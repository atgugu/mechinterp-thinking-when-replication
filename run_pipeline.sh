#!/usr/bin/env bash
# End-to-end production pipeline for the replication.
# Override with env vars (defaults sized for a multi-day single-machine run):
#   PAIR=qwen-7b N_ROLLOUTS=300 N_EVAL=30 MAX_NEW=2048
#   STEER_EXAMPLES=100 STEER_EPOCHS=25
# Activate your virtualenv before running, or set VENV to its activate script.
set -euo pipefail

PAIR="${PAIR:-qwen-7b}"
N_ROLLOUTS="${N_ROLLOUTS:-300}"
N_EVAL="${N_EVAL:-30}"
MAX_NEW="${MAX_NEW:-2048}"
N_GSM8K="${N_GSM8K:-50}"
STEER_EXAMPLES="${STEER_EXAMPLES:-100}"
STEER_EPOCHS="${STEER_EPOCHS:-25}"

cd "$(dirname "$0")"
[ -n "${VENV:-}" ] && [ -f "$VENV" ] && source "$VENV"
export ML_MEMORY_LIMIT_GB="${ML_MEMORY_LIMIT_GB:-0}"

echo "============================================================"
echo "  Paper 4 Replication Pipeline"
echo "  pair=$PAIR  rollouts=$N_ROLLOUTS  eval=$N_EVAL  max_new=$MAX_NEW"
echo "============================================================"

run() { echo; echo "▸ $*"; "$@"; }

run python -m data.prepare_mmlu_pro --n "$N_ROLLOUTS"
run python -m data.prepare_gsm8k    --n "$N_GSM8K"
run python -m data.prepare_math500  --n "$N_EVAL"

run python -m rollouts.generate_rollouts --pair "$PAIR" --n "$N_ROLLOUTS" --max_new_tokens "$MAX_NEW"

run python -m extraction.extract_activations --pair "$PAIR" --shard_size 500

run python -m saes.train_saes --pair "$PAIR"
run python -m saes.score_taxonomy --pair "$PAIR" --skip_consistency
run python -m saes.label_clusters --pair "$PAIR"

run python -m steering.train_steering_vectors --pair "$PAIR" --max_examples_per_cat "$STEER_EXAMPLES" --epochs "$STEER_EPOCHS"
run python -m steering.train_bias_vector --pair "$PAIR" --n_pairs "$STEER_EXAMPLES" --epochs "$STEER_EPOCHS"

for mode in base thinking full only_bias random_firing random_vectors; do
  run python -m hybrid.infer_hybrid --pair "$PAIR" --mode "$mode" --dataset math500 --n "$N_EVAL" --max_new_tokens "$MAX_NEW"
done
for mode in base thinking full; do
  run python -m hybrid.infer_hybrid --pair "$PAIR" --mode "$mode" --dataset gsm8k --n "$N_GSM8K" --max_new_tokens 512
done

run python -m evaluation.score_all --pair "$PAIR" --print_recovery

run python -m visualizations.render_all --pair "$PAIR"

echo
echo "============================================================"
echo "  Done. See results/scores_$PAIR.json and figures/."
echo "============================================================"
