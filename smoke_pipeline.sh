#!/usr/bin/env bash
# Tight-loop smoke pipeline — runs the full pipeline end-to-end on minimal
# data so we can validate that every stage produces an artifact.
# Assumes rollouts have already been generated.
# Activate your virtualenv before running, or set VENV to its activate script.
set -euo pipefail

cd "$(dirname "$0")"
[ -n "${VENV:-}" ] && [ -f "$VENV" ] && source "$VENV"
export ML_MEMORY_LIMIT_GB="${ML_MEMORY_LIMIT_GB:-0}"

PAIR="qwen-7b"

run() { echo; echo "▸ $*"; "$@"; }

# Skip rollouts (assume already generated)
# For smoke, only extract at 2 layers (paper uses 6 distributed)
run python -m extraction.extract_activations --pair "$PAIR" --shard_size 200 --layers 13 17

# Small SAE sweep (only 2 layers, 3 dict sizes for smoke)
run python -m saes.train_saes --pair "$PAIR" --layers 13 17 --dict_sizes 10 15 20
run python -m saes.score_taxonomy --pair "$PAIR" --skip_consistency
run python -m saes.label_clusters --pair "$PAIR"

# Steering vectors (few examples per cat, few epochs)
run python -m steering.train_steering_vectors --pair "$PAIR" --max_examples_per_cat 8 --epochs 5
run python -m steering.train_bias_vector --pair "$PAIR" --n_pairs 30 --epochs 5

# Hybrid inference on tiny subset
for mode in base thinking full only_bias; do
  run python -m hybrid.infer_hybrid --pair "$PAIR" --mode "$mode" --dataset math500 --n 5 --max_new_tokens 512
done

# Evaluate
run python -m evaluation.score_all --pair "$PAIR" --datasets math500 --print_recovery

# Render whatever figures we can
run python -m visualizations.render_all --pair "$PAIR"

echo
echo "============================================================"
echo "  Smoke pipeline complete. Check figures/ and results/."
echo "============================================================"
