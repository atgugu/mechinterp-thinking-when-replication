"""Extract sentence-pooled residual-stream activations from thinking + base models.

Iterates over rollouts (`results/rollouts/<pair>_thinking_mmlu_pro.jsonl`),
splits the <think> block into sentences, runs each through both base and thinking
models, captures `hook_resid_post` at 6 distributed layers, and saves
mean-pooled per-sentence activations as fp16 shards.

Output layout:
    results/acts/<pair>/<which>/L<layer>/shard_<n>.pt  -- dict with keys:
        "acts": Tensor[n_sent, d_model] fp16
        "ids":  list[str] sentence IDs   ("<rollout_id>::s<idx>")
        "text": list[str]                (the sentence text)

A separate `index.json` per shard directory records what's been done.
"""

import os
import sys
import gc
import json
import argparse
from pathlib import Path

os.environ["TRANSFORMERLENS_ALLOW_MPS"] = "1"
# (memory_guard imported from local package root)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_guard import init_memory_guard, check_memory  # noqa: E402
from const import MODEL_PAIRS, ROLLOUTS_DIR, ACTS_DIR  # noqa: E402
from rollouts.segment_sentences import extract_think_block, iter_sentences_with_spans  # noqa: E402
from extraction.pool_utils import map_sentences_to_token_ranges, mean_pool  # noqa: E402

import torch  # noqa: E402


def load_model(path: str, dtype=torch.float16):
    """Load a HF model on the available accelerator for activation extraction.

    We use raw HF (not TransformerLens) here for speed and to enable cleaner
    hooks on resid_post. We register a forward hook per target decoder layer.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available():
        device_map = {"": "cuda"}
    elif torch.backends.mps.is_available():
        device_map = {"": "mps"}
    else:
        device_map = {"": "cpu"}
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, device_map=device_map, local_files_only=True,
        attn_implementation="sdpa",
        output_hidden_states=True,
    )
    model.eval()
    return model, tokenizer


def iter_rollout_sentences(rollout_path: Path, min_len: int = 8, max_per_rollout: int = 200):
    with open(rollout_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            think = extract_think_block(rec["completion"])
            sents = iter_sentences_with_spans(think, min_len=min_len)
            if not sents:
                continue
            if len(sents) > max_per_rollout:
                sents = sents[:max_per_rollout]
            yield rec["id"], think, sents


def extract_for_model(
    model, tokenizer, model_label: str, pair: str, layers: list[int],
    rollout_path: Path, n_max: int | None = None, shard_size: int = 500,
):
    out_root = ACTS_DIR / pair / model_label
    out_root.mkdir(parents=True, exist_ok=True)
    for L in layers:
        (out_root / f"L{L}").mkdir(exist_ok=True)

    # Determine resume point
    existing_shards: list[int] = []
    L0_dir = out_root / f"L{layers[0]}"
    for f in L0_dir.glob("shard_*.pt"):
        existing_shards.append(int(f.stem.split("_")[1]))
    shard_idx = (max(existing_shards) + 1) if existing_shards else 0
    # Count how many sentences already done (rough — just skip first shard_idx*shard_size)
    skip = shard_idx * shard_size
    print(f"  [{model_label}] Resuming at shard {shard_idx} (skipping ~{skip} sentences)")

    buf_acts = {L: [] for L in layers}
    buf_ids: list[str] = []
    buf_text: list[str] = []
    n_collected = 0
    n_skipped = 0

    # Hook setup: capture resid stream output of each target decoder layer
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(L):
        def h(_module, _inp, output):
            # output is either Tensor (hidden_states) or tuple (Qwen / Llama variant)
            hs = output[0] if isinstance(output, tuple) else output
            captured[L] = hs.detach()
        return h

    # Wire up hooks on `model.model.layers[L]` (standard HF decoder-only layout)
    decoder_layers = model.model.layers
    for L in layers:
        hooks.append(decoder_layers[L].register_forward_hook(make_hook(L)))

    try:
        with torch.inference_mode():
            for r_idx, (rid, think, sents) in enumerate(iter_rollout_sentences(rollout_path)):
                if n_max is not None and r_idx >= n_max:
                    break

                # Skip early shards
                if n_skipped < skip:
                    n_skipped += len(sents)
                    continue

                # Tokenize the whole <think> block once
                enc = tokenizer(
                    think, return_tensors="pt", return_offsets_mapping=True,
                    add_special_tokens=False, truncation=True, max_length=4096,
                )
                offsets = enc["offset_mapping"][0].tolist()
                dev = next(model.parameters()).device
                input_ids = enc["input_ids"].to(dev)
                attn_mask = enc["attention_mask"].to(dev)

                tok_ranges = map_sentences_to_token_ranges(sents, offsets)

                # Forward pass
                captured.clear()
                _ = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)

                # Pool per-layer
                for L in layers:
                    acts_t = captured[L]  # (1, T, D)
                    pooled = mean_pool(acts_t, tok_ranges).to("cpu", dtype=torch.float16)
                    buf_acts[L].append(pooled)

                for si, (s_text, _, _) in enumerate(sents):
                    buf_ids.append(f"{rid}::s{si}")
                    buf_text.append(s_text)
                n_collected += len(sents)

                # Flush if full
                if n_collected >= shard_size:
                    _flush_shard(out_root, layers, shard_idx, buf_acts, buf_ids, buf_text)
                    shard_idx += 1
                    buf_acts = {L: [] for L in layers}
                    buf_ids = []
                    buf_text = []
                    n_collected = 0
                    gc.collect()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    if shard_idx % 5 == 0:
                        check_memory(f"shard {shard_idx} flushed")

            # Final flush
            if n_collected > 0:
                _flush_shard(out_root, layers, shard_idx, buf_acts, buf_ids, buf_text)
    finally:
        for h in hooks:
            h.remove()


def _flush_shard(out_root, layers, shard_idx, buf_acts, buf_ids, buf_text):
    for L in layers:
        acts = torch.cat(buf_acts[L], dim=0)
        torch.save({"acts": acts, "ids": buf_ids, "text": buf_text}, out_root / f"L{L}" / f"shard_{shard_idx:04d}.pt")
    print(f"  Flushed shard {shard_idx:04d} ({len(buf_ids)} sentences) → {out_root}/L<...>/shard_{shard_idx:04d}.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    p.add_argument("--which", default="both", choices=["base", "thinking", "both"])
    p.add_argument("--rollouts", default=None, help="Path to rollout JSONL")
    p.add_argument("--n_max", type=int, default=None, help="Cap rollouts processed")
    p.add_argument("--shard_size", type=int, default=500)
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="Override which layers to extract (default: all in const.MODEL_PAIRS)")
    args = p.parse_args()

    init_memory_guard()

    cfg = MODEL_PAIRS[args.pair]
    layers = args.layers if args.layers else cfg["extract_layers"]
    rollout_path = Path(args.rollouts) if args.rollouts else ROLLOUTS_DIR / f"{args.pair}_thinking_mmlu_pro.jsonl"
    if not rollout_path.exists():
        raise SystemExit(f"No rollouts at {rollout_path}; generate them first.")

    print(f"Pair: {args.pair}")
    print(f"Layers: {layers}")
    print(f"Rollouts: {rollout_path}")

    to_do = []
    if args.which in ("thinking", "both"):
        to_do.append(("thinking", cfg["thinking"]))
    if args.which in ("base", "both"):
        to_do.append(("base", cfg["base"]))

    for label, path in to_do:
        print(f"\n=== Extracting for {label} ({path}) ===")
        model, tokenizer = load_model(path)
        check_memory(f"after {label} load")
        extract_for_model(
            model, tokenizer, label, args.pair, layers, rollout_path,
            n_max=args.n_max, shard_size=args.shard_size,
        )
        del model, tokenizer
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        check_memory(f"after {label} done")


if __name__ == "__main__":
    main()
