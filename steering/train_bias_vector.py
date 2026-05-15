"""Train the general-purpose "bias vector" — a category-agnostic steering vector
trained on random thinking-model rollouts.

This is one of the paper's ablation baselines: when applied alone (only-bias),
it should give a modest performance bump that does NOT match the full hybrid.
"""

import os
import sys
import json
import random
import argparse
import time
from pathlib import Path

# (memory_guard imported from local package root)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["TRANSFORMERLENS_ALLOW_MPS"] = "1"

import torch  # noqa: E402

from memory_guard import init_memory_guard, check_memory  # noqa: E402
from const import (  # noqa: E402
    MODEL_PAIRS, STEERING_DIR, ROLLOUTS_DIR,
    STEERING_LAMBDA, STEERING_LR, STEERING_EPOCHS,
)
from steering.train_steering_vectors import (  # noqa: E402
    load_models, load_rollout_text, train_steering_vector,
)
from rollouts.segment_sentences import extract_think_block, split_sentences  # noqa: E402


def random_pairs_from_rollouts(rollouts: dict[str, str], tokenizer,
                               n_pairs: int = 500, seed: int = 0,
                               max_tokens: int = 320):
    rng = random.Random(seed)
    items = list(rollouts.items())
    rng.shuffle(items)
    pairs: list[tuple[torch.Tensor, int]] = []
    for rid, full in items:
        if len(pairs) >= n_pairs:
            break
        think = extract_think_block(full)
        sents = split_sentences(think)
        if len(sents) < 2:
            continue
        # Pick a random sentence as continuation
        i = rng.randrange(1, len(sents))
        cont = sents[i]
        ctx = " ".join(sents[:i])
        text = (ctx + " " + cont).strip()
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=True, max_length=max_tokens)
        ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        offsets = enc["offset_mapping"]
        pos = (ctx + " ").__len__()
        cont_start = next((j for j, (s, e) in enumerate(offsets) if e > pos), None)
        if cont_start is None or cont_start >= len(ids) - 1:
            continue
        pairs.append((ids, cont_start))
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    p.add_argument("--n_pairs", type=int, default=200)
    p.add_argument("--epochs", type=int, default=STEERING_EPOCHS)
    p.add_argument("--lr", type=float, default=STEERING_LR)
    p.add_argument("--lam", type=float, default=STEERING_LAMBDA)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
    args = p.parse_args()

    init_memory_guard()
    cfg = MODEL_PAIRS[args.pair]

    rollout_path = ROLLOUTS_DIR / f"{args.pair}_thinking_mmlu_pro.jsonl"
    rollouts = load_rollout_text(rollout_path)

    base, thinking, tok = load_models(cfg, device=args.device)
    check_memory("after both models loaded")

    pairs = random_pairs_from_rollouts(rollouts, tok, n_pairs=args.n_pairs)
    print(f"Built {len(pairs)} random training pairs")

    t0 = time.time()
    v, hist = train_steering_vector(
        base, thinking, cfg["steer_layer"], cfg["steer_layer"],
        cfg["d_model"], pairs, lr=args.lr, epochs=args.epochs, lam=args.lam,
        device=args.device,
    )
    print(f"Bias-vector training: {time.time()-t0:.1f}s")
    print(f"Final NLL_T={hist[-1]['nll_thinking']:.3f}  NLL_B={hist[-1]['nll_base']:.3f}")

    out_dir = STEERING_DIR / args.pair
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bias_L{cfg['steer_layer']}.pt"
    torch.save({"vector": v, "history": hist, "layer": cfg["steer_layer"]}, out_path)
    print(f"Saved bias vector → {out_path}")


if __name__ == "__main__":
    main()
