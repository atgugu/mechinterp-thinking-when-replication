"""Prepare MMLU-Pro subset for thinking-model rollouts.

Paper used all 12,102 MMLU-Pro questions; we use a 3k random subset
which yields ~100k sentence-level activations after rollouts — sufficient
for the 10-20 cluster elbow.
"""

import json
import random
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from const import DATA_DIR, MMLU_PRO_SUBSET  # noqa: E402


def format_prompt(question: str, options: list[str]) -> str:
    """Standard MMLU-Pro prompt format."""
    letters = "ABCDEFGHIJ"
    opts = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
    return (
        "Answer the following multiple-choice question. "
        "Think step by step, then give the final answer letter.\n\n"
        f"Question: {question}\n\nOptions:\n{opts}\n\nAnswer:"
    )


def main(n: int = MMLU_PRO_SUBSET, seed: int = 0, force: bool = False):
    out_path = DATA_DIR / "mmlu_pro_subset.jsonl"
    if out_path.exists() and not force:
        with open(out_path) as f:
            count = sum(1 for _ in f)
        print(f"Already cached: {out_path} ({count} examples). Pass --force to redo.")
        return out_path

    from datasets import load_dataset

    print(f"Loading TIGER-Lab/MMLU-Pro ...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    print(f"Total examples: {len(ds)}")

    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    indices = indices[:n]

    with open(out_path, "w") as f:
        for i in indices:
            ex = ds[i]
            rec = {
                "id": f"mmlu_pro_{ex['question_id']}",
                "category": ex["category"],
                "prompt": format_prompt(ex["question"], ex["options"]),
                "answer": ex["answer"],
                "answer_index": ex["answer_index"],
                "raw_question": ex["question"],
                "options": ex["options"],
            }
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {n} examples to {out_path}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=MMLU_PRO_SUBSET)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    main(args.n, args.seed, args.force)
