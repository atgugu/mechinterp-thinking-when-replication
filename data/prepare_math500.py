"""Prepare MATH500 evaluation split."""

import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from const import DATA_DIR  # noqa: E402


def format_prompt(problem: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "Put the final answer in \\boxed{}.\n\n"
        f"Problem: {problem}\n\nSolution:"
    )


def main(n: int | None = None, force: bool = False):
    out_path = DATA_DIR / "math500.jsonl"
    if out_path.exists() and not force:
        with open(out_path) as f:
            count = sum(1 for _ in f)
        print(f"Already cached: {out_path} ({count} examples). Pass --force to redo.")
        return out_path

    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"MATH-500 test: {len(ds)} examples")

    rows = []
    for i, ex in enumerate(ds):
        if n is not None and i >= n:
            break
        rows.append({
            "id": ex.get("unique_id", f"math500_{i}"),
            "prompt": format_prompt(ex["problem"]),
            "raw_problem": ex["problem"],
            "answer": ex["answer"],
            "solution": ex.get("solution", ""),
            "level": ex.get("level"),
            "subject": ex.get("subject"),
        })

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} examples to {out_path}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    main(args.n, args.force)
