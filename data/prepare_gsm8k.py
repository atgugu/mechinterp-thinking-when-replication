"""Prepare GSM8K test split for hybrid-model evaluation."""

import json
import re
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from const import DATA_DIR  # noqa: E402


def extract_gsm8k_answer(answer_field: str) -> str:
    """GSM8K answers end with '#### <number>'."""
    m = re.search(r"####\s*(-?[\d,\.]+)", answer_field)
    if not m:
        return answer_field.strip().splitlines()[-1].strip()
    return m.group(1).replace(",", "").strip()


def format_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "Give the final numeric answer after 'Final Answer:'.\n\n"
        f"Problem: {question}\n\nSolution:"
    )


def main(n: int | None = None, force: bool = False):
    out_path = DATA_DIR / "gsm8k_test.jsonl"
    if out_path.exists() and not force:
        with open(out_path) as f:
            count = sum(1 for _ in f)
        print(f"Already cached: {out_path} ({count} examples). Pass --force to redo.")
        return out_path

    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="test")
    print(f"GSM8K test: {len(ds)} examples")

    rows = []
    for i, ex in enumerate(ds):
        if n is not None and i >= n:
            break
        rows.append({
            "id": f"gsm8k_{i}",
            "prompt": format_prompt(ex["question"]),
            "raw_question": ex["question"],
            "answer": extract_gsm8k_answer(ex["answer"]),
            "full_solution": ex["answer"],
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
