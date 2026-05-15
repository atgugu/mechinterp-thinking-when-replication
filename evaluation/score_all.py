"""Grade hybrid-generation JSONL files against ground truth.

Computes overall accuracy and prints a base→hybrid→thinking gap-recovery
table for the README. Uses the lightweight grader at evaluation/grader_lite.py.
"""

import re
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from const import MODEL_PAIRS, HYBRID_DIR, RESULTS_DIR  # noqa: E402
from evaluation.grader_lite import math_equal_lite as math_equal  # noqa: E402


_BOX_RE = re.compile(r"\\boxed\{(.+?)\}", re.DOTALL)
_FINAL_RE = re.compile(r"(?:final answer|answer)[:\s]*([^\n]+)", re.IGNORECASE)
_GSM8K_RE = re.compile(r"####\s*(-?[\d,\.]+)|final answer[:\s]*([-\d,\.]+)", re.IGNORECASE)


def extract_math500_pred(text: str) -> str:
    """Extract \\boxed{...} content; fall back to final-answer regex, then last number."""
    m = _BOX_RE.findall(text)
    if m:
        return m[-1].strip()
    m = _FINAL_RE.search(text)
    if m:
        return m.group(1).strip()
    # Last-ditch: last number in text
    nums = re.findall(r"-?\d[\d,\.]*", text)
    return nums[-1].replace(",", "") if nums else ""


def extract_gsm8k_pred(text: str) -> str:
    m = _GSM8K_RE.search(text)
    if m:
        return (m.group(1) or m.group(2) or "").replace(",", "").strip()
    nums = re.findall(r"-?\d[\d,\.]*", text)
    return nums[-1].replace(",", "") if nums else ""


def grade(pred: str, target: str, dataset: str) -> bool:
    if not pred or target is None:
        return False
    target = str(target).strip()
    pred = str(pred).strip()
    if dataset == "math500":
        try:
            return bool(math_equal(pred, target))
        except Exception:
            pass
    # numeric compare
    try:
        return abs(float(pred.replace(",", "")) - float(str(target).replace(",", ""))) < 1e-6
    except Exception:
        return pred.lower() == target.lower()


def grade_file(path: Path, dataset: str) -> dict:
    n, correct = 0, 0
    steered_fracs = []
    total_tokens = 0
    rows = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            text = rec["completion"]
            ans = rec.get("answer")
            if dataset == "math500":
                pred = extract_math500_pred(text)
            else:
                pred = extract_gsm8k_pred(text)
            ok = grade(pred, ans, dataset)
            n += 1
            correct += int(ok)
            total_tokens += rec.get("n_tokens", 0)
            if "steered_frac" in rec:
                steered_fracs.append(rec["steered_frac"])
            rows.append({"id": rec["id"], "ok": ok, "pred": pred, "answer": ans})
    return {
        "path": str(path),
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "steered_frac_mean": sum(steered_fracs) / len(steered_fracs) if steered_fracs else 0.0,
        "tokens_total": total_tokens,
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    p.add_argument("--datasets", nargs="*", default=["math500", "gsm8k"])
    p.add_argument("--print_recovery", action="store_true")
    args = p.parse_args()

    summary: dict = {"pair": args.pair, "by_dataset": {}}
    for ds in args.datasets:
        by_mode = {}
        for f in sorted(HYBRID_DIR.glob(f"{args.pair}_{ds}_*.jsonl")):
            mode = f.stem.split("_", 2)[2]  # qwen-7b_math500_<mode>
            res = grade_file(f, ds)
            by_mode[mode] = res
            print(f"  [{ds}] {mode:<14}  acc={res['accuracy']:.3f} ({res['correct']}/{res['n']})  "
                  f"steered={res['steered_frac_mean']:.2%}  tokens={res['tokens_total']}")
        summary["by_dataset"][ds] = by_mode

    # Compute gap recovery if both base and thinking present
    print("\nGap recovery:")
    for ds, modes in summary["by_dataset"].items():
        b = modes.get("base", {}).get("accuracy")
        t = modes.get("thinking", {}).get("accuracy")
        h = modes.get("full", {}).get("accuracy")
        if b is None or t is None or h is None:
            continue
        gap = t - b
        rec = ((h - b) / gap * 100) if abs(gap) > 1e-6 else 0.0
        print(f"  {ds}: base={b:.3f}  hybrid={h:.3f}  thinking={t:.3f}  →  recovery={rec:.1f}%")
        summary["by_dataset"][ds]["__recovery_pct"] = rec

    # Drop heavy per-row data from saved summary
    light = {"pair": args.pair, "by_dataset": {}}
    for ds, modes in summary["by_dataset"].items():
        if not isinstance(modes, dict):
            continue
        light["by_dataset"][ds] = {}
        for mode, r in modes.items():
            if not isinstance(r, dict):
                light["by_dataset"][ds][mode] = r
                continue
            light["by_dataset"][ds][mode] = {k: v for k, v in r.items() if k != "rows"}

    out = RESULTS_DIR / f"scores_{args.pair}.json"
    with open(out, "w") as f:
        json.dump(light, f, indent=2)
    print(f"\nWrote summary → {out}")


if __name__ == "__main__":
    main()
