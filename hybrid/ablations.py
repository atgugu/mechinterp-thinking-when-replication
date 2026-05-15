"""Run all ablation modes over a dataset by repeatedly invoking infer_hybrid."""

import subprocess
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from const import MODEL_PAIRS  # noqa: E402

ABLATION_MODES = ("base", "only_bias", "random_firing", "random_vectors", "full", "thinking")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    p.add_argument("--dataset", default="math500", choices=["math500", "gsm8k"])
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--modes", nargs="*", default=list(ABLATION_MODES))
    args = p.parse_args()

    for mode in args.modes:
        print(f"\n{'='*60}\n  MODE: {mode}\n{'='*60}")
        cmd = [
            sys.executable, "-m", "hybrid.infer_hybrid",
            "--pair", args.pair,
            "--mode", mode,
            "--dataset", args.dataset,
            "--n", str(args.n),
            "--max_new_tokens", str(args.max_new_tokens),
            "--threshold", str(args.threshold),
        ]
        rc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1])).returncode
        if rc != 0:
            print(f"  WARNING: mode={mode} returned {rc}")


if __name__ == "__main__":
    main()
