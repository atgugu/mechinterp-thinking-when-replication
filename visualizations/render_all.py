"""Render every figure in one go.

Each script is best-effort: if its prerequisite artifacts are missing the
script is skipped, not crashed, so partial pipelines still produce what they can.
"""

import subprocess
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from const import MODEL_PAIRS  # noqa: E402

SCRIPTS = [
    "visualizations.viz_sae_quality_grid",
    "visualizations.viz_category_landscape",
    "visualizations.viz_wordclouds",
    "visualizations.viz_twelve_pct_map",
    "visualizations.viz_rollout_animation",
    "visualizations.viz_gap_recovery",
    "visualizations.viz_steering_position",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    args = p.parse_args()
    cwd = str(Path(__file__).resolve().parents[1])
    for mod in SCRIPTS:
        print(f"\n=== {mod} ===")
        rc = subprocess.run(
            [sys.executable, "-m", mod, "--pair", args.pair],
            cwd=cwd,
        ).returncode
        if rc != 0:
            print(f"   (skipped or failed; rc={rc})")


if __name__ == "__main__":
    main()
