"""Figure 2 (animation): side-by-side rollout under base / hybrid / thinking.

Render an animated GIF of the same MATH500 problem unfolding token-by-token
under three regimes. The hybrid track flashes its steering-category color.
"""

import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio.v2 as imageio

from const import HYBRID_DIR, FIGURES_DIR, MODEL_PAIRS  # noqa: E402
from visualizations.style import apply_dark_style, category_color, ACCENT_GOLD  # noqa: E402
from visualizations.viz_twelve_pct_map import load_records, pick_example  # noqa: E402


def render_frame(rec_base, rec_full, rec_think, step: int, max_steps: int):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1, 1]})
    titles = ["Base only", "Hybrid (full)", "Thinking only"]
    recs = [rec_base, rec_full, rec_think]
    cols = 64
    for ax, title, rec in zip(axes, titles, recs):
        ax.axis("off")
        n_done = min(step, rec.get("n_tokens", 0))
        flags = rec.get("steered_flags", []) if title == "Hybrid (full)" else None
        cats = rec.get("cat_trace", []) if title == "Hybrid (full)" else None
        for i in range(n_done):
            r = i // cols
            c = i % cols
            if flags is not None and i < len(flags) and flags[i]:
                color = category_color(cats[i])
            else:
                color = "#475569" if title == "Base only" else ("#34d399" if title == "Thinking only" else "#1f2937")
            ax.add_patch(patches.Rectangle((c, -r - 1), 1, 1, facecolor=color, edgecolor="#1e293b", linewidth=0.1))
        ax.set_xlim(0, cols)
        rows_needed = max(1, (rec.get("n_tokens", 0) + cols - 1) // cols)
        ax.set_ylim(-rows_needed - 0.5, 0.5)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\n{n_done}/{rec.get('n_tokens', 0)} tokens", color="white", fontsize=11)

    fig.suptitle(f"Same problem, three regimes  ·  step {step}/{max_steps}",
                 color=ACCENT_GOLD, fontsize=13, y=1.02)
    fig.patch.set_facecolor("#0d1117")
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = rgba[..., :3].copy()
    plt.close(fig)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    p.add_argument("--dataset", default="auto",
                   help="auto picks the dataset with the highest steering activity")
    p.add_argument("--problem_id", default=None)
    p.add_argument("--steps", type=int, default=24, help="Animation frames")
    p.add_argument("--fps", type=int, default=4)
    args = p.parse_args()

    apply_dark_style()
    if args.dataset == "auto":
        best_ds, best_max = None, -1
        for ds in ("gsm8k", "math500"):
            full = load_records(HYBRID_DIR / f"{args.pair}_{ds}_full.jsonl")
            for _, rec in full.items():
                ns = sum(rec.get("steered_flags", []))
                if ns > best_max:
                    best_max = ns
                    best_ds = ds
        args.dataset = best_ds or "math500"
    full = load_records(HYBRID_DIR / f"{args.pair}_{args.dataset}_full.jsonl")
    base = load_records(HYBRID_DIR / f"{args.pair}_{args.dataset}_base.jsonl")
    think = load_records(HYBRID_DIR / f"{args.pair}_{args.dataset}_thinking.jsonl")
    pid = args.problem_id or pick_example(full, base, think)
    if pid is None:
        raise SystemExit("No suitable problem")
    print(f"Animating problem: {pid}")
    if pid not in base or pid not in think:
        raise SystemExit("Need outputs for base AND thinking modes too")

    rec_full = full[pid]
    rec_base = base[pid]
    rec_think = think[pid]
    max_tokens = max(rec_full["n_tokens"], rec_base["n_tokens"], rec_think["n_tokens"])
    step_size = max(1, max_tokens // args.steps)

    frames = []
    for step in range(step_size, max_tokens + 1, step_size):
        frames.append(render_frame(rec_base, rec_full, rec_think, step, max_tokens))
    # Hold final frame
    for _ in range(args.fps):
        frames.append(frames[-1])

    out = FIGURES_DIR / f"fig2_rollout_animation_{args.pair}_{pid.replace('/', '_')}.gif"
    imageio.mimsave(out, frames, fps=args.fps)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
