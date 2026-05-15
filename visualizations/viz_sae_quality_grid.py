"""Figure 2: SAE quality grid — combined-score vs dict_size, one panel per layer.

Mirrors the paper's Figure 2: shows the elbow at 10-20 clusters.
"""

import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from const import SAES_DIR, FIGURES_DIR, ACCENT_GOLD, MODEL_PAIRS  # noqa: E402
from visualizations.style import apply_dark_style  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    args = p.parse_args()

    apply_dark_style()
    scores_path = SAES_DIR / args.pair / "taxonomy_scores.json"
    if not scores_path.exists():
        raise SystemExit(f"Run saes.score_taxonomy first; missing {scores_path}")
    scores = json.loads(scores_path.read_text())

    # Group by layer
    by_layer: dict[int, list[dict]] = {}
    for r in scores:
        by_layer.setdefault(r["layer"], []).append(r)
    layers = sorted(by_layer)
    n_panels = len(layers)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.0 * nrows),
                              squeeze=False, facecolor="#0d1117")

    # Find global best (layer, K) for emphasis
    best = max(scores, key=lambda r: r["combined"])

    for i, L in enumerate(layers):
        ax = axes[i // ncols][i % ncols]
        rows = sorted(by_layer[L], key=lambda r: r["dict_size"])
        ks = [r["dict_size"] for r in rows]
        for metric, color, label in [
            ("independence", "#3b82f6", "Independence"),
            ("completeness", "#10b981", "Completeness"),
            ("combined", ACCENT_GOLD, "Combined"),
        ]:
            vals = [r[metric] for r in rows]
            ax.plot(ks, vals, "-o", color=color, label=label, lw=1.8, ms=5)

        # Mark elbow band
        ax.axvspan(10, 20, color="#f0b429", alpha=0.07, zorder=0)
        ax.set_title(f"Layer {L}", color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel("Dict size K", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.grid(alpha=0.15, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9,
                      edgecolor="#30363d")

        if best["layer"] == L:
            ax.scatter([best["dict_size"]], [best["combined"]],
                       s=240, facecolors="none", edgecolors=ACCENT_GOLD,
                       linewidths=2.4, zorder=5)
            # Place callout above + to the right so it doesn't overlap curves
            ax.annotate(
                f"best K={best['dict_size']}",
                xy=(best["dict_size"], best["combined"]),
                xytext=(best["dict_size"] + 6, 1.05),
                color=ACCENT_GOLD, fontsize=9.5, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="-", color=ACCENT_GOLD, lw=0.8, alpha=0.6),
            )

    # Hide unused panels
    for j in range(n_panels, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(
        f"Taxonomy quality across layers  ·  elbow at K ∈ [10, 20]  ·  "
        f"best: L{best['layer']} / K={best['dict_size']}",
        color="white", fontsize=15, fontweight="bold", y=0.995)
    out = FIGURES_DIR / f"fig2_sae_quality_grid_{args.pair}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=150, facecolor="#0d1117")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
