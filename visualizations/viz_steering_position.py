"""Figure 6: where steering fires — normalized CoT position × frequency, stacked by category.

Reveals whether backtracking fires late, verification mid-trace, etc.
Most exploratory of the six — likely surfaces a finding worth writing up.
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from const import HYBRID_DIR, SAES_DIR, FIGURES_DIR, MODEL_PAIRS  # noqa: E402
from visualizations.style import apply_dark_style, category_color, PLOTLY_LAYOUT  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    p.add_argument("--dataset", default="math500")
    p.add_argument("--bins", type=int, default=20)
    args = p.parse_args()

    apply_dark_style()
    full_path = HYBRID_DIR / f"{args.pair}_{args.dataset}_full.jsonl"
    if not full_path.exists():
        raise SystemExit(f"Missing {full_path}")

    # Load category labels (if available)
    sae_dir = SAES_DIR / args.pair
    best = json.loads((sae_dir / "best_taxonomy.json").read_text())
    L, K = best["layer"], best["dict_size"]
    label_path = sae_dir / f"L{L}_K{K}_labels.json"
    labels = ({c["cluster_id"]: c["label_info"]["label"]
               for c in json.loads(label_path.read_text())}
              if label_path.exists() else {})

    bins = args.bins
    counts = np.zeros((K, bins), dtype=np.int64)
    total = np.zeros(bins, dtype=np.int64)

    with open(full_path) as f:
        for line in f:
            rec = json.loads(line)
            flags = rec.get("steered_flags", [])
            cats = rec.get("cat_trace", [])
            n = len(flags)
            if n < 5:
                continue
            for i, (fl, c) in enumerate(zip(flags, cats)):
                b = min(int(i * bins / n), bins - 1)
                total[b] += 1
                if fl and c >= 0:
                    counts[c, b] += 1

    # Stacked area: per-bin steering frequency by category
    bins_x = np.linspace(0, 1, bins, endpoint=False) + 0.5 / bins

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(bins, dtype=np.float64)
    legend_cats = []
    for cid in range(K):
        v = counts[cid] / np.maximum(total, 1)
        if v.sum() < 0.01:
            continue
        ax.fill_between(bins_x, bottom, bottom + v, step="mid",
                        color=category_color(cid), alpha=0.85,
                        label=f"{cid}: {labels.get(cid, f'c{cid}')}")
        bottom += v
        legend_cats.append(cid)

    ax.set_xlim(0, 1)
    ymax = float(bottom.max()) if bottom.max() > 0 else 0.05
    ax.set_ylim(0, max(0.01, ymax * 1.15))
    ax.set_xlabel("Normalized position in CoT (start → end)")
    ax.set_ylabel("Steering frequency")
    ax.set_title(f"Where steering fires in the chain  ·  {args.pair} / {args.dataset}", color="white")
    ax.legend(ncol=2, fontsize=8, loc="upper right", framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    out = FIGURES_DIR / f"fig6_steering_position_{args.pair}_{args.dataset}.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")

    # Interactive Plotly version
    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    figp = go.Figure()
    bottom = np.zeros(bins)
    for cid in legend_cats:
        v = counts[cid] / np.maximum(total, 1)
        figp.add_trace(go.Scatter(
            x=bins_x, y=bottom + v, name=f"{cid}: {labels.get(cid, f'c{cid}')}",
            fill="tonexty" if cid != legend_cats[0] else "tozeroy",
            line=dict(width=0.4, color=category_color(cid)),
            fillcolor=category_color(cid),
        ))
        bottom += v
    figp.update_layout(
        title=f"Where steering fires — {args.pair} / {args.dataset}",
        xaxis_title="Normalized CoT position",
        yaxis_title="Steering frequency",
        **PLOTLY_LAYOUT,
    )
    out_html = FIGURES_DIR / f"fig6_steering_position_{args.pair}_{args.dataset}.html"
    figp.write_html(out_html)
    print(f"Wrote {out_html}")


if __name__ == "__main__":
    main()
