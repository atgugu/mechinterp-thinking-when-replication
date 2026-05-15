"""Figure 3: UMAP landscape of sentence activations, colored by SAE category.

Shows whether the SAE's categories form coherent clusters in activation space.
Interactive Plotly version with hover sentences, plus static PNG.
"""

import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import matplotlib.pyplot as plt

from const import SAES_DIR, ACTS_DIR, FIGURES_DIR, MODEL_PAIRS  # noqa: E402
from saes.topk_sae import TopKSAE  # noqa: E402
from saes.train_saes import load_layer_acts  # noqa: E402
from visualizations.style import apply_dark_style, category_color, PLOTLY_LAYOUT  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    p.add_argument("--n_max", type=int, default=8000, help="Cap points for UMAP (speed)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    apply_dark_style()
    sae_dir = SAES_DIR / args.pair
    best = json.loads((sae_dir / "best_taxonomy.json").read_text())
    L, K = best["layer"], best["dict_size"]

    bundle = torch.load(sae_dir / f"L{L}_K{K}.pt", map_location="cpu", weights_only=False)
    sae = TopKSAE(bundle["config"]["d_input"], bundle["config"]["dict_size"],
                  k=bundle["config"]["k"], normalize_decoder=True)
    sae.load_state_dict(bundle["state_dict"])
    sae.eval()

    # Load labels (if labeling has been run)
    label_path = sae_dir / f"L{L}_K{K}_labels.json"
    cat_labels: dict[int, str] = {}
    if label_path.exists():
        labels = json.loads(label_path.read_text())
        for c in labels:
            cat_labels[c["cluster_id"]] = c["label_info"]["label"]

    acts, _ids, texts = load_layer_acts(args.pair, "thinking", L)
    if len(acts) > args.n_max:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(acts), args.n_max, replace=False)
        acts = acts[idx]
        texts = [texts[i] for i in idx]
    with torch.no_grad():
        cats = sae.assign_category(acts).cpu().numpy()

    print(f"Running UMAP on {len(acts)} points ({acts.shape[1]}d → 2d) ...")
    import umap
    reducer = umap.UMAP(n_components=2, random_state=args.seed, n_neighbors=15, min_dist=0.1)
    emb = reducer.fit_transform(acts.numpy())

    # Static PNG with density-aware hulls
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="#0d1117")
    try:
        from scipy.spatial import ConvexHull
    except Exception:
        ConvexHull = None

    # Clip axis limits tightly to the 2nd-98th percentile (drop visual outliers)
    xlo, xhi = np.percentile(emb[:, 0], [2, 98])
    ylo, yhi = np.percentile(emb[:, 1], [2, 98])
    pad_x = (xhi - xlo) * 0.06
    pad_y = (yhi - ylo) * 0.06

    # Track centroids for labels so we can offset overlapping ones
    centroids = []

    for cid in range(K):
        m = cats == cid
        if m.sum() == 0:
            continue
        lbl = cat_labels.get(cid, f"c{cid}")
        pts_all = emb[m]
        color = category_color(cid)
        # Drop visual outliers from BOTH the hull and the scatter
        if len(pts_all) >= 10:
            cx_q = np.percentile(pts_all[:, 0], [5, 95])
            cy_q = np.percentile(pts_all[:, 1], [5, 95])
            in_core = ((pts_all[:, 0] >= cx_q[0]) & (pts_all[:, 0] <= cx_q[1])
                       & (pts_all[:, 1] >= cy_q[0]) & (pts_all[:, 1] <= cy_q[1]))
            pts_core = pts_all[in_core]
        else:
            pts_core = pts_all

        if ConvexHull is not None and len(pts_core) >= 4:
            try:
                hull = ConvexHull(pts_core)
                poly = pts_core[hull.vertices]
                ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.13, zorder=1)
                ax.plot(np.append(poly[:, 0], poly[0, 0]),
                        np.append(poly[:, 1], poly[0, 1]),
                        color=color, alpha=0.55, lw=1.4, zorder=2)
            except Exception:
                pass
        # Scatter only the core points (drops the lone outliers that wasted space)
        ax.scatter(pts_core[:, 0], pts_core[:, 1], s=10, c=color, alpha=0.55,
                   label=f"{cid} · {lbl.replace('_', ' ')}",
                   edgecolor="none", zorder=3, clip_on=True)
        cx, cy = np.median(pts_core[:, 0]), np.median(pts_core[:, 1])
        centroids.append((cid, cx, cy, color))

    # Draw centroid labels only for clusters whose median is inside the plot limits
    for cid, cx, cy, color in centroids:
        if not (xlo - pad_x <= cx <= xhi + pad_x and ylo - pad_y <= cy <= yhi + pad_y):
            continue
        ax.text(cx, cy, str(cid), color="white", fontsize=11, fontweight="bold",
                ha="center", va="center", clip_on=True,
                bbox=dict(facecolor=color, edgecolor="white", alpha=0.92,
                          boxstyle="circle,pad=0.28"), zorder=5)

    # Lock axis limits AFTER plotting so points outside are clipped, not shown
    ax.set_xlim(xlo - pad_x, xhi + pad_x, auto=False)
    ax.set_ylim(ylo - pad_y, yhi + pad_y, auto=False)
    # Drop clusters that fell entirely outside the limits from the legend
    handles, lbls_legend = ax.get_legend_handles_labels()
    ax.set_xlabel("UMAP 1", fontsize=11, color="#cbd5e1")
    ax.set_ylabel("UMAP 2", fontsize=11, color="#cbd5e1")
    ax.set_title(
        f"Reasoning-category landscape  ·  {args.pair}  ·  L{L}, K={K}",
        color="white", fontsize=14, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.88,
              title="categories", title_fontsize=9.5,
              labelcolor="#e6edf3", edgecolor="#30363d",
              ncol=2, columnspacing=0.6)
    ax.grid(alpha=0.12, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.10)
    out_png = FIGURES_DIR / f"fig3_category_landscape_{args.pair}.png"
    fig.savefig(out_png, facecolor="#0d1117", dpi=160)
    print(f"Wrote {out_png}")

    # Interactive HTML
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed; skipping HTML")
        return
    fig = go.Figure()
    for cid in range(K):
        m = cats == cid
        if m.sum() == 0:
            continue
        lbl = cat_labels.get(cid, f"c{cid}")
        hover = [f"<b>cat {cid}: {lbl}</b><br>" + t[:200].replace('\n', ' ') for t in (texts[i] for i in np.where(m)[0])]
        fig.add_trace(go.Scattergl(
            x=emb[m, 0], y=emb[m, 1], mode="markers",
            marker=dict(size=4, color=category_color(cid)),
            name=f"{cid}: {lbl}", hovertext=hover, hoverinfo="text",
        ))
    fig.update_layout(
        title=f"Reasoning-category landscape (L{L}, K={K}) — {args.pair}",
        xaxis_title="UMAP 1", yaxis_title="UMAP 2",
        legend=dict(itemsizing="constant"),
        **PLOTLY_LAYOUT,
    )
    out_html = FIGURES_DIR / f"fig3_category_landscape_{args.pair}.html"
    fig.write_html(out_html)
    print(f"Wrote {out_html}")


if __name__ == "__main__":
    main()
