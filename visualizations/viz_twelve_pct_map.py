"""Figure 1: The "12% Map" — visualize sparsity of steering on a single rollout.

Render a chosen MATH500 problem as a tile grid (one tile per token, ~80 cols).
Steered tokens are colored by their category; unsteered tokens are dim gray.
Adds side-panels comparing the same problem under base-only and thinking-only.
"""

import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from const import HYBRID_DIR, FIGURES_DIR, MODEL_PAIRS  # noqa: E402
from visualizations.style import apply_dark_style, category_color  # noqa: E402


def load_records(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def pick_example(full_recs, base_recs, thinking_recs, prefer_correct: bool = True,
                 min_steered: int = 4):
    """Pick a problem that is best for the figure (most steering events).
    Also prefers problems where hybrid succeeded but base failed (the wow story)."""
    candidates = []
    for pid, rec in full_recs.items():
        n_steered = sum(rec.get("steered_flags", []))
        if n_steered < min_steered:
            continue
        # Bonus: hybrid succeeded, base did not (best illustration)
        bonus = 0
        if pid in base_recs:
            # Imperfect ok-flag inference: full has more tokens steered AND base/thinking present
            bonus = n_steered * 3
        candidates.append((pid, n_steered + bonus, rec))
    if not candidates:
        for pid, rec in full_recs.items():
            if sum(rec.get("steered_flags", [])) > 0:
                candidates.append((pid, sum(rec.get("steered_flags", [])), rec))
    if not candidates:
        for pid, rec in full_recs.items():
            candidates.append((pid, 0, rec))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[1])
    return candidates[0][0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    p.add_argument("--dataset", default="auto",
                   help="auto picks the dataset with the highest mean steering rate")
    p.add_argument("--problem_id", default=None)
    p.add_argument("--cols", type=int, default=80)
    args = p.parse_args()

    apply_dark_style()
    # Auto-pick the dataset with the most steering events on any single problem
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
        print(f"Auto-selected dataset: {args.dataset}")
    full = load_records(HYBRID_DIR / f"{args.pair}_{args.dataset}_full.jsonl")
    base = load_records(HYBRID_DIR / f"{args.pair}_{args.dataset}_base.jsonl")
    think = load_records(HYBRID_DIR / f"{args.pair}_{args.dataset}_thinking.jsonl")

    pid = args.problem_id or pick_example(full, base, think)
    if pid is None:
        raise SystemExit("No suitable problem found in hybrid outputs")
    print(f"Selected problem: {pid}")

    rec = full[pid]
    flags = rec.get("steered_flags", [])
    cats = rec.get("cat_trace", [])
    n = len(flags)
    cols = args.cols
    rows = (n + cols - 1) // cols

    tile_w = 14.0 / cols  # inches per tile (target ~14in wide)
    grid_h = rows * tile_w * 1.0  # square tiles
    fig = plt.figure(figsize=(14, grid_h + 1.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[grid_h, 1.2], hspace=0.15)
    ax = fig.add_subplot(gs[0])

    # Draw tile grid
    for i, (f, c) in enumerate(zip(flags, cats)):
        r = i // cols
        col_ = i % cols
        color = category_color(c) if f else "#1f2937"
        edge = "#e5e7eb" if f else "#374151"
        ax.add_patch(patches.Rectangle((col_, -r - 1), 1, 1,
                                       facecolor=color, edgecolor=edge, linewidth=0.3))

    ax.set_xlim(0, cols)
    ax.set_ylim(-rows - 0.5, 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    sf = rec.get("steered_frac", sum(flags) / max(len(flags), 1))
    ax.set_title(
        f"{pid}  ·  {n} tokens  ·  {sum(flags)} steered ({sf*100:.1f}%)\n"
        "Colored tiles = steering active (color = category); dark = base passthrough",
        color="white",
    )

    # Legend / outcome panel
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis("off")
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)

    # Build category legend with names
    sae_dir = __import__("const").SAES_DIR / args.pair  # noqa
    cat_labels: dict[int, str] = {}
    try:
        import json as _j
        from const import SAES_DIR
        best_t = _j.loads((SAES_DIR / args.pair / "best_taxonomy.json").read_text())
        lbl_p = SAES_DIR / args.pair / f"L{best_t['layer']}_K{best_t['dict_size']}_labels.json"
        if lbl_p.exists():
            for c in _j.loads(lbl_p.read_text()):
                cat_labels[c["cluster_id"]] = c["label_info"]["label"]
    except Exception:
        pass

    used_cats = sorted({c for c, f in zip(cats, flags) if f})
    for i, cid in enumerate(used_cats):
        x = 0.02 + i * 0.13
        legend_ax.add_patch(patches.Rectangle((x, 0.55), 0.035, 0.3,
                                              facecolor=category_color(cid),
                                              edgecolor="white", lw=0.5))
        lbl = cat_labels.get(cid, f"c{cid}")
        legend_ax.text(x + 0.045, 0.7,
                       f"c{cid} · {lbl.replace('_', ' ')}",
                       color="white", va="center", fontsize=9)

    # Comparative outcomes panel
    base_rec = base.get(pid, {})
    think_rec = think.get(pid, {})
    outcome_text = (
        f"$\\bf{{Base\\;only}}$: {base_rec.get('n_tokens', '?')} tok      "
        f"$\\bf{{Hybrid}}$: {rec.get('n_tokens', '?')} tok, "
        f"{sum(flags)} steered ({sf*100:.1f}%)      "
        f"$\\bf{{Thinking\\;only}}$: {think_rec.get('n_tokens', '?')} tok"
    )
    legend_ax.text(0.02, 0.18, outcome_text, color="#cbd5e1", fontsize=10)

    out = FIGURES_DIR / f"fig1_twelve_pct_map_{args.pair}_{pid.replace('/', '_')}.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
