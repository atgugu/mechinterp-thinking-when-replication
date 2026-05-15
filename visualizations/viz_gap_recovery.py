"""Figure 5: gap-recovery panel — accuracy bars for {base, only_bias, random_*, full, thinking}
across each evaluation dataset, with Wilson 95 % CI and a headline gap-recovery
annotation where the dataset actually exhibits a positive base→thinking gap.
"""

import json
import math
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from const import RESULTS_DIR, FIGURES_DIR, ACCENT_GOLD, ACCENT_AMBER, MODEL_PAIRS  # noqa: E402
from visualizations.style import apply_dark_style  # noqa: E402


MODE_ORDER = ["base", "only_bias", "random_firing", "random_vectors", "full", "thinking"]
MODE_LABELS = {
    "base": "Base only",
    "only_bias": "Bias only",
    "random_firing": "Random\nfiring",
    "random_vectors": "Random\nvectors",
    "full": "Hybrid",
    "thinking": "Thinking",
}
MODE_COLORS = {
    "base": "#475569",
    "only_bias": "#94a3b8",
    "random_firing": "#a78bfa",
    "random_vectors": "#c084fc",
    "full": ACCENT_GOLD,
    "thinking": "#34d399",
}


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return p, p
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    halfw = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - halfw), min(1.0, center + halfw)


def _panel(ax, ds: str, by_mode: dict):
    """Draw one dataset's bars + recovery annotation on `ax`."""
    accs, cis, labels, colors, modes = [], [], [], [], []
    for m in MODE_ORDER:
        if m not in by_mode:
            continue
        r = by_mode[m]
        n = r["n"]
        p = r["accuracy"]
        lo, hi = wilson_ci(p, n)
        accs.append(p)
        cis.append((lo, hi))
        labels.append(MODE_LABELS[m])
        colors.append(MODE_COLORS[m])
        modes.append(m)
    if not accs:
        return

    x = np.arange(len(accs))
    bars = ax.bar(x, accs, color=colors, edgecolor="#cbd5e1", linewidth=0.6,
                   width=0.7, zorder=3)
    err_low = [a - lo for a, (lo, _) in zip(accs, cis)]
    err_high = [hi - a for a, (_, hi) in zip(accs, cis)]
    ax.errorbar(x, accs, yerr=[err_low, err_high], fmt="none", ecolor="#e2e8f0",
                capsize=4, capthick=1.2, alpha=0.7, zorder=4)
    for xi, a, (lo_ci, hi_ci) in zip(x, accs, cis):
        # Place label above error-bar top to avoid overlap with bar/callout
        ax.text(xi, hi_ci + 0.018, f"{a*100:.0f}%", ha="center", va="bottom",
                color="white", fontsize=10, fontweight="bold", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=9, color="#e6edf3")
    # Headroom for callout
    top = max(hi for _, hi in cis) * 1.15
    ax.set_ylim(0, max(0.65, top + 0.15))
    ax.set_ylabel("Accuracy", fontsize=10, color="#e6edf3")
    n_used = max(by_mode[m]["n"] for m in modes)
    ax.set_title(f"{ds.upper()}    n = {n_used} problems", color=ACCENT_GOLD,
                 fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Recovery annotation only when thinking > base by > 3pp
    base_p = by_mode.get("base", {}).get("accuracy")
    full_p = by_mode.get("full", {}).get("accuracy")
    think_p = by_mode.get("thinking", {}).get("accuracy")
    if (base_p is not None and full_p is not None and think_p is not None
            and (think_p - base_p) > 0.03):
        gap = think_p - base_p
        rec_pct = (full_p - base_p) / gap * 100
        x_base = modes.index("base") if "base" in modes else 0
        x_full = modes.index("full") if "full" in modes else len(modes) - 1
        x_think = modes.index("thinking") if "thinking" in modes else len(modes) - 1
        # Shaded recovery band
        ax.fill_between([x_base - 0.4, x_think + 0.4],
                        base_p, think_p,
                        color=ACCENT_AMBER, alpha=0.08, zorder=1)
        # Horizontal reference lines
        ax.axhline(base_p, color="#94a3b8", linestyle=":", lw=0.8, alpha=0.6, zorder=2)
        ax.axhline(think_p, color="#34d399", linestyle=":", lw=0.8, alpha=0.6, zorder=2)
        # Curved arrow base → full (curve goes ABOVE so the label can sit on it)
        ax.annotate(
            "", xy=(x_full, full_p), xytext=(x_base, base_p),
            arrowprops=dict(arrowstyle="->", color=ACCENT_AMBER, lw=2.4,
                            connectionstyle="arc3,rad=-0.3"),
            zorder=6,
        )
        # Recovery callout — placed above the arrow's apex, well clear of bars
        callout_x = (x_base + x_full) / 2
        # Estimate arc apex (rough): above the chord by ~rad*chord_length
        chord_len_y = abs(full_p - base_p) + 0.001
        callout_y = max(base_p, full_p) + 0.10 + chord_len_y * 0.15
        ax.text(
            callout_x, callout_y, f"{rec_pct:.0f}%  gap recovered",
            color=ACCENT_AMBER, ha="center", va="center", fontsize=11,
            fontweight="bold",
            bbox=dict(facecolor="#0d1117", edgecolor=ACCENT_AMBER,
                      alpha=0.95, boxstyle="round,pad=0.35", lw=1.4),
            zorder=7,
        )
        # Gap explanation as compact footer instead of squeezed under title
        ax.text(0.5, -0.18,
                f"thinking − base = {gap*100:.0f} pp",
                transform=ax.transAxes, ha="center", va="top",
                color="#94a3b8", fontsize=9, style="italic")
    else:
        # Honest no-signal note
        ax.text(0.985, 0.965,
                "no positive base→thinking gap on this slice",
                transform=ax.transAxes, ha="right", va="top",
                color="#fbbf24", fontsize=9.5,
                bbox=dict(facecolor="#0d1117", edgecolor="#fbbf24",
                          alpha=0.85, boxstyle="round,pad=0.25", lw=0.8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    args = p.parse_args()

    apply_dark_style()
    scores_path = RESULTS_DIR / f"scores_{args.pair}.json"
    if not scores_path.exists():
        raise SystemExit(f"No scores at {scores_path}; run evaluation/score_all.py first")
    summary = json.loads(scores_path.read_text())

    datasets = list(summary["by_dataset"])
    # Sort: positive-gap datasets first (puts the wow story on the left)
    def has_gap(ds):
        m = summary["by_dataset"][ds]
        if not isinstance(m, dict) or "base" not in m or "thinking" not in m:
            return -1
        return m["thinking"]["accuracy"] - m["base"]["accuracy"]
    datasets.sort(key=has_gap, reverse=True)

    n_panels = len(datasets)
    # Variable widths: panels with more modes get wider
    widths = []
    for ds in datasets:
        n_modes = sum(1 for m in MODE_ORDER if m in summary["by_dataset"][ds])
        widths.append(max(4.5, 0.95 * n_modes))
    fig = plt.figure(figsize=(sum(widths) + 1.2, 5.4), facecolor="#0d1117")
    gs = fig.add_gridspec(1, n_panels, width_ratios=widths, wspace=0.22,
                          left=0.07, right=0.98, top=0.84, bottom=0.16)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
    for ax, ds in zip(axes, datasets):
        _panel(ax, ds, summary["by_dataset"][ds])

    # Pick title based on best dataset's recovery
    best_rec = 0
    best_ds = None
    for ds in datasets:
        m = summary["by_dataset"][ds]
        if isinstance(m, dict) and "base" in m and "thinking" in m and "full" in m:
            b, t, f_ = m["base"]["accuracy"], m["thinking"]["accuracy"], m["full"]["accuracy"]
            if t - b > 0.03:
                r = (f_ - b) / (t - b) * 100
                if r > best_rec:
                    best_rec = r
                    best_ds = ds
    if best_ds:
        title = (f"Hybrid recovers {best_rec:.0f} % of the base→thinking gap on "
                 f"{best_ds.upper()}  ·  {args.pair}")
        fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=0.96)
    else:
        fig.suptitle(f"Hybrid inference modes side-by-side  ·  {args.pair}",
                     color="white", fontsize=14, fontweight="bold", y=0.96)

    out = FIGURES_DIR / f"fig5_gap_recovery_{args.pair}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="#0d1117")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
