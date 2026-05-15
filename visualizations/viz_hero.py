"""Hero/splash figure for the README — production-result edition.

Four-panel composite:
  A. Cluster sizes per SAE category
  B. The "n% map" — steering events on one real rollout (auto-picked)
  C. Gap recovery: best dataset with positive base→thinking gap
  D. Category labels + exemplar text
"""

import json
import sys
import re
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from const import (  # noqa: E402
    SAES_DIR, HYBRID_DIR, RESULTS_DIR, FIGURES_DIR,
    MODEL_PAIRS, ACCENT_GOLD, ACCENT_AMBER,
)
from visualizations.style import apply_dark_style, category_color  # noqa: E402


_FW_CHARS = re.compile(r"[｜ﾞﾟ◀▶▷◁<>]|<\|[^|]*\|>")


def _clean(s: str) -> str:
    s = _FW_CHARS.sub("", s)
    return "".join(ch for ch in s if ord(ch) < 0x2600)


def _pick_rollout_dataset(pair: str) -> str:
    best_ds, best_max = "math500", -1
    for ds in ("gsm8k", "math500"):
        path = HYBRID_DIR / f"{pair}_{ds}_full.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                ns = sum(r.get("steered_flags", []))
                if ns > best_max:
                    best_max = ns
                    best_ds = ds
    return best_ds


def _pick_rollout(pair: str, dataset: str):
    path = HYBRID_DIR / f"{pair}_{dataset}_full.jsonl"
    best_rec, best_ns = None, -1
    if path.exists():
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                ns = sum(r.get("steered_flags", []))
                if ns > best_ns:
                    best_ns = ns
                    best_rec = r
    return best_rec


def _pick_recovery_dataset(scores: dict) -> tuple[str | None, float]:
    best_ds, best_rec = None, -1.0
    for ds, by_mode in scores.get("by_dataset", {}).items():
        if not (isinstance(by_mode, dict) and {"base", "thinking", "full"} <= set(by_mode.keys())):
            continue
        b = by_mode["base"]["accuracy"]
        t = by_mode["thinking"]["accuracy"]
        h = by_mode["full"]["accuracy"]
        if t - b > 0.03:
            rec = (h - b) / (t - b) * 100
            if rec > best_rec:
                best_rec = rec
                best_ds = ds
    return best_ds, best_rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    args = p.parse_args()

    apply_dark_style()
    sae_dir = SAES_DIR / args.pair
    best = json.loads((sae_dir / "best_taxonomy.json").read_text())
    L, K = best["layer"], best["dict_size"]

    ex = json.loads((sae_dir / f"L{L}_K{K}_exemplars.json").read_text())
    label_path = sae_dir / f"L{L}_K{K}_labels.json"
    labels = ({c["cluster_id"]: c["label_info"]["label"]
               for c in json.loads(label_path.read_text())}
              if label_path.exists() else {})

    scores_path = RESULTS_DIR / f"scores_{args.pair}.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {"by_dataset": {}}
    rollout_ds = _pick_rollout_dataset(args.pair)
    rollout = _pick_rollout(args.pair, rollout_ds)
    best_ds, best_rec = _pick_recovery_dataset(scores)

    fig = plt.figure(figsize=(17, 11), facecolor="#0d1117")
    gs = GridSpec(3, 4, figure=fig,
                  height_ratios=[1.5, 1.5, 0.7],
                  width_ratios=[1, 1, 1, 1],
                  hspace=0.85, wspace=0.4,
                  left=0.05, right=0.98, top=0.88, bottom=0.05)

    # ── Header ──
    title_text = "Base models already know how to reason — they just need a nudge"
    fig.suptitle(title_text, color="white", fontsize=19, fontweight="bold", y=0.965)
    subt = f"Replication of arXiv:2510.07364  ·  {args.pair} pair"
    if best_ds:
        subt += f"  ·  {best_rec:.0f}% gap recovered on {best_ds.upper()}"
    fig.text(0.5, 0.925, subt, color="#94a3b8", fontsize=11, ha="center")

    # ── Panel A: cluster sizes ──
    axA = fig.add_subplot(gs[0, :2])
    sizes = [c["size"] for c in ex]
    cids = [c["cluster_id"] for c in ex]
    colors = [category_color(c) for c in cids]
    axA.bar(range(len(cids)), sizes, color=colors, edgecolor="#cbd5e1", linewidth=0.5)
    # Compact labels: drop the prefix word duplication and keep them readable
    tick_lbls = []
    for c in cids:
        lbl = labels.get(c, "").replace("_", " ")
        # Shorten common labels to fit
        short = {"deduction": "ded.", "backtracking": "backtr.",
                 "definition": "defn.", "uncategorized": "uncat.",
                 "subgoal setting": "subgl.", "case analysis": "case",
                 "verification": "verify"}.get(lbl, lbl[:7])
        tick_lbls.append(f"c{c}\n{short}")
    axA.set_xticks(range(len(cids)))
    axA.set_xticklabels(tick_lbls, fontsize=8.5, color="#e6edf3")
    for i, s in enumerate(sizes):
        axA.text(i, s + max(sizes) * 0.02, str(s),
                 ha="center", va="bottom", fontsize=8, color="white")
    axA.set_title(f"(A) Top-K SAE found {len(cids)} reasoning categories",
                  color=ACCENT_GOLD, fontsize=13, fontweight="bold", pad=10)
    axA.set_ylabel("sentences in cluster", fontsize=9.5)
    axA.set_ylim(0, max(sizes) * 1.16)
    axA.grid(axis="y", alpha=0.2, linestyle="--")
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)

    # ── Panel B: steering trace ──
    axB = fig.add_subplot(gs[0, 2:])
    if rollout is not None:
        flags = rollout.get("steered_flags", [])
        cats = rollout.get("cat_trace", [])
        cols = 80
        n = len(flags)
        rows = (n + cols - 1) // cols
        for i, (f_, c_) in enumerate(zip(flags, cats)):
            r_ = i // cols
            col_ = i % cols
            color = category_color(c_) if f_ else "#252e3b"
            axB.add_patch(patches.Rectangle((col_, -r_ - 1), 1, 1,
                                             facecolor=color, edgecolor="#0d1117", lw=0.18))
        axB.set_xlim(0, cols)
        axB.set_ylim(-rows - 0.5, 0.5)
        axB.set_aspect("equal")
        n_steered = sum(flags)
        pct = 100 * n_steered / max(n, 1)
        axB.set_title(
            f"(B) {n_steered} of {n} tokens steered "
            f"({pct:.1f}%)  ·  one {rollout_ds.upper()} rollout",
            color=ACCENT_GOLD, fontsize=13, fontweight="bold", pad=10)
    else:
        axB.text(0.5, 0.5, "(B) no hybrid rollout yet", ha="center", va="center",
                 color="#94a3b8", fontsize=12, transform=axB.transAxes)
        axB.set_title("(B) Steering trace", color=ACCENT_GOLD,
                      fontsize=13, fontweight="bold", pad=10)
    axB.axis("off")

    # ── Panel C: gap recovery for best dataset ──
    axC = fig.add_subplot(gs[1, :2])
    target_ds = best_ds or "math500"
    by_mode = scores.get("by_dataset", {}).get(target_ds, {})
    mode_order = ["base", "only_bias", "random_firing", "random_vectors", "full", "thinking"]
    mode_labels = {"base": "Base", "only_bias": "Only\nbias",
                   "random_firing": "Random\nfiring", "random_vectors": "Random\nvectors",
                   "full": "Hybrid", "thinking": "Thinking"}
    mode_colors = {"base": "#475569", "only_bias": "#94a3b8",
                   "random_firing": "#a78bfa", "random_vectors": "#c084fc",
                   "full": ACCENT_GOLD, "thinking": "#34d399"}
    accs, lbls, cols2 = [], [], []
    for m in mode_order:
        if m in by_mode:
            accs.append(by_mode[m]["accuracy"])
            lbls.append(mode_labels[m])
            cols2.append(mode_colors[m])
    if accs:
        x = np.arange(len(accs))
        axC.bar(x, accs, color=cols2, edgecolor="#cbd5e1", linewidth=0.5, width=0.72)
        for xi, a in zip(x, accs):
            axC.text(xi, a + 0.015, f"{a*100:.0f}%", ha="center", va="bottom",
                     fontsize=9.5, color="white", fontweight="bold")
        axC.set_xticks(x)
        axC.set_xticklabels(lbls, fontsize=9, color="#e6edf3")
        axC.set_ylim(0, max(accs) * 1.32)
        # Recovery arrow if applicable
        if best_ds == target_ds and "base" in by_mode and "full" in by_mode:
            b_ = by_mode["base"]["accuracy"]
            f_ = by_mode["full"]["accuracy"]
            x_b = mode_order.index("base")
            x_f = mode_order.index("full")
            axC.annotate("", xy=(x_f, f_), xytext=(x_b, b_),
                         arrowprops=dict(arrowstyle="->", color=ACCENT_AMBER, lw=2,
                                         connectionstyle="arc3,rad=-0.25"))
            axC.text((x_b + x_f) / 2, max(b_, f_) + 0.08,
                     f"{best_rec:.0f}% recovered",
                     color=ACCENT_AMBER, ha="center", fontsize=10, fontweight="bold",
                     bbox=dict(facecolor="#0d1117", edgecolor=ACCENT_AMBER,
                               alpha=0.95, boxstyle="round,pad=0.3", lw=1.2))
    else:
        axC.text(0.5, 0.5, "(no scores yet)", ha="center", va="center",
                 color="#94a3b8", transform=axC.transAxes)
    axC.set_ylabel(f"{target_ds.upper()} accuracy", fontsize=9.5)
    axC.set_title(f"(C) Gap recovery on {target_ds.upper()}",
                  color=ACCENT_GOLD, fontsize=13, fontweight="bold", pad=10)
    axC.grid(axis="y", alpha=0.2, linestyle="--")
    axC.spines["top"].set_visible(False)
    axC.spines["right"].set_visible(False)

    # ── Panel D: category exemplars ──
    axD = fig.add_subplot(gs[1, 2:])
    axD.axis("off")
    axD.set_title("(D) What each reasoning category looks like",
                  color=ACCENT_GOLD, fontsize=13, fontweight="bold", pad=10, loc="left")
    top4 = sorted(ex, key=lambda c: -c["size"])[:4]
    y = 0.92
    for c in top4:
        cid = c["cluster_id"]
        lbl = labels.get(cid, f"c{cid}").replace("_", " ")
        color = category_color(cid)
        first_ex = (c.get("exemplars") or ["(none)"])[0]
        first_ex = _clean(first_ex)
        if len(first_ex) > 105:
            first_ex = first_ex[:102] + "…"
        axD.add_patch(patches.Rectangle((0.0, y - 0.04), 0.024, 0.08,
                                         facecolor=color, transform=axD.transAxes))
        axD.text(0.04, y, f"c{cid} · {lbl}  ({c['size']:,} sentences)",
                 color="white", fontsize=11, fontweight="bold", transform=axD.transAxes,
                 va="center")
        axD.text(0.04, y - 0.09, f'"{first_ex}"',
                 color="#cbd5e1", fontsize=9.5, transform=axD.transAxes,
                 va="center", style="italic")
        y -= 0.22

    # ── Bottom takeaway ──
    axE = fig.add_subplot(gs[2, :])
    axE.axis("off")
    axE.text(0.5, 0.65,
             "Train a Top-K SAE on thinking-model traces · "
             "inject the right category vector at the right moment · "
             "recover a real fraction of the gap.",
             color="white", fontsize=13, ha="center", style="italic",
             transform=axE.transAxes)
    axE.text(0.5, 0.18,
             "github.com/atgugu/mechinterp-thinking-when-replication  ·  fp16",
             color="#94a3b8", fontsize=10, ha="center", transform=axE.transAxes)

    out = FIGURES_DIR / f"hero_{args.pair}.png"
    fig.savefig(out, facecolor="#0d1117", dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
