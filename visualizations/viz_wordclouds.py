"""Figure 4: per-category exemplar wordclouds.

One wordcloud per discovered category, in a grid. Plain-language summary of
what the SAE found, no math jargon.
"""

import json
import sys
import re
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from wordcloud import WordCloud

from const import SAES_DIR, FIGURES_DIR, MODEL_PAIRS  # noqa: E402
from visualizations.style import apply_dark_style, category_color  # noqa: E402


_STOPWORDS = set("""
the a an of and or to in for on with at by from is are was were be been being have has had
do does did this that these those it its as not no but if then so than which who whom whose
i we you he she they me him her us them my our your their its his hers our theirs
will would shall should may might can could just only also too very more most less
yes no thus hence therefore however moreover meanwhile here there where when why how
okay got would could should one two three four five six seven eight nine ten thing things
make makes made take takes took look looks looked thought thinks thinking know knew
say said says way ways ok well still really probably about give given let lets
""".split())


def clean_text(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-z']+", text.lower())
    return [t for t in toks if t not in _STOPWORDS and len(t) > 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    args = p.parse_args()

    apply_dark_style()
    sae_dir = SAES_DIR / args.pair
    best = json.loads((sae_dir / "best_taxonomy.json").read_text())
    L, K = best["layer"], best["dict_size"]
    src = sae_dir / f"L{L}_K{K}_exemplars.json"
    if not src.exists():
        raise SystemExit(f"Missing exemplars {src}; run score_taxonomy first")
    clusters = json.loads(src.read_text())

    # Optional labels
    label_path = sae_dir / f"L{L}_K{K}_labels.json"
    labels = {c["cluster_id"]: c["label_info"]["label"]
              for c in json.loads(label_path.read_text())} if label_path.exists() else {}

    # Compute global word frequencies to identify "generic" terms across all clusters
    from collections import Counter
    global_counts: Counter = Counter()
    cluster_word_lists: list[list[str]] = []
    for c in clusters:
        words: list[str] = []
        for ex in c.get("exemplars", []):
            words.extend(clean_text(ex))
        cluster_word_lists.append(words)
        global_counts.update(set(words))  # set → count clusters containing each word
    n_total = max(1, sum(1 for w in cluster_word_lists if w))
    # Word appears in >50 % of clusters → consider it generic, downweight
    generic = {w for w, c in global_counts.items() if c > n_total * 0.5}

    n = K
    # 2-row layouts give tight grids; force a square-ish aspect ratio per panel
    if n <= 4:
        ncols = n
    elif n <= 6:
        ncols = 3
    elif n <= 8:
        ncols = 4
    elif n <= 12:
        ncols = (n + 1) // 2
    else:
        ncols = (n + 2) // 3
    nrows = (n + ncols - 1) // ncols
    # Per-panel size: 4" wide × 3" tall — wordclouds look better with rectangular tiles
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows),
                              squeeze=False, facecolor="#0d1117")

    for i, c in enumerate(clusters):
        ax = axes[i // ncols][i % ncols]
        ax.axis("off")
        cid = c["cluster_id"]
        words = cluster_word_lists[i] if i < len(cluster_word_lists) else []
        # Downweight generic words by 0.3 (instead of removing entirely)
        filtered = [w for w in words if w not in generic] + \
                   [w for w in words if w in generic][: max(0, len(words) // 6)]
        if not filtered:
            ax.set_title(f"c{cid}: no terms", color="#94a3b8", fontsize=10)
            continue
        text_blob = " ".join(filtered)
        wc = WordCloud(
            width=900, height=600,
            background_color="#0d1117",
            color_func=lambda *args_, color=category_color(cid), **kw: color,
            max_words=24, prefer_horizontal=0.92,
            relative_scaling=0.45,
            margin=6,
            min_font_size=10,
        ).generate(text_blob)
        ax.imshow(wc, interpolation="bilinear")
        lbl = labels.get(cid, f"c{cid}")
        for spine in ax.spines.values():
            spine.set_edgecolor(category_color(cid))
            spine.set_linewidth(1.5)
        ax.set_title(f"c{cid} · {lbl.replace('_', ' ').title()}    {c['size']:,} sentences",
                     color="white", fontsize=11.5, pad=8, fontweight="bold")

    # Hide unused panels
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"What each reasoning category sounds like  ·  "
                 f"{args.pair}  ·  L{L}, K={K}",
                 color="white", fontsize=16, fontweight="bold", y=0.995)
    out = FIGURES_DIR / f"fig4_wordclouds_{args.pair}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, facecolor="#0d1117", dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
