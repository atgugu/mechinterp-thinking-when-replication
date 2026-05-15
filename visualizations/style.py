"""Shared plotting style for the paper4 figures (matches paper2 / paper3 portfolio)."""

import matplotlib as mpl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from const import PLOT_BG, PLOT_FG, ACCENT_GOLD, ACCENT_AMBER, CATEGORY_PALETTE  # noqa: E402


def apply_dark_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": PLOT_BG,
        "axes.facecolor": PLOT_BG,
        "savefig.facecolor": PLOT_BG,
        "axes.edgecolor": PLOT_FG,
        "axes.labelcolor": PLOT_FG,
        "axes.titlecolor": PLOT_FG,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "axes.titleweight": "bold",
        "xtick.color": PLOT_FG,
        "ytick.color": PLOT_FG,
        "text.color": PLOT_FG,
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#30363d",
        "legend.labelcolor": PLOT_FG,
        "grid.color": "#30363d",
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=PLOT_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(color=PLOT_FG, family="DejaVu Sans, sans-serif"),
    margin=dict(l=60, r=30, t=70, b=50),
)


def category_color(cid: int) -> str:
    return CATEGORY_PALETTE[cid % len(CATEGORY_PALETTE)]
