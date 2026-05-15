"""Interactive HTML explorer for the discovered SAE categories.

Renders a single self-contained HTML file where the viewer can:
  - Browse each reasoning category (one tab/section per cluster)
  - See the auto-assigned label, exemplar count, top-firing sentences
  - Get an at-a-glance summary of WHAT each "reasoning primitive" sounds like

Pure HTML/CSS/JS — no server needed. Looks great on a dark background and
mirrors the paper2 `feature_explorer.html` interaction style.
"""

import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from const import SAES_DIR, FIGURES_DIR, MODEL_PAIRS, CATEGORY_PALETTE  # noqa: E402


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Reasoning Category Explorer — {pair}</title>
<style>
  :root {{
    --bg: #0d1117;
    --fg: #e6edf3;
    --panel: #161b22;
    --muted: #8b949e;
    --border: #30363d;
    --accent: #f0b429;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0;
    background: var(--bg); color: var(--fg);
    font-family: 'DejaVu Sans', system-ui, sans-serif;
    line-height: 1.55;
  }}
  header {{
    padding: 28px 36px 18px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, #161b22 0%, var(--bg) 100%);
  }}
  header h1 {{ margin: 0 0 8px; font-size: 24px; }}
  header p {{ margin: 0; color: var(--muted); font-size: 13px; }}
  .tabs {{
    display: flex; flex-wrap: wrap; gap: 6px;
    padding: 16px 36px; border-bottom: 1px solid var(--border);
    position: sticky; top: 0; background: var(--bg); z-index: 10;
  }}
  .tab {{
    padding: 8px 14px; border-radius: 999px; cursor: pointer; border: 1px solid var(--border);
    background: var(--panel); color: var(--fg); font-size: 12px; transition: 0.15s;
    display: flex; align-items: center; gap: 8px;
  }}
  .tab .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .tab:hover {{ border-color: var(--accent); }}
  .tab.active {{ background: var(--fg); color: var(--bg); border-color: var(--fg); }}
  main {{ padding: 28px 36px 80px; }}
  .panel {{
    display: none;
    border: 1px solid var(--border); border-radius: 12px;
    padding: 24px; background: var(--panel);
  }}
  .panel.active {{ display: block; }}
  .panel h2 {{ margin: 0 0 6px; font-size: 22px; display: flex; align-items: center; gap: 12px; }}
  .panel h2 .pill {{
    font-size: 11px; padding: 3px 10px; border-radius: 999px;
    border: 1px solid currentColor; opacity: 0.85;
  }}
  .panel .meta {{ color: var(--muted); margin-bottom: 24px; font-size: 13px; }}
  .exemplars {{ display: grid; gap: 10px; }}
  .ex {{
    padding: 12px 16px; border-radius: 8px;
    background: rgba(255,255,255,0.03);
    border-left: 3px solid var(--accent);
    font-size: 14px;
  }}
  .ex .activation {{
    float: right; font-size: 11px; color: var(--muted);
    font-family: monospace; padding-left: 12px;
  }}
  footer {{
    padding: 20px 36px; color: var(--muted); font-size: 12px;
    border-top: 1px solid var(--border);
  }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<header>
  <h1>🧠 Reasoning Category Explorer</h1>
  <p>
    Each cluster below is a feature discovered by a Top-K=3 sparse autoencoder
    trained on {n_sent} sentence-pooled residuals from <strong>{pair}</strong> thinking-model
    rollouts. Click a tab to browse the top exemplars of that "reasoning primitive."
  </p>
</header>

<div class="tabs">{tabs}</div>

<main>{panels}</main>

<footer>
  Replication of Venhoff et al. 2025
  (<a href="https://arxiv.org/abs/2510.07364">arXiv:2510.07364</a>) &middot;
  layer {layer}, dict size {dict_size}, Top-K=3, unit-norm decoder
</footer>

<script>
  const tabs = document.querySelectorAll('.tab');
  const panels = document.querySelectorAll('.panel');
  tabs.forEach(t => t.addEventListener('click', () => {{
    tabs.forEach(x => x.classList.remove('active'));
    panels.forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById(t.dataset.target).classList.add('active');
  }}));
</script>
</body>
</html>
"""


def category_color(cid: int) -> str:
    return CATEGORY_PALETTE[cid % len(CATEGORY_PALETTE)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    args = p.parse_args()

    sae_dir = SAES_DIR / args.pair
    best = json.loads((sae_dir / "best_taxonomy.json").read_text())
    L, K = best["layer"], best["dict_size"]
    ex_path = sae_dir / f"L{L}_K{K}_exemplars.json"
    lbl_path = sae_dir / f"L{L}_K{K}_labels.json"
    if not ex_path.exists():
        raise SystemExit(f"Missing {ex_path}; run score_taxonomy first")
    clusters = json.loads(ex_path.read_text())
    labels = ({c["cluster_id"]: c["label_info"]
               for c in json.loads(lbl_path.read_text())}
              if lbl_path.exists() else {})

    total_sentences = sum(c.get("size", 0) for c in clusters)

    tabs_html = []
    panels_html = []
    for i, c in enumerate(clusters):
        cid = c["cluster_id"]
        color = category_color(cid)
        lbl_info = labels.get(cid, {"label": f"c{cid}", "confidence": 0})
        label_name = lbl_info["label"].replace("_", " ").title()
        is_active = "active" if i == 0 else ""
        tabs_html.append(
            f'<div class="tab {is_active}" data-target="panel-{cid}" '
            f'style="border-color:{color}">'
            f'<span class="dot" style="background:{color}"></span>'
            f'<strong>{cid}</strong> · {label_name}</div>'
        )
        exemplars_html = ""
        for ex in c.get("exemplars", []):
            ex_safe = (ex.replace("&", "&amp;").replace("<", "&lt;")
                          .replace(">", "&gt;").replace("\n", "<br>"))
            exemplars_html += f'<div class="ex" style="border-left-color:{color}">{ex_safe}</div>'
        if not exemplars_html:
            exemplars_html = '<div class="ex">(no exemplars)</div>'
        panels_html.append(
            f'<div class="panel {is_active}" id="panel-{cid}" style="border-color:{color}">'
            f'<h2 style="color:{color}">'
            f'  Category {cid}: {label_name} '
            f'  <span class="pill">{c.get("size", 0)} sentences · '
            f'  confidence {lbl_info.get("confidence", 0):.2f}</span>'
            f'</h2>'
            f'<div class="meta">Top exemplars (sorted by SAE activation magnitude):</div>'
            f'<div class="exemplars">{exemplars_html}</div>'
            f'</div>'
        )

    out_path = FIGURES_DIR / f"category_explorer_{args.pair}.html"
    html = HTML_TEMPLATE.format(
        pair=args.pair,
        n_sent=total_sentences,
        layer=L,
        dict_size=K,
        tabs="".join(tabs_html),
        panels="".join(panels_html),
    )
    out_path.write_text(html)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
