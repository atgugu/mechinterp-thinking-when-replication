"""Animated "live thinking" HTML: scrub through a rollout token-by-token.

Picks one MATH500 problem that has full / base / thinking completions, renders
all three side-by-side. The hybrid track highlights tokens that were steered
with their category color. Playable, pausable, scrubbable.

The output is a single self-contained HTML file — no server needed.
"""

import json
import sys
import argparse
import html as _html
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from const import HYBRID_DIR, SAES_DIR, FIGURES_DIR, MODEL_PAIRS, CATEGORY_PALETTE  # noqa: E402


def category_color(cid: int) -> str:
    if cid is None or cid < 0:
        return "#1f2937"
    return CATEGORY_PALETTE[cid % len(CATEGORY_PALETTE)]


def load_records(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                out[rec["id"]] = rec
            except Exception:
                continue
    return out


def pick_problem(full: dict, base: dict, think: dict) -> str | None:
    """Pick the problem with the most steering events that exists in all three modes."""
    best_id, best_steered = None, -1
    for pid, rec in full.items():
        if pid not in base or pid not in think:
            continue
        n_steered = sum(rec.get("steered_flags", []))
        if n_steered > best_steered:
            best_steered = n_steered
            best_id = pid
    return best_id


def render_token_strip(tokens: list[str], flags: list[int] | None = None,
                       cats: list[int] | None = None, default_color: str = "#475569"):
    """Render a list of HTML token spans with optional steering color."""
    spans: list[str] = []
    for i, t in enumerate(tokens):
        safe = _html.escape(t).replace("\n", "↵<br>")
        color = default_color
        title = ""
        if flags is not None and cats is not None and i < len(flags) and flags[i]:
            color = category_color(cats[i])
            title = f"steered → category {cats[i]}"
        spans.append(
            f'<span class="tok" data-i="{i}" '
            f'data-flag="{flags[i] if flags and i < len(flags) else 0}" '
            f'style="background:{color}1a;border-bottom:2px solid {color}" '
            f'title="{title}">{safe}</span>'
        )
    return "".join(spans)


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Live thinking — {pair}</title>
<style>
:root {{
  --bg: #0d1117; --fg: #e6edf3; --panel: #161b22; --border: #30363d;
  --muted: #8b949e; --accent: #f0b429;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--fg);
  font-family: 'DejaVu Sans', system-ui, sans-serif; line-height: 1.5; }}
header {{ padding: 24px 32px 16px; border-bottom: 1px solid var(--border); }}
header h1 {{ margin: 0 0 6px; font-size: 22px; }}
header p {{ margin: 0; color: var(--muted); font-size: 13px; }}
header .pid {{ color: var(--accent); font-family: monospace; }}
.controls {{
  display: flex; align-items: center; gap: 14px;
  padding: 14px 32px; background: var(--panel); border-bottom: 1px solid var(--border);
}}
.controls button {{
  padding: 7px 14px; border-radius: 6px; border: 1px solid var(--border);
  background: var(--bg); color: var(--fg); cursor: pointer; font-size: 13px;
}}
.controls button:hover {{ border-color: var(--accent); color: var(--accent); }}
.controls input[type=range] {{ flex: 1; accent-color: var(--accent); }}
.controls .step {{ font-family: monospace; color: var(--muted); font-size: 12px; min-width: 110px; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; padding: 24px 32px; }}
.col {{
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px 18px; height: 540px; overflow-y: auto; font-size: 14px;
}}
.col h3 {{ margin: 0 0 12px; font-size: 14px; display: flex; align-items: center; gap: 10px; }}
.col .badge {{ font-size: 10px; padding: 2px 8px; border-radius: 999px;
  background: rgba(255,255,255,0.06); color: var(--muted); font-weight: normal; }}
.col.base h3 {{ color: #64748b; }}
.col.hybrid h3 {{ color: var(--accent); }}
.col.thinking h3 {{ color: #34d399; }}
.tok {{ display: inline; transition: opacity 0.2s; padding: 0 1px; }}
.tok.future {{ opacity: 0.15; filter: blur(0.3px); }}
.legend {{
  display: flex; flex-wrap: wrap; gap: 10px; padding: 8px 32px 24px; font-size: 11px;
}}
.legend .item {{ display: flex; align-items: center; gap: 6px; }}
.legend .swatch {{ width: 12px; height: 12px; border-radius: 3px; }}
footer {{ padding: 16px 32px 28px; color: var(--muted); font-size: 12px;
  border-top: 1px solid var(--border); }}
</style>
</head>
<body>
<header>
  <h1>🎬 Live thinking · base vs hybrid vs thinking</h1>
  <p>Watch the same MATH500 problem (<span class="pid">{pid}</span>) decoded by all three models token-by-token.
  In the <em>Hybrid</em> column, each underline color shows which reasoning category fired at that step.</p>
</header>

<div class="controls">
  <button id="play">▶ Play</button>
  <button id="reset">⟳ Reset</button>
  <input type="range" id="scrub" min="0" max="{maxn}" value="0">
  <div class="step"><span id="cur">0</span> / {maxn} tokens</div>
</div>

<div class="grid">
  <div class="col base">
    <h3>Base only <span class="badge">{base_n} tokens · base failed</span></h3>
    <div id="track-base">{base_html}</div>
  </div>
  <div class="col hybrid">
    <h3>Hybrid (full) <span class="badge">{full_n} tokens · {n_steered} steered ({steered_pct:.1f}%)</span></h3>
    <div id="track-hybrid">{full_html}</div>
  </div>
  <div class="col thinking">
    <h3>Thinking only <span class="badge">{think_n} tokens · thinking succeeded</span></h3>
    <div id="track-thinking">{think_html}</div>
  </div>
</div>

<div class="legend">{legend}</div>

<footer>
  Replication of <a style="color:var(--accent)" href="https://arxiv.org/abs/2510.07364">Venhoff et al. 2025</a>
  &middot; the hybrid model uses the base model's weights with category-conditioned residual-stream
  steering at layer {steer_layer}. SAE trained at layer {sae_layer}, K={K}.
</footer>

<script>
const tracks = ['base','hybrid','thinking'].map(k => document.getElementById('track-'+k));
const tokens = tracks.map(t => t.querySelectorAll('.tok'));
const MAXN = {maxn};
const scrub = document.getElementById('scrub');
const cur = document.getElementById('cur');
let playing = false, timer = null;

function show(n) {{
  tokens.forEach(arr => {{
    arr.forEach((s, i) => {{
      if (i < n) s.classList.remove('future');
      else s.classList.add('future');
    }});
  }});
  cur.textContent = n;
  scrub.value = n;
}}
show(0);
scrub.addEventListener('input', e => show(parseInt(e.target.value)));
document.getElementById('reset').addEventListener('click', () => {{ playing = false; show(0); }});
document.getElementById('play').addEventListener('click', () => {{
  if (playing) {{ playing = false; clearInterval(timer); return; }}
  playing = true; let n = parseInt(scrub.value);
  timer = setInterval(() => {{
    n += 2;
    if (n > MAXN) {{ playing = false; clearInterval(timer); n = MAXN; }}
    show(n);
  }}, 60);
}});
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    p.add_argument("--dataset", default="auto",
                   help="auto picks the dataset with the most steering activity")
    p.add_argument("--problem_id", default=None)
    args = p.parse_args()

    pair = args.pair
    if args.dataset == "auto":
        best_ds, best_max = "math500", -1
        for ds in ("gsm8k", "math500"):
            p_full = HYBRID_DIR / f"{pair}_{ds}_full.jsonl"
            if not p_full.exists():
                continue
            with open(p_full) as _f:
                for line in _f:
                    r = json.loads(line)
                    ns = sum(r.get("steered_flags", []))
                    if ns > best_max:
                        best_max = ns
                        best_ds = ds
        args.dataset = best_ds
    full = load_records(HYBRID_DIR / f"{pair}_{args.dataset}_full.jsonl")
    base = load_records(HYBRID_DIR / f"{pair}_{args.dataset}_base.jsonl")
    think = load_records(HYBRID_DIR / f"{pair}_{args.dataset}_thinking.jsonl")
    pid = args.problem_id or pick_problem(full, base, think)
    if pid is None:
        raise SystemExit("Need outputs for all three modes (full / base / thinking)")

    print(f"Picked problem: {pid}")
    rec_full = full[pid]
    rec_base = base[pid]
    rec_think = think[pid]

    from transformers import AutoTokenizer
    cfg = MODEL_PAIRS[pair]
    tok = AutoTokenizer.from_pretrained(cfg["base"], local_files_only=True)

    def tokenize(text):
        return [tok.decode([tid], skip_special_tokens=False) for tid in tok.encode(text, add_special_tokens=False)]

    full_tokens = tokenize(rec_full["completion"])
    base_tokens = tokenize(rec_base["completion"])
    think_tokens = tokenize(rec_think["completion"])

    flags = rec_full.get("steered_flags", [])
    cats = rec_full.get("cat_trace", [])

    # Load labels for legend
    sae_dir = SAES_DIR / pair
    best = json.loads((sae_dir / "best_taxonomy.json").read_text())
    L, K = best["layer"], best["dict_size"]
    label_path = sae_dir / f"L{L}_K{K}_labels.json"
    labels = {}
    if label_path.exists():
        for c in json.loads(label_path.read_text()):
            labels[c["cluster_id"]] = c["label_info"]["label"]

    used_cats = sorted({c for f, c in zip(flags, cats) if f and c >= 0})
    legend_html = ""
    for cid in used_cats:
        col = category_color(cid)
        lab = labels.get(cid, f"c{cid}").replace("_", " ")
        legend_html += (
            f'<div class="item"><div class="swatch" style="background:{col}"></div>'
            f'<span>{cid} · {lab}</span></div>'
        )
    if not legend_html:
        legend_html = '<div class="item" style="color:var(--muted)">No categories triggered in this rollout.</div>'

    n_steered = sum(flags)
    maxn = max(len(full_tokens), len(base_tokens), len(think_tokens))

    out_html = HTML.format(
        pair=pair,
        pid=pid,
        steer_layer=cfg["steer_layer"],
        sae_layer=L,
        K=K,
        maxn=maxn,
        base_html=render_token_strip(base_tokens),
        full_html=render_token_strip(full_tokens, flags, cats),
        think_html=render_token_strip(think_tokens),
        base_n=len(base_tokens),
        full_n=len(full_tokens),
        think_n=len(think_tokens),
        n_steered=n_steered,
        steered_pct=100 * n_steered / max(len(full_tokens), 1),
        legend=legend_html,
    )
    safe_pid = pid.replace("/", "_").replace(".json", "")
    out_path = FIGURES_DIR / f"live_thinking_{pair}_{safe_pid}.html"
    out_path.write_text(out_html)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
