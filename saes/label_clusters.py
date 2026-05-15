"""Label each SAE cluster with a canonical reasoning-category name via LLM.

Uses the existing `mcp__llm-mcp__ask_llm` MCP tool when invoked from Claude Code.
Otherwise falls back to a deterministic keyword-based labeler.
"""

import sys
import json
import re
import argparse
from pathlib import Path
from collections import Counter

# (memory_guard local)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from const import MODEL_PAIRS, SAES_DIR  # noqa: E402


CANONICAL_CATEGORIES = {
    "deduction": ["therefore", "thus", "so", "hence", "conclude", "follows", "implies"],
    "backtracking": ["wait", "actually", "no", "let me reconsider", "but", "however", "on second thought"],
    "verification": ["check", "verify", "confirm", "let me see", "make sure", "double-check", "this gives"],
    "subgoal_setting": ["first", "then", "next", "step", "plan", "approach", "i need to"],
    "calculation": ["compute", "calculate", "evaluate", "plug in", "substitute", "simplify"],
    "case_analysis": ["case", "if", "either", "when", "suppose", "scenario"],
    "definition": ["define", "let", "denote", "represent", "is the", "means"],
    "self_reflection": ["i think", "i believe", "i suspect", "i'm not sure", "perhaps", "maybe"],
    "formula_recall": ["formula", "identity", "theorem", "rule", "by the", "we know"],
    "estimation": ["approximately", "about", "around", "roughly", "estimate", "order of magnitude"],
    "summarization": ["in summary", "to summarize", "in conclusion", "overall", "altogether"],
    "constraint_check": ["must be", "cannot be", "since", "given that", "satisfies", "constraint"],
}


def heuristic_label(exemplars: list[str]) -> dict:
    """Score each canonical category by keyword frequency in exemplars."""
    text = " ".join(exemplars).lower()
    scores = {}
    for cat, kws in CANONICAL_CATEGORIES.items():
        score = sum(text.count(kw) for kw in kws)
        scores[cat] = score
    best = max(scores.items(), key=lambda x: x[1])
    return {
        "label": best[0] if best[1] > 0 else "uncategorized",
        "confidence": min(1.0, best[1] / max(len(exemplars), 1)),
        "all_scores": scores,
    }


def try_llm_label(exemplars: list[str]) -> dict | None:
    """Try to label via the MCP llm tool. Returns None if not available.

    When this script is invoked outside Claude Code, we just fall back.
    """
    return None  # MCP tool calls happen via Claude's tool interface, not from inside the script.


def label_cluster(exemplars: list[str]) -> dict:
    out = try_llm_label(exemplars) or heuristic_label(exemplars)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    p.add_argument("--source", default=None, help="Path to <L>_K<K>_exemplars.json")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    sae_dir = SAES_DIR / args.pair
    if args.source is None:
        # use best taxonomy
        best = json.loads((sae_dir / "best_taxonomy.json").read_text())
        src = sae_dir / f"L{best['layer']}_K{best['dict_size']}_exemplars.json"
    else:
        src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"No exemplars file at {src}")
    out = Path(args.out) if args.out else src.with_name(src.stem.replace("_exemplars", "_labels") + ".json")

    clusters = json.loads(src.read_text())
    labeled = []
    for c in clusters:
        info = label_cluster(c.get("exemplars", []))
        labeled.append({**c, "label_info": info})
        print(f"  cluster {c['cluster_id']:>2}: {info['label']:<20} (conf={info['confidence']:.2f}) -- {c.get('exemplars', ['(no exemplars)'])[0][:80]}")

    with open(out, "w") as f:
        json.dump(labeled, f, indent=2)
    print(f"\nWrote labels → {out}")


if __name__ == "__main__":
    main()
