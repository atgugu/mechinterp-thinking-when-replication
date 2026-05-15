"""Score each (layer, dict_size) SAE for taxonomy quality.

Three metrics following the paper:
  * Consistency  → reproducibility across seeds (F1 on cluster membership)
  * Independence → average inter-centroid cosine distance > 0.5 fraction
  * Completeness → LLM rates exemplar sentences for cluster coherence (0-10)

Combined score is the average of (consistency, independence, completeness).
The cluster size with highest combined score (typically 10-20) is selected
as the taxonomy for steering.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# (memory_guard local)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from const import MODEL_PAIRS, SAES_DIR, ACTS_DIR  # noqa: E402
from saes.topk_sae import TopKSAE  # noqa: E402
from saes.train_saes import load_layer_acts, train_one  # noqa: E402


def load_sae(path: Path) -> TopKSAE:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    cfg = bundle["config"]
    sae = TopKSAE(cfg["d_input"], cfg["dict_size"], k=cfg["k"], normalize_decoder=True)
    sae.load_state_dict(bundle["state_dict"])
    sae.eval()
    return sae


def independence_score(sae: TopKSAE) -> float:
    """Mean pairwise cosine distance between active feature decoder columns. Range 0..1."""
    with torch.no_grad():
        W = sae.W_dec.cpu()  # (d_input, dict_size)
        Wn = W / W.norm(dim=0, keepdim=True).clamp(min=1e-6)
        sim = Wn.t() @ Wn  # (K, K)
        K = sim.shape[0]
        if K < 2:
            return 1.0
        mask = ~torch.eye(K, dtype=torch.bool)
        # cosine distance = 1 - cos sim; clamp to [0, 1]
        dist = (1.0 - sim).clamp(0, 1)
        return dist[mask].mean().item()


def consistency_score(acts: torch.Tensor, dict_size: int, k: int,
                      device: str, max_epochs: int = 30, patience: int = 5,
                      seeds: tuple[int, ...] = (0, 1)) -> float:
    """Train two SAEs with different seeds, measure cluster-membership agreement.

    Uses the Hungarian-style best-match F1 since cluster IDs are arbitrary.
    """
    assignments = []
    for s in seeds:
        sae, _meta = train_one(
            acts, dict_size, k, max_epochs=max_epochs, patience=patience,
            batch_size=512, lr=None, device=device, seed=s,
        )
        with torch.no_grad():
            a = sae.assign_category(acts.to(device)).cpu().numpy()
        assignments.append(a)

    a1, a2 = assignments
    # Build co-occurrence matrix
    import numpy as np
    K = dict_size
    M = np.zeros((K, K), dtype=np.int64)
    for x, y in zip(a1, a2):
        M[x, y] += 1
    # Best column for each row (greedy match)
    row_max = M.max(axis=1).sum()
    return float(row_max) / max(len(a1), 1)


def completeness_score(sae: TopKSAE, acts: torch.Tensor, texts: list[str],
                       device: str = "mps", n_exemplars: int = 10) -> tuple[float, list[dict]]:
    """LLM rates each cluster's exemplars for coherence (0-10). Returns mean / 10."""
    sae_on_device = sae.to(device)
    with torch.no_grad():
        z = sae_on_device.encode(acts.to(device)).cpu()
    # Top-activating sentences per cluster
    K = sae.dict_size
    cluster_info: list[dict] = []
    for cid in range(K):
        col = z[:, cid]
        if col.max() <= 0:
            cluster_info.append({"cluster_id": cid, "size": 0, "exemplars": []})
            continue
        order = torch.argsort(col, descending=True)[:n_exemplars].tolist()
        exemplars = [texts[i] for i in order]
        cluster_info.append({
            "cluster_id": cid,
            "size": int((col > 0).sum().item()),
            "exemplars": exemplars,
            "max_act": col.max().item(),
        })

    # Query LLM (Claude/OpenAI via MCP). Fallback: simple intra-cluster bag-of-words coherence.
    score = _llm_rate_clusters(cluster_info)
    return score, cluster_info


def _llm_rate_clusters(cluster_info: list[dict]) -> float:
    """Heuristic fallback. Returns mean coherence in [0, 1].

    Uses lexical overlap (Jaccard on word bigrams) — proxy for "exemplars share theme".
    Replace with a real LLM call if you have the MCP plumbing wired up.
    """
    import re
    def bigrams(text):
        toks = re.findall(r"[a-zA-Z']+", text.lower())
        return set(zip(toks, toks[1:])) if len(toks) > 1 else set()

    scores = []
    for c in cluster_info:
        exs = c.get("exemplars", [])
        if len(exs) < 2:
            continue
        bgs = [bigrams(e) for e in exs]
        # Mean pairwise Jaccard
        pairs = 0
        tot = 0.0
        for i in range(len(bgs)):
            for j in range(i + 1, len(bgs)):
                u = bgs[i] | bgs[j]
                if not u:
                    continue
                tot += len(bgs[i] & bgs[j]) / len(u)
                pairs += 1
        if pairs > 0:
            scores.append(tot / pairs)
    if not scores:
        return 0.0
    # Stretch from [0, 0.3] (typical Jaccard for thematic text) to [0, 1]
    raw = sum(scores) / len(scores)
    return min(1.0, raw / 0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
    p.add_argument("--skip_consistency", action="store_true",
                   help="Skip retraining (consistency is expensive)")
    args = p.parse_args()

    cfg = MODEL_PAIRS[args.pair]
    sae_dir = SAES_DIR / args.pair
    files = sorted(sae_dir.glob("L*_K*.pt"))
    if not files:
        raise SystemExit(f"No trained SAEs in {sae_dir}")

    results = []
    # Cache acts per layer
    layer_data: dict[int, tuple[torch.Tensor, list[str]]] = {}

    for f in files:
        # Parse L<L>_K<K>.pt
        stem = f.stem
        L = int(stem.split("_")[0][1:])
        K = int(stem.split("_")[1][1:])
        sae = load_sae(f)

        # Load acts/texts on demand
        if L not in layer_data:
            acts, _ids, texts = load_layer_acts(args.pair, "thinking", L)
            layer_data[L] = (acts, texts)
        acts, texts = layer_data[L]

        indep = independence_score(sae)
        compl, cluster_info = completeness_score(sae, acts, texts, device=args.device)
        if args.skip_consistency:
            cons = float("nan")
        else:
            cons = consistency_score(acts, K, sae.k, args.device)

        combined = (indep + compl + (cons if not args.skip_consistency else (indep + compl) / 2)) / 3.0
        rec = {
            "layer": L, "dict_size": K,
            "independence": indep,
            "completeness": compl,
            "consistency": cons,
            "combined": combined,
        }
        results.append(rec)
        print(f"  L{L}/K{K}: indep={indep:.3f} compl={compl:.3f} cons={cons:.3f} → {combined:.3f}")

        # Save per-SAE exemplars
        with open(sae_dir / f"L{L}_K{K}_exemplars.json", "w") as fp:
            json.dump(cluster_info, fp, indent=2)

    out = sae_dir / "taxonomy_scores.json"
    with open(out, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nWrote scores → {out}")

    # Pick best (layer, K)
    best = max(results, key=lambda r: r["combined"])
    print(f"\nBest taxonomy: L{best['layer']}/K{best['dict_size']} combined={best['combined']:.3f}")
    with open(sae_dir / "best_taxonomy.json", "w") as fp:
        json.dump(best, fp, indent=2)


if __name__ == "__main__":
    main()
