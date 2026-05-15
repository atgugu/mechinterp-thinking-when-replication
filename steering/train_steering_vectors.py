"""Train one steering vector per discovered category (Dunefsky & Cohan loss).

For each category c discovered by the best SAE:
  1. Collect ~STEERING_SENTENCES_PER_CAT sentences from thinking-model rollouts
     whose top-firing SAE feature is c.
  2. For each sentence, build (context, continuation) pairs from the rollout.
  3. Train v_c by gradient descent: minimize NLL_T - λ·NLL_B with v_c injected
     into both models at the same layer.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

# (memory_guard imported from local package root)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["TRANSFORMERLENS_ALLOW_MPS"] = "1"

import torch  # noqa: E402

from memory_guard import init_memory_guard, check_memory  # noqa: E402
from const import (  # noqa: E402
    MODEL_PAIRS, SAES_DIR, STEERING_DIR, ROLLOUTS_DIR,
    STEERING_LAMBDA, STEERING_LR, STEERING_EPOCHS, STEERING_SENTENCES_PER_CAT,
)
from saes.topk_sae import TopKSAE  # noqa: E402
from saes.train_saes import load_layer_acts  # noqa: E402
from steering.dunefsky_loss import (  # noqa: E402
    SteeringHook, SteeringVector, teacher_forced_nll,
)


def load_models(cfg, dtype=torch.float16, device=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["base"], local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base"], dtype=dtype, device_map={"": device}, local_files_only=True,
        attn_implementation="sdpa",
    )
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    thinking = AutoModelForCausalLM.from_pretrained(
        cfg["thinking"], dtype=dtype, device_map={"": device}, local_files_only=True,
        attn_implementation="sdpa",
    )
    thinking.eval()
    for p in thinking.parameters():
        p.requires_grad_(False)

    return base, thinking, tok


def collect_category_examples(pair: str, layer: int, dict_size: int,
                              sae_path: Path, max_per_cat: int = STEERING_SENTENCES_PER_CAT):
    """Return dict[category_id, list[(rollout_id, sentence_idx, sentence_text)]]."""
    # Load SAE
    bundle = torch.load(sae_path, map_location="cpu", weights_only=False)
    cfg = bundle["config"]
    sae = TopKSAE(cfg["d_input"], cfg["dict_size"], k=cfg["k"], normalize_decoder=True)
    sae.load_state_dict(bundle["state_dict"])
    sae.eval()

    acts, ids, texts = load_layer_acts(pair, "thinking", layer)
    with torch.no_grad():
        assignments = sae.assign_category(acts).cpu().numpy()
        max_acts = sae.encode(acts).max(dim=-1).values.cpu().numpy()

    # Bucket
    by_cat: dict[int, list[tuple[float, str, str]]] = {c: [] for c in range(cfg["dict_size"])}
    for sid, cat, ma, txt in zip(ids, assignments, max_acts, texts):
        by_cat[int(cat)].append((float(ma), sid, txt))

    # Top-K most-activating per category
    out: dict[int, list[tuple[str, str]]] = {}
    for cat, items in by_cat.items():
        items.sort(key=lambda x: -x[0])
        out[cat] = [(sid, txt) for _ma, sid, txt in items[:max_per_cat]]
    return out, sae


def load_rollout_text(rollout_path: Path) -> dict[str, str]:
    """Map rollout id → full completion text."""
    out = {}
    with open(rollout_path) as f:
        for line in f:
            rec = json.loads(line)
            out[rec["id"]] = rec["completion"]
    return out


def build_training_pairs(
    category_sentences: dict[int, list[tuple[str, str]]],
    rollouts: dict[str, str], tokenizer,
    max_context_tokens: int = 192, max_continuation_tokens: int = 80,
    max_per_cat: int = 200,
):
    """For each category, build a list of (input_ids, continuation_start) pairs.

    Truncate the context from the left so the continuation always fits.
    """
    pairs: dict[int, list[tuple[torch.Tensor, int]]] = {}
    total_kept, total_dropped = 0, 0
    for cat, sents in category_sentences.items():
        cat_pairs = []
        for rid_idx, txt in sents[:max_per_cat]:
            rid = rid_idx.split("::s")[0] if "::s" in rid_idx else rid_idx
            full = rollouts.get(rid)
            if not full:
                total_dropped += 1
                continue
            pos = full.find(txt)
            if pos < 0:
                total_dropped += 1
                continue
            context = full[:pos]
            continuation = txt

            # 1) Tokenize the continuation first
            cont_enc = tokenizer(continuation, add_special_tokens=False, truncation=True,
                                 max_length=max_continuation_tokens)
            cont_ids = cont_enc["input_ids"]
            if len(cont_ids) < 2:
                total_dropped += 1
                continue

            # 2) Tokenize the context, KEEP last `max_context_tokens` so we don't lose recent context
            ctx_enc = tokenizer(context, add_special_tokens=False)
            ctx_ids = ctx_enc["input_ids"][-max_context_tokens:]
            if len(ctx_ids) < 2:
                total_dropped += 1
                continue

            # 3) Combine
            ids = torch.tensor(ctx_ids + cont_ids, dtype=torch.long)
            cont_start = len(ctx_ids)
            if cont_start >= len(ids) - 1:
                total_dropped += 1
                continue
            cat_pairs.append((ids, cont_start))
            total_kept += 1
        if cat_pairs:
            pairs[cat] = cat_pairs
    print(f"  Training pairs: kept {total_kept}, dropped {total_dropped}")
    return pairs


def train_steering_vector(
    base, thinking, base_layer, thinking_layer, d_model: int,
    pairs: list[tuple[torch.Tensor, int]],
    lr: float, epochs: int, lam: float,
    device: str = "mps",
):
    sv = SteeringVector(d_model).to(device)
    sv.v.data = sv.v.data.to(torch.float32)
    opt = torch.optim.AdamW(sv.parameters(), lr=lr)

    history = []
    for ep in range(epochs):
        running_t, running_b, n = 0.0, 0.0, 0
        for ids, cont_start in pairs:
            ids_b = ids.unsqueeze(0).to(device)

            hook_t = SteeringHook(sv())
            hook_t.install(thinking.model.layers[thinking_layer])
            hook_b = SteeringHook(sv())
            hook_b.install(base.model.layers[base_layer])

            try:
                nll_t = teacher_forced_nll(thinking, ids_b, cont_start)
                nll_b = teacher_forced_nll(base, ids_b, cont_start)
                if nll_t is None or nll_b is None:
                    continue
                loss = nll_t - lam * nll_b  # maximize p_T, minimize p_B
                opt.zero_grad()
                loss.backward()
                opt.step()
            finally:
                hook_t.remove()
                hook_b.remove()

            running_t += nll_t.item()
            running_b += nll_b.item()
            n += 1
        if n > 0:
            history.append({
                "epoch": ep,
                "nll_thinking": running_t / n,
                "nll_base": running_b / n,
                "loss": running_t / n - lam * (running_b / n),
            })
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"      ep {ep:2d}: NLL_T={running_t/max(n,1):.3f}  NLL_B={running_b/max(n,1):.3f}")

    return sv.v.detach().cpu(), history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    p.add_argument("--sae", default=None, help="Path to SAE bundle (default: best_taxonomy)")
    p.add_argument("--max_examples_per_cat", type=int, default=64,
                   help="Cap training examples per category (full 1000 is too slow)")
    p.add_argument("--epochs", type=int, default=STEERING_EPOCHS)
    p.add_argument("--lr", type=float, default=STEERING_LR)
    p.add_argument("--lam", type=float, default=STEERING_LAMBDA)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
    args = p.parse_args()

    init_memory_guard()

    cfg = MODEL_PAIRS[args.pair]
    sae_dir = SAES_DIR / args.pair

    if args.sae is None:
        best = json.loads((sae_dir / "best_taxonomy.json").read_text())
        sae_path = sae_dir / f"L{best['layer']}_K{best['dict_size']}.pt"
        layer = best["layer"]
        K = best["dict_size"]
    else:
        sae_path = Path(args.sae)
        layer = int(sae_path.stem.split("_")[0][1:])
        K = int(sae_path.stem.split("_")[1][1:])

    print(f"Using SAE: {sae_path} (L={layer}, K={K})")
    print(f"Steering layer: {cfg['steer_layer']}")

    # Collect category-tagged training sentences
    cat_sents, sae = collect_category_examples(
        args.pair, layer, K, sae_path, max_per_cat=args.max_examples_per_cat,
    )
    for cid, sents in cat_sents.items():
        print(f"  cat {cid}: {len(sents)} examples")

    # Load rollouts to recover full text
    rollout_path = ROLLOUTS_DIR / f"{args.pair}_thinking_mmlu_pro.jsonl"
    rollouts = load_rollout_text(rollout_path)

    # Load both models
    print(f"\nLoading base + thinking models ...")
    base, thinking, tok = load_models(cfg, device=args.device)
    check_memory("after both models loaded")

    # Build training pairs
    pairs_per_cat = build_training_pairs(
        cat_sents, rollouts, tok,
        max_per_cat=args.max_examples_per_cat,
    )

    out_dir = STEERING_DIR / args.pair
    out_dir.mkdir(parents=True, exist_ok=True)

    vectors: dict[int, torch.Tensor] = {}
    histories: dict[int, list] = {}
    for cat, pairs in pairs_per_cat.items():
        if not pairs:
            continue
        print(f"\n=== Category {cat} ({len(pairs)} pairs) ===")
        t0 = time.time()
        v, hist = train_steering_vector(
            base, thinking, cfg["steer_layer"], cfg["steer_layer"],
            cfg["d_model"], pairs, lr=args.lr, epochs=args.epochs, lam=args.lam,
            device=args.device,
        )
        vectors[cat] = v
        histories[cat] = hist
        if hist:
            print(f"  done in {time.time()-t0:.1f}s, final NLL_T={hist[-1]['nll_thinking']:.3f}")
        else:
            print(f"  done in {time.time()-t0:.1f}s, NO valid training pairs — vector unchanged")

        check_memory(f"after cat {cat}")

    # Save
    out_path = out_dir / f"steering_L{cfg['steer_layer']}_K{K}.pt"
    torch.save({
        "vectors": {int(k): v for k, v in vectors.items()},
        "histories": histories,
        "layer": cfg["steer_layer"],
        "sae_path": str(sae_path),
        "dict_size": K,
        "lr": args.lr,
        "epochs": args.epochs,
        "lam": args.lam,
    }, out_path)
    print(f"\nSaved {len(vectors)} steering vectors → {out_path}")


if __name__ == "__main__":
    main()
