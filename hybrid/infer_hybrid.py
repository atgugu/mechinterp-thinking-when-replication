"""Hybrid-model inference: base-model decoding with selective per-token steering.

At each generation step:
  1. Forward base model on (prompt + generated_so_far) — captures residual at sae_layer
     and unsteered next-token logits.
  2. SAE encodes that residual → picks top-firing category.
  3. If max activation exceeds threshold, build a "steered" candidate by re-running
     the base model with a hook that adds the category's learned vector to
     `steer_layer`'s residual.
  4. Score each candidate token under the thinking model's parallel KV cache.
  5. Commit the higher-scoring token; track whether it was steered + which category.

Implementation note: we recompute the base forward from scratch each step (no KV
cache) so the steered branch can use the same exact-prior-context as the unsteered
branch without needing to clone HF's DynamicCache. The thinking model uses a
proper KV cache for speed since it's only scoring.

Modes (--mode):
  full          : SAE-chosen category + learned steering vector (fallback: bias)
  only_bias     : every steering-trigger uses the bias vector
  random_firing : random category + learned vectors (fallback: bias)
  random_vectors: SAE-chosen category + Gaussian-random direction
  base          : no steering (passes through base model only)
  thinking      : run thinking model directly, no steering
"""

import os
import sys
import json
import time
import random as _random
import argparse
from pathlib import Path

# (memory_guard imported from local package root)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["TRANSFORMERLENS_ALLOW_MPS"] = "1"

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from memory_guard import init_memory_guard, check_memory  # noqa: E402
from const import MODEL_PAIRS, STEERING_DIR, SAES_DIR, HYBRID_DIR  # noqa: E402
from saes.topk_sae import TopKSAE  # noqa: E402


MODES = ("full", "only_bias", "random_firing", "random_vectors", "base", "thinking")


def load_bundle(pair: str):
    """Load SAE, learned vectors, bias vector, and best-taxonomy config."""
    sae_dir = SAES_DIR / pair
    st_dir = STEERING_DIR / pair
    best = json.loads((sae_dir / "best_taxonomy.json").read_text())
    sae_path = sae_dir / f"L{best['layer']}_K{best['dict_size']}.pt"
    sae_b = torch.load(sae_path, map_location="cpu", weights_only=False)
    sae = TopKSAE(sae_b["config"]["d_input"], sae_b["config"]["dict_size"],
                  k=sae_b["config"]["k"], normalize_decoder=True)
    sae.load_state_dict(sae_b["state_dict"])
    sae.eval()

    cfg = MODEL_PAIRS[pair]
    vec_path = st_dir / f"steering_L{cfg['steer_layer']}_K{best['dict_size']}.pt"
    bias_path = st_dir / f"bias_L{cfg['steer_layer']}.pt"
    learned = torch.load(vec_path, map_location="cpu", weights_only=False) if vec_path.exists() else None
    bias = torch.load(bias_path, map_location="cpu", weights_only=False) if bias_path.exists() else None

    return sae, learned, bias, best


def get_steering_vector(mode: str, cat_id: int, learned: dict, bias: dict,
                        d_model: int, rng_random_vec: torch.Tensor):
    """Pick the steering vector for this step under the given mode. Falls back
    to the bias vector when a category-specific vector is not available."""
    if mode == "full":
        v = learned["vectors"].get(int(cat_id)) if learned is not None else None
        if v is None and bias is not None:
            v = bias.get("vector")
        return v
    if mode == "only_bias":
        return bias["vector"] if bias is not None else None
    if mode == "random_firing":
        v = learned["vectors"].get(int(cat_id)) if learned is not None else None
        if v is None and bias is not None:
            v = bias.get("vector")
        return v
    if mode == "random_vectors":
        return rng_random_vec
    return None


class HookHolder:
    """Add `scale * v` to the residual output of a target decoder layer."""

    def __init__(self, model, layer_idx: int, vector: torch.Tensor, scale: float = 1.0):
        self.layer = model.model.layers[layer_idx]
        self.vector = vector
        self.scale = scale
        self.handle = None

    def __enter__(self):
        v = self.vector
        scale = self.scale
        def hook(_module, _inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            v_local = (v.to(hs.device).to(hs.dtype) * scale)
            hs = hs + v_local
            return (hs,) + output[1:] if isinstance(output, tuple) else hs
        self.handle = self.layer.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        if self.handle is not None:
            self.handle.remove()


class ResidCapture:
    """Hook that captures the residual stream output of a target layer."""

    def __init__(self, model, layer_idx: int):
        self.layer = model.model.layers[layer_idx]
        self.handle = None
        self.captured = None

    def __enter__(self):
        def hook(_module, _inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            self.captured = hs.detach()
        self.handle = self.layer.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        if self.handle is not None:
            self.handle.remove()


def _greedy_next_token(logits: torch.Tensor) -> int:
    return int(logits.argmax(dim=-1).item())


def hybrid_generate(
    base, thinking, tokenizer, prompt: str,
    sae: TopKSAE | None, learned, bias, best,
    cfg, mode: str = "full", max_new_tokens: int = 512,
    steering_threshold: float = 20.0,
    steering_scale: float = 1.0,
    consider_steering: bool = True,
    seed: int = 0,
    device: str = "mps",
    log_every: int = 32,
) -> dict:
    """Generate a completion using selective steering.

    Returns dict with `completion`, `tokens`, `steering_trace`.
    """
    _random.seed(seed)
    torch.manual_seed(seed)
    steer_layer = cfg["steer_layer"]
    sae_layer = best["layer"] if best else steer_layer
    d_model = cfg["d_model"]

    msgs = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    prompt_ids = enc["input_ids"].to(device)
    n_prompt = prompt_ids.shape[1]

    rng_vec = torch.randn(d_model, generator=torch.Generator().manual_seed(seed)) * 1e-2

    generated: list[int] = []
    steered_flags: list[int] = []
    cat_trace: list[int] = []
    max_act_trace: list[float] = []

    # Thinking model state (parallel KV cache)
    thinking_state = {"past": None}
    # Base model KV cache — advances each step on chosen tokens.
    base_past = None

    eos_id = tokenizer.eos_token_id

    t0 = time.time()
    for step in range(max_new_tokens):
        # 1) Unsteered forward — KV-cached for speed.
        #    First step: feed the full prompt; subsequent steps: feed just the previously chosen token.
        with torch.inference_mode():
            with ResidCapture(base, sae_layer) as cap:
                if base_past is None:
                    out = base(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                               use_cache=True)
                else:
                    last_tok = torch.tensor([[generated[-1]]], device=device, dtype=torch.long)
                    out = base(input_ids=last_tok, past_key_values=base_past, use_cache=True)
            base_past = out.past_key_values
            unsteered_logits = out.logits[0, -1, :]

        # 2) Decide whether to steer (uses captured residual at sae_layer)
        do_steer = False
        cat_id = -1
        max_act = 0.0
        if consider_steering and mode != "base" and sae is not None:
            resid = cap.captured[0, -1, :].cpu().float().unsqueeze(0)
            with torch.no_grad():
                z = sae.encode(resid)
                max_act = z.max().item()
                cat_id = int(z.argmax(dim=-1).item())
            if max_act > steering_threshold:
                do_steer = True
            if mode == "random_firing":
                cat_id = _random.randrange(sae.dict_size)
                do_steer = max_act > steering_threshold

        # 3) Build steered candidate via true layer-injection.
        #    The steered branch needs the same context as unsteered with a vector
        #    added at steer_layer. Since KV-cache state can't be safely cloned
        #    here, we recompute the FULL forward from scratch for the steered
        #    branch only when do_steer fires (~12 % of tokens at production scale).
        candidates = [("unsteered", unsteered_logits, -1)]
        if do_steer:
            v = get_steering_vector(mode, cat_id, learned, bias, d_model, rng_vec)
            if v is not None:
                full_ids = (torch.cat([prompt_ids,
                                       torch.tensor([generated], device=device, dtype=torch.long)], dim=1)
                            if generated else prompt_ids)
                attn_full = torch.ones_like(full_ids)
                with torch.inference_mode():
                    with HookHolder(base, steer_layer, v, scale=steering_scale):
                        out_s = base(input_ids=full_ids, attention_mask=attn_full, use_cache=False)
                    steered_logits = out_s.logits[0, -1, :]
                candidates.append(("steered", steered_logits, cat_id))

        # 5) Thinking-model logprob distribution (advances its KV state)
        thinking_input = prompt_ids if step == 0 else torch.tensor([[generated[-1]]], device=device, dtype=torch.long)
        thinking_attn = torch.ones_like(prompt_ids) if step == 0 else None
        if thinking is not None and thinking is not base:
            with torch.inference_mode():
                if thinking_state["past"] is None:
                    out_t = thinking(input_ids=thinking_input, attention_mask=thinking_attn, use_cache=True)
                else:
                    out_t = thinking(input_ids=thinking_input, past_key_values=thinking_state["past"], use_cache=True)
                thinking_state["past"] = out_t.past_key_values
                lp_dist = F.log_softmax(out_t.logits[0, -1, :].float(), dim=-1)
        else:
            lp_dist = None

        # 6) Pick winner by thinking-model logprob (if scoring available)
        chosen_label, chosen_logits, chosen_cat = candidates[0]
        if len(candidates) > 1 and lp_dist is not None:
            best_lp = -float("inf")
            best_idx = 0
            for i, (label, logits_, cid) in enumerate(candidates):
                tok = _greedy_next_token(logits_)
                lp = float(lp_dist[tok].item())
                if lp > best_lp:
                    best_lp = lp
                    best_idx = i
            chosen_label, chosen_logits, chosen_cat = candidates[best_idx]

        next_tok = _greedy_next_token(chosen_logits)
        generated.append(next_tok)
        steered_flags.append(1 if chosen_label == "steered" else 0)
        cat_trace.append(chosen_cat)
        max_act_trace.append(round(max_act, 3))

        if next_tok == eos_id:
            break

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed
            sf = sum(steered_flags) / len(steered_flags)
            print(f"    step {step+1}: {rate:.1f}tok/s | steered_frac={sf:.2%} | last_max_act={max_act:.1f}")

    completion = tokenizer.decode(generated, skip_special_tokens=False)
    return {
        "completion": completion,
        "tokens": generated,
        "steered_flags": steered_flags,
        "cat_trace": cat_trace,
        "max_act_trace": max_act_trace,
        "steered_frac": sum(steered_flags) / max(len(steered_flags), 1),
        "n_tokens": len(generated),
        "wall_s": time.time() - t0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    p.add_argument("--mode", default="full", choices=MODES)
    p.add_argument("--dataset", default="math500", choices=["math500", "gsm8k"])
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--threshold", type=float, default=20.0,
                   help="SAE activation threshold for triggering steering")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Multiplier on the steering vector magnitude")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    init_memory_guard()
    cfg = MODEL_PAIRS[args.pair]

    from data.prepare_math500 import main as prep_math500
    from data.prepare_gsm8k import main as prep_gsm8k
    if args.dataset == "math500":
        ds_path = prep_math500()
    else:
        ds_path = prep_gsm8k()
    examples = []
    with open(ds_path) as f:
        for line in f:
            examples.append(json.loads(line))
    examples = examples[: args.n]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["base"], local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading models for mode={args.mode} ...")
    needs_base = args.mode != "thinking"
    needs_thinking = args.mode in ("full", "only_bias", "random_firing", "random_vectors", "thinking")
    base_model = None
    thinking_model = None
    if needs_base:
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg["base"], dtype=torch.float16, device_map={"": ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))},
            local_files_only=True, attn_implementation="sdpa",
        )
        base_model.eval()
    if needs_thinking:
        thinking_model = AutoModelForCausalLM.from_pretrained(
            cfg["thinking"], dtype=torch.float16, device_map={"": ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))},
            local_files_only=True, attn_implementation="sdpa",
        )
        thinking_model.eval()
    if args.mode == "thinking":
        eval_model = thinking_model
    elif args.mode == "base":
        eval_model = base_model
    else:
        eval_model = base_model
    check_memory("after model load")

    if args.mode in ("full", "only_bias", "random_firing", "random_vectors"):
        sae, learned, bias, best = load_bundle(args.pair)
    else:
        sae, learned, bias, best = None, None, None, None

    out_path = Path(args.out) if args.out else HYBRID_DIR / f"{args.pair}_{args.dataset}_{args.mode}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

    f_out = open(out_path, "a")
    for i, ex in enumerate(examples):
        if ex["id"] in done_ids:
            continue
        print(f"\n[{i+1}/{len(examples)}] {ex['id']}")
        # Fast path: base or thinking only → use HF generate with KV cache
        if args.mode in ("thinking", "base"):
            msgs = [{"role": "user", "content": ex["prompt"]}]
            try:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = ex["prompt"]
            inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
            t0 = time.time()
            with torch.inference_mode():
                out = eval_model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                )
            completion = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            rec = {
                "id": ex["id"], "mode": args.mode,
                "completion": completion,
                "answer": ex.get("answer"),
                "n_tokens": int(out.shape[1] - inputs["input_ids"].shape[1]),
                "wall_s": time.time() - t0,
                "steered_frac": 0.0,
            }
        else:
            res = hybrid_generate(
                eval_model,
                thinking_model if args.mode not in ("base",) else None,
                tok, ex["prompt"], sae, learned, bias, best, cfg,
                mode=args.mode if args.mode != "base" else "full",
                consider_steering=(args.mode != "base"),
                max_new_tokens=args.max_new_tokens,
                steering_threshold=args.threshold,
                steering_scale=args.scale,
                seed=i,
            )
            rec = {
                "id": ex["id"], "mode": args.mode,
                "completion": res["completion"],
                "answer": ex.get("answer"),
                "n_tokens": res["n_tokens"],
                "wall_s": res["wall_s"],
                "steered_frac": res["steered_frac"],
                "cat_trace": res["cat_trace"],
                "steered_flags": res["steered_flags"],
            }
        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f_out.flush()
        print(f"  {rec['n_tokens']} toks | steered={rec.get('steered_frac', 0):.2%} | {rec['wall_s']:.1f}s")
        if (i + 1) % 10 == 0:
            check_memory(f"after {i+1}")

    f_out.close()
    print(f"\nDone → {out_path}")


if __name__ == "__main__":
    main()
