"""Model loading for base + thinking model pairs."""

import os
import sys
import gc
from pathlib import Path

# Allow transformer-lens to use any available accelerator; harmless elsewhere.
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")
sys.path.insert(0, str(Path(__file__).parent))

import torch
from memory_guard import init_memory_guard, check_memory  # type: ignore  # noqa: E402

from const import MODEL_PAIRS  # noqa: E402


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _select_device()


def cleanup():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()


def load_hf(model_path: str, dtype=torch.float16):
    """Load a HuggingFace model + tokenizer from local path. Stays on CPU until moved."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading HF: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="cpu",
        local_files_only=True,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_hooked(model_path: str, tl_arch: str, n_ctx: int = 4096, dtype=torch.float16):
    """Load model as TransformerLens HookedTransformer for activation extraction."""
    from transformer_lens import HookedTransformer

    hf_model, tokenizer = load_hf(model_path, dtype=dtype)
    check_memory(f"after HF load ({Path(model_path).name})")

    print(f"Wrapping as HookedTransformer with arch={tl_arch}")
    model = HookedTransformer.from_pretrained(
        tl_arch,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=DEVICE,
        dtype=dtype,
        n_ctx=n_ctx,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    del hf_model
    cleanup()
    check_memory(f"after HookedTransformer ({Path(model_path).name})")
    return model, tokenizer


def load_pair(pair_key: str, mode: str = "hooked", n_ctx: int = 4096):
    """Load a (base, thinking) pair for the given key.

    mode: "hooked" → HookedTransformer (for activation extraction)
          "hf"     → HuggingFace model (for generation)
    Returns (base_model, base_tok, thinking_model, thinking_tok, cfg).
    """
    if pair_key not in MODEL_PAIRS:
        raise ValueError(f"Unknown pair: {pair_key}")
    cfg = MODEL_PAIRS[pair_key]

    if mode == "hf":
        base_model, base_tok = load_hf(cfg["base"])
        check_memory("after base HF")
        thinking_model, thinking_tok = load_hf(cfg["thinking"])
        check_memory("after thinking HF")
    elif mode == "hooked":
        base_model, base_tok = load_hooked(cfg["base"], cfg["tl_arch"], n_ctx=n_ctx)
        thinking_model, thinking_tok = load_hooked(cfg["thinking"], cfg["tl_arch"], n_ctx=n_ctx)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return base_model, base_tok, thinking_model, thinking_tok, cfg


if __name__ == "__main__":
    init_memory_guard()
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    p.add_argument("--mode", default="hf", choices=["hf", "hooked"])
    p.add_argument("--which", default="thinking", choices=["base", "thinking", "both"])
    args = p.parse_args()

    cfg = MODEL_PAIRS[args.pair]
    print(f"=== Smoke test: load {args.which} for pair={args.pair} mode={args.mode} ===\n")

    if args.which in ("base", "both"):
        if args.mode == "hf":
            m, t = load_hf(cfg["base"])
        else:
            m, t = load_hooked(cfg["base"], cfg["tl_arch"])
        print(f"Base model loaded. n_params={sum(p.numel() for p in m.parameters())/1e9:.2f}B\n")
        del m, t
        cleanup()
    if args.which in ("thinking", "both"):
        if args.mode == "hf":
            m, t = load_hf(cfg["thinking"])
        else:
            m, t = load_hooked(cfg["thinking"], cfg["tl_arch"])
        print(f"Thinking model loaded. n_params={sum(p.numel() for p in m.parameters())/1e9:.2f}B\n")
        del m, t
        cleanup()

    check_memory("end")
