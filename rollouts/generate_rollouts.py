"""Generate thinking-model rollouts on MMLU-Pro prompts (resumable, sharded JSONL).

Usage:
    python -m rollouts.generate_rollouts --pair qwen-7b --input data/cache/mmlu_pro_subset.jsonl
"""

import os
import sys
import gc
import json
import time
import argparse
from pathlib import Path

os.environ["TRANSFORMERLENS_ALLOW_MPS"] = "1"
# (memory_guard imported from local package root)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_guard import init_memory_guard, check_memory  # noqa: E402
from const import (  # noqa: E402
    MODEL_PAIRS, ROLLOUTS_DIR, DATA_DIR,
    ROLLOUT_TEMPERATURE, ROLLOUT_TOP_P, ROLLOUT_MAX_NEW_TOKENS,
)

import torch  # noqa: E402


def iter_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def already_done(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec["id"])
                except Exception:
                    continue
    return done


def build_thinking_prompt(tokenizer, user_prompt: str) -> str:
    """Wrap the user prompt with the thinking-model's chat template."""
    msgs = [{"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user_prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    p.add_argument("--input", default=str(DATA_DIR / "mmlu_pro_subset.jsonl"))
    p.add_argument("--out", default=None)
    p.add_argument("--n", type=int, default=None, help="Cap on number of examples")
    p.add_argument("--max_new_tokens", type=int, default=ROLLOUT_MAX_NEW_TOKENS)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    init_memory_guard()

    cfg = MODEL_PAIRS[args.pair]
    out_path = Path(args.out) if args.out else ROLLOUTS_DIR / f"{args.pair}_thinking_mmlu_pro.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(out_path)
    print(f"Already done: {len(done)} examples; writing to {out_path}")

    # Stream input to find remaining work
    all_examples = list(iter_jsonl(args.input))
    if args.n is not None:
        all_examples = all_examples[: args.n]
    todo = [ex for ex in all_examples if ex["id"] not in done]
    print(f"Total: {len(all_examples)} | TODO: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    # Load thinking model in HF mode (faster generate)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading thinking model: {cfg['thinking']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["thinking"], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["thinking"],
        dtype=torch.float16,
        device_map={"": ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))},
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.eval()
    check_memory("after thinking load")

    torch.manual_seed(args.seed)

    out_f = open(out_path, "a")
    t0 = time.time()
    for i, ex in enumerate(todo):
        prompt = build_thinking_prompt(tokenizer, ex["prompt"])
        inputs = tokenizer(prompt, return_tensors="pt").to(("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))

        t_start = time.time()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=ROLLOUT_TEMPERATURE,
                top_p=ROLLOUT_TOP_P,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t_start

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = out[0, input_len:].tolist()
        rollout = tokenizer.decode(new_tokens, skip_special_tokens=False)

        rec = {
            "id": ex["id"],
            "prompt": ex["prompt"],
            "completion": rollout,
            "answer": ex.get("answer"),
            "n_new_tokens": len(new_tokens),
            "wall_s": round(elapsed, 2),
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()

        if (i + 1) % 5 == 0 or i == 0:
            total_elapsed = time.time() - t0
            rate = (i + 1) / total_elapsed
            eta = (len(todo) - (i + 1)) / rate / 60
            print(f"  [{i+1}/{len(todo)}] {elapsed:.1f}s ({len(new_tokens)} toks) | "
                  f"rate={rate:.2f}/s ETA={eta:.1f}m")

        # Memory check + cleanup every 50 examples
        if (i + 1) % 50 == 0:
            torch.mps.empty_cache()
            gc.collect()
            check_memory(f"after {i+1} rollouts")

    out_f.close()
    print(f"\nDone. Total elapsed: {(time.time()-t0)/60:.1f}m")
    check_memory("end")


if __name__ == "__main__":
    main()
