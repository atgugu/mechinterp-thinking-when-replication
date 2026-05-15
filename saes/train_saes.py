"""Train Top-K SAEs across (layer, dict_size) sweep.

For each layer in extract_layers and each dict_size in SAE_DICT_SIZES,
train a Top-K SAE on the THINKING model's sentence-pooled activations.
The thinking model is used as the source because thinking traces have
the rich category structure we want to discover.

Saves trained SAE state_dict + training metadata to
`results/saes/<pair>/L<layer>_K<dict_size>.pt`.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

# (memory_guard imported from local package root)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["TRANSFORMERLENS_ALLOW_MPS"] = "1"

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from memory_guard import init_memory_guard, check_memory  # noqa: E402
from const import (  # noqa: E402
    MODEL_PAIRS, ACTS_DIR, SAES_DIR,
    SAE_K, SAE_DICT_SIZES, SAE_BATCH_SIZE, SAE_MAX_EPOCHS, SAE_PATIENCE,
)
from saes.topk_sae import TopKSAE  # noqa: E402


def load_layer_acts(pair: str, which: str, layer: int) -> tuple[torch.Tensor, list[str], list[str]]:
    """Load all activation shards for (pair, which, layer) into a single tensor."""
    shard_dir = ACTS_DIR / pair / which / f"L{layer}"
    shards = sorted(shard_dir.glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No shards found in {shard_dir}")

    acts_list: list[torch.Tensor] = []
    ids_list: list[str] = []
    text_list: list[str] = []
    for s in shards:
        d = torch.load(s, map_location="cpu", weights_only=False)
        acts_list.append(d["acts"])
        ids_list.extend(d["ids"])
        text_list.extend(d["text"])
    acts = torch.cat(acts_list, dim=0).to(torch.float32)
    return acts, ids_list, text_list


def train_one(
    acts: torch.Tensor, dict_size: int, k: int,
    max_epochs: int, patience: int, batch_size: int, lr: float | None,
    device: str = "mps", seed: int = 0,
) -> tuple[TopKSAE, dict]:
    d_input = acts.shape[1]
    if lr is None:
        # 1 / d scaling rule
        lr = 1.0 / d_input
    torch.manual_seed(seed)

    # 90/10 train/val split
    n = acts.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    cut = int(0.9 * n)
    train_acts = acts[perm[:cut]].clone()
    val_acts = acts[perm[cut:]].clone()

    # Initialize b_pre to mean of training data (whitening trick)
    sae = TopKSAE(d_input, dict_size, k=k, normalize_decoder=True).to(device)
    with torch.no_grad():
        sae.b_pre.copy_(train_acts.mean(dim=0).to(device))

    opt = torch.optim.AdamW(sae.parameters(), lr=lr, weight_decay=0.0)
    loader = DataLoader(
        TensorDataset(train_acts), batch_size=batch_size, shuffle=True, drop_last=True
    )

    best_val = float("inf")
    epochs_without_improve = 0
    history: list[dict] = []
    t0 = time.time()
    for ep in range(max_epochs):
        sae.train()
        running = 0.0
        n_batches = 0
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            x_hat, _z = sae(xb)
            loss = (x_hat - xb).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                sae._normalize_decoder_inplace()
            running += loss.item()
            n_batches += 1
        train_loss = running / max(n_batches, 1)

        # Validation
        sae.eval()
        with torch.no_grad():
            vb = val_acts.to(device)
            x_hat, _z = sae(vb)
            val_loss = (x_hat - vb).pow(2).mean().item()
            # Variance-normalised reconstruction quality
            var = (vb - vb.mean(dim=0, keepdim=True)).pow(2).mean().item()
            recon_frac = 1.0 - val_loss / max(var, 1e-8)

        history.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss, "recon_frac": recon_frac})

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    meta = {
        "best_val_loss": best_val,
        "final_recon_frac": history[-1]["recon_frac"],
        "epochs_run": len(history),
        "elapsed_s": time.time() - t0,
        "history": history,
        "dict_size": dict_size,
        "k": k,
        "d_input": d_input,
        "lr": lr,
    }
    return sae, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="qwen-7b", choices=list(MODEL_PAIRS))
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--dict_sizes", type=int, nargs="*", default=None)
    p.add_argument("--k", type=int, default=SAE_K)
    p.add_argument("--max_epochs", type=int, default=SAE_MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=SAE_PATIENCE)
    p.add_argument("--batch_size", type=int, default=SAE_BATCH_SIZE)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
    p.add_argument("--dry_run", action="store_true", help="1 epoch sanity check")
    args = p.parse_args()

    init_memory_guard()

    cfg = MODEL_PAIRS[args.pair]
    layers = args.layers or cfg["extract_layers"]
    dict_sizes = args.dict_sizes or SAE_DICT_SIZES
    out_dir = SAES_DIR / args.pair
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        layers = layers[:1]
        dict_sizes = dict_sizes[:1]
        max_epochs = 1
    else:
        max_epochs = args.max_epochs

    summary: list[dict] = []
    for L in layers:
        print(f"\n=== Layer {L}: loading thinking-model activations ===")
        try:
            acts, _ids, _texts = load_layer_acts(args.pair, "thinking", L)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        print(f"  Acts: {tuple(acts.shape)}  (mean={acts.mean():.3f}, std={acts.std():.3f})")

        for K in dict_sizes:
            out_path = out_dir / f"L{L}_K{K}.pt"
            if out_path.exists() and not args.dry_run:
                print(f"  L{L}/K{K}: exists, skipping")
                continue
            print(f"  Training L{L}/K{K} ...")
            sae, meta = train_one(
                acts, K, args.k, max_epochs, args.patience, args.batch_size,
                lr=None, device=args.device, seed=0,
            )
            torch.save({
                "state_dict": sae.state_dict(),
                "config": {"d_input": acts.shape[1], "dict_size": K, "k": args.k},
                "meta": meta,
            }, out_path)
            summary.append({"layer": L, "dict_size": K, **{k: v for k, v in meta.items() if k != "history"}})
            print(f"    → recon_frac={meta['final_recon_frac']:.3f} epochs={meta['epochs_run']} t={meta['elapsed_s']:.1f}s")
            check_memory(f"after L{L}/K{K}")

        # Free per-layer activations
        del acts

    with open(out_dir / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote sweep summary → {out_dir / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
