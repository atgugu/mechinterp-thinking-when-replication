"""Device-agnostic memory guard for the replication pipeline.

Sets a soft cap on accelerator memory and reports usage at checkpoints.
Auto-detects CUDA or torch MPS; falls back to CPU.

Usage:
    from memory_guard import init_memory_guard, check_memory
    init_memory_guard()
    check_memory("after model load")

Configure via environment:
    ML_MEMORY_LIMIT_GB   — soft cap for accelerator allocations
    ML_MAX_SEQ_LEN       — max tokens before truncation (default: 5000)
"""

from __future__ import annotations

import gc
import os
import torch

MAX_ML_MEMORY_GB = float(os.environ.get("ML_MEMORY_LIMIT_GB", "0") or 0)
MAX_SEQ_LEN = int(os.environ.get("ML_MAX_SEQ_LEN", "5000"))

_initialized = False


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _alloc_gb() -> tuple[float, float]:
    """Return (currently_allocated_gb, reserved_gb)."""
    dev = _device()
    if dev == "cuda":
        return torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9
    if dev == "mps":
        return (
            torch.mps.current_allocated_memory() / 1e9,
            torch.mps.driver_allocated_memory() / 1e9,
        )
    return 0.0, 0.0


def _sys_mem_pct() -> tuple[float, float, float]:
    try:
        import psutil
        m = psutil.virtual_memory()
        return m.used / 1e9, m.total / 1e9, float(m.percent)
    except Exception:
        return 0.0, 0.0, 0.0


def _empty_cache() -> None:
    dev = _device()
    if dev == "cuda":
        torch.cuda.empty_cache()
    elif dev == "mps":
        torch.mps.empty_cache()


def init_memory_guard() -> None:
    """Set a soft cap on accelerator memory. Call once at script start."""
    global _initialized
    if _initialized:
        return
    dev = _device()
    if dev == "cpu":
        print("Memory guard: no accelerator detected, running on CPU")
        _initialized = True
        return

    if MAX_ML_MEMORY_GB > 0:
        try:
            if dev == "mps":
                recommended = torch.mps.recommended_max_memory() / 1e9
                fraction = min(MAX_ML_MEMORY_GB / max(recommended, 1.0), 1.0)
                torch.mps.set_per_process_memory_fraction(fraction)
                print(f"Memory guard: {MAX_ML_MEMORY_GB:.0f}GB cap "
                      f"({fraction:.0%} of {recommended:.0f}GB recommended)")
            elif dev == "cuda":
                # CUDA per-process fraction is per-device
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                fraction = min(MAX_ML_MEMORY_GB / max(total, 1.0), 1.0)
                torch.cuda.set_per_process_memory_fraction(fraction)
                print(f"Memory guard: {MAX_ML_MEMORY_GB:.0f}GB cap "
                      f"({fraction:.0%} of {total:.0f}GB total)")
        except Exception as e:
            print(f"Memory guard: cap not applied ({e})")
    print(f"Device: {dev}  ·  Max sequence length: {MAX_SEQ_LEN} tokens")
    _initialized = True
    check_memory("init")


def check_memory(label: str = "") -> float:
    """Print accelerator + system memory usage. Returns allocated GB."""
    alloc, reserved = _alloc_gb()
    used_gb, total_gb, pct = _sys_mem_pct()
    dev = _device()
    print(
        f"[MEM {label}] {dev}: {alloc:.1f}GB (reserved: {reserved:.1f}GB) | "
        f"system: {used_gb:.0f}/{total_gb:.0f}GB ({pct:.0f}%)"
    )
    if pct > 85:
        print("  WARNING: system memory pressure >85%")
    if pct > 92:
        print("  CRITICAL: system memory >92% — freeing caches")
        _empty_cache()
        gc.collect()
    return alloc


def truncate_tokens(tokens, model=None):
    """Truncate to MAX_SEQ_LEN if longer; return possibly-shortened tensor."""
    if tokens.shape[1] <= MAX_SEQ_LEN:
        return tokens
    orig = tokens.shape[1]
    tokens = tokens[:, :MAX_SEQ_LEN]
    print(f"Truncated {orig} → {MAX_SEQ_LEN} tokens")
    return tokens
