"""Constants for the replication pipeline.

Model pairs, layer maps, and paper-headline ground truth.
Paper: Venhoff et al., "Base Models Know How to Reason, Thinking Models Learn When"
       NeurIPS 2025 MechInterp Workshop, arxiv 2510.07364
"""

import os
from pathlib import Path

# ── Model location ────────────────────────────────────────────────────────────
# Override with env var MODELS_DIR. Default: ./models/ relative to this repo.
# Each pair points to a directory containing the HF model files.
MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).parent / "models"))

MODEL_PAIRS = {
    "qwen-7b": {
        "base": str(MODELS_DIR / "Qwen2.5-7B"),
        "thinking": str(MODELS_DIR / "DeepSeek-R1-Distill-Qwen-7B"),
        "tl_arch": "Qwen/Qwen2.5-7B",
        "n_layers": 28,
        "d_model": 3584,
        "n_heads": 28,
        # 6 distributed layers (paper uses ~{6,10,14,18,22,26} of 32; scaled to 28-layer model)
        "extract_layers": [5, 9, 13, 17, 21, 25],
        # ~37% of depth → layer 10 for steering vector application
        "steer_layer": 10,
    },
    "qwen-14b": {
        "base": str(MODELS_DIR / "Qwen2.5-14B"),
        "thinking": str(MODELS_DIR / "DeepSeek-R1-Distill-Qwen-14B"),
        "tl_arch": "Qwen/Qwen2.5-14B",
        "n_layers": 48,
        "d_model": 5120,
        "n_heads": 40,
        "extract_layers": [8, 16, 24, 32, 40, 47],
        "steer_layer": 18,  # 37% of 48
    },
}

# ── Paper headline metrics (for verification) ────────────────────────────────

PAPER_HEADLINES = {
    "qwen-14b": {
        "math500_base": 59.1,
        "math500_thinking": None,  # paper Table 2
        "math500_hybrid": 74.6,
        "gap_recovery_pct_math500": None,  # computed
    },
    "qwen-32b": {
        "math500_base": 63.4,
        "math500_thinking": 86.5,
        "math500_hybrid": 84.4,
        "gap_recovery_pct_math500": 91.0,
        "gsm8k_base": 92.6,
        "gsm8k_hybrid": 94.8,
    },
    "tokens_steered_pct_target": 12.0,  # paper's headline ~12 %
    "elbow_cluster_range": (10, 20),
}

# ── Pipeline hyperparameters ─────────────────────────────────────────────────

MMLU_PRO_SUBSET = 3000          # paper used 12,102; 3k yields ~100k sentences
ROLLOUT_TEMPERATURE = 0.6
ROLLOUT_TOP_P = 0.95
ROLLOUT_MAX_NEW_TOKENS = 2048

SAE_K = 3                       # Top-K sparsity
SAE_DICT_SIZES = [5, 10, 15, 20, 25, 30, 40, 50]
SAE_BATCH_SIZE = 512
SAE_MAX_EPOCHS = 300
SAE_PATIENCE = 10

STEERING_LAMBDA = 1.0           # weight on base-model suppression term
STEERING_LR = 1e-3
STEERING_EPOCHS = 50
STEERING_SENTENCES_PER_CAT = 1000

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
DATA_DIR = REPO_ROOT / "data" / "cache"
ROLLOUTS_DIR = RESULTS_DIR / "rollouts"
ACTS_DIR = RESULTS_DIR / "acts"
SAES_DIR = RESULTS_DIR / "saes"
STEERING_DIR = RESULTS_DIR / "steering"
HYBRID_DIR = RESULTS_DIR / "hybrid"

for d in [RESULTS_DIR, FIGURES_DIR, DATA_DIR, ROLLOUTS_DIR, ACTS_DIR, SAES_DIR, STEERING_DIR, HYBRID_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Style (matches paper2 / paper3 portfolio) ────────────────────────────────

PLOT_BG = "#0d1117"
PLOT_FG = "#e6edf3"
ACCENT_GOLD = "#f0b429"
ACCENT_AMBER = "#de911d"
CATEGORY_PALETTE = [
    "#f0b429", "#3b82f6", "#10b981", "#ef4444", "#a78bfa",
    "#06b6d4", "#f97316", "#ec4899", "#84cc16", "#8b5cf6",
    "#14b8a6", "#f43f5e", "#22d3ee", "#facc15", "#a3e635",
    "#fb923c", "#e879f9", "#34d399", "#fbbf24", "#60a5fa",
]
