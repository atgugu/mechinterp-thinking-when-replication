"""Convenience entrypoint: grade all GSM8K mode outputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.score_all import main as score_all_main
import sys as _sys

if __name__ == "__main__":
    _sys.argv = ["score_all", "--pair", "qwen-7b", "--datasets", "gsm8k", "--print_recovery"]
    score_all_main()
