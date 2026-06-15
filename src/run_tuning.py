"""
src/run_tuning.py — Standalone Optuna tuning script
Run from project root: python src/run_tuning.py

Advantages over notebook:
  - No kernel timeout issues
  - Checkpoint saves after each outer fold
  - Resume from checkpoint if interrupted
  - Ctrl+C safe — progress not lost
"""

import os
import sys
import pickle
import logging
from pathlib import Path

# ── Project root setup ──────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

import yaml
import numpy as np
import pandas as pd

from src.ingestion import load_config
from src.models import setup_mlflow, tune_lgbm_optuna

# ── Logging to file + console ───────────────────────────────────────────────
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "tuning.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
cfg         = load_config("configs/config.yaml")
N_TRIALS    = int(os.environ.get("N_TRIALS", 20))   # override: N_TRIALS=50 python src/run_tuning.py
CHECKPOINT  = Path("data/processed/tuning_checkpoint.pkl")

setup_mlflow(cfg)

# ── Load data ───────────────────────────────────────────────────────────────
splits_path = Path(cfg["data"]["splits_path"])
X_train = pd.read_parquet(splits_path / "X_train.parquet")
y_train = pd.read_parquet(splits_path / "y_train.parquet").squeeze()
logger.info(f"Loaded X_train: {X_train.shape} | fraud: {y_train.mean()*100:.2f}%")

# ── Resume check ─────────────────────────────────────────────────────────────
if CHECKPOINT.exists():
    logger.info(f"⚡ Checkpoint found at {CHECKPOINT}")
    with open(CHECKPOINT, "rb") as f:
        checkpoint = pickle.load(f)
    completed_folds   = checkpoint["completed_folds"]
    outer_scores      = checkpoint["outer_scores"]
    best_params_folds = checkpoint["best_params_per_fold"]
    logger.info(f"  Resuming from fold {completed_folds + 1}/5")
else:
    completed_folds   = 0
    outer_scores      = []
    best_params_folds = []
    logger.info("No checkpoint found — starting fresh")

# ── Run tuning with checkpoint support ──────────────────────────────────────
logger.info(f"Starting Optuna tuning — {N_TRIALS} trials per outer fold")
logger.info(f"Remaining folds: {5 - completed_folds}")

try:
    tuning_results = tune_lgbm_optuna(
        X_train, y_train, cfg,
        n_trials=N_TRIALS,
        checkpoint_path=str(CHECKPOINT),
        completed_folds=completed_folds,
        outer_scores_so_far=outer_scores,
        best_params_so_far=best_params_folds,
    )
except KeyboardInterrupt:
    logger.info("\n⚠️  Interrupted by user — checkpoint saved, resume with same command")
    sys.exit(0)

# ── Save results ─────────────────────────────────────────────────────────────
logger.info(f"\n{'='*55}")
logger.info(f"TUNING COMPLETE")
logger.info(f"  Mean outer PR-AUC : {tuning_results['mean_outer_pr_auc']:.4f} ± {tuning_results['std_outer_pr_auc']:.4f}")
logger.info(f"  Best params saved : configs/best_params.yaml")
logger.info(f"{'='*55}")

# Cleanup checkpoint
if CHECKPOINT.exists():
    CHECKPOINT.unlink()
    logger.info("✅ Checkpoint cleaned up")

print("\n✅ Tuning complete! Now run 07_retrain.ipynb in Jupyter.")
