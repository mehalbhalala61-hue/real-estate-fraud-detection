"""
src/evaluate.py — Real Estate Fraud Detection
Metrics — PR-AUC, Recall@95Precision, calibration, confusion matrix.

Primary metric: PR-AUC    — fraud class imbalanced (3-5%), ROC-AUC misleading
Secondary:      Recall@95P — cost of missing fraud >> cost of false alarm
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core metric functions
# ─────────────────────────────────────────────────────────────────────────────

def pr_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Precision-Recall AUC (average_precision_score).
    Primary metric — handles class imbalance correctly.
    """
    return float(average_precision_score(y_true, y_prob))


def recall_at_precision(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_precision: float = 0.95,
) -> float:
    """
    Recall at a given precision threshold.
    'At 95% precision, what fraction of fraud are we catching?'
    Returns 0.0 if precision never reaches target.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    # precision/recall arrays are in decreasing threshold order
    mask = precision >= target_precision
    if not mask.any():
        return 0.0
    return float(recall[mask].max())


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.70,
) -> Dict[str, float]:
    """
    Compute full metric suite at given threshold.
    threshold: from config api.fraud_threshold_suspicious
    """
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "pr_auc":           pr_auc_score(y_true, y_prob),
        "roc_auc":          float(roc_auc_score(y_true, y_prob)),
        "recall_at_95p":    recall_at_precision(y_true, y_prob, 0.95),
        "recall_at_90p":    recall_at_precision(y_true, y_prob, 0.90),
        "precision":        float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":               float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold":        threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    plots_dir: str = "reports/plots",
    ax=None,
    show: bool = True,
) -> plt.Axes:
    """Precision-Recall curve with PR-AUC annotation."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    auc = pr_auc_score(y_true, y_prob)
    baseline = float(y_true.mean())

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(recall, precision, lw=2, label=f"{model_name} (PR-AUC={auc:.4f})")
    ax.axhline(baseline, color="grey", ls="--", lw=1,
               label=f"Random baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    if own_fig:
        plt.tight_layout()
        Path(plots_dir).mkdir(parents=True, exist_ok=True)
        out = f"{plots_dir}/pr_curve_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(out, dpi=150)
        if show:
            plt.show()
        plt.close()

    return ax


def plot_all_pr_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    y_true: np.ndarray,
    plots_dir: str = "reports/plots",
    show: bool = True,
) -> None:
    """
    Plot all model PR curves on one figure.
    results: {'Model Name': y_prob_array, ...}
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    baseline = float(y_true.mean())
    ax.axhline(baseline, color="grey", ls="--", lw=1,
               label=f"Random baseline ({baseline:.3f})")

    for model_name, y_prob in results.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auc = pr_auc_score(y_true, y_prob)
        ax.plot(recall, precision, lw=2, label=f"{model_name} (AUC={auc:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Baseline Models")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()

    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    out = f"{plots_dir}/pr_curves_comparison.png"
    plt.savefig(out, dpi=150)
    logger.info(f"PR curves saved → {out}")
    if show:
        plt.show()
    plt.close()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    plots_dir: str = "reports/plots",
    n_bins: int = 10,
    show: bool = True,
) -> None:
    """
    Calibration curve: predicted prob vs actual fraction.
    Perfect calibration = diagonal line.
    p=0.8 should mean 80% of those listings are actually fraud.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Calibration plot
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    axes[0].plot(prob_pred, prob_true, "s-", lw=2,
                 label=f"{model_name}", color="#1E88E5")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Fraction of Positives")
    axes[0].set_title(f"Calibration Curve — {model_name}")
    axes[0].legend()
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])

    # Score histogram
    axes[1].hist(y_prob[y_true == 0], bins=50, alpha=0.6,
                 color="#1E88E5", label="Normal", density=True)
    axes[1].hist(y_prob[y_true == 1], bins=50, alpha=0.6,
                 color="#E53935", label="Fraud", density=True)
    axes[1].set_xlabel("Predicted Fraud Probability")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Score Distribution by True Label")
    axes[1].legend()

    plt.suptitle(f"Calibration Analysis — {model_name}", fontsize=13)
    plt.tight_layout()

    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    out = f"{plots_dir}/calibration_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    logger.info(f"Calibration plot saved → {out}")
    if show:
        plt.show()
    plt.close()


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    plots_dir: str = "reports/plots",
    show: bool = True,
) -> pd.DataFrame:
    """
    Precision, Recall, F1 vs threshold sweep.
    Shows why 0.70 was chosen (from threshold_decisions.md).
    Returns table of metrics at key thresholds.
    """
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "Threshold":    t,
            "Precision":    round(precision_score(y_true, y_pred, zero_division=0), 4),
            "Recall":       round(recall_score(y_true, y_pred, zero_division=0), 4),
            "F1":           round(f1_score(y_true, y_pred, zero_division=0), 4),
            "FP/1000":      int(((y_pred == 1) & (y_true == 0)).sum() / len(y_true) * 1000),
            "FN/1000":      int(((y_pred == 0) & (y_true == 1)).sum() / len(y_true) * 1000),
        })
    df = pd.DataFrame(rows)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df["Threshold"], df["Precision"], "o-", label="Precision", color="#1E88E5")
    axes[0].plot(df["Threshold"], df["Recall"],    "s-", label="Recall",    color="#E53935")
    axes[0].plot(df["Threshold"], df["F1"],        "^-", label="F1",        color="#43A047")
    axes[0].axvline(0.70, color="black", ls="--", alpha=0.5, label="Chosen (0.70)")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Precision / Recall / F1 vs Threshold")
    axes[0].legend()

    axes[1].plot(df["Threshold"], df["FP/1000"], "o-", label="False Positives/1000", color="#FF9800")
    axes[1].plot(df["Threshold"], df["FN/1000"], "s-", label="False Negatives/1000", color="#9C27B0")
    axes[1].axvline(0.70, color="black", ls="--", alpha=0.5, label="Chosen (0.70)")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Count per 1000 listings")
    axes[1].set_title("FP/FN Trade-off vs Threshold")
    axes[1].legend()

    plt.suptitle(f"Threshold Analysis — {model_name}", fontsize=13)
    plt.tight_layout()

    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    out = f"{plots_dir}/threshold_analysis_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close()

    return df


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    threshold: float = 0.70,
    plots_dir: str = "reports/plots",
    show: bool = True,
) -> None:
    """Confusion matrix at chosen threshold."""
    import seaborn as sns

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Fraud"],
        yticklabels=["Normal", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}\n(threshold={threshold})")
    plt.tight_layout()

    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    out = f"{plots_dir}/confusion_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation report
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    cfg: dict,
    plots_dir: str = "reports/plots",
    show: bool = True,
) -> Dict:
    """
    Run full evaluation suite:
      - All metrics at configured threshold
      - PR curve
      - Calibration curve
      - Threshold analysis
      - Confusion matrix
    Returns metrics dict.
    """
    threshold = cfg["api"]["fraud_threshold_suspicious"]
    metrics   = compute_all_metrics(y_true, y_prob, threshold)

    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluation — {model_name}")
    logger.info(f"  PR-AUC        : {metrics['pr_auc']:.4f}")
    logger.info(f"  ROC-AUC       : {metrics['roc_auc']:.4f}")
    logger.info(f"  Recall@95P    : {metrics['recall_at_95p']:.4f}")
    logger.info(f"  Precision @{threshold}: {metrics['precision']:.4f}")
    logger.info(f"  Recall    @{threshold}: {metrics['recall']:.4f}")
    logger.info(f"  F1        @{threshold}: {metrics['f1']:.4f}")
    logger.info(f"{'='*50}")

    # Target checks
    pr_auc_target = 0.80
    recall_target = 0.40
    logger.info(
        f"  Target PR-AUC > {pr_auc_target}: "
        f"{'✅' if metrics['pr_auc'] > pr_auc_target else '❌'}"
    )
    logger.info(
        f"  Target Recall@95P > {recall_target}: "
        f"{'✅' if metrics['recall_at_95p'] > recall_target else '❌'}"
    )

    plot_pr_curve(y_true, y_prob, model_name, plots_dir, show=show)
    plot_calibration_curve(y_true, y_prob, model_name, plots_dir, show=show)
    plot_threshold_analysis(y_true, y_prob, model_name, plots_dir, show=show)
    plot_confusion_matrix(y_true, y_prob, model_name, threshold, plots_dir, show=show)

    return metrics