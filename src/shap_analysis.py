"""
src/shap_analysis.py — Real Estate Fraud Detection
SHAP explainability — TreeExplainer for LightGBM.

Visualizations:
  - Summary plot (beeswarm)  : overall feature importance
  - Bar plot                 : feature ranking — portfolio main plot
  - Waterfall plot           : individual fraud case breakdown
  - Dependence plot          : price_per_sqft SHAP vs actual value
  - Force plot               : push/pull for fraud vs normal

Interview point:
  Production mein sirf is_fraud=True bolna kaafi nahi.
  Investigator ko specific signal pata hona chahiye — SHAP deta hai.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP value computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(
    model,
    X: pd.DataFrame,
    cfg: dict,
    sample_n: int = 2000,
    random_state: int = 42,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values using TreeExplainer (fast for LightGBM).

    Samples data if X is large — SHAP plots dont need full dataset.
    Returns (shap_values, X_sample).
    """
    try:
        import shap
    except ImportError:
        raise ImportError("pip install shap")

    # Sample for speed
    if len(X) > sample_n:
        X_sample = X.sample(sample_n, random_state=random_state).reset_index(drop=True)
    else:
        X_sample = X.reset_index(drop=True)

    logger.info(f"Computing SHAP values — {len(X_sample):,} samples")

    # TreeExplainer — fast, exact for tree models
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # LightGBM binary: shap_values may be list [neg_class, pos_class]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]   # positive class (fraud)

    logger.info(f"SHAP values shape: {shap_values.shape}")
    return shap_values, X_sample


def get_feature_importance_df(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build feature importance DataFrame sorted by mean |SHAP|.
    """
    importance = pd.DataFrame({
        "feature":    X_sample.columns,
        "mean_shap":  np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_shap", ascending=False).reset_index(drop=True)
    importance["rank"] = importance.index + 1
    return importance


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    plots_dir: str = "reports/plots",
    max_display: int = 20,
    show: bool = True,
) -> None:
    """
    Beeswarm summary plot — top 20 features, each dot = one listing.
    Color = feature value (red = high, blue = low).
    """
    try:
        import shap
    except ImportError:
        raise ImportError("pip install shap")

    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Summary — Top Feature Impact on Fraud Score", fontsize=13, pad=12)
    plt.tight_layout()
    out = f"{plots_dir}/shap_summary_beeswarm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    logger.info(f"Summary plot saved → {out}")
    if show:
        plt.show()
    plt.close()


def plot_bar_importance(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    plots_dir: str = "reports/plots",
    max_display: int = 20,
    show: bool = True,
) -> pd.DataFrame:
    """
    Bar plot — mean |SHAP| per feature.
    Portfolio main plot — clean, easy to explain in interviews.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("pip install shap")

    importance_df = get_feature_importance_df(shap_values, X_sample)
    top_n = importance_df.head(max_display)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#E53935" if i < 5 else "#1E88E5" if i < 10 else "#78909C"
              for i in range(len(top_n))]
    ax.barh(top_n["feature"][::-1], top_n["mean_shap"][::-1],
            color=colors[::-1], alpha=0.85, edgecolor="white")
    ax.set_xlabel("Mean |SHAP Value| (impact on fraud probability)")
    ax.set_title(f"Feature Importance — Top {max_display} Features\n(SHAP — Real Estate Fraud Detection)", fontsize=12)

    # Annotate top 5
    for i, (_, row) in enumerate(top_n.head(5).iterrows()):
        ax.text(row["mean_shap"] + max(top_n["mean_shap"]) * 0.01,
                max_display - 1 - i,
                f'#{row["rank"]}', va="center", fontsize=9, color="#E53935")

    plt.tight_layout()
    out = f"{plots_dir}/shap_bar_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    logger.info(f"Bar plot saved → {out}")
    if show:
        plt.show()
    plt.close()

    return importance_df


def plot_waterfall(
    model,
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    y_sample: pd.Series,
    plots_dir: str = "reports/plots",
    n_cases: int = 3,
    show: bool = True,
) -> List[int]:
    """
    Waterfall plots for N specific fraud cases.
    Shows exactly which features pushed the score high.
    Interview: 'Investigator ko pata chalega kaunsa signal tha.'
    """
    try:
        import shap
    except ImportError:
        raise ImportError("pip install shap")

    # Pick top fraud cases (highest predicted score)
    fraud_idx = y_sample[y_sample == 1].index.tolist()
    if len(fraud_idx) == 0:
        logger.warning("No fraud cases in sample — using top predicted scores")
        fraud_idx = list(range(min(n_cases, len(X_sample))))

    # Sort by SHAP magnitude — most extreme fraud cases first
    fraud_shap_sums = [(i, shap_values[i].sum()) for i in fraud_idx]
    fraud_shap_sums.sort(key=lambda x: x[1], reverse=True)
    selected_idx = [i for i, _ in fraud_shap_sums[:n_cases]]

    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    for case_num, idx in enumerate(selected_idx):
        plt.figure(figsize=(10, 6))
        shap_exp = shap.Explanation(
            values=shap_values[idx],
            base_values=shap_values.mean(),
            data=X_sample.iloc[idx].values,
            feature_names=list(X_sample.columns),
        )
        shap.waterfall_plot(shap_exp, max_display=12, show=False)
        plt.title(f"Fraud Case #{case_num+1} — Feature Contribution Breakdown\n"
                  f"(Listing index: {idx})", fontsize=11)
        plt.tight_layout()
        out = f"{plots_dir}/shap_waterfall_fraud_{case_num+1}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        logger.info(f"Waterfall plot {case_num+1} saved → {out}")
        if show:
            plt.show()
        plt.close()

    return selected_idx


def plot_dependence(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    feature: str = "price_per_sqft",
    interaction_feature: str = "price_vs_city_median",
    plots_dir: str = "reports/plots",
    show: bool = True,
) -> None:
    """
    Dependence plot — SHAP value vs actual feature value.
    Shows how price_per_sqft impacts fraud score across its range.
    Color = interaction feature (price_vs_city_median).
    """
    try:
        import shap
    except ImportError:
        raise ImportError("pip install shap")

    if feature not in X_sample.columns:
        logger.warning(f"Feature '{feature}' not in X_sample — skipping dependence plot")
        return

    feat_idx  = list(X_sample.columns).index(feature)
    inter_idx = (list(X_sample.columns).index(interaction_feature)
                 if interaction_feature in X_sample.columns else "auto")

    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feat_idx,
        shap_values,
        X_sample,
        interaction_index=inter_idx,
        show=False,
        alpha=0.4,
    )
    plt.title(
        f"SHAP Dependence Plot — {feature}\n"
        f"(Color = {interaction_feature})", fontsize=12
    )
    plt.tight_layout()
    out = f"{plots_dir}/shap_dependence_{feature}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    logger.info(f"Dependence plot saved → {out}")
    if show:
        plt.show()
    plt.close()


def plot_force_comparison(
    model,
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    y_sample: pd.Series,
    plots_dir: str = "reports/plots",
    show: bool = True,
) -> None:
    """
    Side-by-side force plots — one fraud vs one normal listing.
    Shows push/pull of each feature on final fraud score.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("pip install shap")

    fraud_indices  = y_sample[y_sample == 1].index.tolist()
    normal_indices = y_sample[y_sample == 0].index.tolist()

    if not fraud_indices or not normal_indices:
        logger.warning("Need both fraud and normal cases for force comparison")
        return

    # Pick highest-confidence fraud and lowest-confidence normal
    fraud_idx  = max(fraud_indices,  key=lambda i: shap_values[i].sum())
    normal_idx = min(normal_indices, key=lambda i: shap_values[i].sum())

    base_val = shap_values.mean()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for ax, idx, label, color in [
        (axes[0], fraud_idx,  "FRAUD listing",  "#E53935"),
        (axes[1], normal_idx, "NORMAL listing", "#1E88E5"),
    ]:
        sv     = shap_values[idx]
        feats  = list(X_sample.columns)
        vals   = X_sample.iloc[idx].values

        # Sort by |SHAP| — top 10
        top_idx = np.argsort(np.abs(sv))[-10:][::-1]
        top_sv  = sv[top_idx]
        top_f   = [feats[i] for i in top_idx]
        top_v   = vals[top_idx]

        bar_colors = ["#E53935" if s > 0 else "#1E88E5" for s in top_sv]
        ax.barh(range(len(top_sv)), top_sv[::-1], color=bar_colors[::-1], alpha=0.85)
        ax.set_yticks(range(len(top_sv)))
        ax.set_yticklabels(
            [f"{f} = {v:.2f}" for f, v in zip(top_f[::-1], top_v[::-1])],
            fontsize=9
        )
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{label} — Top Feature Contributions (red=fraud push, blue=pulls back)",
                     color=color, fontsize=11)
        ax.set_xlabel("SHAP Value")

    plt.suptitle("Force Comparison: Fraud vs Normal Listing", fontsize=13)
    plt.tight_layout()
    out = f"{plots_dir}/shap_force_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    logger.info(f"Force comparison saved → {out}")
    if show:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Top-3 SHAP for inference (used in FastAPI response)
# ─────────────────────────────────────────────────────────────────────────────

def get_top3_shap(
    model,
    X_row: pd.DataFrame,
) -> List[Dict]:
    """
    Get top 3 SHAP features for a single prediction.
    Used in FastAPI /predict response and Streamlit dashboard.

    Returns: [{'feature': str, 'impact': float, 'value': float}, ...]
    """
    try:
        import shap
    except ImportError:
        raise ImportError("pip install shap")

    explainer   = shap.TreeExplainer(model)
    shap_vals   = explainer.shap_values(X_row)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    sv   = shap_vals[0]
    top3 = np.argsort(np.abs(sv))[-3:][::-1]

    return [
        {
            "feature": str(X_row.columns[i]),
            "impact":  round(float(sv[i]), 4),
            "value":   round(float(X_row.iloc[0, i]), 4),
        }
        for i in top3
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Business insights writer
# ─────────────────────────────────────────────────────────────────────────────

def write_business_insights(
    importance_df: pd.DataFrame,
    cfg: dict,
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
) -> str:
    """
    Auto-generate reports/business_insights.md from SHAP analysis.
    """
    top5      = importance_df.head(5)
    fraud_df  = X_sample[y_sample == 1]
    normal_df = X_sample[y_sample == 0]

    rows = ""
    for _, row in top5.iterrows():
        feat = row["feature"]
        if feat in fraud_df.columns:
            f_mean = fraud_df[feat].mean()
            n_mean = normal_df[feat].mean()
            ratio  = f_mean / n_mean if n_mean != 0 else float("inf")
            rows += f"| `{feat}` | #{int(row['rank'])} | {row['mean_shap']:.4f} | {f_mean:.2f} | {n_mean:.2f} | {ratio:.2f}x |\n"

    content = f"""# Business Insights — SHAP Analysis
**Real Estate Fraud Detection**

---

## Key Finding

**Top fraud signal:** `{top5.iloc[0]['feature']}` — highest mean |SHAP| value

These features are most predictive of fraudulent listings based on SHAP analysis.

---

## Top 5 Fraud-Indicating Features

| Feature | Rank | Mean |SHAP| | Fraud Mean | Normal Mean | Ratio |
|---------|------|------------|------------|-------------|-------|
{rows}
---

## Interview Talking Points

1. **Why SHAP over feature importance?**
   Feature importance tells you *which* features matter globally.
   SHAP tells you *how* and *how much* each feature affected *each prediction*.

2. **Production use:**
   Every fraud alert includes top-3 SHAP features — investigator knows
   exactly which signal triggered the alert, not just is_fraud=True.

3. **Price signal:**
   Listings priced far below city median (price_vs_city_median < 0.3)
   strongly push fraud score up — consistent with fake listing fraud pattern.

4. **Acre lot anomaly:**
   Unusually large acre_lot values with normal house_size = data entry fraud signal.

---

*Generated by: `notebooks/09_shap.ipynb`*
"""

    out_path = Path(cfg["paths"]["business_insights"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(content)
    logger.info(f"Business insights saved → {out_path}")
    return str(out_path)
