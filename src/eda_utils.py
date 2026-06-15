"""
src/eda_utils.py -- Real Estate Fraud Detection
Reusable EDA helper functions used in Day 3 notebook.
All config-driven -- no hardcoded column names or thresholds.

Functions:
    univariate_numerical()    -- distributions + skewness for all num cols
    univariate_categorical()  -- bar charts + cardinality for cat cols
    bivariate_correlation()   -- Pearson heatmap + fraud correlation
    fraud_vs_normal()         -- side-by-side distributions per feature
    mann_whitney_test()       -- statistical significance per feature
    fraud_by_geography()      -- state + city fraud rate charts
    price_per_sqft_analysis() -- ppsf distribution + fraud overlay
    scatter_fraud_highlight()  -- price vs size scatter, fraud=red
    generate_eda_summary()    -- dict of key numbers for eda_findings.md
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plot style defaults
# ---------------------------------------------------------------------------
FRAUD_COLOR  = "#E53935"   # red
NORMAL_COLOR = "#1E88E5"   # blue
NEUTRAL_COLOR = "#78909C"  # grey

def _save(fig, path: str, dpi: int = 150) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info(f"Plot saved -> {path}")


# ---------------------------------------------------------------------------
# 1. Univariate -- Numerical
# ---------------------------------------------------------------------------
def univariate_numerical(
    df: pd.DataFrame,
    cfg: dict,
    plots_dir: str = "reports/plots",
    show: bool = True,
) -> Dict:
    """
    For each numerical column:
      - Raw histogram with skewness annotation
      - log1p histogram
      - Boxplot (clipped at p99)
    Returns dict of skewness values.
    """
    num_cols = [c for c in cfg["columns"]["numerical"] if c in df.columns]
    skew_thresh = cfg["data_quality"]["skewness_log_transform_threshold"]
    results = {}

    fig, axes = plt.subplots(3, len(num_cols), figsize=(4 * len(num_cols), 10))
    if len(num_cols) == 1:
        axes = axes.reshape(-1, 1)

    for i, col in enumerate(num_cols):
        data = df[col].dropna()
        skewness = float(data.skew())
        results[col] = {"skewness": round(skewness, 3),
                        "log_transform": abs(skewness) > skew_thresh}
        p99 = data.quantile(0.99)

        # Row 0 -- raw histogram
        axes[0, i].hist(data.clip(upper=p99), bins=60,
                        color=NEUTRAL_COLOR, alpha=0.8, edgecolor="white")
        axes[0, i].set_title(f"{col}\nskew={skewness:.2f}", fontsize=10)
        axes[0, i].set_xlabel(col)
        if i == 0:
            axes[0, i].set_ylabel("Raw")

        # Row 1 -- log1p histogram
        log_data = np.log1p(data.clip(lower=0))
        axes[1, i].hist(log_data, bins=60,
                        color="#43A047", alpha=0.8, edgecolor="white")
        axes[1, i].set_title(f"log1p({col})\nskew={log_data.skew():.2f}", fontsize=10)
        axes[1, i].set_xlabel(f"log1p({col})")
        if i == 0:
            axes[1, i].set_ylabel("Log-transformed")

        # Row 2 -- boxplot
        axes[2, i].boxplot(data.clip(upper=p99).dropna(),
                           patch_artist=True,
                           boxprops=dict(facecolor=NEUTRAL_COLOR, alpha=0.6))
        axes[2, i].set_title(f"{col} boxplot\n(clipped p99)", fontsize=10)
        if i == 0:
            axes[2, i].set_ylabel("Boxplot")

    plt.suptitle("Numerical Feature Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, f"{plots_dir}/univariate_numerical.png")
    if show:
        plt.show()
    else:
        plt.close()

    logger.info(f"Univariate numerical -- needs log transform: "
                f"{[c for c, v in results.items() if v['log_transform']]}")
    return results


# ---------------------------------------------------------------------------
# 2. Univariate -- Categorical
# ---------------------------------------------------------------------------
def univariate_categorical(
    df: pd.DataFrame,
    cfg: dict,
    plots_dir: str = "reports/plots",
    show: bool = True,
) -> Dict:
    """Bar charts + cardinality for categorical + high-cardinality columns."""
    cat_cols = [c for c in cfg["columns"]["categorical"] if c in df.columns]
    hc_cols  = [c for c in cfg["columns"]["high_cardinality"] if c in df.columns]
    all_cols = cat_cols + hc_cols
    results  = {}

    for col in all_cols:
        n_unique = df[col].nunique()
        top_n = min(30, n_unique)
        top   = df[col].value_counts().head(top_n)

        fig, ax = plt.subplots(figsize=(max(10, top_n * 0.4), 4))
        ax.bar(top.index.astype(str), top.values,
               color=NEUTRAL_COLOR, alpha=0.85, edgecolor="white")
        ax.set_title(f"{col} -- Top {top_n} values  (total unique: {n_unique:,})", fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        _save(fig, f"{plots_dir}/univariate_{col}.png")
        if show:
            plt.show()
        else:
            plt.close()

        results[col] = {"n_unique": n_unique, "top_value": top.index[0],
                        "top_count": int(top.values[0])}

    return results


# ---------------------------------------------------------------------------
# 3. Correlation heatmap
# ---------------------------------------------------------------------------
def bivariate_correlation(
    df: pd.DataFrame,
    cfg: dict,
    plots_dir: str = "reports/plots",
    show: bool = True,
) -> pd.DataFrame:
    """
    Pearson correlation heatmap for numerical features + is_fraud.
    Returns correlation matrix.
    """
    num_cols = [c for c in cfg["columns"]["numerical"] if c in df.columns]
    target   = cfg["columns"]["target"]
    cols     = num_cols + ([target] if target in df.columns else [])

    corr = df[cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(len(cols) + 1, len(cols)))
    mask = np.zeros_like(corr, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1,
        mask=mask, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Pearson Correlation Matrix\n(numerical features + is_fraud)", fontsize=12)
    plt.tight_layout()
    _save(fig, f"{plots_dir}/correlation_heatmap.png")
    if show:
        plt.show()
    else:
        plt.close()

    # Log top correlations with is_fraud
    if target in corr.columns:
        fraud_corr = corr[target].drop(target).abs().sort_values(ascending=False)
        logger.info(f"Top correlations with {target}:")
        for feat, val in fraud_corr.head(5).items():
            logger.info(f"  {feat}: {val:.3f}")

    return corr


# ---------------------------------------------------------------------------
# 4. Fraud vs Normal -- side-by-side distributions
# ---------------------------------------------------------------------------
def fraud_vs_normal(
    df: pd.DataFrame,
    cfg: dict,
    plots_dir: str = "reports/plots",
    show: bool = True,
) -> pd.DataFrame:
    """
    Side-by-side density plots: fraud (red) vs normal (blue) per feature.
    Returns mean comparison table.
    """
    target   = cfg["columns"]["target"]
    num_cols = [c for c in cfg["columns"]["numerical"] if c in df.columns]

    if target not in df.columns:
        logger.warning(f"'{target}' column not found -- skipping fraud_vs_normal")
        return pd.DataFrame()

    fraud  = df[df[target] == 1]
    normal = df[df[target] == 0]

    fig, axes = plt.subplots(1, len(num_cols), figsize=(4.5 * len(num_cols), 4))
    if len(num_cols) == 1:
        axes = [axes]

    for i, col in enumerate(num_cols):
        p99 = df[col].quantile(0.99)
        bins = 50
        kw   = dict(bins=bins, density=True, alpha=0.55, edgecolor="none")
        axes[i].hist(normal[col].clip(upper=p99).dropna(),
                     color=NORMAL_COLOR, label="Normal", **kw)
        axes[i].hist(fraud[col].clip(upper=p99).dropna(),
                     color=FRAUD_COLOR,  label="Fraud",  **kw)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel(col)
        if i == 0:
            axes[i].set_ylabel("Density")
            axes[i].legend(fontsize=8)

    plt.suptitle(
        f"Feature Distributions: Fraud (n={len(fraud):,}) vs Normal (n={len(normal):,})",
        fontsize=13,
    )
    plt.tight_layout()
    _save(fig, f"{plots_dir}/fraud_vs_normal.png")
    if show:
        plt.show()
    else:
        plt.close()

    # Mean comparison table
    comparison = pd.DataFrame({
        "Normal mean":  normal[num_cols].mean().round(2),
        "Fraud mean":   fraud[num_cols].mean().round(2),
        "Ratio F/N":    (fraud[num_cols].mean() / normal[num_cols].mean()).round(3),
        "Diff %":       ((fraud[num_cols].mean() - normal[num_cols].mean())
                         / normal[num_cols].mean() * 100).round(1),
    })
    return comparison


# ---------------------------------------------------------------------------
# 5. Mann-Whitney U test -- statistical significance
# ---------------------------------------------------------------------------
def mann_whitney_test(
    df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Mann-Whitney U test for each numerical feature between fraud and normal.
    Returns DataFrame sorted by p-value (most significant first).
    """
    target   = cfg["columns"]["target"]
    num_cols = [c for c in cfg["columns"]["numerical"] if c in df.columns]

    if target not in df.columns:
        return pd.DataFrame()

    fraud  = df[df[target] == 1]
    normal = df[df[target] == 0]
    rows   = []

    for col in num_cols:
        f_vals = fraud[col].dropna()
        n_vals = normal[col].dropna()
        if len(f_vals) < 5 or len(n_vals) < 5:
            continue
        # Sample for speed on 2M rows
        f_sample = f_vals.sample(min(10000, len(f_vals)), random_state=42)
        n_sample = n_vals.sample(min(10000, len(n_vals)), random_state=42)
        stat, pval = stats.mannwhitneyu(f_sample, n_sample, alternative="two-sided")
        rows.append({
            "feature":        col,
            "fraud_median":   round(float(f_vals.median()), 2),
            "normal_median":  round(float(n_vals.median()), 2),
            "ratio_F_N":      round(float(f_vals.median() / max(n_vals.median(), 1e-9)), 3),
            "mw_statistic":   round(float(stat), 0),
            "p_value":        float(pval),
            "significant":    pval < 0.05,
        })

    result = pd.DataFrame(rows).sort_values("p_value")
    logger.info("Mann-Whitney U Test results:")
    for _, row in result.iterrows():
        sig = "[OK] significant" if row["significant"] else "[NO] not significant"
        logger.info(f"  {row['feature']:<15}: p={row['p_value']:.4f}  {sig}")
    return result


# ---------------------------------------------------------------------------
# 6. Fraud by geography
# ---------------------------------------------------------------------------
def fraud_by_geography(df: pd.DataFrame, cfg: dict, plots_dir: str = "reports/plots", show: bool = True,min_city_listings: int = 200, 
                       min_state_listings: int=50,) -> Tuple[pd.DataFrame, pd.DataFrame]:
   
    target = cfg["columns"]["target"]
    
    if target not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    national_avg = df[target].mean() * 100

    # -- State fraud rate ------------------------------------------------
    state_fraud = (
        df.groupby("state")[target]
        .agg(total="count", fraud="sum")
        .assign(fraud_rate=lambda x: x["fraud"] / x["total"] * 100)
        .query(f"total >= {min_state_listings}")
        .sort_values("fraud_rate", ascending=False)
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # State fraud rate -- top 30
    top30 = state_fraud.head(30)
    colors = [
        FRAUD_COLOR if r > national_avg * 1.5
        else "#FF7043" if r > national_avg
        else NORMAL_COLOR
        for r in top30["fraud_rate"]
    ]
    axes[0].barh(top30["state"][::-1], top30["fraud_rate"][::-1],
                 color=colors[::-1], alpha=0.85)
    axes[0].axvline(national_avg, color="black", ls="--", lw=1.2,
                    alpha=0.6, label=f"National avg {national_avg:.2f}%")
    axes[0].set_xlabel("Fraud Rate (%)")
    axes[0].set_title("Fraud Rate by State")
    axes[0].legend(fontsize=9)

    # Absolute fraud count by state
    top15_abs = state_fraud.sort_values("fraud", ascending=False).head(15)
    axes[1].bar(top15_abs["state"], top15_abs["fraud"],
                color="#7B1FA2", alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("State")
    axes[1].set_ylabel("Fraud Count")
    axes[1].set_title("Absolute Fraud Count by State (Top 15)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else str(int(x))
    ))
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

    plt.suptitle("Geographic Distribution of Fraud -- State Level", fontsize=13)
    plt.tight_layout()
    _save(fig, f"{plots_dir}/fraud_by_state.png")
    if show:
        plt.show()
    else:
        plt.close()

    # -- City fraud rate -------------------------------------------------
    city_fraud = (
        df.groupby("city")[target]
        .agg(total="count", fraud="sum")
        .query(f"total >= {min_city_listings}")
        .assign(fraud_rate=lambda x: x["fraud"] / x["total"] * 100)
        .sort_values("fraud_rate", ascending=False)
        .reset_index()
    )

    top20_cities = city_fraud.head(20)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors_c = [FRAUD_COLOR if r > national_avg * 2 else "#FF7043"
                for r in top20_cities["fraud_rate"]]
    ax.bar(top20_cities["city"], top20_cities["fraud_rate"],
           color=colors_c, alpha=0.85, edgecolor="white")
    ax.axhline(national_avg, color="black", ls="--", lw=1.2, alpha=0.6,
               label=f"National avg {national_avg:.2f}%")
    ax.set_xlabel("City")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_title(f"Top 20 Cities by Fraud Rate (min {min_city_listings:,} listings)")
    ax.legend(fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save(fig, f"{plots_dir}/fraud_by_city_top20.png")
    if show:
        plt.show()
    else:
        plt.close()

    logger.info(f"Geographic analysis complete -- "
                f"{len(state_fraud)} states, "
                f"{len(city_fraud)} cities (>={min_city_listings} listings)")
    return state_fraud, city_fraud


# ---------------------------------------------------------------------------
# 7. Price-per-sqft analysis
# ---------------------------------------------------------------------------
def price_per_sqft_analysis(
    df: pd.DataFrame,
    cfg: dict,
    plots_dir: str = "reports/plots",
    show: bool = True,
) -> pd.DataFrame:
    """
    Compute price_per_sqft and show fraud vs normal distribution.
    Returns summary stats.
    """
    target = cfg["columns"]["target"]

    tmp = df.copy()
    tmp["price_per_sqft"] = np.where(
        tmp["house_size"] > 0,
        tmp["price"] / tmp["house_size"],
        np.nan,
    )
    # Remove impossible ppsf values
    p01  = tmp["price_per_sqft"].quantile(0.01)
    p99  = tmp["price_per_sqft"].quantile(0.99)
    tmp  = tmp[(tmp["price_per_sqft"] >= p01) & (tmp["price_per_sqft"] <= p99)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution by fraud label
    if target in tmp.columns:
        fraud  = tmp[tmp[target] == 1]["price_per_sqft"].dropna()
        normal = tmp[tmp[target] == 0]["price_per_sqft"].dropna()

        axes[0].hist(normal, bins=60, alpha=0.55, density=True,
                     color=NORMAL_COLOR, label="Normal", edgecolor="none")
        axes[0].hist(fraud,  bins=60, alpha=0.55, density=True,
                     color=FRAUD_COLOR,  label="Fraud",  edgecolor="none")
        axes[0].axvline(normal.median(), color=NORMAL_COLOR, ls="--", lw=1.5,
                        label=f"Normal median ${normal.median():.0f}/sqft")
        axes[0].axvline(fraud.median(),  color=FRAUD_COLOR,  ls="--", lw=1.5,
                        label=f"Fraud median ${fraud.median():.0f}/sqft")
        axes[0].set_xlabel("Price per sqft ($)")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Price/sqft: Fraud vs Normal")
        axes[0].legend(fontsize=8)

    # Top 20 cities by median price_per_sqft
    city_ppsf = (
        tmp.groupby("city")["price_per_sqft"]
        .agg(median="median", count="count")
        .query("count >= 100")
        .sort_values("median", ascending=False)
        .head(20)
        .reset_index()
    )
    axes[1].barh(city_ppsf["city"][::-1], city_ppsf["median"][::-1],
                 color=NEUTRAL_COLOR, alpha=0.85, edgecolor="white")
    axes[1].set_xlabel("Median Price/sqft ($)")
    axes[1].set_title("Top 20 Cities by Median Price/sqft")

    plt.suptitle("Price per Square Foot Analysis", fontsize=13)
    plt.tight_layout()
    _save(fig, f"{plots_dir}/price_per_sqft_analysis.png")
    if show:
        plt.show()
    else:
        plt.close()

    summary = tmp.groupby(target)["price_per_sqft"].describe().round(2) \
        if target in tmp.columns else tmp["price_per_sqft"].describe().round(2)
    return summary


# ---------------------------------------------------------------------------
# 8. Price vs house_size scatter -- fraud highlighted
# ---------------------------------------------------------------------------
def scatter_fraud_highlight(
    df: pd.DataFrame,
    cfg: dict,
    plots_dir: str = "reports/plots",
    show: bool = True,
    sample_n: int = 8000,
) -> None:
    """Scatter: price vs house_size, fraud=red, normal=blue (sampled)."""
    target = cfg["columns"]["target"]
    sample = df.sample(min(sample_n, len(df)), random_state=42)

    p99_size  = df["house_size"].quantile(0.99)
    p99_price = df["price"].quantile(0.99)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Scatter 1: price vs house_size
    if target in sample.columns:
        normal_s = sample[sample[target] == 0]
        fraud_s  = sample[sample[target] == 1]
        axes[0].scatter(
            normal_s["house_size"].clip(upper=p99_size),
            normal_s["price"].clip(upper=p99_price),
            alpha=0.15, s=6, c=NORMAL_COLOR, label="Normal",
        )
        axes[0].scatter(
            fraud_s["house_size"].clip(upper=p99_size),
            fraud_s["price"].clip(upper=p99_price),
            alpha=0.5, s=10, c=FRAUD_COLOR, label="Fraud",
        )
        axes[0].legend(fontsize=9)
    else:
        axes[0].scatter(
            sample["house_size"].clip(upper=p99_size),
            sample["price"].clip(upper=p99_price),
            alpha=0.2, s=6, c=NEUTRAL_COLOR,
        )

    axes[0].set_xlabel("House Size (sqft)")
    axes[0].set_ylabel("Price (USD)")
    axes[0].set_title(f"Price vs House Size\n(sample {sample_n:,}, clipped at p99)")
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}K")
    )

    # Scatter 2: log price vs log house_size
    log_price = np.log1p(sample["price"].clip(lower=0))
    log_size  = np.log1p(sample["house_size"].clip(lower=0))

    if target in sample.columns:
        normal_s = sample[sample[target] == 0]
        fraud_s  = sample[sample[target] == 1]
        axes[1].scatter(
            np.log1p(normal_s["house_size"].clip(lower=0)),
            np.log1p(normal_s["price"].clip(lower=0)),
            alpha=0.15, s=6, c=NORMAL_COLOR, label="Normal",
        )
        axes[1].scatter(
            np.log1p(fraud_s["house_size"].clip(lower=0)),
            np.log1p(fraud_s["price"].clip(lower=0)),
            alpha=0.5, s=10, c=FRAUD_COLOR, label="Fraud",
        )
        axes[1].legend(fontsize=9)

    axes[1].set_xlabel("log1p(House Size)")
    axes[1].set_ylabel("log1p(Price)")
    axes[1].set_title("log(Price) vs log(House Size)\n(clearer pattern in log space)")

    plt.suptitle("Price vs House Size -- Fraud Highlighted", fontsize=13)
    plt.tight_layout()
    _save(fig, f"{plots_dir}/scatter_price_vs_size.png")
    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# 9. Fraud score distribution analysis
# ---------------------------------------------------------------------------
def fraud_score_analysis(
    df: pd.DataFrame,
    cfg: dict,
    plots_dir: str = "reports/plots",
    show: bool = True,
) -> None:
    """Fraud score distribution + rule co-occurrence heatmap."""
    from src.fraud_labeler import FraudLabeler

    rule_cols = [c for c in FraudLabeler.RULE_COLS if c in df.columns]
    threshold = cfg["fraud_rules"]["min_fraud_score_threshold"]

    if "fraud_score" not in df.columns or not rule_cols:
        logger.warning("fraud_score or rule columns not found -- skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Fraud score bar chart
    score_dist = df["fraud_score"].value_counts().sort_index()
    bar_colors = [FRAUD_COLOR if s >= threshold else NORMAL_COLOR
                  for s in score_dist.index]
    axes[0].bar(score_dist.index, score_dist.values,
                color=bar_colors, alpha=0.85, edgecolor="white")
    axes[0].axvline(threshold - 0.5, color="red", ls="--", lw=1.5,
                    label=f"Threshold >= {threshold}")
    axes[0].set_xlabel("Fraud Score (rules fired)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Fraud Score Distribution")
    axes[0].legend(fontsize=9)
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
    )
    for s, c in zip(score_dist.index, score_dist.values):
        axes[0].text(s, c * 1.02, f"{c/1000:.0f}K", ha="center", fontsize=8)

    # Rule co-occurrence heatmap
    rule_short = [c.replace("rule_", "") for c in rule_cols]
    co = df[rule_cols].T.dot(df[rule_cols]).astype(int)
    co.index   = rule_short
    co.columns = rule_short
    mask = np.eye(len(rule_cols), dtype=bool)
    sns.heatmap(co, annot=True, fmt="d", cmap="Blues",
                mask=mask, linewidths=0.5, ax=axes[1],
                cbar_kws={"label": "Co-occurrence"})
    axes[1].set_title("Rule Co-occurrence\n(how often 2 rules fire together)")

    plt.suptitle("Fraud Score & Rule Co-occurrence Analysis", fontsize=13)
    plt.tight_layout()
    _save(fig, f"{plots_dir}/fraud_score_analysis.png")
    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# 10. Generate EDA summary dict (numbers for eda_findings.md)
# ---------------------------------------------------------------------------
def generate_eda_summary(
    df: pd.DataFrame,
    cfg: dict,
    skew_results: Optional[Dict] = None,
    state_fraud: Optional[pd.DataFrame] = None,
    city_fraud: Optional[pd.DataFrame] = None,
    mw_results: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Collect all key numbers into a single dict.
    Used to auto-fill eda_findings.md template.
    """
    target   = cfg["columns"]["target"]
    num_cols = [c for c in cfg["columns"]["numerical"] if c in df.columns]

    fraud_rate  = df[target].mean() if target in df.columns else None
    fraud_count = int(df[target].sum()) if target in df.columns else None

    # Missing values
    missing = {col: round(df[col].isnull().mean() * 100, 2) for col in df.columns}

    # Price stats
    price_stats = {}
    if "price" in df.columns:
        p = df["price"].dropna()
        price_stats = {
            "mean":     round(float(p.mean()), 0),
            "median":   round(float(p.median()), 0),
            "std":      round(float(p.std()), 0),
            "skewness": round(float(p.skew()), 3),
            "p1":  round(float(p.quantile(0.01)), 0),
            "p5":  round(float(p.quantile(0.05)), 0),
            "p95": round(float(p.quantile(0.95)), 0),
            "p99": round(float(p.quantile(0.99)), 0),
        }

    # Top fraud states / cities
    top_fraud_states = []
    if state_fraud is not None and len(state_fraud):
        top_fraud_states = state_fraud.head(5)[
            ["state", "total", "fraud", "fraud_rate"]
        ].to_dict(orient="records")

    top_fraud_cities = []
    if city_fraud is not None and len(city_fraud):
        top_fraud_cities = city_fraud.head(5)[
            ["city", "total", "fraud", "fraud_rate"]
        ].to_dict(orient="records")

    # Most significant features
    sig_features = []
    if mw_results is not None and len(mw_results):
        sig_features = mw_results[mw_results["significant"]]["feature"].tolist()

    summary = {
        "shape":          df.shape,
        "total_rows":     len(df),
        "total_cols":     len(df.columns),
        "fraud_rate":     round(fraud_rate * 100, 2) if fraud_rate else None,
        "fraud_count":    fraud_count,
        "missing_pct":    missing,
        "price_stats":    price_stats,
        "skewness":       skew_results or {},
        "top_fraud_states": top_fraud_states,
        "top_fraud_cities": top_fraud_cities,
        "significant_features": sig_features,
        "national_avg_fraud_pct": round(fraud_rate * 100, 2) if fraud_rate else None,
    }

    logger.info("EDA summary generated -- key numbers:")
    logger.info(f"  Fraud rate  : {summary['fraud_rate']}%")
    logger.info(f"  Price median: ${price_stats.get('median', 'N/A'):,}")
    logger.info(f"  Sig features: {sig_features}")

    return summary