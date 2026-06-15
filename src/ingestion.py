"""
src/ingestion.py — Real Estate Fraud Detection
Data loading, schema validation, dtype enforcement, and summary logging.
All behaviour driven by configs/config.yaml — no hardcoded values.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config(config_path: str = "configs/config.yaml") -> dict:
    # Path(__file__) = src/ingestion.py → .parent = src/ → .parent = project root
    # Isse "python src/ingestion.py" aur "cd src && python ingestion.py" dono kaam karte hain
    config_path = Path(config_path)
    if not config_path.is_absolute() and not config_path.exists():
        root_relative = Path(__file__).parent.parent / config_path
        if root_relative.exists():
            config_path = root_relative
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    logger.info(f"Config loaded from {config_path} — project: {cfg['project']['name']} v{cfg['project']['version']}")
    return cfg


# ---------------------------------------------------------------------------
# Raw data loader
# ---------------------------------------------------------------------------
def load_raw_data(cfg: dict) -> pd.DataFrame:
    """
    Load CSV from path defined in config.
    Enforces dtypes from config['dtypes'].
    Drops columns listed in config['columns']['drop'].
    """
    raw_path = Path(cfg["data"]["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at: {raw_path}\n"
            f"Then place CSV at: {raw_path}"
        )

    logger.info(f"Loading raw data from: {raw_path}")
    
    df = pd.read_csv(raw_path, low_memory=False)
    logger.info(f"Raw shape: {df.shape}")

    # Enforce dtypes from config
    dtype_map = cfg.get("dtypes", {})
    for col, dtype in dtype_map.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError) as e:
                logger.warning(f"  dtype cast failed for '{col}' → {dtype}: {e}")

    # Drop columns listed in config
    drop_cols = [c for c in cfg["columns"].get("drop", []) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info(f"Dropped columns: {drop_cols}")

    logger.info(f"Shape after drops: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame, cfg: dict) -> None:
    """
    Assert expected columns are present.
    Crash loudly if critical columns are missing — better than silent bad results.
    """
    # Columns we actually need (after drops)
    drop_set = set(cfg["columns"].get("drop", []))
    expected = [c for c in cfg["columns"]["all"] if c not in drop_set]

    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"Schema validation FAILED — missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Critical columns must exist — is_fraud yahan check NAHI hoga
    # (wo fraud_labeler.py banayega, ingestion ke time df mein hoti hi nahi)
    critical = cfg["columns"]["numerical"] + cfg["columns"]["categorical"]
    
    critical_missing = [c for c in critical if c not in df.columns]
    
    if critical_missing:
        raise ValueError(f"Critical columns missing: {critical_missing}")

    logger.info(f"Schema validation PASSED — {len(df.columns)} columns present")


# ---------------------------------------------------------------------------
# Duplicate check
# ---------------------------------------------------------------------------
def assert_no_duplicates(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Check for duplicate rows. Log count, drop them, warn loudly.
    Uses subset of numerical + categorical columns (not all — some may be ID-like).
    """
    # city aur zip_code bhi include karo — warna same-city duplicate listings miss ho jayenge
    # (fraud rule: same bed+bath+size in same zip = duplicate listing)
    subset_cols = (
        cfg["columns"]["numerical"]
        + cfg["columns"]["categorical"]
        + cfg["columns"].get("high_cardinality", [])
    )
    subset_cols = [c for c in subset_cols if c in df.columns]

    n_before = len(df)
    n_dupes = df.duplicated(subset=subset_cols).sum()

    if n_dupes > 0:
        logger.warning(f"Found {n_dupes} duplicate rows ({n_dupes/n_before*100:.2f}%) — dropping")
        df = df.drop_duplicates(subset=subset_cols).reset_index(drop=True)
    else:
        logger.info(f"No duplicate rows found (checked subset of {len(subset_cols)} columns)")

    logger.info(f"Shape after dedup: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Missing value analysis
# ---------------------------------------------------------------------------
def analyze_missing(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Compute missing value stats for every column.
    Warn if above warn threshold, raise if above critical threshold.
    Returns a summary DataFrame (also useful in notebooks).
    """
    warn_thresh = cfg["data_quality"]["max_null_pct_warn"]
    critical_thresh = cfg["data_quality"]["max_null_pct_critical"]

    null_pct = (df.isnull().sum() / len(df) * 100).round(2)
    null_count = df.isnull().sum()

    summary = pd.DataFrame({
        "column": df.columns,
        "null_count": null_count.values,
        "null_pct": null_pct.values,
        "dtype": df.dtypes.values,
    }).sort_values("null_pct", ascending=False).reset_index(drop=True)

    # Log warnings / critical
    for _, row in summary.iterrows():
        if row["null_pct"] >= critical_thresh:
            logger.error(f"CRITICAL NULL: '{row['column']}' has {row['null_pct']:.1f}% missing — consider dropping")
        elif row["null_pct"] >= warn_thresh:
            logger.warning(f"HIGH NULL: '{row['column']}' has {row['null_pct']:.1f}% missing")

    cols_ok = (summary["null_pct"] < warn_thresh).sum()
    logger.info(f"Missing value analysis: {cols_ok}/{len(df.columns)} columns below {warn_thresh}% threshold")
    return summary


# ---------------------------------------------------------------------------
# Data summary logger
# ---------------------------------------------------------------------------
def log_data_summary(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Comprehensive data summary — shape, dtypes, nulls, numeric stats, cardinality.
    Returns a dict so notebooks can display it cleanly.
    """
    min_rows = cfg["data_quality"]["min_rows_required"]
    if len(df) < min_rows:
        raise ValueError(f"Dataset too small: {len(df)} rows < required {min_rows}")

    num_cols = [c for c in cfg["columns"]["numerical"] if c in df.columns]
    cat_cols = [c for c in cfg["columns"]["categorical"] if c in df.columns]
    hc_cols  = [c for c in cfg["columns"].get("high_cardinality", []) if c in df.columns]

    # Skewness check
    skew_thresh = cfg["data_quality"]["skewness_log_transform_threshold"]
    skewness = df[num_cols].skew().round(3)
    high_skew = skewness[skewness.abs() > skew_thresh]

    summary = {
        "shape": df.shape,
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "numerical_cols": num_cols,
        "categorical_cols": cat_cols,
        "high_cardinality_cols": hc_cols,
        "numerical_describe": df[num_cols].describe().round(2).to_dict() if num_cols else {},
        "cardinality": {c: df[c].nunique() for c in cat_cols + hc_cols if c in df.columns},
        "skewness": skewness.to_dict(),
        "high_skew_cols": high_skew.to_dict(),
        "needs_log_transform": list(high_skew.index),
    }

    # Print structured summary
    logger.info("=" * 60)
    logger.info(f"DATA SUMMARY — {cfg['project']['name']}")
    logger.info("=" * 60)
    logger.info(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"  Numerical cols : {num_cols}")
    logger.info(f"  Categorical    : {cat_cols}")
    logger.info(f"  High cardinality: {hc_cols}")
    logger.info(f"  Cardinality    : { {c: df[c].nunique() for c in hc_cols if c in df.columns} }")
    if not high_skew.empty:
        logger.info(f"  High-skew cols (|skew| > {skew_thresh}): {high_skew.to_dict()}")
        logger.info(f"  → Log transform recommended for: {list(high_skew.index)}")
    logger.info("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# Price distribution check
# ---------------------------------------------------------------------------
def check_price_distribution(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Analyse price column distribution.
    Returns skewness, percentiles — used in Day 1 audit notebook.
    """
    price_col = "price"
    if price_col not in df.columns:
        logger.warning("'price' column not found — skipping price distribution check")
        return {}

    price = df[price_col].dropna()
    skewness = float(price.skew())
    skew_thresh = cfg["data_quality"]["skewness_log_transform_threshold"]

    result = {
        "count": len(price),
        "mean": round(float(price.mean()), 2),
        "median": round(float(price.median()), 2),
        "std": round(float(price.std()), 2),
        "min": round(float(price.min()), 2),
        "max": round(float(price.max()), 2),
        "skewness": round(skewness, 3),
        "log_transform_recommended": abs(skewness) > skew_thresh,
        "percentiles": {
            f"p{p}": round(float(np.percentile(price, p)), 2)
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        },
    }

    logger.info(f"Price distribution: mean=${result['mean']:,.0f}, median=${result['median']:,.0f}, skew={skewness:.3f}")
    if result["log_transform_recommended"]:
        logger.info(f"  → Log transform recommended (|skew|={abs(skewness):.3f} > {skew_thresh})")

    return result


# ---------------------------------------------------------------------------
# Leakage risk report
# ---------------------------------------------------------------------------
def generate_leakage_report(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Build feature semantics + leakage risk table.
    Reads leakage info from problem_contract column definitions.
    """
    # Leakage risk map — driven by config column categories
    drop_cols   = set(cfg["columns"].get("drop", []))
    hc_cols     = set(cfg["columns"].get("high_cardinality", []))
    temporal    = set(cfg["columns"].get("temporal", []))
    numerical   = set(cfg["columns"].get("numerical", []))
    categorical = set(cfg["columns"].get("categorical", []))

    # df.columns mein sirf remaining columns hain — drop ho chuke columns bhi report mein dikhao
    # warna leakage_report.csv mein street aur brokered_by missing dikhenge
    all_cols_to_report = list(df.columns) + [c for c in drop_cols if c not in df.columns]

    rows = []
    for col in all_cols_to_report:
        if col in drop_cols:
            risk = "HIGH — DROPPED"
            action = "Already dropped"
        elif col in temporal:
            risk = "MEDIUM"
            action = "Extract year/month features only"
        elif col in hc_cols:
            risk = "MEDIUM"
            action = "Frequency or target encoding (OOF-safe)"
        elif col in numerical:
            risk = "LOW"
            action = "Keep — standard scaling"
        elif col in categorical:
            risk = "LOW"
            action = "Ordinal or OHE encoding"
        else:
            risk = "UNKNOWN"
            action = "Review manually"

        rows.append({
            "column": col,
            "dtype": str(df[col].dtype) if col in df.columns else "N/A — dropped",
            "nunique": df[col].nunique() if col in df.columns else "N/A",
            "null_pct": round(df[col].isnull().mean() * 100, 2) if col in df.columns else "N/A",
            "leakage_risk": risk,
            "action": action,
        })

    report = pd.DataFrame(rows)
    logger.info(f"Leakage report generated — {len(report)} columns assessed")
    return report


# ---------------------------------------------------------------------------
# Master ingestion pipeline
# ---------------------------------------------------------------------------
def run_ingestion(config_path: str = "configs/config.yaml") -> Tuple[pd.DataFrame, dict]:
    """
    Full Day 1 ingestion pipeline:
      1. Load config
      2. Load raw CSV
      3. Validate schema
      4. Remove duplicates
      5. Analyse missing values
      6. Log data summary
      7. Check price distribution
      8. Generate leakage report
    Returns (df, summary_dict)
    """
    cfg = load_config(config_path)

    df = load_raw_data(cfg)
    validate_schema(df, cfg)
    df = assert_no_duplicates(df, cfg)

    missing_summary = analyze_missing(df, cfg)
    data_summary    = log_data_summary(df, cfg)
    price_info      = check_price_distribution(df, cfg)
    leakage_report  = generate_leakage_report(df, cfg)

    # Save processed snapshot
    processed_path = Path(cfg["data"]["processed_path"])
    processed_path.mkdir(parents=True, exist_ok=True)

    # reports/plots/ auto-create — baad ke scripts (EDA, SHAP) ko manually banana nahi padega
    plots_path = Path(cfg["paths"]["plots"])
    plots_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directories exist: {processed_path}, {plots_path}")
    out_path = processed_path / "ingested.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Ingested data saved → {out_path}")

    # Save leakage report
    leakage_path = processed_path / "leakage_report.csv"
    leakage_report.to_csv(leakage_path, index=False)
    logger.info(f"Leakage report saved → {leakage_path}")

    full_summary = {
        **data_summary,
        "price_distribution": price_info,
        "missing_summary": missing_summary.to_dict(orient="records"),
    }

    return df, full_summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    df, summary = run_ingestion(config_path)
    print(f"\n✅ Ingestion complete — {summary['total_rows']:,} rows, {summary['total_cols']} cols")
    if summary.get("needs_log_transform"):
        print(f"⚠️  Log transform needed for: {summary['needs_log_transform']}")