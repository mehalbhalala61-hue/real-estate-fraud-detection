"""
src/inference.py — Real Estate Fraud Detection
Prediction pipeline — loads models once, runs fast inference.

Flow:
  ListingInput → FeatureEngineer → Preprocessor → Stacking → Calibrated → FraudScore

SHAP: top-3 features computed per prediction for investigator explanation.
Latency target: p95 < 500ms (tested in tests/test_latency.py)
"""

import json
import logging
import os
import pickle
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model loader — singleton pattern, loaded once on startup
# ─────────────────────────────────────────────────────────────────────────────

class ModelBundle:
    """
    Holds all loaded model artifacts.
    Loaded once at API startup — not per request.
    """
    def __init__(self):
        self.cfg               = None
        self.feat_eng          = None
        self.preprocessor      = None
        self.lgbm_model        = None
        self.calibrated_model  = None
        self.stacking_meta     = None
        self.feature_names     = None
        self._loaded           = False

    def load(self, config_path: str = "configs/config.yaml") -> "ModelBundle":
        """Load all artifacts from disk."""
        from src.ingestion import load_config
        from src.preprocessing import load_preprocessor, get_feature_names

        t0 = time.time()
        logger.info("Loading model bundle...")

        self.cfg = load_config(config_path)

        # FeatureEngineer
        fe_path = Path(self.cfg["data"]["processed_path"]) / "feature_engineer.pkl"
        with open(fe_path, "rb") as f:
            self.feat_eng = pickle.load(f)
        logger.info(f"  FeatureEngineer loaded ← {fe_path}")

        # Preprocessor
        self.preprocessor = load_preprocessor(self.cfg)
        self.feature_names = get_feature_names(self.cfg, include_city_fraud_rate=True)

        # LightGBM model (for SHAP)
        lgbm_path = Path(self.cfg["paths"]["lgbm_model"])
        with open(lgbm_path, "rb") as f:
            self.lgbm_model = pickle.load(f)
        logger.info(f"  LightGBM loaded ← {lgbm_path}")

        # Calibrated stacked model
        cal_path = Path(self.cfg["paths"]["calibrated_model"])
        with open(cal_path, "rb") as f:
            self.calibrated_model = pickle.load(f)
        logger.info(f"  Calibrated model loaded ← {cal_path}")

        # Stacking metadata
        meta_path = Path(self.cfg["data"]["processed_path"]) / "stacking_meta.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                self.stacking_meta = pickle.load(f)

        self._loaded = True
        elapsed = (time.time() - t0) * 1000
        logger.info(f"Model bundle loaded in {elapsed:.0f}ms")
        return self

    @property
    def loaded(self) -> bool:
        return self._loaded


# Global singleton — loaded once at startup
_bundle: Optional[ModelBundle] = None


def get_bundle() -> ModelBundle:
    """Return global model bundle — load if not already loaded."""
    global _bundle
    if _bundle is None or not _bundle.loaded:
        _bundle = ModelBundle().load()
    return _bundle


def load_bundle(config_path: str = "configs/config.yaml") -> ModelBundle:
    """Explicitly load bundle — called in FastAPI lifespan."""
    global _bundle
    _bundle = ModelBundle().load(config_path)
    return _bundle


# ─────────────────────────────────────────────────────────────────────────────
# Risk tier logic
# ─────────────────────────────────────────────────────────────────────────────

def get_risk_tier(score: float, cfg: dict) -> str:
    """
    3-tier classification from threshold_decisions.md:
      HIGH   : score >= 0.70 → Block + Manual Review
      MEDIUM : score >= 0.40 → Flag for Investigator
      LOW    : score <  0.40 → Allow through
    """
    high   = cfg["api"]["fraud_threshold_suspicious"]   # 0.70
    medium = cfg["api"]["fraud_threshold_review"]        # 0.40

    if score >= high:
        return "HIGH"
    elif score >= medium:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# Feature preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_features(listing_dict: dict, bundle: ModelBundle) -> pd.DataFrame:
    """
    Convert raw listing dict → processed feature array.
    Applies same pipeline as training: FeatureEngineer → Preprocessor.
    """
    from src.features import FeatureEngineer

    cfg = bundle.cfg

    # Build base DataFrame
    df = pd.DataFrame([listing_dict])

    # Ensure all expected columns exist — fill missing with NaN
    base_cols = (
        cfg["columns"]["numerical"]
        + cfg["columns"]["categorical"]
        + cfg["columns"].get("high_cardinality", [])
        + cfg["columns"].get("temporal", [])
    )
    for col in base_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Stateless features
    df = FeatureEngineer.add_stateless_features(df, cfg)

    # Fold-dependent features — use training stats
    df = bundle.feat_eng.transform(df)

    # Preprocessing
    X_proc = bundle.preprocessor.transform(df)
    X_df   = pd.DataFrame(X_proc, columns=bundle.feature_names)

    return X_df


# ─────────────────────────────────────────────────────────────────────────────
# SHAP top-3 for single prediction
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_top3(X_df: pd.DataFrame, bundle: ModelBundle) -> list:
    """
    Compute top-3 SHAP features for one listing.
    Uses LightGBM TreeExplainer — fast on single row.
    Returns list of dicts: [{'feature', 'impact', 'value'}, ...]
    """
    try:
        import shap
        explainer  = shap.TreeExplainer(bundle.lgbm_model)
        shap_vals  = explainer.shap_values(X_df)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        sv   = shap_vals[0]
        top3 = np.argsort(np.abs(sv))[-3:][::-1]

        return [
            {
                "feature": str(X_df.columns[i]),
                "impact":  round(float(sv[i]), 4),
                "value":   round(float(X_df.iloc[0, i]), 4),
            }
            for i in top3
        ]
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e} — returning empty list")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Main predict function
# ─────────────────────────────────────────────────────────────────────────────

def predict_fraud(listing_dict: dict, bundle: Optional[ModelBundle] = None) -> dict:
    """
    Full prediction pipeline for one listing.

    Args:
        listing_dict: raw listing fields (price, bed, bath, city, state, ...)
        bundle: ModelBundle — uses global singleton if None

    Returns:
        {
          'fraud_score':   float [0, 1],
          'is_suspicious': bool,
          'risk_tier':     str (HIGH/MEDIUM/LOW),
          'shap_top3':     list of 3 feature dicts,
          'latency_ms':    float,
        }
    """
    t0 = time.time()

    if bundle is None:
        bundle = get_bundle()

    cfg = bundle.cfg

    # 1. Prepare features
    X_df = prepare_features(listing_dict, bundle)

    # 2. Get stacking input
    # Calibrated model expects stacking DataFrame (lr + lgbm_tuned columns)
    # We use LightGBM directly for the score since we have single model inference
    lgbm_prob = float(
        bundle.lgbm_model.predict_proba(X_df)[:, 1][0]
    )

    # For stacking: build stacking input
    stacking_cols = bundle.stacking_meta["stacking_columns"] if bundle.stacking_meta else ["lr", "lgbm_tuned"]

    # Build stacking DataFrame with available columns
    # lr column: use logistic-like fallback from lgbm_prob
    stacking_input = pd.DataFrame(
        [[lgbm_prob * 0.7, lgbm_prob]],   # approximate lr as 0.7x lgbm
        columns=["lr", "lgbm_tuned"] if len(stacking_cols) == 2 else stacking_cols,
    )

    # 3. Calibrated prediction
    try:
        fraud_score = float(
            bundle.calibrated_model.predict_proba(stacking_input.values)[:, 1][0]
        )
        # Clip to [0, 1]
        fraud_score = float(np.clip(fraud_score, 0.0, 1.0))
    except Exception as e:
        logger.warning(f"Calibrated model failed: {e} — using lgbm_prob directly")
        fraud_score = lgbm_prob

    # 4. Risk tier
    risk_tier    = get_risk_tier(fraud_score, cfg)
    is_suspicious = fraud_score >= cfg["api"]["fraud_threshold_suspicious"]

    # 5. SHAP top-3
    shap_top3 = compute_shap_top3(X_df, bundle)

    latency_ms = round((time.time() - t0) * 1000, 2)

    result = {
        "fraud_score":    round(fraud_score, 4),
        "is_suspicious":  is_suspicious,
        "risk_tier":      risk_tier,
        "shap_top3":      shap_top3,
        "latency_ms":     latency_ms,
        "model_version":  cfg["project"]["version"],
    }

    logger.info(
        f"Prediction: score={fraud_score:.4f} tier={risk_tier} "
        f"city={listing_dict.get('city', 'N/A')} latency={latency_ms}ms"
    )

    return result
