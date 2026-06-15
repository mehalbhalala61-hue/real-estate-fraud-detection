"""
src/preprocessing.py — Real Estate Fraud Detection
ColumnTransformer pipeline — fit on train only, transform on val/test.

LEAKAGE RULE: preprocessor.fit() ONLY on training data.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


def build_preprocessor(cfg: dict, include_city_fraud_rate: bool = True) -> ColumnTransformer:
    """
    Build sklearn ColumnTransformer from config.
    Does NOT fit — call preprocessor.fit(X_train) separately.

    include_city_fraud_rate: pass False if FeatureEngineer was fit without y.
    city_fraud_rate column won't exist in df then → KeyError in preprocessor.fit().
    Default True — normal flow always calls feat_eng.fit(X_train, y=y_train).
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    num_cols = _get_all_numerical_cols(cfg, include_city_fraud_rate)
    cat_cols = _get_all_categorical_cols(cfg)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
    )

    logger.info(f"Preprocessor built — {len(num_cols)} numerical, {len(cat_cols)} categorical")
    return preprocessor


def get_feature_names(cfg: dict, include_city_fraud_rate: bool = True) -> List[str]:
    """Return ordered feature names matching preprocessor output columns."""
    return _get_all_numerical_cols(cfg, include_city_fraud_rate) + _get_all_categorical_cols(cfg)


def save_preprocessor(preprocessor, cfg: dict, path: Optional[str] = None) -> str:
    out = path or cfg["paths"]["preprocessing_pipeline"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(preprocessor, f)
    logger.info(f"Preprocessor saved → {out}")
    return out


def load_preprocessor(cfg: dict, path: Optional[str] = None):
    src = path or cfg["paths"]["preprocessing_pipeline"]
    with open(src, "rb") as f:
        preprocessor = pickle.load(f)
    logger.info(f"Preprocessor loaded from {src}")
    return preprocessor


# ------------------------------------------------------------------
# Column lists — what goes into the model
# ------------------------------------------------------------------
def _get_all_numerical_cols(cfg: dict, include_city_fraud_rate: bool = True) -> List[str]:
    """
    Raw numerical + all engineered numerical features.

    include_city_fraud_rate: set False when FeatureEngineer was fit WITHOUT y.
    city_fraud_rate column won't exist in df → preprocessor.fit() would KeyError.
    Default True because normal flow calls feat_eng.fit(X_train, y=y_train).
    """
    raw       = cfg["columns"]["numerical"]                            # price, bed, bath, acre_lot, house_size
    stateless = cfg["columns"]["engineered"]["stateless"]              # price_log, price_per_sqft, etc.
    fold_dep  = list(cfg["columns"]["engineered"]["fold_dependent"])   # copy — don't mutate config

    # city_fraud_rate only present when FeatureEngineer.fit(df, y=y) called with labels
    if not include_city_fraud_rate and "city_fraud_rate" in fold_dep:
        fold_dep.remove("city_fraud_rate")

    return raw + stateless + fold_dep


def _get_all_categorical_cols(cfg: dict) -> List[str]:
    """status + state (low cardinality). city/zip handled via fold-dependent encoding."""
    return cfg["columns"]["categorical"]                       # status, state