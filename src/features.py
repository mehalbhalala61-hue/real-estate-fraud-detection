"""
src/features.py — Real Estate Fraud Detection
Feature engineering — stateless + fold-dependent features.

Stateless : computed from the row itself — safe on any split
Fold-dependent : computed from training data only — must use fit/transform pattern

LEAKAGE RULE: fit() on train fold only, transform() on val/test
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Two-step feature engineering:
      1. add_stateless_features() — no training dependency, safe anywhere
      2. fit() + transform()      — fold-dependent features (city stats, zip freq)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._fitted = False
        self.city_median_   : Optional[pd.Series] = None
        self.city_fraud_rate_: Optional[pd.Series] = None
        self.zip_freq_      : Optional[pd.Series] = None
        self.state_median_  : Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Stateless features — call on any split, any time
    # ------------------------------------------------------------------
    @staticmethod
    def add_stateless_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
        """
        Compute features that need only the row itself.
        Safe to call before train/test split.
        """
        df = df.copy()
        fill = cfg.get("feature_defaults", {}).get("bath_per_bed_zero_bed_fill", 0.0)

        # FIX: pd.to_numeric(..., errors=coerce) — None/NoneType ko NaN mein convert karo
        # np.log1p directly NoneType handle nahi kar sakta — inference mein crash hota tha
        price      = pd.to_numeric(df["price"],      errors="coerce").fillna(0).clip(lower=0)
        house_size = pd.to_numeric(df["house_size"], errors="coerce").fillna(0).clip(lower=0)
        acre_lot   = pd.to_numeric(df["acre_lot"],   errors="coerce").fillna(0).clip(lower=0)
        bed        = pd.to_numeric(df["bed"],         errors="coerce").fillna(0)
        bath       = pd.to_numeric(df["bath"],        errors="coerce").fillna(0)

        # Price features
        df["price_log"]       = np.log1p(price)

        # Size features
        df["house_size_log"]  = np.log1p(house_size)
        df["acre_lot_log"]    = np.log1p(acre_lot)

        # Ratio features
        df["price_per_sqft"]  = np.where(
            house_size > 0,
            price / house_size,
            np.nan,
        )
        df["bath_per_bed"]    = np.where(
            bed > 0,
            bath / bed,
            fill,
        )

        # Binary flag
        df["is_large_property"] = (house_size > 3000).astype(int)

        # Temporal feature — days since last sale
        if "prev_sold_date" in df.columns:
            df = _add_temporal_features(df, cfg)

        n_new = 6 + (3 if "prev_sold_date" in df.columns else 0)
        logger.info(f"Stateless features added — {n_new} new columns")
        return df

    # ------------------------------------------------------------------
    # Fold-dependent features — fit on train only
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """
        Compute city/state/zip aggregation stats from training data.
        y required for city_fraud_rate (target encoding).
        """
        logger.info("FeatureEngineer.fit() — computing fold-dependent stats")

        min_city  = self.cfg["fraud_rules"].get("min_city_listings_for_stats", 30)
        min_state = self.cfg["fraud_rules"].get("min_state_listings_for_stats", 10)

        # City median price
        city_counts = df.groupby("city")["price"].count()
        valid_cities = city_counts[city_counts >= min_city].index
        self.city_median_ = (
            df[df["city"].isin(valid_cities)]
            .groupby("city")["price"]
            .median()
            .rename("city_median_price")
        )

        # State median price
        state_counts = df.groupby("state")["price"].count()
        valid_states = state_counts[state_counts >= min_state].index
        self.state_median_ = (
            df[df["state"].isin(valid_states)]
            .groupby("state")["price"]
            .median()
            .rename("state_median_price")
        )

        # Zip frequency encoding
        self.zip_freq_ = (
            df["zip_code"].value_counts().rename("zip_listing_count")
        )

        # City fraud rate — target encoding (OOF safe when called inside CV)
        if y is not None:
            tmp = df.copy()
            tmp["_target"] = y.values if hasattr(y, "values") else y
            self.city_fraud_rate_ = (
                tmp[tmp["city"].isin(valid_cities)]
                .groupby("city")["_target"]
                .mean()
                .rename("city_fraud_rate")
            )

        self._fitted = True
        logger.info(
            f"  Cities: {len(self.city_median_)} | "
            f"States: {len(self.state_median_)} | "
            f"Zip codes: {len(self.zip_freq_)}"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add fold-dependent features using stats from fit().
        Unknown cities/zips get national/global fallback.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() on training data before transform()")

        df = df.copy()

        # National fallbacks
        national_price_median = float(self.city_median_.median())

        # city_median_price
        df["city_median_price"] = (
            df["city"].map(self.city_median_).fillna(national_price_median)
        )

        # price_vs_city_median — key fraud signal
        df["price_vs_city_median"] = np.where(
            df["city_median_price"] > 0,
            df["price"] / df["city_median_price"],
            1.0,
        )

        # state_median_price
        state_national = float(self.state_median_.median())
        df["state_median_price"] = (
            df["state"].map(self.state_median_).fillna(state_national)
        )

        # zip_listing_count — frequency encoding
        df["zip_listing_count"] = (
            df["zip_code"].map(self.zip_freq_).fillna(0).astype(int)
        )

        # city_fraud_rate — only if fit with y
        if self.city_fraud_rate_ is not None:
            global_fraud_rate = float(self.city_fraud_rate_.mean())
            df["city_fraud_rate"] = (
                df["city"].map(self.city_fraud_rate_).fillna(global_fraud_rate)
            )

        logger.info(f"Fold-dependent features added — {len(df.columns)} total cols")
        return df

    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)


# ------------------------------------------------------------------
# Temporal helper — kept private, called from add_stateless_features
# ------------------------------------------------------------------
def _add_temporal_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fill_days = cfg.get("temporal_features", {}).get(
        "prev_sold_date", {}
    ).get("fill_missing_days", 9999)

    parsed = pd.to_datetime(df["prev_sold_date"], errors="coerce")
    reference_date = pd.Timestamp("2024-01-01")

    df["days_since_last_sale"] = (
        (reference_date - parsed).dt.days.clip(lower=0).fillna(fill_days).astype(int)
    )
    df["sold_year"]  = parsed.dt.year.fillna(0).astype(int)
    df["sold_month"] = parsed.dt.month.fillna(0).astype(int)
    return df


# ------------------------------------------------------------------
# Convenience — full list of feature names after engineering
# ------------------------------------------------------------------
STATELESS_FEATURES = [
    "price_log",
    "house_size_log",
    "acre_lot_log",
    "price_per_sqft",
    "bath_per_bed",
    "is_large_property",
    "days_since_last_sale",
    "sold_year",
    "sold_month",
]

FOLD_DEPENDENT_FEATURES = [
    "city_median_price",
    "price_vs_city_median",
    "state_median_price",
    "zip_listing_count",
    "city_fraud_rate",      # only when fit with y
]