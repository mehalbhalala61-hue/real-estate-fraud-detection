"""
src/fraud_labeler.py — Real Estate Fraud Detection
Synthetic fraud label generation via rule-based domain logic.

All thresholds driven by configs/config.yaml → fraud_rules section.
No hardcoded numbers anywhere.

REAL DATA FIX (USA Realtor dataset):
  The dataset contains extreme corrupted values:
    price max  = 2,147,483,600  (int32 overflow — not real price)
    bath max   = 830            (data entry error)
    house_size = 1,040,400,400  (data corruption)
  These corrupt city/state medians used in R1/R2/R7, causing 50%+ fraud rate.
  Fix: clip price/house_size at 99th percentile BEFORE computing stats.
  Raw values still fire R4 (impossible_dims) and R5 (size_disconnect).

CRITICAL LEAKAGE RULE:
  fit()       → call ONLY on training data
  transform() → call on any split with same pre-fitted stats
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FraudLabeler:
    """
    Config-driven synthetic fraud label generator for real estate listings.

    Rules (all thresholds from config.yaml → fraud_rules):
      R1  price_low        price < multiplier × city_median
      R2  price_high       price > multiplier × city_median
      R3  price_per_sqft   price_per_sqft < city 10th-percentile
      R4  impossible_dims  bed>thresh OR bath>thresh OR acre_lot>thresh
      R5  size_disconnect  house_size>sqft_thresh AND price<price_thresh
      R6  duplicate        same (bed,bath,house_size,zip_code) diff price
      R7  state_anomaly    price < lo×state_median OR >hi×state_median
    """

    RULE_COLS = [
        "rule_price_low",
        "rule_price_high",
        "rule_price_per_sqft",
        "rule_impossible_dims",
        "rule_size_disconnect",
        "rule_duplicate",
        "rule_state_anomaly",
    ]

    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.rules  = cfg["fraud_rules"]
        self._fitted = False
        self.city_stats_      : Optional[pd.DataFrame] = None
        self.state_stats_     : Optional[pd.DataFrame] = None
        self.national_median_ : Optional[float]        = None
        self.city_sqft_p10_   : Optional[pd.Series]   = None
        # Store clip thresholds so transform() uses same bounds
        self._price_clip_val      : Optional[float]    = None
        self._house_size_clip_val : Optional[float]    = None

    # ------------------------------------------------------------------
    # fit — compute reference stats from training data ONLY
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "FraudLabeler":
        """
        Compute city/state price stats from df (training data only).
        Outlier clipping is applied before stat computation to prevent
        corrupted values (int32 overflow etc.) from distorting medians.
        """
        logger.info("FraudLabeler.fit() — computing city/state stats from training data")

        r           = self.rules
        min_city    = r["min_city_listings_for_stats"]
        min_state   = r["min_state_listings_for_stats"]
        price_pctile    = r.get("price_clip_percentile", 99)
        hs_pctile       = r.get("house_size_clip_percentile", 99)

        # ── Step 1: Compute clip thresholds from training data ──────────
        raw_price = df["price"].replace(0, np.nan).dropna()
        self._price_clip_val = float(np.percentile(raw_price, price_pctile))

        raw_hs = df["house_size"].replace(0, np.nan).dropna()
        self._house_size_clip_val = float(np.percentile(raw_hs, hs_pctile))

        logger.info(
            f"  Clip thresholds (p{price_pctile}): "
            f"price≤${self._price_clip_val:,.0f}  "
            f"house_size≤{self._house_size_clip_val:,.0f} sqft"
        )

        # ── Step 2: Clean copy for stat computation only ────────────────
        tmp = df.copy()
        tmp["price"]      = tmp["price"].clip(upper=self._price_clip_val)
        tmp["house_size"] = tmp["house_size"].clip(upper=self._house_size_clip_val)
        tmp["price"]      = tmp["price"].replace(0, np.nan)   # $0 = missing

        self.national_median_ = float(tmp["price"].median())

        # ── Step 3: City stats ──────────────────────────────────────────
        city_counts  = tmp.groupby("city")["price"].count()
        valid_cities = city_counts[city_counts >= min_city].index

        self.city_stats_ = (
            tmp[tmp["city"].isin(valid_cities)]
            .groupby("city")["price"]
            .agg(city_median="median", city_std="std")
            .reset_index()
        )

        # City price_per_sqft 10th percentile (R3)
        tmp["_ppsf"] = np.where(
            tmp["house_size"] > 0,
            tmp["price"] / tmp["house_size"],
            np.nan
        )
        self.city_sqft_p10_ = (
            tmp[tmp["city"].isin(valid_cities)]
            .groupby("city")["_ppsf"]
            .quantile(r["price_per_sqft_percentile"] / 100)
            .rename("city_ppsf_p10")
        )

        # ── Step 4: State stats ─────────────────────────────────────────
        state_counts  = tmp.groupby("state")["price"].count()
        valid_states  = state_counts[state_counts >= min_state].index
        self.state_stats_ = (
            tmp[tmp["state"].isin(valid_states)]
            .groupby("state")["price"]
            .agg(state_median="median")
            .reset_index()
        )

        self._fitted = True
        logger.info(
            f"  Cities: {len(self.city_stats_)} (≥{min_city} listings) | "
            f"States: {len(self.state_stats_)} | "
            f"National median: ${self.national_median_:,.0f}"
        )
        return self

    # ------------------------------------------------------------------
    # transform — apply rules using pre-computed stats
    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all 7 fraud rules using stats from fit().
        Returns df with rule_* columns, fraud_score, is_fraud.
        """
        if not self._fitted:
            raise RuntimeError(
                "FraudLabeler.fit() must be called before transform(). "
                "Call fit() on training data only."
            )

        logger.info(f"FraudLabeler.transform() — {len(df):,} rows")
        df = df.copy()

        # ── Merge city/state stats ──────────────────────────────────────
        df = df.merge(self.city_stats_,  on="city",  how="left")
        df = df.merge(self.state_stats_, on="state", how="left")
        df = df.merge(self.city_sqft_p10_.reset_index(), on="city", how="left")

        # Fill missing city/state with national fallback
        df["city_median"]   = df["city_median"].fillna(self.national_median_)
        df["city_std"]      = df["city_std"].fillna(df["city_median"] * 0.3)
        df["state_median"]  = df["state_median"].fillna(self.national_median_)
        national_ppsf_p10   = self.national_median_ / max(df["house_size"].median(), 1)
        df["city_ppsf_p10"] = df["city_ppsf_p10"].fillna(national_ppsf_p10)

        r = self.rules

        # Use stat-safe price (clipped) for comparison rules R1/R2/R3/R7
        # but keep original price for R4/R5 so corrupted values still fire
        safe_price = df["price"].clip(upper=self._price_clip_val)

        # R1 — price too low vs city median
        df["rule_price_low"] = (
            safe_price < r["price_low_multiplier"] * df["city_median"]
        ).astype(int)

        # R2 — price too high vs city median
        df["rule_price_high"] = (
            safe_price > r["price_high_multiplier"] * df["city_median"]
        ).astype(int)

        # R3 — price_per_sqft below city 10th percentile
        safe_hs = df["house_size"].clip(upper=self._house_size_clip_val)
        ppsf = np.where(safe_hs > 0, safe_price / safe_hs, np.nan)
        df["rule_price_per_sqft"] = (
            pd.Series(ppsf, index=df.index) < df["city_ppsf_p10"]
        ).fillna(False).astype(int)

        # R4 — impossible dimensions (use RAW values — catches corrupted data)
        df["rule_impossible_dims"] = (
            (df["bed"]      > r["impossible_bed_threshold"])  |
            (df["bath"]     > r["impossible_bath_threshold"]) |
            (df["acre_lot"] > r["impossible_acre_threshold"])
        ).astype(int)

        # R5 — price-size disconnect (use RAW values)
        df["rule_size_disconnect"] = (
            (df["house_size"] > r["price_size_disconnect_sqft"]) &
            (df["price"]      < r["price_size_disconnect_price"])
        ).astype(int)

        # R6 — duplicate listing
        df["rule_duplicate"] = self._flag_duplicates(df)

        # R7 — state price anomaly (use safe_price)
        df["rule_state_anomaly"] = (
            (safe_price < r["state_low_multiplier"]  * df["state_median"]) |
            (safe_price > r["state_high_multiplier"] * df["state_median"])
        ).astype(int)

        # ── Aggregate ───────────────────────────────────────────────────
        df["fraud_score"] = df[self.RULE_COLS].sum(axis=1)
        df["is_fraud"]    = (
            df["fraud_score"] >= r["min_fraud_score_threshold"]
        ).astype(int)

        # Clean up merge helper cols
        for col in ["city_median", "city_std", "state_median", "city_ppsf_p10"]:
            df = df.drop(columns=[col], errors="ignore")

        fraud_rate = df["is_fraud"].mean()
        logger.info(
            f"  Fraud rate: {fraud_rate*100:.2f}% — "
            f"{df['is_fraud'].sum():,} fraud / {len(df):,} total"
        )
        self._validate_fraud_rate(fraud_rate)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit + transform on same df. Use ONLY on training data."""
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # save / load stats
    # ------------------------------------------------------------------
    def save_stats(self, path: Optional[str] = None) -> str:
        if not self._fitted:
            raise RuntimeError("Call fit() before save_stats()")
        out = path or self.cfg["data"]["fraud_labeler_stats_path"]
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "national_median":     self.national_median_,
            "price_clip_val":      self._price_clip_val,
            "house_size_clip_val": self._house_size_clip_val,
            "city_stats":          self.city_stats_.to_dict(orient="records"),
            "state_stats":         self.state_stats_.to_dict(orient="records"),
            "city_sqft_p10":       self.city_sqft_p10_.to_dict(),
        }
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"FraudLabeler stats saved → {out}")
        return out

    def load_stats(self, path: Optional[str] = None) -> "FraudLabeler":
        src = path or self.cfg["data"]["fraud_labeler_stats_path"]
        with open(src) as f:
            s = json.load(f)
        self.national_median_     = s["national_median"]
        self._price_clip_val      = s.get("price_clip_val")
        self._house_size_clip_val = s.get("house_size_clip_val")
        self.city_stats_          = pd.DataFrame(s["city_stats"])
        self.state_stats_         = pd.DataFrame(s["state_stats"])
        self.city_sqft_p10_       = pd.Series(s["city_sqft_p10"]).rename("city_ppsf_p10")
        self.city_sqft_p10_.index.name = "city"
        self._fitted = True
        logger.info(f"FraudLabeler stats loaded from {src}")
        return self

    # ------------------------------------------------------------------
    # Label report
    # ------------------------------------------------------------------
    def label_report(self, df: pd.DataFrame) -> Dict:
        report = {
            "total_rows":  len(df),
            "fraud_count": int(df["is_fraud"].sum()),
            "fraud_rate":  float(df["is_fraud"].mean()),
            "rules": {},
        }
        for col in self.RULE_COLS:
            if col in df.columns:
                n = int(df[col].sum())
                report["rules"][col] = {"count": n,
                                        "pct": round(n / len(df) * 100, 3)}
        score_dist = df["fraud_score"].value_counts().sort_index()
        report["fraud_score_distribution"] = {
            int(k): int(v) for k, v in score_dist.items()
        }
        report["top_rules"] = sorted(
            {c: int(df[c].sum()) for c in self.RULE_COLS if c in df.columns}.items(),
            key=lambda x: x[1], reverse=True
        )
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _flag_duplicates(self, df):
        """
        Flag listings sharing (bed, bath, house_size, zip_code) with different prices.
        FIX: Skip zip_codes with < duplicate_min_zip_listings to avoid
        null/sparse zip false positives (was causing 23% false duplicate rate).
        """
        key_cols  = ["bed", "bath", "house_size", "zip_code"]
        available = [c for c in key_cols if c in df.columns]
        if len(available) < 3:
            return pd.Series(0, index=df.index)

        min_zip = self.rules.get("duplicate_min_zip_listings", 5)
        result  = pd.Series(0, index=df.index)

        tmp = df[available + ["price"]].copy()
        if "house_size" in tmp.columns:
            tmp["house_size"] = (tmp["house_size"] / 10).round() * 10

        # Only check duplicates within zip_codes that have enough listings
        if "zip_code" in tmp.columns:
            zip_counts = tmp["zip_code"].value_counts()
            valid_zips = zip_counts[zip_counts >= min_zip].index
            mask = tmp["zip_code"].isin(valid_zips)
        else:
            mask = pd.Series(True, index=tmp.index)

        if mask.sum() > 0:
            dup_flags = (
                tmp[mask].groupby(available)["price"]
                         .transform("nunique")
                         .gt(1)
                         .astype(int)
            )
            result[mask] = dup_flags.values

        logger.debug(f"  Duplicate rule: {result.sum():,} ({result.mean()*100:.2f}%)")
        return result

    def _validate_fraud_rate(self, rate: float) -> None:
        lo = self.rules["target_fraud_rate_min"]
        hi = self.rules["target_fraud_rate_max"]
        if rate < lo:
            logger.warning(
                f"⚠️  Fraud rate {rate:.3f} BELOW min {lo} — "
                "consider lowering price_low_multiplier or min_fraud_score_threshold"
            )
        elif rate > hi:
            logger.warning(
                f"⚠️  Fraud rate {rate:.3f} ABOVE max {hi} — "
                "consider raising min_fraud_score_threshold (currently "
                f"{self.rules['min_fraud_score_threshold']})"
            )
        else:
            logger.info(f"  ✅ Fraud rate {rate:.3f} within target [{lo}, {hi}]")


# ---------------------------------------------------------------------------
# Standalone pipeline runner
# ---------------------------------------------------------------------------
def run_labeling(cfg: dict) -> Tuple[pd.DataFrame, Dict]:
    """Full Day 2 labeling pipeline — load → fit → transform → save."""
    ingested = Path(cfg["data"]["processed_path"]) / "ingested.parquet"
    if not ingested.exists():
        raise FileNotFoundError(
            f"Not found: {ingested}\n"
            "Run ingestion first: python main.py --stage ingest"
        )

    df = pd.read_parquet(ingested)
    labeler    = FraudLabeler(cfg)
    df_labeled = labeler.fit_transform(df)

    out = Path(cfg["data"]["processed_path"]) / "labeled.parquet"
    df_labeled.to_parquet(out, index=False)
    labeler.save_stats()

    report = labeler.label_report(df_labeled)
    logger.info("=" * 55)
    logger.info("LABEL REPORT")
    logger.info("=" * 55)
    logger.info(f"  Total rows : {report['total_rows']:,}")
    logger.info(f"  Fraud count: {report['fraud_count']:,}")
    logger.info(f"  Fraud rate : {report['fraud_rate']*100:.2f}%")
    for rule, stats in report["rules"].items():
        logger.info(f"  {rule:<30}: {stats['count']:>8,}  ({stats['pct']:.2f}%)")
    logger.info("=" * 55)

    return df_labeled, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, yaml
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    df_labeled, report = run_labeling(cfg)
    print(f"\n✅ Done — fraud_rate={report['fraud_rate']*100:.2f}%  "
          f"count={report['fraud_count']:,}")