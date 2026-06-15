"""
src/drift_monitor.py — Real Estate Fraud Detection
Monthly data drift detection — city_median_price baseline comparison.

★ NEW: Production hardening — model accuracy silently degrade hoti hai
when real estate prices shift. This script detects that.

Usage:
  python src/drift_monitor.py                    # run drift check
  python src/drift_monitor.py --threshold 0.15   # custom threshold

Interview point:
  "Maine data drift monitoring add kiya — har month city_median_price
  statistics retrain baseline se compare hoti hain. 20% drift pe
  automatic retrain alert generate hota hai."
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

# Project root setup
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# ─────────────────────────────────────────────────────────────────────────────
# DriftMonitor class
# ─────────────────────────────────────────────────────────────────────────────

class DriftMonitor:
    """
    Monthly data drift detection for Real Estate Fraud Detection.

    Compares current city_median_price statistics against baseline
    computed at training time. Flags cities with >threshold% change.

    Usage:
        monitor = DriftMonitor('configs/training_stats.yaml')
        drift   = monitor.detect_drift(new_monthly_data, threshold=0.20)
        report  = monitor.generate_report(drift)
        if report['retrain_recommended']:
            print("ACTION: Model retraining recommended!")
    """

    def __init__(self, baseline_stats_path: str = "configs/training_stats.yaml"):
        self.baseline_stats_path = Path(baseline_stats_path)

        if not self.baseline_stats_path.exists():
            raise FileNotFoundError(
                f"Baseline stats not found: {self.baseline_stats_path}\n"
                f"Run save_training_stats() first after Day 4 feature engineering."
            )

        with open(self.baseline_stats_path, "r") as f:
            self.baseline = yaml.safe_load(f)

        logger.info(
            f"DriftMonitor initialized — baseline: {self.baseline_stats_path} | "
            f"{len(self.baseline)} cities tracked"
        )

    # ── Stats computation ──────────────────────────────────────────────────────

    def compute_current_stats(self, df: pd.DataFrame) -> Dict:
        """
        Compute current city_median_price stats from new data.
        Same computation as training — ensures apples-to-apples comparison.
        """
        stats = (
            df.groupby("city")["price"]
            .agg(["median", "std", "count"])
            .to_dict("index")
        )
        logger.info(f"Current stats computed — {len(stats)} cities")
        return stats

    # ── Drift detection ────────────────────────────────────────────────────────

    def detect_drift(
        self,
        current_df: pd.DataFrame,
        threshold: float = 0.20,
    ) -> List[Dict]:
        """
        Detect cities where price median drifted beyond threshold.

        Args:
            current_df: new monthly listings DataFrame
            threshold:  fraction change (0.20 = 20%) to flag as drift

        Returns:
            List of drifted city dicts with baseline/current/pct_change
        """
        current_stats = self.compute_current_stats(current_df)
        drift_cities  = []

        for city, curr in current_stats.items():
            if city not in self.baseline:
                continue  # new city — skip

            if curr["count"] < 10:
                continue  # too few listings — unreliable

            base_median = self.baseline[city]["median"]
            curr_median = curr["median"]

            if base_median == 0:
                continue

            pct_change = abs(curr_median - base_median) / base_median

            if pct_change > threshold:
                drift_cities.append({
                    "city":             city,
                    "baseline_median":  round(base_median, 2),
                    "current_median":   round(curr_median, 2),
                    "pct_change":       round(pct_change * 100, 1),
                    "direction":        "UP" if curr_median > base_median else "DOWN",
                    "current_count":    curr["count"],
                })

        drift_cities.sort(key=lambda x: x["pct_change"], reverse=True)
        logger.info(f"Drift detected: {len(drift_cities)} cities above {threshold*100:.0f}% threshold")
        return drift_cities

    # ── Report generation ──────────────────────────────────────────────────────

    def generate_report(
        self,
        drift_cities: List[Dict],
        retrain_threshold: int = 5,
    ) -> Dict:
        """
        Generate drift report and save to reports/drift_report.json.

        retrain_recommended = True if more than retrain_threshold cities drifted.

        Args:
            drift_cities:       output from detect_drift()
            retrain_threshold:  how many drifted cities trigger retrain alert
        """
        retrain_recommended = len(drift_cities) > retrain_threshold

        report = {
            "run_date":             datetime.now().isoformat(),
            "drifted_cities":       len(drift_cities),
            "retrain_recommended":  retrain_recommended,
            "retrain_threshold":    retrain_threshold,
            "details":              drift_cities,
            "summary": {
                "max_drift_city":   drift_cities[0]["city"]        if drift_cities else None,
                "max_drift_pct":    drift_cities[0]["pct_change"]  if drift_cities else 0,
                "cities_up":        sum(1 for c in drift_cities if c["direction"] == "UP"),
                "cities_down":      sum(1 for c in drift_cities if c["direction"] == "DOWN"),
            }
        }

        # Save report
        report_path = Path("reports/drift_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Drift report saved → {report_path}")

        if retrain_recommended:
            logger.warning(
                f"⚠️  ACTION NEEDED: Model retraining recommended!\n"
                f"   {len(drift_cities)} cities drifted > threshold.\n"
                f"   Largest drift: {report['summary']['max_drift_city']} "
                f"({report['summary']['max_drift_pct']}%)"
            )
        else:
            logger.info(
                f"✅ No retraining needed — only {len(drift_cities)} cities drifted."
            )

        return report


# ─────────────────────────────────────────────────────────────────────────────
# Save training stats — call this after Day 4 feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def save_training_stats(
    df_train: pd.DataFrame,
    output_path: str = "configs/training_stats.yaml",
) -> str:
    """
    Save city_median_price stats from training data.
    Call ONCE after Day 4 splits are created — before any CV.

    These stats become the drift detection baseline.
    """
    stats = (
        df_train.groupby("city")["price"]
        .agg(["median", "std", "count"])
        .to_dict("index")
    )

    # Convert to plain Python types for YAML
    clean_stats = {
        city: {
            "median": float(v["median"]),
            "std":    float(v["std"]) if pd.notna(v["std"]) else 0.0,
            "count":  int(v["count"]),
        }
        for city, v in stats.items()
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(clean_stats, f, default_flow_style=False)

    logger.info(f"Training stats saved → {out_path} ({len(clean_stats)} cities)")
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point — for monthly cron job
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real Estate Fraud — Drift Monitor")
    parser.add_argument("--data",      default="data/processed/labeled.parquet",
                        help="Path to current month data")
    parser.add_argument("--baseline",  default="configs/training_stats.yaml",
                        help="Path to training baseline stats")
    parser.add_argument("--threshold", type=float, default=0.20,
                        help="Drift threshold (default: 0.20 = 20%%)")
    parser.add_argument("--retrain-threshold", type=int, default=5,
                        help="Cities drifted to trigger retrain alert (default: 5)")
    args = parser.parse_args()

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        sys.exit(1)

    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path) if data_path.suffix == ".parquet" \
         else pd.read_csv(data_path)
    logger.info(f"  Loaded {len(df):,} rows")

    # Run drift detection
    monitor = DriftMonitor(args.baseline)
    drift   = monitor.detect_drift(df, threshold=args.threshold)
    report  = monitor.generate_report(drift, retrain_threshold=args.retrain_threshold)

    # Print summary
    print(f"\n{'='*50}")
    print(f"  DRIFT MONITOR REPORT — {report['run_date'][:10]}")
    print(f"{'='*50}")
    print(f"  Drifted cities     : {report['drifted_cities']}")
    print(f"  Retrain recommended: {report['retrain_recommended']}")
    if drift:
        print(f"\n  Top 5 drifted cities:")
        for c in drift[:5]:
            arrow = "⬆" if c["direction"] == "UP" else "⬇"
            print(f"    {c['city']:<20} {arrow} {c['pct_change']}%  "
                  f"(${c['baseline_median']:,.0f} → ${c['current_median']:,.0f})")
    print(f"\n  Report saved → reports/drift_report.json")
    print(f"{'='*50}\n")

    if report["retrain_recommended"]:
        print("⚠️  ACTION NEEDED: Run model retraining pipeline!")
        sys.exit(1)   # non-zero exit for cron job alerting
    else:
        print("✅ No action needed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
