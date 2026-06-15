"""
tests/test_inference.py — Real Estate Fraud Detection
End-to-end inference tests + latency assertion.

Run:
  pytest tests/test_inference.py -v
  pytest tests/test_inference.py -v -k "latency"
"""

import os
import sys
import time
from pathlib import Path

import pytest

# Project root setup
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def bundle():
    """Load model bundle once for all tests."""
    from src.inference import load_bundle
    return load_bundle()


@pytest.fixture
def normal_listing():
    return {
        "price": 450000.0, "bed": 4.0, "bath": 3.0,
        "house_size": 2200.0, "acre_lot": 0.25,
        "city": "Denver", "state": "CO",
        "zip_code": "80201", "status": "for_sale",
    }


@pytest.fixture
def fraud_listing():
    """Listing designed to trigger fraud rules."""
    return {
        "price": 25000.0,   # far below city median
        "bed": 3.0, "bath": 2.0,
        "house_size": 1500.0, "acre_lot": 0.15,
        "city": "Austin", "state": "TX",
        "zip_code": "78701", "status": "for_sale",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Basic inference tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInference:

    def test_predict_returns_valid_score(self, bundle, normal_listing):
        """fraud_score must be in [0, 1]."""
        from src.inference import predict_fraud
        result = predict_fraud(normal_listing, bundle)
        assert 0.0 <= result["fraud_score"] <= 1.0

    def test_predict_returns_risk_tier(self, bundle, normal_listing):
        """risk_tier must be HIGH, MEDIUM, or LOW."""
        from src.inference import predict_fraud
        result = predict_fraud(normal_listing, bundle)
        assert result["risk_tier"] in ("HIGH", "MEDIUM", "LOW")

    def test_predict_returns_shap_top3(self, bundle, normal_listing):
        """shap_top3 must have exactly 3 features."""
        from src.inference import predict_fraud
        result = predict_fraud(normal_listing, bundle)
        assert len(result["shap_top3"]) == 3
        for feat in result["shap_top3"]:
            assert "feature" in feat
            assert "impact" in feat
            assert "value" in feat

    def test_fraud_listing_higher_score(self, bundle, normal_listing, fraud_listing):
        """Fraud listing should score higher than normal listing."""
        from src.inference import predict_fraud
        normal_result = predict_fraud(normal_listing, bundle)
        fraud_result  = predict_fraud(fraud_listing, bundle)
        assert fraud_result["fraud_score"] > normal_result["fraud_score"]

    def test_is_suspicious_matches_threshold(self, bundle, normal_listing):
        """is_suspicious must match score >= threshold."""
        from src.inference import predict_fraud
        result    = predict_fraud(normal_listing, bundle)
        threshold = bundle.cfg["api"]["fraud_threshold_suspicious"]
        expected  = result["fraud_score"] >= threshold
        assert result["is_suspicious"] == expected

    def test_latency_ms_present(self, bundle, normal_listing):
        """latency_ms must be present and positive."""
        from src.inference import predict_fraud
        result = predict_fraud(normal_listing, bundle)
        assert "latency_ms" in result
        assert result["latency_ms"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Edge case tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_missing_optional_fields(self, bundle):
        """Only price required — rest should be handled by imputation."""
        from src.inference import predict_fraud
        minimal = {"price": 300000.0}
        result  = predict_fraud(minimal, bundle)
        assert result is not None
        assert 0.0 <= result["fraud_score"] <= 1.0

    def test_unknown_city(self, bundle):
        """Unknown city — OrdinalEncoder fallback (-1). Should not crash."""
        from src.inference import predict_fraud
        listing = {
            "price": 250000.0, "bed": 3.0, "bath": 2.0,
            "house_size": 1800.0, "city": "FakeCityXYZ123", "state": "TX",
        }
        result = predict_fraud(listing, bundle)
        assert 0.0 <= result["fraud_score"] <= 1.0

    def test_zero_bedrooms_studio(self, bundle):
        """0 bedrooms = studio apartment — valid, should not crash."""
        from src.inference import predict_fraud
        listing = {
            "price": 300000.0, "bed": 0.0, "bath": 1.0,
            "house_size": 500.0, "city": "Chicago", "state": "IL",
        }
        result = predict_fraud(listing, bundle)
        assert result is not None

    def test_very_high_price(self, bundle):
        """$10M listing — valid, should return score in [0, 1]."""
        from src.inference import predict_fraud
        listing = {
            "price": 10_000_000.0, "bed": 8.0, "bath": 10.0,
            "house_size": 12000.0, "city": "Beverly Hills", "state": "CA",
        }
        result = predict_fraud(listing, bundle)
        assert 0.0 <= result["fraud_score"] <= 1.0

    def test_city_case_insensitive(self, bundle):
        """Austin / AUSTIN / austin — all should work."""
        from src.inference import predict_fraud
        base = {"price": 350000.0, "bed": 3.0, "bath": 2.0,
                "house_size": 1600.0, "state": "TX"}
        for city in ["Austin", "austin", "AUSTIN"]:
            result = predict_fraud({**base, "city": city}, bundle)
            assert result is not None

    def test_100_random_scores_in_range(self, bundle):
        """100 random inputs — all fraud_scores must be in [0, 1]."""
        import numpy as np
        from src.inference import predict_fraud

        rng = np.random.default_rng(42)
        for _ in range(100):
            listing = {
                "price":      float(rng.uniform(10_000, 5_000_000)),
                "bed":        float(rng.integers(0, 10)),
                "bath":       float(rng.uniform(0, 8)),
                "house_size": float(rng.uniform(200, 10_000)),
                "city":       "Houston",
                "state":      "TX",
            }
            result = predict_fraud(listing, bundle)
            assert 0.0 <= result["fraud_score"] <= 1.0, \
                f"Score out of range: {result['fraud_score']}"


# ─────────────────────────────────────────────────────────────────────────────
# Latency tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLatency:

    def test_p95_latency_under_500ms(self, bundle):
        """
        p95 inference latency must be < 500ms.
        Plan requirement: 'p95 inference < 500ms — Streamlit interactive hona chahiye'
        """
        from src.inference import predict_fraud

        listing = {
            "price": 350000.0, "bed": 3.0, "bath": 2.0,
            "house_size": 1800.0, "city": "Austin", "state": "TX",
        }

        # Warmup — first call loads SHAP explainer
        predict_fraud(listing, bundle)

        # Measure 20 calls
        latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            predict_fraud(listing, bundle)
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        p50 = latencies[len(latencies) // 2]

        print(f"\nLatency results (20 calls):")
        print(f"  p50: {p50:.1f}ms")
        print(f"  p95: {p95:.1f}ms")
        print(f"  max: {max(latencies):.1f}ms")

        assert p95 < 500, f"p95 latency {p95:.1f}ms exceeds 500ms limit"

    def test_single_call_under_2s(self, bundle):
        """Single call (including SHAP) must complete under 2 seconds."""
        from src.inference import predict_fraud

        listing = {
            "price": 200000.0, "bed": 3.0, "bath": 2.0,
            "house_size": 1500.0, "city": "Portland", "state": "OR",
        }

        t0     = time.perf_counter()
        result = predict_fraud(listing, bundle)
        elapsed = (time.perf_counter() - t0) * 1000

        assert elapsed < 2000, f"Single call took {elapsed:.0f}ms > 2000ms"
        assert result["latency_ms"] < 2000
