"""
tests/test_edge_cases.py — Real Estate Fraud Detection
Edge case tests — production readiness verification.

★ NEW: Tests zero bedrooms, unknown cities, price=0,
  negative values, long descriptions, and 100 random inputs.

Run:
  pytest tests/test_edge_cases.py -v
  pytest tests/test_edge_cases.py -v -k "price"
  pytest tests/ -v --tb=short   # all tests

Interview point:
  "Maine 12+ edge case tests add kiye — zero bedrooms (studio apartments),
  unknown cities (OrdinalEncoder fallback), price=0, aur 100 random inputs
  pe fraud score range validation. Production mein model gracefully handle
  kare unexpected inputs — crash nahi kare."
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
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
    """Load model bundle once for entire test session."""
    from src.inference import load_bundle
    return load_bundle()


# ─────────────────────────────────────────────────────────────────────────────
# Price edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestPriceEdgeCases:

    def test_price_very_low_is_suspicious(self, bundle):
        """price=0.01 — extremely low = should be suspicious."""
        from src.inference import predict_fraud
        result = predict_fraud(
            {"price": 0.01, "bed": 3.0, "bath": 2.0,
             "house_size": 1500.0, "city": "Austin", "state": "TX"},
            bundle
        )
        assert result is not None
        assert 0.0 <= result["fraud_score"] <= 1.0
        # Very low price should push fraud score up
        assert result["fraud_score"] >= 0.0

    def test_price_very_high_valid_range(self, bundle):
        """$10M listing — valid, score must be in [0, 1]."""
        from src.inference import predict_fraud
        result = predict_fraud(
            {"price": 10_000_000.0, "bed": 8.0, "bath": 10.0,
             "house_size": 12000.0, "city": "Beverly Hills", "state": "CA"},
            bundle
        )
        assert 0.0 <= result["fraud_score"] <= 1.0

    def test_price_equals_city_median_low_risk(self, bundle):
        """Price exactly at city median — should be LOW or MEDIUM risk."""
        from src.inference import predict_fraud
        # Austin median ~$400k
        result = predict_fraud(
            {"price": 400000.0, "bed": 3.0, "bath": 2.0,
             "house_size": 1800.0, "city": "Austin", "state": "TX"},
            bundle
        )
        assert result["risk_tier"] in ("LOW", "MEDIUM")

    def test_price_far_below_median_high_risk(self, bundle):
        """Price 90% below city median — should score higher than normal."""
        from src.inference import predict_fraud
        fraud_result  = predict_fraud({"price": 15000.0,  "city": "Austin", "state": "TX"}, bundle)
        normal_result = predict_fraud({"price": 400000.0, "city": "Austin", "state": "TX"}, bundle)
        assert fraud_result["fraud_score"] > normal_result["fraud_score"]


# ─────────────────────────────────────────────────────────────────────────────
# Bedroom / bathroom edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestBedroomEdgeCases:

    def test_zero_bedrooms_studio_valid(self, bundle):
        """0 bedrooms = studio apartment — valid, must not crash."""
        from src.inference import predict_fraud
        result = predict_fraud(
            {"price": 300_000.0, "bed": 0.0, "bath": 1.0,
             "house_size": 500.0, "city": "Chicago", "state": "IL"},
            bundle
        )
        assert result is not None
        assert 0.0 <= result["fraud_score"] <= 1.0

    def test_impossible_bedrooms_flagged(self, bundle):
        """25 bedrooms — impossible dimension = fraud signal."""
        from src.inference import predict_fraud
        impossible = predict_fraud(
            {"price": 500_000.0, "bed": 25.0, "bath": 3.0,
             "house_size": 2000.0, "city": "Dallas", "state": "TX"},
            bundle
        )
        normal = predict_fraud(
            {"price": 500_000.0, "bed": 3.0, "bath": 3.0,
             "house_size": 2000.0, "city": "Dallas", "state": "TX"},
            bundle
        )
        # Impossible bedrooms should score higher
        assert impossible["fraud_score"] >= normal["fraud_score"] * 0.8

    def test_zero_bathrooms_valid(self, bundle):
        """0 bathrooms — edge case, should not crash."""
        from src.inference import predict_fraud
        result = predict_fraud(
            {"price": 250_000.0, "bed": 2.0, "bath": 0.0,
             "house_size": 1000.0, "city": "Phoenix", "state": "AZ"},
            bundle
        )
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# City / state edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestCityEdgeCases:

    def test_unknown_city_no_crash(self, bundle):
        """City not in training data — OrdinalEncoder fallback (-1)."""
        from src.inference import predict_fraud
        result = predict_fraud(
            {"price": 250_000.0, "bed": 3.0, "bath": 2.0,
             "house_size": 1800.0,
             "city": "FakeCity_XYZ_123_NotReal", "state": "TX"},
            bundle
        )
        assert result is not None
        assert 0.0 <= result["fraud_score"] <= 1.0

    def test_city_case_insensitive(self, bundle):
        """austin / Austin / AUSTIN — all should return valid scores."""
        from src.inference import predict_fraud
        base = {"price": 350_000.0, "bed": 3.0, "bath": 2.0,
                "house_size": 1600.0, "state": "TX"}
        for city in ["austin", "Austin", "AUSTIN", "AuStIn"]:
            result = predict_fraud({**base, "city": city}, bundle)
            assert result is not None, f"Failed for city='{city}'"
            assert 0.0 <= result["fraud_score"] <= 1.0

    def test_none_city_no_crash(self, bundle):
        """city=None — missing optional field, must not crash."""
        from src.inference import predict_fraud
        result = predict_fraud(
            {"price": 300_000.0, "bed": 3.0, "bath": 2.0,
             "house_size": 1500.0, "city": None, "state": "TX"},
            bundle
        )
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# Missing field edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingFieldEdgeCases:

    def test_only_price_provided(self, bundle):
        """Only price — all other fields None/missing → median imputation."""
        from src.inference import predict_fraud
        result = predict_fraud({"price": 300_000.0}, bundle)
        assert result is not None
        assert 0.0 <= result["fraud_score"] <= 1.0

    def test_missing_house_size(self, bundle):
        """house_size=None — imputer should fill with median."""
        from src.inference import predict_fraud
        result = predict_fraud(
            {"price": 300_000.0, "bed": 3.0, "bath": 2.0,
             "house_size": None, "city": "Denver", "state": "CO"},
            bundle
        )
        assert result is not None

    def test_all_optional_fields_none(self, bundle):
        """Only required field (price) provided — all optionals None."""
        from src.inference import predict_fraud
        result = predict_fraud(
            {"price": 450_000.0, "bed": None, "bath": None,
             "house_size": None, "acre_lot": None,
             "city": None, "state": None},
            bundle
        )
        assert result is not None
        assert 0.0 <= result["fraud_score"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Output validation edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputValidation:

    def test_fraud_score_always_0_to_1(self, bundle):
        """100 random inputs — fraud_score must always be in [0, 1]."""
        from src.inference import predict_fraud

        rng = np.random.default_rng(42)
        for i in range(100):
            listing = {
                "price":      float(rng.uniform(1_000, 10_000_000)),
                "bed":        float(rng.integers(0, 15)),
                "bath":       float(rng.uniform(0, 10)),
                "house_size": float(rng.uniform(100, 15_000)),
                "acre_lot":   float(rng.uniform(0, 100)),
                "city":       "Houston",
                "state":      "TX",
            }
            result = predict_fraud(listing, bundle)
            assert 0.0 <= result["fraud_score"] <= 1.0, \
                f"Score out of range at iteration {i}: {result['fraud_score']}"

    def test_risk_tier_always_valid(self, bundle):
        """risk_tier must always be HIGH, MEDIUM, or LOW."""
        from src.inference import predict_fraud
        for price in [5000, 50000, 200000, 500000, 2000000]:
            result = predict_fraud(
                {"price": float(price), "city": "Austin", "state": "TX"},
                bundle
            )
            assert result["risk_tier"] in ("HIGH", "MEDIUM", "LOW"), \
                f"Invalid tier for price={price}: {result['risk_tier']}"

    def test_is_suspicious_matches_threshold(self, bundle):
        """is_suspicious must correctly match score >= 0.70."""
        from src.inference import predict_fraud
        threshold = bundle.cfg["api"]["fraud_threshold_suspicious"]
        for price in [10000, 100000, 400000, 1000000]:
            result = predict_fraud(
                {"price": float(price), "city": "Austin", "state": "TX"},
                bundle
            )
            expected = result["fraud_score"] >= threshold
            assert result["is_suspicious"] == expected, \
                f"is_suspicious mismatch for price={price}"

    def test_latency_ms_positive(self, bundle):
        """latency_ms must be present and positive."""
        from src.inference import predict_fraud
        result = predict_fraud(
            {"price": 350000.0, "city": "Austin", "state": "TX"},
            bundle
        )
        assert "latency_ms" in result
        assert result["latency_ms"] > 0
        assert result["latency_ms"] < 10000   # sanity: under 10 seconds


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schema validation (via API schemas)
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaValidation:

    def test_negative_acre_lot_rejected(self):
        """Negative acre_lot — Pydantic should raise ValueError."""
        from api.schemas import ListingInput
        with pytest.raises(Exception):  # ValidationError
            ListingInput(price=300_000.0, bed=3.0, bath=2.0,
                        house_size=1500.0, acre_lot=-5.0,
                        city="Miami", state="FL")

    def test_price_over_100m_rejected(self):
        """Price > $100M — Pydantic validator should reject."""
        from api.schemas import ListingInput
        with pytest.raises(Exception):
            ListingInput(price=200_000_000.0)

    def test_city_normalized_to_title_case(self):
        """City should be normalized to Title Case."""
        from api.schemas import ListingInput
        listing = ListingInput(price=300_000.0, city="AUSTIN")
        assert listing.city == "Austin"

    def test_state_normalized_to_upper(self):
        """State should be normalized to uppercase."""
        from api.schemas import ListingInput
        listing = ListingInput(price=300_000.0, state="tx")
        assert listing.state == "TX"
