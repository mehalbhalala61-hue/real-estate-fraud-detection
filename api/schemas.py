"""
api/schemas.py — Real Estate Fraud Detection
Pydantic models for FastAPI request/response validation.

Input  : ListingInput  — raw listing fields
Output : PredictionResponse — fraud score + SHAP + tier
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Input schema
# ─────────────────────────────────────────────────────────────────────────────

class ListingInput(BaseModel):
    """
    Real estate listing — all fields optional except price.
    Missing fields handled by median imputation in preprocessing.
    """
    price:       float  = Field(...,  gt=0,  description="Listing price in USD")
    bed:         Optional[float] = Field(None, ge=0,  description="Number of bedrooms (0 = studio)")
    bath:        Optional[float] = Field(None, ge=0,  description="Number of bathrooms")
    house_size:  Optional[float] = Field(None, ge=0,  description="House size in sqft")
    acre_lot:    Optional[float] = Field(None, ge=0,  description="Lot size in acres")
    city:        Optional[str]   = Field(None, max_length=100)
    state:       Optional[str]   = Field(None, max_length=50)
    zip_code:    Optional[str]   = Field(None, max_length=20)
    status:      Optional[str]   = Field(None, max_length=50)
    prev_sold_date: Optional[str] = Field(None, description="Previous sale date YYYY-MM-DD")
    description: Optional[str]   = Field(None, description="Listing description text")

    @field_validator("price")
    @classmethod
    def price_reasonable(cls, v):
        if v > 100_000_000:
            raise ValueError("Price exceeds maximum allowed value ($100M)")
        return v

    @field_validator("acre_lot")
    @classmethod
    def acre_lot_positive(cls, v):
        if v is not None and v < 0:
            raise ValueError("acre_lot cannot be negative")
        return v

    @field_validator("city")
    @classmethod
    def normalize_city(cls, v):
        """Normalize city casing — 'austin', 'AUSTIN' → 'Austin'"""
        if v is not None:
            return v.strip().title()
        return v

    @field_validator("state")
    @classmethod
    def normalize_state(cls, v):
        if v is not None:
            return v.strip().upper()
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "price": 85000,
                "bed": 3,
                "bath": 2,
                "house_size": 1500,
                "acre_lot": 0.15,
                "city": "Austin",
                "state": "TX",
                "zip_code": "78701",
                "status": "for_sale",
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output schema
# ─────────────────────────────────────────────────────────────────────────────

class SHAPFeature(BaseModel):
    """One SHAP feature contribution."""
    feature: str   = Field(..., description="Feature name")
    impact:  float = Field(..., description="SHAP value — positive = fraud signal")
    value:   float = Field(..., description="Actual feature value for this listing")


class PredictionResponse(BaseModel):
    """
    Full fraud detection response.
    fraud_score is calibrated — 0.8 means ~80% probability of fraud.
    """
    fraud_score:    float          = Field(..., ge=0, le=1, description="Calibrated fraud probability [0, 1]")
    is_suspicious:  bool           = Field(..., description="True if score >= threshold (0.70)")
    risk_tier:      str            = Field(..., description="HIGH / MEDIUM / LOW")
    shap_top3:      List[SHAPFeature] = Field(..., description="Top 3 features driving this prediction")
    latency_ms:     float          = Field(..., description="Inference latency in milliseconds")
    model_version:  str            = Field(..., description="Model version")

    model_config = {
        "json_schema_extra": {
            "example": {
                "fraud_score": 0.82,
                "is_suspicious": True,
                "risk_tier": "HIGH",
                "shap_top3": [
                    {"feature": "price_vs_city_median", "impact": 1.21, "value": 0.31},
                    {"feature": "price_per_sqft",       "impact": 0.84, "value": 56.67},
                    {"feature": "city_fraud_rate",       "impact": 0.52, "value": 0.08},
                ],
                "latency_ms": 42.3,
                "model_version": "1.1.0",
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# History / Analytics schemas
# ─────────────────────────────────────────────────────────────────────────────

class PredictionHistoryItem(BaseModel):
    """One row in prediction history."""
    id:           int
    created_at:   str
    city:         Optional[str]
    state:        Optional[str]
    price:        Optional[float]
    fraud_score:  float
    risk_tier:    str
    latency_ms:   Optional[float]

    model_config = {"from_attributes": True}


class FraudStatsResponse(BaseModel):
    """Aggregate fraud statistics — Streamlit Analytics page."""
    total_predictions: int
    suspicious_count:  int
    fraud_rate:        float
    avg_fraud_score:   float
    tier_counts:       dict
    top_fraud_cities:  list


class HealthResponse(BaseModel):
    """Health check response."""
    status:        str
    db_connected:  bool
    model_loaded:  bool
    version:       str
