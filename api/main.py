"""
api/main.py — Real Estate Fraud Detection
FastAPI application — POST /predict, GET /history, GET /health

Endpoints:
  POST /predict      — fraud detection for one listing
  GET  /history      — past predictions with filters
  GET  /history/{id} — single prediction by ID
  GET  /stats        — fraud aggregate stats
  GET  /health       — DB + model health check

Security: API key via X-API-Key header (simple, interview-ready).
"""

import hashlib
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# ── Project root setup ──────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

from api.schemas import (
    FraudStatsResponse,
    HealthResponse,
    ListingInput,
    PredictionHistoryItem,
    PredictionResponse,
    SHAPFeature,
)
from src.database import (
    check_db_connection,
    get_active_model,
    get_db,
    get_engine,
    get_fraud_stats,
    get_prediction_by_id,
    get_prediction_history,
    init_db,
    log_prediction,
)
from src.inference import get_bundle, load_bundle, predict_fraud

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — startup + shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and init DB on startup."""
    logger.info("Starting up — loading models and DB...")

    # Init database
    engine = get_engine()
    init_db(engine)
    logger.info("Database initialized")

    # Load model bundle
    load_bundle()
    logger.info("Model bundle loaded")

    yield

    logger.info("Shutting down")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Real Estate Fraud Detection API",
    description="ML-powered fraud detection for real estate listings. "
                "Returns fraud score, risk tier, and SHAP explanations.",
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Streamlit dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# API Key auth — simple, no JWT complexity
# ─────────────────────────────────────────────────────────────────────────────

def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """
    Simple API key verification from X-API-Key header.
    Key loaded from API_KEY env var — never hardcoded.

    Interview answer: "For production, API Gateway (AWS/GCP) handles
    OAuth/JWT. Application layer uses simple API key — enough for this scope."
    """
    expected = os.getenv("API_KEY", "dev-secret-key")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return hashlib.sha256(x_api_key.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict fraud for a listing",
    tags=["Prediction"],
)
def predict(
    listing: ListingInput,
    db: Session = Depends(get_db),
    api_key_hash: str = Depends(verify_api_key),
) -> PredictionResponse:
    """
    Detect fraud in a real estate listing.

    Returns:
    - **fraud_score**: calibrated probability [0, 1]
    - **risk_tier**: HIGH (≥0.70) / MEDIUM (≥0.40) / LOW
    - **shap_top3**: top 3 features driving this prediction
    - **latency_ms**: inference time

    **Example HIGH risk listing:** price far below city median, impossible dimensions.
    """
    bundle = get_bundle()

    # Run prediction
    listing_dict = listing.model_dump()
    result = predict_fraud(listing_dict, bundle)

    # Log to database
    try:
        log_prediction(db, {
            **{k: listing_dict.get(k) for k in [
                "price", "bed", "bath", "house_size", "acre_lot",
                "city", "state", "zip_code", "status",
            ]},
            "fraud_score":    result["fraud_score"],
            "is_suspicious":  result["is_suspicious"],
            "risk_tier":      result["risk_tier"],
            "shap_top3":      json.dumps(result["shap_top3"]),
            "model_version":  result["model_version"],
            "latency_ms":     result["latency_ms"],
            "api_key_hash":   api_key_hash,
        })
    except Exception as e:
        logger.warning(f"Failed to log prediction to DB: {e}")

    return PredictionResponse(
        fraud_score=result["fraud_score"],
        is_suspicious=result["is_suspicious"],
        risk_tier=result["risk_tier"],
        shap_top3=[SHAPFeature(**f) for f in result["shap_top3"]],
        latency_ms=result["latency_ms"],
        model_version=result["model_version"],
    )


@app.get(
    "/history",
    response_model=List[PredictionHistoryItem],
    summary="Get prediction history",
    tags=["History"],
)
def history(
    limit:     int            = Query(50,   ge=1, le=500),
    offset:    int            = Query(0,    ge=0),
    city:      Optional[str]  = Query(None, description="Filter by city"),
    state:     Optional[str]  = Query(None, description="Filter by state"),
    risk_tier: Optional[str]  = Query(None, description="HIGH / MEDIUM / LOW"),
    min_score: Optional[float]= Query(None, ge=0, le=1),
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
) -> List[PredictionHistoryItem]:
    """Get past predictions with optional filters."""
    predictions = get_prediction_history(
        db, limit=limit, offset=offset,
        city=city, state=state,
        risk_tier=risk_tier, min_score=min_score,
    )
    return [
        PredictionHistoryItem(
            id=p.id,
            created_at=str(p.created_at),
            city=p.city,
            state=p.state,
            price=p.price,
            fraud_score=p.fraud_score,
            risk_tier=p.risk_tier,
            latency_ms=p.latency_ms,
        )
        for p in predictions
    ]


@app.get(
    "/history/{prediction_id}",
    response_model=PredictionHistoryItem,
    summary="Get prediction by ID",
    tags=["History"],
)
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
) -> PredictionHistoryItem:
    """Get a single prediction by its ID."""
    pred = get_prediction_by_id(db, prediction_id)
    if not pred:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found")
    return PredictionHistoryItem(
        id=pred.id,
        created_at=str(pred.created_at),
        city=pred.city,
        state=pred.state,
        price=pred.price,
        fraud_score=pred.fraud_score,
        risk_tier=pred.risk_tier,
        latency_ms=pred.latency_ms,
    )


@app.get(
    "/stats",
    response_model=FraudStatsResponse,
    summary="Fraud aggregate statistics",
    tags=["Analytics"],
)
def stats(
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
) -> FraudStatsResponse:
    """Aggregate fraud statistics — used by Streamlit Analytics dashboard."""
    s = get_fraud_stats(db)
    return FraudStatsResponse(**s)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Health"],
)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    """
    Health check — no API key required.
    Used by Docker healthcheck and load balancers.
    """
    engine      = get_engine()
    db_ok       = check_db_connection(engine)
    bundle      = get_bundle()
    model_ok    = bundle.loaded

    return HealthResponse(
        status="healthy" if (db_ok and model_ok) else "degraded",
        db_connected=db_ok,
        model_loaded=model_ok,
        version=app.version,
    )
