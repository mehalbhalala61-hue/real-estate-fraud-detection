"""
src/database.py — Real Estate Fraud Detection
PostgreSQL + SQLAlchemy — predictions store & query.

Tables:
  prediction_logs  — every API prediction logged here
  model_registry   — model versions + metrics

SECURITY: DATABASE_URL from environment variable only.
Never hardcode credentials in code or config.yaml.

Usage:
  from src.database import get_db, PredictionLog, init_db
"""

import logging
import os
from datetime import datetime
from typing import Generator, List, Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer,
    String, Text, create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Database URL — from environment only
# ─────────────────────────────────────────────────────────────────────────────

def get_database_url() -> str:
    """
    Read DATABASE_URL from environment.
    Never hardcode — security issue.

    Local dev: set in .env file
      DATABASE_URL=postgresql://fraud_user:fraud_pass@localhost:5432/fraud_db

    Docker: set in docker-compose.yml environment section.
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        # Fallback for local development only
        url = "postgresql://fraud_user:fraud_pass@localhost:5432/fraud_db"
        logger.warning(
            "DATABASE_URL not set — using default local URL. "
            "Set DATABASE_URL env var for production."
        )
    return url


# ─────────────────────────────────────────────────────────────────────────────
# SQLAlchemy setup
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


def create_db_engine(database_url: Optional[str] = None, echo: bool = False):
    """
    Create SQLAlchemy engine with connection pooling.
    echo=True logs all SQL — useful for debugging, False for production.
    """
    url = database_url or get_database_url()
    engine = create_engine(
        url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,    # verify connection before using
        echo=echo,
    )
    logger.info(f"Database engine created — {url.split('@')[-1]}")  # log host only, not password
    return engine


def create_session_factory(engine):
    """Create SQLAlchemy session factory."""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ─────────────────────────────────────────────────────────────────────────────
# ORM Models
# ─────────────────────────────────────────────────────────────────────────────

class PredictionLog(Base):
    """
    One row per API prediction.
    Stores input features, fraud score, SHAP top-3, latency.
    """
    __tablename__ = "prediction_logs"

    id               = Column(Integer, primary_key=True, index=True)
    created_at       = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Input features
    price            = Column(Float,   nullable=True)
    bed              = Column(Float,   nullable=True)
    bath             = Column(Float,   nullable=True)
    house_size       = Column(Float,   nullable=True)
    acre_lot         = Column(Float,   nullable=True)
    city             = Column(String(100), nullable=True)
    state            = Column(String(50),  nullable=True)
    zip_code         = Column(String(20),  nullable=True)
    status           = Column(String(50),  nullable=True)

    # Prediction output
    fraud_score      = Column(Float,   nullable=False)
    is_suspicious    = Column(Boolean, nullable=False)
    risk_tier        = Column(String(20), nullable=False)  # HIGH / MEDIUM / LOW

    # SHAP top-3 features (stored as JSON string)
    shap_top3        = Column(Text, nullable=True)

    # Metadata
    model_version    = Column(String(50), nullable=True)
    latency_ms       = Column(Float,      nullable=True)
    api_key_hash     = Column(String(64), nullable=True)   # hashed — never store raw key

    def __repr__(self):
        return (
            f"<PredictionLog id={self.id} "
            f"score={self.fraud_score:.3f} "
            f"tier={self.risk_tier} "
            f"city={self.city}>"
        )


class ModelRegistry(Base):
    """
    Track model versions — metrics, paths, active status.
    """
    __tablename__ = "model_registry"

    id               = Column(Integer, primary_key=True, index=True)
    registered_at    = Column(DateTime, default=datetime.utcnow, nullable=False)

    model_name       = Column(String(100), nullable=False)
    version          = Column(String(50),  nullable=False)
    stage            = Column(String(20),  nullable=False)   # Production / Staging / Archived

    # Metrics
    pr_auc           = Column(Float, nullable=True)
    recall_at_95p    = Column(Float, nullable=True)
    threshold        = Column(Float, nullable=True)

    # Paths
    model_path       = Column(String(500), nullable=True)
    mlflow_run_id    = Column(String(100), nullable=True)

    # Active flag — only one model should be Production at a time
    is_active        = Column(Boolean, default=False, nullable=False)
    notes            = Column(Text, nullable=True)

    def __repr__(self):
        return (
            f"<ModelRegistry {self.model_name} v{self.version} "
            f"stage={self.stage} pr_auc={self.pr_auc}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Database initialization
# ─────────────────────────────────────────────────────────────────────────────

def init_db(engine) -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified: prediction_logs, model_registry")


def drop_all_tables(engine) -> None:
    """Drop all tables — use only in testing/reset."""
    Base.metadata.drop_all(bind=engine)
    logger.warning("All tables dropped!")


# ─────────────────────────────────────────────────────────────────────────────
# Dependency injection — FastAPI uses this
# ─────────────────────────────────────────────────────────────────────────────

# Module-level engine + session factory
# Initialized lazily on first use
_engine        = None
_SessionLocal  = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_db_engine()
    return _engine


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency — yields DB session, auto-closes on request end.

    Usage in FastAPI:
        @app.post('/predict')
        def predict(db: Session = Depends(get_db)):
            ...
    """
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = create_session_factory(get_engine())

    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# CRUD operations
# ─────────────────────────────────────────────────────────────────────────────

def log_prediction(
    db: Session,
    prediction_data: dict,
) -> PredictionLog:
    """
    Insert one prediction into prediction_logs.

    prediction_data keys:
      price, bed, bath, house_size, acre_lot, city, state, zip_code, status,
      fraud_score, is_suspicious, risk_tier, shap_top3 (JSON str),
      model_version, latency_ms, api_key_hash
    """
    log = PredictionLog(**prediction_data)
    db.add(log)
    db.commit()
    db.refresh(log)
    logger.debug(f"Prediction logged — id={log.id} score={log.fraud_score:.3f}")
    return log


def get_prediction_history(
    db: Session,
    limit: int = 100,
    offset: int = 0,
    city: Optional[str] = None,
    state: Optional[str] = None,
    risk_tier: Optional[str] = None,
    min_score: Optional[float] = None,
) -> List[PredictionLog]:
    """
    Query prediction history with optional filters.
    Used by GET /history endpoint and Streamlit History page.
    """
    query = db.query(PredictionLog)

    if city:
        query = query.filter(PredictionLog.city.ilike(f"%{city}%"))
    if state:
        query = query.filter(PredictionLog.state == state)
    if risk_tier:
        query = query.filter(PredictionLog.risk_tier == risk_tier)
    if min_score is not None:
        query = query.filter(PredictionLog.fraud_score >= min_score)

    return (
        query
        .order_by(PredictionLog.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_prediction_by_id(db: Session, prediction_id: int) -> Optional[PredictionLog]:
    """Get single prediction by ID."""
    return db.query(PredictionLog).filter(PredictionLog.id == prediction_id).first()


def get_fraud_stats(db: Session) -> dict:
    """
    Aggregate fraud statistics — used in Streamlit Analytics page.
    Returns counts by risk tier, fraud rate over time, top fraud cities.
    """
    from sqlalchemy import func

    total = db.query(func.count(PredictionLog.id)).scalar() or 0
    suspicious = db.query(func.count(PredictionLog.id)).filter(
        PredictionLog.is_suspicious == True
    ).scalar() or 0

    # Count by risk tier
    tier_counts = dict(
        db.query(PredictionLog.risk_tier, func.count(PredictionLog.id))
        .group_by(PredictionLog.risk_tier)
        .all()
    )

    # Top 10 cities by fraud count
    top_cities = (
        db.query(PredictionLog.city, func.count(PredictionLog.id).label("count"))
        .filter(PredictionLog.is_suspicious == True)
        .filter(PredictionLog.city.isnot(None))
        .group_by(PredictionLog.city)
        .order_by(func.count(PredictionLog.id).desc())
        .limit(10)
        .all()
    )

    # Average fraud score
    avg_score = db.query(func.avg(PredictionLog.fraud_score)).scalar() or 0.0

    return {
        "total_predictions": total,
        "suspicious_count":  suspicious,
        "fraud_rate":        round(suspicious / total, 4) if total > 0 else 0.0,
        "avg_fraud_score":   round(float(avg_score), 4),
        "tier_counts":       tier_counts,
        "top_fraud_cities":  [{"city": c, "count": n} for c, n in top_cities],
    }


def register_model(
    db: Session,
    model_name: str,
    version: str,
    stage: str,
    pr_auc: float,
    recall_at_95p: float,
    threshold: float,
    model_path: str,
    mlflow_run_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> ModelRegistry:
    """
    Register a model version in model_registry.
    Marks previous Production models as Archived.
    """
    # Archive existing production model
    if stage == "Production":
        db.query(ModelRegistry).filter(
            ModelRegistry.stage == "Production",
            ModelRegistry.is_active == True,
        ).update({"stage": "Archived", "is_active": False})

    model = ModelRegistry(
        model_name=model_name,
        version=version,
        stage=stage,
        pr_auc=pr_auc,
        recall_at_95p=recall_at_95p,
        threshold=threshold,
        model_path=model_path,
        mlflow_run_id=mlflow_run_id,
        is_active=(stage == "Production"),
        notes=notes,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    logger.info(f"Model registered — {model_name} v{version} ({stage})")
    return model


def get_active_model(db: Session) -> Optional[ModelRegistry]:
    """Get currently active (Production) model from registry."""
    return (
        db.query(ModelRegistry)
        .filter(ModelRegistry.is_active == True)
        .first()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

def check_db_connection(engine) -> bool:
    """Ping database — used in FastAPI /health endpoint."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False
