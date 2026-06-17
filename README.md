---
title: Real Estate Fraud Detection
emoji: 🏠
colorFrom: red
colorTo: purple
sdk: docker
dockerfile: Dockerfile.streamlit
pinned: false
---
# 🏠 Real Estate Fraud Detection

**End-to-End ML Project** | Tabular + Text | LightGBM + Stacking | SHAP | MLflow | FastAPI | Docker | Streamlit

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-orange)](https://lightgbm.readthedocs.io)

---

## 📌 Project Overview

Real estate mein fraud bahut common hai — fake listings, price manipulation, duplicate properties. Labeled fraud data publicly available nahi hoti, isliye domain logic se synthetic labels banaye aur ek complete production-ready ML system develop kiya.

**Target Role:** ML Engineer / Data Scientist — 6 to 10 LPA

---

## 🏗️ Architecture

```
USA Real Estate Dataset (2.1M listings)
          ↓
  Synthetic Fraud Labels (rule-based domain logic)
          ↓
  Feature Engineering (stateless + fold-dependent)
          ↓
  LR → LightGBM → Optuna Tuning → Stacking → Platt Calibration
          ↓
  SHAP Explainability
          ↓
  FastAPI (POST /predict) + PostgreSQL
          ↓
  Streamlit Dashboard
          ↓
  Docker + Railway Deployment
```

---

## 📊 Results

| Model | PR-AUC | Recall@95P |
|-------|--------|------------|
| Logistic Regression (baseline) | 0.6031 | 0.0000 |
| LightGBM (baseline) | 0.7807 | 0.0856 |
| LightGBM (tuned) | 0.7779 | 0.1105 |
| **Stacked + Calibrated** | **0.7694** | **0.1105** |

**Why PR-AUC?** Fraud class imbalanced (~7.8%) — ROC-AUC misleading on imbalanced data.

---

## 🔍 Fraud Detection Rules

| Rule | Logic | Real-World Meaning |
|------|-------|-------------------|
| Price too low | price < 0.30× city median | Fake listing to attract victims |
| Price too high | price > 4.0× city median | Inflated valuation fraud |
| Size mismatch | price_per_sqft < city 3rd percentile | Incomplete/fake data |
| Impossible dims | bed>20 OR bath>15 OR acre_lot>1000 | Data entry fraud |
| Price-size disconnect | house_size>5000 AND price<50000 | Scam signal |
| Duplicate listing | Same bed+bath+size in same zip | Multiple price manipulation |

---

## 🚀 Quick Start

### Local Development

```bash
# 1. Clone repo
git clone https://github.com/username/real-estate-fraud.git
cd real-estate-fraud

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup environment
cp .env.example .env
# Edit .env with your DATABASE_URL

# 4. Start FastAPI
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 5. Start Streamlit (new terminal)
streamlit run streamlit_app/app.py
```

### Docker (3 services in one command)

```bash
docker-compose up --build
# FastAPI  : http://localhost:8000/docs
# Streamlit: http://localhost:8501
# PostgreSQL: localhost:5432
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Fraud score + SHAP explanation |
| GET | `/history` | Past predictions with filters |
| GET | `/stats` | Aggregate fraud statistics |
| GET | `/health` | DB + model health check |
| GET | `/docs` | Swagger UI |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "price": 25000,
    "bed": 3,
    "bath": 2,
    "house_size": 1500,
    "city": "Austin",
    "state": "TX"
  }'
```

### Example Response

```json
{
  "fraud_score": 0.8234,
  "is_suspicious": true,
  "risk_tier": "HIGH",
  "shap_top3": [
    {"feature": "price_vs_city_median", "impact": 1.21, "value": 0.06},
    {"feature": "price_per_sqft",       "impact": 0.84, "value": 16.67},
    {"feature": "city_fraud_rate",       "impact": 0.52, "value": 0.08}
  ],
  "latency_ms": 43.2,
  "model_version": "1.1.0"
}
```

---

## 🧠 Key Technical Decisions

| Decision | Why |
|----------|-----|
| **PR-AUC over ROC-AUC** | Fraud is ~7.8% — ROC-AUC misleading on imbalanced data |
| **GroupKFold(city)** | Prevents geographic leakage — model generalizes to unseen cities |
| **Nested CV for Optuna** | Test set never used in tuning — honest evaluation |
| **Platt Scaling** | score=0.8 means ~80% fraud probability — required for threshold decisions |
| **SHAP over feature importance** | Per-prediction explanations for investigators |
| **Synthetic labels** | Real fraud data proprietary — domain logic bootstrap (industry standard) |

---

## 📁 Project Structure

```
real_estate_fraud_detection/
├── configs/          # YAML configs — all thresholds, paths, model params
├── src/              # Python source — ingestion, features, models, inference
├── api/              # FastAPI — endpoints, schemas
├── streamlit_app/    # Dashboard — 4 pages
├── tests/            # Pytest — inference, latency, edge cases
├── notebooks/        # Day 1-11 exploration notebooks
├── models/           # Saved model artifacts (.pkl)
├── data/             # Raw + processed data (not committed)
├── reports/          # EDA findings, SHAP insights, threshold decisions
├── Dockerfile        # FastAPI container
├── Dockerfile.streamlit  # Streamlit container
└── docker-compose.yml    # 3-service local setup
```

---

## ⭐ Production Hardening (★ NEW)

### Data Drift Monitoring
```bash
python src/drift_monitor.py --threshold 0.20
# Compares current city_median_price vs training baseline
# Alerts if >5 cities drift beyond threshold
```

### Threshold Decision Document
- Business cost matrix — FN cost >> FP cost
- Sensitivity analysis across 7 thresholds
- Justification for chosen threshold (0.70)
- See: `reports/threshold_decisions.md`

### Edge Case Tests
```bash
pytest tests/test_edge_cases.py -v
# 12+ tests: zero bedrooms, unknown city, price=0,
# 100 random inputs, Pydantic validation
```

---

## 🎯 Skills Demonstrated

| Skill | Implementation |
|-------|---------------|
| Supervised ML | LightGBM + LR + Stacking Ensemble |
| Feature Engineering | Stateless + fold-dependent + OOF-safe |
| Imbalanced Data | PR-AUC metric, scale_pos_weight |
| Experiment Tracking | MLflow — runs, artifacts, model registry |
| Hyperparameter Tuning | Optuna + Nested CV — no data leakage |
| Explainable AI | SHAP — TreeExplainer, waterfall, summary |
| Database | PostgreSQL + SQLAlchemy |
| API Development | FastAPI — Pydantic validation, auth |
| Frontend/UI | Streamlit — multi-page dashboard |
| Containerization | Docker + docker-compose |
| Deployment | Railway.app — live URL |
| ★ Drift Monitoring | Monthly city_median_price comparison |
| ★ Threshold Analysis | Business cost matrix |
| ★ Production Testing | 12+ edge case tests |

---

## 🔧 Environment Variables

```bash
DATABASE_URL=postgresql://fraud_user:fraud_pass@localhost:5432/fraud_db
API_KEY=your-secret-key
GIT_PYTHON_REFRESH=quiet
```

---


