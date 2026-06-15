# Problem Contract — Real Estate Fraud Detection

**Version:** 1.0.0  
**Date:** Day 1 — System Freeze  
**Status:** LOCKED — Do not change without team discussion

---

## 1. Task Definition

| Field | Value |
|-------|-------|
| **Task Type** | Binary Classification |
| **Prediction Unit** | One real estate listing = one row |
| **Positive Class** | `is_fraud = 1` (fraudulent listing) |
| **Negative Class** | `is_fraud = 0` (legitimate listing) |

---

## 2. Dataset

| Field | Value |
|-------|-------|
| **Source** | USA Real Estate Dataset (Kaggle) |
| **Kaggle ID** | `ahmedshahriarsakib/usa-real-estate-dataset` |
| **Size** | ~200,000+ listings |
| **Geography** | USA + Puerto Rico |
| **Fraud Labels** | **Synthetically generated** via rule-based domain logic |

**Why synthetic labels?**  
Real labeled fraud data is proprietary. Companies like Zillow and Realtor.com have internal fraud teams, but they never release labeled datasets. We use domain logic (price anomalies, impossible dimensions, duplicates) to create realistic fraud patterns — **this is exactly how real fraud detection teams bootstrap without historical labels**.

---

## 3. Metrics

### Primary Metric: PR-AUC (Precision-Recall AUC)
- **Why not ROC-AUC?** Fraud class is ~3-5% of data (severely imbalanced). ROC-AUC is misleading on imbalanced datasets because it counts true negatives — and we have millions of true negatives. PR-AUC focuses only on the minority class.
- **Target:** PR-AUC > 0.80

### Secondary Metric: Recall@95Precision  
- **Why?** In production, false negatives (missing real fraud) cost more than false positives (manual review of legitimate listings). We want: "at 95% precision, what fraction of fraud are we catching?"
- **Target:** Recall@95P > 0.40

### Operational Constraint: Latency
- **p95 inference < 500ms** — Streamlit dashboard must be interactive
- Measured via `pytest tests/test_latency.py`

---

## 4. What Goes Into Training vs Test

| Split | Data | Purpose |
|-------|------|---------|
| **Train (80%)** | Used for CV, feature fitting, model training | |
| **Test (20%)** | **FROZEN** — touched only at final evaluation | Never use for tuning decisions |
| **city_stats** | Computed ONLY on train fold | Prevent target leakage |
| **Fraud labels (rule-based)** | Applied AFTER train/test split | city_stats fit on train only |

⚠️ **CRITICAL LEAKAGE RULES:**
1. `FraudLabeler.fit()` — ONLY on training data
2. `ColumnTransformer.fit()` — ONLY on training data  
3. Target encoding (city_fraud_rate) — OOF inside CV loop only
4. Test set never used for hyperparameter decisions

---

## 5. Fraud Pattern Definitions

| Pattern | Rule | Real-World Meaning |
|---------|------|-------------------|
| Price too low | price < 0.5× city_median | Fake listing to attract victims |
| Price too high | price > 3.0× city_median | Inflated valuation fraud |
| Size mismatch | price_per_sqft < city 10th percentile | Incomplete/fake data |
| Impossible dimensions | bed>20 OR bath>15 OR acre_lot>1000 | Data entry fraud |
| Price-size disconnect | house_size>5000 AND price<50000 | Scam signal |
| Duplicate listing | Same bed+bath+size in same zip | Multiple price manipulation |
| State anomaly | price < 0.4× OR > 4.0× state_median | Geographic price fraud |

---

## 6. Feature Leakage Assessment

| Column | Leakage Risk | Action |
|--------|-------------|--------|
| `price` | Low | Keep — target feature |
| `bed`, `bath`, `house_size` | Low | Keep |
| `acre_lot` | Low | Keep |
| `city`, `state` | Low | Keep — use carefully in encoding |
| `status` | Low | Keep |
| `zip_code` | Medium | Keep — use frequency encoding only |
| `prev_sold_date` | Medium | Keep — extract year/month features |
| `street` | **HIGH** | **DROP** — near-unique ID, no signal |
| `brokered_by` | **HIGH** | **DROP** — agent ID, not property feature |

---

## 7. Success Criteria

The project is considered successful when:
- [ ] PR-AUC > 0.80 on held-out test set
- [ ] Recall@95P > 0.40 on held-out test set
- [ ] p95 inference latency < 500ms (tested)
- [ ] SHAP explanations available for every prediction
- [ ] Docker `docker-compose up` starts all 3 services successfully
- [ ] Deployed to Render.com with live URL

---

## 8. Out of Scope (This Version)

- Real-time streaming predictions (batch API only)  
- Multi-language support
- Image/photo analysis of listings
- User authentication in Streamlit

---

*This document is locked after Day 1. Any changes require explicit review.*
