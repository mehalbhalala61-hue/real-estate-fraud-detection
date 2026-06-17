"""
streamlit_app/pages/2_analytics.py — Analytics Dashboard
Fraud rate trends, tier breakdown, top fraud cities, model metrics
"""

import os
import sys
from pathlib import Path

import streamlit as st
import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

st.set_page_config(page_title="Analytics — Fraud Detector", page_icon="📊", layout="wide")

API_URL = st.session_state.get("api_base_url", "http://localhost:8000")
API_KEY = st.session_state.get("api_key", os.getenv("API_KEY", "dev-secret-key"))
HEADERS = {"X-API-Key": API_KEY}

st.title("📊 Analytics Dashboard")
st.markdown("Aggregate fraud statistics and model performance overview.")

# ── Fetch stats from API ──────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch_stats():
    try:
        resp = httpx.get(f"{API_URL}/stats", headers=HEADERS, timeout=10.0)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None

@st.cache_data(ttl=30)
def fetch_history(limit=500):
    try:
        resp = httpx.get(f"{API_URL}/history?limit={limit}", headers=HEADERS, timeout=10.0)
        return pd.DataFrame(resp.json()) if resp.status_code == 200 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

if st.button("🔄 Refresh"):
    st.cache_data.clear()

stats = fetch_stats()
df    = fetch_history()

if stats is None:
    st.error("Cannot reach API — make sure FastAPI is running")
    st.stop()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.markdown("### Key Metrics")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Predictions",  f"{stats['total_predictions']:,}")
col2.metric("Suspicious Count",   f"{stats['suspicious_count']:,}")
col3.metric("Fraud Rate",         f"{stats['fraud_rate']*100:.1f}%")
col4.metric("Avg Fraud Score",    f"{stats['avg_fraud_score']:.4f}")

st.markdown("---")

# ── Tier Breakdown ────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### Risk Tier Distribution")
    tier_counts = stats.get("tier_counts", {})
    if tier_counts:
        tier_df = pd.DataFrame(
            list(tier_counts.items()), columns=["Tier", "Count"]
        )
        color_map = {"HIGH": "#E53935", "MEDIUM": "#FF9800", "LOW": "#43A047"}
        fig = px.pie(
            tier_df, names="Tier", values="Count",
            color="Tier", color_discrete_map=color_map,
            hole=0.4,
        )
        fig.update_layout(height=300, margin=dict(t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No tier data yet — make some predictions first!")

with col_right:
    st.markdown("### Top Fraud Cities")
    top_cities = stats.get("top_fraud_cities", [])
    if top_cities:
        city_df = pd.DataFrame(top_cities)
        fig = px.bar(
            city_df.head(10), x="count", y="city",
            orientation="h", color="count",
            color_continuous_scale=["#FFEBEE", "#E53935"],
            labels={"count": "Suspicious listings", "city": "City"},
        )
        fig.update_layout(height=300, margin=dict(t=20, b=0),
                          showlegend=False, coloraxis_showscale=False)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No fraud city data yet")

st.markdown("---")

# ── Fraud Score Distribution ──────────────────────────────────────────────────
if not df.empty and "fraud_score" in df.columns:
    st.markdown("### Fraud Score Distribution")
    fig = px.histogram(
        df, x="fraud_score", nbins=40,
        color_discrete_sequence=["#1E88E5"],
        labels={"fraud_score": "Fraud Score", "count": "Count"},
    )
    fig.add_vline(x=0.70, line_dash="dash", line_color="#E53935",
                  annotation_text="HIGH threshold (0.70)")
    fig.add_vline(x=0.40, line_dash="dash", line_color="#FF9800",
                  annotation_text="MEDIUM threshold (0.40)")
    fig.update_layout(height=300, margin=dict(t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Fraud by State ────────────────────────────────────────────────────────
    if "state" in df.columns:
        st.markdown("### Fraud Rate by State")
        state_df = (
            df.groupby("state")
            .agg(total=("fraud_score", "count"),
                 suspicious=("risk_tier", lambda x: (x == "HIGH").sum()))
            .assign(fraud_rate=lambda x: x["suspicious"] / x["total"] * 100)
            .reset_index()
            .query("total >= 2")
            .sort_values("fraud_rate", ascending=False)
            .head(20)
        )
        if not state_df.empty:
            fig = px.bar(
                state_df, x="state", y="fraud_rate",
                color="fraud_rate",
                color_continuous_scale=["#E8F5E9", "#E53935"],
                labels={"fraud_rate": "Fraud Rate (%)", "state": "State"},
            )
            fig.update_layout(height=300, margin=dict(t=20, b=0),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Latency distribution ──────────────────────────────────────────────────
    if "latency_ms" in df.columns:
        st.markdown("### Inference Latency")
        lat_col1, lat_col2, lat_col3 = st.columns(3)
        lat_col1.metric("p50 Latency", f"{df['latency_ms'].median():.0f}ms")
        lat_col2.metric("p95 Latency", f"{df['latency_ms'].quantile(0.95):.0f}ms")
        lat_col3.metric("Max Latency", f"{df['latency_ms'].max():.0f}ms",
                        delta="Target: <500ms",
                        delta_color="inverse" if df['latency_ms'].quantile(0.95) > 500 else "normal")

# ── Model Info ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Model Information")
model_col1, model_col2 = st.columns(2)
with model_col1:
    st.markdown("""
    | Property | Value |
    |----------|-------|
    | Model | Stacked + Calibrated |
    | Base models | LR + LightGBM Tuned |
    | Calibration | Platt Scaling |
    | Fraud threshold | 0.70 (HIGH) |
    | Review threshold | 0.40 (MEDIUM) |
    """)
with model_col2:
    st.markdown("""
    | Metric | Value |
    |--------|-------|
    | PR-AUC | 0.7694 |
    | Recall@95P | 0.1105 |
    | Dataset | 300k listings |
    | Features | 21 |
    | CV Strategy | GroupKFold(city) |
    """)
