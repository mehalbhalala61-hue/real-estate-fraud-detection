"""
streamlit_app/pages/3_history.py — History Page
Filterable prediction history table + CSV download
"""

import os
import sys
from pathlib import Path

import streamlit as st
import httpx
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

st.set_page_config(page_title="History — Fraud Detector", page_icon="📋", layout="wide")

API_URL = st.session_state.get("api_base_url", "http://localhost:8000")
API_KEY = st.session_state.get("api_key", "dev-secret-key")
HEADERS = {"X-API-Key": API_KEY}

st.title("📋 Prediction History")
st.markdown("Browse, filter, and export past fraud predictions.")

# ── Filters ───────────────────────────────────────────────────────────────────
st.markdown("### Filters")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    risk_tier = st.selectbox("Risk Tier", ["All", "HIGH", "MEDIUM", "LOW"])
with col2:
    state_filter = st.text_input("State", placeholder="e.g. TX")
with col3:
    city_filter = st.text_input("City", placeholder="e.g. Austin")
with col4:
    min_score = st.slider("Min Fraud Score", 0.0, 1.0, 0.0, 0.01)
with col5:
    limit = st.selectbox("Show rows", [50, 100, 200, 500], index=1)

# ── Fetch data ────────────────────────────────────────────────────────────────
params = f"limit={limit}&min_score={min_score}"
if risk_tier != "All":
    params += f"&risk_tier={risk_tier}"
if state_filter:
    params += f"&state={state_filter.upper()}"
if city_filter:
    params += f"&city={city_filter}"

try:
    resp = httpx.get(f"{API_URL}/history?{params}", headers=HEADERS, timeout=10.0)
    if resp.status_code == 200:
        data = resp.json()
        df   = pd.DataFrame(data)
    else:
        st.error(f"API error: {resp.status_code}")
        st.stop()
except httpx.ConnectError:
    st.error("Cannot reach API — make sure FastAPI is running")
    st.stop()

# ── Results ───────────────────────────────────────────────────────────────────
st.markdown(f"### Results — {len(df)} predictions found")

if df.empty:
    st.info("No predictions found with current filters. Make some predictions first!")
    st.stop()

# Color code risk tier
def color_tier(val):
    colors = {"HIGH": "background-color: #FFEBEE; color: #E53935; font-weight: bold",
              "MEDIUM": "background-color: #FFF8E1; color: #FF8F00; font-weight: bold",
              "LOW": "background-color: #E8F5E9; color: #2E7D32; font-weight: bold"}
    return colors.get(val, "")

def color_score(val):
    try:
        v = float(val)
        if v >= 0.70:   return "color: #E53935; font-weight: bold"
        elif v >= 0.40: return "color: #FF8F00"
        return "color: #2E7D32"
    except:
        return ""

# Display columns
display_cols = [c for c in ["id","created_at","city","state","price",
                              "fraud_score","risk_tier","latency_ms"] if c in df.columns]
display_df = df[display_cols].copy()

if "fraud_score" in display_df.columns:
    display_df["fraud_score"] = display_df["fraud_score"].round(4)
if "price" in display_df.columns:
    display_df["price"] = display_df["price"].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
    )
if "latency_ms" in display_df.columns:
    display_df["latency_ms"] = display_df["latency_ms"].apply(
        lambda x: f"{x:.0f}ms" if pd.notna(x) else "N/A"
    )

styled = display_df.style.applymap(color_tier, subset=["risk_tier"]) \
    if "risk_tier" in display_df.columns else display_df.style

st.dataframe(styled, use_container_width=True, height=400)

# ── Summary stats ─────────────────────────────────────────────────────────────
st.markdown("### Summary")
s_col1, s_col2, s_col3 = st.columns(3)
if "risk_tier" in df.columns:
    tier_counts = df["risk_tier"].value_counts()
    s_col1.metric("HIGH risk",   tier_counts.get("HIGH",   0))
    s_col2.metric("MEDIUM risk", tier_counts.get("MEDIUM", 0))
    s_col3.metric("LOW risk",    tier_counts.get("LOW",    0))

# ── CSV Download ──────────────────────────────────────────────────────────────
st.markdown("### Export")
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download CSV",
    data=csv_data,
    file_name="fraud_predictions.csv",
    mime="text/csv",
    use_container_width=True,
)
