"""
streamlit_app/app.py — Real Estate Fraud Detection Dashboard
Main entry point — run with: streamlit run streamlit_app/app.py

Pages:
  1_predict.py   — Submit listing → fraud score + SHAP
  2_analytics.py — Fraud trends + geographic analysis
  3_history.py   — Past predictions table + filters
  4_shap.py      — Global SHAP feature importance
"""

import os
import sys
from pathlib import Path

import streamlit as st

# ── Project root setup ──────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from dotenv import load_dotenv
load_dotenv()

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real Estate Fraud Detector",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared config (accessible from all pages via session_state) ──────────────
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = os.getenv("API_BASE_URL", "https://real-estate-fraud-detection.onrender.com")
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("API_KEY", "dev-secret-key")

# ── Home page ────────────────────────────────────────────────────────────────
st.title("🏠 Real Estate Fraud Detection")
st.markdown("**ML-powered fraud detection for real estate listings**")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 🔍 Predict")
    st.markdown("Submit a listing and get instant fraud score with SHAP explanation")
    if st.button("Go to Predict →", use_container_width=True):
        st.switch_page("pages/1_predict.py")

with col2:
    st.markdown("### 📊 Analytics")
    st.markdown("Fraud trends, geographic analysis, and model performance metrics")
    if st.button("Go to Analytics →", use_container_width=True):
        st.switch_page("pages/2_analytics.py")

with col3:
    st.markdown("### 📋 History")
    st.markdown("Browse past predictions with filters and download CSV")
    if st.button("Go to History →", use_container_width=True):
        st.switch_page("pages/3_history.py")

with col4:
    st.markdown("### 🧠 SHAP Viewer")
    st.markdown("Global feature importance from SHAP analysis")
    if st.button("Go to SHAP →", use_container_width=True):
        st.switch_page("pages/4_shap.py")

st.markdown("---")

# ── API Health check ─────────────────────────────────────────────────────────
st.markdown("### API Status")
try:
    import httpx
    resp = httpx.get(
        f"{st.session_state.api_base_url}/health",
        timeout=3.0,
    )
    health = resp.json()
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("API", "✅ Online" if health["status"] == "healthy" else "⚠️ Degraded")
    col_b.metric("Database", "✅ Connected" if health["db_connected"] else "❌ Down")
    col_c.metric("Model", "✅ Loaded" if health["model_loaded"] else "❌ Not loaded")
except Exception:
    st.error(
        "❌ API not reachable — make sure FastAPI is running:\n\n"
        "```\nuvicorn api.main:app --reload --host 0.0.0.0 --port 8000\n```"
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_url = st.text_input("API URL", value=st.session_state.api_base_url)
    api_key = st.text_input("API Key", value=st.session_state.api_key, type="password")
    if st.button("Save Settings"):
        st.session_state.api_base_url = api_url
        st.session_state.api_key = api_key
        st.success("Settings saved!")

    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown(f"- Threshold HIGH: `0.70`")
    st.markdown(f"- Threshold MED: `0.40`")
    st.markdown(f"- Primary metric: `PR-AUC`")
