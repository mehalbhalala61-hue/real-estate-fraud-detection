"""
streamlit_app/pages/1_predict.py — Predict Page
Submit listing → fraud score gauge → SHAP top 3 features
"""

import os
import sys
from pathlib import Path

import streamlit as st
import httpx

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

st.set_page_config(page_title="Predict — Fraud Detector", page_icon="🔍", layout="wide")


API_URL = st.session_state.get("api_base_url", os.getenv("API_BASE_URL", "http://localhost:8000"))
API_KEY = st.session_state.get("api_key", os.getenv("API_KEY", "dev-secret-key"))
HEADERS = {"X-API-Key": API_KEY}

st.title("🔍 Fraud Prediction")
st.markdown("Enter listing details to get a fraud score with explanation.")

# ── Input Form ───────────────────────────────────────────────────────────────
with st.form("predict_form"):
    st.markdown("### Listing Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        price      = st.number_input("Price (USD) *", min_value=1.0,
                                      value=350000.0, step=1000.0)
        bed        = st.number_input("Bedrooms",      min_value=0.0,
                                      value=3.0, step=1.0)
        bath       = st.number_input("Bathrooms",     min_value=0.0,
                                      value=2.0, step=0.5)

    with col2:
        house_size = st.number_input("House Size (sqft)", min_value=0.0,
                                      value=1800.0, step=100.0)
        acre_lot   = st.number_input("Lot Size (acres)",  min_value=0.0,
                                      value=0.2, step=0.01)
        status     = st.selectbox("Status", ["for_sale", "sold", ""])

    with col3:
        city       = st.text_input("City",     value="Austin")
        state      = st.text_input("State",    value="TX")
        zip_code   = st.text_input("Zip Code", value="78701")

    submitted = st.form_submit_button("🔍 Check for Fraud", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    payload = {
        "price":      price,
        "bed":        bed,
        "bath":       bath,
        "house_size": house_size,
        "acre_lot":   acre_lot,
        "city":       city or None,
        "state":      state or None,
        "zip_code":   zip_code or None,
        "status":     status or None,
    }

    with st.spinner("Analyzing listing..."):
        try:
            resp   = httpx.post(f"{API_URL}/predict", json=payload,
                                headers=HEADERS, timeout=30.0)
            result = resp.json()

            if resp.status_code != 200:
                st.error(f"API Error {resp.status_code}: {result}")
                st.stop()

        except httpx.ConnectError:
            st.error("Cannot connect to API — make sure FastAPI is running on port 8000")
            st.stop()

    score    = result["fraud_score"]
    tier     = result["risk_tier"]
    shap_top = result.get("shap_top3", [])
    latency  = result.get("latency_ms", 0)

    st.markdown("---")
    st.markdown("### 📊 Result")

    # ── Score + Tier ──────────────────────────────────────────────────────────
    col_score, col_tier, col_lat = st.columns(3)

    color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(tier, "⚪")

    col_score.metric("Fraud Score", f"{score:.4f}", help="Calibrated probability [0, 1]")
    col_tier.metric("Risk Tier", f"{color} {tier}")
    col_lat.metric("Latency", f"{latency:.0f}ms")

    # ── Gauge ─────────────────────────────────────────────────────────────────
    import plotly.graph_objects as go

    gauge_color = (
        "#E53935" if score >= 0.70 else
        "#FF9800" if score >= 0.40 else
        "#43A047"
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1},
            "bar": {"color": gauge_color, "thickness": 0.3},
            "steps": [
                {"range": [0.00, 0.40], "color": "#E8F5E9"},
                {"range": [0.40, 0.70], "color": "#FFF8E1"},
                {"range": [0.70, 1.00], "color": "#FFEBEE"},
            ],
            "threshold": {
                "line": {"color": "#E53935", "width": 3},
                "thickness": 0.85,
                "value": 0.70,
            },
        },
        title={"text": "Fraud Probability", "font": {"size": 16}},
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

    # ── Alert box ─────────────────────────────────────────────────────────────
    if tier == "HIGH":
        st.error("🚨 **HIGH RISK** — Block listing + Queue for manual review")
    elif tier == "MEDIUM":
        st.warning("⚠️ **MEDIUM RISK** — Flag for investigator review")
    else:
        st.success("✅ **LOW RISK** — Listing appears normal")

    # ── SHAP top 3 ────────────────────────────────────────────────────────────
    if shap_top:
        st.markdown("### 🧠 Why this score? (SHAP Top 3)")
        st.markdown("*Red = pushing fraud score UP | Blue = pulling it DOWN*")

        import plotly.express as px
        import pandas as pd

        shap_df = pd.DataFrame(shap_top)
        shap_df["direction"] = shap_df["impact"].apply(
            lambda x: "🔴 Fraud signal" if x > 0 else "🔵 Normal signal"
        )
        shap_df["label"] = shap_df.apply(
            lambda r: f"{r['feature']} = {r['value']:.2f}", axis=1
        )

        for _, row in shap_df.iterrows():
            bar_color = "#E53935" if row["impact"] > 0 else "#1E88E5"
            direction = "pushes score UP ↑" if row["impact"] > 0 else "pulls score DOWN ↓"
            st.markdown(
                f"**{row['feature']}** (value: `{row['value']:.3f}`) — "
                f"impact: `{row['impact']:+.4f}` — *{direction}*"
            )
            st.progress(
                min(abs(row["impact"]) / 2.0, 1.0),
            )
    else:
        st.info("SHAP explanation not available for this prediction.")

    # ── Raw JSON ──────────────────────────────────────────────────────────────
    with st.expander("📄 Raw API Response"):
        st.json(result)
