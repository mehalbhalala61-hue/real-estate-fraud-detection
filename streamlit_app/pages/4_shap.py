"""
streamlit_app/pages/4_shap.py — SHAP Viewer
Global feature importance from SHAP analysis (Day 9 artifacts)
"""

import os
import sys
import pickle
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

st.set_page_config(page_title="SHAP Viewer — Fraud Detector", page_icon="🧠", layout="wide")

st.title("🧠 SHAP Feature Importance")
st.markdown("Global feature importance from SHAP analysis on LightGBM model.")

# ── Load SHAP artifacts ───────────────────────────────────────────────────────
@st.cache_resource
def load_shap_artifacts():
    shap_path = Path("data/processed/shap_artifacts.pkl")
    if not shap_path.exists():
        return None
    with open(shap_path, "rb") as f:
        return pickle.load(f)

artifacts = load_shap_artifacts()

if artifacts is None:
    st.error(
        "SHAP artifacts not found — run `09_shap.ipynb` first to generate them.\n\n"
        "Expected file: `data/processed/shap_artifacts.pkl`"
    )
    st.stop()

importance_df = artifacts["importance_df"]
shap_values   = artifacts["shap_values"]
X_sample      = artifacts["X_sample"]
y_sample      = artifacts["y_sample"]
top5          = artifacts.get("top5_features", importance_df.head(5)["feature"].tolist())

st.success(f"✅ SHAP artifacts loaded — {len(X_sample):,} samples, {len(importance_df)} features")

st.markdown("---")

# ── Feature Importance Bar Chart ──────────────────────────────────────────────
st.markdown("### Feature Importance — Mean |SHAP Value|")
st.markdown("*Higher value = more impact on fraud score*")

n_features = st.slider("Show top N features", 5, len(importance_df), 20)
top_df = importance_df.head(n_features).copy()

colors = ["#E53935" if i < 5 else "#1E88E5" if i < 10 else "#78909C"
          for i in range(len(top_df))]

fig = go.Figure(go.Bar(
    x=top_df["mean_shap"][::-1],
    y=top_df["feature"][::-1],
    orientation="h",
    marker_color=colors[::-1],
    marker_line_width=0,
    opacity=0.85,
))
fig.update_layout(
    xaxis_title="Mean |SHAP Value| (impact on fraud probability)",
    yaxis_title="",
    height=max(400, n_features * 22),
    margin=dict(t=10, b=40, l=10, r=10),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Top 5 Insights ────────────────────────────────────────────────────────────
st.markdown("### 🔑 Top 5 Fraud Signals")

import numpy as np
fraud_mask  = y_sample == 1
normal_mask = y_sample == 0

cols = st.columns(5)
for i, feat in enumerate(top5[:5]):
    if feat in X_sample.columns:
        f_mean = X_sample.loc[fraud_mask,  feat].mean()
        n_mean = X_sample.loc[normal_mask, feat].mean()
        diff   = (f_mean - n_mean) / (abs(n_mean) + 1e-9) * 100
        shap_m = importance_df[importance_df["feature"] == feat]["mean_shap"].values[0]

        with cols[i]:
            st.markdown(f"**#{i+1}**")
            st.markdown(f"`{feat}`")
            st.metric("Mean |SHAP|", f"{shap_m:.4f}")
            st.markdown(f"Fraud avg: `{f_mean:.2f}`")
            st.markdown(f"Normal avg: `{n_mean:.2f}`")
            st.markdown(f"Diff: `{diff:+.0f}%`")

st.markdown("---")

# ── SHAP Value Distribution ───────────────────────────────────────────────────
st.markdown("### SHAP Value Distribution — Fraud vs Normal")
selected_feat = st.selectbox("Select feature", importance_df["feature"].tolist())

if selected_feat in X_sample.columns:
    feat_idx  = list(X_sample.columns).index(selected_feat)
    shap_vals = shap_values[:, feat_idx]

    plot_df = pd.DataFrame({
        "SHAP Value":     shap_vals,
        "Feature Value":  X_sample[selected_feat].values,
        "Label":          y_sample.values,
    })
    plot_df["Label"] = plot_df["Label"].map({1: "Fraud", 0: "Normal"})

    col_left, col_right = st.columns(2)

    with col_left:
        fig = px.histogram(
            plot_df, x="SHAP Value", color="Label",
            barmode="overlay", opacity=0.7, nbins=40,
            color_discrete_map={"Fraud": "#E53935", "Normal": "#1E88E5"},
            title=f"SHAP Distribution — {selected_feat}",
        )
        fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
        fig.update_layout(height=320, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig2 = px.scatter(
            plot_df, x="Feature Value", y="SHAP Value",
            color="Label", opacity=0.4,
            color_discrete_map={"Fraud": "#E53935", "Normal": "#1E88E5"},
            title=f"Feature Value vs SHAP — {selected_feat}",
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        fig2.update_layout(height=320, margin=dict(t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Saved plots ───────────────────────────────────────────────────────────────
st.markdown("### Saved SHAP Plots (from Day 9)")

plots_dir = Path("reports/plots")
shap_plots = sorted(plots_dir.glob("shap_*.png")) if plots_dir.exists() else []

if shap_plots:
    plot_names = {p.name: p for p in shap_plots}
    selected_plot = st.selectbox("Select plot", list(plot_names.keys()))
    st.image(str(plot_names[selected_plot]), use_column_width=True)
else:
    st.info("No SHAP plots found in reports/plots/ — run 09_shap.ipynb to generate them")

# ── Full importance table ─────────────────────────────────────────────────────
with st.expander("📄 Full Feature Importance Table"):
    st.dataframe(importance_df, use_container_width=True)
    csv = importance_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "shap_importance.csv", "text/csv")
