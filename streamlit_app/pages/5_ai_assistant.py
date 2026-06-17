"""
streamlit_app/pages/5_ai_assistant.py — AI Fraud Investigation Assistant
RAG-powered chatbot using Gemini (free) + ChromaDB + sentence-transformers.
"""

import os
import sys
from pathlib import Path

import streamlit as st
import httpx

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="AI Assistant — Fraud Detector",
    page_icon="🤖",
    layout="wide",
)

API_URL = (
    st.session_state.get("api_base_url")
    or os.getenv("API_BASE_URL", "http://localhost:8000")
)
API_KEY = st.session_state.get("api_key", os.getenv("API_KEY", "dev-secret-key"))
HEADERS = {"X-API-Key": API_KEY}

st.title("🤖 AI Fraud Investigation Assistant")
st.markdown(
    "**RAG-powered** — Gemini answers using actual project reports "
    "(EDA findings, SHAP analysis, threshold decisions)."
)
st.caption("Free: Google Gemini 1.5 Flash | sentence-transformers (local) | ChromaDB (local)")

# Check API key
google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not google_key:
    st.warning(
        "⚠️ GOOGLE_API_KEY not set.\n\n"
        "**Free key lena:**\n"
        "1. `aistudio.google.com` → Sign in with Google\n"
        "2. **Get API Key** → Copy\n"
        "3. `.env` mein add karo: `GOOGLE_API_KEY=AIzaSy...`\n"
        "4. Streamlit restart karo"
    )

@st.cache_resource(show_spinner="📚 Knowledge base index ho rahi hai...")
def load_rag():
    from src.rag_pipeline import get_rag
    return get_rag()

tab1, tab2, tab3 = st.tabs(["🔍 Analyze Listing", "💬 Chat", "📚 Knowledge Base"])

# ── TAB 1 ────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Submit listing → ML prediction + RAG explanation")

    col1, col2 = st.columns(2)
    with col1:
        price      = st.number_input("Price (USD) *", min_value=1.0, value=85000.0, step=1000.0)
        bed        = st.number_input("Bedrooms",      min_value=0.0, value=3.0, step=1.0)
        bath       = st.number_input("Bathrooms",     min_value=0.0, value=2.0, step=0.5)
        house_size = st.number_input("House Size (sqft)", min_value=0.0, value=1500.0, step=100.0)
    with col2:
        city     = st.text_input("City",     value="Austin")
        state    = st.text_input("State",    value="TX")
        zip_code = st.text_input("Zip Code", value="78701")
        status   = st.selectbox("Status", ["for_sale", "sold", ""])

    analyze_btn = st.button("🤖 Analyze with AI", use_container_width=True, disabled=not google_key)

    if analyze_btn:
        listing_dict = {
            "price": price, "bed": bed, "bath": bath,
            "house_size": house_size, "city": city,
            "state": state, "zip_code": zip_code, "status": status,
        }

        with st.spinner("🔍 ML prediction running..."):
            try:
                resp   = httpx.post(f"{API_URL}/predict", json=listing_dict, headers=HEADERS, timeout=30.0)
                result = resp.json()
            except httpx.ConnectError:
                st.error("FastAPI not running — uvicorn start karo")
                st.stop()

        score = result["fraud_score"]
        tier  = result["risk_tier"]
        col_s, col_t, col_l = st.columns(3)
        col_s.metric("Fraud Score", f"{score:.4f}")
        tier_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(tier, "⚪")
        col_t.metric("Risk Tier", f"{tier_icon} {tier}")
        col_l.metric("Latency", f"{result.get('latency_ms', 0):.0f}ms")

        if tier == "HIGH":     st.error("🚨 HIGH RISK — Block + Manual review")
        elif tier == "MEDIUM": st.warning("⚠️ MEDIUM RISK — Flag for investigator")
        else:                  st.success("✅ LOW RISK — Normal listing")

        st.markdown("---")
        st.markdown("### 🧠 AI Explanation (RAG-powered)")

        with st.spinner("📚 Knowledge base se context retrieve ho raha hai..."):
            try:
                rag         = load_rag()
                explanation = rag.explain_prediction(
                    listing_dict={**listing_dict},
                    prediction_result={**result, "city": city, "price": price},
                )
                st.markdown(explanation)
                st.session_state["last_prediction"] = {**result, "city": city, "price": price}
            except Exception as e:
                st.error(f"AI error: {e}")

        if result.get("shap_top3"):
            st.markdown("### 📊 SHAP Top 3")
            for f in result["shap_top3"]:
                icon = "⬆️" if f["impact"] > 0 else "⬇️"
                st.markdown(f"{icon} **{f['feature']}** = `{f['value']:.3f}` | impact `{f['impact']:+.4f}`")

        st.info("💡 Chat tab mein follow-up sawaal puch sakte ho!")

# ── TAB 2 ────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 💬 Fraud Investigation Chatbot")
    st.caption("Gemini project ke actual reports se answer deta hai")

    if not google_key:
        st.error("GOOGLE_API_KEY chahiye — .env mein add karo")
        st.stop()

    if "rag_chat_messages" not in st.session_state:
        st.session_state.rag_chat_messages = []

    if "last_prediction" in st.session_state:
        pred = st.session_state["last_prediction"]
        st.info(f"📌 Context: Score=`{pred['fraud_score']:.4f}` | Tier=`{pred['risk_tier']}` | City=`{pred.get('city','N/A')}`")

    # Quick questions
    st.markdown("**Quick questions:**")
    q1, q2, q3, q4 = st.columns(4)
    quick_qs = {
        "Threshold 0.70 kyun?": "Why was threshold 0.70 chosen for HIGH risk?",
        "Top fraud signals?":   "What are the top 5 fraud-indicating features from SHAP?",
        "Fraud rate kya hai?":  "What is the overall fraud rate in the dataset?",
        "HIGH pe kya karu?":    "What action should I take for a HIGH risk listing?",
    }
    for (label, question), col in zip(quick_qs.items(), [q1, q2, q3, q4]):
        if col.button(label, use_container_width=True):
            st.session_state.rag_chat_messages.append({"role": "user", "content": question})

    for msg in st.session_state.rag_chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 Sources"):
                    for src in msg["sources"]:
                        st.caption(f"📄 {Path(src['source']).name} (relevance: {src['score']:.2f})")

    if prompt := st.chat_input("Fraud detection ke baare mein puchho..."):
        st.session_state.rag_chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("📚 Knowledge base search ho rahi hai..."):
                try:
                    rag    = load_rag()
                    chunks = rag.retrieve(prompt, n_results=4)
                    reply  = rag.answer(
                        query=prompt,
                        context_chunks=chunks,
                        prediction_context=st.session_state.get("last_prediction"),
                        chat_history=st.session_state.rag_chat_messages[:-1],
                    )
                    st.markdown(reply)
                    with st.expander("📚 Sources"):
                        for chunk in chunks:
                            st.caption(f"📄 {Path(chunk['source']).name} (relevance: {chunk['score']:.2f})")
                    st.session_state.rag_chat_messages.append({
                        "role": "assistant", "content": reply, "sources": chunks
                    })
                except Exception as e:
                    err = f"Error: {e}"
                    st.error(err)
                    st.session_state.rag_chat_messages.append({"role": "assistant", "content": err})

    if st.session_state.rag_chat_messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.rag_chat_messages = []
            st.rerun()

# ── TAB 3 ────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📚 Knowledge Base — Indexed Documents")
    st.caption("Yeh documents ChromaDB mein indexed hain — RAG inhi se answers deta hai")

    docs_info = [
        ("reports/eda_findings.md",        "EDA Findings",        "Dataset stats, fraud rate, distributions"),
        ("reports/business_insights.md",   "Business Insights",   "SHAP analysis, top fraud signals"),
        ("reports/threshold_decisions.md", "Threshold Decisions", "Why 0.70, cost matrix, sensitivity"),
        ("configs/problem_contract.md",    "Problem Contract",    "Task definition, metrics, fraud patterns"),
        ("reports/shap_importance.csv",    "SHAP Importance",     "Feature rankings from Day 9"),
    ]

    for path_str, name, desc in docs_info:
        path   = Path(path_str)
        exists = path.exists()
        icon   = "✅" if exists else "❌"
        with st.expander(f"{icon} {name} — `{path_str}`"):
            st.caption(desc)
            if exists:
                content = path.read_text(encoding="utf-8")
                st.text_area("Preview", content[:400] + "...", height=120, disabled=True)
            else:
                st.warning("File nahi mili — corresponding notebook run karo")

    st.markdown("---")
    if st.button("🔄 Re-index Knowledge Base"):
        with st.spinner("Re-indexing..."):
            try:
                from src.rag_pipeline import RAGPipeline
                rag = RAGPipeline()
                n   = rag.index(force_reindex=True)
                st.cache_resource.clear()
                st.success(f"✅ {n} chunks re-indexed")
            except Exception as e:
                st.error(f"Error: {e}")
