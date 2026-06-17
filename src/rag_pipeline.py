"""
src/rag_pipeline.py — Gemini-powered RAG (no local embeddings)
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_DOCS = [
    "reports/eda_findings.md",
    "reports/business_insights.md",
    "reports/threshold_decisions.md",
    "configs/problem_contract.md",
]

def load_documents() -> str:
    """Load all documents into a single context string."""
    context = ""
    for path_str in KNOWLEDGE_BASE_DOCS:
        path = Path(path_str)
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                context += f"\n\n=== {path.name} ===\n{content[:1500]}"
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
    return context

class RAGPipeline:
    def __init__(self):
        self._context = None
        self._indexed = False

    def index(self, force_reindex: bool = False) -> int:
        self._context = load_documents()
        self._indexed = True
        logger.info("Knowledge base loaded into memory")
        return len(self._context)

    def retrieve(self, query: str, n_results: int = 4) -> List[dict]:
        if not self._indexed:
            self.index()
        return [{"content": self._context, "source": "knowledge_base", "score": 1.0}]

    def answer(self, query: str, context_chunks=None, prediction_context=None, chat_history=None) -> str:
        if not self._indexed:
            self.index()

        pred_text = ""
        if prediction_context:
            pred_text = f"""
CURRENT LISTING:
- Price: ${prediction_context.get('price', 'N/A'):,.0f}
- City: {prediction_context.get('city', 'N/A')}
- Fraud Score: {prediction_context.get('fraud_score', 'N/A')}
- Risk Tier: {prediction_context.get('risk_tier', 'N/A')}
"""

        prompt = f"""You are a real estate fraud detection expert.
Answer using the knowledge base below.

KNOWLEDGE BASE:
{self._context[:3000]}

{pred_text}

QUESTION: {query}

Answer concisely and professionally:"""

        return _call_gemini(prompt)

    def explain_prediction(self, listing_dict: dict, prediction_result: dict) -> str:
        if not self._indexed:
            self.index()

        score = prediction_result.get("fraud_score", 0)
        tier  = prediction_result.get("risk_tier", "UNKNOWN")
        shap_top3 = prediction_result.get("shap_top3", [])

        shap_text = ""
        for f in shap_top3:
            direction = "fraud signal" if f["impact"] > 0 else "normal signal"
            shap_text += f"\n  - {f['feature']} = {f['value']:.3f} → {direction}"

        prompt = f"""Explain this fraud prediction in plain English.

LISTING:
- Price: ${listing_dict.get('price', 0):,.0f}
- City: {listing_dict.get('city', 'N/A')}, {listing_dict.get('state', 'N/A')}

PREDICTION:
- Fraud Score: {score:.4f}
- Risk Tier: {tier}
- SHAP Signals: {shap_text}

CONTEXT:
{self._context[:2000]}

Write 2-3 clear paragraphs explaining the risk, signals, and recommended action:"""

        return _call_gemini(prompt)


def _call_gemini(prompt: str, max_tokens: int = 500) -> str:
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("pip install google-generativeai")

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")

    genai.configure(api_key=api_key)
    model    = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.3,
        ),
    )
    return response.text


_rag: Optional[RAGPipeline] = None

def get_rag() -> RAGPipeline:
    global _rag
    if _rag is None:
        _rag = RAGPipeline()
        _rag.index()
    return _rag