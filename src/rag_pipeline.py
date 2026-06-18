"""
src/rag_pipeline.py — Real Estate Fraud Detection
LIGHTWEIGHT RAG — no sentence-transformers, no ChromaDB.
Uses direct REST API call to Gemini — no library version issues.
"""

import logging
import os
import re
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_DOCS = [
    "reports/eda_findings.md",
    "reports/business_insights.md",
    "reports/threshold_decisions.md",
    "configs/problem_contract.md",
]


def load_documents(doc_paths: Optional[List[str]] = None) -> List[dict]:
    paths = doc_paths or KNOWLEDGE_BASE_DOCS
    docs  = []

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            logger.warning(f"Document not found — skipping: {path}")
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            docs.append({"content": content, "source": str(path)})
            logger.info(f"Loaded {path.name} ({len(content)} chars)")
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")

    csv_path = Path("reports/shap_importance.csv")
    if csv_path.exists():
        try:
            import pandas as pd
            df    = pd.read_csv(csv_path)
            top10 = df.head(10).to_string(index=False)
            docs.append({
                "content": f"SHAP Feature Importance (Top 10):\n{top10}",
                "source":  str(csv_path),
            })
        except Exception as e:
            logger.warning(f"Failed to load SHAP CSV: {e}")

    logger.info(f"Total documents loaded: {len(docs)}")
    return docs


_STOPWORDS = {
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "but",
    "in", "with", "to", "for", "of", "as", "by", "what", "why", "how",
    "this", "that", "these", "those", "be", "was", "were", "are", "do",
    "does", "did", "can", "could", "should", "would", "i", "you", "it",
}


def _score_relevance(query: str, doc_content: str) -> float:
    query_words = set(re.findall(r"\w+", query.lower())) - _STOPWORDS
    if not query_words:
        return 0.0
    doc_lower = doc_content.lower()
    matches   = sum(1 for w in query_words if w in doc_lower)
    return matches / len(query_words)


class RAGPipeline:

    def __init__(self):
        self._docs    = []
        self._indexed = False

    def index(self, force_reindex: bool = False) -> int:
        if self._indexed and not force_reindex:
            return len(self._docs)
        self._docs    = load_documents()
        self._indexed = True
        logger.info(f"Lightweight index ready — {len(self._docs)} documents")
        return len(self._docs)

    def retrieve(self, query: str, n_results: int = 4) -> List[dict]:
        if not self._indexed:
            self.index()
        if not self._docs:
            return []

        scored = [
            {
                "content": d["content"],
                "source":  d["source"],
                "score":   round(_score_relevance(query, d["content"]), 4),
            }
            for d in self._docs
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)

        top = scored[:n_results]
        if all(s["score"] == 0 for s in top):
            top = scored

        logger.info(f"Retrieved {len(top)} docs for query: '{query[:40]}'")
        return top

    def answer(
        self,
        query: str,
        context_chunks: Optional[List[dict]] = None,
        prediction_context: Optional[dict] = None,
        chat_history: Optional[List[dict]] = None,
    ) -> str:
        if context_chunks is None:
            context_chunks = self.retrieve(query)

        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            source  = Path(chunk["source"]).name
            content = chunk["content"][:2000]
            context_text += f"\n[Source {i}: {source}]\n{content}\n"

        pred_text = ""
        if prediction_context:
            pred_text = f"""
CURRENT LISTING:
- Price: ${prediction_context.get('price', 'N/A'):,.0f}
- City: {prediction_context.get('city', 'N/A')}
- Fraud Score: {prediction_context.get('fraud_score', 'N/A')}
- Risk Tier: {prediction_context.get('risk_tier', 'N/A')}
"""

        history_text = ""
        if chat_history:
            for msg in chat_history[-4:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"

        prompt = f"""You are a real estate fraud detection assistant. Answer using ONLY the provided context.

KNOWLEDGE BASE:
{context_text}
{pred_text}
RECENT CONVERSATION:
{history_text}

QUESTION: {query}

Be concise (3-4 sentences max), reference specific numbers from context, no hallucination.

ANSWER:"""

        return _call_gemini(prompt)

    def explain_prediction(self, listing_dict: dict, prediction_result: dict) -> str:
        query   = f"threshold risk tier {prediction_result.get('risk_tier','')} price anomaly fraud signal"
        chunks  = self.retrieve(query, n_results=2)
        context = "\n".join([c["content"][:1500] for c in chunks])

        score     = prediction_result.get("fraud_score", 0)
        tier      = prediction_result.get("risk_tier", "UNKNOWN")
        shap_top3 = prediction_result.get("shap_top3", [])

        shap_text = ""
        for f in shap_top3:
            direction = "fraud signal" if f["impact"] > 0 else "normal signal"
            shap_text += f"\n  - {f['feature']} = {f['value']:.3f} → {direction} (impact: {f['impact']:+.4f})"

        prompt = f"""Real estate fraud expert: explain this prediction in plain English.

LISTING: Price ${listing_dict.get('price', 0):,.0f}, {listing_dict.get('city', 'N/A')}, {listing_dict.get('state', 'N/A')}
PREDICTION: Score={score:.4f}, Tier={tier}
TOP SIGNALS:{shap_text}

CONTEXT: {context[:1000]}

Write 2 short paragraphs (no jargon): (1) risk level + meaning, (2) specific action for investigator."""

        return _call_gemini(prompt, max_tokens=300)


def _call_gemini(prompt: str, max_tokens: int = 400) -> str:
    """Direct REST API call to Gemini — no library version issues."""
    import httpx

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set. Get free key: aistudio.google.com")

    url = (
        "https://generativelanguage.googleapis.com/v1/models/"
        f"gemini-1.5-flash:generateContent?key={api_key}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.3,
        },
    }

    resp = httpx.post(url, json=payload, timeout=25)
    data = resp.json()

    if resp.status_code != 200:
        raise ValueError(f"{resp.status_code} {data}")

    return data["candidates"][0]["content"]["parts"][0]["text"]


_rag: Optional[RAGPipeline] = None


def get_rag() -> RAGPipeline:
    global _rag
    if _rag is None:
        _rag = RAGPipeline()
        _rag.index()
    return _rag