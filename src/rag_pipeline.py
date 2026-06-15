"""
src/rag_pipeline.py — Real Estate Fraud Detection
RAG Pipeline — Document indexing + retrieval + Gemini free LLM.

Stack (all FREE):
  Embeddings  : sentence-transformers (local, no API)
  Vector store: chromadb (local, no API)
  LLM         : Google Gemini 1.5 Flash (free tier — 15 req/min, 1M tokens/day)

Documents indexed:
  - reports/eda_findings.md
  - reports/business_insights.md
  - reports/threshold_decisions.md
  - configs/problem_contract.md
  - reports/shap_importance.csv

Interview point:
  "Maine RAG implement kiya — project ke actual reports ko ChromaDB mein
  index kiya. Investigator koi bhi sawaal puche, Gemini project ke real
  data se answer deta hai — hallucinate nahi karta."
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Documents to index — project reports + configs
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE_DOCS = [
    "reports/eda_findings.md",
    "reports/business_insights.md",
    "reports/threshold_decisions.md",
    "configs/problem_contract.md",
]


def load_documents(doc_paths: Optional[List[str]] = None) -> List[dict]:
    """
    Load documents from disk for indexing.
    Returns list of {content, source, chunk_id} dicts.
    """
    paths   = doc_paths or KNOWLEDGE_BASE_DOCS
    docs    = []
    chunk_size = 500   # characters per chunk

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            logger.warning(f"Document not found — skipping: {path}")
            continue

        try:
            content = path.read_text(encoding="utf-8")

            # Split into chunks for better retrieval
            chunks = _chunk_text(content, chunk_size=chunk_size)
            for i, chunk in enumerate(chunks):
                docs.append({
                    "content":  chunk,
                    "source":   str(path),
                    "chunk_id": f"{path.stem}_chunk_{i}",
                    "doc_id":   f"{path.stem}_{i}",
                })
            logger.info(f"Loaded {len(chunks)} chunks from {path.name}")

        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")

    # Add fraud stats from CSV if available
    csv_path = Path("reports/shap_importance.csv")
    if csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            top10 = df.head(10).to_string(index=False)
            docs.append({
                "content":  f"SHAP Feature Importance (Top 10):\n{top10}",
                "source":   str(csv_path),
                "chunk_id": "shap_importance_0",
                "doc_id":   "shap_importance_0",
            })
            logger.info("Loaded SHAP importance CSV")
        except Exception as e:
            logger.warning(f"Failed to load SHAP CSV: {e}")

    logger.info(f"Total chunks loaded: {len(docs)}")
    return docs


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for better retrieval."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start  = 0
    while start < len(text):
        end   = min(start + chunk_size, len(text))
        chunk = text[start:end]

        # Try to break at paragraph or sentence boundary
        if end < len(text):
            last_newline = chunk.rfind('\n')
            last_period  = chunk.rfind('. ')
            break_point  = max(last_newline, last_period)
            if break_point > chunk_size // 2:
                chunk = chunk[:break_point + 1]
                end   = start + break_point + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return [c for c in chunks if c.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB Vector Store
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    RAG Pipeline:
      1. load_documents()  — read markdown/CSV files
      2. index()           — embed + store in ChromaDB
      3. retrieve()        — find relevant chunks for a query
      4. answer()          — Gemini generates answer from context
    """

    COLLECTION_NAME = "fraud_detection_kb"
    PERSIST_DIR     = "data/processed/chromadb"

    def __init__(self):
        self._client     = None
        self._collection = None
        self._embedder   = None
        self._indexed    = False

    def _init_chromadb(self):
        """Initialize ChromaDB client."""
        try:
            import chromadb
        except ImportError:
            raise ImportError("pip install chromadb")

        Path(self.PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.PERSIST_DIR)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB initialized — collection: {self.COLLECTION_NAME}")

    def _init_embedder(self):
        """Initialize sentence-transformers embedder (local, free)."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("pip install sentence-transformers")

        # Small, fast model — good for retrieval
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedder loaded: all-MiniLM-L6-v2")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using sentence-transformers."""
        if self._embedder is None:
            self._init_embedder()
        embeddings = self._embedder.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def index(self, force_reindex: bool = False) -> int:
        """
        Index all knowledge base documents into ChromaDB.
        Skips if already indexed (unless force_reindex=True).
        Returns number of chunks indexed.
        """
        self._init_chromadb()

        # Check if already indexed
        existing = self._collection.count()
        if existing > 0 and not force_reindex:
            logger.info(f"Already indexed — {existing} chunks in ChromaDB. Skipping.")
            self._indexed = True
            return existing

        # Clear and reindex
        if force_reindex and existing > 0:
            self._client.delete_collection(self.COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        docs = load_documents()
        if not docs:
            logger.warning("No documents to index!")
            return 0

        # Batch embed + add to ChromaDB
        batch_size = 32
        total = 0
        for i in range(0, len(docs), batch_size):
            batch     = docs[i:i + batch_size]
            texts     = [d["content"]  for d in batch]
            ids       = [d["doc_id"]   for d in batch]
            metadatas = [{"source": d["source"], "chunk_id": d["chunk_id"]}
                         for d in batch]

            embeddings = self._embed(texts)
            self._collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            total += len(batch)
            logger.info(f"Indexed {total}/{len(docs)} chunks")

        self._indexed = True
        logger.info(f"✅ RAG indexing complete — {total} chunks in ChromaDB")
        return total

    def retrieve(
        self,
        query: str,
        n_results: int = 4,
    ) -> List[dict]:
        """
        Find most relevant chunks for a query.
        Returns list of {content, source, score} dicts.
        """
        if not self._indexed:
            self.index()

        query_embedding = self._embed([query])[0]

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "content":    doc,
                "source":     meta.get("source", "unknown"),
                "score":      round(1 - dist, 4),   # cosine similarity
            })

        logger.info(f"Retrieved {len(chunks)} chunks for query: '{query[:50]}...'")
        return chunks

    def answer(
        self,
        query: str,
        context_chunks: Optional[List[dict]] = None,
        prediction_context: Optional[dict] = None,
        chat_history: Optional[List[dict]] = None,
    ) -> str:
        """
        Generate answer using Gemini + retrieved context.

        Args:
            query:              user's question
            context_chunks:     pre-retrieved chunks (or None to auto-retrieve)
            prediction_context: current listing prediction for context
            chat_history:       previous conversation turns

        Returns:
            Gemini's answer string
        """
        # Retrieve context if not provided
        if context_chunks is None:
            context_chunks = self.retrieve(query)

        # Build context string
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            source = Path(chunk["source"]).name
            context_text += f"\n[Source {i}: {source}]\n{chunk['content']}\n"

        # Build prediction context
        pred_text = ""
        if prediction_context:
            pred_text = f"""
CURRENT LISTING UNDER REVIEW:
- Price: ${prediction_context.get('price', 'N/A'):,.0f}
- City: {prediction_context.get('city', 'N/A')}
- Fraud Score: {prediction_context.get('fraud_score', 'N/A')}
- Risk Tier: {prediction_context.get('risk_tier', 'N/A')}
- Top Features: {prediction_context.get('shap_top3', [])}
"""

        # Build chat history string
        history_text = ""
        if chat_history:
            for msg in chat_history[-4:]:   # last 4 turns
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"

        prompt = f"""You are an expert real estate fraud detection assistant.
Answer the investigator's question using ONLY the provided context.
If the answer is not in the context, say so clearly.

KNOWLEDGE BASE CONTEXT:
{context_text}

{pred_text}

CONVERSATION HISTORY:
{history_text}

INVESTIGATOR'S QUESTION: {query}

Instructions:
- Be concise and professional
- Reference specific numbers/stats from the context when available
- If recommending action, be specific
- Do not hallucinate information not in the context

ANSWER:"""

        return _call_gemini(prompt)

    def explain_prediction(
        self,
        listing_dict: dict,
        prediction_result: dict,
    ) -> str:
        """
        Generate natural language explanation for a fraud prediction.
        Uses RAG context for enriched explanation.
        """
        # Get relevant context
        query  = f"fraud detection threshold risk tier {prediction_result.get('risk_tier', '')} price anomaly"
        chunks = self.retrieve(query, n_results=3)
        context_text = "\n".join([c["content"] for c in chunks])

        score     = prediction_result.get("fraud_score", 0)
        tier      = prediction_result.get("risk_tier", "UNKNOWN")
        shap_top3 = prediction_result.get("shap_top3", [])

        shap_text = ""
        for f in shap_top3:
            direction = "fraud signal" if f["impact"] > 0 else "normal signal"
            shap_text += f"\n  - {f['feature']} = {f['value']:.3f} → {direction} (impact: {f['impact']:+.4f})"

        prompt = f"""You are a real estate fraud detection expert. Explain this ML prediction in plain English.

LISTING:
- Price: ${listing_dict.get('price', 0):,.0f}
- Bedrooms: {listing_dict.get('bed', 'N/A')}
- City: {listing_dict.get('city', 'N/A')}, {listing_dict.get('state', 'N/A')}

ML PREDICTION:
- Fraud Score: {score:.4f} / 1.0
- Risk Tier: {tier}
- Top SHAP Signals: {shap_text}

RELEVANT KNOWLEDGE:
{context_text}

Write 2-3 clear paragraphs:
1. Risk level aur uska matlab plain English mein
2. Top fraud signals human-readable explanation mein
3. Investigator ke liye specific action

Professional, concise, no ML jargon."""

        return _call_gemini(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Gemini API call — free tier
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini(prompt: str, max_tokens: int = 500) -> str:
    """
    Call Google Gemini 1.5 Flash — free tier.
    Free: 15 requests/min, 1M tokens/day, no credit card.
    API key: aistudio.google.com
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("pip install google-generativeai")

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not set.\n"
            "Get free key at: aistudio.google.com\n"
            "Add to .env: GOOGLE_API_KEY=AIzaSy..."
        )

    genai.configure(api_key=api_key)
    model    = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.3,   # low temperature = more factual
        ),
    )
    return response.text


# ─────────────────────────────────────────────────────────────────────────────
# Singleton — load once
# ─────────────────────────────────────────────────────────────────────────────

_rag: Optional[RAGPipeline] = None


def get_rag() -> RAGPipeline:
    """Return global RAG pipeline — index on first call."""
    global _rag
    if _rag is None:
        _rag = RAGPipeline()
        _rag.index()
    return _rag
