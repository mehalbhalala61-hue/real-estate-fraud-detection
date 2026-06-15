"""
src/text_features.py — Real Estate Fraud Detection
TF-IDF + TruncatedSVD text feature pipeline.

USA Real Estate Dataset has NO description column — text.enabled = false in config.
This file exists so imports don't crash. Set text.enabled = true in config when
a description column is available.

Architecture: TfidfVectorizer → TruncatedSVD(50) → 50 dense features
Latency: fits in <500ms p95 (CPU) because SVD reduces to 50 dims only.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def is_text_enabled(cfg: dict) -> bool:
    """Check config whether text modality is active."""
    return cfg.get("text_features", {}).get("enabled", False)


class TextPipeline:
    """
    TF-IDF + TruncatedSVD pipeline for listing description text.

    Usage (when text.enabled = true):
        pipeline = TextPipeline(cfg)
        pipeline.fit(X_train["description"])
        X_train_text = pipeline.transform(X_train["description"])  # shape (n, 50)
        pipeline.save()

    When text.enabled = false (current — no description column):
        transform() returns zero array of shape (n, n_svd_components)
        so downstream code doesn't crash.
    """

    def __init__(self, cfg: dict):
        self.cfg      = cfg
        self.enabled  = is_text_enabled(cfg)
        self._fitted  = False
        self.vectorizer_  = None
        self.svd_         = None

        tf_cfg = cfg.get("text_features", {})
        self.column          = tf_cfg.get("column", "description")
        self.max_features    = tf_cfg.get("max_features", 5000)
        self.n_components    = tf_cfg.get("n_svd_components", 50)

    def fit(self, series: pd.Series) -> "TextPipeline":
        """
        Fit TF-IDF + SVD on training text. Call on training data only.
        If text disabled, does nothing.
        """
        if not self.enabled:
            logger.info("TextPipeline.fit() skipped — text_features.enabled = false")
            self._fitted = True
            return self

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
        except ImportError:
            raise ImportError("scikit-learn required for TextPipeline")

        text = series.fillna("").astype(str)
        logger.info(f"TextPipeline.fit() — {len(text):,} samples, max_features={self.max_features}")

        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=5,
        )
        tfidf_matrix = self.vectorizer_.fit_transform(text)

        self.svd_ = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.svd_.fit(tfidf_matrix)

        explained = self.svd_.explained_variance_ratio_.sum()
        logger.info(
            f"  TF-IDF vocab: {len(self.vectorizer_.vocabulary_):,} | "
            f"SVD explained variance: {explained*100:.1f}%"
        )
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        """
        Transform text to SVD features. Shape: (n_samples, n_components).
        Returns zero array if text disabled — so downstream concat doesn't break.
        """
        n = len(series)
        if not self.enabled:
            return np.zeros((n, self.n_components), dtype=np.float32)

        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        text        = series.fillna("").astype(str)
        tfidf_mat   = self.vectorizer_.transform(text)
        svd_features = self.svd_.transform(tfidf_mat).astype(np.float32)
        logger.info(f"TextPipeline.transform() — output shape: {svd_features.shape}")
        return svd_features

    def fit_transform(self, series: pd.Series) -> np.ndarray:
        """Fit + transform on same series. Use ONLY on training data."""
        return self.fit(series).transform(series)

    def get_feature_names(self) -> list:
        """Column names for the SVD output features."""
        return [f"text_svd_{i}" for i in range(self.n_components)]

    def save(self, path: Optional[str] = None) -> str:
        """Save fitted pipeline to disk."""
        out = path or self.cfg["paths"].get("text_pipeline", "models/text_pipeline.pkl")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"TextPipeline saved → {out}")
        return out

    @classmethod
    def load(cls, path: str) -> "TextPipeline":
        """Load saved pipeline from disk."""
        with open(path, "rb") as f:
            pipeline = pickle.load(f)
        logger.info(f"TextPipeline loaded from {path}")
        return pipeline