"""Адаптер лексического бейзлайна BM25."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from meme_organizer.retrieval.base import BaseRetrieverAdapter


class BM25RetrieverAdapter(BaseRetrieverAdapter):
    """Адаптер для алгоритма BM25 (аналог SQLite FTS5)."""

    def __init__(self, model_name: str = "bm25-baseline", **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
        self._bm25: BM25Okapi | None = None
        self._corpus_tokens: list[list[str]] = []

    def load_or_build(self) -> None:
        """Для BM25 модель строится непосредственно на корпусе в encode_corpus."""

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Простая и быстрая токенизация: нижний регистр, только буквы и цифры."""
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        return [t for t in clean.split() if len(t) > 1]

    def encode_corpus(self, corpus_docs: list[dict[str, Any]]) -> list[list[str]]:
        """Индексирует текстовые поля каждого мема."""
        self._corpus_tokens = []
        for doc in corpus_docs:
            user_caption = doc.get("user_caption") or ""
            rich_desc = doc.get("rich_description") or ""
            transcript = doc.get("transcript_or_text") or ""
            tags = " ".join(doc.get("tags") or [])

            full_text = f"{user_caption} {tags} {transcript} {rich_desc}"
            tokens = self._tokenize(full_text)
            self._corpus_tokens.append(tokens)

        self._bm25 = BM25Okapi(self._corpus_tokens)
        return self._corpus_tokens

    def encode_queries(self, queries: list[str]) -> list[list[str]]:
        """Токенизирует список запросов."""
        return [self._tokenize(q) for q in queries]

    def predict_scores(self, query_repr: list[list[str]], corpus_repr: Any) -> np.ndarray:
        """Считает матрицу BM25-скоров (N_queries, N_corpus)."""
        if self._bm25 is None:
            raise RuntimeError("BM25 index is not initialized. Run encode_corpus() first.")

        n_queries = len(query_repr)
        n_corpus = len(self._corpus_tokens)
        scores_matrix = np.zeros((n_queries, n_corpus), dtype=np.float32)

        for i, q_tokens in enumerate(query_repr):
            if q_tokens:
                scores_matrix[i] = self._bm25.get_scores(q_tokens)

        return scores_matrix

    def get_model_size_mb(self) -> float:
        """BM25 не весит ничего как отдельная нейросеть (память только под словарь)."""
        return 0.1
