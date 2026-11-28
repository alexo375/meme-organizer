"""Вычисление матрицы косинусного сходства для плотных эмбеддингов."""

from __future__ import annotations

import numpy as np


class CosineScorer:
    """Вычисляет матрицу косинусного сходства между эмбеддингами запросов и корпуса."""

    @staticmethod
    def score(query_embeddings: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
        """Возвращает матрицу косинусного сходства формы (n_queries, n_docs)."""
        q_norm = query_embeddings / np.maximum(np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12)
        c_norm = corpus_embeddings / np.maximum(np.linalg.norm(corpus_embeddings, axis=1, keepdims=True), 1e-12)
        return np.asarray(q_norm @ c_norm.T, dtype=np.float32)
