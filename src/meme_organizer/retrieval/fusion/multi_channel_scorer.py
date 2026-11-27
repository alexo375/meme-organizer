"""Вычисление и объединение многоканальных оценок для BM25, Meta, OCR и ASR."""

from __future__ import annotations

from typing import Any

import numpy as np


class MultiChannelScorer:
    """Нормализация оценок, RRF-объединение и построение маски доступности каналов."""

    @staticmethod
    def min_max_normalize(scores: np.ndarray) -> np.ndarray:
        """Нормализует каждую строку оценок запроса в диапазон [0, 1]."""
        row_min = scores.min(axis=1, keepdims=True)
        row_max = scores.max(axis=1, keepdims=True)
        denom = np.where((row_max - row_min) > 1e-9, row_max - row_min, 1.0)
        return np.asarray((scores - row_min) / denom, dtype=np.float32)

    @staticmethod
    def rrf_fuse(channel_matrices: list[np.ndarray], k: int = 60) -> np.ndarray:
        """Объединяет матрицы оценок (n_queries, n_docs) методом Reciprocal Rank Fusion."""
        n_q, n_d = channel_matrices[0].shape
        fused = np.zeros((n_q, n_d), dtype=np.float32)
        for mat in channel_matrices:
            ranks = np.argsort(np.argsort(-mat, axis=1), axis=1) + 1
            fused += 1.0 / (k + ranks)
        return fused

    @staticmethod
    def build_channel_mask(
        ocr_texts: dict[str, str],
        asr_texts: dict[str, str],
        corpus_docs: list[dict[str, Any]],
    ) -> np.ndarray:
        """Возвращает булеву маску (n_docs, 4) доступности каналов [BM25, Meta, OCR, ASR]."""
        mask = np.ones((len(corpus_docs), 4), dtype=bool)
        for idx, doc in enumerate(corpus_docs):
            fh = doc["file_hash"]
            mask[idx, 2] = bool(ocr_texts.get(fh, "").strip())
            mask[idx, 3] = bool(asr_texts.get(fh, "").strip())
        return mask
