"""Абстрактный контракт для моделей поиска (Retrieval Adapters)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from meme_organizer.retrieval.tracking import ExperimentTracker


class BaseRetrieverAdapter(ABC):
    """Единый контракт для любой поисковой модели (BM25, Bi-Encoder, Cross-Encoder, etc)."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.params = kwargs

    @abstractmethod
    def load_or_build(self) -> None:
        """Инициализирует или загружает веса/индексы модели."""

    def fit(
        self,
        train_pairs: list[dict[str, str]],
        corpus_docs: list[dict[str, Any]],
        val_pairs: list[dict[str, str]] | None = None,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        """Опциональное обучение/дообучение (по умолчанию no-op для zero-shot)."""
        _ = (train_pairs, corpus_docs, val_pairs, tracker)

    @abstractmethod
    def encode_corpus(self, corpus_docs: list[dict[str, Any]]) -> Any:
        """Кодирует базу документов (возвращает матрицы эмбеддингов или инвертированный индекс)."""

    @abstractmethod
    def encode_queries(self, queries: list[str]) -> Any:
        """Кодирует запросы (возвращает векторы или список токенов)."""

    @abstractmethod
    def predict_scores(self, query_repr: Any, corpus_repr: Any) -> np.ndarray:
        """Рассчитывает матрицу сходства shape (N_queries, N_corpus)."""

    def export_onnx(self, output_path: Path) -> Path | None:
        """Опциональный экспорт модели в ONNX формат."""
        _ = output_path
        return None

    def get_model_size_mb(self) -> float:
        """Возвращает размер модели на диске в МБ (для профилирования)."""
        return 0.0
