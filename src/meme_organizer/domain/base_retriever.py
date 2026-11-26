"""Базовый контракт моделей поиска."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from meme_organizer.evaluation.metrics.experiment_tracker import ExperimentTracker


class BaseRetriever(ABC):
    """Абстрактный контракт модели поиска (разреженной, плотной, гибридной, реранкера)."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.params = kwargs

    @abstractmethod
    def load_or_build(self) -> None:
        """Инициализирует или загружает веса и индексы модели."""

    def fit(
        self,
        train_pairs: list[dict[str, str]],
        corpus_docs: list[dict[str, Any]],
        val_pairs: list[dict[str, str]] | None = None,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        """Опционально обучает или калибрует модель (по умолчанию без действия)."""
        _ = (train_pairs, corpus_docs, val_pairs, tracker)

    @abstractmethod
    def encode_corpus(self, corpus_docs: list[dict[str, Any]]) -> Any:
        """Кодирует корпус документов в представления (эмбеддинги или инвертированный индекс)."""

    @abstractmethod
    def encode_queries(self, queries: list[str]) -> Any:
        """Кодирует запросы в представления."""

    @abstractmethod
    def predict_scores(self, query_repr: Any, corpus_repr: Any) -> np.ndarray:
        """Вычисляет матрицу сходства формы (N_queries, N_corpus)."""

    def export_onnx(self, output_path: Path) -> Path | None:
        """Опционально экспортирует модель в формат ONNX и возвращает путь или None."""
        _ = output_path
        return None

    def get_onnx_paths(self) -> list[Path]:
        """Возвращает пути к сгенерированным ONNX-моделям для логирования в MLflow."""
        return []

    def get_artifact_dirs(self) -> dict[str, Path]:
        """Возвращает директории артефактов (например, {'hf_model': Path(...)}) для загрузки в MLflow."""
        return {}

    def get_model_size_mb(self) -> float:
        """Возвращает размер модели на диске в МБ."""
        return 0.0
