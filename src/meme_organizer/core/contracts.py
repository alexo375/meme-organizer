"""Base contract for retrieval models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from meme_organizer.core.tracker import ExperimentTracker


class BaseRetriever(ABC):
    """Abstract contract for any retrieval model (sparse, dense, hybrid, reranker)."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.params = kwargs

    @abstractmethod
    def load_or_build(self) -> None:
        """Initialize or load weights/indices."""

    def fit(
        self,
        train_pairs: list[dict[str, str]],
        corpus_docs: list[dict[str, Any]],
        val_pairs: list[dict[str, str]] | None = None,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        """Optional training or calibration (default no-op)."""
        _ = (train_pairs, corpus_docs, val_pairs, tracker)

    @abstractmethod
    def encode_corpus(self, corpus_docs: list[dict[str, Any]]) -> Any:
        """Encode documents corpus into representations (embeddings or inverted index)."""

    @abstractmethod
    def encode_queries(self, queries: list[str]) -> Any:
        """Encode queries into representations."""

    @abstractmethod
    def predict_scores(self, query_repr: Any, corpus_repr: Any) -> np.ndarray:
        """Compute similarity matrix of shape (N_queries, N_corpus)."""

    def export_onnx(self, output_path: Path) -> Path | None:
        """Optional ONNX export."""
        _ = output_path
        return None

    def get_onnx_paths(self) -> list[Path]:
        """Return generated ONNX model paths to log into MLflow."""
        return []

    def get_model_size_mb(self) -> float:
        """Return model disk footprint in MB."""
        return 0.0
