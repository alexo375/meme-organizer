"""Базовый контракт для чистых текстовых энкодеров (без логики поиска и скоринга)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


class BaseTextEncoder(ABC):
    """Abstract interface for text representation models."""

    name: str = "base_text_encoder"

    @abstractmethod
    def load_or_build(self) -> None:
        """Load encoder weights or initialize vocabulary."""

    @abstractmethod
    def encode_corpus(self, corpus_docs: list[dict[str, Any]]) -> np.ndarray | Any:
        """Encode corpus documents into embeddings or sparse representations."""

    @abstractmethod
    def encode_queries(self, queries: list[str]) -> np.ndarray | Any:
        """Encode query strings into embeddings or sparse representations."""

    def export_onnx(self, output_path: Path) -> Path:
        """Export encoder to ONNX format if supported."""
        raise NotImplementedError(f"{self.name} does not support ONNX export.")
