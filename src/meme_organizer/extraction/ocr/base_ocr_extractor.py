"""Base contract and result schema for OCR extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class OcrResult:
    """Structured output from an OCR extractor."""

    text: str
    lines: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    latency_ms: float = 0.0


class BaseOcrExtractor(ABC):
    """Abstract interface for on-device and baseline OCR engines."""

    name: str = "base"

    @abstractmethod
    def extract_text(self, image: np.ndarray) -> OcrResult:
        """Extract text lines and confidences from an RGB/BGR image array."""

    @abstractmethod
    def model_size_mb(self) -> float:
        """Return total model weight size on disk in megabytes."""
