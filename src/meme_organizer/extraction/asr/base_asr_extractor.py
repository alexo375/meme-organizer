"""Base contract and result schema for ASR extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class AsrResult:
    """Structured output from an ASR extractor."""

    text: str
    segments: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


class BaseAsrExtractor(ABC):
    """Abstract interface for on-device ASR engines."""

    name: str = "base_asr"

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> AsrResult:
        """Transcribe 1D float32 PCM audio into text and segments."""

    @abstractmethod
    def model_size_mb(self) -> float:
        """Return total model weight size on disk in megabytes."""
