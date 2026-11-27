"""Базовый контракт для доменных адаптеров экспериментов."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from meme_organizer.evaluation.metrics.experiment_tracker import ExperimentTracker


class BaseExperimentAdapter(ABC):
    """Abstract adapter defining the lifecycle hooks for any ML experiment domain."""

    run_name: str = "experiment"
    tags: dict[str, str]

    def __init__(self, run_name: str, tags: dict[str, str] | None = None) -> None:
        """Initialize adapter with MLflow run name and optional tags."""
        self.run_name = run_name
        self.tags = tags or {}

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return hyperparameters and configuration dictionary to log."""

    @abstractmethod
    def prepare(self) -> None:
        """Load weights, datasets, or build indices before execution."""

    @abstractmethod
    def execute(self, tracker: ExperimentTracker | None = None) -> None:
        """Run training/inference and record timing and predictions."""

    @abstractmethod
    def compute_quality_metrics(self) -> dict[str, float]:
        """Return domain-specific quality metrics (e.g., CER/WER or MRR/Recall)."""

    @abstractmethod
    def compute_performance_metrics(self) -> dict[str, float]:
        """Return latency, throughput, and model size metrics."""

    def get_error_table(self) -> list[dict[str, Any]]:
        """Return top-K worst error rows for CSV artifact logging."""
        return []

    def get_model_artifacts(self) -> dict[str, Path]:
        """Return mapping of artifact label to file path (e.g. raw/quantized ONNX)."""
        return {}
