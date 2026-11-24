"""Профилировщик производительности On-Device (задержка CPU и размер артефакта)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from meme_organizer.retrieval.base import BaseRetrieverAdapter


@dataclass(frozen=True)
class ProfilingResult:
    """Результаты замера скорости и веса модели."""

    cpu_latency_ms: float
    cpu_latency_p95_ms: float
    model_size_mb: float


class ModelProfiler:
    """Замеряет latency на CPU и размер артефактов."""

    @staticmethod
    def benchmark(
        adapter: BaseRetrieverAdapter,
        sample_query: str = "когда ругаются матом",
        num_warmup: int = 5,
        num_runs: int = 30,
    ) -> ProfilingResult:
        """Измеряет время обработки одного запроса на CPU."""
        # Warmup
        for _ in range(num_warmup):
            _ = adapter.encode_queries([sample_query])

        durations: list[float] = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = adapter.encode_queries([sample_query])
            durations.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = float(np.mean(durations))
        p95_ms = float(np.percentile(durations, 95))
        size_mb = adapter.get_model_size_mb()

        return ProfilingResult(
            cpu_latency_ms=round(mean_ms, 2),
            cpu_latency_p95_ms=round(p95_ms, 2),
            model_size_mb=round(size_mb, 2),
        )
