"""Профилировщик задержки на CPU и размера модели."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from meme_organizer.domain.base_retriever import BaseRetriever


@dataclass
class ProfilingResult:
    """Результат замера производительности модели для запуска на устройстве."""

    cpu_latency_ms: float
    cpu_latency_p95_ms: float
    model_size_mb: float


class ModelProfiler:
    """Измеряет задержку на CPU и размер моделей поиска на диске."""

    @staticmethod
    def benchmark(
        retriever: BaseRetriever,
        sample_query: str = "когда друг несет чушь",
        corpus_repr: Any = None,
        warmup_runs: int = 3,
        benchmark_runs: int = 15,
    ) -> ProfilingResult:
        """Измеряет среднюю и p95 задержку на один запрос (batch=1) и размер модели на диске."""
        # 1. Warm-up
        for _ in range(warmup_runs):
            q_repr = retriever.encode_queries([sample_query])
            c_repr = corpus_repr if corpus_repr is not None else q_repr
            _ = retriever.predict_scores(q_repr, c_repr)

        # 2. Timing
        latencies_ms: list[float] = []
        for _ in range(benchmark_runs):
            t_start = time.perf_counter()
            q_repr = retriever.encode_queries([sample_query])
            c_repr = corpus_repr if corpus_repr is not None else q_repr
            _ = retriever.predict_scores(q_repr, c_repr)
            latencies_ms.append((time.perf_counter() - t_start) * 1000.0)

        cpu_latency_ms = round(float(np.mean(latencies_ms)), 2)
        cpu_latency_p95_ms = round(float(np.percentile(latencies_ms, 95)), 2)
        model_size_mb = round(retriever.get_model_size_mb(), 2)

        return ProfilingResult(
            cpu_latency_ms=cpu_latency_ms,
            cpu_latency_p95_ms=cpu_latency_p95_ms,
            model_size_mb=model_size_mb,
        )
