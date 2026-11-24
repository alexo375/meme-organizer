"""CPU latency and model size profiler."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from meme_organizer.core.contracts import BaseRetriever


@dataclass
class ProfilingResult:
    """Benchmark profiling result for on-device deployment."""

    cpu_latency_ms: float
    cpu_latency_p95_ms: float
    model_size_mb: float


class ModelProfiler:
    """Profiles CPU latency and memory footprint of retrieval models."""

    @staticmethod
    def benchmark(
        retriever: BaseRetriever,
        sample_query: str = "когда друг несет чушь",
        corpus_repr: Any = None,
        warmup_runs: int = 3,
        benchmark_runs: int = 15,
    ) -> ProfilingResult:
        """Measure mean/p95 latency per single query (batch=1) and model disk size."""
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
