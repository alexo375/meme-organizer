"""Оценка качества и производительности ретривера на подготовленном сплите."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from meme_organizer.domain.ranking_metrics import RankingMetrics
from meme_organizer.evaluation.metrics.error_analyzer import ErrorAnalyzer
from meme_organizer.evaluation.metrics.model_profiler import ModelProfiler, ProfilingResult

if TYPE_CHECKING:
    from meme_organizer.domain.base_retriever import BaseRetriever
    from meme_organizer.evaluation.pipeline.evaluation_dataset_preparer import EvaluationSplitData


@dataclass(frozen=True)
class EvaluationResult:
    """Результаты оценки качества, ошибок и производительности ретривера."""

    metrics: RankingMetrics
    profile: ProfilingResult
    worst_queries: list[dict[str, Any]]
    eval_inference_time_sec: float
    throughput_qps: float


class RetrievalEvaluator:
    """Вычислитель метрик качества, ошибок и профиля производительности."""

    @staticmethod
    def evaluate(retriever: BaseRetriever, data: EvaluationSplitData) -> EvaluationResult:
        """Кодирует корпус и запросы, вычисляет метрики, ошибки и профиль модели.

        Вход: инициализированный ретривер и подготовленные данные сплита.
        Выход: объект EvaluationResult.
        """
        t_eval_start = time.perf_counter()
        corpus_repr = retriever.encode_corpus(data.corpus_docs)
        query_repr = retriever.encode_queries(data.queries)
        scores_matrix = retriever.predict_scores(query_repr, corpus_repr)
        eval_inference_time_sec = round(time.perf_counter() - t_eval_start, 3)

        throughput_qps = round(len(data.queries) / max(eval_inference_time_sec, 0.001), 2)
        metrics = RankingMetrics.calculate(scores_matrix, data.ground_truth_indices, top_k=5)
        worst_queries = ErrorAnalyzer.find_worst_queries(
            scores_matrix,
            data.ground_truth_indices,
            data.queries,
            data.corpus_hashes,
            top_k=15,
        )

        sample_q = data.queries[0] if data.queries else "тестовый запрос"
        profile = ModelProfiler.benchmark(retriever, sample_query=sample_q, corpus_repr=corpus_repr)

        return EvaluationResult(
            metrics=metrics,
            profile=profile,
            worst_queries=worst_queries,
            eval_inference_time_sec=eval_inference_time_sec,
            throughput_qps=throughput_qps,
        )
