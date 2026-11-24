"""Шаблонный пайплайн Retrieval (Template Method)."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from meme_organizer.evaluation.metrics import RankingMetrics, RetrievalEvaluator
from meme_organizer.retrieval.profiler import ModelProfiler
from meme_organizer.retrieval.tracking import ExperimentTracker

if TYPE_CHECKING:
    from pathlib import Path

    from meme_organizer.retrieval.base import BaseRetrieverAdapter

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Оркестратор эксперимента: запускает сквозной цикл, считает метрики и логирует в MLflow."""

    def __init__(self, tracker: ExperimentTracker | None = None) -> None:
        self.tracker = tracker or ExperimentTracker()

    def run(
        self,
        adapter: BaseRetrieverAdapter,
        gold_dataset_path: Path,
        split_path: Path,
        split_name: str = "test",  # "val" или "test"
        train_mode: bool = False,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> tuple[RankingMetrics, str]:
        """Жесткий шаблонный цикл эксперимента без возможности забыть шаг или метрику."""
        logger.info("Старт эксперимента для модели: %s (сплит: %s)", adapter.model_name, split_name)

        # 1. Загрузка данных
        with open(gold_dataset_path, encoding="utf-8") as f:
            corpus_docs = [json.loads(line) for line in f if line.strip()]

        with open(split_path, encoding="utf-8") as f:
            split_info = json.load(f)

        corpus_hash_to_idx = {doc["file_hash"]: i for i, doc in enumerate(corpus_docs)}
        corpus_hashes = [doc["file_hash"] for doc in corpus_docs]

        pairs: list[dict[str, str]] = split_info.get(f"{split_name}_pairs", [])
        if not pairs:
            raise ValueError(f"Сплит '{split_name}_pairs' пуст в {split_path}")

        queries = [p["query"] for p in pairs]
        ground_truth_indices = [corpus_hash_to_idx[p["file_hash"]] for p in pairs]

        # 2. Инициализация и опциональное дообучение
        adapter.load_or_build()
        if train_mode:
            train_pairs = split_info.get("train_pairs", [])
            logger.info("Дообучение адаптера на %d парах...", len(train_pairs))
            adapter.fit(train_pairs)

        # 3. Кодирование корпуса и запросов
        logger.info("Кодирование корпуса (%d документов)...", len(corpus_docs))
        corpus_repr = adapter.encode_corpus(corpus_docs)

        logger.info("Кодирование запросов (%d запросов)...", len(queries))
        query_repr = adapter.encode_queries(queries)

        # 4. Расчет скоров
        logger.info("Расчет матрицы предсказаний...")
        scores_matrix = adapter.predict_scores(query_repr, corpus_repr)

        # 5. Оценка качества
        metrics, worst_queries = RetrievalEvaluator.compute_metrics(
            scores_matrix=scores_matrix,
            ground_truth_indices=ground_truth_indices,
            queries=queries,
            corpus_hashes=corpus_hashes,
            top_k=5,
        )

        logger.info(
            "Метрики [%s]: Recall@1=%.4f | Recall@5=%.4f | MRR@5=%.4f | NDCG@5=%.4f",
            split_name,
            metrics.recall_at_1,
            metrics.recall_at_5,
            metrics.mrr_at_5,
            metrics.ndcg_at_5,
        )

        # 6. Профилирование задержки и размера
        sample_q = queries[0] if queries else "тестовый запрос"
        profile = ModelProfiler.benchmark(adapter, sample_query=sample_q)
        logger.info("Профиль: CPU latency=%.2f ms | Size=%.2f MB", profile.cpu_latency_ms, profile.model_size_mb)

        # 7. Автоматическое логирование в MLflow
        effective_run_name = run_name or f"{adapter.model_name}_{split_name}"
        params: dict[str, Any] = {
            "model_name": adapter.model_name,
            "split_name": split_name,
            "split_version": split_info.get("version", "v1"),
            "corpus_size": len(corpus_docs),
            "num_queries": len(queries),
            **adapter.params,
        }

        all_tags = {"domain": "retrieval", "split": split_name}
        if tags:
            all_tags.update(tags)

        run_id = self.tracker.log_run(
            run_name=effective_run_name,
            params=params,
            metrics=metrics,
            profile=profile,
            worst_queries=worst_queries,
            tags=all_tags,
        )

        return metrics, run_id
