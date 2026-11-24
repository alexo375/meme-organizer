"""Template Method orchestrator running retrieval evaluation end-to-end."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from meme_organizer.core.profiler import ModelProfiler
from meme_organizer.core.tracker import ExperimentTracker
from meme_organizer.dataset.loader import DatasetLoader
from meme_organizer.metrics.error_analysis import find_worst_queries
from meme_organizer.metrics.ranking import RankingMetrics, calculate_ranking_metrics

if TYPE_CHECKING:
    from pathlib import Path

    from meme_organizer.core.contracts import BaseRetriever

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Rigid evaluation pipeline template: load -> fit -> encode -> score -> metrics -> mlflow."""

    def __init__(self, tracker: ExperimentTracker | None = None) -> None:
        self.tracker = tracker or ExperimentTracker()

    def run(
        self,
        retriever: BaseRetriever,
        gold_dataset_path: Path,
        split_path: Path,
        split_name: str = "test",
        train_mode: bool = False,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> tuple[RankingMetrics, str]:
        """Execute the end-to-end evaluation cycle with precise timing and logging."""
        t_total_start = time.perf_counter()
        logger.info("Старт эксперимента: %s (сплит: %s)", retriever.model_name, split_name)

        corpus_docs = DatasetLoader.load_corpus(gold_dataset_path)
        split_info = DatasetLoader.load_split_info(split_path)

        corpus_hash_to_idx = {doc["file_hash"]: i for i, doc in enumerate(corpus_docs)}
        corpus_hashes = [doc["file_hash"] for doc in corpus_docs]

        pairs = split_info.get(f"{split_name}_pairs", [])
        if not pairs:
            raise ValueError(f"Сплит '{split_name}_pairs' пуст в {split_path}")

        queries = [p["query"] for p in pairs]
        ground_truth_indices = [corpus_hash_to_idx[p["file_hash"]] for p in pairs]

        effective_run_name = run_name or f"{retriever.model_name}_{split_name}"
        params: dict[str, Any] = {
            "model_name": retriever.model_name,
            "split_name": split_name,
            "split_version": split_info.get("version", "v1"),
            "corpus_size": len(corpus_docs),
            "num_queries": len(queries),
            "train_mode": train_mode,
            **retriever.params,
        }

        all_tags = {"domain": "retrieval", "split": split_name}
        if tags:
            all_tags.update(tags)

        with self.tracker.start_run(run_name=effective_run_name, tags=all_tags) as run_id:
            retriever.load_or_build()

            training_time_sec = 0.0
            if train_mode or retriever.model_name.startswith("hybrid-optuna"):
                train_pairs = split_info.get("train_pairs", [])
                val_pairs = split_info.get("val_pairs", [])
                logger.info(
                    "Обучение/калибровка ретривера на %d train / %d val парах...", len(train_pairs), len(val_pairs)
                )
                t_train_start = time.perf_counter()
                retriever.fit(
                    train_pairs=train_pairs,
                    corpus_docs=corpus_docs,
                    val_pairs=val_pairs,
                    tracker=self.tracker,
                )
                training_time_sec = round(time.perf_counter() - t_train_start, 3)
                logger.info("Обучение завершено за %.2f сек", training_time_sec)

            # Оценка инференса
            t_eval_start = time.perf_counter()
            corpus_repr = retriever.encode_corpus(corpus_docs)
            query_repr = retriever.encode_queries(queries)
            scores_matrix = retriever.predict_scores(query_repr, corpus_repr)
            eval_inference_time_sec = round(time.perf_counter() - t_eval_start, 3)

            throughput_qps = round(len(queries) / max(eval_inference_time_sec, 0.001), 2)
            metrics = calculate_ranking_metrics(scores_matrix, ground_truth_indices, top_k=5)
            worst_queries = find_worst_queries(scores_matrix, ground_truth_indices, queries, corpus_hashes, top_k=15)

            logger.info(
                "Метрики [%s]: Recall@1=%.4f | Recall@5=%.4f | MRR@5=%.4f | NDCG@5=%.4f",
                split_name,
                metrics.recall_at_1,
                metrics.recall_at_5,
                metrics.mrr_at_5,
                metrics.ndcg_at_5,
            )

            sample_q = queries[0] if queries else "тестовый запрос"
            profile = ModelProfiler.benchmark(retriever, sample_query=sample_q, corpus_repr=corpus_repr)

            total_run_time_sec = round(time.perf_counter() - t_total_start, 3)
            timing_metrics = {
                "total_run_time_sec": total_run_time_sec,
                "training_time_sec": training_time_sec,
                "eval_inference_time_sec": eval_inference_time_sec,
                "eval_throughput_qps": throughput_qps,
            }

            self.tracker.log_run_results(
                params=params,
                metrics=metrics,
                profile=profile,
                worst_queries=worst_queries,
                timing_metrics=timing_metrics,
            )

            for onnx_file in retriever.get_onnx_paths():
                self.tracker.log_model_artifact(onnx_file, artifact_path_in_run="onnx")

        return metrics, run_id
