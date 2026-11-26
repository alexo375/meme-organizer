"""Оркестратор пайплайна оценки ретриверов (Template Method)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from meme_organizer.evaluation.metrics.experiment_tracker import ExperimentTracker
from meme_organizer.evaluation.pipeline.evaluation_dataset_preparer import EvaluationDatasetPreparer
from meme_organizer.evaluation.pipeline.retrieval_evaluator import RetrievalEvaluator

if TYPE_CHECKING:
    from pathlib import Path

    from meme_organizer.domain.base_retriever import BaseRetriever
    from meme_organizer.domain.ranking_metrics import RankingMetrics

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Шаблонный пайплайн оценки: подготовка данных -> обучение -> оценка -> логирование."""

    def __init__(
        self,
        tracker: ExperimentTracker | None = None,
        preparer: EvaluationDatasetPreparer | None = None,
        evaluator: RetrievalEvaluator | None = None,
    ) -> None:
        """Инициализирует пайплайн с явным внедрением зависимостей."""
        self.tracker = tracker or ExperimentTracker()
        self.preparer = preparer or EvaluationDatasetPreparer()
        self.evaluator = evaluator or RetrievalEvaluator()

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
        """Запускает полный цикл оценки ретривера с замером времени и логированием в MLflow.

        Вход: ретривер, пути к датасету и сплиту, имя сплита, флаг обучения, имя запуска и теги.
        Выход: кортеж (RankingMetrics, run_id).
        """
        t_total_start = time.perf_counter()
        logger.info("Старт эксперимента: %s (сплит: %s)", retriever.model_name, split_name)

        data = self.preparer.prepare(gold_dataset_path, split_path, split_name)
        effective_run_name = run_name or f"{retriever.model_name}_{split_name}"
        params: dict[str, Any] = {
            "model_name": retriever.model_name,
            "split_name": split_name,
            "split_version": data.split_version,
            "corpus_size": len(data.corpus_docs),
            "num_queries": len(data.queries),
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
                logger.info(
                    "Обучение/калибровка ретривера на %d train / %d val парах...",
                    len(data.train_pairs),
                    len(data.val_pairs),
                )
                t_train_start = time.perf_counter()
                retriever.fit(
                    train_pairs=data.train_pairs,
                    corpus_docs=data.corpus_docs,
                    val_pairs=data.val_pairs,
                    tracker=self.tracker,
                )
                training_time_sec = round(time.perf_counter() - t_train_start, 3)
                logger.info("Обучение завершено за %.2f сек", training_time_sec)

            eval_res = self.evaluator.evaluate(retriever, data)
            logger.info(
                "Метрики [%s]: Recall@1=%.4f | Recall@5=%.4f | MRR@5=%.4f | NDCG@5=%.4f",
                split_name,
                eval_res.metrics.recall_at_1,
                eval_res.metrics.recall_at_5,
                eval_res.metrics.mrr_at_5,
                eval_res.metrics.ndcg_at_5,
            )

            timing_metrics = {
                "total_run_time_sec": round(time.perf_counter() - t_total_start, 3),
                "training_time_sec": training_time_sec,
                "eval_inference_time_sec": eval_res.eval_inference_time_sec,
                "eval_throughput_qps": eval_res.throughput_qps,
            }

            self.tracker.log_run_results(
                params=params,
                metrics=eval_res.metrics,
                profile=eval_res.profile,
                worst_queries=eval_res.worst_queries,
                timing_metrics=timing_metrics,
            )

            for onnx_file in retriever.get_onnx_paths():
                self.tracker.log_model_artifact(onnx_file, artifact_path_in_run="onnx")
            for art_name, art_dir in retriever.get_artifact_dirs().items():
                self.tracker.log_artifact_dir(art_dir, artifact_path_in_run=art_name)

        return eval_res.metrics, run_id
