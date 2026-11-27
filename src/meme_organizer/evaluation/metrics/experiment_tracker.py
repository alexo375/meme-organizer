"""Трекер экспериментов MLflow для логирования конфигураций, метрик, профилирования и артефактов."""

from __future__ import annotations

import csv
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow

from meme_organizer.config import MlflowConfig
from meme_organizer.evaluation.pipeline.base_experiment_pipeline import BaseExperimentPipeline

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import TracebackType

    from meme_organizer.domain.base_experiment_adapter import BaseExperimentAdapter
    from meme_organizer.domain.ranking_metrics import RankingMetrics
    from meme_organizer.evaluation.metrics.model_profiler import ProfilingResult

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Инкапсулирует логирование экспериментов в MLflow."""

    def __init__(
        self,
        experiment_name: str = MlflowConfig.experiment_fusion,
        tracking_uri: str | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or MlflowConfig.default_tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)

    def __enter__(self) -> ExperimentTracker:
        """Вход в контекстный менеджер, возвращает настроенный трекер."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Корректный выход из контекстного менеджера."""

    def create_pipeline(self, adapter: BaseExperimentAdapter) -> BaseExperimentPipeline:
        """Создаёт BaseExperimentPipeline, привязанный к текущему трекеру и адаптеру."""
        return BaseExperimentPipeline(tracker=self, adapter=adapter)

    def log_epoch_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
        """Логирует метрики обучения для заданной эпохи."""
        if mlflow.active_run():
            mlflow.log_metrics(metrics, step=epoch)

    def log_optuna_trial(self, trial_num: int, trial_mrr: float, best_mrr: float, alpha: float) -> None:
        """Логирует прогресс испытания оптимизации гиперпараметров Optuna."""
        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "optuna_trial_mrr_at_5": trial_mrr,
                    "optuna_best_mrr_at_5": best_mrr,
                    "optuna_trial_alpha": alpha,
                },
                step=trial_num,
            )

    @contextmanager
    def start_run(self, run_name: str, tags: dict[str, str] | None = None) -> Generator[str]:
        """Создает активный запуск MLflow в контекстном менеджере и возвращает его run_id."""
        with mlflow.start_run(run_name=run_name) as run:
            if tags:
                mlflow.set_tags(tags)
            yield str(run.info.run_id)

    def log_run_results(
        self,
        params: dict[str, Any],
        metrics: RankingMetrics,
        profile: ProfilingResult,
        worst_queries: list[dict[str, Any]],
        timing_metrics: dict[str, float] | None = None,
    ) -> None:
        """Логирует параметры, метрики ранжирования, задержку и CSV с худшими запросами."""
        clean_params = {k: str(v) for k, v in params.items()}
        mlflow.log_params(clean_params)

        all_metrics: dict[str, float] = {
            "mrr_at_5": metrics.mrr_at_5,
            "recall_at_1": metrics.recall_at_1,
            "recall_at_5": metrics.recall_at_5,
            "ndcg_at_5": metrics.ndcg_at_5,
            "total_queries": float(metrics.total_queries),
            "cpu_latency_ms": profile.cpu_latency_ms,
            "cpu_latency_p95_ms": profile.cpu_latency_p95_ms,
            "model_size_mb": profile.model_size_mb,
        }

        if timing_metrics:
            all_metrics.update(timing_metrics)

        mlflow.log_metrics(all_metrics)

        if worst_queries:
            with tempfile.TemporaryDirectory() as tmp_dir:
                csv_path = Path(tmp_dir) / "worst_queries.csv"
                with open(csv_path, "w", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["query", "target_hash", "target_rank", "predicted_top"],
                    )
                    writer.writeheader()
                    for row in worst_queries:
                        writer.writerow(
                            {
                                "query": row["query"],
                                "target_hash": row["target_hash"],
                                "target_rank": row["target_rank"],
                                "predicted_top": ";".join(row.get("predicted_top", [])),
                            }
                        )
                mlflow.log_artifact(str(csv_path), artifact_path="error_analysis")

    def log_model_artifact(self, artifact_path: Path, artifact_path_in_run: str = "onnx") -> None:
        """Загружает сгенерированную ONNX-модель в хранилище артефактов запуска MLflow."""
        if artifact_path.exists():
            logger.info("Загрузка модели %s в MLflow артефакты (%s)...", artifact_path.name, artifact_path_in_run)
            mlflow.log_artifact(str(artifact_path), artifact_path=artifact_path_in_run)
            data_file = artifact_path.parent / f"{artifact_path.name}.data"
            if data_file.exists():
                mlflow.log_artifact(str(data_file), artifact_path=artifact_path_in_run)

    def log_artifact_dir(self, local_dir: Path, artifact_path_in_run: str = "hf_model") -> None:
        """Загружает директорию артефактов модели (например, FP32-веса HuggingFace) в MLflow."""
        if local_dir.exists() and local_dir.is_dir():
            logger.info("Загрузка директории %s в MLflow артефакты (%s)...", local_dir.name, artifact_path_in_run)
            mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path_in_run)
