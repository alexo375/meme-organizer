"""Компактный логгер для MLflow с поддержкой замеров времени и по-эпохорных метрик."""

from __future__ import annotations

import csv
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow

if TYPE_CHECKING:
    from collections.abc import Generator

    from meme_organizer.evaluation.metrics import RankingMetrics
    from meme_organizer.retrieval.profiler import ProfilingResult

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Логирует в MLflow параметры, метрики, профили задержки, тайминги и worst queries."""

    def __init__(
        self,
        experiment_name: str = "meme-organizer-retrieval",
        tracking_uri: str | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or "http://localhost:5000"
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)

    def log_epoch_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
        """Логирует промежуточные метрики эпохи обучения для построения графиков."""
        active_run = mlflow.active_run()
        if active_run:
            mlflow.log_metrics(metrics, step=epoch)

    @contextmanager
    def start_run(self, run_name: str, tags: dict[str, str] | None = None) -> Generator[str]:
        """Открывает активный контекст рана в MLflow."""
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
        """Сохраняет параметры, метрики и артефакты в уже активный ран."""
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
