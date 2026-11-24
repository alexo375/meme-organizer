"""Компактный логгер для MLflow (строго по архитектурному манифесту)."""

from __future__ import annotations

import csv
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow

if TYPE_CHECKING:
    from meme_organizer.evaluation.metrics import RankingMetrics
    from meme_organizer.retrieval.profiler import ProfilingResult

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Логирует в MLflow строго 4 сущности: конфиг, метрики, профиль и worst queries."""

    def __init__(
        self,
        experiment_name: str = "meme-organizer-retrieval",
        tracking_uri: str | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or "http://localhost:5000"
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)

    def log_run(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: RankingMetrics,
        profile: ProfilingResult,
        worst_queries: list[dict[str, Any]],
        tags: dict[str, str] | None = None,
    ) -> str:
        """Создает ран и сохраняет необходимые артефакты."""
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            # 1. Параметры/Конфиг
            clean_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(clean_params)

            # 2. Метрики качества
            mlflow.log_metrics(
                {
                    "mrr_at_5": metrics.mrr_at_5,
                    "recall_at_1": metrics.recall_at_1,
                    "recall_at_5": metrics.recall_at_5,
                    "ndcg_at_5": metrics.ndcg_at_5,
                    "total_queries": float(metrics.total_queries),
                    # 3. Профиль On-Device
                    "cpu_latency_ms": profile.cpu_latency_ms,
                    "cpu_latency_p95_ms": profile.cpu_latency_p95_ms,
                    "model_size_mb": profile.model_size_mb,
                }
            )

            # Теги
            if tags:
                mlflow.set_tags(tags)

            # 4. Таблица ошибок: Top-15 Worst Queries (CSV)
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

            logger.info("Ран успешно сохранен в MLflow [run_id=%s, run_name=%s]", run_id, run_name)
            return str(run_id)
