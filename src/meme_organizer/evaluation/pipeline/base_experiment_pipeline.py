"""Унифицированный шаблонный пайплайн для запуска и логирования доменных экспериментов."""

from __future__ import annotations

import csv
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow

from meme_organizer.config import MlflowConfig

if TYPE_CHECKING:
    from meme_organizer.domain.base_experiment_adapter import BaseExperimentAdapter
    from meme_organizer.evaluation.metrics.experiment_tracker import ExperimentTracker


class BaseExperimentPipeline:
    """Оркестрирует цикл prepare -> execute -> metrics -> artifacts для любого адаптера."""

    def __init__(self, tracker: ExperimentTracker, adapter: BaseExperimentAdapter) -> None:
        """Привязывает ExperimentTracker и доменный BaseExperimentAdapter."""
        self.tracker = tracker
        self.adapter = adapter

    def run(self) -> dict[str, Any]:
        """Выполняет полный жизненный цикл эксперимента и логирует все группы метрик в MLflow."""
        t0 = time.perf_counter()
        with self.tracker.start_run(run_name=self.adapter.run_name, tags=self.adapter.tags) as run_id:
            mlflow.log_params(self.adapter.get_params())
            self.adapter.prepare()
            self.adapter.execute(tracker=self.tracker)

            quality = self.adapter.compute_quality_metrics()
            performance = self.adapter.compute_performance_metrics()
            performance["total_pipeline_time_sec"] = round(time.perf_counter() - t0, 3)

            if quality:
                mlflow.log_metrics(quality)
            if performance:
                mlflow.log_metrics(performance)

            self._log_error_table(self.adapter.get_error_table()[: MlflowConfig.top_k_worst_errors])
            self._log_model_artifacts(self.adapter.get_model_artifacts())

        return {
            "run_id": run_id,
            "quality": quality,
            "performance": performance,
        }

    @staticmethod
    def _log_error_table(rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "top15_errors.csv"
            fieldnames = list(rows[0].keys())
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            mlflow.log_artifact(str(csv_path), artifact_path="errors")

    @staticmethod
    def _log_model_artifacts(artifacts: dict[str, Path]) -> None:
        for path in artifacts.values():
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="model")
