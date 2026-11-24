"""Скрипт полного сброса MLflow: очистка PostgreSQL таблиц, моделей, алиасов и артефактов."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def reset_mlflow(tracking_uri: str | None = None) -> None:
    """Полностью и перманентно очищает MLflow через SQL Truncate и пересоздает эксперимент."""
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    logger.info("Сброс базы данных MLflow в PostgreSQL контейнере...")

    sql_cmd = (
        "TRUNCATE TABLE metrics, params, tags, latest_metrics, model_versions, registered_models, runs CASCADE; "
        "DELETE FROM experiments WHERE experiment_id != '0'; "
        "UPDATE experiments SET lifecycle_stage='active' WHERE experiment_id='0';"
    )

    try:
        docker_bin = shutil.which("docker") or "docker"
        subprocess.run(  # noqa: S603
            [
                docker_bin,
                "compose",
                "exec",
                "-T",
                "postgres",
                "psql",
                "-U",
                "mlflow",
                "-d",
                "mlflow_db",
                "-c",
                sql_cmd,
            ],
            check=True,
            capture_output=True,
        )
        logger.info("Таблицы базы данных успешно очищены")
    except Exception as e:
        logger.warning("Не удалось выполнить SQL очистку: %s", e)

    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    exp_name = "meme-organizer-retrieval"
    try:
        new_id = client.create_experiment(exp_name)
        logger.info("Создан чистый эксперимент: %s (ID=%s)", exp_name, new_id)
    except Exception:
        existing = client.get_experiment_by_name(exp_name)
        if existing:
            logger.info("Эксперимент %s активен (ID=%s)", exp_name, existing.experiment_id)

    # Очистка артефактов внутри Docker контейнера
    try:
        docker_bin = shutil.which("docker") or "docker"
        subprocess.run(  # noqa: S603
            [docker_bin, "compose", "exec", "-T", "mlflow", "sh", "-c", "rm -rf /mlartifacts/*"],
            check=False,
            capture_output=True,
        )
        logger.info("Хранилище /mlartifacts в контейнере успешно очищено")
    except Exception as e:
        logger.warning("Не удалось очистить /mlartifacts в контейнере: %s", e)

    # Очистка локальной папки кэша
    local_cache = Path(".cache")
    if local_cache.exists():
        shutil.rmtree(local_cache, ignore_errors=True)

    logger.info("MLflow готов к новым запускам!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    reset_mlflow()
