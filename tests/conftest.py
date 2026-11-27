"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent test runs from logging to the live MLflow server."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
