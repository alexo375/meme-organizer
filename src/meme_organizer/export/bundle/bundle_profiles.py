"""Реестр профилей мобильных бандлов, загружаемых из JSON-конфигураций."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ModelBundleProfile(BaseModel):
    """Параметры профиля модели для экспорта мобильного ONNX-бандла."""

    profile_id: str
    model_name: str
    base_hf_id: str
    embedding_dim: int
    max_length: int
    query_prefix: str = ""
    passage_prefix: str = ""
    dense_weight: float
    bm25_weight: float
    epochs: int = 50
    patience: int = 10
    lr: float = 2e-4
    batch_size: int = 32
    lora_r: int = 8
    lora_alpha: int = 16
    seed: int = 42
    extra_metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def load_all(cls, profiles_dir: Path | None = None) -> dict[str, ModelBundleProfile]:
        """Загружает и валидирует все JSON-профили из директории ресурсов."""
        target_dir = profiles_dir or Path(__file__).resolve().parents[2] / "resources" / "profiles"
        profiles: dict[str, ModelBundleProfile] = {}
        for json_path in sorted(target_dir.glob("*.json")):
            data = json.loads(json_path.read_text(encoding="utf-8"))
            profile = cls.model_validate(data)
            profiles[profile.profile_id] = profile
        return profiles


PROFILES: dict[str, ModelBundleProfile] = ModelBundleProfile.load_all()
