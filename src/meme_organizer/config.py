"""Настройки конфигурации и SSoT-константы Meme Organizer."""

from dataclasses import dataclass
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class MlflowConfig:
    """SSoT-конфигурация экспериментов MLflow."""

    default_tracking_uri: str = "http://localhost:5000"
    experiment_text: str = "meme-organizer-text"
    experiment_ocr: str = "meme-organizer-ocr"
    experiment_asr: str = "meme-organizer-asr"
    experiment_visual: str = "meme-organizer-visual"
    experiment_fusion: str = "meme-organizer-fusion"
    top_k_worst_errors: int = 15


MLFLOW_EXPERIMENT_TEXT = MlflowConfig.experiment_text
MLFLOW_EXPERIMENT_OCR = MlflowConfig.experiment_ocr
MLFLOW_EXPERIMENT_ASR = MlflowConfig.experiment_asr
MLFLOW_EXPERIMENT_VISUAL = MlflowConfig.experiment_visual
MLFLOW_EXPERIMENT_FUSION = MlflowConfig.experiment_fusion
CYRILLIC_OCR_MODEL_PATH = Path("models/ocr/cyrillic_ppocrv5/inference.onnx")
CYRILLIC_OCR_DICT_PATH = Path("models/ocr/cyrillic_ppocrv5/dict.txt")


class AppConfig(BaseSettings):
    """Глобальная конфигурация, загружаемая из переменных окружения или значений по умолчанию."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mlflow_tracking_uri: str = Field(default=MlflowConfig.default_tracking_uri, alias="MLFLOW_TRACKING_URI")
    experiment_name: str = Field(default=MlflowConfig.experiment_fusion, alias="MLFLOW_EXPERIMENT_NAME")
    gold_dataset_path: Path = Field(default=Path("data/gold/gold_dataset.jsonl"))
    splits_path: Path = Field(default=Path("data/splits/splits_v1.json"))
    cache_dir: Path = Field(default=Path(".cache"))
    random_seed: int = Field(default=42, alias="RANDOM_SEED")
