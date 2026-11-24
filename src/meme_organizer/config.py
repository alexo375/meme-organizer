"""Configuration settings for Meme Organizer retrieval pipeline."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Global configuration loaded from environment or defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mlflow_tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    experiment_name: str = Field(default="meme-organizer-retrieval", alias="MLFLOW_EXPERIMENT_NAME")
    gold_dataset_path: Path = Field(default=Path("data/gold/gold_dataset.jsonl"))
    splits_path: Path = Field(default=Path("data/splits/splits_v1.json"))
    cache_dir: Path = Field(default=Path(".cache"))
    random_seed: int = Field(default=42, alias="RANDOM_SEED")
