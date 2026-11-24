from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LabelingConfig(BaseSettings):
    """Конфигурация пайплайна авторазметки bronze -> silver, загружаемая из .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: SecretStr = Field(default=SecretStr(""), alias="GEMINI_API_KEY")
    model_name: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    rpm: float = Field(default=8.0, gt=0.0, alias="LABELING_RPM")
    max_retries: int = Field(default=4, ge=0, le=10)
    base_backoff_sec: float = Field(default=4.0, gt=0.0)
    random_seed: int = Field(default=42, alias="RANDOM_SEED")

    bronze_dir: Path = Field(default=Path("data/bronze"))
    silver_dir: Path = Field(default=Path("data/silver"))
    gold_dir: Path = Field(default=Path("data/gold"))
    rename_in_place: bool = Field(default=False)

    image_max_side: int = Field(default=1280, ge=256)
    image_max_mb: float = Field(default=4.0, gt=0.0)
    video_max_mb: float = Field(default=12.0, gt=0.0)
    video_target_height: int = Field(default=480, ge=240)
    inline_max_mb: float = Field(default=10.0, gt=0.0)
