from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class MediaType(StrEnum):
    """Поддерживаемые типы медиафайлов мемов."""

    IMAGE = "image"
    GIF = "gif"
    VIDEO = "video"


class FailureReason(StrEnum):
    """Классификация причин неуспешной разметки файла."""

    SAFETY_BLOCKED = "safety_blocked"
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    MEDIA_ERROR = "media_error"


class MediaFileInfo(BaseModel):
    """Метаданные исходного медиафайла из bronze-слоя."""

    original_path: Path
    file_hash: str
    media_type: MediaType
    size_bytes: int
    duration_ms: int | None = None
    width: int | None = None
    height: int | None = None


class MemeLabelResponse(BaseModel):
    """Структурированный ответ VLM для одного мема (Structured Output)."""

    slug: str = Field(
        description=(
            "Short, descriptive English dash-case filename without extension (2-5 lowercase words, "
            "a-z and digits only, e.g. 'di-caprio-pointing-tv' or 'cat-judging-stare')."
        )
    )
    user_caption: str = Field(
        description=(
            "Ultra-short casual Russian caption (2-6 words max, lowercase), as a lazy user would "
            "quickly tag a saved reaction meme (e.g. 'уолтер уайт не матерись', 'фродо готов домой')."
        )
    )
    rich_description: str = Field(
        description=(
            "Concise 1-2 sentence description in Russian for model training/distillation: "
            "what happens visually, emotion/vibe, and recognizable movie/character/source if known."
        )
    )
    tags: list[str] = Field(
        min_length=2,
        max_length=8,
        description="3-6 short lowercase Russian tags (emotions, characters, objects, meme templates).",
    )
    transcript_or_text: str = Field(
        default="",
        description=(
            "Exact text visible on the image/subtitles and/or spoken quote from video audio "
            "(preserve original language and slang/profanity verbatim; empty string if none)."
        ),
    )
    queries: list[str] = Field(
        min_length=3,
        max_length=6,
        description=(
            "4 diverse, short (1-5 words), natural Russian search queries a user would actually type "
            "inside a meme organizer (NEVER include words like 'мем', 'картинка', 'видео', 'гифка'): "
            "(1) visual/emotion, (2) character/source, (3) quote/slang/sound, (4) chat situation."
        ),
    )


class SilverRecord(BaseModel):
    """Сохранённая запись размеченного мема в data/silver/labels.jsonl."""

    file_hash: str
    original_filename: str
    silver_filename: str
    media_type: MediaType
    duration_ms: int | None = None
    width: int | None = None
    height: int | None = None
    model_name: str
    labeled_at: datetime
    label: MemeLabelResponse


class FailedRecord(BaseModel):
    """Запись о файле, который не удалось разметить, в data/silver/failed.jsonl."""

    file_hash: str
    original_filename: str
    reason: FailureReason
    error_message: str
    failed_at: datetime


class PipelineSummary(BaseModel):
    """Итоговая статистика запуска пайплайна разметки."""

    total_discovered: int = 0
    skipped_already_labeled: int = 0
    processed_success: int = 0
    failed_count: int = 0
    stopped_by_quota: bool = False
