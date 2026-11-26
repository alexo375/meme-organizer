"""Схемы данных для эталонного датасета и пар оценки запросов."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryPair(BaseModel):
    """Пара оценки, связывающая текст запроса с хэшем файла и группой сцен."""

    query: str
    file_hash: str
    scene_group: str


class GoldRecord(BaseModel):
    """Схема записи мема в файле data/gold/gold_dataset.jsonl."""

    file_hash: str
    silver_filename: str
    original_filename: str
    media_type: str
    duration_ms: int = 0
    width: int
    height: int
    user_caption: str = ""
    rich_description: str = ""
    tags: list[str] = Field(default_factory=list)
    transcript_or_text: str = ""
    queries: list[str] = Field(default_factory=list)
    status: str = "curated"
    notes: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)
