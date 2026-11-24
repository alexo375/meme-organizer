"""Schemas for gold dataset and query evaluation pairs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryPair(BaseModel):
    """Evaluation query pair mapping query to target media hash."""

    query: str
    file_hash: str
    scene_group: str


class GoldRecord(BaseModel):
    """Schema representing a verified meme record in data/gold/gold_dataset.jsonl."""

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
