from enum import StrEnum

from pydantic import BaseModel


class CurationStatus(StrEnum):
    """Статус верификации записи человеком."""

    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"


class CuratedRecord(BaseModel):
    """Отредактированная и проверенная запись для Gold датасета."""

    file_hash: str
    silver_filename: str
    original_filename: str
    media_type: str
    duration_ms: int | None = None
    width: int | None = None
    height: int | None = None
    user_caption: str
    rich_description: str
    tags: list[str]
    transcript_or_text: str = ""
    queries: list[str]
    status: CurationStatus = CurationStatus.PENDING
    notes: str = ""
