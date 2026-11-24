import json
import logging
from pathlib import Path
from typing import ClassVar

from meme_organizer.schemas.curation import CuratedRecord, CurationStatus
from meme_organizer.schemas.labeling import SilverRecord

logger = logging.getLogger(__name__)


class CurationStore:
    """Управляет состоянием разметки и драфтом gold-датасета."""

    GOLD_FILENAME: ClassVar[str] = "gold_dataset.jsonl"
    DRAFT_FILENAME: ClassVar[str] = "curation_draft.json"

    def __init__(self, silver_dir: Path, gold_dir: Path) -> None:
        self._silver_dir = silver_dir
        self._gold_dir = gold_dir
        self._draft_path = gold_dir / self.DRAFT_FILENAME
        self._gold_path = gold_dir / self.GOLD_FILENAME
        self._records: dict[str, CuratedRecord] = {}
        self._load()

    def get_all(self) -> list[CuratedRecord]:
        """Возвращает все текущие записи в драфте."""
        return list(self._records.values())

    def get(self, file_hash: str) -> CuratedRecord | None:
        """Возвращает запись по ее хешу."""
        return self._records.get(file_hash)

    def update_record(self, record: CuratedRecord) -> None:
        """Обновляет запись и сохраняет драфт."""
        self._records[record.file_hash] = record
        self._save_draft()

    def delete_record(self, file_hash: str, *, delete_files: bool = True) -> bool:
        """Удаляет запись из драфта, silver/labels.jsonl и опционально удаляет файл из silver/bronze."""
        record = self._records.pop(file_hash, None)
        if not record:
            return False

        if delete_files:
            silver_file = self._silver_dir / record.silver_filename
            if silver_file.exists():
                silver_file.unlink()

            bronze_file = self._silver_dir.parent / "bronze" / record.original_filename
            if bronze_file.exists():
                bronze_file.unlink()

        self._sync_silver_labels()
        self._save_draft()
        return True

    def export_gold(self) -> int:
        """Экспортирует проверенные записи в data/gold/gold_dataset.jsonl."""
        self._gold_dir.mkdir(parents=True, exist_ok=True)
        verified = [r for r in self._records.values() if r.status == CurationStatus.VERIFIED]

        with self._gold_path.open("w", encoding="utf-8") as f:
            for rec in verified:
                f.write(rec.model_dump_json() + "\n")
        return len(verified)

    def _sync_silver_labels(self) -> None:
        """Синхронизирует silver/labels.jsonl с активными записями драфта."""
        labels_path = self._silver_dir / "labels.jsonl"
        if not labels_path.exists():
            return

        lines = labels_path.read_text(encoding="utf-8").splitlines()
        remaining_lines: list[str] = []
        for line in lines:
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("file_hash") in self._records:
                remaining_lines.append(line)

        labels_path.write_text("\n".join(remaining_lines) + ("\n" if remaining_lines else ""), encoding="utf-8")

    def _load(self) -> None:
        labels_path = self._silver_dir / "labels.jsonl"
        if not labels_path.exists():
            return

        # Если драфт уже существует, берем его за основу, синхронизируя с labels
        draft_map: dict[str, CuratedRecord] = {}
        if self._draft_path.exists():
            try:
                data = json.loads(self._draft_path.read_text(encoding="utf-8"))
                for raw in data:
                    rec = CuratedRecord.model_validate(raw)
                    draft_map[rec.file_hash] = rec
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                logger.warning("Не удалось прочитать драфт: %s. Пересоздаем.", exc)

        for line in labels_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            silver = SilverRecord.model_validate_json(line)
            if silver.file_hash in draft_map:
                self._records[silver.file_hash] = draft_map[silver.file_hash]
            else:
                curated = CuratedRecord(
                    file_hash=silver.file_hash,
                    silver_filename=silver.silver_filename,
                    original_filename=silver.original_filename,
                    media_type=silver.media_type.value,
                    duration_ms=silver.duration_ms,
                    width=silver.width,
                    height=silver.height,
                    user_caption=silver.label.user_caption,
                    rich_description=silver.label.rich_description,
                    tags=silver.label.tags,
                    transcript_or_text=silver.label.transcript_or_text,
                    queries=silver.label.queries,
                    status=CurationStatus.PENDING,
                )
                self._records[curated.file_hash] = curated

        self._save_draft()

    def _save_draft(self) -> None:
        self._gold_dir.mkdir(parents=True, exist_ok=True)
        items = [r.model_dump(mode="json") for r in self._records.values()]
        self._draft_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
