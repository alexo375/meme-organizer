import logging
from pathlib import Path
from typing import ClassVar

from pydantic import ValidationError

from meme_organizer.schemas.labeling import FailedRecord, SilverRecord

logger = logging.getLogger(__name__)


class LabelingCheckpointStore:
    """Хранилище чекпоинтов разметки (labels.jsonl и failed.jsonl) с индексацией по file_hash."""

    LABELS_FILENAME: ClassVar[str] = "labels.jsonl"
    FAILED_FILENAME: ClassVar[str] = "failed.jsonl"

    def __init__(self, silver_dir: Path) -> None:
        self._silver_dir = silver_dir
        self._labels_path = silver_dir / self.LABELS_FILENAME
        self._failed_path = silver_dir / self.FAILED_FILENAME
        self._labels: dict[str, SilverRecord] = {}
        self._failed: dict[str, FailedRecord] = {}
        self._load()

    @property
    def labels_path(self) -> Path:
        """Путь к файлу успешных разметок."""
        return self._labels_path

    @property
    def failed_path(self) -> Path:
        """Путь к файлу ошибок разметки."""
        return self._failed_path

    def is_labeled(self, file_hash: str) -> bool:
        """Проверяет, был ли файл уже успешно размечен."""
        return file_hash in self._labels

    def is_failed(self, file_hash: str) -> bool:
        """Проверяет, находится ли файл в списке неудачных попыток."""
        return file_hash in self._failed

    def all_labels(self) -> list[SilverRecord]:
        """Возвращает все успешно размеченные записи."""
        return list(self._labels.values())

    def all_failed(self) -> list[FailedRecord]:
        """Возвращает все записи об ошибках."""
        return list(self._failed.values())

    def save_label(self, record: SilverRecord) -> None:
        """Сохраняет успешную запись в labels.jsonl и удаляет file_hash из failed.jsonl (если был там)."""
        self._silver_dir.mkdir(parents=True, exist_ok=True)
        self._labels[record.file_hash] = record
        with self._labels_path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
            handle.flush()

        if record.file_hash in self._failed:
            del self._failed[record.file_hash]
            self._rewrite_failed()

    def save_failure(self, record: FailedRecord) -> None:
        """Сохраняет запись об ошибке в failed.jsonl."""
        self._silver_dir.mkdir(parents=True, exist_ok=True)
        self._failed[record.file_hash] = record
        self._rewrite_failed()

    def _load(self) -> None:
        if self._labels_path.exists():
            for line in self._labels_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = SilverRecord.model_validate_json(line)
                    self._labels[rec.file_hash] = rec
                except ValidationError as exc:
                    logger.warning("Пропущена повреждённая строка в %s: %s", self._labels_path, exc)

        if self._failed_path.exists():
            for line in self._failed_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    failed = FailedRecord.model_validate_json(line)
                    if failed.file_hash not in self._labels:
                        self._failed[failed.file_hash] = failed
                except ValidationError as exc:
                    logger.warning("Пропущена повреждённая строка в %s: %s", self._failed_path, exc)

    def _rewrite_failed(self) -> None:
        with self._failed_path.open("w", encoding="utf-8") as handle:
            for item in self._failed.values():
                handle.write(item.model_dump_json() + "\n")
            handle.flush()
