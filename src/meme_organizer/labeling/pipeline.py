import logging
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm

from meme_organizer.labeling.exceptions import (
    LabelingError,
    QuotaExhaustedError,
    SafetyBlockedError,
    TransientApiError,
)
from meme_organizer.labeling.gemini_client import MediaLabeler
from meme_organizer.labeling.rate_limiter import RateLimiter
from meme_organizer.labeling.store import LabelingCheckpointStore
from meme_organizer.media.preprocessor import MediaPreprocessor
from meme_organizer.media.scanner import MediaScanner
from meme_organizer.media.slugger import SlugManager
from meme_organizer.schemas.labeling import (
    FailedRecord,
    FailureReason,
    MediaFileInfo,
    PipelineSummary,
    SilverRecord,
)

logger = logging.getLogger(__name__)


class BronzeLabelingPipeline:
    """Оркестрирует разметку data/bronze -> data/silver с чекпоинтами, rate-limiting и dash-case копированием."""

    def __init__(
        self,
        *,
        scanner: MediaScanner,
        preprocessor: MediaPreprocessor,
        labeler: MediaLabeler,
        store: LabelingCheckpointStore,
        slugger: SlugManager,
        rate_limiter: RateLimiter,
    ) -> None:
        self._scanner = scanner
        self._preprocessor = preprocessor
        self._labeler = labeler
        self._store = store
        self._slugger = slugger
        self._rate_limiter = rate_limiter

    def run(
        self,
        bronze_dir: Path,
        *,
        max_files: int | None = None,
        retry_failed: bool = False,
        show_progress: bool = True,
    ) -> PipelineSummary:
        """Запускает разметку новых (или ранее упавших) файлов с авто-продолжением."""
        all_files = self._scanner.scan(bronze_dir)
        summary = PipelineSummary(total_discovered=len(all_files))

        queue = self._select_queue(all_files, retry_failed=retry_failed, summary=summary)
        if max_files is not None:
            queue = queue[:max_files]

        iterator = tqdm(queue, desc="Labeling bronze -> silver", disable=not show_progress)
        for info in iterator:
            try:
                self._process_single(info)
                summary.processed_success += 1
            except QuotaExhaustedError as exc:
                logger.error("Остановка: %s. Прогресс сохранён, запусти команду позже для продолжения.", exc)
                self._record_failure(info, FailureReason.RATE_LIMIT, str(exc))
                summary.failed_count += 1
                summary.stopped_by_quota = True
                break
            except SafetyBlockedError as exc:
                logger.warning("Файл %s заблокирован safety-фильтром: %s", info.original_path.name, exc)
                self._record_failure(info, FailureReason.SAFETY_BLOCKED, str(exc))
                summary.failed_count += 1
            except TransientApiError as exc:
                logger.warning("Исчерпаны ретраи для %s: %s", info.original_path.name, exc)
                self._record_failure(info, FailureReason.RATE_LIMIT, str(exc))
                summary.failed_count += 1
            except LabelingError as exc:
                logger.warning("Ошибка API для %s: %s", info.original_path.name, exc)
                self._record_failure(info, FailureReason.API_ERROR, str(exc))
                summary.failed_count += 1
            except OSError as exc:
                logger.warning("Ошибка чтения/обработки медиа %s: %s", info.original_path.name, exc)
                self._record_failure(info, FailureReason.MEDIA_ERROR, str(exc))
                summary.failed_count += 1

        return summary

    def _select_queue(
        self,
        all_files: list[MediaFileInfo],
        *,
        retry_failed: bool,
        summary: PipelineSummary,
    ) -> list[MediaFileInfo]:
        queue: list[MediaFileInfo] = []
        seen_hashes: set[str] = set()

        for info in all_files:
            if info.file_hash in seen_hashes:
                continue
            seen_hashes.add(info.file_hash)

            if self._store.is_labeled(info.file_hash):
                summary.skipped_already_labeled += 1
                continue

            if retry_failed and not self._store.is_failed(info.file_hash):
                continue

            if not retry_failed and self._store.is_failed(info.file_hash):
                continue

            queue.append(info)
        return queue

    def _process_single(self, info: MediaFileInfo) -> None:
        with self._preprocessor.prepare(info) as prepared:
            label = self._rate_limiter.execute(lambda: self._labeler.label(info, prepared))

        dest_path = self._slugger.materialize(
            info.original_path,
            label.slug,
            fallback_hash=info.file_hash,
        )

        record = SilverRecord(
            file_hash=info.file_hash,
            original_filename=info.original_path.name,
            silver_filename=dest_path.name,
            media_type=info.media_type,
            duration_ms=info.duration_ms,
            width=info.width,
            height=info.height,
            model_name=self._labeler.model_name,
            labeled_at=datetime.now(UTC),
            label=label,
        )
        self._store.save_label(record)

    def _record_failure(self, info: MediaFileInfo, reason: FailureReason, message: str) -> None:
        failed = FailedRecord(
            file_hash=info.file_hash,
            original_filename=info.original_path.name,
            reason=reason,
            error_message=message,
            failed_at=datetime.now(UTC),
        )
        self._store.save_failure(failed)
