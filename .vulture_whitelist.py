# Whitelist для vulture: поля Pydantic-моделей, свойства публичного API и атрибуты сторонних библиотек.
from meme_organizer.config import LabelingConfig
from meme_organizer.curation.server import CurationAppFactory
from meme_organizer.labeling.store import LabelingCheckpointStore
from meme_organizer.media.deduplicator import DuplicateCandidate
from meme_organizer.schemas.curation import CuratedRecord, CurationStatus
from meme_organizer.schemas.labeling import (
    FailedRecord,
    MediaFileInfo,
    MemeLabelResponse,
    SilverRecord,
)

benchmark = None  # torch.backends.cudnn.benchmark

_ = (
    LabelingConfig.model_config,
    LabelingCheckpointStore.labels_path,
    LabelingCheckpointStore.failed_path,
    MediaFileInfo,
    MemeLabelResponse.transcript_or_text,
    SilverRecord.labeled_at,
    FailedRecord.error_message,
    FailedRecord.failed_at,
    benchmark,
    CurationAppFactory,
    DuplicateCandidate.source_filename,
    DuplicateCandidate.target_filename,
    DuplicateCandidate.distance,
    CurationStatus.REJECTED,
    CurationStatus.MERGED,
    CuratedRecord.notes,
)
