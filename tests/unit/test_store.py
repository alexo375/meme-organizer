from datetime import UTC, datetime
from pathlib import Path

from meme_organizer.labeling.store import LabelingCheckpointStore
from meme_organizer.schemas.labeling import (
    FailedRecord,
    FailureReason,
    MediaType,
    MemeLabelResponse,
    SilverRecord,
)


def _sample_record(file_hash: str, slug: str) -> SilverRecord:
    return SilverRecord(
        file_hash=file_hash,
        original_filename="orig.jpg",
        silver_filename=f"{slug}.jpg",
        media_type=MediaType.IMAGE,
        model_name="gemini-2.5-flash",
        labeled_at=datetime.now(UTC),
        label=MemeLabelResponse(
            slug=slug,
            user_caption="кот осуждает",
            rich_description="Кот осуждающе смотрит в камеру",
            tags=["кот", "осуждение", "реакция"],
            transcript_or_text="",
            queries=["кот осуждает", "кот смотрит", "кринж", "когда друг написал бред"],
        ),
    )


def test_store_persists_and_clears_failure_on_retry(tmp_path: Path) -> None:
    store = LabelingCheckpointStore(tmp_path)
    failed = FailedRecord(
        file_hash="hash_1",
        original_filename="a.jpg",
        reason=FailureReason.SAFETY_BLOCKED,
        error_message="blocked",
        failed_at=datetime.now(UTC),
    )
    store.save_failure(failed)
    assert store.is_failed("hash_1")

    reloaded = LabelingCheckpointStore(tmp_path)
    assert reloaded.is_failed("hash_1")
    assert not reloaded.is_labeled("hash_1")

    reloaded.save_label(_sample_record("hash_1", "cat-judging"))
    assert reloaded.is_labeled("hash_1")
    assert not reloaded.is_failed("hash_1")

    reloaded_again = LabelingCheckpointStore(tmp_path)
    assert len(reloaded_again.all_labels()) == 1
    assert len(reloaded_again.all_failed()) == 0
