from pathlib import Path

from PIL import Image

from meme_organizer.media.deduplicator import MediaDeduplicator
from meme_organizer.schemas.labeling import MediaType, MemeLabelResponse, SilverRecord


def _make_silver_record(
    file_hash: str, filename: str, media_type: MediaType, duration_ms: int | None = None
) -> SilverRecord:
    from datetime import UTC, datetime

    return SilverRecord(
        file_hash=file_hash,
        original_filename=filename,
        silver_filename=filename,
        media_type=media_type,
        duration_ms=duration_ms,
        width=100,
        height=100,
        model_name="test",
        labeled_at=datetime.now(UTC),
        label=MemeLabelResponse(
            slug="test",
            user_caption="test caption",
            rich_description="test rich description",
            tags=["test1", "test2"],
            queries=["test query 1", "test query 2", "test query 3"],
        ),
    )


def test_deduplicator_detects_identical_and_near_images(tmp_path: Path) -> None:
    img1_path = tmp_path / "img1.jpg"
    img2_path = tmp_path / "img2.jpg"
    img3_path = tmp_path / "img3.jpg"

    # Создаем 2 визуально одинаковых с градиентом и 1 с противоположным градиентом
    img1 = Image.new("L", (64, 64))
    img2 = Image.new("L", (64, 64))
    img3 = Image.new("L", (64, 64))

    for x in range(64):
        for y in range(64):
            img1.putpixel((x, y), x * 4)
            img2.putpixel((x, y), x * 4 + 1)
            img3.putpixel((x, y), (63 - x) * 4)

    img1.save(img1_path, format="JPEG")
    img2.save(img2_path, format="JPEG")
    img3.save(img3_path, format="JPEG")

    records = [
        _make_silver_record("h1", "img1.jpg", MediaType.IMAGE),
        _make_silver_record("h2", "img2.jpg", MediaType.IMAGE),
        _make_silver_record("h3", "img3.jpg", MediaType.IMAGE),
    ]

    dedup = MediaDeduplicator(tmp_path)
    dups = dedup.find_duplicates(records)

    assert len(dups) == 1
    assert dups[0].source_hash == "h1"
    assert dups[0].target_hash == "h2"


def test_deduplicator_detects_video_duration_matches(tmp_path: Path) -> None:
    records = [
        _make_silver_record("v1", "vid1.mp4", MediaType.VIDEO, duration_ms=10500),
        _make_silver_record("v2", "vid2.mp4", MediaType.VIDEO, duration_ms=10700),  # diff 200ms
        _make_silver_record("v3", "vid3.mp4", MediaType.VIDEO, duration_ms=25000),
    ]

    dedup = MediaDeduplicator(tmp_path)
    dups = dedup.find_duplicates(records)

    assert len(dups) == 1
    assert dups[0].source_hash == "v1"
    assert dups[0].target_hash == "v2"
