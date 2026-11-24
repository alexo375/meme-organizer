from pathlib import Path

from meme_organizer.curation.store import CurationStore
from meme_organizer.schemas.curation import CuratedRecord, CurationStatus


def test_curation_store_update_delete_and_export(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    silver.mkdir()
    gold.mkdir()

    f1 = silver / "rec1.jpg"
    f2 = silver / "rec2.jpg"
    f1.write_bytes(b"1")
    f2.write_bytes(b"2")

    store = CurationStore(silver, gold)
    r1 = CuratedRecord(
        file_hash="h1",
        silver_filename="rec1.jpg",
        original_filename="rec1.jpg",
        media_type="image",
        user_caption="caption 1",
        rich_description="rich 1",
        tags=["tag1", "common"],
        queries=["query 1"],
        status=CurationStatus.VERIFIED,
    )
    r2 = CuratedRecord(
        file_hash="h2",
        silver_filename="rec2.jpg",
        original_filename="rec2.jpg",
        media_type="image",
        user_caption="caption 2",
        rich_description="rich 2",
        tags=["tag2", "common"],
        queries=["query 2"],
        status=CurationStatus.PENDING,
    )

    store.update_record(r1)
    store.update_record(r2)

    assert len(store.get_all()) == 2
    assert store.get("h1") is not None

    # Удаление h2
    deleted = store.delete_record("h2", delete_files=True)
    assert deleted is True
    assert not f2.exists()
    assert store.get("h2") is None
    assert len(store.get_all()) == 1

    # Экспорт
    count = store.export_gold()
    assert count == 1
    assert (gold / "gold_dataset.jsonl").exists()
