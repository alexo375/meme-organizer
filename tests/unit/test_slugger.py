from pathlib import Path

from meme_organizer.media.slugger import SlugManager


def test_sanitize_dash_case_and_fallback() -> None:
    assert SlugManager.sanitize("  Di Caprio_Pointing!! TV ", fallback_hash="abcdef1234") == "di-caprio-pointing-tv"
    assert SlugManager.sanitize("кириллица!!!", fallback_hash="abcdef1234") == "meme-abcdef12"
    long_slug = "a" * 100
    assert len(SlugManager.sanitize(long_slug, fallback_hash="abcdef1234")) == SlugManager.MAX_SLUG_LENGTH


def test_allocate_deduplicates_collisions(tmp_path: Path) -> None:
    (tmp_path / "cat-stare.jpg").write_bytes(b"existing")
    manager = SlugManager(tmp_path)

    first = manager.allocate_filename("cat stare", ".JPG", fallback_hash="11112222")
    second = manager.allocate_filename("cat stare", ".jpg", fallback_hash="33334444")

    assert first == "cat-stare-2.jpg"
    assert second == "cat-stare-3.jpg"


def test_materialize_copies_without_mutating_bronze(tmp_path: Path) -> None:
    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"
    bronze.mkdir()
    src = bronze / "IMG_9999.jpg"
    src.write_bytes(b"meme-bytes")

    manager = SlugManager(silver, rename_in_place=False)
    dest = manager.materialize(src, "funny cat", fallback_hash="deadbeef")

    assert dest == silver / "funny-cat.jpg"
    assert dest.read_bytes() == b"meme-bytes"
    assert src.exists()


def test_materialize_rename_in_place(tmp_path: Path) -> None:
    bronze = tmp_path / "bronze"
    bronze.mkdir()
    src = bronze / "IMG_1234.mp4"
    src.write_bytes(b"video-bytes")

    manager = SlugManager(tmp_path / "silver", rename_in_place=True)
    dest = manager.materialize(src, "epic fail", fallback_hash="cafebabe")

    assert dest == bronze / "epic-fail.mp4"
    assert dest.read_bytes() == b"video-bytes"
    assert not src.exists()
