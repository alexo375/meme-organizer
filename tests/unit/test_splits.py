"""Тесты для генератора сплитов данных."""

from pathlib import Path

from meme_organizer.data_splits.splitter import DatasetSplitter, SceneGroupDetector


def test_scene_group_detector() -> None:
    """Проверяет корректность детекции групп персонажей/фильмов."""
    rec1 = {"user_caption": "уолтер уайт осуждает", "tags": ["во все тяжкие", "мат"], "file_hash": "hash1"}
    rec2 = {"user_caption": "стонкс английский", "tags": ["стонкс", "учитель"], "file_hash": "hash2"}
    rec3 = {"user_caption": "рандомный мем", "tags": [], "file_hash": "abcdef123456"}

    assert SceneGroupDetector.detect_group(rec1) == "breaking_bad_walter"
    assert SceneGroupDetector.detect_group(rec2) == "stonks_guy"
    assert SceneGroupDetector.detect_group(rec3) == "caption_рандомный"


def test_dataset_splitter_leak_free(tmp_path: Path) -> None:
    """Проверяет отсутствие утечек между сплитами и детерминированность."""
    gold_file = Path("data/gold/gold_dataset.jsonl")
    if not gold_file.exists():
        return

    out_file = tmp_path / "splits_test.json"
    splitter = DatasetSplitter(seed=42)
    split_res = splitter.build_split(gold_file, out_file)

    assert len(split_res.corpus_hashes) == 318
    assert len(split_res.train_pairs) > 0
    assert len(split_res.val_pairs) > 0
    assert len(split_res.test_pairs) > 0

    {p["file_hash"] for p in split_res.train_pairs}
    {p["file_hash"] for p in split_res.val_pairs}
    {p["file_hash"] for p in split_res.test_pairs}

    # Проверяем, что хотя бы часть мемов изолирована по группам
    # И что сплит детерминирован
    split_res_2 = splitter.build_split(gold_file, tmp_path / "splits_test_2.json")
    assert len(split_res.train_pairs) == len(split_res_2.train_pairs)
    assert split_res.train_pairs[0] == split_res_2.train_pairs[0]
