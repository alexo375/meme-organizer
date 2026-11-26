import json
from pathlib import Path

from meme_organizer.data.leak_free_splitter import LeakFreeSplitter


def test_leak_free_splitter(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    splits_path = tmp_path / "splits.json"

    records = [
        {
            "file_hash": "hash_1",
            "silver_filename": "s1.jpg",
            "original_filename": "o1.jpg",
            "media_type": "image",
            "width": 100,
            "height": 100,
            "extra": {"scene_group": "group_A"},
            "queries": ["q1", "q2"],
        },
        {
            "file_hash": "hash_2",
            "silver_filename": "s2.jpg",
            "original_filename": "o2.jpg",
            "media_type": "image",
            "width": 100,
            "height": 100,
            "extra": {"scene_group": "group_A"},
            "queries": ["q3"],
        },
        {
            "file_hash": "hash_3",
            "silver_filename": "s3.jpg",
            "original_filename": "o3.jpg",
            "media_type": "image",
            "width": 100,
            "height": 100,
            "extra": {"scene_group": "group_B"},
            "queries": ["q4", "q5"],
        },
        {
            "file_hash": "hash_4",
            "silver_filename": "s4.jpg",
            "original_filename": "o4.jpg",
            "media_type": "image",
            "width": 100,
            "height": 100,
            "extra": {"scene_group": "group_C"},
            "queries": ["q6"],
        },
    ]

    with open(gold_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    splitter = LeakFreeSplitter(gold_dataset_path=gold_path, output_splits_path=splits_path, random_seed=42)
    splits = splitter.generate_splits(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)

    assert len(splits["corpus_hashes"]) == 4
    total_pairs = len(splits["train_pairs"]) + len(splits["val_pairs"]) + len(splits["test_pairs"])
    assert total_pairs == 6
