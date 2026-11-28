import json
from pathlib import Path

from meme_organizer.evaluation.pipeline.retrieval_pipeline import RetrievalPipeline
from meme_organizer.retrieval.text.bm25_retriever import BM25Retriever


def test_retrieval_pipeline_execution(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    splits_path = tmp_path / "splits.json"

    records = [
        {
            "file_hash": "h1",
            "silver_filename": "s1.jpg",
            "original_filename": "o1.jpg",
            "media_type": "image",
            "width": 100,
            "height": 100,
            "user_caption": "кот спит",
            "queries": ["котик спит"],
        },
        {
            "file_hash": "h2",
            "silver_filename": "s2.jpg",
            "original_filename": "o2.jpg",
            "media_type": "image",
            "width": 100,
            "height": 100,
            "user_caption": "собака бежит",
            "queries": ["пес бежит"],
        },
    ]

    with open(gold_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    splits_data = {
        "version": "v1",
        "corpus_hashes": ["h1", "h2"],
        "test_pairs": [
            {"query": "котик спит", "file_hash": "h1", "scene_group": "g1"},
            {"query": "пес бежит", "file_hash": "h2", "scene_group": "g2"},
        ],
    }

    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits_data, f)

    pipeline = RetrievalPipeline()
    bm25 = BM25Retriever(model_name="test-bm25")

    metrics, run_id = pipeline.run(
        retriever=bm25,
        gold_dataset_path=gold_path,
        split_path=splits_path,
        split_name="test",
    )

    assert metrics.total_queries == 2
    assert metrics.recall_at_1 >= 0.5
    assert len(run_id) > 0
