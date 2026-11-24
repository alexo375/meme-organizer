"""Тесты для конвейера RetrievalPipeline."""

from pathlib import Path
from typing import Any

import numpy as np

from meme_organizer.retrieval.base import BaseRetrieverAdapter
from meme_organizer.retrieval.pipeline import RetrievalPipeline


class DummyAdapter(BaseRetrieverAdapter):
    """Тестовый адаптер-заглушка."""

    def load_or_build(self) -> None:
        pass

    def encode_corpus(self, corpus_docs: list[dict[str, Any]]) -> int:
        return len(corpus_docs)

    def encode_queries(self, queries: list[str]) -> int:
        return len(queries)

    def predict_scores(self, query_repr: Any, corpus_repr: Any) -> np.ndarray:
        # Единичная матрица (идеальные предсказания для первых N)
        n_q = query_repr
        n_c = corpus_repr
        scores = np.zeros((n_q, n_c), dtype=np.float32)
        for i in range(min(n_q, n_c)):
            scores[i, i] = 1.0
        return scores


def test_retrieval_pipeline_mock_run(tmp_path: Path) -> None:
    """Проверяет прогон пайплайна с mock-трекером."""
    import json

    gold_path = tmp_path / "gold.jsonl"
    with open(gold_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"file_hash": "h1", "user_caption": "m1"}) + "\n")
        f.write(json.dumps({"file_hash": "h2", "user_caption": "m2"}) + "\n")

    split_path = tmp_path / "splits.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": "v1",
                "test_pairs": [
                    {"query": "q1", "file_hash": "h1"},
                    {"query": "q2", "file_hash": "h2"},
                ],
            },
            f,
        )

    class MockTracker:
        def log_run(self, **kwargs: Any) -> str:
            return "mock-run-id-123"

    pipeline = RetrievalPipeline(tracker=MockTracker())  # type: ignore[arg-type]
    metrics, run_id = pipeline.run(
        adapter=DummyAdapter("dummy-model"),
        gold_dataset_path=gold_path,
        split_path=split_path,
        split_name="test",
    )

    assert run_id == "mock-run-id-123"
    assert metrics.recall_at_1 == 1.0
    assert metrics.mrr_at_5 == 1.0
