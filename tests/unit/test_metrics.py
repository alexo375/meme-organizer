"""Тесты для модуля расчета метрик ранжирования."""

import numpy as np

from meme_organizer.evaluation.metrics import RetrievalEvaluator


def test_retrieval_evaluator_ideal_ranking() -> None:
    """Проверяет случай идеального ранжирования (целевой документ всегда на 1 месте)."""
    # 2 запроса, 3 документа в корпусе
    scores = np.array(
        [
            [10.0, 5.0, 1.0],  # doc 0 is best
            [2.0, 8.0, 1.0],  # doc 1 is best
        ]
    )
    ground_truth = [0, 1]
    queries = ["q1", "q2"]
    corpus_hashes = ["h0", "h1", "h2"]

    metrics, worst = RetrievalEvaluator.compute_metrics(scores, ground_truth, queries, corpus_hashes, top_k=5)

    assert metrics.recall_at_1 == 1.0
    assert metrics.recall_at_5 == 1.0
    assert metrics.mrr_at_5 == 1.0
    assert metrics.ndcg_at_5 == 1.0
    assert len(worst) == 2
    assert worst[0]["target_rank"] == 1


def test_retrieval_evaluator_mixed_ranking() -> None:
    """Проверяет случай частичного совпадения."""
    scores = np.array(
        [
            [10.0, 5.0, 1.0],  # rank of doc 1 is 2
            [1.0, 2.0, 3.0],  # rank of doc 0 is 3
        ]
    )
    ground_truth = [1, 0]
    queries = ["q1", "q2"]
    corpus_hashes = ["h0", "h1", "h2"]

    metrics, worst = RetrievalEvaluator.compute_metrics(scores, ground_truth, queries, corpus_hashes, top_k=5)

    # q1 target=1, rank=2 -> RR = 0.5, R@1 = 0, R@5 = 1
    # q2 target=0, rank=3 -> RR = 0.3333, R@1 = 0, R@5 = 1
    assert metrics.recall_at_1 == 0.0
    assert metrics.recall_at_5 == 1.0
    assert np.isclose(metrics.mrr_at_5, (0.5 + 1.0 / 3.0) / 2.0)
    assert worst[0]["target_rank"] == 3
