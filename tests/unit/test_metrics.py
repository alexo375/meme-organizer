import numpy as np

from meme_organizer.metrics.error_analysis import find_worst_queries
from meme_organizer.metrics.ranking import calculate_ranking_metrics


def test_ranking_metrics_perfect() -> None:
    scores = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    ground_truth = [0, 1]
    metrics = calculate_ranking_metrics(scores, ground_truth, top_k=3)

    assert metrics.recall_at_1 == 1.0
    assert metrics.recall_at_5 == 1.0
    assert metrics.mrr_at_5 == 1.0
    assert metrics.ndcg_at_5 == 1.0
    assert metrics.total_queries == 2


def test_find_worst_queries() -> None:
    scores = np.array(
        [
            [0.1, 0.9, 0.8],
            [0.9, 0.1, 0.0],
        ]
    )
    ground_truth = [0, 0]
    queries = ["query_1", "query_2"]
    corpus_hashes = ["hash_0", "hash_1", "hash_2"]

    worst = find_worst_queries(scores, ground_truth, queries, corpus_hashes, top_k=2)
    assert len(worst) == 2
    assert worst[0]["query"] == "query_1"
    assert worst[0]["target_rank"] == 3
