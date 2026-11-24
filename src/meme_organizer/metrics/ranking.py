"""Ranking evaluation metrics calculation (Recall@K, MRR@K, NDCG@K)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class RankingMetrics:
    """Core evaluation metrics for retrieval."""

    recall_at_1: float
    recall_at_5: float
    mrr_at_5: float
    ndcg_at_5: float
    total_queries: int

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return asdict(self)


def calculate_ranking_metrics(
    scores_matrix: np.ndarray,
    ground_truth_indices: list[int],
    top_k: int = 5,
) -> RankingMetrics:
    """Vectorized calculation of Recall@1, Recall@5, MRR@5, and NDCG@5."""
    n_queries = len(ground_truth_indices)
    if scores_matrix.shape[0] != n_queries:
        raise ValueError("Количество строк матрицы должно совпадать с числом запросов")

    top_indices = np.argsort(-scores_matrix, axis=1)[:, :top_k]

    r1_count = 0
    r5_count = 0
    reciprocal_ranks: list[float] = []
    ndcg_scores: list[float] = []

    discounts = 1.0 / np.log2(np.arange(top_k) + 2)

    for i, target_idx in enumerate(ground_truth_indices):
        predicted_row = top_indices[i]
        matches = np.where(predicted_row == target_idx)[0]

        if len(matches) > 0:
            rank = matches[0] + 1
            if rank == 1:
                r1_count += 1
            r5_count += 1
            reciprocal_ranks.append(1.0 / rank)
            ndcg_scores.append(float(discounts[rank - 1]))
        else:
            reciprocal_ranks.append(0.0)
            ndcg_scores.append(0.0)

    recall_at_1 = round(r1_count / n_queries, 4)
    recall_at_5 = round(r5_count / n_queries, 4)
    mrr_at_5 = round(float(np.mean(reciprocal_ranks)), 4)
    ndcg_at_5 = round(float(np.mean(ndcg_scores)), 4)

    return RankingMetrics(
        recall_at_1=recall_at_1,
        recall_at_5=recall_at_5,
        mrr_at_5=mrr_at_5,
        ndcg_at_5=ndcg_at_5,
        total_queries=n_queries,
    )
