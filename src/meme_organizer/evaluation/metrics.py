"""Модуль расчета метрик ранжирования (IR / Retrieval Metrics)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RankingMetrics:
    """Метрики ранжирования качества поиска."""

    mrr_at_5: float
    recall_at_1: float
    recall_at_5: float
    ndcg_at_5: float
    total_queries: int


class RetrievalEvaluator:
    """Оценивает результаты ранжирования по матрице предсказанных скоров."""

    @staticmethod
    def compute_metrics(
        scores_matrix: np.ndarray,  # shape: (N_queries, N_corpus)
        ground_truth_indices: list[int],  # shape: (N_queries,) - индекс правильного мема в корпусе
        queries: list[str],
        corpus_hashes: list[str],
        top_k: int = 5,
    ) -> tuple[RankingMetrics, list[dict[str, Any]]]:
        """Рассчитывает Recall@1, Recall@5, MRR@5, NDCG@5 и Top-15 Worst Queries."""
        n_queries = len(ground_truth_indices)
        if n_queries == 0:
            return RankingMetrics(0.0, 0.0, 0.0, 0.0, 0), []

        reciprocal_ranks = []
        recall_1_list = []
        recall_5_list = []
        ndcg_5_list = []
        query_errors = []

        # Для каждой строки находим отсортированные индексы документов по убыванию скора
        ranked_indices = np.argsort(-scores_matrix, axis=1)

        for q_idx in range(n_queries):
            target_idx = ground_truth_indices[q_idx]
            ranked = ranked_indices[q_idx]

            # Ищем позицию (1-based rank) правильного документа
            matches = np.where(ranked == target_idx)[0]
            rank = int(matches[0]) + 1 if len(matches) > 0 else 999999

            # MRR@top_k
            if rank <= top_k:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

            # Recall@1
            recall_1_list.append(1.0 if rank == 1 else 0.0)

            # Recall@5
            recall_5_list.append(1.0 if rank <= top_k else 0.0)

            # NDCG@top_k (binary relevance: 1 if target in top_k else 0, DCG = 1 / log2(rank + 1), IDCG = 1 / log2(2) = 1.0)
            if rank <= top_k:
                dcg = 1.0 / math.log2(rank + 1)
                idcg = 1.0 / math.log2(2)
                ndcg_5_list.append(dcg / idcg)
            else:
                ndcg_5_list.append(0.0)

            target_hash = corpus_hashes[target_idx]
            predicted_top = [corpus_hashes[idx] for idx in ranked[:top_k]]
            query_errors.append(
                {
                    "query": queries[q_idx],
                    "target_hash": target_hash,
                    "target_rank": rank,
                    "predicted_top": predicted_top,
                }
            )

        metrics = RankingMetrics(
            mrr_at_5=float(np.mean(reciprocal_ranks)),
            recall_at_1=float(np.mean(recall_1_list)),
            recall_at_5=float(np.mean(recall_5_list)),
            ndcg_at_5=float(np.mean(ndcg_5_list)),
            total_queries=n_queries,
        )

        # Сортируем худшие запросы по убыванию ранга правильного ответа (чем больше ранг, тем хуже)
        query_errors.sort(key=lambda x: float(str(x["target_rank"])), reverse=True)
        top_15_worst = query_errors[:15]

        return metrics, top_15_worst
