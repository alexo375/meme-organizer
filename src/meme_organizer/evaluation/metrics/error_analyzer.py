"""Анализ ошибок: поиск запросов с наихудшими позициями релевантных документов."""

from __future__ import annotations

from typing import Any

import numpy as np


class ErrorAnalyzer:
    """Анализатор ошибок ранжирования."""

    @staticmethod
    def find_worst_queries(
        scores_matrix: np.ndarray,
        ground_truth_indices: list[int],
        queries: list[str],
        corpus_hashes: list[str],
        top_k: int = 15,
    ) -> list[dict[str, Any]]:
        """Находит запросы с худшим рангом целевого документа.

        Вход: матрица скоров, индексы целевых документов, тексты запросов, хэши корпуса, лимит top_k.
        Выход: список словарей с полями query, target_hash, target_rank, predicted_top.
        """
        worst_list: list[dict[str, Any]] = []

        for i, target_idx in enumerate(ground_truth_indices):
            row = scores_matrix[i]
            sorted_indices = np.argsort(-row)
            matches = np.where(sorted_indices == target_idx)[0]
            actual_rank = int(matches[0] + 1) if len(matches) > 0 else len(sorted_indices)

            top_pred_hashes = [corpus_hashes[idx] for idx in sorted_indices[:5]]

            worst_list.append(
                {
                    "query": queries[i],
                    "target_hash": corpus_hashes[target_idx],
                    "target_rank": actual_rank,
                    "predicted_top": top_pred_hashes,
                }
            )

        worst_list.sort(key=lambda x: x["target_rank"], reverse=True)
        return worst_list[:top_k]
