"""Rank Fusion strategies (RRF and Optuna Convex Blending)."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence
import numpy as np


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine ranked lists using Reciprocal Rank Fusion: score = sum(1 / (k + rank))."""
    scores: dict[str, float] = defaultdict(float)

    for ranking in rankings:
        for rank_idx, (doc_id, _) in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank_idx + 1)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize score matrix row-wise (per query) into [0, 1]."""
    min_vals = np.min(scores, axis=1, keepdims=True)
    max_vals = np.max(scores, axis=1, keepdims=True)
    diff = max_vals - min_vals
    diff[diff == 0] = 1e-9
    normalized: np.ndarray = (scores - min_vals) / diff
    return normalized


def combine_scores(
    scores_dense: np.ndarray,
    scores_bm25: np.ndarray,
    alpha: float = 0.5,
    normalize: bool = True,
) -> np.ndarray:
    """Combine dense and sparse scores: alpha * dense + (1 - alpha) * bm25."""
    if normalize:
        s_dense = min_max_normalize(scores_dense)
        s_bm25 = min_max_normalize(scores_bm25)
    else:
        s_dense = scores_dense
        s_bm25 = scores_bm25

    return alpha * s_dense + (1.0 - alpha) * s_bm25


class FusionOptimizer:
    """Optimizes blending weight alpha on validation set using Optuna."""

    def __init__(self, n_trials: int = 50) -> None:
        self.n_trials = n_trials
        self.best_alpha: float | None = None
        self.best_mrr: float | None = None

    def fit(
        self,
        scores_dense: np.ndarray,
        scores_bm25: np.ndarray,
        query_target_indices: list[int],
    ) -> float:
        """Run Optuna study maximizing MRR@5 with live MLflow tracking."""
        import mlflow
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        s_dense_norm = min_max_normalize(scores_dense)
        s_bm25_norm = min_max_normalize(scores_bm25)

        targets = np.array(query_target_indices)
        best_so_far = 0.0

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_so_far
            alpha = trial.suggest_float("alpha", 0.0, 1.0)
            combined = alpha * s_dense_norm + (1.0 - alpha) * s_bm25_norm
            top5_indices = np.argsort(-combined, axis=1)[:, :5]

            reciprocal_ranks = []
            for i, target in enumerate(targets):
                matches = np.where(top5_indices[i] == target)[0]
                if len(matches) > 0:
                    reciprocal_ranks.append(1.0 / (matches[0] + 1))
                else:
                    reciprocal_ranks.append(0.0)

            trial_mrr = float(np.mean(reciprocal_ranks))
            if trial_mrr > best_so_far:
                best_so_far = trial_mrr

            if mlflow.active_run():
                mlflow.log_metrics(
                    {
                        "optuna_trial_mrr_at_5": trial_mrr,
                        "optuna_best_mrr_at_5": best_so_far,
                        "optuna_trial_alpha": alpha,
                    },
                    step=trial.number,
                )

            return trial_mrr

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        self.best_alpha = float(study.best_params["alpha"])
        self.best_mrr = float(study.best_value)
        return self.best_alpha
