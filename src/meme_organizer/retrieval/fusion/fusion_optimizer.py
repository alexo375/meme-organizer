"""Стратегии объединения ранжирований: RRF и выпуклое смешивание через Optuna с K-Fold кросс-валидацией."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import KFold

if TYPE_CHECKING:
    from collections.abc import Sequence


class FusionOptimizer:
    """Оптимизатор весов смешивания скоров через 5-Fold CV Optuna."""

    @staticmethod
    def reciprocal_rank_fusion(
        rankings: Sequence[Sequence[tuple[str, float]]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """Объединяет ранжированные списки методом Reciprocal Rank Fusion."""
        scores: dict[str, float] = defaultdict(float)

        for ranking in rankings:
            for rank_idx, (doc_id, _) in enumerate(ranking):
                scores[doc_id] += 1.0 / (k + rank_idx + 1)

        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    @staticmethod
    def min_max_normalize(scores: np.ndarray) -> np.ndarray:
        """Нормализует матрицу скоров построчно в диапазон [0, 1]."""
        min_vals = np.min(scores, axis=1, keepdims=True)
        max_vals = np.max(scores, axis=1, keepdims=True)
        diff = max_vals - min_vals
        diff[diff == 0] = 1e-9
        normalized: np.ndarray = (scores - min_vals) / diff
        return normalized

    @staticmethod
    def combine_scores(
        scores_dense: np.ndarray,
        scores_bm25: np.ndarray,
        alpha: float = 0.5,
        normalize: bool = True,
    ) -> np.ndarray:
        """Смешивает плотные и разреженные скоры с весом alpha."""
        if normalize:
            s_dense = FusionOptimizer.min_max_normalize(scores_dense)
            s_bm25 = FusionOptimizer.min_max_normalize(scores_bm25)
        else:
            s_dense = scores_dense
            s_bm25 = scores_bm25

        combined: np.ndarray = alpha * s_dense + (1.0 - alpha) * s_bm25
        return combined

    def __init__(
        self,
        n_trials: int = 50,
        alpha_range: tuple[float, float] = (0.25, 0.75),
        n_folds: int = 5,
        l2_reg: float = 0.01,
    ) -> None:
        self.n_trials = n_trials
        self.alpha_range = alpha_range
        self.n_folds = n_folds
        self.l2_reg = l2_reg
        self.best_alpha: float | None = None
        self.best_mrr: float | None = None

    @staticmethod
    def _mean_mrr_at_5(combined_scores: np.ndarray, targets: np.ndarray) -> float:
        top5_indices = np.argsort(-combined_scores, axis=1)[:, :5]
        reciprocal_ranks = []
        for i, target in enumerate(targets):
            matches = np.where(top5_indices[i] == target)[0]
            reciprocal_ranks.append(1.0 / (matches[0] + 1) if len(matches) > 0 else 0.0)
        return float(np.mean(reciprocal_ranks))

    def fit(
        self,
        scores_dense: np.ndarray,
        scores_bm25: np.ndarray,
        query_target_indices: list[int],
    ) -> float:
        """Подбирает вес alpha через Optuna, максимизируя 5-Fold CV MRR@5 с L2-регуляризацией к 0.5."""
        import mlflow
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        s_dense_norm = FusionOptimizer.min_max_normalize(scores_dense)
        s_bm25_norm = FusionOptimizer.min_max_normalize(scores_bm25)
        targets = np.array(query_target_indices)

        n_samples = len(targets)
        if n_samples >= self.n_folds:
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            folds = [val_idx for _, val_idx in kf.split(targets)]
        else:
            folds = [np.arange(n_samples)]

        best_so_far = 0.0
        low_a, high_a = self.alpha_range

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_so_far
            alpha = trial.suggest_float("alpha", low_a, high_a)
            combined = alpha * s_dense_norm + (1.0 - alpha) * s_bm25_norm

            fold_mrrs = [self._mean_mrr_at_5(combined[idx], targets[idx]) for idx in folds]
            cv_mrr = float(np.mean(fold_mrrs))
            reg_objective = cv_mrr - self.l2_reg * ((alpha - 0.5) ** 2)

            if cv_mrr > best_so_far:
                best_so_far = cv_mrr

            if mlflow.active_run():
                mlflow.log_metrics(
                    {
                        "optuna_trial_mrr_at_5": cv_mrr,
                        "optuna_best_mrr_at_5": best_so_far,
                        "optuna_trial_alpha": alpha,
                    },
                    step=trial.number,
                )

            return reg_objective

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials)

        self.best_alpha = float(study.best_params["alpha"])
        self.best_mrr = best_so_far
        return self.best_alpha
