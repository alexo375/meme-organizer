"""Weighted Convex Score Combination and Optuna Optimizer."""

import numpy as np


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
    """Combine dense and sparse scores with weight alpha.

    Final_Score = alpha * Score_dense + (1 - alpha) * Score_bm25
    """
    if normalize:
        s_dense = min_max_normalize(scores_dense)
        s_bm25 = min_max_normalize(scores_bm25)
    else:
        s_dense = scores_dense
        s_bm25 = scores_bm25

    return alpha * s_dense + (1.0 - alpha) * s_bm25


class FusionOptimizer:
    """Finds optimal weight alpha on validation set using Optuna."""

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
        """Run Optuna study to maximize MRR@5 on validation pairs."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        s_dense_norm = min_max_normalize(scores_dense)
        s_bm25_norm = min_max_normalize(scores_bm25)

        targets = np.array(query_target_indices)

        def objective(trial: optuna.Trial) -> float:
            alpha = trial.suggest_float("alpha", 0.0, 1.0)
            combined = alpha * s_dense_norm + (1.0 - alpha) * s_bm25_norm

            # Векторизованный расчет MRR@5
            # Сортируем индексы документов по убыванию скора
            top5_indices = np.argsort(-combined, axis=1)[:, :5]

            # Проверяем совпадение с targets
            reciprocal_ranks = []
            for i, target in enumerate(targets):
                matches = np.where(top5_indices[i] == target)[0]
                if len(matches) > 0:
                    reciprocal_ranks.append(1.0 / (matches[0] + 1))
                else:
                    reciprocal_ranks.append(0.0)

            return float(np.mean(reciprocal_ranks))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        self.best_alpha = float(study.best_params["alpha"])
        self.best_mrr = float(study.best_value)
        return self.best_alpha
