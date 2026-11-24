"""Reciprocal Rank Fusion (RRF) for late rank aggregation."""

from collections import defaultdict
from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    Formula:
        RRF_score(doc) = sum_{rankings} (1.0 / (k + rank))
    where rank is 1-indexed.

    Args:
        rankings: A list of rankings, where each ranking is a list of (doc_id, score)
                  sorted in descending order of score.
        k: Smoothing constant (standard default is 60).

    Returns:
        Combined list of (doc_id, rrf_score) sorted by rrf_score descending.

    """
    scores: dict[str, float] = defaultdict(float)

    for ranking in rankings:
        for rank_idx, (doc_id, _) in enumerate(ranking):
            # rank is 1-indexed
            scores[doc_id] += 1.0 / (k + rank_idx + 1)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
