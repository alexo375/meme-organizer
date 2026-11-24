import numpy as np

from meme_organizer.retrieval.fusion.rrf import reciprocal_rank_fusion
from meme_organizer.retrieval.fusion.weighted import FusionOptimizer, combine_scores


def test_rrf() -> None:
    # 2 rank lists
    r1 = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
    r2 = [("doc2", 0.95), ("doc3", 0.85), ("doc1", 0.6)]

    fused = reciprocal_rank_fusion([r1, r2], k=60)
    # doc2 is #2 in r1 and #1 in r2 -> score = 1/(60+2) + 1/(60+1)
    # doc1 is #1 in r1 and #3 in r2 -> score = 1/(60+1) + 1/(60+3)
    # doc2 score > doc1 score
    assert fused[0][0] == "doc2"
    assert len(fused) == 3


def test_combine_scores() -> None:
    dense = np.array([[0.1, 0.9], [0.8, 0.2]])
    bm25 = np.array([[10.0, 0.0], [5.0, 5.0]])

    comb = combine_scores(dense, bm25, alpha=0.5, normalize=True)
    assert comb.shape == (2, 2)
    assert np.all(comb >= 0.0)
    assert np.all(comb <= 1.0)


def test_fusion_optimizer() -> None:
    dense = np.array([[0.1, 0.9], [0.8, 0.2]])
    bm25 = np.array([[0.9, 0.1], [0.2, 0.8]])
    targets = [1, 0]  # Dense is perfect for these targets, BM25 is opposite

    opt = FusionOptimizer(n_trials=10)
    alpha = opt.fit(dense, bm25, targets)
    assert 0.0 <= alpha <= 1.0
    assert opt.best_mrr == 1.0
