"""Unit tests for GatedAttentionFusion and multi-channel scoring."""

from __future__ import annotations

import numpy as np
import torch

from meme_organizer.retrieval.fusion.gated_attention_fusion import GatedAttentionFusion
from meme_organizer.retrieval.fusion.multi_channel_scorer import MultiChannelScorer


def test_gated_attention_masks_missing_modalities() -> None:
    model = GatedAttentionFusion(query_dim=16, num_channels=4, hidden_dim=8)
    q = torch.randn(3, 16)
    scores = torch.rand(3, 5, 4)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
            [True, True, False, True],
            [True, True, True, True],
            [True, True, False, False],
        ]
    )
    out = model(q, scores, mask)
    assert out.shape == (3, 5)
    assert torch.isfinite(out).all()


def test_train_gated_attention_and_rrf() -> None:
    q = np.random.randn(6, 16).astype(np.float32)
    s = np.random.rand(6, 4, 4).astype(np.float32)
    gt = [0, 1, 2, 3, 0, 1]
    mask = MultiChannelScorer.build_channel_mask(
        {"a": "text"},
        {"b": "speech"},
        [{"file_hash": "a"}, {"file_hash": "b"}, {"file_hash": "c"}, {"file_hash": "d"}],
    )
    assert mask.shape == (4, 4)
    assert mask[0, 2]
    assert not mask[0, 3]
    model = GatedAttentionFusion.train_model(q, s, gt, mask, epochs=5)
    with torch.no_grad():
        fused = model(torch.from_numpy(q), torch.from_numpy(s), torch.from_numpy(mask))
    assert fused.shape == (6, 4)
    rrf = MultiChannelScorer.rrf_fuse([s[:, :, 0], s[:, :, 1]])
    assert rrf.shape == (6, 4)
