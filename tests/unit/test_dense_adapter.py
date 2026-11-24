"""Тесты для универсального DenseTextRetrieverAdapter."""

import numpy as np

from meme_organizer.retrieval.adapters.dense_text import DenseTextRetrieverAdapter


def test_dense_text_adapter_dimensions_and_cosine() -> None:
    """Проверяет косинусное сходство и корректность размерностей."""
    adapter = DenseTextRetrieverAdapter("dummy-dense", "dummy-hf-id")

    # Синтетические эмбеддинги
    q_emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    c_emb = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)

    scores = adapter.predict_scores(q_emb, c_emb)
    assert scores.shape == (2, 3)
    assert np.isclose(scores[0, 0], 1.0)
    assert np.isclose(scores[0, 1], 0.0)
    assert np.isclose(scores[1, 1], 1.0)
    assert np.isclose(scores[1, 0], 0.0)
