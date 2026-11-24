"""Инициализация пакета adapters."""

from meme_organizer.retrieval.adapters.dense_text import DenseTextRetrieverAdapter
from meme_organizer.retrieval.adapters.finetuned_dense import FineTunedDenseAdapter

__all__ = ["DenseTextRetrieverAdapter", "FineTunedDenseAdapter"]
