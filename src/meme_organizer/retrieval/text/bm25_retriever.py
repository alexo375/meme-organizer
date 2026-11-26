"""Лексический ретривер на базе BM25 для точного поиска по ключевым словам и субтитрам."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from meme_organizer.domain.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """Лексический ретривер на основе BM25Okapi."""

    def __init__(self, model_name: str = "bm25-baseline", **kwargs: Any) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self.bm25: BM25Okapi | None = None
        self.corpus_tokenized: list[list[str]] = []

    def load_or_build(self) -> None:
        """Индекс BM25 строится в памяти при кодировании корпуса."""

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Приводит текст к нижнему регистру, разбивает по небуквенно-цифровым символам и возвращает токены длиной больше 1."""
        return [w for w in re.split(r"[^\w]+", text.lower()) if len(w) > 1]

    def encode_corpus(self, corpus_docs: list[dict[str, Any]]) -> list[list[str]]:
        """Индексирует поля документов в BM25Okapi и возвращает список токенизированных документов."""
        logger.info("Построение BM25 индекса для %d документов...", len(corpus_docs))
        self.corpus_tokenized = []
        for doc in corpus_docs:
            combined = (
                f"{doc.get('user_caption', '')} "
                f"{doc.get('rich_description', '')} "
                f"{doc.get('transcript_or_text', '')} "
                f"{' '.join(doc.get('tags', []))}"
            )
            self.corpus_tokenized.append(self.tokenize(combined))

        self.bm25 = BM25Okapi(self.corpus_tokenized)
        return self.corpus_tokenized

    def encode_queries(self, queries: list[str]) -> list[list[str]]:
        """Токенизирует список запросов queries и возвращает списки токенов."""
        return [self.tokenize(q) for q in queries]

    def predict_scores(self, query_repr: list[list[str]], corpus_repr: Any) -> np.ndarray:
        """Вычисляет матрицу оценок BM25 для всех пар запрос-документ; если индекс не построен, вызывает RuntimeError."""
        _ = corpus_repr
        if self.bm25 is None:
            raise RuntimeError("BM25 индекс должен быть инициализирован через encode_corpus")

        scores_list = []
        for q_tokens in query_repr:
            doc_scores = self.bm25.get_scores(q_tokens)
            scores_list.append(doc_scores)

        return np.array(scores_list, dtype=np.float32)

    def get_model_size_mb(self) -> float:
        """Возвращает оценку размера индекса в памяти в мегабайтах."""
        return 0.1
