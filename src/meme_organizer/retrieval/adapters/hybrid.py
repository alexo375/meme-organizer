"""Hybrid Retriever Adapter combining Lexical (BM25) and Dense ONNX with optional Re-ranking."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from meme_organizer.retrieval.adapters.finetuned_dense import FineTunedDenseAdapter
from meme_organizer.retrieval.base import BaseRetrieverAdapter
from meme_organizer.retrieval.baselines.bm25 import BM25RetrieverAdapter
from meme_organizer.retrieval.fusion.rrf import reciprocal_rank_fusion
from meme_organizer.retrieval.fusion.weighted import FusionOptimizer, combine_scores
from meme_organizer.retrieval.rerank.cross_encoder import CrossEncoderReranker

if TYPE_CHECKING:
    from meme_organizer.retrieval.tracking import ExperimentTracker

logger = logging.getLogger(__name__)


class HybridRetrieverAdapter(BaseRetrieverAdapter):
    """Hybrid Retriever Adapter supporting RRF, Optuna Convex Blending, and Cross-Encoder Re-ranking."""

    def __init__(
        self,
        mode: str = "optuna",  # "rrf", "optuna", "rerank_ce"
        model_name: str | None = None,
        dense_model_name: str = "multilingual-e5-small-finetuned",
        alpha: float = 0.5,
        rerank_top_k: int = 20,
        **kwargs: Any,
    ) -> None:
        name = model_name or f"hybrid-{mode}"
        super().__init__(model_name=name, **kwargs)
        self.mode = mode
        self.dense_model_name = dense_model_name
        self.alpha = alpha
        self.rerank_top_k = rerank_top_k

        self.bm25_adapter = BM25RetrieverAdapter(model_name=f"{name}_bm25")

        # Настройка параметров E5
        if "e5" in dense_model_name:
            self.dense_adapter = FineTunedDenseAdapter(
                model_name=dense_model_name,
                base_model_id="intfloat/multilingual-e5-small",
                query_prefix="query: ",
                passage_prefix="passage: ",
            )
        else:
            self.dense_adapter = FineTunedDenseAdapter(
                model_name=dense_model_name,
                base_model_id="cointegrated/rubert-tiny2",
            )

        self.reranker: CrossEncoderReranker | None = None
        if self.mode == "rerank_ce":
            self.reranker = CrossEncoderReranker()

        self.corpus_docs: list[dict[str, Any]] = []
        self._current_queries: list[str] = []

    def load_or_build(self) -> None:
        """Инициализирует базовые адаптеры BM25, Dense ONNX и Cross-Encoder."""
        self.bm25_adapter.load_or_build()
        self.dense_adapter.load_or_build()
        if self.reranker:
            self.reranker.export_and_quantize()

    def fit(
        self,
        train_pairs: list[dict[str, str]],
        corpus_docs: list[dict[str, Any]],
        val_pairs: list[dict[str, str]] | None = None,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        """Обучает/калибрует адаптер. Для Optuna ищет оптимальный alpha на val_pairs."""
        logger.info("Fit гибридного адаптера (режим: %s)...", self.mode)
        self.corpus_docs = corpus_docs
        self.bm25_adapter.fit(train_pairs, corpus_docs, val_pairs, tracker)
        self.dense_adapter.fit(train_pairs, corpus_docs, val_pairs, tracker)

        if self.mode == "optuna" and val_pairs:
            logger.info("Подбор оптимального alpha через Optuna на %d валидационных парах...", len(val_pairs))
            # Индексируем корпус в bm25 и dense
            bm25_corpus = self.bm25_adapter.encode_corpus(corpus_docs)
            dense_corpus = self.dense_adapter.encode_corpus(corpus_docs)

            val_queries = [p["query"] for p in val_pairs]
            target_hashes = [p["media_hash"] for p in val_pairs]
            hash_to_idx = {d["media_hash"]: idx for idx, d in enumerate(corpus_docs)}
            target_indices = [hash_to_idx[h] for h in target_hashes if h in hash_to_idx]

            # Получаем скоры на val
            bm25_query_repr = self.bm25_adapter.encode_queries(val_queries)
            bm25_scores = self.bm25_adapter.predict_scores(bm25_query_repr, bm25_corpus)

            dense_query_repr = self.dense_adapter.encode_queries(val_queries)
            dense_scores = self.dense_adapter.predict_scores(dense_query_repr, dense_corpus)

            optimizer = FusionOptimizer(n_trials=50)
            self.alpha = optimizer.fit(dense_scores, bm25_scores, target_indices)
            logger.info("Подобран оптимальный alpha: %.4f (Val MRR@5: %.4f)", self.alpha, optimizer.best_mrr or 0.0)

    def encode_corpus(self, corpus_docs: list[dict[str, Any]]) -> dict[str, Any]:
        """Кодирует базу документов двумя адаптерами: BM25 и Dense."""
        self.corpus_docs = corpus_docs
        bm25_repr = self.bm25_adapter.encode_corpus(corpus_docs)
        dense_repr = self.dense_adapter.encode_corpus(corpus_docs)
        return {"bm25": bm25_repr, "dense": dense_repr}

    def encode_queries(self, queries: list[str]) -> dict[str, Any]:
        """Кодирует список запросов двумя адаптерами."""
        self._current_queries = queries
        bm25_repr = self.bm25_adapter.encode_queries(queries)
        dense_repr = self.dense_adapter.encode_queries(queries)
        return {"bm25": bm25_repr, "dense": dense_repr}

    def predict_scores(self, query_repr: Any, corpus_repr: Any) -> np.ndarray:
        """Рассчитывает финальные матричные скоры сходства."""
        bm25_scores = self.bm25_adapter.predict_scores(query_repr["bm25"], corpus_repr["bm25"])
        dense_scores = self.dense_adapter.predict_scores(query_repr["dense"], corpus_repr["dense"])

        if self.mode == "rrf":
            n_queries, n_docs = dense_scores.shape
            final_scores = np.zeros((n_queries, n_docs), dtype=np.float32)

            for q_idx in range(n_queries):
                dense_ranked = [
                    (str(d_idx), float(dense_scores[q_idx, d_idx])) for d_idx in np.argsort(-dense_scores[q_idx])
                ]
                bm25_ranked = [
                    (str(d_idx), float(bm25_scores[q_idx, d_idx])) for d_idx in np.argsort(-bm25_scores[q_idx])
                ]
                rrf_res = reciprocal_rank_fusion([dense_ranked, bm25_ranked], k=60)
                for d_str, score in rrf_res:
                    final_scores[q_idx, int(d_str)] = score
            return final_scores

        # Default or optuna convex score combination
        combined = combine_scores(dense_scores, bm25_scores, alpha=self.alpha, normalize=True)

        if self.mode == "rerank_ce" and self.reranker:
            logger.info("Stage 2: Cross-Encoder реранкинг топ-%d кандидатов...", self.rerank_top_k)
            n_queries, n_docs = combined.shape
            reranked_scores = combined.copy()

            for q_idx, query in enumerate(self._current_queries):
                candidate_indices = np.argsort(-combined[q_idx])[: self.rerank_top_k]
                pairs = []
                for c_idx in candidate_indices:
                    doc = self.corpus_docs[c_idx]
                    doc_text = f"{doc.get('user_caption', '')} {doc.get('rich_description', '')} {doc.get('transcript_or_text', '')} {' '.join(doc.get('tags', []))}"
                    pairs.append((query, doc_text))

                ce_scores = self.reranker.predict_pair_scores(pairs)
                if len(ce_scores) > 1 and np.max(ce_scores) != np.min(ce_scores):
                    norm_ce = (ce_scores - np.min(ce_scores)) / (np.max(ce_scores) - np.min(ce_scores))
                else:
                    norm_ce = np.zeros_like(ce_scores)

                for rank_pos, c_idx in enumerate(candidate_indices):
                    # Прибавляем вес Cross-Encoder к исходному скору первой стадии
                    reranked_scores[q_idx, c_idx] = 0.5 * float(combined[q_idx, c_idx]) + 0.5 * float(norm_ce[rank_pos])

            return reranked_scores

        return combined

    def get_model_size_mb(self) -> float:
        """Суммарный размер моделей на диске в МБ."""
        size = self.bm25_adapter.get_model_size_mb() + self.dense_adapter.get_model_size_mb()
        if self.reranker and self.reranker.onnx_path.exists():
            size += self.reranker.onnx_path.stat().st_size / (1024 * 1024)
        return size
