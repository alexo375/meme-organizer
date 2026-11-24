"""Hybrid Retriever combining Lexical (BM25) and Dense (E5) with optional Re-ranking."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from meme_organizer.core.contracts import BaseRetriever
from meme_organizer.models.bm25 import BM25Retriever
from meme_organizer.models.cross_encoder import CrossEncoderReranker
from meme_organizer.models.finetuned_dense import FineTunedDenseRetriever
from meme_organizer.models.fusion import FusionOptimizer, combine_scores, reciprocal_rank_fusion

if TYPE_CHECKING:
    from pathlib import Path

    from meme_organizer.core.tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """Hybrid Retriever supporting RRF, Optuna Convex Blending, and Cross-Encoder Re-ranking."""

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

        self.bm25_retriever = BM25Retriever(model_name=f"{name}_bm25")

        if "e5" in dense_model_name:
            self.dense_retriever = FineTunedDenseRetriever(
                model_name=dense_model_name,
                base_model_id="intfloat/multilingual-e5-small",
                query_prefix="query: ",
                passage_prefix="passage: ",
            )
        else:
            self.dense_retriever = FineTunedDenseRetriever(
                model_name=dense_model_name,
                base_model_id="cointegrated/rubert-tiny2",
            )

        self.reranker: CrossEncoderReranker | None = None
        if self.mode == "rerank_ce":
            self.reranker = CrossEncoderReranker()

        self.corpus_docs: list[dict[str, Any]] = []
        self._current_queries: list[str] = []

    def load_or_build(self) -> None:
        """Initialize components."""
        self.bm25_retriever.load_or_build()
        self.dense_retriever.load_or_build()
        if self.reranker:
            self.reranker.export_and_quantize()

    def fit(
        self,
        train_pairs: list[dict[str, str]],
        corpus_docs: list[dict[str, Any]],
        val_pairs: list[dict[str, str]] | None = None,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        """Calibrate blending weights on validation pairs if mode is optuna."""
        logger.info("Fit гибридного ретривера (режим: %s)...", self.mode)
        self.corpus_docs = corpus_docs
        self.bm25_retriever.fit(train_pairs, corpus_docs, val_pairs, tracker)
        self.dense_retriever.fit(train_pairs, corpus_docs, val_pairs, tracker)

        if self.mode == "optuna" and val_pairs:
            logger.info("Подбор оптимального alpha через Optuna на %d валидационных парах...", len(val_pairs))
            bm25_corpus = self.bm25_retriever.encode_corpus(corpus_docs)
            dense_corpus = self.dense_retriever.encode_corpus(corpus_docs)

            val_queries = [p["query"] for p in val_pairs]
            target_hashes = [p["file_hash"] for p in val_pairs]
            hash_to_idx = {d["file_hash"]: idx for idx, d in enumerate(corpus_docs)}
            target_indices = [hash_to_idx[h] for h in target_hashes if h in hash_to_idx]

            bm25_query_repr = self.bm25_retriever.encode_queries(val_queries)
            bm25_scores = self.bm25_retriever.predict_scores(bm25_query_repr, bm25_corpus)

            dense_query_repr = self.dense_retriever.encode_queries(val_queries)
            dense_scores = self.dense_retriever.predict_scores(dense_query_repr, dense_corpus)

            optimizer = FusionOptimizer(n_trials=50)
            self.alpha = optimizer.fit(dense_scores, bm25_scores, target_indices)
            logger.info("Подобран оптимальный alpha: %.4f (Val MRR@5: %.4f)", self.alpha, optimizer.best_mrr or 0.0)

    def encode_corpus(self, corpus_docs: list[dict[str, Any]]) -> dict[str, Any]:
        """Encode corpus using both retrievers."""
        self.corpus_docs = corpus_docs
        bm25_repr = self.bm25_retriever.encode_corpus(corpus_docs)
        dense_repr = self.dense_retriever.encode_corpus(corpus_docs)
        return {"bm25": bm25_repr, "dense": dense_repr}

    def encode_queries(self, queries: list[str]) -> dict[str, Any]:
        """Encode queries using both retrievers."""
        self._current_queries = queries
        bm25_repr = self.bm25_retriever.encode_queries(queries)
        dense_repr = self.dense_retriever.encode_queries(queries)
        return {"bm25": bm25_repr, "dense": dense_repr}

    def predict_scores(self, query_repr: Any, corpus_repr: Any) -> np.ndarray:
        """Compute fused similarity scores."""
        bm25_scores = self.bm25_retriever.predict_scores(query_repr["bm25"], corpus_repr["bm25"])
        dense_scores = self.dense_retriever.predict_scores(query_repr["dense"], corpus_repr["dense"])

        if self.mode == "rrf":
            n_queries, n_docs = dense_scores.shape
            final_scores = np.zeros((n_queries, n_docs), dtype=np.float32)

            for q_idx in range(n_queries):
                dense_ranked = [
                    (str(d_idx), float(dense_scores[q_idx, d_idx]))
                    for d_idx in np.argsort(-dense_scores[q_idx])
                ]
                bm25_ranked = [
                    (str(d_idx), float(bm25_scores[q_idx, d_idx]))
                    for d_idx in np.argsort(-bm25_scores[q_idx])
                ]
                rrf_res = reciprocal_rank_fusion([dense_ranked, bm25_ranked], k=60)
                for d_str, score in rrf_res:
                    final_scores[q_idx, int(d_str)] = score
            return final_scores

        combined = combine_scores(dense_scores, bm25_scores, alpha=self.alpha, normalize=True)

        if self.mode == "rerank_ce" and self.reranker:
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
                    reranked_scores[q_idx, c_idx] = 0.5 * float(combined[q_idx, c_idx]) + 0.5 * float(norm_ce[rank_pos])

            return reranked_scores

        return combined

    def get_onnx_paths(self) -> list[Path]:
        """Return ONNX paths for both dense and reranker components."""
        paths = list(self.dense_retriever.get_onnx_paths())
        if self.reranker and self.reranker.onnx_path.exists():
            paths.append(self.reranker.onnx_path)
        return paths

    def get_model_size_mb(self) -> float:
        """Total disk footprint of hybrid pipeline."""
        size = self.bm25_retriever.get_model_size_mb() + self.dense_retriever.get_model_size_mb()
        if self.reranker and self.reranker.onnx_path.exists():
            size += self.reranker.onnx_path.stat().st_size / (1024 * 1024)
        return size
