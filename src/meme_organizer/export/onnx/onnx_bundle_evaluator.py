"""Оценка паритета и качества квантованной ONNX-модели на валидации и тесте."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from meme_organizer.data.dataset_loader import DatasetLoader
from meme_organizer.domain.ranking_metrics import RankingMetrics
from meme_organizer.export.generators.smoke_test_generator import SmokeTestGenerator
from meme_organizer.export.onnx.onnx_exporter import NormalizedEmbeddingPipeline
from meme_organizer.retrieval.fusion.fusion_optimizer import FusionOptimizer
from meme_organizer.retrieval.text.bm25_retriever import BM25Retriever

if TYPE_CHECKING:
    import onnxruntime
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from meme_organizer.config import AppConfig
    from meme_organizer.export.bundle.bundle_profiles import ModelBundleProfile

logger = logging.getLogger(__name__)


class OnnxBundleEvaluator:
    """Проверяет паритет PyTorch/ONNX и калибрует веса гибридного поиска."""

    @staticmethod
    def verify_pytorch_onnx_parity(
        profile: ModelBundleProfile,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        session: onnxruntime.InferenceSession,
    ) -> tuple[float, float]:
        """Сравнивает эталонные эмбеддинги PyTorch FP32 с квантованным ONNX INT8.

        Вход: профиль, PyTorch-модель, токенизатор, сессия ONNX Runtime.
        Выход: кортеж (среднее косинусное сходство, максимальное абсолютное отклонение).
        """
        test_texts = [f"{profile.query_prefix}{q}" for q in SmokeTestGenerator.SAMPLE_QUERIES] + [
            f"{profile.passage_prefix}{p}" for p in SmokeTestGenerator.SAMPLE_PASSAGES
        ]

        wrapper = NormalizedEmbeddingPipeline(model)
        wrapper.eval()

        enc = tokenizer(
            test_texts,
            padding="max_length",
            max_length=profile.max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            pt_embs = wrapper(enc["input_ids"], enc["attention_mask"]).numpy()

        onnx_embs = OnnxBundleEvaluator.encode_onnx(profile, session, tokenizer, test_texts)

        cos_sims = np.sum(pt_embs * onnx_embs, axis=1)
        mean_cos = float(np.mean(cos_sims))
        max_diff = float(np.max(np.abs(pt_embs - onnx_embs)))
        return mean_cos, max_diff

    @staticmethod
    def encode_onnx(
        profile: ModelBundleProfile,
        session: onnxruntime.InferenceSession,
        tokenizer: PreTrainedTokenizerBase,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Кодирует тексты батчами через сессию ONNX Runtime.

        Вход: профиль, сессия ONNX, токенизатор, список текстов, размер батча.
        Выход: матрица эмбеддингов.
        """
        all_embs: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding="max_length",
                max_length=profile.max_length,
                truncation=True,
                return_tensors="np",
            )
            feed = {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            }
            out = session.run(["sentence_embedding"], feed)[0]
            all_embs.append(out)
        return np.vstack(all_embs)

    @staticmethod
    def evaluate_and_calibrate_hybrid(
        profile: ModelBundleProfile,
        session: onnxruntime.InferenceSession,
        tokenizer: PreTrainedTokenizerBase,
        config: AppConfig,
    ) -> tuple[float, float, dict[str, float]]:
        """Оценивает ONNX INT8 на val/test и калибрует веса гибридного поиска через Optuna.

        Вход: профиль, сессия ONNX, токенизатор, конфигурация приложения.
        Выход: кортеж (вес dense, вес bm25, словарь метрик).
        """
        corpus = DatasetLoader.load_corpus(config.gold_dataset_path)
        val_pairs = DatasetLoader.load_query_pairs(config.splits_path, "val")
        test_pairs = DatasetLoader.load_query_pairs(config.splits_path, "test")

        hash_to_idx = {d["file_hash"]: idx for idx, d in enumerate(corpus)}

        corpus_texts = [
            f"{profile.passage_prefix}"
            f"{d.get('user_caption', '')} {d.get('rich_description', '')} "
            f"{d.get('transcript_or_text', '')} {' '.join(d.get('tags', []))}".strip()
            for d in corpus
        ]
        corpus_embs = OnnxBundleEvaluator.encode_onnx(profile, session, tokenizer, corpus_texts)

        bm25 = BM25Retriever()
        bm25_corpus = bm25.encode_corpus(corpus)

        val_queries = [f"{profile.query_prefix}{p['query']}" for p in val_pairs]
        val_targets = [hash_to_idx[p["file_hash"]] for p in val_pairs]
        val_embs = OnnxBundleEvaluator.encode_onnx(profile, session, tokenizer, val_queries)
        val_dense_scores = np.matmul(val_embs, corpus_embs.T)
        val_bm25_scores = bm25.predict_scores(bm25.encode_queries([p["query"] for p in val_pairs]), bm25_corpus)

        optimizer = FusionOptimizer(n_trials=50)
        best_dense_weight = optimizer.fit(val_dense_scores, val_bm25_scores, val_targets)
        best_bm25_weight = 1.0 - best_dense_weight

        test_queries = [f"{profile.query_prefix}{p['query']}" for p in test_pairs]
        test_targets = [hash_to_idx[p["file_hash"]] for p in test_pairs]
        test_embs = OnnxBundleEvaluator.encode_onnx(profile, session, tokenizer, test_queries)
        test_dense_scores = np.matmul(test_embs, corpus_embs.T)
        test_bm25_scores = bm25.predict_scores(bm25.encode_queries([p["query"] for p in test_pairs]), bm25_corpus)

        dense_metrics = RankingMetrics.calculate(test_dense_scores, test_targets, top_k=5)
        hybrid_scores = FusionOptimizer.combine_scores(
            test_dense_scores, test_bm25_scores, alpha=best_dense_weight, normalize=True
        )
        hybrid_metrics = RankingMetrics.calculate(hybrid_scores, test_targets, top_k=5)

        metrics_dict = {
            "onnx_dense_mrr_at_5": round(dense_metrics.mrr_at_5, 4),
            "onnx_dense_recall_at_1": round(dense_metrics.recall_at_1, 4),
            "onnx_dense_recall_at_5": round(dense_metrics.recall_at_5, 4),
            "onnx_hybrid_mrr_at_5": round(hybrid_metrics.mrr_at_5, 4),
            "onnx_hybrid_recall_at_1": round(hybrid_metrics.recall_at_1, 4),
            "onnx_hybrid_recall_at_5": round(hybrid_metrics.recall_at_5, 4),
            "onnx_hybrid_ndcg_at_5": round(hybrid_metrics.ndcg_at_5, 4),
        }
        logger.info("Измеренные ONNX INT8 метрики (Test Set, %d запросов): %s", len(test_pairs), metrics_dict)
        return best_dense_weight, best_bm25_weight, metrics_dict
