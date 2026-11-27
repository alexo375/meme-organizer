"""Генератор дымовых тестов паритета мобильного клиента с Python."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pydantic import BaseModel, Field

from meme_organizer.retrieval.fusion.fusion_optimizer import FusionOptimizer
from meme_organizer.retrieval.text.bm25_retriever import BM25Retriever

if TYPE_CHECKING:
    import onnxruntime
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class SmokeSample(BaseModel):
    """Тестовый пример с ожидаемыми токенами и эмбеддингом."""

    sample_id: str
    text_type: str  # "query" or "passage"
    raw_text: str
    formatted_text: str
    bm25_tokens: list[str]
    input_ids: list[int]
    attention_mask: list[int]
    embedding_dim: int
    l2_norm: float
    embedding_head: list[float] = Field(description="First 10 dimensions of normalized embedding")
    full_embedding: list[float] = Field(description="Full embedding for exact match check")


class SmokePairTest(BaseModel):
    """Пара запроса и документа с ожидаемым косинусным сходством."""

    query_id: str
    passage_id: str
    query_text: str
    passage_text: str
    expected_cosine_similarity: float


class SmokeHybridRankingTest(BaseModel):
    """Тест гибридного ранжирования по 4 тестовым документам."""

    query_id: str
    query_text: str
    dense_weight: float
    bm25_weight: float
    passage_ids: list[str]
    dense_raw_scores: list[float]
    dense_norm_scores: list[float]
    bm25_raw_scores: list[float]
    bm25_norm_scores: list[float]
    hybrid_scores: list[float]
    expected_ranking: list[str]


class SmokeTestPack(BaseModel):
    """Набор эталонных данных для проверки паритета мобильной реализации."""

    description: str = Field(
        default="Golden parity test suite for mobile ONNX Runtime & Hybrid Ranking verification against Python"
    )
    model_name: str
    embedding_dimension: int
    samples: list[SmokeSample]
    similarity_tests: list[SmokePairTest]
    hybrid_ranking_tests: list[SmokeHybridRankingTest]
    fusion_tests: list[dict[str, Any]] = Field(default_factory=list)


class SmokeTestGenerator:
    """Генерирует эталонные тесты паритета с использованием сессии ONNX, BM25 и токенизатора."""

    SAMPLE_QUERIES: ClassVar[list[str]] = [
        "кот в капюшоне",
        "когда упал прод в пятницу",
        "паника и хаос в офисе",
        "жиза мем с собакой",
    ]

    SAMPLE_PASSAGES: ClassVar[list[str]] = [
        "Мем с грустным котом в капюшоне сидит и думает о жизни",
        "Инженер смотрит на упавший сервер в пятницу вечером перед релизом",
        "Собака сидит в горящей комнате и говорит this is fine",
        "Кот спит на клавиатуре ноутбука программиста",
    ]

    @classmethod
    def generate(
        cls,
        onnx_session: onnxruntime.InferenceSession,
        tokenizer: PreTrainedTokenizerBase,
        model_name: str,
        embedding_dim: int,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        max_length: int = 64,
        dense_weight: float = 0.52,
        bm25_weight: float = 0.48,
    ) -> SmokeTestPack:
        """Выполняет инференс, вычисляет оценки гибридного ранжирования и возвращает SmokeTestPack с эталонными данными."""
        samples: list[SmokeSample] = []
        sample_embs: dict[str, np.ndarray] = {}

        # 1. Process queries
        for idx, q in enumerate(cls.SAMPLE_QUERIES):
            sid = f"query_{idx + 1}"
            fmt = f"{query_prefix}{q}"
            s, emb = cls._run_single(
                session=onnx_session,
                tokenizer=tokenizer,
                sample_id=sid,
                text_type="query",
                raw_text=q,
                formatted_text=fmt,
                max_length=max_length,
                dim=embedding_dim,
            )
            samples.append(s)
            sample_embs[sid] = emb

        # 2. Process passages
        passage_ids: list[str] = []
        for idx, p in enumerate(cls.SAMPLE_PASSAGES):
            sid = f"passage_{idx + 1}"
            passage_ids.append(sid)
            fmt = f"{passage_prefix}{p}"
            s, emb = cls._run_single(
                session=onnx_session,
                tokenizer=tokenizer,
                sample_id=sid,
                text_type="passage",
                raw_text=p,
                formatted_text=fmt,
                max_length=max_length,
                dim=embedding_dim,
            )
            samples.append(s)
            sample_embs[sid] = emb

        # 3. Pairwise cosine similarity tests
        similarity_tests: list[SmokePairTest] = []
        pairs_to_test = [
            ("query_1", "passage_1"),
            ("query_2", "passage_2"),
            ("query_3", "passage_3"),
            ("query_1", "passage_2"),
        ]

        for q_id, p_id in pairs_to_test:
            sim = float(np.dot(sample_embs[q_id], sample_embs[p_id]))
            similarity_tests.append(
                SmokePairTest(
                    query_id=q_id,
                    passage_id=p_id,
                    query_text=next(s.raw_text for s in samples if s.sample_id == q_id),
                    passage_text=next(s.raw_text for s in samples if s.sample_id == p_id),
                    expected_cosine_similarity=round(sim, 4),
                )
            )

        # 4. End-to-end Hybrid Ranking parity tests over the 4 sample passages
        bm25 = BM25Retriever()
        mini_corpus = [{"user_caption": p} for p in cls.SAMPLE_PASSAGES]
        corpus_repr = bm25.encode_corpus(mini_corpus)
        q_repr = bm25.encode_queries(cls.SAMPLE_QUERIES)
        bm25_raw = bm25.predict_scores(q_repr, corpus_repr)

        q_mat = np.stack([sample_embs[f"query_{i + 1}"] for i in range(len(cls.SAMPLE_QUERIES))])
        p_mat = np.stack([sample_embs[f"passage_{i + 1}"] for i in range(len(cls.SAMPLE_PASSAGES))])
        dense_raw = np.matmul(q_mat, p_mat.T)

        dense_norm = FusionOptimizer.min_max_normalize(dense_raw)
        bm25_norm = FusionOptimizer.min_max_normalize(bm25_raw)
        hybrid_mat = FusionOptimizer.combine_scores(dense_raw, bm25_raw, alpha=dense_weight, normalize=True)

        hybrid_tests: list[SmokeHybridRankingTest] = []
        for idx, q_text in enumerate(cls.SAMPLE_QUERIES):
            order = np.argsort(-hybrid_mat[idx])
            expected_ranking = [passage_ids[int(j)] for j in order]
            hybrid_tests.append(
                SmokeHybridRankingTest(
                    query_id=f"query_{idx + 1}",
                    query_text=q_text,
                    dense_weight=dense_weight,
                    bm25_weight=bm25_weight,
                    passage_ids=passage_ids,
                    dense_raw_scores=[round(float(x), 5) for x in dense_raw[idx]],
                    dense_norm_scores=[round(float(x), 5) for x in dense_norm[idx]],
                    bm25_raw_scores=[round(float(x), 5) for x in bm25_raw[idx]],
                    bm25_norm_scores=[round(float(x), 5) for x in bm25_norm[idx]],
                    hybrid_scores=[round(float(x), 5) for x in hybrid_mat[idx]],
                    expected_ranking=expected_ranking,
                )
            )

        return SmokeTestPack(
            model_name=model_name,
            embedding_dimension=embedding_dim,
            samples=samples,
            similarity_tests=similarity_tests,
            hybrid_ranking_tests=hybrid_tests,
        )

    @staticmethod
    def _run_single(
        session: onnxruntime.InferenceSession,
        tokenizer: PreTrainedTokenizerBase,
        sample_id: str,
        text_type: str,
        raw_text: str,
        formatted_text: str,
        max_length: int,
        dim: int,
    ) -> tuple[SmokeSample, np.ndarray]:
        inputs = tokenizer(
            formatted_text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="np",
        )
        feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        outputs = session.run(["sentence_embedding"], feed)
        emb = outputs[0][0]  # shape: [dim]
        l2_norm = float(np.linalg.norm(emb))

        sample = SmokeSample(
            sample_id=sample_id,
            text_type=text_type,
            raw_text=raw_text,
            formatted_text=formatted_text,
            bm25_tokens=BM25Retriever.tokenize(raw_text),
            input_ids=inputs["input_ids"][0].tolist(),
            attention_mask=inputs["attention_mask"][0].tolist(),
            embedding_dim=dim,
            l2_norm=round(l2_norm, 5),
            embedding_head=[round(float(x), 5) for x in emb[:10]],
            full_embedding=[round(float(x), 6) for x in emb],
        )
        return sample, emb
