"""Smoke test generator for testing mobile parity with Python."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import onnxruntime
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class SmokeSample(BaseModel):
    """Single test sample with expected tokenization and embeddings."""

    sample_id: str
    text_type: str  # "query" or "passage"
    raw_text: str
    formatted_text: str
    input_ids: list[int]
    attention_mask: list[int]
    embedding_dim: int
    l2_norm: float
    embedding_head: list[float] = Field(description="First 10 dimensions of normalized embedding")
    full_embedding: list[float] = Field(description="Full 384-dim embedding for exact match check")


class SmokePairTest(BaseModel):
    """Pair of query and passage with expected cosine similarity."""

    query_id: str
    passage_id: str
    query_text: str
    passage_text: str
    expected_cosine_similarity: float


class SmokeTestPack(BaseModel):
    """Comprehensive parity verification package for mobile testing."""

    description: str = Field(
        default="Golden parity test suite for mobile ONNX Runtime verification against Python reference"
    )
    model_name: str
    embedding_dimension: int
    samples: list[SmokeSample]
    similarity_tests: list[SmokePairTest]


class SmokeTestGenerator:
    """Generates golden parity test samples using an ONNX session and tokenizer."""

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
    ) -> SmokeTestPack:
        """Run inferences and record exact golden input/output pairs."""
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
        for idx, p in enumerate(cls.SAMPLE_PASSAGES):
            sid = f"passage_{idx + 1}"
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

        # 3. Calculate pairwise cosine similarity for test pairs
        similarity_tests: list[SmokePairTest] = []
        pairs_to_test = [
            ("query_1", "passage_1"),  # кот в капюшоне -> Мем с грустным котом... (HIGH)
            ("query_2", "passage_2"),  # прод в пятницу -> упавший сервер в пятницу... (HIGH)
            ("query_3", "passage_3"),  # паника и хаос -> собака this is fine (MEDIUM/HIGH)
            ("query_1", "passage_2"),  # кот -> сервер в пятницу (LOW / IRRELEVANT)
        ]

        for q_id, p_id in pairs_to_test:
            q_emb = sample_embs[q_id]
            p_emb = sample_embs[p_id]
            sim = float(np.dot(q_emb, p_emb))
            similarity_tests.append(
                SmokePairTest(
                    query_id=q_id,
                    passage_id=p_id,
                    query_text=next(s.raw_text for s in samples if s.sample_id == q_id),
                    passage_text=next(s.raw_text for s in samples if s.sample_id == p_id),
                    expected_cosine_similarity=round(sim, 4),
                )
            )

        return SmokeTestPack(
            model_name=model_name,
            embedding_dimension=embedding_dim,
            samples=samples,
            similarity_tests=similarity_tests,
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
            input_ids=inputs["input_ids"][0].tolist(),
            attention_mask=inputs["attention_mask"][0].tolist(),
            embedding_dim=dim,
            l2_norm=round(l2_norm, 5),
            embedding_head=[round(float(x), 5) for x in emb[:10]],
            full_embedding=[round(float(x), 6) for x in emb],
        )
        return sample, emb
