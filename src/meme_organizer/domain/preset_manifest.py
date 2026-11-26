"""Схемы манифеста и структур данных для мобильного пакета модели."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HybridScoringConfig(BaseModel):
    """Параметры гибридного ранжирования на стороне клиента."""

    formula: str = Field(
        default="dense_weight * minmax(cosine_similarity) + bm25_weight * minmax(bm25_score)",
        description="Exact scoring formula matching Python fusion.py",
    )
    dense_weight: float = Field(default=0.52, description="Weight for min-max normalized dense cosine similarity")
    bm25_weight: float = Field(default=0.48, description="Weight for min-max normalized lexical BM25 score")
    normalization: str = Field(
        default="per_query_min_max",
        description="Both channels must be min-max normalized into [0, 1] across candidate docs per query",
    )
    bm25_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "algorithm": "BM25Okapi",
            "k1": 1.5,
            "b": 0.75,
            "epsilon": 0.25,
            "tokenizer_regex": r"[^\w]+",
            "min_token_length": 2,
            "lowercase": True,
            "indexed_fields": ["user_caption", "rich_description", "transcript_or_text", "tags"],
        },
        description="BM25 parameters matching Python BM25Retriever",
    )


class TokenizerSpec(BaseModel):
    """Параметры токенизации и идентификаторы служебных токенов для мобильного клиента."""

    recommended_asset: str = Field(default="tokenizer/tokenizer.json")
    fallback_sentencepiece: str = Field(default="tokenizer/sentencepiece.bpe.model")
    padding: str = Field(default="max_length")
    truncation: bool = Field(default=True)
    max_length: int = Field(default=64)
    special_token_ids: dict[str, int] = Field(
        default_factory=lambda: {"bos": 0, "pad": 1, "eos": 2, "unk": 3},
        description="Special token IDs used by the model tokenizer",
    )


class BundleProvenance(BaseModel):
    """Метаданные сборки, контрольные суммы и метрики верификации пакета."""

    git_commit: str
    dataset_path: str
    splits_path: str
    onnx_sha256: str
    pytorch_onnx_mean_cosine_similarity: float
    pytorch_onnx_max_abs_diff: float
    evaluated_onnx_test_metrics: dict[str, float]


class PresetManifest(BaseModel):
    """Спецификация контракта мобильного пакета модели."""

    schema_version: str = Field(default="1.1.0", description="Contract schema version")
    release_tag: str | None = Field(default=None, description="GitHub Release tag (e.g. v0.2.0-mobile)")
    component: str = Field(default="text_encoder", description="Component name")
    model_name: str = Field(..., description="Canonical model identifier")
    model_file: str = Field(default="text_encoder_int8.onnx", description="ONNX binary filename")
    quantization: str = Field(
        default="INT8 Dynamic (CPUExecutionProvider)",
        description="Quantization type and execution target",
    )
    embedding_dimension: int = Field(default=384, description="Output embedding vector length")
    max_sequence_length: int = Field(default=64, description="Max token sequence length")
    inputs: dict[str, str] = Field(
        default_factory=lambda: {
            "input_ids": "int64[batch_size, sequence_length]",
            "attention_mask": "int64[batch_size, sequence_length]",
        }
    )
    outputs: dict[str, str] = Field(
        default_factory=lambda: {
            "sentence_embedding": "float32[batch_size, embedding_dimension]",
        }
    )
    prefixes: dict[str, str] = Field(
        default_factory=lambda: {
            "query": "query: ",
            "passage": "passage: ",
        }
    )
    tokenizer: TokenizerSpec = Field(default_factory=TokenizerSpec)
    hybrid_scoring: HybridScoringConfig = Field(default_factory=HybridScoringConfig)
    provenance: BundleProvenance | None = Field(default=None, description="Measured metrics and build checksums")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional metadata")
