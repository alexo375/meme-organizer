"""Preset manifest schema and data structures for Mobile Delivery Bundle."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HybridScoringConfig(BaseModel):
    """Configuration for client-side hybrid ranking."""

    formula: str = Field(
        default="0.52 * bm25_score_normalized + 0.48 * cosine_similarity",
        description="Scoring formula evaluated on device",
    )
    alpha_bm25: float = Field(default=0.52, description="Weight for lexical BM25 matching")
    beta_dense: float = Field(default=0.48, description="Weight for semantic dense embeddings")


class PresetManifest(BaseModel):
    """Declarative specification of the mobile model package contract."""

    schema_version: str = Field(default="1.0.0", description="Contract schema version")
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
    tokenizer_type: str = Field(
        default="SentencePiece / XLM-RoBERTa (tokenizer.json, sentencepiece.bpe.model)",
        description="Tokenizer specification",
    )
    hybrid_scoring: HybridScoringConfig = Field(default_factory=HybridScoringConfig)
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional metadata")
