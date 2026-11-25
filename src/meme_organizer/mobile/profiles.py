"""Preset model profiles for mobile bundle exports."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelBundleProfile:
    """Settings profile defining how to build a model bundle."""

    profile_id: str
    model_name: str
    base_hf_id: str
    embedding_dim: int
    max_length: int
    query_prefix: str
    passage_prefix: str
    dense_weight: float
    bm25_weight: float
    epochs: int = 50
    patience: int = 10
    lr: float = 2e-4
    batch_size: int = 32
    lora_r: int = 8
    lora_alpha: int = 16
    seed: int = 42
    extra_metadata: dict[str, str] = field(default_factory=dict)


# SOTA Champion: Fine-tuned multilingual-e5-small (weights will be refined/verified during build)
TEXT_CHAMPION = ModelBundleProfile(
    profile_id="text-champion",
    model_name="multilingual-e5-small-finetuned",
    base_hf_id="intfloat/multilingual-e5-small",
    embedding_dim=384,
    max_length=64,
    query_prefix="query: ",
    passage_prefix="passage: ",
    dense_weight=0.52,
    bm25_weight=0.48,
    epochs=50,
    patience=10,
    lr=2e-4,
    batch_size=32,
    lora_r=8,
    lora_alpha=16,
    seed=42,
)

# Lightweight RuBERT-tiny2 option (~15 MB, ultra-fast)
RUBERT_TINY2_FT = ModelBundleProfile(
    profile_id="rubert-tiny2-finetuned",
    model_name="rubert-tiny2-finetuned",
    base_hf_id="cointegrated/rubert-tiny2",
    embedding_dim=312,
    max_length=64,
    query_prefix="",
    passage_prefix="",
    dense_weight=0.50,
    bm25_weight=0.50,
    epochs=50,
    patience=10,
    lr=2e-4,
    batch_size=32,
    lora_r=8,
    lora_alpha=16,
    seed=42,
)

PROFILES: dict[str, ModelBundleProfile] = {
    TEXT_CHAMPION.profile_id: TEXT_CHAMPION,
    RUBERT_TINY2_FT.profile_id: RUBERT_TINY2_FT,
}
