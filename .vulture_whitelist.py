# Whitelist для vulture: поля Pydantic-моделей, контрактные методы и конфигурационные константы.
from meme_organizer.config import MlflowConfig
from meme_organizer.domain.base_retriever import BaseRetriever
from meme_organizer.domain.preset_manifest import (
    Bm25Config,
    DatasetInfo,
    EvaluationMetrics,
    ModelArtifact,
    PresetManifest,
    ScoringFormula,
    TokenizerConfig,
)
from meme_organizer.domain.schemas import GoldRecord, QueryPair
from meme_organizer.encoders.text.base_text_encoder import BaseTextEncoder
from meme_organizer.evaluation.metrics.experiment_tracker import ExperimentTracker
from meme_organizer.export.bundle.bundle_profiles import PROFILES
from meme_organizer.export.generators.smoke_test_generator import SmokeTestPack

_ = (
    MlflowConfig.experiment_text,
    MlflowConfig.experiment_visual,
    BaseRetriever.export_onnx,
    BaseTextEncoder.export_onnx,
    ExperimentTracker.log_optuna_trial,
    PROFILES,
    ScoringFormula.formula,
    ScoringFormula.bm25_params,
    TokenizerConfig.recommended_asset,
    TokenizerConfig.fallback_sentencepiece,
    TokenizerConfig.padding,
    TokenizerConfig.truncation,
    DatasetInfo.dataset_path,
    ModelArtifact.model_file,
    ModelArtifact.quantization,
    PresetManifest.metadata,
    Bm25Config,
    EvaluationMetrics,
    QueryPair.file_hash,
    GoldRecord.file_hash,
    GoldRecord.silver_filename,
    GoldRecord.original_filename,
    GoldRecord.duration_ms,
    GoldRecord.user_caption,
    GoldRecord.rich_description,
    GoldRecord.status,
    GoldRecord.extra,
    SmokeTestPack.bm25_tokens,
    SmokeTestPack.embedding_head,
    SmokeTestPack.full_embedding,
)
