"""Builder for compiling the complete Mobile Delivery Bundle."""

from __future__ import annotations

import logging
import shutil
import tarfile
from typing import TYPE_CHECKING

import onnxruntime
from transformers import AutoModel, AutoTokenizer

from meme_organizer.config import AppConfig
from meme_organizer.dataset.loader import DatasetLoader
from meme_organizer.mobile.manifest import HybridScoringConfig, PresetManifest
from meme_organizer.mobile.profiles import PROFILES, ModelBundleProfile
from meme_organizer.mobile.readme_generator import ReadmeGenerator
from meme_organizer.mobile.smoke_test import SmokeTestGenerator
from meme_organizer.models.finetuned_dense import FineTunedDenseRetriever
from meme_organizer.models.onnx_exporter import OnnxExporter

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class MobileBundleBuilder:
    """Builds and packages mobile bundle containing ONNX INT8, Tokenizer, Manifest, Smoke tests and README."""

    def __init__(self, profile: ModelBundleProfile, work_dir: Path) -> None:
        self.profile = profile
        self.work_dir = work_dir
        self.bundle_dir = work_dir / f"meme-organizer-{profile.profile_id}"

    def build(self, output_tar: Path | None = None) -> Path:
        """Run the complete bundle build process."""
        if self.bundle_dir.exists():
            shutil.rmtree(self.bundle_dir)
        self.bundle_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=== Запуск сборки Mobile Bundle: %s ===", self.profile.profile_id)

        # 1. Загрузка или дообучение модели
        model, tokenizer = self._prepare_model_and_tokenizer()

        # 2. Экспорт в ONNX INT8 со встроенным Mean Pooling и L2 Normalization
        onnx_model_path = self.bundle_dir / "text_encoder_int8.onnx"
        logger.info("Экспорт ONNX с in-graph Mean-Pooling & L2 Norm...")
        exported_path = OnnxExporter.export_text_encoder(
            model=model,
            tokenizer=tokenizer,
            output_dir=self.bundle_dir,
            base_name="text_encoder",
            max_length=self.profile.max_length,
            include_pooling=True,
        )
        if exported_path != onnx_model_path:
            shutil.move(exported_path, onnx_model_path)

        # 3. Экспорт токенизатора
        tokenizer_dir = self.bundle_dir / "tokenizer"
        logger.info("Экспорт файлов токенизатора в %s...", tokenizer_dir)
        OnnxExporter.export_tokenizer_assets(tokenizer, tokenizer_dir)

        # 4. Создание сессии ONNX и генерация Smoke-Test Pack
        logger.info("Генерация Smoke-Test Pack для тестирования паритета на Android...")
        session = onnxruntime.InferenceSession(str(onnx_model_path), providers=["CPUExecutionProvider"])
        smoke_pack = SmokeTestGenerator.generate(
            onnx_session=session,
            tokenizer=tokenizer,
            model_name=self.profile.model_name,
            embedding_dim=self.profile.embedding_dim,
            query_prefix=self.profile.query_prefix,
            passage_prefix=self.profile.passage_prefix,
            max_length=self.profile.max_length,
        )
        smoke_path = self.bundle_dir / "smoke_test_pack.json"
        with open(smoke_path, "w", encoding="utf-8") as f:
            f.write(smoke_pack.model_dump_json(indent=2))

        # 5. Генерация Preset Manifest
        logger.info("Генерация preset_manifest.json...")
        manifest = PresetManifest(
            model_name=self.profile.model_name,
            model_file="text_encoder_int8.onnx",
            embedding_dimension=self.profile.embedding_dim,
            max_sequence_length=self.profile.max_length,
            prefixes={
                "query": self.profile.query_prefix,
                "passage": self.profile.passage_prefix,
            },
            hybrid_scoring=HybridScoringConfig(
                alpha_bm25=self.profile.alpha_bm25,
                beta_dense=self.profile.beta_dense,
                formula=f"{self.profile.alpha_bm25} * bm25_score_normalized + {self.profile.beta_dense} * cosine_similarity",
            ),
            metadata=self.profile.extra_metadata,
        )
        manifest_path = self.bundle_dir / "preset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))

        # 6. Генерация README.md для Android
        logger.info("Генерация документации README.md...")
        readme_content = ReadmeGenerator.generate(manifest)
        readme_path = self.bundle_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

        # 7. Упаковка в tar.gz
        if output_tar is None:
            output_tar = self.work_dir / f"meme-organizer-{self.profile.profile_id}.tar.gz"

        output_tar.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Архивация бандла в %s...", output_tar)
        with tarfile.open(output_tar, "w:gz") as tar:
            tar.add(self.bundle_dir, arcname=self.bundle_dir.name)

        size_mb = output_tar.stat().st_size / (1024 * 1024)
        logger.info("✅ Пакет поставки успешно собран: %s (размер: %.2f МБ)", output_tar, size_mb)
        return output_tar

    def _prepare_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Obtain fine-tuned or base model and tokenizer."""
        config = AppConfig()
        tokenizer = AutoTokenizer.from_pretrained(self.profile.base_hf_id)

        # Если это fine-tuned модель, обучаем ее через LoRA
        if "finetuned" in self.profile.model_name:
            logger.info("Дообучение адаптера для модели %s...", self.profile.model_name)
            retriever = FineTunedDenseRetriever(
                model_name=self.profile.model_name,
                base_model_id=self.profile.base_hf_id,
                epochs=self.profile.epochs,
                lr=self.profile.lr,
                batch_size=self.profile.batch_size,
                max_length=self.profile.max_length,
                lora_r=self.profile.lora_r,
                lora_alpha=self.profile.lora_alpha,
                query_prefix=self.profile.query_prefix,
                passage_prefix=self.profile.passage_prefix,
            )
            train_pairs = DatasetLoader.load_query_pairs(config.splits_path, "train")
            val_pairs = DatasetLoader.load_query_pairs(config.splits_path, "val")
            corpus = DatasetLoader.load_corpus(config.gold_dataset_path)

            retriever.fit(train_pairs=train_pairs, corpus_docs=corpus, val_pairs=val_pairs)
            return retriever._model, tokenizer

        model = AutoModel.from_pretrained(self.profile.base_hf_id)
        return model, tokenizer


def get_profile(profile_id: str) -> ModelBundleProfile:
    """Get profile by ID or raise KeyError."""
    if profile_id not in PROFILES:
        valid = ", ".join(PROFILES.keys())
        raise KeyError(f"Неизвестный профиль '{profile_id}'. Доступные: {valid}")
    return PROFILES[profile_id]
