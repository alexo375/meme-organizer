"""Сборщик мобильного пакета поставки (ONNX INT8, токенизатор, манифест, smoke-тесты, README)."""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import tarfile
from typing import TYPE_CHECKING

import onnxruntime

from meme_organizer.config import AppConfig
from meme_organizer.domain.preset_manifest import (
    BundleProvenance,
    HybridScoringConfig,
    PresetManifest,
    TokenizerSpec,
)
from meme_organizer.export.bundle.bundle_model_loader import BundleModelLoader
from meme_organizer.export.bundle.bundle_profiles import PROFILES, ModelBundleProfile
from meme_organizer.export.generators.readme_generator import ReadmeGenerator
from meme_organizer.export.generators.smoke_test_generator import SmokeTestGenerator
from meme_organizer.export.onnx.onnx_bundle_evaluator import OnnxBundleEvaluator
from meme_organizer.export.onnx.onnx_exporter import OnnxExporter
from meme_organizer.export.onnx.onnx_quantizer import OnnxQuantizer

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class MobileBundleBuilder:
    """Оркестратор сборки и упаковки мобильного бандла."""

    def __init__(
        self,
        profile: ModelBundleProfile,
        work_dir: Path,
        run_full_eval: bool = True,
        model_loader: BundleModelLoader | None = None,
        evaluator: OnnxBundleEvaluator | None = None,
    ) -> None:
        """Инициализирует сборщик бандла с явным внедрением зависимостей."""
        self.profile = profile
        self.work_dir = work_dir
        self.run_full_eval = run_full_eval
        self.bundle_dir = work_dir / f"meme-organizer-{profile.profile_id}"
        self.model_loader = model_loader or BundleModelLoader()
        self.evaluator = evaluator or OnnxBundleEvaluator()

    def build(self, output_tar: Path | None = None, release_tag: str | None = None) -> Path:
        """Выполняет полный цикл сборки, оценки и архивации мобильного бандла.

        Вход: опциональный путь к tar.gz и тег релиза.
        Выход: путь к собранному архиву tar.gz.
        """
        if self.bundle_dir.exists():
            shutil.rmtree(self.bundle_dir)
        self.bundle_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=== Запуск сборки Mobile Bundle: %s ===", self.profile.profile_id)
        config = AppConfig()

        model, tokenizer = self.model_loader.load_or_train(self.profile, config)

        logger.info("Экспорт FP32 ONNX с in-graph Mean-Pooling & L2 Norm...")
        fp32_onnx_path = OnnxExporter.export_to_fp32_onnx(
            model=model,
            tokenizer=tokenizer,
            output_dir=self.bundle_dir,
            base_name="text_encoder",
            max_length=self.profile.max_length,
            include_pooling=True,
            overwrite=True,
        )

        onnx_model_path = OnnxQuantizer.quantize_dynamic_int8(
            fp32_onnx_path=fp32_onnx_path,
            int8_onnx_path=self.bundle_dir / "text_encoder_int8.onnx",
            cleanup_fp32=True,
        )

        tokenizer_dir = self.bundle_dir / "tokenizer"
        logger.info("Экспорт файлов токенизатора в %s...", tokenizer_dir)
        OnnxExporter.export_tokenizer_assets(tokenizer, tokenizer_dir)

        session = onnxruntime.InferenceSession(str(onnx_model_path), providers=["CPUExecutionProvider"])
        mean_cos, max_diff = self.evaluator.verify_pytorch_onnx_parity(self.profile, model, tokenizer, session)
        logger.info(
            "Паритет PyTorch FP32 <-> ONNX INT8: mean_cosine=%.5f, max_abs_diff=%.5f",
            mean_cos,
            max_diff,
        )

        dense_weight = self.profile.dense_weight
        bm25_weight = self.profile.bm25_weight
        eval_metrics: dict[str, float] = {}

        if self.run_full_eval:
            dense_weight, bm25_weight, eval_metrics = self.evaluator.evaluate_and_calibrate_hybrid(
                profile=self.profile,
                session=session,
                tokenizer=tokenizer,
                config=config,
            )

        logger.info("Генерация Smoke-Test Pack для тестирования паритета на Android...")
        smoke_pack = SmokeTestGenerator.generate(
            onnx_session=session,
            tokenizer=tokenizer,
            model_name=self.profile.model_name,
            embedding_dim=self.profile.embedding_dim,
            query_prefix=self.profile.query_prefix,
            passage_prefix=self.profile.passage_prefix,
            max_length=self.profile.max_length,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )
        smoke_path = self.bundle_dir / "smoke_test_pack.json"
        with open(smoke_path, "w", encoding="utf-8") as f:
            f.write(smoke_pack.model_dump_json(indent=2))

        logger.info("Генерация preset_manifest.json...")
        provenance = BundleProvenance(
            git_commit=self._get_git_commit(),
            dataset_path=str(config.gold_dataset_path),
            splits_path=str(config.splits_path),
            onnx_sha256=self._compute_sha256(onnx_model_path),
            pytorch_onnx_mean_cosine_similarity=round(mean_cos, 5),
            pytorch_onnx_max_abs_diff=round(max_diff, 5),
            evaluated_onnx_test_metrics=eval_metrics,
        )
        tokenizer_spec = TokenizerSpec(
            max_length=self.profile.max_length,
            special_token_ids={
                "bos": int(tokenizer.bos_token_id or 0),
                "pad": int(tokenizer.pad_token_id or 1),
                "eos": int(tokenizer.eos_token_id or 2),
                "unk": int(tokenizer.unk_token_id or 3),
            },
        )
        manifest = PresetManifest(
            release_tag=release_tag,
            model_name=self.profile.model_name,
            model_file="text_encoder_int8.onnx",
            embedding_dimension=self.profile.embedding_dim,
            max_sequence_length=self.profile.max_length,
            prefixes={
                "query": self.profile.query_prefix,
                "passage": self.profile.passage_prefix,
            },
            tokenizer=tokenizer_spec,
            hybrid_scoring=HybridScoringConfig(
                dense_weight=round(dense_weight, 4),
                bm25_weight=round(bm25_weight, 4),
                formula=f"{round(dense_weight, 4)} * minmax(cosine_similarity) + {round(bm25_weight, 4)} * minmax(bm25_score)",
            ),
            provenance=provenance,
            metadata=self.profile.extra_metadata,
        )
        manifest_path = self.bundle_dir / "preset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))

        logger.info("Генерация документации README.md...")
        readme_content = ReadmeGenerator.generate(manifest)
        readme_path = self.bundle_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

        if output_tar is None:
            output_tar = self.work_dir / f"meme-organizer-{self.profile.profile_id}.tar.gz"

        output_tar.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Архивация бандла в %s...", output_tar)
        with tarfile.open(output_tar, "w:gz") as tar:
            tar.add(self.bundle_dir, arcname=self.bundle_dir.name)

        size_mb = output_tar.stat().st_size / (1024 * 1024)
        logger.info("✅ Пакет поставки успешно собран: %s (размер: %.2f МБ)", output_tar, size_mb)
        return output_tar

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _get_git_commit() -> str:
        git_bin = shutil.which("git")
        if not git_bin:
            return "unknown"
        try:
            out = subprocess.check_output(  # noqa: S603
                [git_bin, "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            return out.decode("utf-8").strip()
        except (subprocess.SubprocessError, OSError):
            return "unknown"

    @staticmethod
    def get_profile(profile_id: str) -> ModelBundleProfile:
        """Возвращает профиль сборки по идентификатору или выбрасывает KeyError."""
        if profile_id not in PROFILES:
            valid = ", ".join(PROFILES.keys())
            raise KeyError(f"Неизвестный профиль '{profile_id}'. Доступные: {valid}")
        return PROFILES[profile_id]
