"""Builder for compiling the complete Mobile Delivery Bundle."""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import tarfile
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime
import torch
from transformers import AutoModel, AutoTokenizer

from meme_organizer.config import AppConfig
from meme_organizer.dataset.loader import DatasetLoader
from meme_organizer.metrics.ranking import calculate_ranking_metrics
from meme_organizer.mobile.manifest import (
    BundleProvenance,
    HybridScoringConfig,
    PresetManifest,
    TokenizerSpec,
)
from meme_organizer.mobile.profiles import PROFILES, ModelBundleProfile
from meme_organizer.mobile.quantizer import OnnxQuantizer
from meme_organizer.mobile.readme_generator import ReadmeGenerator
from meme_organizer.mobile.smoke_test import SmokeTestGenerator
from meme_organizer.models.bm25 import BM25Retriever
from meme_organizer.models.dense import DenseTextRetriever
from meme_organizer.models.finetuned_dense import FineTunedDenseRetriever
from meme_organizer.models.fusion import FusionOptimizer, combine_scores
from meme_organizer.models.onnx_exporter import NormalizedEmbeddingPipeline, OnnxExporter

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class MobileBundleBuilder:
    """Builds and packages mobile bundle containing ONNX INT8, Tokenizer, Manifest, Smoke tests and README."""

    def __init__(self, profile: ModelBundleProfile, work_dir: Path, run_full_eval: bool = True) -> None:
        self.profile = profile
        self.work_dir = work_dir
        self.run_full_eval = run_full_eval
        self.bundle_dir = work_dir / f"meme-organizer-{profile.profile_id}"

    def build(self, output_tar: Path | None = None, release_tag: str | None = None) -> Path:
        """Run the complete bundle build process."""
        if self.bundle_dir.exists():
            shutil.rmtree(self.bundle_dir)
        self.bundle_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=== Запуск сборки Mobile Bundle: %s ===", self.profile.profile_id)
        config = AppConfig()

        # 1. Загрузка или дообучение модели
        model, tokenizer = self._prepare_model_and_tokenizer(config)

        # 2. Экспорт в чистый FP32 ONNX со встроенным Mean Pooling и L2 Normalization
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

        # 3. Отдельный этап мобильного квантования в INT8
        onnx_model_path = OnnxQuantizer.quantize_dynamic_int8(
            fp32_onnx_path=fp32_onnx_path,
            int8_onnx_path=self.bundle_dir / "text_encoder_int8.onnx",
            cleanup_fp32=True,
        )

        # 3. Экспорт токенизатора
        tokenizer_dir = self.bundle_dir / "tokenizer"
        logger.info("Экспорт файлов токенизатора в %s...", tokenizer_dir)
        OnnxExporter.export_tokenizer_assets(tokenizer, tokenizer_dir)

        # 4. Создание сессии ONNX INT8 и проверка паритета PyTorch FP32 <-> ONNX INT8
        session = onnxruntime.InferenceSession(str(onnx_model_path), providers=["CPUExecutionProvider"])
        mean_cos, max_diff = self._verify_pytorch_onnx_parity(model, tokenizer, session)
        logger.info(
            "Паритет PyTorch FP32 <-> ONNX INT8: mean_cosine=%.5f, max_abs_diff=%.5f",
            mean_cos,
            max_diff,
        )

        # 5. Честная оценка ONNX INT8 модели и калибровка весов гибрида (при run_full_eval=True)
        dense_weight = self.profile.dense_weight
        bm25_weight = self.profile.bm25_weight
        eval_metrics: dict[str, float] = {}

        if self.run_full_eval:
            dense_weight, bm25_weight, eval_metrics = self._evaluate_onnx_and_calibrate_hybrid(
                session=session,
                tokenizer=tokenizer,
                config=config,
            )

        # 6. Генерация Smoke-Test Pack (с учетом откалиброванных весов)
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

        # 7. Формирование Preset Manifest с provenance и точными параметрами токенизатора
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

        # 8. Генерация README.md для Android
        logger.info("Генерация документации README.md...")
        readme_content = ReadmeGenerator.generate(manifest)
        readme_path = self.bundle_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

        # 9. Упаковка в tar.gz
        if output_tar is None:
            output_tar = self.work_dir / f"meme-organizer-{self.profile.profile_id}.tar.gz"

        output_tar.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Архивация бандла в %s...", output_tar)
        with tarfile.open(output_tar, "w:gz") as tar:
            tar.add(self.bundle_dir, arcname=self.bundle_dir.name)

        size_mb = output_tar.stat().st_size / (1024 * 1024)
        logger.info("✅ Пакет поставки успешно собран: %s (размер: %.2f МБ)", output_tar, size_mb)
        return output_tar

    def _prepare_model_and_tokenizer(self, config: AppConfig) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Obtain fine-tuned FP32 model (from MLflow if available, else train) or base HF model."""
        if "finetuned" in self.profile.model_name:
            dense_loader = DenseTextRetriever(
                model_name=self.profile.model_name,
                hf_model_id=self.profile.base_hf_id,
                mlflow_run_name=f"{self.profile.model_name}_test",
            )
            dense_loader.load_or_build()
            if dense_loader._loaded_from_mlflow:
                logger.info("Используются чистые FP32 веса из MLflow для сборки бандла.")
                return dense_loader._model, dense_loader._tokenizer

            logger.info("Дообучение адаптера для модели %s (MLflow недоступен)...", self.profile.model_name)
            retriever = FineTunedDenseRetriever(
                model_name=self.profile.model_name,
                base_model_id=self.profile.base_hf_id,
                epochs=self.profile.epochs,
                patience=self.profile.patience,
                lr=self.profile.lr,
                batch_size=self.profile.batch_size,
                max_length=self.profile.max_length,
                lora_r=self.profile.lora_r,
                lora_alpha=self.profile.lora_alpha,
                query_prefix=self.profile.query_prefix,
                passage_prefix=self.profile.passage_prefix,
                seed=self.profile.seed,
            )
            train_pairs = DatasetLoader.load_query_pairs(config.splits_path, "train")
            val_pairs = DatasetLoader.load_query_pairs(config.splits_path, "val")
            corpus = DatasetLoader.load_corpus(config.gold_dataset_path)

            retriever.fit(train_pairs=train_pairs, corpus_docs=corpus, val_pairs=val_pairs)
            return retriever._model, retriever._tokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.profile.base_hf_id)
        model = AutoModel.from_pretrained(self.profile.base_hf_id)
        return model, tokenizer

    def _verify_pytorch_onnx_parity(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        session: onnxruntime.InferenceSession,
    ) -> tuple[float, float]:
        """Compare PyTorch FP32 reference embeddings against exported ONNX INT8."""
        test_texts = [f"{self.profile.query_prefix}{q}" for q in SmokeTestGenerator.SAMPLE_QUERIES] + [
            f"{self.profile.passage_prefix}{p}" for p in SmokeTestGenerator.SAMPLE_PASSAGES
        ]

        wrapper = NormalizedEmbeddingPipeline(model)
        wrapper.eval()

        enc = tokenizer(
            test_texts,
            padding="max_length",
            max_length=self.profile.max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            pt_embs = wrapper(enc["input_ids"], enc["attention_mask"]).numpy()

        onnx_embs = self._encode_onnx(session, tokenizer, test_texts)

        cos_sims = np.sum(pt_embs * onnx_embs, axis=1)
        mean_cos = float(np.mean(cos_sims))
        max_diff = float(np.max(np.abs(pt_embs - onnx_embs)))
        return mean_cos, max_diff

    def _encode_onnx(
        self,
        session: onnxruntime.InferenceSession,
        tokenizer: PreTrainedTokenizerBase,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode texts in batches using ONNX Runtime session."""
        all_embs: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding="max_length",
                max_length=self.profile.max_length,
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

    def _evaluate_onnx_and_calibrate_hybrid(
        self,
        session: onnxruntime.InferenceSession,
        tokenizer: PreTrainedTokenizerBase,
        config: AppConfig,
    ) -> tuple[float, float, dict[str, float]]:
        """Evaluate exported INT8 ONNX on val/test splits and calibrate Optuna hybrid weights."""
        corpus = DatasetLoader.load_corpus(config.gold_dataset_path)
        val_pairs = DatasetLoader.load_query_pairs(config.splits_path, "val")
        test_pairs = DatasetLoader.load_query_pairs(config.splits_path, "test")

        hash_to_idx = {d["file_hash"]: idx for idx, d in enumerate(corpus)}

        corpus_texts = [
            f"{self.profile.passage_prefix}"
            f"{d.get('user_caption', '')} {d.get('rich_description', '')} "
            f"{d.get('transcript_or_text', '')} {' '.join(d.get('tags', []))}".strip()
            for d in corpus
        ]
        corpus_embs = self._encode_onnx(session, tokenizer, corpus_texts)

        bm25 = BM25Retriever()
        bm25_corpus = bm25.encode_corpus(corpus)

        # Calibrate weights on Validation set using Optuna
        val_queries = [f"{self.profile.query_prefix}{p['query']}" for p in val_pairs]
        val_targets = [hash_to_idx[p["file_hash"]] for p in val_pairs]
        val_embs = self._encode_onnx(session, tokenizer, val_queries)
        val_dense_scores = np.matmul(val_embs, corpus_embs.T)
        val_bm25_scores = bm25.predict_scores(bm25.encode_queries([p["query"] for p in val_pairs]), bm25_corpus)

        optimizer = FusionOptimizer(n_trials=50)
        best_dense_weight = optimizer.fit(val_dense_scores, val_bm25_scores, val_targets)
        best_bm25_weight = 1.0 - best_dense_weight

        # Evaluate on Test set
        test_queries = [f"{self.profile.query_prefix}{p['query']}" for p in test_pairs]
        test_targets = [hash_to_idx[p["file_hash"]] for p in test_pairs]
        test_embs = self._encode_onnx(session, tokenizer, test_queries)
        test_dense_scores = np.matmul(test_embs, corpus_embs.T)
        test_bm25_scores = bm25.predict_scores(bm25.encode_queries([p["query"] for p in test_pairs]), bm25_corpus)

        dense_metrics = calculate_ranking_metrics(test_dense_scores, test_targets, top_k=5)
        hybrid_scores = combine_scores(test_dense_scores, test_bm25_scores, alpha=best_dense_weight, normalize=True)
        hybrid_metrics = calculate_ranking_metrics(hybrid_scores, test_targets, top_k=5)

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


def get_profile(profile_id: str) -> ModelBundleProfile:
    """Get profile by ID or raise KeyError."""
    if profile_id not in PROFILES:
        valid = ", ".join(PROFILES.keys())
        raise KeyError(f"Неизвестный профиль '{profile_id}'. Доступные: {valid}")
    return PROFILES[profile_id]
