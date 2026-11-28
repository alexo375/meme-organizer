"""Сборщик мобильного пакета поставки (ONNX INT8, токенизатор, манифест, smoke-тесты, README)."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import shutil
import subprocess
import tarfile
import wave
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime
import rapidocr_onnxruntime
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

from meme_organizer.config import AppConfig, MlflowConfig
from meme_organizer.domain.preset_manifest import (
    BundleProvenance,
    ComponentSpec,
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
from meme_organizer.retrieval.fusion.gated_attention_fusion import GatedAttentionFusion

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
        components: list[ComponentSpec] = [self._build_text_encoder_spec(onnx_model_path)]
        if isinstance(self.profile.extra_metadata.get("domains"), dict):
            components.extend(self._bundle_multimodal_assets(config))
            self._append_multimodal_smoke_tests()

        manifest_metadata = {**self.profile.extra_metadata}
        if release_tag:
            manifest_metadata["release_tag"] = release_tag

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
            components=components,
            metadata=manifest_metadata,
        )
        manifest_path = self.bundle_dir / "preset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))

        logger.info("Генерация документации README.md...")
        readme_content = ReadmeGenerator.generate(manifest)
        readme_path = self.bundle_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

        return self._archive_bundle(output_tar)

    def _archive_bundle(self, output_tar: Path | None) -> Path:
        """Архивирует собранную директорию бандла в tar.gz."""
        if output_tar is None:
            output_tar = self.work_dir / f"meme-organizer-{self.profile.profile_id}.tar.gz"
        output_tar.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Архивация бандла в %s...", output_tar)
        with tarfile.open(output_tar, "w:gz") as tar:
            tar.add(self.bundle_dir, arcname=self.bundle_dir.name)
        size_mb = output_tar.stat().st_size / (1024 * 1024)
        logger.info("✅ Пакет поставки успешно собран: %s (размер: %.2f МБ)", output_tar, size_mb)
        return output_tar

    def _build_text_encoder_spec(self, onnx_model_path: Path) -> ComponentSpec:
        """Формирует спецификацию текстового энкодера для манифеста."""
        tok_path = self.bundle_dir / "tokenizer" / "tokenizer.json"
        aux = {"tokenizer/tokenizer.json": self._compute_sha256(tok_path)} if tok_path.exists() else {}
        return ComponentSpec(
            name="text_encoder",
            path="text_encoder_int8.onnx",
            sha256=self._compute_sha256(onnx_model_path),
            runtime="onnxruntime",
            inputs=[
                {"name": "input_ids", "dtype": "int64", "shape": ["batch", self.profile.max_length]},
                {"name": "attention_mask", "dtype": "int64", "shape": ["batch", self.profile.max_length]},
            ],
            outputs=[
                {"name": "sentence_embedding", "dtype": "float32", "shape": ["batch", self.profile.embedding_dim]},
            ],
            auxiliary_files=aux,
        )

    def _bundle_multimodal_assets(self, config: AppConfig) -> list[ComponentSpec]:
        """Упаковывает OCR, ASR и GatedAttentionFusion ONNX в мультимодальный бандл."""
        specs: list[ComponentSpec] = []

        # 1. OCR (detector + recognizer)
        ocr_dst = self.bundle_dir / "ocr"
        ocr_dst.mkdir(parents=True, exist_ok=True)
        ocr_src = Path("models/ocr/cyrillic_ppocrv5")
        for fname in ("inference.onnx", "dict.txt"):
            if (ocr_src / fname).exists():
                shutil.copy2(ocr_src / fname, ocr_dst / fname)
        det_candidates = list(Path(rapidocr_onnxruntime.__file__).parent.rglob("*det*.onnx"))
        if det_candidates:
            shutil.copy2(det_candidates[0], ocr_dst / "det.onnx")
            specs.append(
                ComponentSpec(
                    name="ocr_det",
                    path="ocr/det.onnx",
                    sha256=self._compute_sha256(ocr_dst / "det.onnx"),
                    runtime="onnxruntime",
                    inputs=[{"name": "x", "dtype": "float32", "shape": [1, 3, "h", "w"]}],
                    outputs=[{"name": "sigmoid_0.tmp_0", "dtype": "float32", "shape": [1, 1, "h", "w"]}],
                    preprocessing={
                        "color": "BGR",
                        "scale": 1.0 / 255.0,
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                        "resize_multiple": 32,
                        "thresh": 0.3,
                        "box_thresh": 0.5,
                        "unclip_ratio": 1.6,
                        "max_candidates": 1000,
                    },
                )
            )
        if (ocr_dst / "inference.onnx").exists():
            dict_path = ocr_dst / "dict.txt"
            ocr_aux = {"ocr/dict.txt": self._compute_sha256(dict_path)} if dict_path.exists() else {}
            specs.append(
                ComponentSpec(
                    name="ocr_rec",
                    path="ocr/inference.onnx",
                    sha256=self._compute_sha256(ocr_dst / "inference.onnx"),
                    runtime="onnxruntime",
                    inputs=[{"name": "x", "dtype": "float32", "shape": ["batch", 3, 48, "w"]}],
                    outputs=[{"name": "softmax_2.tmp_0", "dtype": "float32", "shape": ["batch", "seq", "vocab"]}],
                    preprocessing={
                        "height": 48,
                        "padding_value": 0,
                        "mean": [0.5, 0.5, 0.5],
                        "std": [0.5, 0.5, 0.5],
                        "decoding": "ctc_greedy",
                    },
                    auxiliary_files=ocr_aux,
                )
            )

        # 2. ASR (GigaAM-CTC INT8)
        asr_dst = self.bundle_dir / "asr"
        asr_dst.mkdir(parents=True, exist_ok=True)
        asr_repo = "csukuangfj/sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24"
        for fname in ("model.int8.onnx", "tokens.txt"):
            src_file = hf_hub_download(repo_id=asr_repo, filename=fname, revision="main")  # nosec B615
            shutil.copy2(src_file, asr_dst / fname)
        tok_asr_path = asr_dst / "tokens.txt"
        asr_aux = {"asr/tokens.txt": self._compute_sha256(tok_asr_path)} if tok_asr_path.exists() else {}
        specs.append(
            ComponentSpec(
                name="asr",
                path="asr/model.int8.onnx",
                sha256=self._compute_sha256(asr_dst / "model.int8.onnx"),
                runtime="sherpa-onnx:1.10.37",
                inputs=[{"name": "audio_signal", "dtype": "float32", "shape": ["batch", 80, "frames"]}],
                outputs=[{"name": "logprobs", "dtype": "float32", "shape": ["batch", "t", "vocab"]}],
                preprocessing={
                    "sample_rate": 16000,
                    "pcm": "mono_float32",
                    "fbank_by": "sherpa-onnx",
                    "chunk_seconds": 25,
                },
                auxiliary_files=asr_aux,
            )
        )

        # 3. GatedAttentionFusion ONNX
        client = mlflow.MlflowClient(tracking_uri=config.mlflow_tracking_uri)
        exp = client.get_experiment_by_name(MlflowConfig.experiment_fusion)
        if exp is not None:
            runs = client.search_runs(
                [exp.experiment_id],
                filter_string="tags.mlflow.runName = 'fusion_gated_attention_4ch' and attributes.status = 'FINISHED'",
                max_results=1,
            )
            if runs:
                pt_path = mlflow.artifacts.download_artifacts(
                    run_id=runs[0].info.run_id,
                    artifact_path="model/gated_attention_fusion.pt",
                    tracking_uri=config.mlflow_tracking_uri,
                )
                gate = GatedAttentionFusion(query_dim=self.profile.embedding_dim, num_channels=4, hidden_dim=64)
                gate.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
                fusion_path = self.bundle_dir / "gated_attention_fusion.onnx"
                gate.export_onnx(str(fusion_path), query_dim=self.profile.embedding_dim, num_channels=4)
                specs.append(
                    ComponentSpec(
                        name="fusion",
                        path="gated_attention_fusion.onnx",
                        sha256=self._compute_sha256(fusion_path),
                        runtime="onnxruntime",
                        inputs=[
                            {
                                "name": "query_emb",
                                "dtype": "float32",
                                "shape": ["n_queries", self.profile.embedding_dim],
                            },
                            {"name": "channel_scores", "dtype": "float32", "shape": ["n_queries", "n_docs", 4]},
                            {"name": "channel_mask", "dtype": "bool", "shape": ["n_docs", 4]},
                        ],
                        outputs=[{"name": "fused_scores", "dtype": "float32", "shape": ["n_queries", "n_docs"]}],
                    )
                )
        return specs

    def _append_multimodal_smoke_tests(self) -> None:
        """Добавляет эталонные тесты для fusion, OCR и ASR в smoke_test_pack.json."""
        smoke_path = self.bundle_dir / "smoke_test_pack.json"
        if not smoke_path.exists():
            return
        pack = json.loads(smoke_path.read_text(encoding="utf-8"))

        fusion_path = self.bundle_dir / "gated_attention_fusion.onnx"
        if fusion_path.exists():
            rng = np.random.default_rng(42)
            q_emb = rng.standard_normal((1, self.profile.embedding_dim), dtype=np.float32)
            c_scores = np.array([[[0.9, 0.8, 0.7, 0.0], [0.2, 0.3, 0.0, 0.6]]], dtype=np.float32)
            c_mask = np.array([[True, True, True, False], [True, True, False, True]], dtype=bool)
            sess = onnxruntime.InferenceSession(str(fusion_path), providers=["CPUExecutionProvider"])
            fused = sess.run(None, {"query_emb": q_emb, "channel_scores": c_scores, "channel_mask": c_mask})[0]
            pack["fusion_tests"] = [
                {
                    "query_emb": [round(float(x), 6) for x in q_emb[0]],
                    "channel_scores": c_scores.tolist(),
                    "channel_mask": c_mask.tolist(),
                    "expected_fused_scores": [[round(float(x), 6) for x in fused[0]]],
                    "atol": 1e-4,
                }
            ]

        img = Image.new("RGB", (200, 60), color=(255, 255, 255))
        ImageDraw.Draw(img).text((20, 20), "MEME", fill=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        pack["ocr_tests"] = [
            {
                "scope": "end_to_end_text",
                "image_base64_png": base64.b64encode(buf.getvalue()).decode("ascii"),
                "expected_text": "MEME",
            }
        ]

        wav_silence_buf = io.BytesIO()
        pcm_silence = np.zeros(8000, dtype=np.int16)
        with wave.open(wav_silence_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm_silence.tobytes())

        asr_tests = [
            {
                "case": "silence",
                "sample_rate": 16000,
                "wav_base64": base64.b64encode(wav_silence_buf.getvalue()).decode("ascii"),
                "expected_text": "",
                "expected_segments": [],
            }
        ]

        speech_wav_path = Path(__file__).resolve().parent.parent.parent / "resources" / "speech_sample_16k.wav"
        if speech_wav_path.exists():
            asr_tests.append(
                {
                    "case": "speech_phrase",
                    "sample_rate": 16000,
                    "wav_base64": base64.b64encode(speech_wav_path.read_bytes()).decode("ascii"),
                    "expected_text": "и сказки конец",
                    "expected_segments": ["и сказки конец"],
                }
            )

        pack["asr_tests"] = asr_tests
        smoke_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")

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
