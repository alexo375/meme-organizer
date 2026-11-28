"""Unit tests for Mobile Delivery Bundle export, manifest validation, and ONNX parity."""

from __future__ import annotations

import json
import tarfile
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime
import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from meme_organizer.domain.preset_manifest import PresetManifest
from meme_organizer.export.bundle.bundle_profiles import ModelBundleProfile
from meme_organizer.export.bundle.mobile_bundle_builder import MobileBundleBuilder
from meme_organizer.export.bundle.release_publisher import GitHubReleasePublisher
from meme_organizer.export.generators.smoke_test_generator import SmokeTestPack
from meme_organizer.export.onnx.onnx_exporter import OnnxExporter
from meme_organizer.export.onnx.onnx_quantizer import OnnxQuantizer
from meme_organizer.retrieval.fusion.fusion_optimizer import FusionOptimizer
from meme_organizer.retrieval.text.dense_text_retriever import DenseTextRetriever
from meme_organizer.retrieval.text.fine_tuned_dense_retriever import FineTunedDenseRetriever

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def dummy_profile() -> ModelBundleProfile:
    """Fixture for lightweight testing profile using rubert-tiny2 without training."""
    return ModelBundleProfile(
        profile_id="test-rubert-tiny2",
        model_name="cointegrated/rubert-tiny2",
        base_hf_id="cointegrated/rubert-tiny2",
        embedding_dim=312,
        max_length=32,
        query_prefix="",
        passage_prefix="",
        dense_weight=0.55,
        bm25_weight=0.45,
        epochs=1,
    )


def test_onnx_exporter_in_graph_pooling(tmp_path: Path) -> None:
    """Verify FP32 ONNX export and separate INT8 quantization output normalized [batch, dim] embeddings."""
    model_id = "cointegrated/rubert-tiny2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)

    fp32_path = OnnxExporter.export_to_fp32_onnx(
        model=model,
        tokenizer=tokenizer,
        output_dir=tmp_path / "onnx",
        base_name="test_text_encoder",
        max_length=32,
        include_pooling=True,
    )
    assert fp32_path.exists()

    onnx_path = OnnxQuantizer.quantize_dynamic_int8(
        fp32_onnx_path=fp32_path,
        int8_onnx_path=tmp_path / "onnx" / "test_text_encoder_int8.onnx",
        cleanup_fp32=True,
    )
    assert onnx_path.exists()
    assert not fp32_path.exists()

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = tokenizer("тестовый запрос", padding="max_length", max_length=32, return_tensors="np")
    feed = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }

    outputs = session.run(["sentence_embedding"], feed)
    embeddings = outputs[0]

    # Verify shape: [batch_size=1, dim=312]
    assert embeddings.shape == (1, 312)

    # Verify L2 normalization: ||v|| == 1.0
    norm = np.linalg.norm(embeddings[0])
    assert pytest.approx(norm, abs=1e-4) == 1.0


def test_lora_updates_across_multiple_epochs(tmp_path: Path) -> None:
    """Verify optimizer keeps updating LoRA weights after epoch 1 validation."""
    retriever = FineTunedDenseRetriever(
        model_name="rubert-tiny2-test-ft",
        base_model_id="cointegrated/rubert-tiny2",
        epochs=2,
        lr=1e-2,
        batch_size=2,
        max_length=16,
        cache_dir=tmp_path / "hf",
    )
    corpus = [
        {"file_hash": "h1", "user_caption": "кот в шапке", "tags": ["кот"]},
        {"file_hash": "h2", "user_caption": "упал прод", "tags": ["прод"]},
    ]
    pairs = [
        {"query": "кот", "file_hash": "h1"},
        {"query": "прод", "file_hash": "h2"},
    ]

    losses: list[float] = []

    class SpyTracker:
        def log_epoch_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
            _ = epoch
            losses.append(metrics["train_loss"])

    retriever.fit(train_pairs=pairs, corpus_docs=corpus, val_pairs=pairs, tracker=SpyTracker())  # type: ignore[arg-type]

    assert len(losses) == 2
    assert losses[1] < losses[0]
    assert retriever._model is not None
    assert not hasattr(retriever._model, "peft_config")
    # Verify clean FP32 HF directory is saved (and no INT8 ONNX in training artifact)
    art_dirs = retriever.get_artifact_dirs()
    assert "hf_model" in art_dirs
    assert (art_dirs["hf_model"] / "config.json").exists()
    assert retriever.get_onnx_paths() == []
    with torch.no_grad():
        emb = retriever.encode_queries(["кот"])
        assert emb.shape == (1, 312)


def test_fusion_optimizer_balanced_range() -> None:
    """Verify FusionOptimizer stays within balanced [0.25, 0.75] range using K-Fold CV."""
    rng = np.random.default_rng(42)
    s_dense = rng.random((20, 10))
    s_bm25 = rng.random((20, 10))
    targets = [int(i % 10) for i in range(20)]

    opt = FusionOptimizer(n_trials=15, alpha_range=(0.25, 0.75), n_folds=5)
    alpha = opt.fit(s_dense, s_bm25, targets)
    assert 0.25 <= alpha <= 0.75


def test_dense_mlflow_fast_fail_when_unreachable() -> None:
    """Verify MLflow port check fails fast (<1s) without urllib3 retry hangs."""
    import time

    t0 = time.perf_counter()
    reachable = DenseTextRetriever._is_mlflow_reachable("http://127.0.0.1:59999")
    elapsed = time.perf_counter() - t0
    assert reachable is False
    assert elapsed < 1.0


def test_release_publisher_utils(tmp_path: Path) -> None:
    """Verify GitHub remote parsing and SHA256SUMS generation."""
    assert GitHubReleasePublisher.parse_github_repo("git@github.com:alexo375/meme-organizer.git") == (
        "alexo375",
        "meme-organizer",
    )
    assert GitHubReleasePublisher.parse_github_repo("https://github.com/alexo375/meme-organizer") == (
        "alexo375",
        "meme-organizer",
    )

    dummy_tar = tmp_path / "bundle.tar.gz"
    dummy_tar.write_bytes(b"hello-world")
    sums_path = GitHubReleasePublisher.write_sha256sums(dummy_tar)
    assert sums_path.exists()
    content = sums_path.read_text(encoding="utf-8")
    assert "bundle.tar.gz" in content
    assert len(content.split()[0]) == 64


def test_mobile_bundle_builder(tmp_path: Path, dummy_profile: ModelBundleProfile) -> None:
    """Verify end-to-end bundle creation: archive contents, manifest, provenance, and smoke pack."""
    work_dir = tmp_path / "work"
    output_tar = tmp_path / "dist" / f"bundle-{dummy_profile.profile_id}.tar.gz"

    builder = MobileBundleBuilder(profile=dummy_profile, work_dir=work_dir, run_full_eval=False)
    res_tar = builder.build(output_tar=output_tar, release_tag="v0.2.0-mobile")

    assert res_tar.exists()
    assert res_tar.stat().st_size > 0

    with tarfile.open(res_tar, "r:gz") as tar:
        names = tar.getnames()
        root = f"meme-organizer-{dummy_profile.profile_id}"

        assert f"{root}/text_encoder_int8.onnx" in names
        assert f"{root}/preset_manifest.json" in names
        assert f"{root}/smoke_test_pack.json" in names
        assert f"{root}/README.md" in names
        assert any(n.startswith(f"{root}/tokenizer/") for n in names)

        # Validate manifest content and provenance
        manifest_file = tar.extractfile(f"{root}/preset_manifest.json")
        assert manifest_file is not None
        manifest_data = json.load(manifest_file)
        manifest = PresetManifest.model_validate(manifest_data)
        assert manifest.release_tag == "v0.2.0-mobile"
        assert manifest.component == "text_encoder"
        assert manifest.embedding_dimension == 312
        assert manifest.hybrid_scoring.dense_weight == 0.55
        assert manifest.hybrid_scoring.bm25_weight == 0.45
        assert manifest.hybrid_scoring.normalization == "per_query_min_max"
        assert manifest.provenance is not None
        assert len(manifest.provenance.onnx_sha256) == 64
        assert manifest.provenance.pytorch_onnx_mean_cosine_similarity >= 0.95

        # Validate smoke test pack content including hybrid ranking tests
        smoke_file = tar.extractfile(f"{root}/smoke_test_pack.json")
        assert smoke_file is not None
        smoke_data = json.load(smoke_file)
        smoke_pack = SmokeTestPack.model_validate(smoke_data)
        assert len(smoke_pack.samples) == 8
        assert len(smoke_pack.similarity_tests) == 4
        assert len(smoke_pack.hybrid_ranking_tests) == 4

        # Validate README contains MinMax, SimpleBm25Engine, and exact weights
        readme_file = tar.extractfile(f"{root}/README.md")
        assert readme_file is not None
        readme_text = readme_file.read().decode("utf-8")
        assert "class TextEmbeddingEngine" in readme_text
        assert "class SimpleBm25Engine" in readme_text
        assert "minMaxNormalize" in readme_text
        assert "WEIGHT_DENSE = 0.55f" in readme_text
        assert "WEIGHT_BM25 = 0.45f" in readme_text
