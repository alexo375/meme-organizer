"""Unit tests for Mobile Delivery Bundle export, manifest validation, and ONNX parity."""

from __future__ import annotations

import json
import tarfile
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime
import pytest
from transformers import AutoModel, AutoTokenizer

from meme_organizer.mobile.bundle_builder import MobileBundleBuilder
from meme_organizer.mobile.manifest import PresetManifest
from meme_organizer.mobile.profiles import ModelBundleProfile
from meme_organizer.mobile.smoke_test import SmokeTestPack
from meme_organizer.models.onnx_exporter import OnnxExporter

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
        alpha_bm25=0.5,
        beta_dense=0.5,
        epochs=1,
    )


def test_onnx_exporter_in_graph_pooling(tmp_path: Path) -> None:
    """Verify that exported ONNX model outputs normalized [batch, dim] embeddings directly."""
    model_id = "cointegrated/rubert-tiny2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)

    onnx_path = OnnxExporter.export_text_encoder(
        model=model,
        tokenizer=tokenizer,
        output_dir=tmp_path / "onnx",
        base_name="test_text_encoder",
        max_length=32,
        include_pooling=True,
    )

    assert onnx_path.exists()

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


def test_mobile_bundle_builder(tmp_path: Path, dummy_profile: ModelBundleProfile) -> None:
    """Verify end-to-end bundle creation: archive contents, manifest and smoke pack."""
    work_dir = tmp_path / "work"
    output_tar = tmp_path / "dist" / f"bundle-{dummy_profile.profile_id}.tar.gz"

    builder = MobileBundleBuilder(profile=dummy_profile, work_dir=work_dir)
    res_tar = builder.build(output_tar=output_tar)

    assert res_tar.exists()
    assert res_tar.stat().st_size > 0

    # Inspect tar archive contents
    with tarfile.open(res_tar, "r:gz") as tar:
        names = tar.getnames()
        root = f"meme-organizer-{dummy_profile.profile_id}"

        assert f"{root}/text_encoder_int8.onnx" in names
        assert f"{root}/preset_manifest.json" in names
        assert f"{root}/smoke_test_pack.json" in names
        assert f"{root}/README.md" in names
        assert any(n.startswith(f"{root}/tokenizer/") for n in names)

        # Validate manifest content
        manifest_file = tar.extractfile(f"{root}/preset_manifest.json")
        assert manifest_file is not None
        manifest_data = json.load(manifest_file)
        manifest = PresetManifest.model_validate(manifest_data)
        assert manifest.component == "text_encoder"
        assert manifest.embedding_dimension == 312

        # Validate smoke test pack content
        smoke_file = tar.extractfile(f"{root}/smoke_test_pack.json")
        assert smoke_file is not None
        smoke_data = json.load(smoke_file)
        smoke_pack = SmokeTestPack.model_validate(smoke_data)
        assert len(smoke_pack.samples) == 8
        assert len(smoke_pack.similarity_tests) == 4

        # Validate README mentions Kotlin and formula
        readme_file = tar.extractfile(f"{root}/README.md")
        assert readme_file is not None
        readme_text = readme_file.read().decode("utf-8")
        assert "class TextEmbeddingEngine" in readme_text
        assert "0.5f * bm25NormalizedScore + 0.5f * cosineSimilarity" in readme_text
