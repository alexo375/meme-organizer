"""Integration tests for ExperimentTracker + BaseExperimentPipeline + OcrExperimentAdapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from meme_organizer.config import MlflowConfig
from meme_organizer.evaluation.metrics.experiment_tracker import ExperimentTracker
from meme_organizer.extraction.ocr.ocr_experiment_adapter import OcrExperimentAdapter
from meme_organizer.extraction.ocr.ocr_preprocessor import OcrPreprocessor
from meme_organizer.extraction.ocr.rapid_ocr_extractor import RapidOcrExtractor
from meme_organizer.extraction.video.keyframe_sampler import KeyframeSampler

if TYPE_CHECKING:
    from pathlib import Path


def test_ocr_experiment_adapter_pipeline_logs_all_groups(tmp_path: Path) -> None:
    img_path = tmp_path / "meme.jpg"
    img = np.zeros((160, 400, 3), dtype=np.uint8)
    cv2.putText(img, "TEST MEME", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.imwrite(str(img_path), img)

    records = [
        {
            "file_hash": "abc123",
            "silver_filename": "meme.jpg",
            "media_type": "image",
            "transcript_or_text": "test meme",
        }
    ]
    dummy_onnx = tmp_path / "model.onnx"
    dummy_onnx.write_bytes(b"onnx")

    with ExperimentTracker(experiment_name=MlflowConfig.experiment_ocr) as exp:
        adapter = OcrExperimentAdapter(
            run_name="test_ocr_adapter",
            extractor=RapidOcrExtractor(min_confidence=0.3),
            preprocessor=OcrPreprocessor(mode="raw"),
            sampler=KeyframeSampler(strategy="uniform_3"),
            records=records,
            silver_dir=tmp_path,
            rec_model_path=dummy_onnx,
        )
        result = exp.create_pipeline(adapter).run()

    assert "cer" in result["quality"]
    assert "wer" in result["quality"]
    assert "empty_rate" in result["quality"]
    assert "avg_confidence" in result["quality"]
    assert "sample_p50_ms" in result["performance"]
    assert "preprocess_p50_ms" in result["performance"]
    assert "inference_p50_ms" in result["performance"]
    assert "throughput_items_per_sec" in result["performance"]
    assert "model_size_mb" in result["performance"]
