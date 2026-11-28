"""Unit tests for AudioExtractor and AsrExperimentAdapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from meme_organizer.config import MlflowConfig
from meme_organizer.evaluation.metrics.experiment_tracker import ExperimentTracker
from meme_organizer.extraction.asr.asr_experiment_adapter import AsrExperimentAdapter
from meme_organizer.extraction.asr.base_asr_extractor import AsrResult, BaseAsrExtractor
from meme_organizer.extraction.audio.audio_extractor import AudioExtractor

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


class _DummyAsrExtractor(BaseAsrExtractor):
    name = "dummy_asr"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> AsrResult:
        return AsrResult(text="test phrase", segments=["test phrase"], latency_ms=1.5)

    def model_size_mb(self) -> float:
        return 12.0


def test_audio_extractor_handles_missing_file(tmp_path: Path) -> None:
    audio = AudioExtractor().extract(tmp_path / "missing.mp4")
    assert audio.size == 0


def test_asr_experiment_adapter_pipeline(tmp_path: Path) -> None:
    records = [
        {
            "file_hash": "vid1",
            "silver_filename": "missing.mp4",
            "media_type": "video",
            "transcript_or_text": "test phrase",
        }
    ]
    with ExperimentTracker(experiment_name=MlflowConfig.experiment_asr) as exp:
        adapter = AsrExperimentAdapter(
            run_name="test_asr_adapter",
            extractor=_DummyAsrExtractor(),
            audio_extractor=AudioExtractor(),
            records=records,
            silver_dir=tmp_path,
        )
        res = exp.create_pipeline(adapter).run()

    assert res["quality"]["cer"] == 0.0
    assert res["performance"]["model_size_mb"] == 12.0
