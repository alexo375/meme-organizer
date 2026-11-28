"""ASR experiment adapter for BaseExperimentPipeline."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import jiwer
import numpy as np

from meme_organizer.domain.base_experiment_adapter import BaseExperimentAdapter

if TYPE_CHECKING:
    from pathlib import Path

    from meme_organizer.evaluation.metrics.experiment_tracker import ExperimentTracker
    from meme_organizer.extraction.asr.base_asr_extractor import BaseAsrExtractor
    from meme_organizer.extraction.audio.audio_extractor import AudioExtractor


class AsrExperimentAdapter(BaseExperimentAdapter):
    """Runs ASR transcription over video records and collects quality + component timings."""

    def __init__(
        self,
        run_name: str,
        extractor: BaseAsrExtractor,
        audio_extractor: AudioExtractor,
        records: list[dict[str, Any]],
        silver_dir: Path,
        model_artifacts: dict[str, Path] | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Initialize ASR adapter with ASR extractor, audio extractor, and dataset records."""
        super().__init__(run_name=run_name, tags=tags or {"domain": "asr"})
        self.extractor = extractor
        self.audio_extractor = audio_extractor
        self.records = records
        self.silver_dir = silver_dir
        self._model_artifacts = model_artifacts or {}

        self.transcriptions: dict[str, str] = {}
        self._refs: list[str] = []
        self._hyps: list[str] = []
        self._audio_ms: list[float] = []
        self._inference_ms: list[float] = []
        self._total_ms: list[float] = []
        self._errors: list[dict[str, Any]] = []
        self._exec_sec: float = 0.0

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().strip().split())

    def get_params(self) -> dict[str, Any]:
        """Return ASR engine and dataset configuration."""
        return {
            "engine": self.extractor.name,
            "sample_rate": self.audio_extractor.sample_rate,
            "n_records": len(self.records),
        }

    def prepare(self) -> None:
        """Reset per-run accumulators before execution."""
        self.transcriptions.clear()
        self._refs.clear()
        self._hyps.clear()
        self._audio_ms.clear()
        self._inference_ms.clear()
        self._total_ms.clear()
        self._errors.clear()

    def execute(self, tracker: ExperimentTracker | None = None) -> None:
        """Extract audio and run ASR across all records."""
        t_start = time.perf_counter()
        for rec in self.records:
            media_path = self.silver_dir / rec["silver_filename"]
            ref = self._normalize(rec.get("transcript_or_text", ""))

            t0 = time.perf_counter()
            audio = self.audio_extractor.extract(media_path)
            a_ms = (time.perf_counter() - t0) * 1000.0

            res = self.extractor.transcribe(audio, self.audio_extractor.sample_rate)
            tot_ms = a_ms + res.latency_ms
            hyp = self._normalize(res.text)

            self.transcriptions[rec["file_hash"]] = res.text
            self._refs.append(ref)
            self._hyps.append(hyp or " ")
            self._audio_ms.append(a_ms)
            self._inference_ms.append(res.latency_ms)
            self._total_ms.append(tot_ms)

            cer_item = jiwer.cer(ref, hyp or " ") if ref else 0.0
            self._errors.append(
                {
                    "filename": rec["silver_filename"],
                    "media_type": rec["media_type"],
                    "cer": round(float(cer_item), 4),
                    "ref": ref,
                    "hyp": hyp,
                    "total_ms": round(tot_ms, 1),
                }
            )
        self._exec_sec = time.perf_counter() - t_start

    def compute_quality_metrics(self) -> dict[str, float]:
        """Compute CER, WER, and empty_rate."""
        if not self._refs:
            return {}
        empty_count = sum(1 for h in self._hyps if h == " ")
        return {
            "cer": round(float(jiwer.cer(self._refs, self._hyps)), 4),
            "wer": round(float(jiwer.wer(self._refs, self._hyps)), 4),
            "empty_rate": round(empty_count / len(self._hyps), 4),
        }

    def compute_performance_metrics(self) -> dict[str, float]:
        """Compute audio extraction, inference, total P50/P95 latencies, throughput, and size."""
        if not self._total_ms:
            return {}
        return {
            "audio_extract_p50_ms": round(float(np.percentile(self._audio_ms, 50)), 2),
            "inference_p50_ms": round(float(np.percentile(self._inference_ms, 50)), 2),
            "inference_p95_ms": round(float(np.percentile(self._inference_ms, 95)), 2),
            "total_p50_ms": round(float(np.percentile(self._total_ms, 50)), 2),
            "total_p95_ms": round(float(np.percentile(self._total_ms, 95)), 2),
            "throughput_items_per_sec": round(len(self._total_ms) / max(self._exec_sec, 1e-6), 2),
            "model_size_mb": self.extractor.model_size_mb(),
        }

    def get_error_table(self) -> list[dict[str, Any]]:
        """Return records sorted by descending CER."""
        return sorted(self._errors, key=lambda x: float(x["cer"]), reverse=True)

    def get_model_artifacts(self) -> dict[str, Path]:
        """Return configured ONNX model artifacts."""
        return self._model_artifacts
