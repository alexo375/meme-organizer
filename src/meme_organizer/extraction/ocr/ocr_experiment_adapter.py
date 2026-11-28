"""OCR / Video / GIF experiment adapter for BaseExperimentPipeline."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import cv2
import jiwer
import numpy as np

from meme_organizer.domain.base_experiment_adapter import BaseExperimentAdapter

if TYPE_CHECKING:
    from pathlib import Path

    from meme_organizer.evaluation.metrics.experiment_tracker import ExperimentTracker
    from meme_organizer.extraction.ocr.base_ocr_extractor import BaseOcrExtractor
    from meme_organizer.extraction.ocr.ocr_preprocessor import OcrPreprocessor
    from meme_organizer.extraction.video.keyframe_sampler import KeyframeSampler


class OcrExperimentAdapter(BaseExperimentAdapter):
    """Runs OCR extraction over image/video records and collects quality + component timings."""

    def __init__(
        self,
        run_name: str,
        extractor: BaseOcrExtractor,
        preprocessor: OcrPreprocessor,
        sampler: KeyframeSampler,
        records: list[dict[str, Any]],
        silver_dir: Path,
        rec_model_path: Path | None = None,
        rec_keys_path: Path | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Initialize OCR adapter with extractor, preprocessor, sampler, and dataset records."""
        super().__init__(run_name=run_name, tags=tags or {"domain": "ocr"})
        self.extractor = extractor
        self.preprocessor = preprocessor
        self.sampler = sampler
        self.records = records
        self.silver_dir = silver_dir
        self.rec_model_path = rec_model_path
        self.rec_keys_path = rec_keys_path

        self.extracted_texts: dict[str, str] = {}
        self._refs: list[str] = []
        self._hyps: list[str] = []
        self._confs: list[float] = []
        self._sample_ms: list[float] = []
        self._preprocess_ms: list[float] = []
        self._inference_ms: list[float] = []
        self._total_ms: list[float] = []
        self._errors: list[dict[str, Any]] = []
        self._exec_sec: float = 0.0

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().strip().split())

    def get_params(self) -> dict[str, Any]:
        """Return OCR engine, preprocessing, and sampling configuration."""
        return {
            "engine": self.extractor.name,
            "preprocess_mode": self.preprocessor.mode,
            "sampler_strategy": self.sampler.strategy,
            "n_records": len(self.records),
        }

    def prepare(self) -> None:
        """Reset per-run accumulators before execution."""
        self.extracted_texts.clear()
        self._refs.clear()
        self._hyps.clear()
        self._confs.clear()
        self._sample_ms.clear()
        self._preprocess_ms.clear()
        self._inference_ms.clear()
        self._total_ms.clear()
        self._errors.clear()

    def execute(self, tracker: ExperimentTracker | None = None) -> None:
        """Run OCR extraction across all records and measure component timings."""
        t_start = time.perf_counter()
        for rec in self.records:
            media_path = self.silver_dir / rec["silver_filename"]
            ref = self._normalize(rec.get("transcript_or_text", ""))
            hyp_raw, s_ms, p_ms, i_ms, confs = self._process_record(media_path, rec["media_type"])
            tot_ms = s_ms + p_ms + i_ms
            hyp = self._normalize(hyp_raw)

            self.extracted_texts[rec["file_hash"]] = hyp_raw
            self._refs.append(ref)
            self._hyps.append(hyp or " ")
            self._confs.extend(confs)
            self._sample_ms.append(s_ms)
            self._preprocess_ms.append(p_ms)
            self._inference_ms.append(i_ms)
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

    def _process_record(self, media_path: Path, media_type: str) -> tuple[str, float, float, float, list[float]]:
        if media_type == "image":
            t0 = time.perf_counter()
            img = cv2.imread(str(media_path))
            s_ms = (time.perf_counter() - t0) * 1000.0
            if img is None:
                return "", s_ms, 0.0, 0.0, []
            t1 = time.perf_counter()
            proc = self.preprocessor.process(img)
            p_ms = (time.perf_counter() - t1) * 1000.0
            res = self.extractor.extract_text(proc)
            return res.text, s_ms, p_ms, res.latency_ms, res.confidences

        t0 = time.perf_counter()
        frames = self.sampler.sample(media_path)
        s_ms = (time.perf_counter() - t0) * 1000.0
        seen: set[str] = set()
        lines: list[str] = []
        confs: list[float] = []
        p_ms = 0.0
        i_ms = 0.0
        for frame in frames:
            t1 = time.perf_counter()
            proc = self.preprocessor.process(frame)
            p_ms += (time.perf_counter() - t1) * 1000.0
            res = self.extractor.extract_text(proc)
            i_ms += res.latency_ms
            confs.extend(res.confidences)
            for line in res.lines:
                norm = self._normalize(line)
                if norm and norm not in seen:
                    seen.add(norm)
                    lines.append(line)
        return " ".join(lines), s_ms, p_ms, i_ms, confs

    def compute_quality_metrics(self) -> dict[str, float]:
        """Compute CER, WER, empty_rate, and avg_confidence."""
        if not self._refs:
            return {}
        empty_count = sum(1 for h in self._hyps if h == " ")
        return {
            "cer": round(float(jiwer.cer(self._refs, self._hyps)), 4),
            "wer": round(float(jiwer.wer(self._refs, self._hyps)), 4),
            "empty_rate": round(empty_count / len(self._hyps), 4),
            "avg_confidence": round(float(np.mean(self._confs)), 4) if self._confs else 0.0,
        }

    def compute_performance_metrics(self) -> dict[str, float]:
        """Compute component-level P50/P95 latencies, throughput, and model size."""
        if not self._total_ms:
            return {}
        return {
            "sample_p50_ms": round(float(np.percentile(self._sample_ms, 50)), 2),
            "sample_p95_ms": round(float(np.percentile(self._sample_ms, 95)), 2),
            "preprocess_p50_ms": round(float(np.percentile(self._preprocess_ms, 50)), 2),
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
        """Return raw ONNX model and dictionary paths if provided."""
        artifacts: dict[str, Path] = {}
        if self.rec_model_path is not None:
            artifacts["onnx_raw"] = self.rec_model_path
        if self.rec_keys_path is not None:
            artifacts["dict"] = self.rec_keys_path
        return artifacts
