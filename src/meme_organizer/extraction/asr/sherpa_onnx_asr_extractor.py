"""Sherpa-ONNX ASR extractor with optional Silero VAD."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from meme_organizer.extraction.asr.base_asr_extractor import AsrResult, BaseAsrExtractor

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


class SherpaOnnxAsrExtractor(BaseAsrExtractor):
    """On-device ONNX ASR extractor using sherpa-onnx (Whisper / GigaAM) and optional Silero VAD."""

    name = "sherpa_onnx"

    def __init__(
        self,
        model_type: str,
        tokens_path: Path,
        encoder_path: Path | None = None,
        decoder_path: Path | None = None,
        nemo_ctc_path: Path | None = None,
        vad_model_path: Path | None = None,
        language: str = "ru",
        num_threads: int = 4,
    ) -> None:
        """Initialize sherpa-onnx offline recognizer and optional Silero VAD."""
        import sherpa_onnx

        self.model_type = model_type
        self.tokens_path = tokens_path
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.nemo_ctc_path = nemo_ctc_path
        self.vad_model_path = vad_model_path
        self.name = f"sherpa_{model_type}"

        if model_type == "whisper":
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=str(encoder_path),
                decoder=str(decoder_path),
                tokens=str(tokens_path),
                language=language,
                task="transcribe",
                num_threads=num_threads,
            )
        elif model_type == "nemo_ctc":
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=str(nemo_ctc_path),
                tokens=str(tokens_path),
                num_threads=num_threads,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self._vad = None
        if vad_model_path is not None:
            vad_cfg = sherpa_onnx.VadModelConfig()
            vad_cfg.silero_vad.model = str(vad_model_path)
            vad_cfg.sample_rate = 16000
            self._vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=60)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> AsrResult:
        """Transcribe audio, optionally filtering silence with Silero VAD."""
        if audio.size == 0:
            return AsrResult(text="", segments=[], latency_ms=0.0)

        t0 = time.perf_counter()
        if self._vad is not None:
            self._vad.reset()
            self._vad.accept_waveform(audio)
            self._vad.flush()
            chunks: list[np.ndarray] = []
            while not self._vad.empty():
                chunks.append(self._vad.front.samples)
                self._vad.pop()
            if not chunks:
                return AsrResult(text="", segments=[], latency_ms=(time.perf_counter() - t0) * 1000.0)
        else:
            chunks = [audio]

        segments: list[str] = []
        for chunk in chunks:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sample_rate, chunk)
            self._recognizer.decode_stream(stream)
            txt = stream.result.text.strip()
            if txt:
                segments.append(txt)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return AsrResult(text=" ".join(segments), segments=segments, latency_ms=latency_ms)

    def model_size_mb(self) -> float:
        """Return total size of configured ONNX model files in MB."""
        total = 0
        for p in (self.encoder_path, self.decoder_path, self.nemo_ctc_path, self.vad_model_path):
            if p is not None and p.exists():
                total += p.stat().st_size
        return round(total / (1024 * 1024), 2)
