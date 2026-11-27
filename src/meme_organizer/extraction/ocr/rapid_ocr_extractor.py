"""RapidOCR (ONNX Runtime PaddleOCR) extractor."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from meme_organizer.extraction.ocr.base_ocr_extractor import BaseOcrExtractor, OcrResult

if TYPE_CHECKING:
    import numpy as np


class RapidOcrExtractor(BaseOcrExtractor):
    """ONNX-native PaddleOCR extractor via rapidocr_onnxruntime."""

    name = "rapidocr_onnx"

    def __init__(
        self,
        min_confidence: float = 0.5,
        rec_model_path: str | None = None,
        rec_keys_path: str | None = None,
    ) -> None:
        """Initialize RapidOCR engine with optional custom recognition model and dictionary."""
        from rapidocr_onnxruntime import RapidOCR

        self.min_confidence = min_confidence
        kwargs: dict[str, str] = {}
        if rec_model_path is not None:
            kwargs["rec_model_path"] = rec_model_path
        if rec_keys_path is not None:
            kwargs["rec_keys_path"] = rec_keys_path
        self._engine = RapidOCR(**kwargs)

    def extract_text(self, image: np.ndarray) -> OcrResult:
        """Run RapidOCR on a BGR image and filter lines below min_confidence."""
        start = time.perf_counter()
        result, _ = self._engine(image)
        latency_ms = (time.perf_counter() - start) * 1000.0

        lines: list[str] = []
        confidences: list[float] = []
        if result:
            for item in result:
                text = str(item[1]).strip()
                conf = float(item[2])
                if text and conf >= self.min_confidence:
                    lines.append(text)
                    confidences.append(conf)

        return OcrResult(
            text=" ".join(lines),
            lines=lines,
            confidences=confidences,
            latency_ms=latency_ms,
        )

    def model_size_mb(self) -> float:
        """Return approximate disk size of bundled PP-OCR ONNX models in MB."""
        return 15.5
