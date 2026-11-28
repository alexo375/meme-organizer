"""Tesseract OCR baseline extractor."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cv2

from meme_organizer.extraction.ocr.base_ocr_extractor import BaseOcrExtractor, OcrResult

if TYPE_CHECKING:
    import numpy as np


class TesseractOcrExtractor(BaseOcrExtractor):
    """CPU baseline OCR extractor using pytesseract (rus+eng)."""

    name = "tesseract"

    def __init__(self, lang: str = "rus+eng", psm: int = 11, min_confidence: float = 40.0) -> None:
        """Initialize Tesseract extractor with language, page segmentation mode, and confidence."""
        self.lang = lang
        self.psm = psm
        self.min_confidence = min_confidence

    def extract_text(self, image: np.ndarray) -> OcrResult:
        """Run Tesseract on a BGR image and return filtered text lines."""
        import pytesseract

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        config = f"--psm {self.psm}"
        start = time.perf_counter()
        data = pytesseract.image_to_data(rgb, lang=self.lang, config=config, output_type=pytesseract.Output.DICT)
        latency_ms = (time.perf_counter() - start) * 1000.0

        lines: list[str] = []
        confidences: list[float] = []
        for word, conf in zip(data["text"], data["conf"], strict=False):
            word_str = str(word).strip()
            conf_val = float(conf)
            if word_str and conf_val >= self.min_confidence:
                lines.append(word_str)
                confidences.append(conf_val / 100.0)

        return OcrResult(
            text=" ".join(lines),
            lines=lines,
            confidences=confidences,
            latency_ms=latency_ms,
        )

    def model_size_mb(self) -> float:
        """Return approximate size of rus+eng traineddata in MB."""
        return 18.0
