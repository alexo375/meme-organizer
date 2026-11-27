"""Unit tests for OCR extraction, preprocessing, and video keyframe sampling."""

from __future__ import annotations

import cv2
import numpy as np

from meme_organizer.extraction.ocr.ocr_preprocessor import OcrPreprocessor
from meme_organizer.extraction.ocr.rapid_ocr_extractor import RapidOcrExtractor


def _make_meme_image(text: str = "TEST MEME 123") -> np.ndarray:
    img = np.zeros((200, 500, 3), dtype=np.uint8)
    cv2.putText(
        img,
        text,
        (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return img


def test_preprocessor_modes_preserve_valid_image() -> None:
    img = _make_meme_image()
    for mode in ("raw", "contrast_clahe", "upscale_2x", "meme_outline"):
        out = OcrPreprocessor(mode=mode).process(img)
        assert out.ndim == 3
        assert out.shape[2] == 3


def test_rapid_ocr_extracts_synthetic_text() -> None:
    img = _make_meme_image("HELLO WORLD")
    extractor = RapidOcrExtractor(min_confidence=0.3)
    result = extractor.extract_text(img)
    assert "HELLO" in result.text.upper()
    assert result.latency_ms > 0.0
    assert extractor.model_size_mb() > 0.0
