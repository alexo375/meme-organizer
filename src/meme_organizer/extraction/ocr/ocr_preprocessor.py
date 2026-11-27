"""Image preprocessing strategies for meme typography."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cv2

if TYPE_CHECKING:
    import numpy as np

PreprocessMode = Literal["raw", "contrast_clahe", "upscale_2x", "meme_outline"]


class OcrPreprocessor:
    """Applies configurable preprocessing transforms before OCR inference."""

    def __init__(self, mode: PreprocessMode = "raw") -> None:
        """Initialize preprocessor with a named transform mode."""
        self.mode = mode

    def process(self, image: np.ndarray) -> np.ndarray:
        """Transform a BGR image array according to the configured mode."""
        if self.mode == "raw":
            return image
        if self.mode == "contrast_clahe":
            return self._apply_clahe(image)
        if self.mode == "upscale_2x":
            return self._apply_upscale(image)
        if self.mode == "meme_outline":
            return self._apply_meme_outline(image)
        raise ValueError(f"Unsupported preprocessing mode: {self.mode}")

    @staticmethod
    def _apply_clahe(image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_chan)
        merged = cv2.merge((cl, a_chan, b_chan))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _apply_upscale(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        if width >= 480:
            return image
        return cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _apply_meme_outline(image: np.ndarray) -> np.ndarray:
        # Isolate bright Impact-style meme text and invert to dark-on-light for OCR.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        inverted = cv2.bitwise_not(mask)
        return cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
