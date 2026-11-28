"""Audio extraction from video/GIF files using ffmpeg."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


class AudioExtractor:
    """Extracts mono 16 kHz float32 PCM audio from media files via ffmpeg."""

    def __init__(self, sample_rate: int = 16000) -> None:
        """Initialize extractor with target sample rate."""
        self.sample_rate = sample_rate

    def extract(self, media_path: Path) -> np.ndarray:
        """Return 1D float32 audio array, or empty array if no audio stream exists."""
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(media_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)  # noqa: S603
        if proc.returncode != 0 or not proc.stdout:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()
