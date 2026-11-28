"""Video and GIF keyframe sampling with scene deduplication."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

SamplingStrategy = Literal["uniform_3", "interval_dedup"]


class KeyframeSampler:
    """Extracts representative BGR frames from video or GIF files."""

    def __init__(
        self,
        strategy: SamplingStrategy = "uniform_3",
        interval_sec: float = 1.5,
        dedup_threshold: float = 0.92,
        max_frames: int = 12,
    ) -> None:
        """Initialize sampler with strategy, interval, similarity threshold, and frame cap."""
        self.strategy = strategy
        self.interval_sec = interval_sec
        self.dedup_threshold = dedup_threshold
        self.max_frames = max_frames

    def sample(self, media_path: Path) -> list[np.ndarray]:
        """Extract BGR frames from a video or animated GIF file."""
        cap = cv2.VideoCapture(str(media_path))
        if not cap.isOpened():
            return []
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
            if total_frames <= 0:
                return []
            if self.strategy == "uniform_3":
                return self._sample_uniform(cap, total_frames, count=3)
            if self.strategy == "interval_dedup":
                return self._sample_interval_dedup(cap, total_frames, fps)
            raise ValueError(f"Unsupported sampling strategy: {self.strategy}")
        finally:
            cap.release()

    def _sample_uniform(self, cap: cv2.VideoCapture, total_frames: int, count: int) -> list[np.ndarray]:
        indices = sorted({int(i) for i in np.linspace(0, max(total_frames - 1, 0), count)})
        frames: list[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
        return frames

    def _sample_interval_dedup(self, cap: cv2.VideoCapture, total_frames: int, fps: float) -> list[np.ndarray]:
        step = max(1, int(fps * self.interval_sec))
        frames: list[np.ndarray] = []
        prev_hist: np.ndarray | None = None

        for idx in range(0, total_frames, step):
            if len(frames) >= self.max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            hist = self._compute_hist(frame)
            if prev_hist is None or cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < self.dedup_threshold:
                frames.append(frame)
                prev_hist = hist
        return frames

    @staticmethod
    def _compute_hist(frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten()
