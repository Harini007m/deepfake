"""
video_input.py
──────────────
Module 1 – Video Input
Handles opening a video file (or webcam) and yielding frames one by one.
"""

from __future__ import annotations
import cv2
from pathlib import Path


class VideoReader:
    """Thin wrapper around cv2.VideoCapture that provides convenient iteration."""

    def __init__(self, source: str | int = 0):
        """
        Parameters
        ----------
        source : str | int
            Path to a video file, or an integer webcam index (default 0).
        """
        if isinstance(source, str):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Video file not found: {path}")
        self.source = source
        self.cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

    # ── Properties ──────────────────────────────────────────────────────────
    @property
    def fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS) or 25.0

    @property
    def total_frames(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def duration_seconds(self) -> float:
        fps = self.fps
        return self.total_frames / fps if fps > 0 else 0.0

    # ── Iteration ────────────────────────────────────────────────────────────
    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, object]:
        """Return (frame_index, BGR frame) or raise StopIteration."""
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        return idx, frame

    def __len__(self) -> int:
        return self.total_frames

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def info(self) -> dict:
        return {
            "source":       str(self.source),
            "fps":          round(self.fps, 2),
            "total_frames": self.total_frames,
            "width":        self.width,
            "height":       self.height,
            "duration_s":   round(self.duration_seconds, 2),
        }
