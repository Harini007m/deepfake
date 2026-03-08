"""
ear_calculator.py
─────────────────
Module 4 – Eye Aspect Ratio (EAR) Calculation

The EAR formula (Soukupová & Čech, 2016):
    EAR = (||p2–p6|| + ||p3–p5||) / (2 × ||p1–p4||)

For the 6 landmarks ordered as:
    p1 (outer corner) – p2 – p3 – p4 (inner corner) – p5 – p6
    (following the MediaPipe numbering order for LEFT_EYE_LANDMARKS /
     RIGHT_EYE_LANDMARKS defined in config.py)
"""

from __future__ import annotations
import numpy as np
from collections import deque
from config import EAR_SMOOTHING_WINDOW


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_ear(eye_pts: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio for one eye.

    Parameters
    ----------
    eye_pts : np.ndarray, shape (6, 2)
        Landmarks in the order: [outer, top1, top2, inner, bot2, bot1]
        which corresponds to the indices in LEFT/RIGHT_EYE_LANDMARKS.

    Returns
    -------
    float – EAR value (typically 0.15 – 0.40)
    """
    # Vertical distances
    A = _euclidean(eye_pts[1], eye_pts[5])   # top1 ↔ bot1
    B = _euclidean(eye_pts[2], eye_pts[4])   # top2 ↔ bot2
    # Horizontal distance
    C = _euclidean(eye_pts[0], eye_pts[3])   # outer ↔ inner
    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


class EARSmoother:
    """Rolling-average smoother for EAR values."""

    def __init__(self, window: int = EAR_SMOOTHING_WINDOW):
        self._window = window
        self._left_buf:  deque[float] = deque(maxlen=window)
        self._right_buf: deque[float] = deque(maxlen=window)

    def update(self, left_ear: float, right_ear: float) -> tuple[float, float]:
        self._left_buf.append(left_ear)
        self._right_buf.append(right_ear)
        return float(np.mean(self._left_buf)), float(np.mean(self._right_buf))

    def reset(self):
        self._left_buf.clear()
        self._right_buf.clear()
