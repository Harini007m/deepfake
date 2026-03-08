"""
eye_landmarks.py
────────────────
Module 3 – Eye Landmark Extraction
Extracts the 6 key points for each eye from a full 468-point landmark array.
"""

from __future__ import annotations
import numpy as np
from config import LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS


def extract_eye_landmarks(
    landmarks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract left and right eye landmark points from a 468-point array.

    Parameters
    ----------
    landmarks : np.ndarray, shape (468, 3)
        Pixel-space (x, y, z) coordinates from MediaPipe FaceMesh.

    Returns
    -------
    left_eye  : np.ndarray, shape (6, 2)  – (x, y) only
    right_eye : np.ndarray, shape (6, 2)
    """
    left_eye  = landmarks[LEFT_EYE_LANDMARKS, :2]   # drop z
    right_eye = landmarks[RIGHT_EYE_LANDMARKS, :2]
    return left_eye, right_eye


def eye_center(eye_pts: np.ndarray) -> np.ndarray:
    """Return the centroid of an eye's 6 landmark points."""
    return eye_pts.mean(axis=0)
