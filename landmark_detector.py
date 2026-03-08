"""
landmark_detector.py
────────────────────
Module 2 – Face Landmark Detection
Uses MediaPipe Face Mesh to detect 468 3-D facial landmarks per frame.
"""

from __future__ import annotations
import numpy as np
import mediapipe as mp


class FaceLandmarkDetector:
    """Wraps MediaPipe FaceMesh for single-face landmark detection."""

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, bgr_frame: np.ndarray) -> list[np.ndarray] | None:
        """
        Detect facial landmarks in a BGR frame.

        Returns
        -------
        list[np.ndarray] | None
            A list of Nx3 arrays (x, y, z) in *pixel* coordinates for each
            detected face, or None if no face was found.
        """
        h, w = bgr_frame.shape[:2]
        rgb = bgr_frame[:, :, ::-1]          # BGR → RGB
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        faces = []
        for face_lm in results.multi_face_landmarks:
            pts = np.array(
                [[lm.x * w, lm.y * h, lm.z * w] for lm in face_lm.landmark],
                dtype=np.float32,
            )
            faces.append(pts)
        return faces

    def close(self):
        self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
