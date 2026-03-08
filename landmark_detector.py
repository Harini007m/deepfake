"""
landmark_detector.py
────────────────────
Module 2 – Face Landmark Detection

Uses the MediaPipe Tasks API (FaceLandmarker) to detect 478 3-D facial
landmarks per frame.  Works with mediapipe >= 0.10.30 (Tasks API).

The model file `face_landmarker.task` must be present in the project root
or the path specified by `model_path`.
"""

from __future__ import annotations
import numpy as np
import mediapipe as mp
from pathlib import Path

# ── Locate model file ────────────────────────────────────────────────────────
_DEFAULT_MODEL = str(Path(__file__).resolve().parent / "face_landmarker.task")

BaseOptions         = mp.tasks.BaseOptions
FaceLandmarker      = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode   = mp.tasks.vision.RunningMode


class FaceLandmarkDetector:
    """Wraps MediaPipe FaceLandmarker for per-frame landmark detection."""

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}\n"
                f"Download it from:\n"
                f"  https://storage.googleapis.com/mediapipe-models/"
                f"face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._frame_ts_ms = 0  # monotonically increasing timestamp

    def detect(self, bgr_frame: np.ndarray) -> list[np.ndarray] | None:
        """
        Detect facial landmarks in a BGR frame.

        Parameters
        ----------
        bgr_frame : np.ndarray
            BGR image from OpenCV.

        Returns
        -------
        list[np.ndarray] | None
            A list of Nx3 arrays (x, y, z) in *pixel* coordinates
            for each detected face, or None if no face was found.
        """
        h, w = bgr_frame.shape[:2]

        # Convert BGR → RGB and wrap in mp.Image
        rgb = bgr_frame[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # detect_for_video requires a monotonically increasing timestamp
        self._frame_ts_ms += 33  # ~30 fps increment
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        if not result.face_landmarks:
            return None

        faces = []
        for face_lm in result.face_landmarks:
            pts = np.array(
                [[lm.x * w, lm.y * h, lm.z * w] for lm in face_lm],
                dtype=np.float32,
            )
            faces.append(pts)
        return faces

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
