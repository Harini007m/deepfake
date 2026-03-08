"""
pipeline.py
───────────
Core Orchestrator – ties all modules together.

Usage (programmatic)
--------------------
    from pipeline import DeepfakeDetectionPipeline

    pipeline = DeepfakeDetectionPipeline("my_video.mp4")
    results  = pipeline.run()            # returns dict with all results
    print(results["analysis"]["verdict"])
"""

from __future__ import annotations
import time
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from video_input        import VideoReader
from landmark_detector  import FaceLandmarkDetector
from eye_landmarks      import extract_eye_landmarks
from ear_calculator     import compute_ear, EARSmoother
from blink_detector     import BlinkDetector
from feature_extractor  import extract_features
from behavioral_analyzer import analyze
from visualizer          import draw_osd, draw_eye_contours, generate_report
from config              import EAR_THRESHOLD, BLINK_CONSEC_FRAMES


class DeepfakeDetectionPipeline:
    """Full blink-based deepfake detection pipeline."""

    def __init__(
        self,
        video_path:     str,
        show_video:     bool = False,
        save_output_video: bool = False,
        output_dir:     str  = "results",
    ):
        self.video_path        = video_path
        self.show_video        = show_video
        self.save_output_video = save_output_video
        self.output_dir        = output_dir

    # ─────────────────────────────────────────────────────────────────────
    def run(self, progress_callback=None) -> dict:
        """
        Run the full pipeline.

        Parameters
        ----------
        progress_callback : callable(current_frame, total_frames) | None

        Returns
        -------
        dict with keys:
            features, analysis, report_path, blinks,
            left_ears, right_ears, frame_times, video_info
        """
        video_path = self.video_path
        t0 = time.time()

        # ── Modules ──────────────────────────────────────────────────────
        reader   = VideoReader(video_path)
        detector = FaceLandmarkDetector()
        smoother = EARSmoother()
        blinker  = BlinkDetector(
            ear_threshold=EAR_THRESHOLD,
            consec_frames=BLINK_CONSEC_FRAMES,
            fps=reader.fps,
        )

        video_info  = reader.info()
        total       = reader.total_frames
        fps         = reader.fps

        left_ears:   list[float] = []
        right_ears:  list[float] = []
        frame_times: list[float] = []

        out_writer = None
        if self.save_output_video:
            import os
            os.makedirs(self.output_dir, exist_ok=True)
            safe = Path(video_path).stem
            out_path = str(Path(self.output_dir) / f"{safe}_annotated.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(
                out_path, fourcc, fps, (reader.width, reader.height)
            )

        # ── Frame loop ────────────────────────────────────────────────────
        pbar = tqdm(total=total, desc="Analysing frames", unit="fr", leave=False)

        for frame_idx, frame in reader:
            pbar.update(1)
            if progress_callback:
                progress_callback(frame_idx + 1, total)

            time_sec  = frame_idx / fps
            left_ear  = 0.0
            right_ear = 0.0
            left_pts  = None
            right_pts = None

            faces = detector.detect(frame)
            if faces:
                landmarks = faces[0]          # analyse the primary face
                left_pts, right_pts = extract_eye_landmarks(landmarks)
                left_ear_raw  = compute_ear(left_pts)
                right_ear_raw = compute_ear(right_pts)
                left_ear, right_ear = smoother.update(left_ear_raw, right_ear_raw)

            blinker.update(frame_idx, left_ear, right_ear)
            left_ears.append(left_ear)
            right_ears.append(right_ear)
            frame_times.append(time_sec)

            # Visual output (optional)
            if self.show_video or out_writer:
                vis = frame.copy()
                if left_pts is not None:
                    is_blink = (left_ear + right_ear) / 2 < EAR_THRESHOLD
                    vis = draw_eye_contours(vis, left_pts, right_pts, is_blink)
                vis = draw_osd(
                    vis,
                    blink_count=blinker.total_blinks(),
                    left_ear=left_ear,
                    right_ear=right_ear,
                    fps=fps,
                )
                if out_writer:
                    out_writer.write(vis)
                if self.show_video:
                    cv2.imshow("Eye-Blink Deepfake Detector", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        pbar.close()
        if self.show_video:
            cv2.destroyAllWindows()
        if out_writer:
            out_writer.release()
        reader.release()
        detector.close()

        # ── Feature extraction ────────────────────────────────────────────
        features = extract_features(
            blinks            = blinker.blinks,
            left_ears         = left_ears,
            right_ears        = right_ears,
            video_duration_s  = reader.duration_seconds,
            fps               = fps,
        )

        # ── Behavioural analysis ──────────────────────────────────────────
        analysis = analyze(features)

        # ── Report generation ─────────────────────────────────────────────
        report_path = generate_report(
            video_name   = video_path,
            left_ears    = left_ears,
            right_ears   = right_ears,
            frame_times  = frame_times,
            blinks       = blinker.blinks,
            features     = features,
            analysis     = analysis,
            output_dir   = self.output_dir,
        )

        elapsed = time.time() - t0

        return {
            "video_info":   video_info,
            "features":     features,
            "analysis":     analysis,
            "report_path":  report_path,
            "blinks":       blinker.blinks,
            "left_ears":    left_ears,
            "right_ears":   right_ears,
            "frame_times":  frame_times,
            "elapsed_s":    round(elapsed, 2),
        }
