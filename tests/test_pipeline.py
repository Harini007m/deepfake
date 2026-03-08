"""
tests/test_pipeline.py
──────────────────────
Unit & integration tests using pytest.
Run:  pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from ear_calculator    import compute_ear, EARSmoother
from blink_detector    import BlinkDetector, BlinkEvent
from feature_extractor import extract_features
from behavioral_analyzer import analyze


# ══════════════════════════════════════════════════════════════════════════════
# EAR Calculator
# ══════════════════════════════════════════════════════════════════════════════

class TestEARCalculator:

    def _open_eye(self):
        """6 points that simulate an open eye (EAR ~ 0.35)."""
        return np.array([
            [0.0, 0.0],   # outer corner
            [1.0, 1.5],   # top1
            [2.0, 1.5],   # top2
            [3.0, 0.0],   # inner corner
            [2.0, -1.5],  # bot2
            [1.0, -1.5],  # bot1
        ], dtype=np.float32)

    def _closed_eye(self):
        """6 points where vertical distance → 0 (EAR ~ 0)."""
        return np.array([
            [0.0, 0.0],
            [1.0, 0.05],
            [2.0, 0.05],
            [3.0, 0.0],
            [2.0, -0.05],
            [1.0, -0.05],
        ], dtype=np.float32)

    def test_open_ear_above_threshold(self):
        ear = compute_ear(self._open_eye())
        assert ear > 0.22, f"Expected EAR > 0.22 for open eye, got {ear:.4f}"

    def test_closed_ear_below_threshold(self):
        ear = compute_ear(self._closed_eye())
        assert ear < 0.15, f"Expected EAR < 0.15 for closed eye, got {ear:.4f}"

    def test_ear_non_negative(self):
        for _ in range(20):
            pts = np.random.rand(6, 2).astype(np.float32) * 100
            assert compute_ear(pts) >= 0.0

    def test_smoother_stabilises(self):
        smoother = EARSmoother(window=3)
        values = [0.30, 0.10, 0.30]
        last_l, last_r = 0.0, 0.0
        for v in values:
            last_l, last_r = smoother.update(v, v)
        # Mean of [0.30, 0.10, 0.30] = 0.2333
        assert abs(last_l - 0.2333) < 0.01


# ══════════════════════════════════════════════════════════════════════════════
# Blink Detector
# ══════════════════════════════════════════════════════════════════════════════

class TestBlinkDetector:

    def _simulate_blink(self, detector, start_frame, n_frames=3, low=0.15, high=0.30, fps=25):
        """Feed low EAR for n_frames then high EAR, return event."""
        event = None
        for i in range(n_frames):
            event = detector.update(start_frame + i, low, low)
        # Eye opens
        e = detector.update(start_frame + n_frames, high, high)
        return e  # should be the completed BlinkEvent

    def test_single_blink_detected(self):
        det = BlinkDetector(fps=25)
        event = self._simulate_blink(det, 0)
        assert event is not None, "Expected a blink event"
        assert isinstance(event, BlinkEvent)

    def test_no_blink_when_above_threshold(self):
        det = BlinkDetector(fps=25)
        for i in range(50):
            e = det.update(i, 0.35, 0.35)
        assert det.total_blinks() == 0

    def test_multiple_blinks(self):
        det = BlinkDetector(fps=25)
        for b in range(5):
            start = b * 30
            self._simulate_blink(det, start)
        assert det.total_blinks() == 5

    def test_blink_duration_positive(self):
        det = BlinkDetector(fps=25)
        self._simulate_blink(det, start_frame=0, n_frames=4, fps=25)
        if det.blinks:
            assert det.blinks[0].duration > 0

    def test_ear_min_recorded(self):
        det = BlinkDetector(fps=25, ear_threshold=0.22, consec_frames=2)
        self._simulate_blink(det, 0, n_frames=3, low=0.10)
        if det.blinks:
            assert det.blinks[0].left_ear_min <= 0.15


# ══════════════════════════════════════════════════════════════════════════════
# Feature Extractor
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureExtractor:

    def _make_blinks(self, n=5, fps=25, gap=60):
        """Create n synthetic blink events spaced `gap` frames apart."""
        blinks = []
        for i in range(n):
            start_f = i * gap
            end_f   = start_f + 3
            b = BlinkEvent(
                frame_start   = start_f,
                frame_end     = end_f,
                time_start    = start_f / fps,
                time_end      = end_f   / fps,
                left_ear_min  = 0.10,
                right_ear_min = 0.11,
            )
            blinks.append(b)
        return blinks

    def test_bpm_reasonable(self):
        fps     = 25
        blinks  = self._make_blinks(n=10, fps=fps, gap=75)  # ~8 blinks/min
        total_f = 10 * 75
        dur     = total_f / fps
        ears    = [0.30] * total_f
        feats   = extract_features(blinks, ears, ears, dur, fps)
        # BPM = 10 / (total_f/fps/60)
        assert feats["bpm"] > 0

    def test_symmetry_zero_for_identical_ears(self):
        ears  = [0.30] * 100
        feats = extract_features([], ears, ears, 4.0, 25)
        assert feats["symmetry_diff"] == pytest.approx(0.0, abs=1e-4)

    def test_zero_blinks(self):
        ears  = [0.35] * 250
        feats = extract_features([], ears, ears, 10.0, 25)
        assert feats["n_blinks"] == 0
        assert feats["bpm"] == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Behavioural Analyser
# ══════════════════════════════════════════════════════════════════════════════

class TestBehavioralAnalyzer:

    def _normal_features(self):
        return {
            "n_blinks":         15,
            "video_duration_s": 60.0,
            "bpm":              15.0,
            "mean_duration_s":  0.2,
            "std_duration_s":   0.02,
            "mean_ibi_s":       3.5,
            "std_ibi_s":        0.3,
            "cv_ibi":           0.08,
            "symmetry_diff":    0.01,
            "mean_left_ear":    0.30,
            "mean_right_ear":   0.30,
            "ear_variability":  0.02,
            "ibis":             [3.5] * 14,
            "durations":        [0.2] * 15,
        }

    def _suspicious_features(self):
        return {
            "n_blinks":         2,
            "video_duration_s": 60.0,
            "bpm":              2.0,          # way below normal
            "mean_duration_s":  0.8,          # too long
            "std_duration_s":   0.3,
            "mean_ibi_s":       25.0,
            "std_ibi_s":        12.0,
            "cv_ibi":           0.9,          # very irregular
            "symmetry_diff":    0.40,         # high asymmetry
            "mean_left_ear":    0.25,
            "mean_right_ear":   0.15,
            "ear_variability":  0.05,
            "ibis":             [25.0],
            "durations":        [0.8, 0.8],
        }

    def test_normal_verdict_is_real(self):
        result = analyze(self._normal_features())
        assert result["verdict"] == "REAL"
        assert result["total_score"] < 30

    def test_suspicious_verdict_not_real(self):
        result = analyze(self._suspicious_features())
        assert result["verdict"] != "REAL"
        assert result["total_score"] >= 30

    def test_output_keys_present(self):
        result = analyze(self._normal_features())
        for key in ("sub_scores", "total_score", "verdict", "explanation"):
            assert key in result

    def test_score_range(self):
        for _ in range(20):
            # Randomly perturbed features
            f = self._normal_features()
            f["bpm"]          = np.random.uniform(0, 50)
            f["cv_ibi"]       = np.random.uniform(0, 2)
            f["symmetry_diff"]= np.random.uniform(0, 0.5)
            result = analyze(f)
            assert 0 <= result["total_score"] <= 100
