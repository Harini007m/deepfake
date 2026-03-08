"""
feature_extractor.py
────────────────────
Module 6 – Behavioural Feature Extraction

Converts a list of BlinkEvent objects and EAR time-series into a rich
feature dictionary ready for the behavioural analyser.
"""

from __future__ import annotations
import numpy as np
from blink_detector import BlinkEvent


def extract_features(
    blinks:       list[BlinkEvent],
    left_ears:    list[float],
    right_ears:   list[float],
    video_duration_s: float,
    fps: float,
) -> dict:
    """
    Parameters
    ----------
    blinks            : detected blink events
    left_ears         : per-frame smoothed left  EAR
    right_ears        : per-frame smoothed right EAR
    video_duration_s  : total video length in seconds
    fps               : frames per second

    Returns
    -------
    dict of float features
    """
    n_blinks = len(blinks)
    duration_min = video_duration_s / 60.0

    # ── Blinks per minute ─────────────────────────────────────────────────
    bpm = n_blinks / duration_min if duration_min > 0 else 0.0

    # ── Blink duration stats ───────────────────────────────────────────────
    durations = [b.duration for b in blinks]
    mean_duration = float(np.mean(durations)) if durations else 0.0
    std_duration  = float(np.std(durations))  if durations else 0.0

    # ── Inter-blink interval (IBI) ─────────────────────────────────────────
    if len(blinks) >= 2:
        ibis = [
            blinks[i+1].time_start - blinks[i].time_end
            for i in range(len(blinks) - 1)
        ]
        mean_ibi = float(np.mean(ibis))
        std_ibi  = float(np.std(ibis))
        cv_ibi   = std_ibi / mean_ibi if mean_ibi > 0 else 0.0   # variability
    else:
        ibis     = []
        mean_ibi = 0.0
        std_ibi  = 0.0
        cv_ibi   = 0.0

    # ── Eye symmetry (mean absolute EAR difference) ───────────────────────
    left_arr  = np.array(left_ears,  dtype=np.float32)
    right_arr = np.array(right_ears, dtype=np.float32)
    symmetry_diff = float(np.mean(np.abs(left_arr - right_arr)))

    mean_left_ear  = float(np.mean(left_arr))
    mean_right_ear = float(np.mean(right_arr))

    # ── EAR variability during open-eye periods ────────────────────────────
    # Collect frames where eyes were open
    open_mask = (left_arr + right_arr) / 2 > 0.22   # above threshold
    open_ears = ((left_arr + right_arr) / 2)[open_mask]
    ear_variability = float(np.std(open_ears)) if open_ears.size > 0 else 0.0

    return {
        "n_blinks":         n_blinks,
        "video_duration_s": round(video_duration_s, 2),
        "bpm":              round(bpm, 3),
        "mean_duration_s":  round(mean_duration, 4),
        "std_duration_s":   round(std_duration,  4),
        "mean_ibi_s":       round(mean_ibi, 4),
        "std_ibi_s":        round(std_ibi,  4),
        "cv_ibi":           round(cv_ibi,   4),   # blink variability
        "symmetry_diff":    round(symmetry_diff, 4),
        "mean_left_ear":    round(mean_left_ear,  4),
        "mean_right_ear":   round(mean_right_ear, 4),
        "ear_variability":  round(ear_variability, 4),
        "ibis":             ibis,
        "durations":        durations,
    }
