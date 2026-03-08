"""
behavioral_analyzer.py
──────────────────────
Module 7 – Behavioural Analysis

Converts extracted features into a suspicion score (0–100) and a
human-readable verdict: REAL / SUSPICIOUS / LIKELY DEEPFAKE.

Scoring logic
─────────────
Each feature sub-score is 0 (normal) → 1 (highly abnormal).
A weighted sum gives a final score in [0, 1]; scaled to [0, 100].

Thresholds are based on published research on human blink behaviour:
  - Stern (1988), Doughty (2002) – BPM norms
  - Ngo (2012)                   – blink duration norms
  - Chugh (2020)                 – deepfake blink irregularity
"""

from __future__ import annotations
import numpy as np
from config import (
    NORMAL_BPM_MIN, NORMAL_BPM_MAX,
    NORMAL_BLINK_DUR_MIN, NORMAL_BLINK_DUR_MAX,
    NORMAL_SYMMETRY_MAX, NORMAL_VARIABILITY_MAX,
    SCORE_WEIGHTS,
)


def _range_score(value: float, low: float, high: float) -> float:
    """Return 0 if value in [low, high], else scale 0→1 with distance."""
    if value < low:
        return min(1.0, (low - value) / low)
    if value > high:
        return min(1.0, (value - high) / high)
    return 0.0


def analyze(features: dict) -> dict:
    """
    Parameters
    ----------
    features : dict produced by feature_extractor.extract_features()

    Returns
    -------
    dict with:
        sub_scores    – per-feature suspicion 0→1
        total_score   – weighted 0→100
        verdict       – "REAL" | "SUSPICIOUS" | "LIKELY DEEPFAKE"
        explanation   – list of human-readable strings
    """
    sub = {}
    exp = []

    bpm = features["bpm"]
    sub["bpm"] = _range_score(bpm, NORMAL_BPM_MIN, NORMAL_BPM_MAX)
    if sub["bpm"] > 0:
        exp.append(
            f"BPM={bpm:.1f} is outside normal range "
            f"[{NORMAL_BPM_MIN}–{NORMAL_BPM_MAX}]."
        )

    dur = features["mean_duration_s"]
    sub["duration"] = _range_score(dur, NORMAL_BLINK_DUR_MIN, NORMAL_BLINK_DUR_MAX)
    if sub["duration"] > 0:
        exp.append(
            f"Mean blink duration={dur*1000:.0f} ms outside normal "
            f"[{NORMAL_BLINK_DUR_MIN*1000:.0f}–{NORMAL_BLINK_DUR_MAX*1000:.0f} ms]."
        )

    sym = features["symmetry_diff"]
    sub["symmetry"] = min(1.0, sym / NORMAL_SYMMETRY_MAX) if NORMAL_SYMMETRY_MAX else 0.0
    if sym > NORMAL_SYMMETRY_MAX:
        exp.append(
            f"Eye symmetry diff={sym:.3f} exceeds threshold {NORMAL_SYMMETRY_MAX}."
        )

    cv = features["cv_ibi"]
    sub["variability"] = min(1.0, cv / NORMAL_VARIABILITY_MAX) if NORMAL_VARIABILITY_MAX else 0.0
    if cv > NORMAL_VARIABILITY_MAX:
        exp.append(
            f"Blink interval CoV={cv:.2f} indicates irregular pattern (>{NORMAL_VARIABILITY_MAX})."
        )

    # Blink count sub-score (penalise 0 blinks heavily)
    n = features["n_blinks"]
    dur_min = features["video_duration_s"] / 60.0
    expected_min = NORMAL_BPM_MIN * dur_min
    sub["count"] = max(0.0, min(1.0, 1.0 - n / max(expected_min, 1)))
    if n == 0:
        exp.append("No blinks detected – very likely deepfake or face not found.")

    # ── Weighted total ────────────────────────────────────────────────────
    weights = SCORE_WEIGHTS
    total = sum(sub[k] * weights[k] for k in sub)
    score_pct = round(total * 100, 1)

    if score_pct < 30:
        verdict = "REAL"
    elif score_pct < 60:
        verdict = "SUSPICIOUS"
    else:
        verdict = "LIKELY DEEPFAKE"

    if not exp:
        exp.append("All blink metrics are within normal human ranges.")

    return {
        "sub_scores":   {k: round(v, 3) for k, v in sub.items()},
        "total_score":  score_pct,
        "verdict":      verdict,
        "explanation":  exp,
    }
