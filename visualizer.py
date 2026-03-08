"""
visualizer.py
─────────────
Module 8 – Result Visualisation

Two responsibilities:
  1. OSD overlay drawn on live video frames during processing.
  2. Post-processing matplotlib report (EAR timeline, blink events,
     metrics bar, score gauge) saved to a PNG file.
"""

from __future__ import annotations
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")                       # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from config import (
    PLOT_DPI, RESULTS_DIR,
    FONT_SCALE, FONT_THICKNESS,
    OSD_COLOR_NORMAL, OSD_COLOR_SUSPICIOUS, OSD_COLOR_INFO,
    EAR_THRESHOLD,
)


# ─────────────────────────────────────────────────────────────────────────────
# OSD helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_osd(
    frame:         np.ndarray,
    blink_count:   int,
    left_ear:      float,
    right_ear:     float,
    fps:           float,
    verdict:       str = "",
    score:         float = -1,
) -> np.ndarray:
    """Draw on-screen display onto a copy of *frame*."""
    out = frame.copy()
    h, w = out.shape[:2]

    verdict_color = (
        OSD_COLOR_NORMAL     if verdict == "REAL"          else
        OSD_COLOR_SUSPICIOUS if verdict == "SUSPICIOUS"    else
        (0, 30, 200)         if verdict == "LIKELY DEEPFAKE" else
        OSD_COLOR_INFO
    )

    lines = [
        (f"Blinks: {blink_count}",                         OSD_COLOR_INFO),
        (f"L-EAR: {left_ear:.3f}  R-EAR: {right_ear:.3f}", OSD_COLOR_INFO),
        (f"FPS: {fps:.1f}",                                 OSD_COLOR_INFO),
    ]
    if verdict:
        lines.append((f"Verdict: {verdict}", verdict_color))
    if score >= 0:
        lines.append((f"Suspicion: {score:.1f}%", verdict_color))

    y = 25
    for text, color in lines:
        cv2.putText(
            out, text, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), FONT_THICKNESS + 2,
        )
        cv2.putText(
            out, text, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, FONT_THICKNESS,
        )
        y += 22
    return out


def draw_eye_contours(
    frame:     np.ndarray,
    left_pts:  np.ndarray,
    right_pts: np.ndarray,
    is_blink:  bool,
) -> np.ndarray:
    """Draw eye contour polygons."""
    color = (0, 0, 220) if is_blink else (0, 220, 0)
    out = frame.copy()
    for pts in (left_pts, right_pts):
        pts_int = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts_int], isClosed=True, color=color, thickness=1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib analytical report
# ─────────────────────────────────────────────────────────────────────────────

def _verdict_color(verdict: str) -> str:
    return {"REAL": "#2ecc71", "SUSPICIOUS": "#f39c12", "LIKELY DEEPFAKE": "#e74c3c"}.get(
        verdict, "#95a5a6"
    )


def generate_report(
    video_name:   str,
    left_ears:    list[float],
    right_ears:   list[float],
    frame_times:  list[float],
    blinks:       list,           # list[BlinkEvent]
    features:     dict,
    analysis:     dict,
    output_dir:   str = RESULTS_DIR,
) -> str:
    """
    Generate and save a multi-panel analytical report as a PNG.

    Returns the path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = os.path.splitext(os.path.basename(video_name))[0]
    out_path  = os.path.join(output_dir, f"{safe_name}_report.png")

    left_arr  = np.array(left_ears)
    right_arr = np.array(right_ears)
    times     = np.array(frame_times)

    vc     = _verdict_color(analysis["verdict"])
    score  = analysis["total_score"]

    fig = plt.figure(figsize=(16, 11), facecolor="#1a1a2e")
    gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Common style
    plt.rcParams.update({
        "text.color":   "white",
        "axes.facecolor":   "#16213e",
        "axes.edgecolor":   "#0f3460",
        "axes.labelcolor":  "white",
        "xtick.color":      "#aaaaaa",
        "ytick.color":      "#aaaaaa",
        "grid.color":       "#0f3460",
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
    })

    # ── Panel 1: EAR Time-Series ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor("#16213e")
    ax1.plot(times, left_arr,  color="#3498db", lw=1.0, label="Left EAR",  alpha=0.9)
    ax1.plot(times, right_arr, color="#e74c3c", lw=1.0, label="Right EAR", alpha=0.9)
    ax1.axhline(EAR_THRESHOLD, color="#f1c40f", lw=1.2, linestyle="--", label=f"Threshold ({EAR_THRESHOLD})")

    for blink in blinks:
        ax1.axvspan(blink.time_start, blink.time_end, alpha=0.25, color="#9b59b6")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("EAR")
    ax1.set_title("Eye Aspect Ratio Over Time", color="white", fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.3, fontsize=8)
    ax1.grid(True)

    # ── Panel 2: Score Gauge ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("#16213e")
    theta = np.linspace(np.pi, 0, 300)
    # Background arc
    ax2.plot(np.cos(theta), np.sin(theta), color="#333355", lw=20, solid_capstyle="butt")
    # Coloured arc up to score
    score_theta = np.linspace(np.pi, np.pi - np.pi * score / 100, 300)
    ax2.plot(np.cos(score_theta), np.sin(score_theta), color=vc, lw=20, solid_capstyle="butt")
    ax2.text(0, -0.15, f"{score:.0f}%", ha="center", va="center",
             fontsize=26, fontweight="bold", color=vc)
    ax2.text(0, -0.45, "Suspicion Score", ha="center", color="#aaaaaa", fontsize=9)
    ax2.text(0, -0.62, analysis["verdict"], ha="center", color=vc,
             fontsize=11, fontweight="bold")
    ax2.set_xlim(-1.3, 1.3); ax2.set_ylim(-0.75, 1.1)
    ax2.axis("off")

    # ── Panel 3: Blink Duration Distribution ────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#16213e")
    durations = features.get("durations", [])
    if durations:
        ax3.hist(durations, bins=max(5, len(durations)//3),
                 color="#3498db", edgecolor="#16213e", alpha=0.85)
    ax3.set_xlabel("Duration (s)"); ax3.set_ylabel("Count")
    ax3.set_title("Blink Duration Dist.", color="white", fontweight="bold")
    ax3.grid(True)

    # ── Panel 4: Inter-Blink Interval ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#16213e")
    ibis = features.get("ibis", [])
    if ibis:
        ax4.plot(range(len(ibis)), ibis, color="#2ecc71", lw=1.5, marker="o",
                 markersize=4, alpha=0.85)
    ax4.set_xlabel("Blink #"); ax4.set_ylabel("IBI (s)")
    ax4.set_title("Inter-Blink Intervals", color="white", fontweight="bold")
    ax4.grid(True)

    # ── Panel 5: Sub-Score Bar Chart ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#16213e")
    sub = analysis["sub_scores"]
    keys   = list(sub.keys())
    values = [sub[k] * 100 for k in keys]
    colors = ["#e74c3c" if v > 50 else "#f39c12" if v > 20 else "#2ecc71"
              for v in values]
    ax5.barh(keys, values, color=colors, edgecolor="#16213e")
    ax5.set_xlim(0, 100)
    ax5.set_xlabel("Sub-score (%)")
    ax5.set_title("Anomaly Sub-Scores", color="white", fontweight="bold")
    ax5.grid(True, axis="x")

    # ── Panel 6: Left vs Right EAR scatter ───────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.set_facecolor("#16213e")
    sample = slice(None, None, max(1, len(left_arr)//500))  # down-sample
    ax6.scatter(left_arr[sample], right_arr[sample],
                s=5, color="#9b59b6", alpha=0.5)
    lim_max = max(left_arr.max(), right_arr.max()) * 1.05
    ax6.plot([0, lim_max], [0, lim_max], color="#f1c40f", lw=1, linestyle="--")
    ax6.set_xlabel("Left EAR"); ax6.set_ylabel("Right EAR")
    ax6.set_title("L vs R EAR Symmetry", color="white", fontweight="bold")
    ax6.grid(True)

    # ── Panel 7: Key Metrics Text Table ─────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.set_facecolor("#16213e")
    ax7.axis("off")

    rows = [
        ["Video",               os.path.basename(video_name)],
        ["Duration (s)",        f"{features['video_duration_s']:.1f}"],
        ["Total Blinks",        str(features["n_blinks"])],
        ["Blinks / min",        f"{features['bpm']:.1f}"],
        ["Mean Blink Dur (ms)", f"{features['mean_duration_s']*1000:.0f}"],
        ["Mean IBI (s)",        f"{features['mean_ibi_s']:.2f}"],
        ["Blink Variability",   f"{features['cv_ibi']:.3f}"],
        ["Eye Symmetry Diff",   f"{features['symmetry_diff']:.3f}"],
        ["Verdict",             analysis["verdict"]],
        ["Suspicion Score",     f"{analysis['total_score']:.1f}%"],
    ]
    table = ax7.table(
        cellText  = rows,
        colLabels = ["Metric", "Value"],
        cellLoc   = "left",
        loc       = "center",
        bbox      = [0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("#1a1a2e" if r % 2 == 0 else "#16213e")
        cell.set_text_props(color="white" if r > 0 else "#f1c40f")
        cell.set_edgecolor("#0f3460")
    ax7.set_title("Summary Metrics", color="white", fontweight="bold", pad=8)

    # ── Super title ──────────────────────────────────────────────────────
    fig.suptitle(
        "Eye-Blink Deepfake Detection Report",
        color="white", fontsize=16, fontweight="bold", y=0.98,
    )

    plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path
