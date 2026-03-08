# =============================================================================
# config.py  –  Global configuration for the Eye-Blink Deepfake Detector
# =============================================================================

# ── MediaPipe Face-Mesh landmark indices for each eye ─────────────────────────
# These are the 6 points used to compute the Eye Aspect Ratio (EAR).
LEFT_EYE_LANDMARKS  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33,  160, 158, 133, 153, 144]

# ── EAR / Blink thresholds ────────────────────────────────────────────────────
EAR_THRESHOLD          = 0.19   # EAR below this ⟹ eye is "closed"
BLINK_CONSEC_FRAMES    = 3      # consecutive closed frames to confirm a blink
EAR_SMOOTHING_WINDOW   = 5      # rolling-average window for EAR smoothing
MAX_BLINK_DURATION_S   = 1.0    # discard "blinks" longer than this (likely face-loss)

# ── Behavioural normality ranges (literature-derived) ─────────────────────────
NORMAL_BPM_MIN         = 6.0    # blinks per minute (lower bound)
NORMAL_BPM_MAX         = 30.0   # blinks per minute (upper bound)
NORMAL_BLINK_DUR_MIN   = 0.08   # seconds
NORMAL_BLINK_DUR_MAX   = 0.50   # seconds
NORMAL_SYMMETRY_MAX    = 0.25   # max|EAR_left – EAR_right| mean diff
NORMAL_VARIABILITY_MAX = 0.60   # coefficient of variation for inter-blink intervals

# ── Scoring weights (must sum to 1.0) ─────────────────────────────────────────
SCORE_WEIGHTS = {
    "bpm":        0.30,
    "duration":   0.20,
    "symmetry":   0.20,
    "variability":0.20,
    "count":      0.10,
}

# ── Visualization ─────────────────────────────────────────────────────────────
PLOT_DPI               = 120
RESULTS_DIR            = "results"
FONT_SCALE             = 0.55
FONT_THICKNESS         = 1
OSD_COLOR_NORMAL       = (0, 220, 0)    # BGR green
OSD_COLOR_SUSPICIOUS   = (0, 60, 230)   # BGR red-ish
OSD_COLOR_INFO         = (220, 220, 0)  # BGR cyan-ish
