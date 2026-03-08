"""
dataset_utils.py
────────────────
Utilities for evaluating the Eye-Blink detector against deepfake datasets.

Natively supports:
  ● Celeb-DF v2   (Celeb-real, YouTube-real, Celeb-synthesis)
  ● UADFV         (real/ + fake/)
  ● Generic       (real/ + fake/)

Usage
-----
    # Celeb-DF v2 (auto-detected from folder names)
    python dataset_utils.py --dataset D:\\archive --output-dir results/celebdf

    # Quick test – 5 videos per class
    python dataset_utils.py --dataset D:\\archive --max-per-class 5

    # Generic layout
    python dataset_utils.py --dataset /path/to/dataset --output-dir results/generic
"""

from __future__ import annotations
import argparse
import json
import csv
import time
from pathlib import Path
from pipeline import DeepfakeDetectionPipeline


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ══════════════════════════════════════════════════════════════════════════════
# Video collection – supports multiple dataset layouts
# ══════════════════════════════════════════════════════════════════════════════

def collect_videos(root: Path) -> dict[str, list[Path]]:
    """
    Discover real / fake videos in a dataset folder.

    Supported layouts (checked in order):

    1. **Celeb-DF v2** layout:
         root/Celeb-real/       → real
         root/YouTube-real/     → real
         root/Celeb-synthesis/  → fake

    2. **Standard** layout:
         root/real/   → real
         root/fake/   → fake

    3. **Filename fallback**:
         files containing "real" → real
         files containing "fake" or "df" → fake
    """
    # ── Layout 1: Celeb-DF v2 ─────────────────────────────────────────────
    celeb_real    = root / "Celeb-real"
    youtube_real  = root / "YouTube-real"
    celeb_synth   = root / "Celeb-synthesis"

    if celeb_real.exists() or youtube_real.exists() or celeb_synth.exists():
        real_videos = []
        if celeb_real.exists():
            real_videos += sorted(
                p for p in celeb_real.rglob("*") if p.suffix.lower() in VIDEO_EXTS
            )
        if youtube_real.exists():
            real_videos += sorted(
                p for p in youtube_real.rglob("*") if p.suffix.lower() in VIDEO_EXTS
            )
        fake_videos = []
        if celeb_synth.exists():
            fake_videos = sorted(
                p for p in celeb_synth.rglob("*") if p.suffix.lower() in VIDEO_EXTS
            )

        print(f"  ✓ Detected Celeb-DF v2 layout")
        print(f"    Celeb-real    : {len([p for p in real_videos if 'Celeb-real' in str(p)])} videos")
        print(f"    YouTube-real  : {len([p for p in real_videos if 'YouTube-real' in str(p)])} videos")
        print(f"    Celeb-synthesis: {len(fake_videos)} videos")
        print(f"    Total real: {len(real_videos)}  |  Total fake: {len(fake_videos)}")
        return {"real": real_videos, "fake": fake_videos}

    # ── Layout 2: Standard real/fake ──────────────────────────────────────
    real_dir = root / "real"
    fake_dir = root / "fake"
    if real_dir.exists() and fake_dir.exists():
        print(f"  ✓ Detected standard real/fake layout")
        return {
            "real": sorted(p for p in real_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS),
            "fake": sorted(p for p in fake_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS),
        }

    # ── Layout 3: Filename fallback ───────────────────────────────────────
    print(f"  ⚠ No known layout detected — inferring labels from filenames")
    all_vids = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    real = [p for p in all_vids if "real" in p.stem.lower()]
    fake = [p for p in all_vids if "fake" in p.stem.lower() or "synth" in p.stem.lower()]
    return {"real": real, "fake": fake}


# ══════════════════════════════════════════════════════════════════════════════
# Dataset processing
# ══════════════════════════════════════════════════════════════════════════════

def process_dataset(
    dataset_root:   str,
    output_dir:     str       = "results/dataset",
    max_per_class:  int | None = None,
    save_reports:   bool      = True,
) -> list[dict]:
    """
    Process all videos in a dataset.

    Returns a list of per-video result dicts.
    """
    root    = Path(dataset_root)
    videos  = collect_videos(root)
    rows: list[dict] = []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_vids = sum(
        min(len(v), max_per_class) if max_per_class else len(v)
        for v in videos.values()
    )
    processed = 0
    errors    = 0
    t0 = time.time()

    for label, paths in videos.items():
        if max_per_class:
            paths = paths[:max_per_class]
        print(f"\n{'═'*60}")
        print(f"  {label.upper()} VIDEOS  ({len(paths)} to process)")
        print(f"{'═'*60}")

        for vp in paths:
            processed += 1
            print(f"  [{processed}/{total_vids}] {vp.name}", end=" … ", flush=True)
            try:
                pipeline = DeepfakeDetectionPipeline(
                    video_path  = str(vp),
                    output_dir  = str(out_dir / label) if save_reports else str(out_dir),
                )
                r = pipeline.run()
                a = r["analysis"]
                f = r["features"]

                # Determine source folder for clarity
                source_folder = vp.parent.name

                row = {
                    "file":          vp.name,
                    "source_folder": source_folder,
                    "label":         label,
                    "n_blinks":      f["n_blinks"],
                    "bpm":           f["bpm"],
                    "mean_dur_ms":   round(f["mean_duration_s"] * 1000, 1),
                    "mean_ibi_s":    f["mean_ibi_s"],
                    "cv_ibi":        f["cv_ibi"],
                    "symmetry":      f["symmetry_diff"],
                    "ear_var":       f["ear_variability"],
                    "score":         a["total_score"],
                    "verdict":       a["verdict"],
                    "correct":       (
                        (label == "real" and a["verdict"] == "REAL") or
                        (label == "fake" and a["verdict"] != "REAL")
                    ),
                }
                rows.append(row)

                emoji = {"REAL": "✅", "SUSPICIOUS": "⚠️", "LIKELY DEEPFAKE": "🚨"}.get(
                    a["verdict"], "❓"
                )
                correct_mark = "✓" if row["correct"] else "✗"
                print(
                    f"{emoji} {a['verdict']}  "
                    f"score={a['total_score']:5.1f}%  "
                    f"blinks={f['n_blinks']:3d}  "
                    f"bpm={f['bpm']:5.1f}  "
                    f"[{correct_mark}]"
                )
            except Exception as exc:
                errors += 1
                print(f"❌ ERROR: {exc}")

    elapsed = time.time() - t0
    print(f"\n{'─'*60}")
    print(f"  Processed {processed} videos in {elapsed:.1f}s  ({errors} errors)")
    print(f"{'─'*60}")

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Metrics computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(rows: list[dict]) -> dict:
    """Compute accuracy, precision, recall, F1 for binary real/fake."""
    tp = sum(1 for r in rows if r["label"] == "fake" and r["verdict"] != "REAL")
    tn = sum(1 for r in rows if r["label"] == "real" and r["verdict"] == "REAL")
    fp = sum(1 for r in rows if r["label"] == "real" and r["verdict"] != "REAL")
    fn = sum(1 for r in rows if r["label"] == "fake" and r["verdict"] == "REAL")

    total = len(rows)
    accuracy  = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # ── Per-class stats ───────────────────────────────────────────────────
    import numpy as np
    real_rows = [r for r in rows if r["label"] == "real"]
    fake_rows = [r for r in rows if r["label"] == "fake"]

    def _class_stats(subset):
        if not subset:
            return {}
        bpms   = [r["bpm"] for r in subset]
        scores = [r["score"] for r in subset]
        return {
            "count":       len(subset),
            "mean_bpm":    round(float(np.mean(bpms)), 2),
            "std_bpm":     round(float(np.std(bpms)), 2),
            "mean_score":  round(float(np.mean(scores)), 2),
            "std_score":   round(float(np.std(scores)), 2),
        }

    return {
        "total":     total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy":  round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall":    round(recall * 100, 2),
        "f1":        round(f1 * 100, 2),
        "real_stats": _class_stats(real_rows),
        "fake_stats": _class_stats(fake_rows),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Results output
# ══════════════════════════════════════════════════════════════════════════════

def print_metrics(metrics: dict):
    print(f"\n{'═'*60}")
    print("     DATASET EVALUATION SUMMARY")
    print(f"{'═'*60}")
    print(f"  Total videos analysed : {metrics['total']}")
    print()
    print(f"  Confusion Matrix:")
    print(f"    True  Positives (fake→detected)  : {metrics['tp']}")
    print(f"    True  Negatives (real→REAL)       : {metrics['tn']}")
    print(f"    False Positives (real→flagged)    : {metrics['fp']}")
    print(f"    False Negatives (fake→missed)     : {metrics['fn']}")
    print()
    print(f"  Accuracy  : {metrics['accuracy']:6.2f}%")
    print(f"  Precision : {metrics['precision']:6.2f}%")
    print(f"  Recall    : {metrics['recall']:6.2f}%")
    print(f"  F1 Score  : {metrics['f1']:6.2f}%")

    for label in ("real", "fake"):
        key = f"{label}_stats"
        if key in metrics and metrics[key]:
            s = metrics[key]
            print(f"\n  {label.upper()} class ({s['count']} videos):")
            print(f"    Mean BPM          : {s['mean_bpm']:.2f} ± {s['std_bpm']:.2f}")
            print(f"    Mean Suspicion    : {s['mean_score']:.2f} ± {s['std_score']:.2f}")

    print(f"{'═'*60}\n")


def save_results(rows: list[dict], metrics: dict, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out / "dataset_results.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  CSV results → {csv_path}")

    # JSON summary
    json_path = out / "dataset_summary.json"
    json_path.write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=2))
    print(f"  JSON summary → {json_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Comparison chart between real vs fake
# ══════════════════════════════════════════════════════════════════════════════

def generate_comparison_charts(rows: list[dict], output_dir: str = "results/dataset"):
    """Generate matplotlib charts comparing real vs fake distributions."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    real_rows = [r for r in rows if r["label"] == "real"]
    fake_rows = [r for r in rows if r["label"] == "fake"]

    if not real_rows or not fake_rows:
        print("  ⚠ Need both real and fake results for comparison charts")
        return

    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor":   "#16213e",
        "axes.edgecolor":   "#0f3460",
        "axes.labelcolor":  "white",
        "text.color":       "white",
        "xtick.color":      "#aaaaaa",
        "ytick.color":      "#aaaaaa",
        "grid.color":       "#0f3460",
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
    })

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#1a1a2e")

    # Helper
    def _hist(ax, key, title, xlabel, bins=20):
        rv = [r[key] for r in real_rows if r[key] is not None]
        fv = [r[key] for r in fake_rows if r[key] is not None]
        ax.hist(rv, bins=bins, alpha=0.65, color="#2ecc71", label="Real", edgecolor="#16213e")
        ax.hist(fv, bins=bins, alpha=0.65, color="#e74c3c", label="Fake", edgecolor="#16213e")
        ax.set_title(title, color="white", fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend(framealpha=0.3)
        ax.grid(True)

    _hist(axes[0, 0], "bpm",       "Blinks Per Minute",       "BPM")
    _hist(axes[0, 1], "n_blinks",  "Total Blink Count",       "Blinks")
    _hist(axes[0, 2], "mean_dur_ms", "Mean Blink Duration",   "Duration (ms)")
    _hist(axes[1, 0], "cv_ibi",    "Blink Variability (CoV)", "CoV")
    _hist(axes[1, 1], "symmetry",  "Eye Symmetry Diff",       "|EAR_L − EAR_R|")
    _hist(axes[1, 2], "score",     "Suspicion Score",          "Score (%)")

    fig.suptitle(
        "Real vs Fake — Blink Behaviour Comparison",
        color="white", fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    chart_path = out / "real_vs_fake_comparison.png"
    plt.savefig(str(chart_path), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Comparison chart → {chart_path}")
    return str(chart_path)


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Eye-Blink detector on a deepfake dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full Celeb-DF v2 evaluation
  python dataset_utils.py --dataset D:\\archive

  # Quick test with 5 videos per class
  python dataset_utils.py --dataset D:\\archive --max-per-class 5

  # Skip individual report PNGs (faster)
  python dataset_utils.py --dataset D:\\archive --no-reports --max-per-class 20
        """,
    )
    parser.add_argument("--dataset",        required=True,  help="Root folder (e.g. D:\\archive)")
    parser.add_argument("--output-dir",     default="results/dataset", help="Where to save results")
    parser.add_argument("--max-per-class",  type=int, default=None,
                        help="Limit videos per class (useful for quick tests)")
    parser.add_argument("--no-reports",     action="store_true",
                        help="Skip generating individual PNG reports (faster)")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print("  EYE-BLINK DEEPFAKE DETECTION — DATASET EVALUATION")
    print(f"{'═'*60}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Output  : {args.output_dir}")
    if args.max_per_class:
        print(f"  Limit   : {args.max_per_class} videos per class")
    print()

    rows = process_dataset(
        dataset_root  = args.dataset,
        output_dir    = args.output_dir,
        max_per_class = args.max_per_class,
        save_reports  = not args.no_reports,
    )

    if rows:
        metrics = compute_metrics(rows)
        print_metrics(metrics)
        save_results(rows, metrics, args.output_dir)
        generate_comparison_charts(rows, args.output_dir)
    else:
        print("  No videos processed.")
