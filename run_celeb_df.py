"""
run_celeb_df.py
───────────────
Quick-start script for running the Eye-Blink Deepfake Detector
on the locally downloaded Celeb-DF v2 dataset.

Dataset location: D:\\archive
    ├── Celeb-real/         → 590 real videos
    ├── YouTube-real/       → 300 real videos
    └── Celeb-synthesis/    → 5639 deepfake videos

Usage
-----
    # Quick test (5 videos per class)
    python run_celeb_df.py --quick

    # Medium run (50 per class)
    python run_celeb_df.py --max 50

    # Full dataset (all 6529 videos – takes a while!)
    python run_celeb_df.py --full

    # Analyse a single video
    python run_celeb_df.py --single D:\\archive\\Celeb-real\\id0_0000.mp4

    # Compare a real and a fake video side by side
    python run_celeb_df.py --compare \\
        --real-video D:\\archive\\Celeb-real\\id0_0000.mp4 \\
        --fake-video D:\\archive\\Celeb-synthesis\\id0_id16_0000.mp4
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from pipeline       import DeepfakeDetectionPipeline
from dataset_utils  import (
    collect_videos,
    process_dataset,
    compute_metrics,
    print_metrics,
    save_results,
    generate_comparison_charts,
)

# ── Default dataset path ──────────────────────────────────────────────────
DEFAULT_DATASET = r"D:\archive"
DEFAULT_OUTPUT  = "results/celeb_df"


# ══════════════════════════════════════════════════════════════════════════════
# Single-video analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_single(video_path: str, show: bool = False):
    """Run the full pipeline on one video and print results."""
    p = Path(video_path)
    if not p.exists():
        print(f"❌ File not found: {video_path}")
        sys.exit(1)

    # Infer label from the parent folder
    parent = p.parent.name.lower()
    if "synth" in parent:
        label = "fake"
    else:
        label = "real"

    print(f"\n{'═'*60}")
    print(f"  SINGLE VIDEO ANALYSIS")
    print(f"{'═'*60}")
    print(f"  File  : {p.name}")
    print(f"  Folder: {p.parent}")
    print(f"  Label : {label.upper()}")
    print()

    pipeline = DeepfakeDetectionPipeline(
        video_path        = str(p),
        show_video        = show,
        save_output_video = False,
        output_dir        = DEFAULT_OUTPUT,
    )
    results = pipeline.run()

    f = results["features"]
    a = results["analysis"]
    v = results["video_info"]

    print(f"  Duration  : {v['duration_s']} s  |  FPS: {v['fps']}")
    print(f"  Frames    : {v['total_frames']}")
    print()
    print(f"  {'─'*40}")
    print(f"  BLINK METRICS")
    print(f"  {'─'*40}")
    print(f"  Total Blinks        : {f['n_blinks']}")
    print(f"  Blinks / min        : {f['bpm']:.2f}")
    print(f"  Mean Blink Duration : {f['mean_duration_s']*1000:.0f} ms")
    print(f"  Std  Blink Duration : {f['std_duration_s']*1000:.0f} ms")
    print(f"  Mean Inter-Blink    : {f['mean_ibi_s']:.3f} s")
    print(f"  Blink Variability   : {f['cv_ibi']:.3f} (CoV)")
    print(f"  Eye Symmetry Diff   : {f['symmetry_diff']:.4f}")
    print(f"  Mean L-EAR          : {f['mean_left_ear']:.4f}")
    print(f"  Mean R-EAR          : {f['mean_right_ear']:.4f}")
    print()
    print(f"  {'─'*40}")
    print(f"  ANALYSIS")
    print(f"  {'─'*40}")
    for k, sv in a["sub_scores"].items():
        bar = "█" * int(sv * 20) + "░" * (20 - int(sv * 20))
        print(f"    {k:<14} [{bar}] {sv*100:5.1f}%")
    print()

    emoji = {"REAL": "✅", "SUSPICIOUS": "⚠️", "LIKELY DEEPFAKE": "🚨"}.get(a["verdict"], "❓")
    correct = (
        (label == "real" and a["verdict"] == "REAL") or
        (label == "fake" and a["verdict"] != "REAL")
    )
    print(f"  Suspicion Score : {a['total_score']:.1f} / 100")
    print(f"  Verdict         : {emoji}  {a['verdict']}")
    print(f"  Ground Truth    : {label.upper()}")
    print(f"  Correct?        : {'✓ Yes' if correct else '✗ No'}")
    print()
    for line in a["explanation"]:
        print(f"    • {line}")
    print()
    print(f"  Report → {results['report_path']}")
    print(f"  Processed in {results['elapsed_s']} s")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Two-video comparison
# ══════════════════════════════════════════════════════════════════════════════

def run_compare(real_path: str, fake_path: str):
    """Analyse one real and one fake video, print side-by-side comparison."""
    print(f"\n{'═'*60}")
    print(f"  REAL vs FAKE — SIDE-BY-SIDE COMPARISON")
    print(f"{'═'*60}")

    results = {}
    for label, path in [("REAL", real_path), ("FAKE", fake_path)]:
        p = Path(path)
        if not p.exists():
            print(f"  ❌ File not found: {path}")
            sys.exit(1)
        print(f"\n  → Analysing {label}: {p.name}")
        pipeline = DeepfakeDetectionPipeline(str(p), output_dir=DEFAULT_OUTPUT)
        results[label] = pipeline.run()

    # Print comparison table
    print(f"\n  {'Metric':<24} {'REAL':>12} {'FAKE':>12}")
    print(f"  {'─'*48}")

    fr = results["REAL"]["features"]
    ff = results["FAKE"]["features"]
    ar = results["REAL"]["analysis"]
    af = results["FAKE"]["analysis"]

    rows = [
        ("Total Blinks",       f"{fr['n_blinks']}",                f"{ff['n_blinks']}"),
        ("Blinks / min",       f"{fr['bpm']:.2f}",                 f"{ff['bpm']:.2f}"),
        ("Mean Duration (ms)", f"{fr['mean_duration_s']*1000:.0f}", f"{ff['mean_duration_s']*1000:.0f}"),
        ("Mean IBI (s)",       f"{fr['mean_ibi_s']:.3f}",          f"{ff['mean_ibi_s']:.3f}"),
        ("Variability (CoV)",  f"{fr['cv_ibi']:.3f}",              f"{ff['cv_ibi']:.3f}"),
        ("Symmetry Diff",      f"{fr['symmetry_diff']:.4f}",       f"{ff['symmetry_diff']:.4f}"),
        ("Suspicion Score",    f"{ar['total_score']:.1f}%",        f"{af['total_score']:.1f}%"),
        ("Verdict",            ar["verdict"],                       af["verdict"]),
    ]
    for name, rv, fv in rows:
        print(f"  {name:<24} {rv:>12} {fv:>12}")

    print(f"\n  Real report → {results['REAL']['report_path']}")
    print(f"  Fake report → {results['FAKE']['report_path']}")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Eye-Blink Deepfake Detection — Celeb-DF v2 Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_celeb_df.py --quick                     # 5 per class
  python run_celeb_df.py --max 50                    # 50 per class
  python run_celeb_df.py --full                      # all 6529 videos
  python run_celeb_df.py --single D:\\archive\\Celeb-real\\id0_0000.mp4
  python run_celeb_df.py --compare --real-video ... --fake-video ...
        """,
    )
    parser.add_argument("--dataset",     default=DEFAULT_DATASET, help="Dataset root (default: D:\\archive)")
    parser.add_argument("--output-dir",  default=DEFAULT_OUTPUT,  help="Output directory")
    parser.add_argument("--quick",       action="store_true",     help="Quick test: 5 videos per class")
    parser.add_argument("--max",         type=int, default=None,  help="Max videos per class")
    parser.add_argument("--full",        action="store_true",     help="Process all videos")
    parser.add_argument("--single",      metavar="VIDEO",         help="Analyse a single video")
    parser.add_argument("--show",        action="store_true",     help="Show live annotated video (single mode)")
    parser.add_argument("--compare",     action="store_true",     help="Compare one real vs one fake")
    parser.add_argument("--real-video",  metavar="PATH",          help="Real video for --compare")
    parser.add_argument("--fake-video",  metavar="PATH",          help="Fake video for --compare")
    parser.add_argument("--no-reports",  action="store_true",     help="Skip individual PNG reports (faster)")

    args = parser.parse_args()

    # ── Mode: single video ────────────────────────────────────────────────
    if args.single:
        run_single(args.single, show=args.show)
        return

    # ── Mode: compare ─────────────────────────────────────────────────────
    if args.compare:
        if not args.real_video or not args.fake_video:
            print("  ❌ --compare requires --real-video and --fake-video")
            sys.exit(1)
        run_compare(args.real_video, args.fake_video)
        return

    # ── Mode: dataset evaluation ──────────────────────────────────────────
    max_per_class = args.max
    if args.quick:
        max_per_class = 5
    elif args.full:
        max_per_class = None
    elif max_per_class is None:
        max_per_class = 10   # sensible default

    print(f"\n{'═'*60}")
    print("  EYE-BLINK DEEPFAKE DETECTION — CELEB-DF v2")
    print(f"{'═'*60}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Output  : {args.output_dir}")
    if max_per_class:
        print(f"  Limit   : {max_per_class} videos per class")
    else:
        print(f"  Mode    : FULL (all videos)")
    print()

    rows = process_dataset(
        dataset_root  = args.dataset,
        output_dir    = args.output_dir,
        max_per_class = max_per_class,
        save_reports  = not args.no_reports,
    )

    if rows:
        metrics = compute_metrics(rows)
        print_metrics(metrics)
        save_results(rows, metrics, args.output_dir)
        generate_comparison_charts(rows, args.output_dir)
        print(f"\n  ✅ All outputs saved to {args.output_dir}/")
    else:
        print("  No videos processed.")


if __name__ == "__main__":
    main()
