"""
cli.py
──────
Command-line interface for the Eye-Blink Deepfake Detection System.

Usage
-----
    python cli.py video.mp4 [options]

    python cli.py video.mp4 --show
    python cli.py video.mp4 --save-video
    python cli.py video.mp4 --output-dir my_results
    python cli.py --batch /path/to/video/folder
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from pipeline import DeepfakeDetectionPipeline


VERDICT_EMOJI = {
    "REAL":            "✅",
    "SUSPICIOUS":      "⚠️",
    "LIKELY DEEPFAKE": "🚨",
}


def print_results(results: dict, indent: bool = True):
    f = results["features"]
    a = results["analysis"]
    v = results["video_info"]
    i = "  " if indent else ""

    print(f"\n{'═'*60}")
    print(f"  EYE-BLINK DEEPFAKE DETECTION — RESULTS")
    print(f"{'═'*60}")
    print(f"{i}Video     : {v['source']}")
    print(f"{i}Duration  : {v['duration_s']} s  |  FPS: {v['fps']}")
    print()
    print(f"{i}{'─'*28}  BLINK METRICS")
    print(f"{i}Total Blinks        : {f['n_blinks']}")
    print(f"{i}Blinks / min        : {f['bpm']:.2f}")
    print(f"{i}Mean Blink Duration : {f['mean_duration_s']*1000:.0f} ms")
    print(f"{i}Mean Inter-Blink IBI: {f['mean_ibi_s']:.3f} s")
    print(f"{i}Blink Variability   : {f['cv_ibi']:.3f} (CoV)")
    print(f"{i}Eye Symmetry Diff   : {f['symmetry_diff']:.4f}")
    print()
    print(f"{i}{'─'*28}  ANALYSIS")
    for k, v_ in a["sub_scores"].items():
        bar = "█" * int(v_ * 20) + "░" * (20 - int(v_ * 20))
        print(f"{i}  {k:<12} [{bar}] {v_*100:5.1f}%")
    print()
    emoji = VERDICT_EMOJI.get(a["verdict"], "❓")
    print(f"{i}Suspicion Score : {a['total_score']:.1f} / 100")
    print(f"{i}Verdict         : {emoji}  {a['verdict']}")
    print()
    print(f"{i}{'─'*28}  EXPLANATION")
    for line in a["explanation"]:
        print(f"{i}  • {line}")
    print()
    print(f"{i}Report saved → {results['report_path']}")
    print(f"{i}Processed in {results['elapsed_s']} s")
    print(f"{'═'*60}\n")


def run_single(args) -> dict:
    pipeline = DeepfakeDetectionPipeline(
        video_path        = args.video,
        show_video        = args.show,
        save_output_video = args.save_video,
        output_dir        = args.output_dir,
    )
    results = pipeline.run()
    print_results(results)

    if args.json:
        j_path = Path(args.output_dir) / (Path(args.video).stem + "_results.json")
        j_path.parent.mkdir(parents=True, exist_ok=True)
        # Make serialisable
        out = {
            "video_info": results["video_info"],
            "features":   results["features"],
            "analysis": {
                "sub_scores": results["analysis"]["sub_scores"],
                "total_score": results["analysis"]["total_score"],
                "verdict": results["analysis"]["verdict"],
                "explanation": results["analysis"]["explanation"],
            },
            "elapsed_s": results["elapsed_s"],
        }
        j_path.write_text(json.dumps(out, indent=2))
        print(f"JSON results saved → {j_path}")

    return results


def run_batch(args):
    folder = Path(args.batch)
    exts   = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)

    if not videos:
        print("No video files found in", folder)
        sys.exit(1)

    print(f"Found {len(videos)} video(s) in {folder}\n")
    summary = []
    for vp in videos:
        print(f"→ {vp.name}")
        try:
            pipeline = DeepfakeDetectionPipeline(
                video_path        = str(vp),
                show_video        = False,
                save_output_video = args.save_video,
                output_dir        = args.output_dir,
            )
            r = pipeline.run()
            summary.append({
                "file":    vp.name,
                "bpm":     r["features"]["bpm"],
                "blinks":  r["features"]["n_blinks"],
                "score":   r["analysis"]["total_score"],
                "verdict": r["analysis"]["verdict"],
            })
            e = VERDICT_EMOJI.get(r["analysis"]["verdict"], "❓")
            print(f"   {e} {r['analysis']['verdict']}  (score {r['analysis']['total_score']:.1f}%)\n")
        except Exception as exc:
            print(f"   ❌ ERROR: {exc}\n")

    print("\n── BATCH SUMMARY ──")
    header = f"{'File':<35} {'BPM':>6} {'Blinks':>7} {'Score':>7}  Verdict"
    print(header)
    print("─" * len(header))
    for s in summary:
        e = VERDICT_EMOJI.get(s["verdict"], "❓")
        print(
            f"{s['file']:<35} {s['bpm']:>6.1f} {s['blinks']:>7} "
            f"{s['score']:>6.1f}%  {e} {s['verdict']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Eye-Blink Based Deepfake Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video",        nargs="?",              help="Path to video file")
    parser.add_argument("--show",       action="store_true",    help="Show annotated video during processing")
    parser.add_argument("--save-video", action="store_true",    help="Save annotated output video")
    parser.add_argument("--output-dir", default="results",      help="Directory for reports (default: results/)")
    parser.add_argument("--json",       action="store_true",    help="Also save JSON results")
    parser.add_argument("--batch",      metavar="FOLDER",       help="Batch-process all videos in FOLDER")

    args = parser.parse_args()

    if args.batch:
        run_batch(args)
    elif args.video:
        run_single(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
