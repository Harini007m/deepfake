"""
dataset_utils.py
────────────────
Utilities for evaluating the detector against standard deepfake datasets
such as Celeb-DF v2 and UADFV.

Assumes dataset is arranged as:
    dataset_root/
        real/     <-- real face videos
        fake/     <-- deepfake videos

Usage
-----
    python dataset_utils.py --dataset /path/to/CelebDF --output-dir results/celebdf
"""

from __future__ import annotations
import argparse
import json
import csv
from pathlib import Path
from pipeline import DeepfakeDetectionPipeline


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def collect_videos(root: Path) -> dict[str, list[Path]]:
    """
    Collect real/fake video paths.

    Supports two layouts
        (a) root/real/*.mp4   root/fake/*.mp4
        (b) root/          (all videos, label inferred from filename)
    """
    real_dir = root / "real"
    fake_dir = root / "fake"

    if real_dir.exists() and fake_dir.exists():
        return {
            "real": sorted(p for p in real_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS),
            "fake": sorted(p for p in fake_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS),
        }
    # Fallback: label by keyword in filename
    all_vids = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    real = [p for p in all_vids if "real" in p.stem.lower()]
    fake = [p for p in all_vids if "fake" in p.stem.lower() or "df" in p.stem.lower()]
    return {"real": real, "fake": fake}


def process_dataset(
    dataset_root: str,
    output_dir:   str = "results/dataset",
    max_per_class: int | None = None,
) -> list[dict]:
    """
    Process all videos in a dataset and return per-video result rows.
    """
    root    = Path(dataset_root)
    videos  = collect_videos(root)
    rows    = []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, paths in videos.items():
        if max_per_class:
            paths = paths[:max_per_class]
        print(f"\n── {label.upper()} ({len(paths)} videos) ──")
        for vp in paths:
            print(f"  → {vp.name}", end=" ", flush=True)
            try:
                pipeline = DeepfakeDetectionPipeline(
                    video_path  = str(vp),
                    output_dir  = str(out_dir / label),
                )
                r = pipeline.run()
                a = r["analysis"]
                f = r["features"]
                row = {
                    "file":     vp.name,
                    "label":    label,
                    "bpm":      f["bpm"],
                    "blinks":   f["n_blinks"],
                    "duration": f["mean_duration_s"],
                    "cv_ibi":   f["cv_ibi"],
                    "symmetry": f["symmetry_diff"],
                    "score":    a["total_score"],
                    "verdict":  a["verdict"],
                    "correct":  (
                        (label == "real" and a["verdict"] == "REAL") or
                        (label == "fake" and a["verdict"] != "REAL")
                    ),
                }
                rows.append(row)
                print(f"[{a['verdict']}  score={a['total_score']:.1f}%]")
            except Exception as exc:
                print(f"[ERROR: {exc}]")

    return rows


def compute_metrics(rows: list[dict]) -> dict:
    """Compute accuracy, precision, recall, F1 for binary real/fake classification."""
    tp = sum(1 for r in rows if r["label"] == "fake" and r["verdict"] != "REAL")
    tn = sum(1 for r in rows if r["label"] == "real" and r["verdict"] == "REAL")
    fp = sum(1 for r in rows if r["label"] == "real" and r["verdict"] != "REAL")
    fn = sum(1 for r in rows if r["label"] == "fake" and r["verdict"] == "REAL")

    total    = len(rows)
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "total":     total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy":  round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall":    round(recall * 100, 2),
        "f1":        round(f1 * 100, 2),
    }


def save_results(rows: list[dict], metrics: dict, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out / "dataset_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # JSON summary
    json_path = out / "dataset_summary.json"
    json_path.write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=2))

    print(f"\nResults saved → {csv_path}")
    print(f"Summary  saved → {json_path}")


def print_metrics(metrics: dict):
    print("\n" + "═" * 44)
    print("  DATASET EVALUATION SUMMARY")
    print("═" * 44)
    print(f"  Total videos : {metrics['total']}")
    print(f"  TP / TN / FP / FN : {metrics['tp']} / {metrics['tn']} / {metrics['fp']} / {metrics['fn']}")
    print(f"  Accuracy   : {metrics['accuracy']:.1f}%")
    print(f"  Precision  : {metrics['precision']:.1f}%")
    print(f"  Recall     : {metrics['recall']:.1f}%")
    print(f"  F1 Score   : {metrics['f1']:.1f}%")
    print("═" * 44 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Eye-Blink detector on a deepfake dataset"
    )
    parser.add_argument("--dataset",     required=True, help="Root folder of the dataset")
    parser.add_argument("--output-dir",  default="results/dataset", help="Where to save results")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Limit videos per class (useful for quick tests)")
    args = parser.parse_args()

    rows    = process_dataset(args.dataset, args.output_dir, args.max_per_class)
    if rows:
        metrics = compute_metrics(rows)
        print_metrics(metrics)
        save_results(rows, metrics, args.output_dir)
    else:
        print("No videos processed.")
