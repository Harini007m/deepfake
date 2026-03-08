"""Test with more videos from the dataset to understand the patterns."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import DeepfakeDetectionPipeline

videos = [
    (r"D:\archive\Celeb-real\id0_0000.mp4",       "real"),
    (r"D:\archive\Celeb-real\id0_0001.mp4",       "real"),
    (r"D:\archive\Celeb-real\id0_0002.mp4",       "real"),
    (r"D:\archive\YouTube-real\00000.mp4",        "real"),
    (r"D:\archive\YouTube-real\00001.mp4",        "real"),
    (r"D:\archive\Celeb-synthesis\id0_id16_0000.mp4", "fake"),
    (r"D:\archive\Celeb-synthesis\id0_id16_0001.mp4", "fake"),
    (r"D:\archive\Celeb-synthesis\id0_id16_0002.mp4", "fake"),
    (r"D:\archive\Celeb-synthesis\id0_id1_0000.mp4",  "fake"),
    (r"D:\archive\Celeb-synthesis\id0_id1_0001.mp4",  "fake"),
]

lines = []
lines.append(f"{'File':<30} {'Label':>5} {'Blinks':>7} {'BPM':>7} {'Dur_ms':>7} {'Sym':>7} {'Score':>7} {'Verdict':<18} {'OK?':>4}")
lines.append("-" * 110)

for path, label in videos:
    name = os.path.basename(path)
    if not os.path.exists(path):
        lines.append(f"{name:<30} MISSING")
        continue
    p = DeepfakeDetectionPipeline(path, output_dir="results/celeb_df")
    r = p.run()
    f = r["features"]
    a = r["analysis"]
    correct = (label == "real" and a["verdict"] == "REAL") or (label == "fake" and a["verdict"] != "REAL")
    lines.append(
        f"{name:<30} {label:>5} {f['n_blinks']:>7} {f['bpm']:>7.1f} "
        f"{f['mean_duration_s']*1000:>7.0f} {f['symmetry_diff']:>7.4f} "
        f"{a['total_score']:>6.1f}% {a['verdict']:<18} {'Y' if correct else 'N':>4}"
    )

with open("multi_results.txt", "w", encoding="utf-8") as fout:
    fout.write("\n".join(lines))
print("Results written to multi_results.txt")
