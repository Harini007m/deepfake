"""
gui.py
──────
Tkinter GUI for the Eye-Blink Deepfake Detection System.

Provides:
  • Drag-and-drop / file-browser video selection
  • Live progress bar during analysis
  • Results panel with colour-coded verdict
  • "Open Report" button to view the saved PNG
"""

from __future__ import annotations
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import subprocess
import sys
import os


class DeepfakeDetectorGUI(tk.Tk):
    """Main application window."""

    # ── Colour palette ────────────────────────────────────────────────────
    BG       = "#0d0d1a"
    SURFACE  = "#16213e"
    SURFACE2 = "#1a1a2e"
    ACCENT   = "#7f5af0"
    GREEN    = "#2cb67d"
    YELLOW   = "#f4c430"
    RED      = "#e53e3e"
    TEXT     = "#e8e8e8"
    SUBTEXT  = "#aaaaaa"

    def __init__(self):
        super().__init__()
        self.title("Eye-Blink Deepfake Detection System")
        self.configure(bg=self.BG)
        self.resizable(True, True)
        self.geometry("780x680")
        self.minsize(680, 560)

        self._video_path = tk.StringVar()
        self._results    = None

        self._build_ui()

    # ── UI Construction ───────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=self.ACCENT, height=5)
        header.pack(fill="x")

        title_frame = tk.Frame(self, bg=self.BG, pady=14)
        title_frame.pack(fill="x")
        tk.Label(
            title_frame,
            text="Eye-Blink Deepfake Detection System",
            font=("Segoe UI", 17, "bold"),
            fg="white", bg=self.BG,
        ).pack()
        tk.Label(
            title_frame,
            text="Behavioural biometrics • No GPU required • Geometric analysis",
            font=("Segoe UI", 9),
            fg=self.SUBTEXT, bg=self.BG,
        ).pack()

        # ── Video selection ──────────────────────────────────────────────
        sel_frame = tk.Frame(self, bg=self.SURFACE2, padx=20, pady=14)
        sel_frame.pack(fill="x", padx=20, pady=(8, 4))

        tk.Label(
            sel_frame, text="Video File", font=("Segoe UI", 10, "bold"),
            fg=self.TEXT, bg=self.SURFACE2,
        ).grid(row=0, column=0, sticky="w")

        entry = tk.Entry(
            sel_frame, textvariable=self._video_path,
            font=("Segoe UI", 10), bg=self.SURFACE, fg="white",
            insertbackground="white", relief="flat", bd=6,
        )
        entry.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        sel_frame.columnconfigure(0, weight=1)

        browse_btn = self._make_button(sel_frame, "Browse…", self._browse_video, self.ACCENT)
        browse_btn.grid(row=1, column=1)

        # ── Options ──────────────────────────────────────────────────────
        opt_frame = tk.Frame(self, bg=self.SURFACE2, padx=20, pady=10)
        opt_frame.pack(fill="x", padx=20, pady=4)

        self._var_show_video  = tk.BooleanVar(value=False)
        self._var_save_video  = tk.BooleanVar(value=False)
        self._var_save_json   = tk.BooleanVar(value=True)

        chk_style = {"bg": self.SURFACE2, "fg": self.TEXT,
                     "activebackground": self.SURFACE2,
                     "selectcolor": self.ACCENT, "font": ("Segoe UI", 9)}

        tk.Checkbutton(opt_frame, text="Show annotated video during analysis",
                       variable=self._var_show_video, **chk_style).grid(row=0, column=0, sticky="w")
        tk.Checkbutton(opt_frame, text="Save annotated output video",
                       variable=self._var_save_video, **chk_style).grid(row=0, column=1, sticky="w", padx=20)
        tk.Checkbutton(opt_frame, text="Export JSON report",
                       variable=self._var_save_json, **chk_style).grid(row=0, column=2, sticky="w")

        # ── Run button ────────────────────────────────────────────────────
        run_frame = tk.Frame(self, bg=self.BG, pady=8)
        run_frame.pack(fill="x", padx=20)
        self._run_btn = self._make_button(run_frame, "[>]  Analyse Video", self._start_analysis, self.GREEN)
        self._run_btn.pack(side="left")

        self._progress_var = tk.DoubleVar(value=0)
        self._progress_lbl = tk.Label(run_frame, text="", fg=self.SUBTEXT, bg=self.BG,
                                      font=("Segoe UI", 9))
        self._progress_lbl.pack(side="left", padx=14)

        self._pbar = ttk.Progressbar(
            self, variable=self._progress_var, maximum=100,
            mode="determinate", length=740,
        )
        self._pbar.pack(padx=20, pady=(0, 8), fill="x")
        self._style_progressbar()

        # ── Results area ─────────────────────────────────────────────────
        res_label = tk.Label(self, text="Results", font=("Segoe UI", 11, "bold"),
                             fg=self.TEXT, bg=self.BG, anchor="w")
        res_label.pack(fill="x", padx=24, pady=(4, 2))

        self._results_frame = tk.Frame(self, bg=self.SURFACE2, padx=20, pady=14)
        self._results_frame.pack(fill="both", expand=True, padx=20, pady=(0, 4))

        self._placeholder_lbl = tk.Label(
            self._results_frame,
            text="Upload a video and click 'Analyse Video' to begin.",
            font=("Segoe UI", 10, "italic"), fg=self.SUBTEXT, bg=self.SURFACE2,
        )
        self._placeholder_lbl.pack(pady=40)

        # ── Footer ────────────────────────────────────────────────────────
        footer = tk.Frame(self, bg=self.BG, pady=6)
        footer.pack(fill="x")
        tk.Label(
            footer,
            text="Powered by OpenCV · MediaPipe Face Mesh · NumPy  |  Eye-Blink Deepfake Detector",
            font=("Segoe UI", 8), fg=self.SUBTEXT, bg=self.BG,
        ).pack()

    # ─────────────────────────────────────────────────────────────────────
    def _style_progressbar(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "TProgressbar",
            troughcolor=self.SURFACE,
            background=self.ACCENT,
            bordercolor=self.SURFACE,
            lightcolor=self.ACCENT,
            darkcolor=self.ACCENT,
        )

    def _make_button(self, parent, text, command, color):
        btn = tk.Button(
            parent, text=text, command=command,
            font=("Segoe UI", 10, "bold"),
            bg=color, fg="white",
            activebackground=color, activeforeground="white",
            relief="flat", bd=0, padx=14, pady=7, cursor="hand2",
        )
        btn.bind("<Enter>", lambda e: btn.config(bg=self._lighten(color)))
        btn.bind("<Leave>", lambda e: btn.config(bg=color))
        return btn

    @staticmethod
    def _lighten(hex_color: str) -> str:
        """Return a slightly lighter version of a #rrggbb colour string."""
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r = min(255, r + 35)
        g = min(255, g + 35)
        b = min(255, b + 35)
        return f"#{r:02x}{g:02x}{b:02x}"

    # ── File selection ────────────────────────────────────────────────────
    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._video_path.set(path)

    # ── Analysis (runs in background thread) ─────────────────────────────
    def _start_analysis(self):
        path = self._video_path.get().strip()
        if not path:
            messagebox.showwarning("No File", "Please select a video file first.")
            return
        if not Path(path).exists():
            messagebox.showerror("File Not Found", f"Cannot find:\n{path}")
            return

        self._run_btn.config(state="disabled", text="Analysing...")
        self._clear_results()
        self._progress_var.set(0)
        self._progress_lbl.config(text="Starting…")

        thread = threading.Thread(target=self._run_pipeline, args=(path,), daemon=True)
        thread.start()

    def _run_pipeline(self, path: str):
        try:
            from pipeline import DeepfakeDetectionPipeline
            import json

            pipeline = DeepfakeDetectionPipeline(
                video_path        = path,
                show_video        = self._var_show_video.get(),
                save_output_video = self._var_save_video.get(),
                output_dir        = "results",
            )

            def _progress(current, total):
                pct = (current / total) * 100 if total else 0
                self.after(0, self._update_progress, pct, current, total)

            results = pipeline.run(progress_callback=_progress)

            # Optional JSON export
            if self._var_save_json.get():
                j_path = Path("results") / (Path(path).stem + "_results.json")
                j_path.parent.mkdir(exist_ok=True)
                out = {
                    "video_info": results["video_info"],
                    "features":   results["features"],
                    "analysis": {
                        "sub_scores":  results["analysis"]["sub_scores"],
                        "total_score": results["analysis"]["total_score"],
                        "verdict":     results["analysis"]["verdict"],
                        "explanation": results["analysis"]["explanation"],
                    },
                    "elapsed_s": results["elapsed_s"],
                }
                j_path.write_text(json.dumps(out, indent=2))

            self.after(0, self._show_results, results)

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self.after(0, self._show_error, str(exc), tb)

    def _update_progress(self, pct: float, current: int, total: int):
        self._progress_var.set(pct)
        self._progress_lbl.config(text=f"Frame {current} / {total}  ({pct:.0f}%)")

    # ── Result display ────────────────────────────────────────────────────
    def _clear_results(self):
        for w in self._results_frame.winfo_children():
            w.destroy()

    def _show_error(self, msg: str, tb: str):
        self._run_btn.config(state="normal", text="[>]  Analyse Video")
        self._progress_lbl.config(text="Error.")
        self._clear_results()
        tk.Label(
            self._results_frame, text=f"[ERROR]  {msg}",
            fg=self.RED, bg=self.SURFACE2, font=("Segoe UI", 10, "bold"),
            wraplength=700, justify="left",
        ).pack(anchor="w", pady=4)
        tk.Label(
            self._results_frame, text=tb,
            fg=self.SUBTEXT, bg=self.SURFACE2, font=("Courier New", 8),
            wraplength=720, justify="left",
        ).pack(anchor="w")

    def _show_results(self, results: dict):
        self._results = results
        self._run_btn.config(state="normal", text="[>]  Analyse Video")
        self._progress_var.set(100)
        self._progress_lbl.config(text=f"Done in {results['elapsed_s']} s")
        self._clear_results()

        f = results["features"]
        a = results["analysis"]

        verdict_map = {
            "REAL":            (self.GREEN,  "[OK]  REAL"),
            "SUSPICIOUS":      (self.YELLOW, "[!!]  SUSPICIOUS"),
            "LIKELY DEEPFAKE": (self.RED,    "[!!]  LIKELY DEEPFAKE"),
        }
        v_color, v_label = verdict_map.get(a["verdict"], (self.SUBTEXT, a["verdict"]))

        # Verdict banner
        banner = tk.Frame(self._results_frame, bg=v_color, padx=12, pady=8)
        banner.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        tk.Label(
            banner, text=v_label,
            font=("Segoe UI", 14, "bold"), fg="white", bg=v_color,
        ).pack(side="left")
        tk.Label(
            banner, text=f"Suspicion Score: {a['total_score']:.1f} / 100",
            font=("Segoe UI", 11), fg="white", bg=v_color,
        ).pack(side="right")

        self._results_frame.columnconfigure(0, weight=1)
        self._results_frame.columnconfigure(1, weight=1)

        # Left column – metrics
        left = tk.Frame(self._results_frame, bg=self.SURFACE2)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self._section(left, "Blink Metrics", [
            ("Total Blinks",        str(f["n_blinks"])),
            ("Blinks / min",        f"{f['bpm']:.2f}"),
            ("Mean Duration",       f"{f['mean_duration_s']*1000:.0f} ms"),
            ("Mean IBI",            f"{f['mean_ibi_s']:.3f} s"),
            ("Blink Variability",   f"{f['cv_ibi']:.3f}"),
            ("Eye Symmetry Diff",   f"{f['symmetry_diff']:.4f}"),
            ("Video Duration",      f"{f['video_duration_s']:.1f} s"),
        ])

        # Right column – sub scores
        right = tk.Frame(self._results_frame, bg=self.SURFACE2)
        right.grid(row=1, column=1, sticky="nsew")
        self._section(right, "Anomaly Sub-Scores", [
            (k, f"{v*100:.1f}%") for k, v in a["sub_scores"].items()
        ])

        # Explanation
        exp_frame = tk.Frame(self._results_frame, bg=self.SURFACE2)
        exp_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        tk.Label(exp_frame, text="Analysis:", font=("Segoe UI", 9, "bold"),
                 fg=self.ACCENT, bg=self.SURFACE2).pack(anchor="w")
        for line in a["explanation"]:
            tk.Label(exp_frame, text=f"  • {line}", font=("Segoe UI", 9),
                     fg=self.TEXT, bg=self.SURFACE2, wraplength=700, justify="left",
                     ).pack(anchor="w")

        # Buttons
        btn_row = tk.Frame(self._results_frame, bg=self.SURFACE2)
        btn_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        self._make_button(btn_row, "[Chart] Open Report PNG", self._open_report, self.ACCENT).pack(side="left", padx=4)
        self._make_button(btn_row, "[Reset] Analyse Another", self._reset, self.SURFACE).pack(side="left", padx=4)

    def _section(self, parent, title: str, rows: list[tuple[str, str]]):
        tk.Label(parent, text=title, font=("Segoe UI", 10, "bold"),
                 fg=self.ACCENT, bg=self.SURFACE2).pack(anchor="w", pady=(0, 4))
        for label, value in rows:
            row = tk.Frame(parent, bg=self.SURFACE2)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, font=("Segoe UI", 9),
                     fg=self.SUBTEXT, bg=self.SURFACE2, width=20, anchor="w").pack(side="left")
            tk.Label(row, text=value, font=("Segoe UI", 9, "bold"),
                     fg="white", bg=self.SURFACE2).pack(side="left")

    def _open_report(self):
        if not self._results:
            return
        path = self._results.get("report_path", "")
        if path and Path(path).exists():
            os.startfile(path) if sys.platform == "win32" else subprocess.Popen(["xdg-open", path])
        else:
            messagebox.showwarning("Not Found", f"Report not found:\n{path}")

    def _reset(self):
        self._video_path.set("")
        self._results = None
        self._progress_var.set(0)
        self._progress_lbl.config(text="")
        self._clear_results()
        self._placeholder_lbl = tk.Label(
            self._results_frame,
            text="Upload a video and click 'Analyse Video' to begin.",
            font=("Segoe UI", 10, "italic"), fg=self.SUBTEXT, bg=self.SURFACE2,
        )
        self._placeholder_lbl.pack(pady=40)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = DeepfakeDetectorGUI()
    app.mainloop()
