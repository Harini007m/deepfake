"""
blink_detector.py
─────────────────
Module 5 – Blink Event Detection

A blink is registered when the averaged EAR drops below EAR_THRESHOLD for
at least BLINK_CONSEC_FRAMES consecutive frames and then rises back above it.
Each blink event records:
    - frame_start / frame_end
    - time_start  / time_end  (seconds)
    - duration    (seconds)
    - left_ear    / right_ear at peak closure
"""

from __future__ import annotations
from dataclasses import dataclass, field
from config import EAR_THRESHOLD, BLINK_CONSEC_FRAMES


@dataclass
class BlinkEvent:
    frame_start:   int
    frame_end:     int
    time_start:    float
    time_end:      float
    left_ear_min:  float
    right_ear_min: float

    @property
    def duration(self) -> float:
        return self.time_end - self.time_start

    @property
    def avg_ear_min(self) -> float:
        return (self.left_ear_min + self.right_ear_min) / 2.0


class BlinkDetector:
    """
    Stateful detector: call `update()` for every frame;
    completed blink events are appended to `blinks`.
    """

    def __init__(
        self,
        ear_threshold: float = EAR_THRESHOLD,
        consec_frames: int   = BLINK_CONSEC_FRAMES,
        fps: float           = 25.0,
    ):
        self.ear_threshold  = ear_threshold
        self.consec_frames  = consec_frames
        self.fps            = fps

        self.blinks: list[BlinkEvent] = []

        # Internal state
        self._consec_count: int   = 0
        self._in_blink:     bool  = False
        self._blink_start_frame: int   = 0
        self._blink_start_time:  float = 0.0
        self._left_ear_min:  float = 1.0
        self._right_ear_min: float = 1.0

    def update(
        self,
        frame_idx:  int,
        left_ear:   float,
        right_ear:  float,
    ) -> BlinkEvent | None:
        """
        Feed one frame's EAR values.

        Returns
        -------
        BlinkEvent if a blink just *completed* this frame, else None.
        """
        avg_ear    = (left_ear + right_ear) / 2.0
        time_sec   = frame_idx / self.fps
        completed  = None

        if avg_ear < self.ear_threshold:
            self._consec_count += 1
            if not self._in_blink and self._consec_count >= self.consec_frames:
                # Blink onset
                self._in_blink        = True
                self._blink_start_frame = frame_idx - self._consec_count + 1
                self._blink_start_time  = self._blink_start_frame / self.fps
                self._left_ear_min    = left_ear
                self._right_ear_min   = right_ear
            elif self._in_blink:
                # Track minimum EAR during blink
                self._left_ear_min  = min(self._left_ear_min,  left_ear)
                self._right_ear_min = min(self._right_ear_min, right_ear)
        else:
            if self._in_blink:
                # Blink completed
                event = BlinkEvent(
                    frame_start   = self._blink_start_frame,
                    frame_end     = frame_idx,
                    time_start    = self._blink_start_time,
                    time_end      = time_sec,
                    left_ear_min  = self._left_ear_min,
                    right_ear_min = self._right_ear_min,
                )
                self.blinks.append(event)
                completed = event
            # Reset state
            self._consec_count  = 0
            self._in_blink      = False
            self._left_ear_min  = 1.0
            self._right_ear_min = 1.0

        return completed

    def total_blinks(self) -> int:
        return len(self.blinks)

    def reset(self):
        self.blinks.clear()
        self._consec_count      = 0
        self._in_blink          = False
        self._left_ear_min      = 1.0
        self._right_ear_min     = 1.0
