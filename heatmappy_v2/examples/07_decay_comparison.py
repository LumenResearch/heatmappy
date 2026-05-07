"""
Example 07 — Decay mode comparison

Demonstrates why heatmaps "contract" at the start of a video when decay is active.

Setup:
  10 gaze points, x evenly spread left→right, y at video centre.
  Points appear one per frame, starting at frame 2.
  Two batches are rendered — 20 % and 30 % diameter — each producing:

  07_d20_no_decay.mp4      20 % diameter | point visible only in its own frame
  07_d20_hard_decay.mp4    20 % diameter | full intensity for 10 frames then vanishes
  07_d20_smooth_decay.mp4  20 % diameter | linear fade to zero over 10 frames
  07_d20_comparison.mp4    20 % diameter | all three side-by-side with labels
  07_d30_no_decay.mp4      30 % diameter | point visible only in its own frame
  07_d30_hard_decay.mp4    30 % diameter | full intensity for 10 frames then vanishes
  07_d30_smooth_decay.mp4  30 % diameter | linear fade to zero over 10 frames
  07_d30_comparison.mp4    30 % diameter | all three side-by-side with labels

Run from the repo root:
    python heatmappy_v2/examples/07_decay_comparison.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from heatmappy2.heatmap import Heatmapper
from heatmappy2.video import VideoHeatmapper, VideoPoint

# ------------------------------------------------------------------ config

FPS = 20.0
FRAME_INTERVAL_MS = 1000.0 / FPS   # 50 ms per frame
N_POINTS = 10
DECAY_FRAMES = 10
DECAY_MS = DECAY_FRAMES * FRAME_INTERVAL_MS  # 500 ms

# Enough frames to show all points appear + full decay window + brief tail
N_FRAMES = N_POINTS + 2 + DECAY_FRAMES + 5   # = 27 frames
DURATION_MS = N_FRAMES * FRAME_INTERVAL_MS   # 1 350 ms

VIDEO_W, VIDEO_H = 640, 360

EXAMPLES_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(EXAMPLES_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------ base image

# Neutral dark-grey background so the heatmap colour is clearly visible.
base_img = np.full((VIDEO_H, VIDEO_W, 3), 30, dtype=np.uint8)

# ------------------------------------------------------------------ helpers

def make_gaze_points(diameter_pct: float) -> list[VideoPoint]:
    """10 points, x evenly distributed left→right, y at centre."""
    pts: list[VideoPoint] = []
    for i in range(N_POINTS):
        x = VIDEO_W * (i + 1) / (N_POINTS + 1)
        y = VIDEO_H / 2.0
        t_ms = (2 + i) * FRAME_INTERVAL_MS
        pts.append(VideoPoint(x=x, y=y, t=t_ms, diameter_pct=diameter_pct))
    return pts


def render_batch(prefix: str, diameter_pct: float) -> None:
    """Render the three decay variants and a comparison grid for one diameter."""
    gaze_points = make_gaze_points(diameter_pct)

    hm = Heatmapper(
        point_strength=0.8,
        normalisation="relative",
        opacity=0.85,
        colormap="jet",
    )

    variants = [
        (f"{prefix}_no_decay",     VideoHeatmapper(hm),                                  "no decay"),
        (f"{prefix}_hard_decay",   VideoHeatmapper(hm, decay_time_ms=DECAY_MS),          "hard decay (10 frames)"),
        (f"{prefix}_smooth_decay", VideoHeatmapper(hm, decay_time_ms=DECAY_MS,
                                                   smooth_decay=True),                   "smooth decay (10 frames)"),
    ]

    for stem, vh, _ in variants:
        out_path = os.path.join(OUTPUT_DIR, f"{stem}.mp4")
        vh.heatmap_on_image(base_img, gaze_points, out_path, duration_ms=DURATION_MS, fps=FPS)
        print(f"Saved: {stem}.mp4")

    # comparison grid (3 × 1)
    caps = [cv2.VideoCapture(os.path.join(OUTPUT_DIR, f"{s}.mp4")) for s, _, _ in variants]
    writer = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, f"{prefix}_comparison.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
        FPS,
        (VIDEO_W * 3, VIDEO_H),
    )
    try:
        while True:
            frames = []
            for cap, (_, _, label) in zip(caps, variants, strict=True):
                ret, frame = cap.read()
                if not ret:
                    frame = None
                else:
                    for thickness, colour in [(2, (0, 0, 0)), (1, (255, 255, 255))]:
                        cv2.putText(frame, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, colour, thickness, cv2.LINE_AA)
                frames.append(frame)
            if any(f is None for f in frames):
                break
            writer.write(np.hstack(frames))  # type: ignore[arg-type]
    finally:
        for cap in caps:
            cap.release()
        writer.release()
    print(f"Saved: {prefix}_comparison.mp4")


# ------------------------------------------------------------------ render both batches

render_batch("07_d20", diameter_pct=0.20)
render_batch("07_d30", diameter_pct=0.30)
render_batch("07_d50", diameter_pct=0.50)
