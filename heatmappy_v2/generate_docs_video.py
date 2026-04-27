"""
Generates a short low-resolution grid video for the README and saves to docs/.
Shows colour + reveal modes × 3 decay types in a 2×3 grid.
Run from the repo root:
    python heatmappy_v2/generate_docs_video.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np

from heatmappy2.heatmap import Heatmapper, Point
from heatmappy2.video import VideoHeatmapper, VideoPoint

SCRIPT_DIR = os.path.dirname(__file__)
DOCS_DIR = os.path.join(SCRIPT_DIR, "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

IMG_PATH = os.path.join(SCRIPT_DIR, "examples", "cat.jpg")
raw = cv2.imread(IMG_PATH)

# Downscale base image so the 2×3 grid is compact
SCALE = 0.45
base = cv2.resize(raw, (0, 0), fx=SCALE, fy=SCALE)
H, W = base.shape[:2]

# ------------------------------------------------------------------ gaze points

rng = np.random.default_rng(42)
DURATION_MS = 6_000.0
FPS = 10.0
INTERVAL = 1000.0 / FPS
N_FRAMES = int(DURATION_MS / INTERVAL)
N_PER_FRAME = 80

gaze: list[VideoPoint] = []
for i in range(N_FRAMES):
    t = i * INTERVAL
    xs = rng.normal(W * 0.45, W * 0.13, N_PER_FRAME).clip(0, W - 1)
    ys = rng.normal(H * 0.38, H * 0.13, N_PER_FRAME).clip(0, H - 1)
    for x, y in zip(xs, ys, strict=True):
        gaze.append(VideoPoint(x=float(x), y=float(y), t=t))

# ------------------------------------------------------------------ render each panel to memory

DECAY_MS = 600.0
DIAM = 40

configs = [
    ("no decay",     Heatmapper(point_diameter=DIAM, mode="colour"), dict()),
    ("hard decay",   Heatmapper(point_diameter=DIAM, mode="colour"), dict(decay_time_ms=DECAY_MS)),
    ("smooth decay", Heatmapper(point_diameter=DIAM, mode="colour"), dict(decay_time_ms=DECAY_MS, smooth_decay=True)),
    ("no decay",     Heatmapper(point_diameter=DIAM, mode="reveal"), dict()),
    ("hard decay",   Heatmapper(point_diameter=DIAM, mode="reveal"), dict(decay_time_ms=DECAY_MS)),
    ("smooth decay", Heatmapper(point_diameter=DIAM, mode="reveal"), dict(decay_time_ms=DECAY_MS, smooth_decay=True)),
]

# Pre-render all frames for each panel
print(f"Rendering {len(configs)} panels × {N_FRAMES} frames ...")
panel_frames: list[list[np.ndarray]] = []
for label, hm, kw in configs:
    vh = VideoHeatmapper(hm, **kw)
    frames = []
    frame_interval_ms = INTERVAL
    snapped = vh._snap_points(gaze, frame_interval_ms)
    for i in range(N_FRAMES):
        t_ms = i * frame_interval_ms
        active = vh._active_points(t_ms, snapped, frame_interval_ms)
        frame = hm.heatmap_on_img(active, base)
        # Add label
        cv2.putText(frame, f"{hm.mode} | {label}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
        frames.append(frame)
    panel_frames.append(frames)
    print(f"  done: {hm.mode} | {label}")

# ------------------------------------------------------------------ stitch 2×3 grid and write

out_path = os.path.join(DOCS_DIR, "heatmap_grid.mp4")
cols, rows = 3, 2
grid_w = W * cols
grid_h = H * rows
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, FPS, (grid_w, grid_h))

for i in range(N_FRAMES):
    top = np.hstack([panel_frames[0][i], panel_frames[1][i], panel_frames[2][i]])
    bot = np.hstack([panel_frames[3][i], panel_frames[4][i], panel_frames[5][i]])
    writer.write(np.vstack([top, bot]))

writer.release()
print(f"Saved: {out_path}  ({grid_w}×{grid_h} @ {FPS}fps, {DURATION_MS/1000:.0f}s)")
