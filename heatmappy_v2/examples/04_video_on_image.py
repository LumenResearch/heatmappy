"""
Example 04 — Heatmap video on a static image

Renders a 5-second heatmap video over cat.jpg at 20 FPS using all three decay modes:
  - 04a_no_decay.mp4      point visible only in its own frame
  - 04b_hard_decay.mp4    full intensity for 500 ms then vanishes
  - 04c_smooth_decay.mp4  linear fade from full intensity to zero over 500 ms

All three videos use the same 100 gaze points per frame, Gaussian-distributed
around the image centre (spread = min(w,h) / 5).

Run from the repo root:
    python heatmappy_v2/examples/04_video_on_image.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from heatmappy2.heatmap import Heatmapper
from heatmappy2.video import VideoHeatmapper, VideoPoint

random.seed(7)

EXAMPLES_DIR = os.path.dirname(__file__)
CAT_PATH = os.path.join(EXAMPLES_DIR, "cat.jpg")

DURATION_MS = 5_000.0
FPS = 20.0


def random_video_points(
    width: int,
    height: int,
    duration_ms: float,
    fps: float,
    points_per_frame: int = 100,
    spread: float | None = None,
) -> list[VideoPoint]:
    """
    Gaussian-distributed gaze points centred on the image.
    Produces exactly points_per_frame points for every frame timestamp.
    spread defaults to min(width, height) / 5.
    """
    if spread is None:
        spread = min(width, height) / 5.0
    cx, cy = width / 2.0, height / 2.0
    frame_interval_ms = 1000.0 / fps
    n_frames = int(duration_ms / frame_interval_ms)
    points: list[VideoPoint] = []
    for i in range(n_frames):
        t = i * frame_interval_ms
        for _ in range(points_per_frame):
            points.append(
                VideoPoint(
                    x=random.gauss(cx, spread),
                    y=random.gauss(cy, spread),
                    t=t,
                )
            )
    return points


img = cv2.imread(CAT_PATH)
if img is None:
    raise FileNotFoundError(f"Could not read: {CAT_PATH}")
H, W = img.shape[:2]

gaze_points = random_video_points(width=W, height=H, duration_ms=DURATION_MS, fps=FPS)

heatmapper = Heatmapper(
    point_diameter=100, point_strength=0.6, normalisation="relative", opacity=0.65
)

# no decay — each point visible only in its own frame
vh_no_decay = VideoHeatmapper(heatmapper)
vh_no_decay.heatmap_on_image(
    img,
    gaze_points,
    os.path.join(EXAMPLES_DIR, "04a_no_decay.mp4"),
    duration_ms=DURATION_MS,
    fps=FPS,
)
print("Saved: 04a_no_decay.mp4")

# hard decay — full intensity for 500 ms then vanishes
vh_hard = VideoHeatmapper(heatmapper, decay_time_ms=500.0)
vh_hard.heatmap_on_image(
    img,
    gaze_points,
    os.path.join(EXAMPLES_DIR, "04b_hard_decay.mp4"),
    duration_ms=DURATION_MS,
    fps=FPS,
)
print("Saved: 04b_hard_decay.mp4")

# smooth decay — linear fade from full intensity to zero over 500 ms
vh_smooth = VideoHeatmapper(heatmapper, decay_time_ms=500.0, smooth_decay=True)
vh_smooth.heatmap_on_image(
    img,
    gaze_points,
    os.path.join(EXAMPLES_DIR, "04c_smooth_decay.mp4"),
    duration_ms=DURATION_MS,
    fps=FPS,
)
print("Saved: 04c_smooth_decay.mp4")
