"""
Example 05 — Heatmap video on an existing video

Renders heatmap overlays over SampleVideo_720x480_1mb.mp4 using all three decay modes.
FPS is read from the source; audio is preserved in all three outputs:
  - 05a_no_decay.mp4      point visible only in its own frame
  - 05b_hard_decay.mp4    full intensity for 400 ms then vanishes
  - 05c_smooth_decay.mp4  linear fade from full intensity to zero over 400 ms

All three videos use the same 100 gaze points per frame, Gaussian-distributed
around the video centre (spread = min(w,h) / 5).

Run from the repo root:
    python heatmappy_v2/examples/05_video_on_video.py
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
VIDEO_PATH = os.path.join(EXAMPLES_DIR, "SampleVideo_720x480_1mb.mp4")

DECAY_MS = 400.0


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


cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open: {VIDEO_PATH}")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

duration_ms = (n_frames / fps) * 1000.0

gaze_points = random_video_points(width=w, height=h, duration_ms=duration_ms, fps=fps)

heatmapper = Heatmapper(
    point_diameter=100, point_strength=0.6, normalisation="relative", opacity=0.65
)

# no decay — each point visible only in its own frame
vh_no_decay = VideoHeatmapper(heatmapper)
vh_no_decay.heatmap_on_video(
    VIDEO_PATH,
    gaze_points,
    os.path.join(EXAMPLES_DIR, "05a_no_decay.mp4"),
)
print("Saved: 05a_no_decay.mp4")

# hard decay — full intensity for 400 ms then vanishes
vh_hard = VideoHeatmapper(heatmapper, decay_time_ms=DECAY_MS)
vh_hard.heatmap_on_video(
    VIDEO_PATH,
    gaze_points,
    os.path.join(EXAMPLES_DIR, "05b_hard_decay.mp4"),
)
print("Saved: 05b_hard_decay.mp4")

# smooth decay — linear fade from full intensity to zero over 400 ms
vh_smooth = VideoHeatmapper(heatmapper, decay_time_ms=DECAY_MS, smooth_decay=True)
vh_smooth.heatmap_on_video(
    VIDEO_PATH,
    gaze_points,
    os.path.join(EXAMPLES_DIR, "05c_smooth_decay.mp4"),
)
print("Saved: 05c_smooth_decay.mp4")
