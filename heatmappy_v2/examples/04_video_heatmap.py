"""
Example 04 — Video heatmap

Demonstrates VideoHeatmapper with two source types and all three decay modes:

  Part A — heatmap_on_image: static cat.jpg → 5-second video at 20 FPS
    - no_decay:     point visible only in its own frame
    - hard_decay:   point holds full intensity for 500 ms then vanishes
    - smooth_decay: point fades linearly to zero over 500 ms

  Part B — heatmap_on_video: SampleVideo_720x480_1mb.mp4 → heatmap overlay
    - smooth_decay over 400 ms; audio is preserved in the output

Run from the repo root:
    python heatmappy_v2/examples/04_video_heatmap.py
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
VIDEO_PATH = os.path.join(EXAMPLES_DIR, "SampleVideo_720x480_1mb.mp4")

# --------------------------------------------------------------------------- helpers


def random_video_points(
    n_users: int,
    points_per_user: int,
    width: int,
    height: int,
    duration_ms: float,
    spread: float = 60.0,
) -> list[VideoPoint]:
    """
    Generate clustered VideoPoints for n_users participants.
    Each user gets a random centre near the image centre, then
    Gaussian-distributed fixations around that centre spread over time.
    """
    cx, cy = width / 2, height / 2
    points: list[VideoPoint] = []
    for _ in range(n_users):
        ux = random.gauss(cx, spread)
        uy = random.gauss(cy, spread)
        for j in range(points_per_user):
            t = (j / points_per_user) * duration_ms
            points.append(
                VideoPoint(
                    x=random.gauss(ux, spread * 0.5),
                    y=random.gauss(uy, spread * 0.5),
                    t=t,
                )
            )
    return points


# --------------------------------------------------------------------------- Part A

img = cv2.imread(CAT_PATH)
if img is None:
    raise FileNotFoundError(f"Could not read: {CAT_PATH}")
H, W = img.shape[:2]

DURATION_MS = 5_000.0
FPS = 20.0

gaze_points = random_video_points(
    n_users=8,
    points_per_user=30,
    width=W,
    height=H,
    duration_ms=DURATION_MS,
)

heatmapper = Heatmapper(point_strength=0.6, normalisation="relative", opacity=0.65)

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

# smooth decay — linear fade from full to zero over 500 ms
vh_smooth = VideoHeatmapper(heatmapper, decay_time_ms=500.0, smooth_decay=True)
vh_smooth.heatmap_on_image(
    img,
    gaze_points,
    os.path.join(EXAMPLES_DIR, "04c_smooth_decay.mp4"),
    duration_ms=DURATION_MS,
    fps=FPS,
)
print("Saved: 04c_smooth_decay.mp4")

# --------------------------------------------------------------------------- Part B

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open: {VIDEO_PATH}")
vfps = cap.get(cv2.CAP_PROP_FPS)
vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vh_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

video_duration_ms = (n_frames / vfps) * 1000.0

video_gaze = random_video_points(
    n_users=6,
    points_per_user=40,
    width=vw,
    height=vh_px,
    duration_ms=video_duration_ms,
)

vh_on_video = VideoHeatmapper(heatmapper, decay_time_ms=400.0, smooth_decay=True)
vh_on_video.heatmap_on_video(
    VIDEO_PATH,
    video_gaze,
    os.path.join(EXAMPLES_DIR, "04d_on_video.mp4"),
)
print("Saved: 04d_on_video.mp4 (audio preserved)")
