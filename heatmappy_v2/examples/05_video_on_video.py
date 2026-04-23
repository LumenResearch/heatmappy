"""
Example 05 — Heatmap video on an existing video

Renders heatmap overlays over SampleVideo_720x480_1mb.mp4.
FPS is read from the source; audio is preserved in all outputs.
Colour and reveal modes, each with all three decay variants:
  - 05a_colour_no_decay.mp4       colour | point visible only in its own frame
  - 05b_colour_hard_decay.mp4     colour | full intensity for 400 ms then vanishes
  - 05c_colour_smooth_decay.mp4   colour | linear fade to zero over 400 ms
  - 05d_reveal_no_decay.mp4       reveal | point visible only in its own frame
  - 05e_reveal_hard_decay.mp4     reveal | full intensity for 400 ms then vanishes
  - 05f_reveal_smooth_decay.mp4   reveal | linear fade to zero over 400 ms

All six videos use the same 100 gaze points per frame, Gaussian-distributed
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

colour_heatmapper = Heatmapper(
    point_diameter=100, point_strength=0.6, normalisation="relative", opacity=0.65
)
reveal_heatmapper = Heatmapper(
    point_diameter=100, point_strength=0.6, normalisation="relative", mode="reveal"
)

for prefix, hm in [
    ("05a_colour", colour_heatmapper),
    ("05d_reveal", reveal_heatmapper),
]:
    # no decay — each point visible only in its own frame
    VideoHeatmapper(hm).heatmap_on_video(
        VIDEO_PATH,
        gaze_points,
        os.path.join(EXAMPLES_DIR, f"{prefix}_no_decay.mp4"),
    )
    print(f"Saved: {prefix}_no_decay.mp4")

    # hard decay — full intensity for DECAY_MS then vanishes
    VideoHeatmapper(hm, decay_time_ms=DECAY_MS).heatmap_on_video(
        VIDEO_PATH,
        gaze_points,
        os.path.join(EXAMPLES_DIR, f"{prefix}_hard_decay.mp4"),
    )
    print(f"Saved: {prefix}_hard_decay.mp4")

    # smooth decay — linear fade from full intensity to zero over DECAY_MS
    VideoHeatmapper(hm, decay_time_ms=DECAY_MS, smooth_decay=True).heatmap_on_video(
        VIDEO_PATH,
        gaze_points,
        os.path.join(EXAMPLES_DIR, f"{prefix}_smooth_decay.mp4"),
    )
    print(f"Saved: {prefix}_smooth_decay.mp4")
