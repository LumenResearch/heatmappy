"""
Example 05 — Heatmap video on an existing video

Renders heatmap overlays over SampleVideo_720x480_1mb.mp4.
FPS is read from the source; audio is preserved in all outputs.
Colour, reveal, and pair (colour+reveal side-by-side) modes, each with all three decay variants:
  - 05a_colour_no_decay.mp4       colour | point visible only in its own frame
  - 05a_colour_hard_decay.mp4     colour | full intensity for 400 ms then vanishes
  - 05a_colour_smooth_decay.mp4   colour | linear fade to zero over 400 ms
  - 05d_reveal_no_decay.mp4       reveal | point visible only in its own frame
  - 05d_reveal_hard_decay.mp4     reveal | full intensity for 400 ms then vanishes
  - 05d_reveal_smooth_decay.mp4   reveal | linear fade to zero over 400 ms
  - 05g_pair_no_decay.mp4         pair   | point visible only in its own frame
  - 05g_pair_hard_decay.mp4       pair   | full intensity for 400 ms then vanishes
  - 05g_pair_smooth_decay.mp4     pair   | linear fade to zero over 400 ms

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
import numpy as np

from heatmappy2.heatmap import Heatmapper
from heatmappy2.video import VideoHeatmapper, VideoPoint

random.seed(7)

EXAMPLES_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(EXAMPLES_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
pair_heatmapper = Heatmapper(
    point_diameter=100, point_strength=0.6, normalisation="relative", opacity=0.65, mode="pair"
)

for prefix, hm in [
    ("05a_colour", colour_heatmapper),
    ("05d_reveal", reveal_heatmapper),
    ("05g_pair", pair_heatmapper),
]:
    # no decay — each point visible only in its own frame
    VideoHeatmapper(hm).heatmap_on_video(
        VIDEO_PATH,
        gaze_points,
        os.path.join(OUTPUT_DIR, f"{prefix}_no_decay.mp4"),
    )
    print(f"Saved: {prefix}_no_decay.mp4")

    # hard decay — full intensity for DECAY_MS then vanishes
    VideoHeatmapper(hm, decay_time_ms=DECAY_MS).heatmap_on_video(
        VIDEO_PATH,
        gaze_points,
        os.path.join(OUTPUT_DIR, f"{prefix}_hard_decay.mp4"),
    )
    print(f"Saved: {prefix}_hard_decay.mp4")

    # smooth decay — linear fade from full intensity to zero over DECAY_MS
    VideoHeatmapper(hm, decay_time_ms=DECAY_MS, smooth_decay=True).heatmap_on_video(
        VIDEO_PATH,
        gaze_points,
        os.path.join(OUTPUT_DIR, f"{prefix}_smooth_decay.mp4"),
    )
    print(f"Saved: {prefix}_smooth_decay.mp4")

# --------------------------------------------------------------------------- windowed clip
# Render only the middle third of the video (start_ms / end_ms).
# Smooth-decay gaze points that started just before the window are still
# visible at the beginning of the clip because _active_points uses absolute
# timestamps — no special handling required.

clip_start_ms = duration_ms / 3
clip_end_ms = 2 * duration_ms / 3

VideoHeatmapper(colour_heatmapper, decay_time_ms=DECAY_MS, smooth_decay=True).heatmap_on_video(
    VIDEO_PATH,
    gaze_points,
    os.path.join(OUTPUT_DIR, "05_windowed_clip.mp4"),
    start_ms=clip_start_ms,
    end_ms=clip_end_ms,
)
print(f"Saved: 05_windowed_clip.mp4  ({clip_start_ms:.0f}–{clip_end_ms:.0f} ms)")

# --------------------------------------------------------------------------- grid

GRID_CELLS = [
    # (path, col_label, row_label)
    ("05a_colour_no_decay.mp4", "no decay", "colour"),
    ("05a_colour_hard_decay.mp4", "hard decay", "colour"),
    ("05a_colour_smooth_decay.mp4", "smooth decay", "colour"),
    ("05d_reveal_no_decay.mp4", "no decay", "reveal"),
    ("05d_reveal_hard_decay.mp4", "hard decay", "reveal"),
    ("05d_reveal_smooth_decay.mp4", "smooth decay", "reveal"),
]
GRID_COLS, GRID_ROWS = 3, 2


def stitch_grid(
    cells: list[tuple[str, str, str]],
    output_path: str,
    n_cols: int,
    n_rows: int,
) -> None:
    caps = [cv2.VideoCapture(os.path.join(OUTPUT_DIR, p)) for p, _, _ in cells]
    fps_out = caps[0].get(cv2.CAP_PROP_FPS)
    cell_w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    cell_h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
        fps_out,
        (cell_w * n_cols, cell_h * n_rows),
    )
    try:
        while True:
            row_frames = []
            done = False
            for cap, (_, col_label, row_label) in zip(caps, cells, strict=True):
                ret, frame = cap.read()
                if not ret:
                    done = True
                    break
                label = f"{row_label} | {col_label}"
                cv2.putText(
                    frame,
                    label,
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    label,
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                row_frames.append(frame)
            if done:
                break
            rows = [np.hstack(row_frames[r * n_cols : (r + 1) * n_cols]) for r in range(n_rows)]
            writer.write(np.vstack(rows))
    finally:
        for cap in caps:
            cap.release()
        writer.release()


stitch_grid(
    GRID_CELLS,
    os.path.join(OUTPUT_DIR, "05_grid.mp4"),
    n_cols=GRID_COLS,
    n_rows=GRID_ROWS,
)
print("Saved: 05_grid.mp4")
