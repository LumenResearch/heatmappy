"""
Example 02 — Grey heatmap

Demonstrates GreyHeatmapper with different normalisation modes, min_intensity,
and the three ways to specify point size:
  - Bulk tuples with defaults (5% of min image dimension)
  - Per-point diameter_pct (percentage of min image dimension)
  - Per-point diameter (absolute pixels)

Run from the repo root:
    python heatmappy_v2/examples/02_grey_heatmap.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from heatmappy2.heatmap import GreyHeatmapper, Point, points_from_tuples

random.seed(42)

IMG_PATH = os.path.join(os.path.dirname(__file__), "cat.jpg")
img = cv2.imread(IMG_PATH)
H, W = img.shape[:2]


def make_random_tuples(n: int) -> list[tuple[float, float]]:
    return [(random.randint(0, W), random.randint(0, H)) for _ in range(n)]


def make_clustered_points(n: int, cx: float, cy: float, spread: float = 40.0) -> list[Point]:
    return [Point(x=random.gauss(cx, spread), y=random.gauss(cy, spread)) for _ in range(n)]


def labelled(grey: np.ndarray, label: str) -> np.ndarray:
    bgr = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    cv2.putText(bgr, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
    return bgr


hm = GreyHeatmapper(point_strength=0.5, normalisation="relative")

# --- Row 1: normalisation modes ---

# 150 random points via bulk tuple conversion — size defaults to 5% of min(W, H)
all_150_points = points_from_tuples(make_random_tuples(150))
p1 = labelled(hm.heatmap(W, H, all_150_points), f"150 people | bulk tuples | default 5% = {int(min(W,H)*0.05)}px")

# 3 singled-out participants — relative (misleading: looks as bright as 150)
three_points = make_clustered_points(3, cx=W // 3, cy=H // 2, spread=30)
p2 = labelled(hm.heatmap(W, H, three_points), "3 people | relative (misleading)")

# 3 people — absolute normalisation against full study size
hm_abs = GreyHeatmapper(point_strength=0.5, normalisation="absolute", ceiling=150)
p3 = labelled(hm_abs.heatmap(W, H, three_points), "3 people | absolute ceiling=150")

# 3 people — absolute + min_intensity floor so they stay visible
hm_floor = GreyHeatmapper(point_strength=0.5, normalisation="absolute", ceiling=150, min_intensity=0.2)
p4 = labelled(hm_floor.heatmap(W, H, three_points), "3 people | absolute + min_intensity=0.2")

# --- Row 2: point sizing modes side by side ---

# Bulk tuples — all get 5% default
bulk_points = points_from_tuples(make_random_tuples(40))
p5 = labelled(
    hm.heatmap(W, H, bulk_points),
    f"bulk default | 5% = {int(min(W, H) * 0.05)}px"
)

# diameter_pct — each point sized relative to image
pct_points = [
    Point(x=W * 0.2, y=H * 0.5, diameter_pct=0.03),
    Point(x=W * 0.5, y=H * 0.5, diameter_pct=0.08),
    Point(x=W * 0.8, y=H * 0.5, diameter_pct=0.15),
]
p6 = labelled(
    hm.heatmap(W, H, pct_points),
    f"diameter_pct | 3% / 8% / 15% of {min(W, H)}px"
)

# diameter — absolute pixels, image-size independent
px_points = [
    Point(x=W * 0.2, y=H * 0.5, diameter=20),
    Point(x=W * 0.5, y=H * 0.5, diameter=60),
    Point(x=W * 0.8, y=H * 0.5, diameter=120),
]
p7 = labelled(hm.heatmap(W, H, px_points), "diameter px | 20 / 60 / 120px")

# mixed — some absolute, some pct, some default
mixed_points = [
    Point(x=W * 0.15, y=H * 0.5),                        # default 5%
    Point(x=W * 0.4,  y=H * 0.5, diameter_pct=0.10),     # 10% pct
    Point(x=W * 0.65, y=H * 0.5, diameter=80),            # 80px absolute
    Point(x=W * 0.85, y=H * 0.5, diameter=30, strength=1.0),  # absolute + custom strength
]
p8 = labelled(hm.heatmap(W, H, mixed_points), "mixed | default / pct / px / px+strength")

row1 = np.hstack([p1, p2, p3, p4])
row2 = np.hstack([p5, p6, p7, p8])
output = np.vstack([row1, row2])

out_path = os.path.join(os.path.dirname(__file__), "02_grey_heatmap.png")
cv2.imwrite(out_path, output)
print(f"Saved: {out_path}")

cv2.imshow("Grey Heatmap Modes", output)
cv2.waitKey(0)
cv2.destroyAllWindows()