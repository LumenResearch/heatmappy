"""
Example 02 — Grey heatmap

Demonstrates GreyHeatmapper with different normalisation modes and min_intensity.
Shows the use case of singling out 3 people from a study of 150.

Run from the repo root:
    python heatmappy_v2/examples/02_grey_heatmap.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import cv2
import numpy as np
from heatmappy2.heatmap import GreyHeatmapper, Point

random.seed(42)

IMG_PATH = os.path.join(os.path.dirname(__file__), 'cat.jpg')
img = cv2.imread(IMG_PATH)
H, W = img.shape[:2]


def make_random_points(n, width, height):
    return [Point(x=random.randint(0, width), y=random.randint(0, height)) for _ in range(n)]


def make_clustered_points(n, cx, cy, spread=40):
    return [
        Point(x=random.gauss(cx, spread), y=random.gauss(cy, spread))
        for _ in range(n)
    ]


# 150 random points — the full study
all_150_points = make_random_points(150, W, H)

# 3 singled-out participants clustered in one area
three_points = make_clustered_points(3, cx=W // 3, cy=H // 2, spread=30)


def labelled(grey, label):
    bgr = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    cv2.putText(bgr, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 220, 255), 1, cv2.LINE_AA)
    return bgr


# --- Panel 1: 150 people, relative ---
hm = GreyHeatmapper(point_diameter=60, point_strength=0.5, normalisation='relative')
p1 = labelled(hm.heatmap(W, H, all_150_points), "150 people | relative")

# --- Panel 2: 3 people, relative (looks same brightness as 150) ---
hm = GreyHeatmapper(point_diameter=60, point_strength=0.5, normalisation='relative')
p2 = labelled(hm.heatmap(W, H, three_points), "3 people | relative (misleading)")

# --- Panel 3: 3 people, absolute ceiling=150 (honest brightness) ---
hm = GreyHeatmapper(point_diameter=60, point_strength=0.5,
                    normalisation='absolute', ceiling=150)
p3 = labelled(hm.heatmap(W, H, three_points), "3 people | absolute ceiling=150")

# --- Panel 4: 3 people, absolute ceiling=150 + min_intensity floor ---
hm = GreyHeatmapper(point_diameter=60, point_strength=0.5,
                    normalisation='absolute', ceiling=150, min_intensity=0.2)
p4 = labelled(hm.heatmap(W, H, three_points), "3 people | absolute + min_intensity=0.2")

# --- Panel 5: mixed point sizes and strengths ---
mixed_points = [
    Point(x=W * 0.2, y=H * 0.5, diameter=40,  strength=0.3),
    Point(x=W * 0.5, y=H * 0.5, diameter=80,  strength=0.7),
    Point(x=W * 0.8, y=H * 0.5, diameter=120, strength=1.0),
]
hm = GreyHeatmapper(normalisation='relative')
p5 = labelled(hm.heatmap(W, H, mixed_points), "mixed diameter & strength | relative")


row1 = np.hstack([p1, p2, p3])
row2_padded = np.hstack([p4, p5, np.zeros_like(p1)])
output = np.vstack([row1, row2_padded])

out_path = os.path.join(os.path.dirname(__file__), '02_grey_heatmap.png')
cv2.imwrite(out_path, output)
print(f"Saved: {out_path}")

cv2.imshow("Grey Heatmap Modes", output)
cv2.waitKey(0)
cv2.destroyAllWindows()