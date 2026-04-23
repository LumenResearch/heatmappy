"""
Example 03 — Colour and reveal heatmaps

Demonstrates Heatmapper with colour and reveal modes across different
colormaps and opacity settings.

Run from the repo root:
    python heatmappy_v2/examples/03_colour_and_reveal.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import cv2
import numpy as np
from heatmappy2.heatmap import Heatmapper, Point

random.seed(42)

IMG_PATH = os.path.join(os.path.dirname(__file__), 'cat.jpg')
img = cv2.imread(IMG_PATH)
H, W = img.shape[:2]


def make_clustered_points(n, cx, cy, spread=60):
    return [Point(x=random.gauss(cx, spread), y=random.gauss(cy, spread)) for _ in range(n)]


points = (
    make_clustered_points(40, W * 0.35, H * 0.35, spread=50) +
    make_clustered_points(20, W * 0.65, H * 0.55, spread=40) +
    make_clustered_points(10, W * 0.50, H * 0.70, spread=30)
)


def labelled(bgr, label):
    out = bgr.copy()
    cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 220, 255), 1, cv2.LINE_AA)
    return out


# --- Row 1: colour mode with different colormaps ---
r1 = []
for cmap in ['jet', 'hot', 'inferno', 'turbo']:
    hm = Heatmapper(point_diameter=60, mode='colour', colormap=cmap, opacity=0.65)
    r1.append(labelled(hm.heatmap_on_img(points, img), f"colour | {cmap}"))

# --- Row 2: opacity comparison + reveal ---
hm_low  = Heatmapper(point_diameter=60, mode='colour', colormap='jet', opacity=0.35)
hm_mid  = Heatmapper(point_diameter=60, mode='colour', colormap='jet', opacity=0.65)
hm_high = Heatmapper(point_diameter=60, mode='colour', colormap='jet', opacity=0.90)
hm_rev  = Heatmapper(point_diameter=60, mode='reveal')

r2 = [
    labelled(hm_low.heatmap_on_img(points, img),  "colour | jet opacity=0.35"),
    labelled(hm_mid.heatmap_on_img(points, img),  "colour | jet opacity=0.65"),
    labelled(hm_high.heatmap_on_img(points, img), "colour | jet opacity=0.90"),
    labelled(hm_rev.heatmap_on_img(points, img),  "reveal"),
]

output = np.vstack([np.hstack(r1), np.hstack(r2)])

out_path = os.path.join(os.path.dirname(__file__), '03_colour_and_reveal.png')
cv2.imwrite(out_path, output)
print(f"Saved: {out_path}")

cv2.imshow("Colour and Reveal", output)
cv2.waitKey(0)
cv2.destroyAllWindows()