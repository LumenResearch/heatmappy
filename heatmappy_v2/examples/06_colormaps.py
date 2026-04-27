"""
Example 06 — Colormap comparison

Renders the same set of points with every available colormap and saves a
side-by-side grid so you can pick the right look for your use case.

Built-in colormaps (OpenCV): jet, hot, inferno, plasma, viridis, turbo, bone
Custom colormaps (from strip PNGs): classic (v1 "default" — red → green → blue)
Modes: reveal is shown as a dedicated panel (it is a mode, not a colormap)

Run from the repo root:
    python heatmappy_v2/examples/06_colormaps.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from heatmappy2.heatmap import ALL_COLORMAPS, Heatmapper, Point

EXAMPLES_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(EXAMPLES_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------ base image

base = cv2.imread(os.path.join(EXAMPLES_DIR, "cat.jpg"))
if base is None:
    raise FileNotFoundError("cat.jpg not found in examples/")
h, w = base.shape[:2]

# ------------------------------------------------------------------ points

rng = np.random.default_rng(42)
n = 200
xs = rng.normal(loc=w * 0.5, scale=w * 0.12, size=n).clip(0, w - 1)
ys = rng.normal(loc=h * 0.35, scale=h * 0.12, size=n).clip(0, h - 1)
points = [Point(x=float(x), y=float(y), diameter_pct=0.08) for x, y in zip(xs, ys, strict=True)]

# ------------------------------------------------------------------ helpers

LABEL_H = 32


def labeled_panel(img: np.ndarray, name: str) -> np.ndarray:
    label_bar = np.zeros((LABEL_H, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        label_bar, name, (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return np.vstack([label_bar, img])


# ------------------------------------------------------------------ render

panels: list[np.ndarray] = []

# All colour-mode colormaps
for name in ALL_COLORMAPS:
    hm = Heatmapper(mode="colour", colormap=name, point_diameter=60)
    panels.append(labeled_panel(hm.heatmap_on_img(points, base), name))

# Reveal mode as a dedicated panel
hm_reveal = Heatmapper(mode="reveal", point_diameter=60)
panels.append(labeled_panel(hm_reveal.heatmap_on_img(points, base), "reveal (mode)"))

print(f"Rendered {len(panels)} panels")

# ------------------------------------------------------------------ stitch grid (3 columns)

cols = 3
rows = (len(panels) + cols - 1) // cols
while len(panels) < rows * cols:
    panels.append(np.zeros_like(panels[0]))

grid_rows = [np.hstack(panels[i * cols : (i + 1) * cols]) for i in range(rows)]
grid = np.vstack(grid_rows)

out_path = os.path.join(OUTPUT_DIR, "06_colormaps.jpg")
cv2.imwrite(out_path, grid)
print(f"Saved: {out_path}")

cv2.imshow("Colormap comparison", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
