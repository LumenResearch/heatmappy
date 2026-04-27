"""
Generates the images used in the README and saves them to docs/.
Run from the repo root:
    python heatmappy_v2/generate_docs_images.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np

from heatmappy2.heatmap import ALL_COLORMAPS, GreyHeatmapper, Heatmapper, Point, points_from_tuples

SCRIPT_DIR = os.path.dirname(__file__)
DOCS_DIR = os.path.join(SCRIPT_DIR, "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

IMG_PATH = os.path.join(SCRIPT_DIR, "examples", "cat.jpg")
img = cv2.imread(IMG_PATH)
H, W = img.shape[:2]  # 349 × 512

LABEL_H = 28


def label(bgr: np.ndarray, text: str) -> np.ndarray:
    out = bgr.copy()
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA)
    return out


def save(img: np.ndarray, name: str) -> None:
    path = os.path.join(DOCS_DIR, name)
    cv2.imwrite(path, img)
    print(f"  saved {path}")


def clustered(rng: np.random.Generator, n: int, cx: float, cy: float, spread: float) -> list[Point]:
    xs = rng.normal(cx, spread, n).clip(0, W - 1)
    ys = rng.normal(cy, spread, n).clip(0, H - 1)
    return [Point(x=float(x), y=float(y)) for x, y in zip(xs, ys, strict=True)]


# ------------------------------------------------------------------ 1. colour + reveal modes

rng1 = np.random.default_rng(42)
pts_mode = (
    clustered(rng1, 40, W * 0.35, H * 0.35, 50)
    + clustered(rng1, 20, W * 0.65, H * 0.55, 35)
    + clustered(rng1, 10, W * 0.50, H * 0.70, 25)
)

r1 = [label(Heatmapper(point_diameter=80, mode="colour", colormap=c).heatmap_on_img(pts_mode, img), f"colour | {c}")
      for c in ["jet", "hot", "inferno", "turbo"]]
r2 = [
    label(Heatmapper(point_diameter=80, mode="colour", colormap="jet", opacity=0.35).heatmap_on_img(pts_mode, img), "opacity=0.35"),
    label(Heatmapper(point_diameter=80, mode="colour", colormap="jet", opacity=0.65).heatmap_on_img(pts_mode, img), "opacity=0.65"),
    label(Heatmapper(point_diameter=80, mode="colour", colormap="jet", opacity=0.90).heatmap_on_img(pts_mode, img), "opacity=0.90"),
    label(Heatmapper(point_diameter=80, mode="reveal").heatmap_on_img(pts_mode, img), "reveal"),
]

print("Generating colour_and_reveal.jpg ...")
save(np.vstack([np.hstack(r1), np.hstack(r2)]), "colour_and_reveal.jpg")

# ------------------------------------------------------------------ 2. colormaps grid
# Use same tight clusters so every panel shows the same density pattern

rng2 = np.random.default_rng(7)
pts_cmap = (
    clustered(rng2, 60, W * 0.40, H * 0.35, 45)
    + clustered(rng2, 30, W * 0.65, H * 0.55, 30)
    + clustered(rng2, 15, W * 0.30, H * 0.65, 20)
)

panels = []
for name in ALL_COLORMAPS:
    result = Heatmapper(mode="colour", colormap=name, point_diameter=80).heatmap_on_img(pts_cmap, img)
    bar = np.zeros((LABEL_H, W, 3), dtype=np.uint8)
    cv2.putText(bar, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    panels.append(np.vstack([bar, result]))

reveal_result = Heatmapper(mode="reveal", point_diameter=80).heatmap_on_img(pts_cmap, img)
bar = np.zeros((LABEL_H, W, 3), dtype=np.uint8)
cv2.putText(bar, "reveal (mode)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
panels.append(np.vstack([bar, reveal_result]))

while len(panels) % 3 != 0:
    panels.append(np.zeros_like(panels[0]))

cols = 3
grid_rows = [np.hstack(panels[i * cols:(i + 1) * cols]) for i in range(len(panels) // cols)]

print("Generating colormaps.jpg ...")
save(np.vstack(grid_rows), "colormaps.jpg")

# ------------------------------------------------------------------ 3. grey normalisation

rng3 = np.random.default_rng(99)
all_pts = points_from_tuples([
    (float(x), float(y))
    for x, y in zip(rng3.integers(0, W, 150), rng3.integers(0, H, 150), strict=True)
])
few_pts = clustered(rng3, 3, W // 3, H // 2, 25)

hm_rel = GreyHeatmapper(point_strength=0.5, normalisation="relative")
hm_abs = GreyHeatmapper(point_strength=0.5, normalisation="absolute", ceiling=150)

grey_panels = [
    label(cv2.cvtColor(hm_rel.heatmap(W, H, all_pts), cv2.COLOR_GRAY2BGR), "150 pts | relative"),
    label(cv2.cvtColor(hm_rel.heatmap(W, H, few_pts), cv2.COLOR_GRAY2BGR), "3 pts | relative"),
    label(cv2.cvtColor(hm_abs.heatmap(W, H, few_pts), cv2.COLOR_GRAY2BGR), "3 pts | absolute ceiling=150"),
]

print("Generating grey_heatmap.jpg ...")
save(np.hstack(grey_panels), "grey_heatmap.jpg")

print("Done.")
