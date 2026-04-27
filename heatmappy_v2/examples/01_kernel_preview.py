"""
Example 01 — Kernel preview

Shows how GaussianKernel generates point kernels with different
diameters, strengths, and sigmas. Saves a side-by-side comparison image.

Run from the repo root:
    python heatmappy_v2/examples/01_kernel_preview.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from heatmappy2.kernels import GaussianKernel


def kernel_to_display(kernel, size=200):
    """Scale kernel by actual values so strength differences are visible."""
    scaled = np.clip(kernel * 255, 0, 255).astype(np.uint8)
    return cv2.resize(scaled, (size, size), interpolation=cv2.INTER_NEAREST)


def add_label(img, text):
    labeled = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        labeled, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA
    )
    return labeled


EXAMPLES_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(EXAMPLES_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

generator = GaussianKernel()

variants = [
    {"diameter": 51, "strength": 0.5, "sigma": None, "label": "d=51 s=0.5 (default sigma)"},
    {"diameter": 51, "strength": 1.0, "sigma": None, "label": "d=51 s=1.0 (full strength)"},
    {"diameter": 101, "strength": 0.5, "sigma": None, "label": "d=101 s=0.5 (larger)"},
    {"diameter": 101, "strength": 4.0, "sigma": None, "label": "d=101 s=0.5 (larger)"},
    {"diameter": 51, "strength": 0.5, "sigma": 5, "label": "d=51 s=0.5 sigma=5 (sharp)"},
    {"diameter": 51, "strength": 0.5, "sigma": 20, "label": "d=51 s=0.5 sigma=20 (soft)"},
]

panels = []
for v in variants:
    kernel = generator.get(diameter=v["diameter"], strength=v["strength"], sigma=v["sigma"])
    display = kernel_to_display(kernel)
    panel = add_label(display, v["label"])
    panels.append(panel)

print(f"Cache size after generating {len(variants)} variants: {generator.cache_size()}")
print("Requesting d=51 s=0.5 again (should hit cache)...")
generator.get(diameter=51, strength=0.5)
print(f"Cache size unchanged: {generator.cache_size()}")

row = np.hstack(panels)
out_path = os.path.join(OUTPUT_DIR, "01_kernel_preview.png")
cv2.imwrite(out_path, row)
print(f"Saved: {out_path}")

cv2.imshow("Kernel variants", row)
cv2.waitKey(0)
cv2.destroyAllWindows()
