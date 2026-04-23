from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from heatmappy2.kernels import GaussianKernel


COLORMAPS = {
    'jet':     cv2.COLORMAP_JET,
    'hot':     cv2.COLORMAP_HOT,
    'inferno': cv2.COLORMAP_INFERNO,
    'plasma':  cv2.COLORMAP_PLASMA,
    'viridis': cv2.COLORMAP_VIRIDIS,
    'turbo':   cv2.COLORMAP_TURBO,
    'bone':    cv2.COLORMAP_BONE,
}


@dataclass
class Point:
    x: float
    y: float
    diameter: Optional[int] = None
    strength: Optional[float] = None
    sigma: Optional[float] = None


class GreyHeatmapper:
    def __init__(self,
                 point_diameter=50,
                 point_strength=0.5,
                 sigma=None,
                 normalisation='relative',
                 ceiling=None,
                 min_intensity=0.0):
        """
        :param point_diameter: default kernel diameter in pixels
        :param point_strength: default kernel peak intensity (0–1)
        :param sigma: default Gaussian sigma; defaults to point_diameter / 6
        :param normalisation: 'relative', 'absolute', or 'raw'
            - relative: hottest point = full brightness
            - absolute: normalise against a fixed ceiling (number of people / fixations)
            - raw: no normalisation, values passed through as-is
        :param ceiling: required for absolute normalisation
        :param min_intensity: floor (0–1) applied to all non-zero pixels after
                              normalisation, preventing sparse data from being invisible
        """
        self._kernel = GaussianKernel(point_diameter, point_strength, sigma)
        self.normalisation = normalisation
        self.ceiling = ceiling
        self.min_intensity = min_intensity

    def heatmap(self, width, height, points):
        """
        :param points: iterable of Point objects
        :return: uint8 greyscale numpy array of shape (height, width)
        """
        canvas = np.zeros((height, width), dtype=np.float32)

        for point in points:
            kernel = self._kernel.get(
                diameter=point.diameter,
                strength=point.strength,
                sigma=point.sigma,
            )
            self._stamp(canvas, kernel, int(point.x), int(point.y))

        return self._normalise(canvas)

    def _stamp(self, canvas, kernel, x, y):
        kh, kw = kernel.shape
        half_h, half_w = kh // 2, kw // 2

        x0, y0 = x - half_w, y - half_h
        x1, y1 = x0 + kw, y0 + kh

        kx0 = max(0, -x0)
        ky0 = max(0, -y0)
        kx1 = kw - max(0, x1 - canvas.shape[1])
        ky1 = kh - max(0, y1 - canvas.shape[0])

        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(canvas.shape[1], x1), min(canvas.shape[0], y1)

        if cx0 >= cx1 or cy0 >= cy1:
            return

        canvas[cy0:cy1, cx0:cx1] += kernel[ky0:ky1, kx0:kx1]

    def _normalise(self, canvas):
        max_val = canvas.max()
        if max_val == 0:
            return np.zeros(canvas.shape, dtype=np.uint8)

        if self.normalisation == 'relative':
            normalised = canvas / max_val
        elif self.normalisation == 'absolute':
            if self.ceiling is None:
                raise ValueError("ceiling must be set when using absolute normalisation")
            normalised = canvas / self.ceiling
        elif self.normalisation == 'raw':
            normalised = canvas.copy()
        else:
            raise ValueError(f"Unknown normalisation mode: '{self.normalisation}'")

        normalised = np.clip(normalised, 0.0, 1.0)

        if self.min_intensity > 0.0:
            mask = normalised > 0
            normalised[mask] = np.clip(normalised[mask], self.min_intensity, 1.0)

        return (normalised * 255).astype(np.uint8)


class Heatmapper:
    def __init__(self,
                 point_diameter=50,
                 point_strength=0.5,
                 sigma=None,
                 normalisation='relative',
                 ceiling=None,
                 min_intensity=0.0,
                 mode='colour',
                 colormap='jet',
                 opacity=0.65):
        """
        :param mode: 'colour' or 'reveal'
            - colour: colourised heatmap composited over the image
            - reveal: image is shown where attention is high, darkened where attention is low
        :param colormap: name from COLORMAPS dict, used in colour mode only
        :param opacity: max opacity of the heatmap overlay (0–1), colour mode only
        All other params are passed through to GreyHeatmapper.
        """
        self._grey = GreyHeatmapper(
            point_diameter=point_diameter,
            point_strength=point_strength,
            sigma=sigma,
            normalisation=normalisation,
            ceiling=ceiling,
            min_intensity=min_intensity,
        )
        self.mode = mode
        self.opacity = opacity

        if isinstance(colormap, str):
            if colormap not in COLORMAPS:
                raise ValueError(f"Unknown colormap '{colormap}'. Choose from: {list(COLORMAPS)}")
            self.colormap = COLORMAPS[colormap]
        else:
            self.colormap = colormap

    def heatmap(self, width, height, points, base_img=None):
        """
        :param base_img: BGR numpy array; if None returns heatmap without background
        :return: BGR numpy array
        """
        grey = self._grey.heatmap(width, height, points)

        if self.mode == 'colour':
            return self._colour(grey, base_img)
        elif self.mode == 'reveal':
            if base_img is None:
                raise ValueError("base_img is required for reveal mode")
            return self._reveal(grey, base_img)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Choose 'colour' or 'reveal'")

    def heatmap_on_img(self, points, img):
        h, w = img.shape[:2]
        return self.heatmap(w, h, points, base_img=img)

    def heatmap_on_img_path(self, points, img_path):
        img = cv2.imread(img_path)
        return self.heatmap_on_img(points, img)

    def _colour(self, grey, base_img):
        coloured = cv2.applyColorMap(grey, self.colormap)

        if base_img is None:
            return coloured

        alpha = (grey / 255.0 * self.opacity)[:, :, np.newaxis]
        return (coloured * alpha + base_img * (1.0 - alpha)).astype(np.uint8)

    def _reveal(self, grey, base_img):
        # hot areas (high attention) reveal image at full brightness
        # cold areas fade to black
        reveal = (grey / 255.0)[:, :, np.newaxis]
        return (base_img * reveal).astype(np.uint8)