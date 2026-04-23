from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator

from heatmappy2.kernels import GaussianKernel

COLORMAPS: dict[str, int] = {
    "jet": cv2.COLORMAP_JET,
    "hot": cv2.COLORMAP_HOT,
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "turbo": cv2.COLORMAP_TURBO,
    "bone": cv2.COLORMAP_BONE,
}

NormalisationMode = Literal["relative", "absolute", "raw"]
HeatmapMode = Literal["colour", "reveal"]
PointList = list["Point"]


class Point(BaseModel):
    """A single gaze or attention point with optional per-point kernel overrides."""

    x: float
    y: float
    diameter: int | None = Field(default=None, gt=0)
    diameter_pct: float | None = Field(default=None, gt=0.0, le=1.0)
    strength: float | None = Field(default=None, ge=0.0, le=1.0)
    sigma: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def diameter_not_both(self) -> Point:
        if self.diameter is not None and self.diameter_pct is not None:
            raise ValueError("specify diameter or diameter_pct, not both")
        return self

    @classmethod
    def from_tuple(cls, t: tuple[float, float]) -> Point:
        """Convenience constructor from a plain (x, y) tuple with default sizing."""
        return cls(x=t[0], y=t[1])


def points_from_tuples(tuples: list[tuple[float, float]]) -> PointList:
    """Convert a list of (x, y) tuples to Points using default sizing."""
    return [Point.from_tuple(t) for t in tuples]


class GreyHeatmapper:
    """
    Renders a greyscale heatmap from a list of Points onto a float32 canvas,
    then normalises and returns a uint8 image.
    """

    def __init__(
        self,
        point_diameter: int = 50,
        point_strength: float = 0.5,
        sigma: float | None = None,
        normalisation: NormalisationMode = "relative",
        ceiling: float | None = None,
        min_intensity: float = 0.0,
    ) -> None:
        """
        :param point_diameter: default kernel diameter in pixels
        :param point_strength: default kernel peak intensity (0–1)
        :param sigma: default Gaussian sigma; defaults to point_diameter / 6
        :param normalisation: 'relative', 'absolute', or 'raw'
        :param ceiling: required for absolute normalisation (e.g. total participant count)
        :param min_intensity: floor (0–1) applied to non-zero pixels after normalisation
        """
        self._kernel = GaussianKernel(point_diameter, point_strength, sigma)
        self.normalisation: NormalisationMode = normalisation
        self.ceiling: float | None = ceiling
        self.min_intensity: float = min_intensity

    def heatmap(
        self,
        width: int,
        height: int,
        points: PointList,
    ) -> NDArray[np.uint8]:
        """
        :param points: list of Point objects
        :return: uint8 greyscale array of shape (height, width)
        """
        canvas = np.zeros((height, width), dtype=np.float32)

        for point in points:
            kernel = self._kernel.get(
                diameter=self._resolve_diameter(point, width, height),
                strength=point.strength,
                sigma=point.sigma,
            )
            self._stamp(canvas, kernel, int(point.x), int(point.y))

        return self._normalise(canvas)

    @staticmethod
    def _resolve_diameter(point: Point, width: int, height: int) -> int | None:
        if point.diameter is not None:
            return point.diameter
        pct = point.diameter_pct if point.diameter_pct is not None else 0.05
        return int(min(width, height) * pct)

    def _stamp(
        self,
        canvas: NDArray[np.float32],
        kernel: NDArray[np.float32],
        x: int,
        y: int,
    ) -> None:
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

    def _normalise(self, canvas: NDArray[np.float32]) -> NDArray[np.uint8]:
        max_val = float(canvas.max())
        if max_val == 0:
            return np.zeros(canvas.shape, dtype=np.uint8)

        if self.normalisation == "relative":
            normalised = canvas / max_val
        elif self.normalisation == "absolute":
            if self.ceiling is None:
                raise ValueError("ceiling must be set when using absolute normalisation")
            normalised = canvas / self.ceiling
        else:  # raw
            normalised = canvas.copy()

        normalised = np.clip(normalised, 0.0, 1.0)

        if self.min_intensity > 0.0:
            mask = normalised > 0
            normalised[mask] = np.clip(normalised[mask], self.min_intensity, 1.0)

        result: NDArray[np.uint8] = (normalised * 255).astype(np.uint8)
        return result


class Heatmapper:
    """
    High-level heatmap renderer. Wraps GreyHeatmapper and applies
    colour or reveal compositing over a base image.
    """

    def __init__(
        self,
        point_diameter: int = 50,
        point_strength: float = 0.5,
        sigma: float | None = None,
        normalisation: NormalisationMode = "relative",
        ceiling: float | None = None,
        min_intensity: float = 0.0,
        mode: HeatmapMode = "colour",
        colormap: str = "jet",
        opacity: float = 0.65,
    ) -> None:
        """
        :param mode: 'colour' overlays a colourised heatmap; 'reveal' shows the image
                     only where attention is high, darkening unattended areas
        :param colormap: one of: jet, hot, inferno, plasma, viridis, turbo, bone
        :param opacity: max heatmap opacity (0–1), colour mode only
        """
        self._grey = GreyHeatmapper(
            point_diameter=point_diameter,
            point_strength=point_strength,
            sigma=sigma,
            normalisation=normalisation,
            ceiling=ceiling,
            min_intensity=min_intensity,
        )
        self.mode: HeatmapMode = mode
        self.opacity: float = opacity

        if colormap not in COLORMAPS:
            raise ValueError(f"Unknown colormap '{colormap}'. Choose from: {list(COLORMAPS)}")
        self.colormap: int = COLORMAPS[colormap]

    def heatmap(
        self,
        width: int,
        height: int,
        points: PointList,
        base_img: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]:
        """
        :param base_img: BGR uint8 numpy array; required for reveal mode
        :return: BGR uint8 numpy array
        """
        grey = self._grey.heatmap(width, height, points)

        if self.mode == "colour":
            return self._colour(grey, base_img)
        else:
            if base_img is None:
                raise ValueError("base_img is required for reveal mode")
            return self._reveal(grey, base_img)

    def heatmap_on_img(
        self,
        points: PointList,
        img: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        h, w = img.shape[:2]
        return self.heatmap(w, h, points, base_img=img)

    def heatmap_on_img_path(
        self,
        points: PointList,
        img_path: str | Path,
    ) -> NDArray[np.uint8]:
        raw: NDArray[np.uint8] | None = cv2.imread(str(img_path))  # type: ignore[assignment]
        if raw is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        return self.heatmap_on_img(points, raw)

    def _colour(
        self,
        grey: NDArray[np.uint8],
        base_img: NDArray[np.uint8] | None,
    ) -> NDArray[np.uint8]:
        coloured: NDArray[np.uint8] = cv2.applyColorMap(grey, self.colormap)  # type: ignore[assignment]
        if base_img is None:
            return coloured
        alpha = (grey / 255.0 * self.opacity)[:, :, np.newaxis]
        blended: np.ndarray = coloured * alpha + base_img * (1.0 - alpha)
        return blended.astype(np.uint8)

    def _reveal(
        self,
        grey: NDArray[np.uint8],
        base_img: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        reveal = (grey / 255.0)[:, :, np.newaxis]
        return (base_img * reveal).astype(np.uint8)
