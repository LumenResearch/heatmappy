from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from heatmappy2.kernels import GaussianKernel

_ASSETS = Path(__file__).parent / "assets"

# Built-in OpenCV colormaps
COLORMAPS: dict[str, int] = {
    "jet": cv2.COLORMAP_JET,
    "hot": cv2.COLORMAP_HOT,
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "turbo": cv2.COLORMAP_TURBO,
    "bone": cv2.COLORMAP_BONE,
}

# Custom LUT colormaps loaded from horizontal PNG strips.
# Keys can be passed as the `colormap` argument just like built-in names.
# Note: v1's "reveal" colormap worked through PNG alpha and does not port to an RGB LUT.
# Use mode="reveal" instead — it achieves the same effect via direct compositing.
CUSTOM_COLORMAPS: dict[str, Path] = {
    "classic": _ASSETS / "classic.png",   # v1 "default" — red → green → blue
}

ALL_COLORMAPS = list(COLORMAPS) + list(CUSTOM_COLORMAPS)

NormalisationMode = Literal["relative", "absolute", "raw"]
HeatmapMode = Literal["colour", "reveal", "pair"]
PointList = list["Point"]


def _lut_from_strip(img_path: Path, reverse: bool = False) -> NDArray[np.uint8]:
    """Load a horizontal colormap strip PNG as a (256, 3) BGR uint8 LUT."""
    img = Image.open(img_path).convert("RGB").resize((256, 1), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.uint8)[0]  # (256, 3) RGB
    lut: NDArray[np.uint8] = arr[:, ::-1]   # → BGR
    if reverse:
        lut = lut[::-1].copy()
    return lut


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
        """Convenience constructor from a plain (x, y) tuple. Defaults to 20% of image size."""
        return cls(x=t[0], y=t[1], diameter_pct=0.20)


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

    def _resolve_diameter(self, point: Point, width: int, height: int) -> int:
        if point.diameter is not None:
            return point.diameter
        if point.diameter_pct is not None:
            return int(min(width, height) * point.diameter_pct)
        return self._kernel.diameter

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
        :param colormap: built-in: jet, hot, inferno, plasma, viridis, turbo, bone;
                         custom: classic (v1 red→green→blue), reveal_lut (v1 transparency mask)
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

        if colormap in COLORMAPS:
            self._colormap_id: int | None = COLORMAPS[colormap]
            self._colormap_lut: NDArray[np.uint8] | None = None
        elif colormap in CUSTOM_COLORMAPS:
            self._colormap_id = None
            # v1 strips are indexed dense→0, sparse→255; v2 is the opposite, so reverse
            self._colormap_lut = _lut_from_strip(CUSTOM_COLORMAPS[colormap], reverse=True)
        else:
            raise ValueError(f"Unknown colormap '{colormap}'. Choose from: {ALL_COLORMAPS}")

    def output_shape(self, height: int, width: int) -> tuple[int, int]:
        """Return (out_height, out_width) for a source frame of the given size."""
        return (height, width * 2) if self.mode == "pair" else (height, width)

    def heatmap(
        self,
        width: int,
        height: int,
        points: PointList,
        base_img: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]:
        """
        :param base_img: BGR uint8 numpy array; required for reveal and pair modes
        :return: BGR uint8 numpy array (double-width for pair mode)
        """
        grey = self._grey.heatmap(width, height, points)

        if self.mode == "colour":
            return self._colour(grey, base_img)
        elif self.mode == "reveal":
            if base_img is None:
                raise ValueError("base_img is required for reveal mode")
            return self._reveal(grey, base_img)
        else:  # pair
            if base_img is None:
                raise ValueError("base_img is required for pair mode")
            return self._pair(grey, base_img)

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
        if self._colormap_lut is not None:
            coloured: NDArray[np.uint8] = self._colormap_lut[grey]
        else:
            coloured = cv2.applyColorMap(grey, self._colormap_id)  # type: ignore[arg-type,assignment]
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

    def _pair(
        self,
        grey: NDArray[np.uint8],
        base_img: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        left: NDArray[np.uint8] = self._colour(grey, base_img)
        right: NDArray[np.uint8] = self._reveal(grey, base_img)
        result: NDArray[np.uint8] = np.hstack([left, right])
        return result
