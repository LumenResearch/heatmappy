from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class GaussianKernel:
    """
    Generates and caches 2D Gaussian kernels for heatmap point rendering.
    Each unique (diameter, strength, sigma, threshold) combination is generated
    once and reused across all heatmap calls.
    """

    def __init__(
        self,
        diameter: int = 50,
        strength: float = 0.5,
        sigma: float | None = None,
        threshold: float = 0.01,
    ) -> None:
        """
        :param diameter: default kernel size in pixels; even values are bumped to odd
        :param strength: default peak intensity (0–1)
        :param sigma: default Gaussian std dev; defaults to diameter / 6
        :param threshold: values below threshold * strength are zeroed to remove
                          invisible fringe pixels that distort min_intensity behaviour
        """
        self.diameter: int = diameter if diameter % 2 == 1 else diameter + 1
        self.strength: float = strength
        self.sigma: float = sigma if sigma is not None else self.diameter / 6
        self.threshold: float = threshold
        self._cache: dict[tuple[int, float, float, float], NDArray[np.float32]] = {}

    def get(
        self,
        diameter: int | None = None,
        strength: float | None = None,
        sigma: float | None = None,
    ) -> NDArray[np.float32]:
        """
        Return a cached 2D float32 kernel for the given parameters.
        Falls back to instance defaults for any parameter not provided.
        When a per-call diameter is given without a sigma, sigma is derived
        from that diameter so the Gaussian scales proportionally.
        """
        d = diameter if diameter is not None else self.diameter
        s = strength if strength is not None else self.strength

        if sigma is not None:
            sig = sigma
        elif diameter is not None:
            sig = d / 6
        else:
            sig = self.sigma

        d = d if d % 2 == 1 else d + 1

        key = (d, s, sig, self.threshold)
        if key not in self._cache:
            self._cache[key] = self._generate(d, s, sig, self.threshold)
        return self._cache[key]

    @staticmethod
    def _generate(
        diameter: int,
        strength: float,
        sigma: float,
        threshold: float,
    ) -> NDArray[np.float32]:
        center = diameter // 2
        ys, xs = np.ogrid[:diameter, :diameter]
        dist_sq = (xs - center) ** 2 + (ys - center) ** 2
        kernel: NDArray[np.float32] = np.exp(-dist_sq / (2 * sigma**2)).astype(np.float32)
        kernel = (kernel / kernel.max()) * strength
        kernel[kernel < threshold * strength] = 0.0
        return kernel

    def clear_cache(self) -> None:
        self._cache.clear()

    def cache_size(self) -> int:
        return len(self._cache)
