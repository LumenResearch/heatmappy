import numpy as np


class GaussianKernel:
    """
    Generates and caches 2D Gaussian kernels for heatmap point rendering.
    Each unique (diameter, strength, sigma) combination is generated once and reused.
    """

    def __init__(self, diameter=50, strength=0.5, sigma=None, threshold=0.01):
        """
        :param diameter: size of the kernel in pixels (must be odd; even values are incremented by 1)
        :param strength: peak intensity of the kernel, between 0 and 1
        :param sigma: standard deviation of the Gaussian; defaults to diameter / 6
        :param threshold: values below this fraction of the peak are zeroed out,
                          preventing invisible fringe pixels from affecting normalisation
        """
        self.diameter = diameter if diameter % 2 == 1 else diameter + 1
        self.strength = strength
        self.sigma = sigma if sigma is not None else self.diameter / 6
        self.threshold = threshold
        self._cache = {}

    def get(self, diameter=None, strength=None, sigma=None):
        """
        Returns a 2D float32 numpy array for the given parameters.
        Uses instance defaults if parameters are not provided.
        Results are cached by (diameter, strength, sigma, threshold).
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
    def _generate(diameter, strength, sigma, threshold):
        center = diameter // 2
        ys, xs = np.ogrid[:diameter, :diameter]
        dist_sq = (xs - center) ** 2 + (ys - center) ** 2
        kernel = np.exp(-dist_sq / (2 * sigma ** 2)).astype(np.float32)
        kernel = (kernel / kernel.max()) * strength
        kernel[kernel < threshold * strength] = 0.0
        return kernel

    def clear_cache(self):
        self._cache.clear()

    def cache_size(self):
        return len(self._cache)
