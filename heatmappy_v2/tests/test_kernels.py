from __future__ import annotations

import numpy as np
import pytest

from heatmappy2.kernels import GaussianKernel


def test_shape_matches_diameter() -> None:
    k = GaussianKernel()
    assert k.get(diameter=51).shape == (51, 51)


def test_even_diameter_bumped_to_odd() -> None:
    k = GaussianKernel()
    assert k.get(diameter=50).shape == (51, 51)


def test_peak_equals_strength() -> None:
    k = GaussianKernel()
    for strength in [0.3, 0.5, 1.0]:
        assert abs(k.get(diameter=51, strength=strength).max() - strength) < 1e-5


def test_cache_hit_returns_same_object() -> None:
    k = GaussianKernel()
    a = k.get(diameter=51, strength=0.5)
    b = k.get(diameter=51, strength=0.5)
    assert a is b


def test_cache_miss_different_params() -> None:
    k = GaussianKernel()
    a = k.get(diameter=51, strength=0.5)
    b = k.get(diameter=51, strength=0.8)
    assert a is not b


def test_larger_sigma_wider_spread() -> None:
    k = GaussianKernel()
    sharp = k.get(diameter=101, strength=1.0, sigma=5)
    soft = k.get(diameter=101, strength=1.0, sigma=30)
    assert (soft > 0).sum() > (sharp > 0).sum()


def test_threshold_zeroes_fringe() -> None:
    strength = 0.5
    threshold = 0.01
    k = GaussianKernel(threshold=threshold)
    kernel = k.get(diameter=51, strength=strength)
    nonzero = kernel[kernel > 0]
    assert (nonzero >= threshold * strength).all()


def test_per_call_diameter_derives_own_sigma() -> None:
    k = GaussianKernel(diameter=51)
    small = k.get(diameter=51, strength=1.0)
    large = k.get(diameter=101, strength=1.0)
    assert (large > 0).sum() > (small > 0).sum()


def test_clear_cache() -> None:
    k = GaussianKernel()
    k.get(diameter=51)
    k.get(diameter=101)
    assert k.cache_size() == 2
    k.clear_cache()
    assert k.cache_size() == 0


def test_dtype_is_float32() -> None:
    k = GaussianKernel()
    assert k.get(diameter=51).dtype == np.float32
