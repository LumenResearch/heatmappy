from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from heatmappy2.heatmap import GreyHeatmapper, Heatmapper, Point, points_from_tuples

# shared 100×80 BGR base image (grey fill)
BASE = np.full((80, 100, 3), 128, dtype=np.uint8)


# ------------------------------------------------------------------ Point


def test_point_diameter_and_pct_conflict() -> None:
    with pytest.raises(ValidationError):
        Point(x=10, y=10, diameter=50, diameter_pct=0.2)


def test_from_tuple_sets_diameter_pct() -> None:
    p = Point.from_tuple((100.0, 200.0))
    assert p.x == 100.0
    assert p.y == 200.0
    assert p.diameter_pct == pytest.approx(0.20)


def test_points_from_tuples_length_and_pct() -> None:
    pts = points_from_tuples([(10.0, 20.0), (30.0, 40.0)])
    assert len(pts) == 2
    assert all(p.diameter_pct == pytest.approx(0.20) for p in pts)


# ------------------------------------------------------------------ GreyHeatmapper


def test_grey_heatmap_shape_and_dtype() -> None:
    hm = GreyHeatmapper()
    result = hm.heatmap(100, 80, [Point(x=50, y=40)])
    assert result.shape == (80, 100)
    assert result.dtype == np.uint8


def test_grey_heatmap_empty_points_all_zero() -> None:
    hm = GreyHeatmapper()
    assert hm.heatmap(100, 80, []).max() == 0


def test_resolve_diameter_priority() -> None:
    hm = GreyHeatmapper(point_diameter=50)
    # explicit diameter wins
    assert hm._resolve_diameter(Point(x=0, y=0, diameter=80), 200, 200) == 80
    # diameter_pct wins over heatmapper default
    assert hm._resolve_diameter(Point(x=0, y=0, diameter_pct=0.10), 200, 200) == 20
    # falls back to heatmapper default (50 → bumped to odd 51)
    assert hm._resolve_diameter(Point(x=0, y=0), 200, 200) == 51


def test_relative_normalisation_max_is_255() -> None:
    hm = GreyHeatmapper(normalisation="relative")
    assert hm.heatmap(100, 100, [Point(x=50, y=50)]).max() == 255


def test_absolute_normalisation_scales_by_ceiling() -> None:
    hm = GreyHeatmapper(normalisation="absolute", ceiling=2.0, point_strength=1.0)
    result = hm.heatmap(100, 100, [Point(x=50, y=50, diameter=10)])
    # peak = 1.0 / 2.0 → ~127; should never exceed 128
    assert result.max() <= 128


def test_absolute_normalisation_requires_ceiling() -> None:
    hm = GreyHeatmapper(normalisation="absolute")
    with pytest.raises(ValueError):
        hm.heatmap(100, 100, [Point(x=50, y=50)])


def test_min_intensity_floor_on_nonzero_pixels() -> None:
    hm = GreyHeatmapper(normalisation="relative", min_intensity=0.3)
    result = hm.heatmap(100, 100, [Point(x=50, y=50)])
    nonzero = result[result > 0]
    assert nonzero.min() >= int(0.3 * 255)


def test_point_stamped_at_correct_location() -> None:
    hm = GreyHeatmapper(normalisation="relative")
    result = hm.heatmap(100, 100, [Point(x=80, y=60, diameter=11)])
    peak_y, peak_x = np.unravel_index(result.argmax(), result.shape)
    assert abs(peak_x - 80) <= 1
    assert abs(peak_y - 60) <= 1


def test_out_of_bounds_point_does_not_crash() -> None:
    hm = GreyHeatmapper()
    hm.heatmap(100, 100, [Point(x=-50, y=-50), Point(x=200, y=200)])


# ------------------------------------------------------------------ Heatmapper


def test_colour_mode_shape_with_base() -> None:
    hm = Heatmapper(mode="colour")
    result = hm.heatmap(100, 80, [Point(x=50, y=40)], base_img=BASE)
    assert result.shape == (80, 100, 3)
    assert result.dtype == np.uint8


def test_colour_mode_works_without_base_img() -> None:
    hm = Heatmapper(mode="colour")
    result = hm.heatmap(100, 80, [Point(x=50, y=40)])
    assert result.shape == (80, 100, 3)


def test_reveal_mode_requires_base_img() -> None:
    hm = Heatmapper(mode="reveal")
    with pytest.raises(ValueError):
        hm.heatmap(100, 80, [Point(x=50, y=40)])


def test_pair_mode_requires_base_img() -> None:
    hm = Heatmapper(mode="pair")
    with pytest.raises(ValueError):
        hm.heatmap(100, 80, [Point(x=50, y=40)])


def test_pair_mode_output_double_width() -> None:
    hm = Heatmapper(mode="pair")
    result = hm.heatmap(100, 80, [Point(x=50, y=40)], base_img=BASE)
    assert result.shape == (80, 200, 3)


def test_output_shape_all_modes() -> None:
    assert Heatmapper(mode="colour").output_shape(80, 100) == (80, 100)
    assert Heatmapper(mode="reveal").output_shape(80, 100) == (80, 100)
    assert Heatmapper(mode="pair").output_shape(80, 100) == (80, 200)


def test_unknown_colormap_raises() -> None:
    with pytest.raises(ValueError):
        Heatmapper(colormap="rainbow")


def test_heatmap_on_img_convenience() -> None:
    hm = Heatmapper(mode="colour")
    result = hm.heatmap_on_img([Point(x=50, y=40)], BASE)
    assert result.shape == BASE.shape
