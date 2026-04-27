from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest

from heatmappy2.heatmap import Heatmapper
from heatmappy2.video import VideoHeatmapper, VideoPoint

# ------------------------------------------------------------------ helpers


def _vh(decay_time_ms: float | None = None, smooth_decay: bool = False) -> VideoHeatmapper:
    return VideoHeatmapper(Heatmapper(point_diameter=20), decay_time_ms=decay_time_ms, smooth_decay=smooth_decay)


def _base(w: int = 64, h: int = 48) -> np.ndarray:
    img = np.full((h, w, 3), 100, dtype=np.uint8)
    return img


def _points(
    n_per_frame: int = 3,
    w: int = 64,
    h: int = 48,
    duration_ms: float = 500.0,
    fps: float = 10.0,
) -> list[VideoPoint]:
    interval = 1000.0 / fps
    pts: list[VideoPoint] = []
    for i in range(int(duration_ms / interval)):
        t = i * interval
        for _ in range(n_per_frame):
            pts.append(VideoPoint(x=w / 2, y=h / 2, t=t))
    return pts


# ------------------------------------------------------------------ _compute_weight


def test_weight_no_decay_in_frame() -> None:
    assert _vh()._compute_weight(0.0, 50.0) == 1.0


def test_weight_no_decay_past_frame() -> None:
    assert _vh()._compute_weight(50.0, 50.0) is None


def test_weight_future_point_is_none() -> None:
    assert _vh()._compute_weight(-1.0, 50.0) is None


def test_weight_hard_decay_within_window() -> None:
    assert _vh(decay_time_ms=500.0)._compute_weight(499.0, 50.0) == 1.0


def test_weight_hard_decay_expired() -> None:
    assert _vh(decay_time_ms=500.0)._compute_weight(501.0, 50.0) is None


def test_weight_smooth_decay_at_start() -> None:
    assert _vh(decay_time_ms=500.0, smooth_decay=True)._compute_weight(0.0, 50.0) == pytest.approx(1.0)


def test_weight_smooth_decay_at_midpoint() -> None:
    assert _vh(decay_time_ms=500.0, smooth_decay=True)._compute_weight(250.0, 50.0) == pytest.approx(0.5)


def test_weight_smooth_decay_at_end() -> None:
    assert _vh(decay_time_ms=500.0, smooth_decay=True)._compute_weight(500.0, 50.0) == pytest.approx(0.0)


def test_smooth_decay_without_decay_time_raises() -> None:
    with pytest.raises(ValueError):
        _vh(smooth_decay=True)


# ------------------------------------------------------------------ _clamp_window


def test_clamp_window_defaults_to_full_range() -> None:
    assert _vh()._clamp_window(0.0, None, 5000.0) == (0.0, 5000.0)


def test_clamp_window_caps_end_at_max() -> None:
    assert _vh()._clamp_window(0.0, 9999.0, 5000.0) == (0.0, 5000.0)


def test_clamp_window_clamps_negative_start() -> None:
    t_start, _ = _vh()._clamp_window(-100.0, 1000.0, 5000.0)
    assert t_start == 0.0


def test_clamp_window_end_before_start_raises() -> None:
    with pytest.raises(ValueError):
        _vh()._clamp_window(1000.0, 500.0, 5000.0)


def test_clamp_window_equal_start_end_raises() -> None:
    with pytest.raises(ValueError):
        _vh()._clamp_window(1000.0, 1000.0, 5000.0)


# ------------------------------------------------------------------ _snap_points


def test_snap_rounds_to_nearest_boundary() -> None:
    snapped = _vh()._snap_points([VideoPoint(x=0, y=0, t=74.0)], 50.0)
    assert snapped[0].t == pytest.approx(50.0)


def test_snap_exact_boundary_unchanged() -> None:
    snapped = _vh()._snap_points([VideoPoint(x=0, y=0, t=100.0)], 50.0)
    assert snapped[0].t == pytest.approx(100.0)


# ------------------------------------------------------------------ _active_points


def test_active_points_includes_current_frame() -> None:
    pts = [VideoPoint(x=10, y=20, t=0.0)]
    active = _vh()._active_points(0.0, pts, 50.0)
    assert len(active) == 1
    assert active[0].x == 10


def test_active_points_excludes_future_points() -> None:
    pts = [VideoPoint(x=10, y=20, t=100.0)]
    assert _vh()._active_points(0.0, pts, 50.0) == []


def test_active_points_smooth_decay_weight_applied() -> None:
    pts = [VideoPoint(x=10, y=20, t=0.0, strength=1.0)]
    active = _vh(decay_time_ms=500.0, smooth_decay=True)._active_points(250.0, pts, 50.0)
    assert len(active) == 1
    assert active[0].strength == pytest.approx(0.5)


def test_active_points_default_strength_is_one() -> None:
    pts = [VideoPoint(x=10, y=20, t=0.0)]  # strength=None
    active = _vh(decay_time_ms=500.0)._active_points(0.0, pts, 50.0)
    assert active[0].strength == pytest.approx(1.0)


# ------------------------------------------------------------------ heatmap_on_image integration


def test_heatmap_on_image_creates_nonempty_file() -> None:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        out = f.name
    try:
        _vh().heatmap_on_image(_base(), _points(), out, duration_ms=500.0, fps=10.0)
        assert os.path.getsize(out) > 0
    finally:
        os.unlink(out)


def test_heatmap_on_image_frame_count() -> None:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        out = f.name
    try:
        _vh().heatmap_on_image(_base(), _points(duration_ms=1000.0), out, duration_ms=1000.0, fps=10.0)
        cap = cv2.VideoCapture(out)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert frames == 10
    finally:
        os.unlink(out)


def test_windowed_clip_is_shorter_than_full() -> None:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f1, \
         tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f2:
        out_full, out_clip = f1.name, f2.name
    pts = _points(duration_ms=2000.0)
    try:
        _vh().heatmap_on_image(_base(), pts, out_full, duration_ms=2000.0, fps=10.0)
        _vh().heatmap_on_image(_base(), pts, out_clip, duration_ms=2000.0, fps=10.0,
                               start_ms=500.0, end_ms=1500.0)
        cap_full = cv2.VideoCapture(out_full)
        cap_clip = cv2.VideoCapture(out_clip)
        full_frames = int(cap_full.get(cv2.CAP_PROP_FRAME_COUNT))
        clip_frames = int(cap_clip.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_full.release()
        cap_clip.release()
        assert clip_frames < full_frames
    finally:
        os.unlink(out_full)
        os.unlink(out_clip)


def test_pair_mode_output_is_double_width() -> None:
    hm = Heatmapper(point_diameter=20, mode="pair")
    vh = VideoHeatmapper(hm)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        out = f.name
    try:
        vh.heatmap_on_image(_base(w=64), _points(), out, duration_ms=500.0, fps=10.0)
        cap = cv2.VideoCapture(out)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        assert w == 128
    finally:
        os.unlink(out)
