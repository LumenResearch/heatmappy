from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import ffmpeg
import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from heatmappy2.heatmap import Heatmapper, Point, PointList


class VideoPoint(Point):
    """A gaze point with a timestamp for video heatmaps."""

    t: float = Field(ge=0.0)  # timestamp in milliseconds


VideoPointList = list[VideoPoint]


class VideoHeatmapper:
    """
    Renders heatmap videos from time-series gaze points.

    Two modes:
    - heatmap_on_image: static base image + gaze points → video at chosen FPS
    - heatmap_on_video: video file + gaze points → video at source FPS with audio preserved

    Decay modes (controlled by decay_time_ms and smooth_decay):
    - Neither set              → point appears only in its own frame
    - decay_time_ms set only   → full intensity for decay_time_ms then vanishes
    - decay_time_ms + smooth   → linear fade from full intensity to zero over decay_time_ms
    """

    def __init__(
        self,
        heatmapper: Heatmapper,
        decay_time_ms: float | None = None,
        smooth_decay: bool = False,
    ) -> None:
        """
        :param heatmapper: configured Heatmapper instance used to render each frame
        :param decay_time_ms: how long (ms) a point persists after its timestamp
        :param smooth_decay: if True, intensity fades linearly over decay_time_ms;
                             requires decay_time_ms to be set
        """
        if smooth_decay and decay_time_ms is None:
            raise ValueError("smooth_decay requires decay_time_ms to be set")

        self._heatmapper = heatmapper
        self.decay_time_ms = decay_time_ms
        self.smooth_decay = smooth_decay

    # ------------------------------------------------------------------ public

    def heatmap_on_image(
        self,
        base_img: NDArray[np.uint8],
        points: VideoPointList,
        output_path: str | Path,
        duration_ms: float,
        fps: float = 20.0,
    ) -> None:
        """
        Render a heatmap video from a static base image.

        :param base_img: BGR uint8 numpy array
        :param points: time-series gaze points
        :param output_path: path to write the output .mp4
        :param duration_ms: total video duration in milliseconds
        :param fps: output frame rate (default 20)
        """
        h, w = base_img.shape[:2]
        frame_interval_ms = 1000.0 / fps
        snapped = self._snap_points(points, frame_interval_ms)
        n_frames = int(duration_ms / frame_interval_ms)

        writer = self._make_writer(str(output_path), fps, w, h)
        try:
            for i in range(n_frames):
                frame_time_ms = i * frame_interval_ms
                active = self._active_points(frame_time_ms, snapped, frame_interval_ms)
                frame = (
                    self._heatmapper.heatmap_on_img(active, base_img) if active else base_img.copy()
                )
                writer.write(frame)
        finally:
            writer.release()

    def heatmap_on_image_path(
        self,
        img_path: str | Path,
        points: VideoPointList,
        output_path: str | Path,
        duration_ms: float,
        fps: float = 20.0,
    ) -> None:
        img: NDArray[np.uint8] | None = cv2.imread(str(img_path))  # type: ignore[assignment]
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        self.heatmap_on_image(img, points, output_path, duration_ms, fps)

    def heatmap_on_video(
        self,
        video_path: str | Path,
        points: VideoPointList,
        output_path: str | Path,
    ) -> None:
        """
        Render a heatmap video over an existing video.
        FPS is read from the source. Audio is preserved if present.

        :param video_path: path to the source video
        :param points: time-series gaze points
        :param output_path: path to write the output .mp4
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_interval_ms = 1000.0 / fps
        snapped = self._snap_points(points, frame_interval_ms)
        has_audio = self._check_audio(str(video_path))

        # if audio needs muxing, write video frames to a temp file first
        tmp_path: str | None = None
        video_out = str(output_path)
        if has_audio:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name
            video_out = tmp_path

        writer = self._make_writer(video_out, fps, w, h)
        try:
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_time_ms = frame_index * frame_interval_ms
                active = self._active_points(frame_time_ms, snapped, frame_interval_ms)
                if active:
                    frame = self._heatmapper.heatmap_on_img(active, frame)  # type: ignore[arg-type]
                writer.write(frame)
                frame_index += 1
        finally:
            cap.release()
            writer.release()

        if has_audio and tmp_path:
            self._mux_audio(str(video_path), tmp_path, str(output_path))
            Path(tmp_path).unlink()

    # ----------------------------------------------------------------- private

    @staticmethod
    def _snap_points(points: VideoPointList, frame_interval_ms: float) -> VideoPointList:
        """Snap each point's timestamp to the nearest frame boundary."""
        return [
            p.model_copy(update={"t": round(p.t / frame_interval_ms) * frame_interval_ms})
            for p in points
        ]

    def _active_points(
        self,
        frame_time_ms: float,
        points: VideoPointList,
        frame_interval_ms: float,
    ) -> PointList:
        """Return Points with effective strengths for the given frame time."""
        active: PointList = []
        for vp in points:
            weight = self._compute_weight(frame_time_ms - vp.t, frame_interval_ms)
            if weight is None or weight <= 0:
                continue
            base_strength = vp.strength if vp.strength is not None else 1.0
            active.append(
                Point(
                    x=vp.x,
                    y=vp.y,
                    diameter=vp.diameter,
                    diameter_pct=vp.diameter_pct,
                    strength=min(1.0, base_strength * weight),
                    sigma=vp.sigma,
                )
            )
        return active

    def _compute_weight(self, age_ms: float, frame_interval_ms: float) -> float | None:
        """
        Returns intensity weight (0–1) for a point of the given age.
        Returns None if the point should not appear in this frame.
        """
        if age_ms < 0:
            return None

        if self.decay_time_ms is None:
            return 1.0 if age_ms < frame_interval_ms else None

        if age_ms > self.decay_time_ms:
            return None

        if self.smooth_decay:
            return 1.0 - (age_ms / self.decay_time_ms)

        return 1.0

    @staticmethod
    def _make_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for: {path}")
        return writer

    @staticmethod
    def _check_audio(video_path: str) -> bool:
        try:
            probe = ffmpeg.probe(video_path)
            return any(s["codec_type"] == "audio" for s in probe["streams"])
        except ffmpeg.Error:
            return False

    @staticmethod
    def _mux_audio(source_video: str, silent_video: str, output_path: str) -> None:
        video_in = ffmpeg.input(silent_video)
        audio_in = ffmpeg.input(source_video).audio
        (
            ffmpeg.output(
                video_in,
                audio_in,
                output_path,
                vcodec="copy",
                acodec="aac",
                shortest=None,
            )
            .overwrite_output()
            .run(quiet=True)
        )
