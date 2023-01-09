from collections import defaultdict
import random
from functools import lru_cache

from moviepy.editor import *
import numpy as np
from PIL import Image
from moviepy.video.VideoClip import DataVideoClip

from heatmappy import Heatmapper


class VideoHeatmapper:
    def __init__(self, img_heatmapper):
        self.img_heatmapper = img_heatmapper

    def heatmap_on_video(self, base_video, points,
                         heat_fps=20,
                         keep_heat=False,
                         heat_decay_s=None,
                         use_lazy_evaluation=True):
        width, height = base_video.size

        frame_points = self._frame_points(
            points,
            fps=heat_fps,
            keep_heat=keep_heat,
            heat_decay_s=heat_decay_s
        )

        if use_lazy_evaluation:
            heatmap_clips = self._lazy_heatmap_clips(width, height, frame_points, heat_fps)
        else:
            heatmap_frames = self._heatmap_frames(width, height, frame_points)
            heatmap_clips = self._heatmap_clips(heatmap_frames, heat_fps)

        return CompositeVideoClip([base_video] + list(heatmap_clips))

    def heatmap_on_video_path(self, video_path, points, heat_fps=20):
        base = VideoFileClip(video_path)
        return self.heatmap_on_video(base, points, heat_fps)

    def heatmap_on_image(self, base_img, points,
                         heat_fps=20,
                         duration_s=None,
                         keep_heat=False,
                         heat_decay_s=None):
        base_img = np.array(base_img)
        points = list(points)
        if not duration_s:
            duration_s = max(t for x, y, t in points) / 1000
        base_video = ImageClip(base_img).set_duration(duration_s)

        return self.heatmap_on_video(
            base_video, points,
            heat_fps=heat_fps,
            keep_heat=keep_heat,
            heat_decay_s=heat_decay_s
        )

    def heatmap_on_image_path(self, base_img_path, points,
                              heat_fps=20,
                              duration_s=None,
                              keep_heat=False,
                              heat_decay_s=None):
        base_img = Image.open(base_img_path)
        return self.heatmap_on_image(
            base_img, points,
            heat_fps=heat_fps,
            duration_s=duration_s,
            keep_heat=keep_heat,
            heat_decay_s=heat_decay_s
        )

    @staticmethod
    def _frame_points(pts, fps, keep_heat=False, heat_decay_s=None):
        interval = 1000 // fps
        frames = defaultdict(list)

        if not keep_heat:
            for x, y, t in pts:
                start = (t // interval) * interval
                frames[start].append((x, y))

            return frames

        pts = list(pts)
        last_interval = max(t for x, y, t in pts)

        for x, y, t in pts:
            start = (t // interval) * interval
            pt_last_interval = int(start + heat_decay_s * 1000) if heat_decay_s else last_interval
            for frame_time in range(start, pt_last_interval + 1, interval):
                frames[frame_time].append((x, y))

        return frames

    @lru_cache(maxsize=8)
    def _heatmap_cache(self, width, height, index):
        return self.img_heatmapper.heatmap(width, height, self._frame_points[index])

    def _lazy_heatmap_clips(self, width, height, frame_points, fps):
        interval = 1000 // fps
        frame_starts, frame_points = zip(*sorted(frame_points.items(), key=lambda pair: pair[0]))
        self._frame_points = defaultdict(list)
        self._frame_points.update(enumerate(frame_points))
        clip_start_frame_index = 0
        for frame_index, frame_start in enumerate(frame_starts):
            clip_start_ms = frame_starts[clip_start_frame_index]
            frame_time_ms = frame_starts[frame_index]
            clip_duration_ms = frame_time_ms - clip_start_ms + 0.99 * interval
            clip_expected_duration_ms = interval * (frame_index - clip_start_frame_index + 1)
            if frame_index < len(frame_starts) - 1 and clip_duration_ms < clip_expected_duration_ms:
                continue
            clip_data = range(clip_start_frame_index, frame_index + 1)
            clip = DataVideoClip(clip_data,
                                 lambda x: np.array(self._heatmap_cache(width, height, x))[:, :, :3], fps)
            clip.mask = DataVideoClip(clip_data,
                                      lambda x: np.array(self._heatmap_cache(width, height, x))[:, :, 3] * (1 / 255),
                                      fps, ismask=True)
            clip_start_frame_index = frame_index
            yield clip.set_start(clip_start_ms / 1000)

    def _heatmap_frames(self, width, height, frame_points):
        for frame_start, points in frame_points.items():
            heatmap = self.img_heatmapper.heatmap(width, height, points)
            yield frame_start, np.array(heatmap)

    @staticmethod
    def _heatmap_clips(heatmap_frames, fps):
        interval = 1000 // fps
        for frame_start, heat in heatmap_frames:
            yield (ImageClip(heat)
                   .set_start(frame_start / 1000)
                   .set_duration(interval / 1000))


def _example_random_points():
    def rand_point(max_x, max_y, max_t):
        return random.randint(0, max_x), random.randint(0, max_y), random.randint(0, max_t)

    return (rand_point(720, 480, 40000) for _ in range(500))


def main():
    example_base_img = os.path.join('assets', 'cat.jpg')

    img_heatmapper = Heatmapper(colours='default', point_strength=0.6)
    video_heatmapper = VideoHeatmapper(img_heatmapper)

    heatmap_video = video_heatmapper.heatmap_on_image_path(
        base_img_path=example_base_img,
        points=_example_random_points(),
        duration_s=40,
        keep_heat=True
    )

    heatmap_video.write_videofile('out_on_image.mp4', bitrate="5000k", fps=24)


if __name__ == '__main__':
    main()
