from collections import defaultdict
import random

from moviepy.editor import *
import numpy as np

from heatmappy import Heatmapper


class VideoHeatmapper:
    def __init__(self, heatmapper):
        self.heatmapper = heatmapper

    def heatmap_on_video(self, video_path, points, heat_fps=25):
        base = VideoFileClip(video_path)
        width, height = base.size

        frame_points = self._frame_points(points, heat_fps)
        heatmap_frames = self._heatmap_frames(width, height, frame_points)
        heatmap_clips = self._heatmap_clips(heatmap_frames, heat_fps)

        return CompositeVideoClip([base] + list(heatmap_clips))

    @staticmethod
    def _frame_points(pts, fps):
        frames = defaultdict(list)
        for x, y, t in pts:
            start = (t // fps) * fps
            frames[start].append((x, y))
        return frames

    def _heatmap_frames(self, width, height, frame_points):
        for frame_start, points in frame_points.items():
            heatmap = self.heatmapper.heatmap(width, height, points)
            yield frame_start, np.array(heatmap)

    @staticmethod
    def _heatmap_clips(heatmap_frames, fps):
        for frame_start, heat in heatmap_frames:
            yield (ImageClip(heat)
                   .set_start(frame_start/1000)
                   .set_duration(fps/1000)
                   .set_fps(fps))


if __name__ == '__main__':
    def rand_point(max_x, max_y, max_t):
        return random.randint(0, max_x), random.randint(0, max_y), random.randint(0, max_t)

    example_points = (rand_point(720, 480, 4000) for _ in range(15000))
    example_vid = 'assets\SampleVideo_720x480_1mb.mp4'

    img_heatmapper = Heatmapper(colours='default')
    video_heatmapper = VideoHeatmapper(img_heatmapper)

    video = heatmap_video = video_heatmapper.heatmap_on_video(
        video_path=example_vid,
        points=example_points
    )

    video.write_videofile('out.mp4', bitrate="5000k", fps=24)

