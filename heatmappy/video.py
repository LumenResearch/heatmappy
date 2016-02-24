from collections import defaultdict
import os
import random

from moviepy.editor import *
import numpy as np

from heatmappy import Heatmapper


class VideoHeatmapper:
    def __init__(self, heatmapper):
        self.heatmapper = heatmapper

    def heatmap_on_video(self, base_video, points, heat_fps=15):
        width, height = base_video.size

        frame_points = self._frame_points(points, heat_fps)
        heatmap_frames = self._heatmap_frames(width, height, frame_points)
        heatmap_clips = self._heatmap_clips(heatmap_frames, heat_fps)

        return CompositeVideoClip([base_video] + list(heatmap_clips))

    def heatmap_on_video_path(self, video_path, points, heat_fps=15):
        base = VideoFileClip(video_path)
        return self.heatmap_on_video(base, points, heat_fps)

    @staticmethod
    def _frame_points(pts, fps):
        frames = defaultdict(list)
        interval = 1000 // fps
        for x, y, t in pts:
            start = (t // interval) * interval
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
                   .set_duration((1000/fps)/1000))


def _example_random_points():
    def rand_point(max_x, max_y, max_t):
        return random.randint(0, max_x), random.randint(0, max_y), random.randint(0, max_t)

    return (rand_point(720, 480, 4000) for _ in range(15000))


def main():
    example_vid = os.path.join('assets', 'SampleVideo_720x480_1mb.mp4')

    img_heatmapper = Heatmapper(colours='default', point_strength=0.6)
    video_heatmapper = VideoHeatmapper(img_heatmapper)

    heatmap_video = video_heatmapper.heatmap_on_video_path(
        video_path=example_vid,
        points=_example_random_points()
    )

    heatmap_video.write_videofile('out.mp4', bitrate="5000k", fps=24)

if __name__ == '__main__':
    main()

