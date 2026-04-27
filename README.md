# heatmappy

Draw image and video heatmaps in Python.

Given a set of 2D points and an image or video, heatmappy overlays a colour or reveal heatmap showing the density or intensity of those points. The points can be anything — gaze coordinates, click events, sensor readings, geographic locations, or any other spatial data.

---

## Versions

This repository contains two independent versions. Each has its own README, install instructions, and examples.

| Version | Folder | Package | Status |
|---------|--------|---------|--------|
| v1 | [`heatmappy_v1/`](heatmappy_v1/) | `heatmappy` | Legacy — uses moviepy and Pillow |
| v2 | [`heatmappy_v2/`](heatmappy_v2/) | `heatmappy2` | Current — uses OpenCV, fully typed |

**New users should start with v2.** v1 is kept for backwards compatibility with existing integrations.

### What changed in v2

- Replaced moviepy with OpenCV and ffmpeg-python — faster, no ffmpeg version conflicts
- Full type annotations, validated with mypy strict
- Pydantic v2 models for `Point` and `VideoPoint` with per-point diameter/strength/sigma overrides
- Three heatmap modes: `colour`, `reveal`, and `pair` (side-by-side)
- Three normalisation modes: `relative`, `absolute`, `raw`
- Three decay modes for video: none, hard cutoff, smooth linear fade
- Time windowing: render any sub-range of the point or video timeline
- Gaussian kernel caching for efficient multi-frame rendering
- pytest test suite

---

## License

MIT License.
