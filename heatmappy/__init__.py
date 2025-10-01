from heatmappy.heatmap import Heatmapper,\
                              GreyHeatMapper,\
                              PILGreyHeatmapper,\
                              PySideGreyHeatmapper

# Optional video support via moviepy
try:
    from .video import VideoHeatmapper  # type: ignore
except Exception:
    class VideoHeatmapper:  # type: ignore
        def __init__(self, *_, **__):
            raise ImportError(
                "VideoHeatmapper requires moviepy. Install it with: pip install moviepy"
            )

__all__ = [
    'Heatmapper', 'GreyHeatMapper', 'PILGreyHeatmapper', 'PySideGreyHeatmapper',
    'VideoHeatmapper'
]

__version__ = '0.3.0'
