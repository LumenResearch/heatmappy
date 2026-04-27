# heatmappy
Draw image and video heatmaps in python

### Image 

![newspaper heatmap](/examples/paper.png?raw=true)

### Video

![video heatmap](/examples/example.gif?raw=true)

# Install

`pip install heatmappy`

# Requirements

- matplotlib
- moviepy
- numpy
- Pillow
- PySide (optional: up to ~20% faster than Pillow alone)

# Examples

### Given some points (co-ordinates) and a base image

```python
from heatmappy import Heatmapper

from PIL import Image

example_points = [(100, 20), (120, 25), (200, 50), (60, 300), (170, 250)]
example_img_path = 'cat.jpg'
example_img = Image.open(example_img_path)
```

### Draw a basic heatmap on the PIL image object

```python
heatmapper = Heatmapper()
heatmap = heatmapper.heatmap_on_img(example_points, example_img)
heatmap.save('heatmap.png')
```
![default cat](/examples/default-cat.png?raw=true)

### Draw a reveal heatmap, given the image path

```python
heatmapper = Heatmapper(opacity=0.9, colours='reveal')
heatmap = heatmapper.heatmap_on_img_path(example_points, example_img_path)
heatmap.save('heatmap.png')
```
![reveal cat](/examples/reveal-cat.png?raw=true)

### Draw a video heatmap

Input points are in the form (x, y, t) where t is in milliseconds.

```python
example_vid = os.path.join('assets', 'some_video.mp4')
example_points = [(100, 100, 25), (112, 92, 67), (17, 100, 36)]

img_heatmapper = Heatmapper()
video_heatmapper = VideoHeatmapper(img_heatmapper)

heatmap_video = video_heatmapper.heatmap_on_video_path(
    video_path=example_vid,
    points=example_points
)

heatmap_video.write_videofile('out.mp4', bitrate="5000k", fps=24)
```

# Heatmap config

The following options are available (shown with their default values):

```python
heatmapper = Heatmapper(
    point_diameter=50,  # the size of each point to be drawn
    point_strength=0.2,  # the strength, between 0 and 1, of each point to be drawn
    opacity=0.65,  # the opacity of the heatmap layer
    colours='default',  # 'default' or 'reveal'
                        # OR a matplotlib LinearSegmentedColorMap object 
                        # OR the path to a horizontal scale image
    grey_heatmapper='PIL'  # The object responsible for drawing the points
                           # Pillow used by default, 'PySide' option available if installed
)

video_heatmapper = VideoHeatmapper(
    heatmapper  # the img heatmapper to use (like the heatmapper above, for example)
)
```

## Provided colour schemes

### default

![default colour scheme](/heatmappy/assets/default.png?raw=true)

### reveal

![reveal colour scheme](/heatmappy/assets/reveal.png?raw=true)


# Coming soon

- Can specify different point size for each point plotted.


# License

MIT License.
