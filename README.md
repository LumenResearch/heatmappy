# heatmap.py
Draw image heatmaps in python

# Install

Soon to be available through pip.

# Requirements

- matplotlib
- numpy
- Pillow
- PySide (soon to be optional)

# Examples

Given some points (co-ordinates) and a base image:

```python
from heatmap import Heatmapper

from PIL import Image

example_points = [(100, 20), (120, 25), (200, 50), (60, 300), (170, 250)]
example_img_path = 'cat.jpg'
example_img = Image.open(example_img_path)
```

Draw a basic heatmap on the PIL image object:

```python
heatmapper = Heatmapper(colours='reveal')
heatmap = heatmapper.heatmap_on_img(example_points, example_img)
heatmap.save('heatmap.png')
```
![default cat](/examples/default-cat.png?raw=true)

Draw a reveal heatmap, given the image path:

```python
heatmapper = Heatmapper(opacity=0.9, colours='reveal')
heatmap = heatmapper.heatmap_on_base_path(example_points, example_img_path)
heatmap.save('heatmap.png')
```
![reveal cat](/examples/reveal-cat.png?raw=true)

# Heatmap config

The following options are available (shown with their default values):

```python
heatmapper = Heatmapper(
    point_diameter=50,  # the size of each point to be drawn
    alpha_strength=0.2,  # the strength, between 0 and 1, of each point to be drawn
    opacity=0.65,  # the opacity of the heatmap layer
    colours='default',  # 'default' or 'reveal'
                        # OR a matplotlib LinearSegmentedColorMap object 
                        # OR the path to a horizontal scale image (like that found at src/default.png)
    grey_heatmapper='PySide'  # The object responsible for drawing the points
                              # Currently only a PySide implementation is available
)
```


# Coming soon

- Alternatives to PySide for drawing points
- Video heatmaps


# License

MIT License.
