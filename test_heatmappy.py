__author__ = 'navid'
from heatmappy import Heatmapper

from PIL import Image

example_points = [(100, 20), (120, 25), (200, 50), (60, 300), (170, 250)]
example_img_path = 'heatmappy/assets/cat.jpg'
example_img = Image.open(example_img_path)
heatmapper = Heatmapper()
heatmap = heatmapper.heatmap_on_img(example_points, example_img)
heatmap.save('heatmap.png')