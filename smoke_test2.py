import os
import random
from heatmappy import Heatmapper

ASSET = os.path.join(os.path.dirname(__file__), 'heatmappy', 'assets', 'cat.jpg')

rng = random.Random(7)
points = [(rng.randint(0, 449), rng.randint(0, 449)) for _ in range(200)]

hm = Heatmapper(point_diameter=30, point_strength=0.5, colours='reveal')
out = hm.heatmap_on_img_path(points, ASSET)
out_path = os.path.join(os.path.dirname(__file__), 'smoke_out2.png')
out.save(out_path)
print('Wrote', out_path)

