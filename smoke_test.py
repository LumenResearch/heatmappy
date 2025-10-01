import os
import random
from PIL import Image
from heatmappy import Heatmapper

# Use bundled asset
ASSET = os.path.join(os.path.dirname(__file__), 'heatmappy', 'assets', 'cat.jpg')


def main():
    base_img = Image.open(ASSET)
    w, h = base_img.size

    # Generate random points
    rng = random.Random(42)
    points = [(rng.randint(0, w - 1), rng.randint(0, h - 1)) for _ in range(300)]

    heatmapper = Heatmapper(point_diameter=40, point_strength=0.35, colours='default')
    out = heatmapper.heatmap_on_img(points, base_img)
    out_path = os.path.join(os.path.dirname(__file__), 'smoke_out.png')
    out.save(out_path)
    print('Wrote', out_path)


if __name__ == '__main__':
    main()

