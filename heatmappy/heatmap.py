from abc import ABCMeta, abstractmethod
from functools import partial
import io
import os
import random
import time
from profilehooks import profile

from matplotlib.colors import LinearSegmentedColormap
import numpy
from PIL import Image

try:
    from PySide import QtCore, QtGui
except ImportError:
    pass


_asset_file = partial(os.path.join, os.path.dirname(__file__), 'assets')


def _img_to_opacity(img, opacity):
        img = img.copy()
        alpha = img.split()[3]
        alpha = alpha.point(lambda p: int(p * opacity))
        img.putalpha(alpha)
        return img


class Heatmapper:
    def __init__(self, point_diameter=50, point_strength=0.2, opacity=0.65,
                 colours='default',
                 grey_heatmapper='PIL'):
        """
        :param opacity: opacity (between 0 and 1) of the generated heatmap overlay
        :param colours: Either 'default', 'reveal',
                        OR the path to horizontal image which will be converted to a scale
                        OR a matplotlib LinearSegmentedColorMap instance.
        :param grey_heatmapper: Required to draw points on an image as a greyscale
                                heatmap. If not using the default, this must be an object
                                which fulfils the GreyHeatmapper interface.
        """

        self.opacity = opacity

        self._colours = None
        self.colours = colours

        if grey_heatmapper == 'PIL':
            self.grey_heatmapper = PILGreyHeatmapper(point_diameter, point_strength)
        elif grey_heatmapper == 'PySide':
            self.grey_heatmapper = PySideGreyHeatmapper(point_diameter, point_strength)
        elif grey_heatmapper == "PILEllipse":
            self.grey_heatmapper = PILGreyEllipseHeatmapper(point_diameter, point_strength)
        elif grey_heatmapper == "PILNPEllipse":
            self.grey_heatmapper = PILNPGreyEllipseHeatmapper(point_diameter, point_strength)
        else:
            self.grey_heatmapper = grey_heatmapper

    @property
    def colours(self):
        return self._colours

    @colours.setter
    def colours(self, colours):
        self._colours = colours

        if isinstance(colours, LinearSegmentedColormap):
            self._cmap = colours
        else:
            files = {
                'default': _asset_file('default.png'),
                'reveal': _asset_file('reveal.png'),
            }
            scale_path = files.get(colours) or colours
            self._cmap = self._cmap_from_image_path(scale_path)

    @property
    def point_diameter(self):
        return self.grey_heatmapper.point_diameter

    @point_diameter.setter
    def point_diameter(self, point_diameter):
        self.grey_heatmapper.point_diameter = point_diameter

    @property
    def point_strength(self):
        return self.grey_heatmapper.point_strength

    @point_strength.setter
    def point_strength(self, point_strength):
        self.grey_heatmapper.point_strength = point_strength

    def heatmap(self, width, height, points, base_path=None, base_img=None):
        """
        :param points: sequence of tuples of (x, y), eg [(9, 20), (7, 3), (19, 12)]
        :return: If base_path of base_img provided, a heat map from the given points
                 is overlayed on the image. Otherwise, the heat map alone is returned
                 with a transparent background.
        """
        heatmap = self.grey_heatmapper.heatmap(width, height, points)
        heatmap = self._colourised(heatmap)
        heatmap = _img_to_opacity(heatmap, self.opacity)

        if not (base_path or base_img):
            return heatmap

        background = Image.open(base_path) if base_path else base_img
        return Image.alpha_composite(background.convert('RGBA'), heatmap)

    def heatmap_on_img_path(self, points, base_path):
        width, height = Image.open(base_path).size
        return self.heatmap(width, height, points, base_path=base_path)

    def heatmap_on_img(self, points, img):
        width, height = img.size
        return self.heatmap(width, height, points, base_img=img)

    def _colourised(self, img):
        """ maps values in greyscale image to colours """
        arr = numpy.array(img)
        rgba_img = self._cmap(arr, bytes=True)
        return Image.fromarray(rgba_img)

    @staticmethod
    def _cmap_from_image_path(img_path):
        img = Image.open(img_path)
        img = img.resize((256, img.height))
        colours = (img.getpixel((x, 0)) for x in range(256))
        colours = [(r/255, g/255, b/255, a/255) for (r, g, b, a) in colours]
        return LinearSegmentedColormap.from_list('from_image', colours)


class GreyHeatMapper(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, point_diameter, point_strength):
        self.point_diameter = point_diameter
        self.point_strength = point_strength

    @abstractmethod
    def heatmap(self, width, height, points):
        """
        :param points: sequence of tuples of (x, y), eg [(9, 20), (7, 3), (19, 12)]
        :return: a white image of size width x height with black areas painted at
                 the given points
        """
        pass


class PySideGreyHeatmapper(GreyHeatMapper):
    def __init__(self, point_diameter, point_strength):
        super().__init__(point_diameter, point_strength)
        self.point_strength = int(point_strength * 255)

    def heatmap(self, width, height, points):
        base_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)
        base_image.fill(QtGui.QColor(255, 255, 255, 255))

        self._paint_points(base_image, points)
        return self._qimage_to_pil_image(base_image).convert('L')

    def _paint_points(self, img, points):
        painter = QtGui.QPainter(img)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 0))
        pen.setWidth(0)
        painter.setPen(pen)

        for point in points:
            self._paint_point(painter, *point)
        painter.end()

    def _paint_point(self, painter, x, y):
        grad = QtGui.QRadialGradient(x, y, self.point_diameter/2)
        grad.setColorAt(0, QtGui.QColor(0, 0, 0, max(self.point_strength, 0)))
        grad.setColorAt(1, QtGui.QColor(0, 0, 0, 0))
        brush = QtGui.QBrush(grad)
        painter.setBrush(brush)
        painter.drawEllipse(
            x - self.point_diameter/2,
            y - self.point_diameter/2,
            self.point_diameter,
            self.point_diameter
        )

    @staticmethod
    def _qimage_to_pil_image(qimg):
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QIODevice.ReadWrite)
        qimg.save(buffer, "PNG")

        bytes_io = io.BytesIO()
        bytes_io.write(buffer.data().data())
        buffer.close()
        bytes_io.seek(0)
        return Image.open(bytes_io)


class PILGreyHeatmapper(GreyHeatMapper):
    def __init__(self, point_diameter, point_strength):
        super().__init__(point_diameter, point_strength)

    def heatmap(self, width, height, points):
        heat = Image.new('L', (width, height), color=255)

        dot = (Image.open(_asset_file('450pxdot.png')).copy()
                    .resize((self.point_diameter, self.point_diameter), resample=Image.ANTIALIAS))
        dot = _img_to_opacity(dot, self.point_strength)

        for x, y in points:
            x, y = int(x - self.point_diameter/2), int(y - self.point_diameter/2)
            heat.paste(dot, (x, y), dot)

        return heat


class PILGreyEllipseHeatmapper(GreyHeatMapper):
    def __init__(self, point_diameter, point_strength):
        super().__init__(point_diameter, point_strength)

    @profile
    def heatmap(self, width, height, points):
        heat = Image.new('L', (width, height), color=255)
        dot_template = Image.open(_asset_file('450pxdot.png')).copy()
        ellipse_canvas_template = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))

        for x, y, w, h, angle, grey_discount in points:
            ellipse = dot_template.copy().resize((w, h), resample=Image.ANTIALIAS)
            ellipse_canvas = ellipse_canvas_template.copy()
            ellipse_canvas.paste(ellipse, (int(ellipse_canvas.size[0]/2-ellipse.size[0]/2), int(ellipse_canvas.size[1]/2-ellipse.size[1]/2)), ellipse)
            ellipse_canvas = ellipse_canvas.rotate(angle)

            x, y = int(x - ellipse_canvas.size[0]/2), int(y - ellipse_canvas.size[1]/2)

            ellipse_canvas = _img_to_opacity(ellipse_canvas, self.point_strength)

            heat.paste(ellipse_canvas, (x, y), ellipse_canvas)

        return heat

class PILNPGreyEllipseHeatmapper(GreyHeatMapper):
    def __init__(self, point_diameter, point_strength):
        super().__init__(point_diameter, point_strength)

    @profile
    def heatmap(self, width, height, points):
        heat = Image.new('L', (width, height), color=255)
        dot_template = Image.open(_asset_file('450pxdot.png')).copy()
        ellipse_canvas_template = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))

        for x, y, w, h, angle, grey_discount in points:
            ellipse = dot_template.copy().resize((w, h), resample=Image.ANTIALIAS)
            ellipse_canvas = ellipse_canvas_template.copy()
            ellipse_canvas.paste(ellipse, (int(ellipse_canvas.size[0]/2-ellipse.size[0]/2), int(ellipse_canvas.size[1]/2-ellipse.size[1]/2)), ellipse)
            ellipse_canvas = ellipse_canvas.rotate(angle)

            x, y = int(x - ellipse_canvas.size[0]/2), int(y - ellipse_canvas.size[1]/2)

            ellipse_canvas = _img_to_opacity(ellipse_canvas, self.point_strength)

            heat.paste(ellipse_canvas, (x, y), ellipse_canvas)

        return heat


def reveal_example(example_img, example_points, sample_size):
    heatmapper = Heatmapper(colours='default')
    heatmapper.colours = 'reveal'
    img = heatmapper.heatmap_on_img(example_points, example_img)
    img.show()

    img.save("out_reveal_{}.png".format(sample_size))


def color_example(example_img, example_points, sample_size):
    heatmapper = Heatmapper(colours='default')
    img = heatmapper.heatmap_on_img(example_points, example_img)
    img.show()
    img.save("out_color_{}.png".format(sample_size))


def ellipse_example(example_img, example_points, sample_size):
    example_img = Image.open(_asset_file('cat.jpg'))
    heatmapper = Heatmapper(colours='default', point_strength=0.2 ,grey_heatmapper='PILEllipse')
    # heatmapper.colours = 'reveal'
    img = heatmapper.heatmap_on_img(example_points, example_img)
    img.show()
    img.save("out_ellipse_{}.png".format(sample_size))


def ellipse_np_example(example_img, example_points, sample_size):
    example_img = Image.open(_asset_file('cat.jpg'))
    heatmapper = Heatmapper(colours='default', point_strength=0.2, grey_heatmapper='PILNPEllipse')
    # heatmapper.colours = 'reveal'
    img = heatmapper.heatmap_on_img(example_points, example_img)
    img.show()
    img.save("out_ellipse_{}.png".format(sample_size))

if __name__ == '__main__':

    sample_size = 1500
    example_img = Image.open(_asset_file('cat.jpg'))
    randpoint = lambda max_x, max_y, max_w, max_h, max_angle, max_grey_discount: (random.randint(0, max_x), random.randint(0, max_y), random.randint(50, max_w), random.randint(50, max_h), random.randint(0, max_angle),random.randint(1, max_grey_discount))
    example_points_ellipse = [(randpoint(*example_img.size, max_w=300, max_h=300, max_angle=360, max_grey_discount=2)) for _ in range(sample_size)]
    example_points_reveal = [(x, y) for x, y, _, _, _, _ in example_points_ellipse]

    # reveal_example(example_img, example_points_reveal)
    # tic = time.time()
    # color_example(example_img, example_points_reveal, sample_size)
    # toc = time.time()
    # print(toc-tic)
    #
    # tic = time.time()
    # ellipse_example(example_img, example_points_ellipse, sample_size)
    # toc = time.time()
    # print(toc-tic)

    tic = time.time()
    ellipse_np_example(example_img, example_points_ellipse, sample_size)
    toc = time.time()
    print(toc-tic)


