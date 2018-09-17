from abc import ABCMeta, abstractmethod
from functools import partial
import io
import os
import random
import time
from profilehooks import profile

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
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
        elif grey_heatmapper == "PILNPAreaDiscountEllipse":
            self.grey_heatmapper = PILNPAreaDiscountGreyEllipseHeatmapper(point_diameter, point_strength)
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
        arr = np.array(img)
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

        for x, y, w, h, angle in points:

            if w == 0 or h == 0:
                continue

            ellipse = dot_template.copy().resize((w, h), resample=Image.ANTIALIAS)
            ellipse_canvas = ellipse_canvas_template.copy()
            ellipse_canvas.paste(ellipse, (int(ellipse_canvas.size[0]/2-ellipse.size[0]/2), int(ellipse_canvas.size[1]/2-ellipse.size[1]/2)), ellipse)
            ellipse_canvas = ellipse_canvas.rotate(angle)

            x, y = int(x - ellipse_canvas.size[0]/2), int(y - ellipse_canvas.size[1]/2)

            ellipse_canvas = _img_to_opacity(ellipse_canvas, self.point_strength)

            heat.paste(ellipse_canvas, (x, y), ellipse_canvas)

            # heat.show()


        # heat.show()
        return heat

class PILNPGreyEllipseHeatmapper(GreyHeatMapper):
    def __init__(self, point_diameter, point_strength):
        super().__init__(point_diameter, point_strength)

    @profile
    def heatmap(self, width, height, points, area_discount=False, unit_area_radius=30):
        original_width = width
        original_height = height
        resize_factor = 4
        print("resize factor is: {}".format(resize_factor))

        width = int(width/resize_factor)
        height = int(height/resize_factor)

        heat = Image.new('L', (width, height), color=0)
        dot_template = Image.open(_asset_file('450pxdot.png')).copy()
        ellipse_canvas_template = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))

        heat_arr = np.asarray(heat, 'float')


        tic = time.time()
        i = 0
        previous_print = 0
        for x, y, w, h, angle in points:
            i += 1
            if (i > previous_print):
                print('procession gaze #{}, time: {}'.format(i, time.time() - tic))
                previous_print += 1000
                tic = time.time()



            x = int(x/resize_factor)
            y = int(y/resize_factor)
            w = int(w/resize_factor)
            h = int(h/resize_factor)

            if w == 0 or h == 0:
                continue

            ellipse = dot_template.copy().resize((w, h), resample=Image.ANTIALIAS)
            ellipse_canvas_rotate = ellipse_canvas_template.copy()
            ellipse_canvas_rotate.paste(ellipse, (int(ellipse_canvas_rotate.size[0]/2-ellipse.size[0]/2), int(ellipse_canvas_rotate.size[1]/2-ellipse.size[1]/2)), ellipse)
            ellipse_canvas_rotate = ellipse_canvas_rotate.rotate(angle)
            # ellipse_canvas_rotate.split()[-1].show()


            x, y = int(x - ellipse_canvas_rotate.size[0]/2), int(y - ellipse_canvas_rotate.size[1]/2)

            ellipse_canvas_translate = ellipse_canvas_template.copy()
            ellipse_canvas_translate.paste(ellipse_canvas_rotate, (x, y), ellipse_canvas_rotate)
            # ellipse_canvas_translate.split()[-1].show()

            ellipse_canvas_translate = _img_to_opacity(ellipse_canvas_translate, self.point_strength).split()[-1]
            # ellipse_canvas_translate.show()

            ellipse_canvas_arr = np.asarray(ellipse_canvas_translate, 'float')
            if area_discount:
                heat_arr *= (unit_area_radius**2)/(w*h)

            heat_arr += ellipse_canvas_arr

        heat_arr *= (255.0/heat_arr.max())
        heat_arr = 255 - heat_arr
        heat_arr = heat_arr.astype('uint8')
        heat = Image.fromarray(heat_arr)

        # turn back to original size
        heat = heat.resize((original_width, original_height), resample=Image.ANTIALIAS)
        # heat.show()
        return heat

class PILNPAreaDiscountGreyEllipseHeatmapper(GreyHeatMapper):
    def __init__(self, point_diameter, point_strength):
        super().__init__(point_diameter, point_strength)

    @profile
    def heatmap(self, width, height, points, area_discount=True, unit_area_radius=100):

        original_width = width
        original_height = height
        resize_factor = 4
        print("resize factor is: {}".format(resize_factor))

        width = int(width/resize_factor)
        height = int(height/resize_factor)

        heat = Image.new('L', (width, height), color=0)
        dot_template = Image.open(_asset_file('450pxdot.png')).copy()
        ellipse_canvas_template = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))

        heat_arr = np.asarray(heat, 'float')

        tic = time.time()
        i = 0
        previous_print = 0
        for x, y, w, h, angle in points:
            i += 1
            if (i > previous_print):
                print('procession gaze #{}, time: {}'.format(i, time.time() - tic))
                previous_print += 1000
                tic = time.time()

            # if (i>5000):
            #     continue

            x = int(x/resize_factor)
            y = int(y/resize_factor)
            w = int(w/resize_factor)
            h = int(h/resize_factor)

            if w == 0 or h == 0:
                continue

            min_hw = min(h,w)
            if min_hw < 100:
                w = int(w*100/min_hw)
                h = int(h*100/min_hw)

            ellipse = dot_template.copy().resize((w, h), resample=Image.ANTIALIAS)
            ellipse_canvas_rotate = ellipse_canvas_template.copy()
            ellipse_canvas_rotate.paste(ellipse, (int(ellipse_canvas_rotate.size[0]/2-ellipse.size[0]/2), int(ellipse_canvas_rotate.size[1]/2-ellipse.size[1]/2)), ellipse)
            ellipse_canvas_rotate = ellipse_canvas_rotate.rotate(angle)
            # ellipse_canvas_rotate.split()[-1].show()


            x, y = int(x - ellipse_canvas_rotate.size[0]/2), int(y - ellipse_canvas_rotate.size[1]/2)

            ellipse_canvas_translate = ellipse_canvas_template.copy()
            ellipse_canvas_translate.paste(ellipse_canvas_rotate, (x, y), ellipse_canvas_rotate)
            # ellipse_canvas_translate.split()[-1].show()

            ellipse_canvas_translate = _img_to_opacity(ellipse_canvas_translate, self.point_strength).split()[-1]
            # ellipse_canvas_translate.show()

            ellipse_canvas_arr = np.asarray(ellipse_canvas_translate, 'float')
            if area_discount:
                d = (unit_area_radius**2)/(max((unit_area_radius**2), (w*h)))
                # d=0.4
                ellipse_canvas_arr *= d

            heat_arr += ellipse_canvas_arr
            # a = Image.fromarray(ellipse_canvas_arr)
            # a.show()






        def rescaled_sigmoid(x, a=0.05):
            sig = 1/(1+np.exp(-a * x))
            return sig * 255

        heat_arr *= (255.0/heat_arr.max())
        heat_arr -= 128
        heat_arr = rescaled_sigmoid(heat_arr)

        # thresh = np.percentile(heat_arr, 99)
        # idx = heat_arr > thresh
        # heat_arr[idx] = 255

        heat_arr = 255 - heat_arr #invert colors
        heat_arr = heat_arr.astype('uint8')
        heat = Image.fromarray(heat_arr)
         # turn back to original size
        heat = heat.resize((original_width, original_height), resample=Image.ANTIALIAS)
        # heat.show()
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
    heatmapper = Heatmapper(colours='default', point_strength=0.1 ,grey_heatmapper='PILEllipse')
    img = heatmapper.heatmap_on_img(example_points, example_img)
    img.show()
    img.save("out_ellipse_{}.png".format(sample_size))


def ellipse_np_example(example_img, example_points, sample_size):
    heatmapper = Heatmapper(colours='default', point_strength=1, grey_heatmapper='PILNPEllipse')
    img = heatmapper.heatmap_on_img(example_points, example_img)
    img.show()
    img.save("out_ellipse_{}.png".format(sample_size))

def ellipse_np_areadiscount_example(example_img, example_points, sample_size):
    heatmapper = Heatmapper(colours='default', point_strength=1, grey_heatmapper='PILNPAreaDiscountEllipse')
    img = heatmapper.heatmap_on_img(example_points, example_img)
    img.show()
    img.save("out_ellipse_{}.png".format(sample_size))


if __name__ == '__main__':

    sample_size = 100
    example_img = Image.open(_asset_file('cat.jpg'))
    randpoint = lambda max_x, max_y, max_w, max_h, max_angle: (random.randint(0, max_x), random.randint(0, max_y), random.randint(50, max_w), random.randint(50, max_h), random.randint(0, max_angle))
    example_points_ellipse = [(randpoint(*example_img.size, max_w=300, max_h=300, max_angle=360)) for _ in range(sample_size)]
    example_points_reveal = [(x, y) for x, y, _, _, _ in example_points_ellipse]

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
    #
    # tic = time.time()
    # ellipse_np_example(example_img, example_points_ellipse, sample_size)
    # toc = time.time()
    # print(toc-tic)

    tic = time.time()
    ellipse_np_areadiscount_example(example_img, example_points_ellipse, sample_size)
    toc = time.time()
    print(toc-tic)


