__all__ = [
    "HeatColorMap",
    "HeatImageNormalisationMethod",
    "HeatImage"
]

from typing import Tuple, List
from enum import Enum

import cv2
import numpy as np

try:
    from .. import lr
    from . import HeatPoint, CirclePoint, HeatPointImageGenerator
except ImportError:
    from opencv import lr
    from opencv.heatmappers import HeatPoint, HeatCircle

logger = lr.setup_logger()


class HeatColorMap(Enum):
    autumn = cv2.COLORMAP_AUTUMN
    bone = cv2.COLORMAP_BONE
    jet = cv2.COLORMAP_JET
    winter = cv2.COLORMAP_WINTER
    rainbow = cv2.COLORMAP_RAINBOW
    ocean = cv2.COLORMAP_OCEAN
    summer = cv2.COLORMAP_SUMMER
    spring = cv2.COLORMAP_SPRING
    cool = cv2.COLORMAP_COOL
    hsv = cv2.COLORMAP_HSV
    pink = cv2.COLORMAP_PINK
    hot = cv2.COLORMAP_HOT
    parula = cv2.COLORMAP_PARULA
    magma = cv2.COLORMAP_MAGMA
    inferno = cv2.COLORMAP_INFERNO
    plasma = cv2.COLORMAP_PLASMA
    viridis = cv2.COLORMAP_VIRIDIS
    cividis = cv2.COLORMAP_CIVIDIS
    twilight = cv2.COLORMAP_TWILIGHT
    twilight_shifted = cv2.COLORMAP_TWILIGHT_SHIFTED
    turbo = cv2.COLORMAP_TURBO
    deepgreen = cv2.COLORMAP_DEEPGREEN


class HeatImageNormalisationMethod(Enum):
    scale_0_255 = "scale_0_255"
    cut_off_at_255 = "cut_off_at_255"


class HeatImage:
    _resizing_warning_issued = False

    @staticmethod
    def add_heatmap(
            background_image: np.ndarray,
            heat_points: List[HeatPoint],
            method: HeatImageNormalisationMethod = HeatImageNormalisationMethod.cut_off_at_255,
            color_map: HeatColorMap = HeatColorMap.jet,
            heat_transparency: float = 0.4
    ) -> Tuple[np.ndarray, np.ndarray]:
        heat_image = HeatImage._get_heat_image(background_image.shape[1], background_image.shape[0], heat_points)
        heat_image = HeatImage._normalise_heat_image(heat_image, method=method)
        heat_res = HeatImage._overlay_on_background(background_image, heat_image, color_map, heat_transparency)
        return heat_res, heat_image

    @staticmethod
    def _overlay_on_background(
            background_image: np.ndarray,
            heat_image: np.ndarray,
            colormap: HeatColorMap,
            alpha: float) -> np.ndarray:
        # Apply a colormap to the heatmap (optional)

        colored_heatmap = cv2.applyColorMap(heat_image, colormap.value)

        # # Keep the zeros in the heatmap uncolored
        mask = heat_image != 0
        # colored_heatmap[mask] = [0, 0, 0]  # Assuming you want to keep the zeros as black

        # Blend the heatmap with the original image
        # Adjust alpha (weight of the original image) and beta (weight of the heatmap) as needed
        beta = 1 - alpha  # Weight for the heatmap
        background_image[mask] = cv2.addWeighted(colored_heatmap, alpha, background_image, beta, 0)[mask]

        return background_image

    @staticmethod
    def _normalise_heat_image(heat_image: np.ndarray, method: HeatImageNormalisationMethod):
        if method is HeatImageNormalisationMethod.scale_0_255:
            # Normalize the array
            min_val = np.min(heat_image)
            max_val = np.max(heat_image)

            # Scale the array to 0-255
            normalized_array = (heat_image - min_val) / (max_val - min_val) * 255

            # Convert to uint8 type if this is for an image
            heat_image = normalized_array.astype(np.uint8)
            return heat_image
        elif method is HeatImageNormalisationMethod.cut_off_at_255:
            heat_image[heat_image > 255] = 255
            heat_image = heat_image.astype(np.uint8)
            return heat_image

    @staticmethod
    def _get_heat_image(
            width: int,
            height: int,
            heat_points: List[HeatPoint]) -> np.ndarray:

        # Generate a new heat image
        heat_img = np.zeros((int(height), int(width)))

        # Add heat points to the heat image
        for hp in heat_points:
            HeatImage._add_point(heat_img, hp.image(), (int(hp.center_x_px), int(hp.center_y_px)))

        return heat_img

    @staticmethod
    def _add_point(heat_img: np.ndarray, heat_point_img: np.ndarray, point: Tuple[int, int]):
        y, x = point

        img_height, img_width = heat_img.shape
        point_height, point_width = heat_point_img.shape

        # Calculate the top-left corner indices for slicing
        start_x = x - point_height // 2
        start_y = y - point_width // 2

        # Calculate the end indices while ensuring they are within the main array bounds
        end_x = max(min(start_x + point_height, img_height), 0)
        end_y = max(min(start_y + point_width, img_width), 0)

        # Adjust start indices for negative values
        start_x = max(start_x, 0)
        start_y = max(start_y, 0)

        # Calculate the corresponding indices in the small array
        small_start_x = max(0, -x + point_height // 2)
        small_start_y = max(0, -y + point_width // 2)
        small_end_x = small_start_x + (end_x - start_x)
        small_end_y = small_start_y + (end_y - start_y)

        # Add the small array to the main array at the specified region
        if start_x > img_width or start_y > img_height:
            return
        try:
            heat_img[start_x:end_x, start_y:end_y] += heat_point_img[small_start_x:small_end_x,
                                                      small_start_y:small_end_y]
        except:
            dbg = 1


if __name__ == '__main__':
    # Example
    import random
    from time import time

    def generate_coordinates_line(num_points, proximity):
        points = [(p * proximity, p * proximity) for p in range(num_points)]
        return points

    def generate_coordinates_random(num_points, max_value, proximity):
        # Generates the first point
        points = [(random.randint(0, max_value), random.randint(0, max_value))]

        for _ in range(num_points - 1):
            last_point = points[-1]

            # Generate a new point within proximity of the last point
            new_x = random.randint(max(last_point[0] - proximity, 0), min(last_point[0] + proximity, max_value))
            new_y = random.randint(max(last_point[1] - proximity, 0), min(last_point[1] + proximity, max_value))

            points.append((new_x, new_y))

        return points

    def generate_single_point(x, y):
        return [(x, y)]

    def gen_circles(x, y):
        return HeatCircle(
            center_x_px=x,
            center_y_px=y,
            strength_10_255=40,
            diameter_px=200,
            color_decay_std_px=0)

    image = cv2.imread("../assets/cat.jpg")
    image = cv2.resize(image, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LINEAR)

    # Generate 100 points
    points_coords = [
        generate_coordinates_random(100, 500, 100),
        generate_coordinates_line(10, 50),
        generate_single_point(100, 500)
    ][1]

    circles = [gen_circles(x, y) for x, y in points_coords]

    tik = time()
    for i in range(1):
        heated_image, _ = HeatImage.add_heatmap(image, circles)

        print(time() - tik)
        cv2.imshow("heat image", heated_image)
    cv2.waitKey(0)

    tik = time()
    for i in range(1):
        heated_image, _ = HeatImage.add_heatmap(image,
                                                circles,
                                                method=HeatImageNormalisationMethod.scale_0_255,
                                                color_map=HeatColorMap.cool,
                                                heat_transparency=0.9)

        print(time() - tik)
        cv2.imshow("heat image", heated_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
