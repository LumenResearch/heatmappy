from typing import Tuple, List
from enum import Enum

import cv2
import numpy as np

try:
    from .. import lr
    from . import HeatPoint, CirclePoint, HeatPointImageGenerator
except ImportError:
    from opencv import lr
    from opencv.heatmappers import HeatPoint, HeatCircle, HeatPointImageGenerator

logger = lr.setup_logger()


class HeatImageNormalisationMethod(Enum):
    scale_0_255 = "scale_0_255"
    cut_off_at_255 = "cut_off_at_255"


class HeatImage:
    _resizing_warning_issued = False

    @staticmethod
    def overlay_on_background(
            background_image: np.ndarray,
            heat_image: np.ndarray,
            colormap: int = cv2.COLORMAP_JET,
            alpha: float = 0.4) -> np.ndarray:
        # Apply a colormap to the heatmap (optional)

        colored_heatmap = cv2.applyColorMap(heat_image, colormap)

        # # Keep the zeros in the heatmap uncolored
        mask = heat_image != 0
        # colored_heatmap[mask] = [0, 0, 0]  # Assuming you want to keep the zeros as black

        # Blend the heatmap with the original image
        # Adjust alpha (weight of the original image) and beta (weight of the heatmap) as needed
        beta = 1 - alpha  # Weight for the heatmap
        background_image[mask] = cv2.addWeighted(colored_heatmap, alpha, background_image, beta, 0)[mask]

        return background_image

    @staticmethod
    def normalise_heat_image(heat_image: np.ndarray, method: HeatImageNormalisationMethod):
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

    @classmethod
    def get_heat_image(cls,
                       width: int,
                       height: int,
                       heat_points: List[HeatPoint],
                       scale: float = 1.0,
                       return_original_scale=False
                       ) -> np.ndarray:

        # Generate a new heat image
        heat_img = np.zeros((int(height * scale), int(width * scale)))

        # Add heat points to the heat image
        for hp in heat_points:
            cls._add_point(heat_img, hp.image(scale=scale), (int(hp.center_x_px * scale), int(hp.center_y_px * scale)))

        # Rescale back to original size if requested
        if scale != 1 and return_original_scale:
            # Issue a warining if rescaling back is requested
            if return_original_scale and not cls._resizing_warning_issued:
                logger.warn("LR: returning to original scale takes longer as resizing an image is demanding")
                cls._resizing_warning_issued = True

            heat_img = cv2.resize(heat_img, (width, height))
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

    import random
    from time import time, sleep

    from opencv.configs import Config


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


    def gen_circles(hig, x, y):
        hp = HeatCircle(
            center_x_px=x,
            center_y_px=y,
            strength_10_255=40,
            image_generator=hig,
            diameter_px=200,
            color_decay_std_px=0)
        return hp


    image = cv2.imread("../assets/cat.jpg")
    image = cv2.resize(image, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LINEAR)

    # Generate 100 points
    points_coords = [
        generate_coordinates_random(100, 500, 100),
        generate_coordinates_line(10, 50),
        generate_single_point(100, 500)
    ][2]

    cfg = Config()
    HeatPointImageGenerator.initialize_class(cache_path=cfg.cache_folder)
    circles = [gen_circles(HeatPointImageGenerator, x, y) for x, y in points_coords]

    tik = time()
    for i in range(1):
        hi = HeatImage.get_heat_image(image.shape[1], image.shape[0], circles, scale=1)
        hi = HeatImage.normalise_heat_image(hi, method=HeatImageNormalisationMethod.cut_off_at_255)
        heated_image = HeatImage.overlay_on_background(image, hi)
    print(time() - tik)
    cv2.imshow("heat image", heated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Sleeping ===============================")
    # sleep(3)

    tik = time()
    for i in range(1):
        hi = HeatImage.get_heat_image(image.shape[1], image.shape[0], circles, scale=1, return_original_scale=True)
        hi = HeatImage.normalise_heat_image(hi, method=HeatImageNormalisationMethod.scale_0_255)
        heated_image = HeatImage.overlay_on_background(image, hi)
    print(time() - tik)
    cv2.imshow("heat image", heated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
