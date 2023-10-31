import os.path

import cv2
import numpy as np
from pydantic import BaseModel, PrivateAttr
from typing import Optional

try:
    from .. import lr
except ImportError:
    from opencv import lr

logger = lr.setup_logger()


class CirclePoint(BaseModel):
    diameter_px: int
    std_px: int
    strength_gery_level: int
    _name: Optional[str] = PrivateAttr(default=None)

    @property
    def name(self):
        if self._name is None:
            self._name = f"circle_{self.diameter_px}_{self.std_px}_{self.strength_gery_level}"
        return self._name


class HeatPoint(BaseModel):
    circles_cache: dict = dict()

    cache_path: str

    def __init__(self, cache_path):
        cache_path = os.path.join(cache_path, 'heat_point')
        os.makedirs(cache_path, exist_ok=True)
        super().__init__(cache_path=cache_path)

    @lr.lr_error_logger(logger)
    @lr.lr_timer(logger)
    def get_circle(self, diameter_px: int, std_px: int, strength_gry_level: int) -> np.ndarray:
        circle = CirclePoint(diameter_px=diameter_px, std_px=std_px, strength_gery_level=strength_gry_level)
        if circle.name not in self.circles_cache:
            circle_img = self._load_circle(circle)
            self.circles_cache[circle.name] = circle_img

        return self.circles_cache[circle.name].copy()

    @lr.lr_timer(logger)
    def _load_circle(self, circle: CirclePoint) -> np.ndarray:
        fpath = os.path.join(self.cache_path, f"{circle.name}.png")
        circle_img = cv2.imread(fpath)
        if circle_img is None:
            circle_img = self._draw_circle(circle)
            cv2.imwrite(fpath, circle_img)

        return circle_img

    @lr.lr_timer(logger)
    def _draw_circle(celf, circle: CirclePoint) -> np.ndarray:

        # Correct the strength of the heatpoint
        new_strength = max(10, min(255, circle.strength_gery_level))  # limit strength between 10 and 255
        if circle.strength_gery_level < 10 or circle.strength_gery_level > 255:
            logger.warn(f"Strength of heat point={circle.strength_gery_level} which is out of range of 10-255. "
                        f"Snapping it to {new_strength}")
        circle.strength_gery_level = new_strength

        # Create a grayscale image with the specified dimensions
        image_size = (circle.diameter_px, circle.diameter_px)
        image = np.zeros(image_size, dtype=np.uint8)

        # Calculate the center of the image
        center = (circle.diameter_px // 2, circle.diameter_px // 2)

        # Loop through each pixel in the image
        for y in range(image_size[1]):
            for x in range(image_size[0]):
                # Calculate the distance from the current pixel to the center
                distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

                # Calculate the whiteness (pixel intensity) based on the distance and std
                whiteness = circle.strength_gery_level * np.exp(-0.5 * (distance / circle.std_px) ** 2)

                # Set the pixel intensity in the image
                image[y, x] = int(whiteness)

        return image

    def _get_ellipse(self, image, width, height, angle):
        # image = self._get_circle()
        #
        # # Resize the image to the specified width and height
        # resized_image = cv2.resize(image, (width, height))
        #
        # # Rotate the resized image by the specified angle
        # rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
        # rotated_image = cv2.warpAffine(resized_image, rotation_matrix, (width, height))
        #
        # return rotated_image
        raise NotImplementedError


if __name__ == '__main__':
    def example_circle():
        from opencv.configs import Config
        cfg = Config()

        # Define the diameter and standard deviation (std) for the gradient
        diameter = 400  # Adjust as needed
        std = 100  # Adjust as needed
        strength = 50

        hp = HeatPoint(cache_path=cfg.cache_folder)

        gradient_circle = hp.get_circle(diameter, std, strength)

        # Display the image
        cv2.imshow("Gradient Circle", gradient_circle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optionally, save the image to a file
        # cv2.imwrite("gradient_circle.png", gradient_circle)


    example_circle()
    example_circle()
    # example_ellipse()
