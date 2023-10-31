import os.path

import cv2
import numpy as np
from pydantic import BaseModel

try:
    from .. import lr
except ImportError:
    from opencv import lr

logger = lr.setup_logger()


class HeatPoint(BaseModel):
    circles_cache: dict = dict()

    cache_path: str

    def __init__(self, cache_path):
        cache_path = os.path.join(cache_path, 'heat_point')
        os.makedirs(cache_path, exist_ok=True)
        super().__init__(cache_path=cache_path)

    @lr.lr_error_logger(logger)
    @lr.lr_timer(logger)
    def get_circle(self, diameter: int, std: int) -> np.ndarray:
        k = (diameter, std)
        if not k in self.circles_cache:
            circle = self._load_circle(*k)
            self.circles_cache[k] = circle

        return self.circles_cache[k].copy()

    @lr.lr_timer(logger)
    def _load_circle(self, diameter: int, std: int) -> np.ndarray:
        fpath = os.path.join(self.cache_path, f"circle_{diameter}_{std}.png")
        circle = cv2.imread(fpath)
        if circle is None:
            circle = self._draw_circle(diameter, std)
            cv2.imwrite(fpath, circle)

        return circle

    @lr.lr_timer(logger)
    def _draw_circle(celf, diameter: int, std: int) -> np.ndarray:
        # Create a grayscale image with the specified dimensions
        image_size = (diameter, diameter)
        image = np.zeros(image_size, dtype=np.uint8)

        # Calculate the center of the image
        center = (diameter // 2, diameter // 2)

        # Loop through each pixel in the image
        for y in range(image_size[1]):
            for x in range(image_size[0]):
                # Calculate the distance from the current pixel to the center
                distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

                # Calculate the whiteness (pixel intensity) based on the distance and std
                whiteness = 255 * np.exp(-0.5 * (distance / std) ** 2)

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

        hp = HeatPoint(cache_path=cfg.cache_folder)

        gradient_circle = hp.get_circle(diameter, std)

        # Display the image
        cv2.imshow("Gradient Circle", gradient_circle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optionally, save the image to a file
        # cv2.imwrite("gradient_circle.png", gradient_circle)


    example_circle()
    example_circle()
    # example_ellipse()
