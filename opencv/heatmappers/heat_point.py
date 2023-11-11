import os.path
from enum import Enum
from abc import ABC, abstractmethod
from functools import lru_cache


import cv2
import numpy as np
from pydantic import BaseModel, PrivateAttr, field_validator
from typing import Optional, Tuple, Type

try:
    from .. import lr
except ImportError:
    from opencv import lr

logger = lr.setup_logger()


class HeatPointType(Enum):
    circle: str = "circle"
    ellipse: str = "ellipse"


class HeatPoint(ABC, BaseModel):
    hp_type: HeatPointType
    center_x_px: int
    center_y_px: int
    color_decay_std_px: int
    strength_10_255: int

    @abstractmethod
    def image(self, scale: float = 1) -> np.ndarray:
        """
        Returns the image corresponding to the point width/height, std, strength, and scale
        :param scale: return the circle at a different scale (e.g. if we need half sized images to speed up heat-mapping
        :return:
        """
        pass


class HeatCircle(HeatPoint):
    hp_type: HeatPointType = HeatPointType.circle
    diameter_px: int

    def image(self, scale: float = 1) -> np.ndarray:
        """
        Returns the image corresponding to the circle diameter, std, strength, and scale
        :param scale: return the circle at a different scale (e.g. if we need half sized images to speed up heat-mapping
        :return:
        """
        if scale == 1:
            return self._draw(self.diameter_px, self.color_decay_std_px, self.strength_10_255)
        else:
            return self._draw(int(self.diameter_px * scale),
                              int(self.color_decay_std_px * scale),
                              int(self.strength_10_255))

    @staticmethod
    @lru_cache(maxsize=None)
    def _draw(diameter_px: int, decay_std_px: int, strength_gery_level: int) -> np.ndarray:
        new_strength = max(10, min(255, strength_gery_level))  # limit strength between 10 and 255
        if strength_gery_level < 10 or strength_gery_level > 255:
            logger.warn(f"Strength of heat point={strength_gery_level} which is out of range of 10-255. "
                        f"Snapping it to {new_strength}")
        strength_gery_level = new_strength

        # Create a grayscale image with the specified dimensions
        image_size = (diameter_px, diameter_px)
        image = np.zeros(image_size, dtype=np.uint8)

        # Calculate the center of the image
        center = (diameter_px // 2, diameter_px // 2)

        # Loop through each pixel in the image
        for y in range(image_size[1]):
            for x in range(image_size[0]):
                # Calculate the distance from the current pixel to the center
                distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

                # if circle.std_px > 0:
                if decay_std_px > 0:
                    # Calculate the whiteness (pixel intensity) based on the distance and std
                    if distance > diameter_px // 2:
                        whiteness = 0
                    else:
                        whiteness = strength_gery_level * np.exp(-0.5 * (distance / decay_std_px) ** 2)
                else:
                    if distance > diameter_px // 2:
                        whiteness = 0
                    else:
                        whiteness = (
                                strength_gery_level -
                                (strength_gery_level - decay_std_px) * min(1, distance / (
                                    diameter_px // 2)))

                # Set the pixel intensity in the image
                image[y, x] = int(whiteness)

        return image


if __name__ == '__main__':
    from time import time, sleep

    def example_circle():
        hp = HeatCircle(
            center_x_px=10,
            center_y_px=10,
            strength_10_255=150,
            diameter_px=400,
            color_decay_std_px=300,
        )
        print(hp.model_dump())
        return hp

    for i in range(10):
        # first time either draws the image and saves it to file or reads from file
        tik = time()
        hp = example_circle()
        print("draw or load from file", (time() - tik) * 1000.0, "ms")
        # Display the image
        cv2.imshow("Gradient Circle scale 2", hp.image(scale=2))
        cv2.waitKey(15)

    # sleep for 1 second so all parallel tasks are finished so timing is more accurate
    sleep(1)

    # second time load from memory
    tik = time()
    hp = example_circle()
    print("from cache", (time() - tik) * 1000.0, "ms")
    # Display the image
    cv2.imshow("Gradient Circle scale 1", hp.image(scale=1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
