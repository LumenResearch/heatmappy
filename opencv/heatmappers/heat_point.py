import os.path
from enum import Enum
from abc import ABC, abstractmethod

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


class HeatPointImageGenerator:
    _initialized: bool = False

    circles_cache: dict = dict()

    cache_path: str

    @classmethod
    def initialize_class(cls, cache_path):
        cls.cache_path = os.path.join(cache_path, 'heat_point')
        os.makedirs(cls.cache_path, exist_ok=True)
        cls._initialized = True

    @classmethod
    @lr.lr_error_logger(logger)
    def get_circle(cls, diameter_px: int, std_px: int, strength_gry_level: int) -> np.ndarray:
        circle = CirclePoint(diameter_px=diameter_px, std_px=std_px, strength_gery_level=strength_gry_level)
        if circle.name not in cls.circles_cache:
            circle_img = cls._load_circle(circle)
            cls.circles_cache[circle.name] = circle_img

        return cls.circles_cache[circle.name].copy()

    @classmethod
    @lr.lr_require_initialization
    def _load_circle(cls, circle: CirclePoint) -> np.ndarray:
        fpath = os.path.join(cls.cache_path, f"{circle.name}.png")
        circle_img = cv2.imread(fpath)
        if circle_img is None:
            circle_img = cls._draw_circle(circle)
            cv2.imwrite(fpath, circle_img)

        return circle_img

    @classmethod
    def _draw_circle(cls, circle: CirclePoint) -> np.ndarray:

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

    @classmethod
    def _get_ellipse(cls, image, width, height, angle):
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


class HeatPoint(ABC, BaseModel):
    hp_type: HeatPointType
    center_x_px: int
    center_y_px: int
    strength_10_255: int
    image_generator: Type[HeatPointImageGenerator]
    scale: float = 1

    @property
    @abstractmethod
    def image(self) -> np.ndarray:
        """
        Returns the image corresponding to the point width/height, std, strength, and scale
        :param scale: return the circle at a different scale (e.g. if we need half sized images to speed up heat-mapping
        :return:
        """
        pass

    @property
    @abstractmethod
    def top_left_corner(self) -> Tuple[int, int]:
        pass

    @field_validator('image_generator')
    def check_image_generator(cls, v):
        if not issubclass(v, HeatPointImageGenerator):
            raise ValueError("image_generator must be HeatPointImageGenerator or a subclass of it")
        return v


class HeatCircle(HeatPoint):
    hp_type: HeatPointType = HeatPointType.circle
    diameter_px: int
    color_decay_std_px: int

    @property
    def image(self) -> np.ndarray:
        """
        Returns the image corresponding to the circle diameter, std, strength, and scale
        :param scale: return the circle at a different scale (e.g. if we need half sized images to speed up heat-mapping
        :return:
        """
        if self.scale == 1:
            return self.image_generator.get_circle(self.diameter_px, self.color_decay_std_px, self.strength_10_255)
        else:
            return self.image_generator.get_circle(
                int(self.diameter_px * self.scale),
                int(self.color_decay_std_px * self.scale),
                int(self.strength_10_255 * self.scale)
            )

    @property
    def top_left_corner(self):
        return self.center_x_px - self.diameter_px // 2, self.center_y_px - self.diameter_px // 2


if __name__ == '__main__':
    from time import time, sleep

    from opencv.configs import Config


    def example_circle(hig):
        hp = HeatCircle(
            center_x_px=10,
            center_y_px=10,
            strength_10_255=255,
            image_generator=HeatPointImageGenerator,
            diameter_px=500,
            color_decay_std_px=100,
            scale=2
        )
        print(hp.model_dump())
        return hp


    cfg = Config()

    # Define the diameter and standard deviation (std) for the gradient
    diameter = 400  # Adjust as needed
    std = 100  # Adjust as needed
    strength = 50

    HeatPointImageGenerator.initialize_class(cache_path=cfg.cache_folder)

    # first time either draws the image and saves it to file or reads from file
    tik = time()
    hp = example_circle(HeatPointImageGenerator)
    print("draw or load from file", (time() - tik) * 1000, "ms")
    # Display the image
    cv2.imshow("Gradient Circle", hp.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # sleep for 1 second so all parallel tasks are finished so timing is more accurate
    sleep(1)

    # second time load from memory
    tik = time()
    hp = example_circle(HeatPointImageGenerator)
    print("from cache", (time() - tik) * 1000, "ms")
    # Display the image
    cv2.imshow("Gradient Circle", hp.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
