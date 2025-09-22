import cv2
import numpy as np
from pathlib import Path


def is_color(img: np.ndarray) -> bool:
    if len(img.shape) == 2:
        return False

    if img.shape[2] != 3:
        return False

    return True


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))

    if is_color(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def save_image(path: Path, img: np.ndarray) -> None:
    if is_color(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(path), img)
