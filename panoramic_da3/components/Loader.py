"""
Reads panorama files into images, keyed by file name.

The only place this package reads one.
"""

import os

import cv2
import numpy as np


def Loader(paths: list[str]) -> dict[str, np.ndarray]:
    """{file name: BGR image}, in the given order. The file name is the pano's
    id everywhere downstream."""
    panoramas = {}
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot read a panorama from {path}")
        panoramas[os.path.basename(path)] = image
    return panoramas
