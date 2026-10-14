"""
One perspective slice of a panorama.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


# Compared by identity: a view holds an image, and two views are the same
# only if they are the same slice.
@dataclass(eq=False)
class View:
    image: np.ndarray  # the slice itself, BGR
    width: int
    height: int
    yaw: float
    pitch: float
    hfov: float
    vfov: float
    focal_px: float

    # --- Identification and Grouping ---
    pano_id: int | str = 0  # To group slices from the same panorama

    depth: Optional[np.ndarray] = None

    def rotation(self):
        """This view's rotation within its pano (camera ray -> pano ray): yaw
        about y, then pitch about the turned x -- intrinsic 'YX', exactly how
        Equirectangular.GetPerspective cuts it. (Extrinsic 'yx' agreed only at
        pitch 0.)"""
        from scipy.spatial.transform import Rotation
        return Rotation.from_euler('YX', [self.yaw, self.pitch], degrees=True).as_matrix()
