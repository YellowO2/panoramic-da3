from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class View:
    path: str  # path to the view image
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


def view_rotation(v):
    """A view's rotation within its pano (camera ray -> pano ray): yaw about
    y, then pitch about the turned x -- intrinsic 'YX', exactly how
    Equirec2Perspec.GetPerspective cuts it. (Extrinsic 'yx' agreed only at
    pitch 0.)"""
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler('YX', [v.yaw, v.pitch], degrees=True).as_matrix()
