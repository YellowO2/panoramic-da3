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
