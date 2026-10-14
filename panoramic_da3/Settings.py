"""
Everything that tunes panoramic_da3, and nothing that flows through it.
"""

from dataclasses import dataclass
from typing import Callable

from panoramic_da3.components.SplatProcessor import CONF_LOWER_PERCENTILE
from panoramic_da3.components.ViewExtractor.Slicer import HFOV


@dataclass
class Settings:
    # DA3's weights: a local snapshot path or a Hugging Face repo id.
    da3_model: str
    # Slicing: one slice every step_degrees of yaw, each hfov wide. 20 gives
    # 18 slices per pano at 90 HFOV, ~78% overlap between neighbours.
    step_degrees: int = 20
    hfov: float = HFOV
    # DA3's consensus filter: a view whose pose is further than this from its
    # pano's consensus is dropped.
    dist_thresh: float = 0.2
    angle_thresh: float = 1
    # A pixel below this percentile of its own view's confidence is dropped
    # before backprojection.
    conf_lower_percentile: float = CONF_LOWER_PERCENTILE
    # Give each PointCloud DA3's confidence for its points, so a caller can
    # trim further later without running DA3 again.
    return_confidence: bool = False
    # Pixels to leave out of the points, e.g. cars and people: given every
    # view's image, returns one boolean array per view (True = drop).
    drop_mask: Callable | None = None
