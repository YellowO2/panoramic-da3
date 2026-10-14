"""
One coloured point cloud, and DA3's confidence per point if asked for.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PointCloud:
    points: np.ndarray                     # (N, 3) world-space positions
    colours: np.ndarray                    # (N, 3) RGB in [0, 1]
    confidence: np.ndarray | None = None   # (N,) DA3's raw confidence, `1 + exp(x)`

    @staticmethod
    def merge(point_clouds) -> "PointCloud":
        """All of them as one, in order."""
        clouds = list(point_clouds)
        if not clouds:
            return PointCloud(np.empty((0, 3)), np.empty((0, 3)))
        confidence = (np.concatenate([c.confidence for c in clouds])
                      if all(c.confidence is not None for c in clouds) else None)
        return PointCloud(np.concatenate([c.points for c in clouds]),
                          np.concatenate([c.colours for c in clouds]), confidence)
