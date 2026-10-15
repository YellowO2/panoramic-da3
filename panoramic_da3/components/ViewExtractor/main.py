"""
Turns panoramas into the perspective views DA3 reads.
"""

from panoramic_da3.components.ViewExtractor.Equirectangular import Equirectangular
from panoramic_da3.components.ViewExtractor.Slicer import Slicer


def ViewExtractor(panoramas, settings):
    """{pano_id: BGR image} -> every pano's slices, in order."""
    views = []
    for pano_id, image in panoramas.items():
        panorama = Equirectangular(image)
        views.extend(Slicer(panorama, pano_id, settings.step_degrees, settings.hfov))
    return views
