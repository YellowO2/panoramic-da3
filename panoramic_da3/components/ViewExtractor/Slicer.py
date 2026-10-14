"""
Cuts one panorama into horizon views for DA3.

16:9 slices, one every step_degrees, each hfov wide (90 reaches about 29
degrees above and below the horizon; wider reaches further).
"""

import math

from panoramic_da3.View import View

HFOV = 90.0  # Default FOV for DA3 slices


def _extract_slice(equ, yaw, pitch, hfov, w, h, pano_id, depth_equ=None) -> View:
    """Extract one perspective slice as a View."""
    img = equ.GetPerspective(hfov, yaw, pitch, h, w)

    focal_px = (w / 2.0) / math.tan(math.radians(hfov) / 2.0)
    vfov = math.degrees(2.0 * math.atan((h / 2.0) / focal_px))

    view = View(
        image=img,
        yaw=yaw,
        pitch=pitch,
        width=int(w),
        height=int(h),
        focal_px=focal_px,
        hfov=hfov,
        vfov=vfov,
        pano_id=pano_id,
    )
    if depth_equ is not None:
        view.depth = depth_equ.GetPerspective(hfov, yaw, pitch, h, w)
    return view


def Slicer(equ, pano_id, step_degrees=20, hfov=HFOV) -> list[View]:
    """`equ` is the panorama as an `Equirectangular`."""
    pano_w = equ._img.shape[1]

    slice_w = max(64, pano_w // 4)
    slice_h = int(slice_w * 9 / 16)

    views = []

    # Horizon views (pitch=0)
    yaw = 0.0
    while yaw < 360.0:
        views.append(_extract_slice(equ, yaw, 0, hfov, slice_w, slice_h, pano_id))
        yaw += step_degrees

    return views
