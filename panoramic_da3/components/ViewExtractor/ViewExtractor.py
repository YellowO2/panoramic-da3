import os
import cv2
import math
from panoramic_da3.components.ViewExtractor import Equirec2Perspec as E2P
from panoramic_da3.datatype import View


def _extract_slice(
    equ, yaw, pitch, hfov, w, h, output_path, pano_id, depth_equ=None
) -> View:
    """Extract one perspective slice, save it, and return a View."""
    img = equ.GetPerspective(hfov, yaw, pitch, h, w)
    cv2.imwrite(output_path, img)

    focal_px = (w / 2.0) / math.tan(math.radians(hfov) / 2.0)
    vfov = math.degrees(2.0 * math.atan((h / 2.0) / focal_px))

    view = View(
        yaw=yaw,
        pitch=pitch,
        path=output_path,
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


HFOV = 90.0  # Default FOV for DA3 slices
RING_VIEWS = 6  # views in each tilted ring (see extract_views_for_da3)


def extract_views_for_da3(
    input_image, output_dir, step_degrees=20, prefix="", pano_id=0,
    hfov=HFOV, ring_pitches=(),
) -> list[View]:
    """Extracts views for Depth Anything 3: 16:9 slices around the horizon,
    one every step_degrees, each hfov wide (90 reaches about 29 degrees
    above and below the horizon; wider reaches further).

    ring_pitches: extra rings of RING_VIEWS slices tilted by these degrees
    (positive looks up, as Equirec2Perspec's PHI), for what the horizon
    ring cannot reach -- building tops, the road nearer the camera."""
    equ = E2P.Equirectangular(input_image)
    pano_w = equ._img.shape[1]

    slice_w = max(64, pano_w // 4)
    slice_h = int(slice_w * 9 / 16)

    views = []
    rings = [(0.0, step_degrees, 0.0)] + [(float(p), 360.0 / RING_VIEWS, 180.0 / RING_VIEWS)
                                          for p in ring_pitches]
    for pitch, step, start in rings:
        yaw = start
        while yaw < 360.0:
            filename = f"{prefix}da3_{int(round(yaw))}_{int(round(pitch))}.jpg"
            views.append(
                _extract_slice(equ, yaw, pitch, hfov, slice_w, slice_h,
                               os.path.join(output_dir, filename), pano_id)
            )
            yaw += step

    return views
