import numpy as np
import cv2
import os

# DA3's own conf output is `1 + exp(x)` (unbounded, not a [0,1] probability,
# not calibrated across scenes) -- see model/utils/head_utils.py's
# "expp1" activation. Matches DA3's own reference export (utils/export/glb.py
# get_conf_thresh) exactly now -- same floor, same lower/upper percentile
# (40/90) -- so a uniformly low-confidence view still gets filtered
# (floor) but a call never discards everything (upper clamp).
CONF_ABS_FLOOR = 1.05
CONF_LOWER_PERCENTILE = 40.0
CONF_UPPER_PERCENTILE = 90.0


def _wedge_bounds_per_pano(views: list) -> dict:
    """For each pano's surviving views, the angular wedge (yaw degrees,
    relative to that view's OWN yaw) it "owns" in the final point cloud --
    a circular Voronoi split of the pano's 360 degrees by view-center yaw,
    computed only from views actually present here (so a view dropped by
    the consensus filter isn't a factor: its neighbors' wedges just expand
    to cover the gap it left, automatically).

    Adjacent views are typically 20-30 degrees apart with a 90 degree HFOV,
    so without this, the same real-world stretch of wall/curb gets
    contributed by 2-3 different views' depth estimates that don't quite
    agree pixel-for-pixel -- inflating point count and adding a soft
    "doubled up" fuzziness. Trimming each view to only its own wedge (its
    least-distorted, most-central region, since perspective crops stretch
    more toward the edges) means every real-world direction comes from
    exactly one view.

    Only handles yaw (not pitch) -- every view this pipeline actually
    generates is sliced at pitch=0 (see extract_views_for_da3), so a 1D
    circular partition is sufficient; there's no multi-row case to cover.

    Returns {view_index: (lo_deg, hi_deg)}, bounds expressed relative to
    that view's own yaw, directly comparable to a per-pixel yaw offset
    computed from that same view's own camera rays -- no absolute-angle
    wraparound bookkeeping needed."""
    by_pano = {}
    for i, v in enumerate(views):
        by_pano.setdefault(v.pano_id, []).append((i, v.yaw % 360.0))

    bounds = {}
    for pano_id, entries in by_pano.items():
        entries.sort(key=lambda e: e[1])
        n = len(entries)
        for k in range(n):
            idx, yaw = entries[k]
            if n == 1:
                bounds[idx] = (-180.0, 180.0)
                continue
            prev_yaw = entries[k - 1][1] - (360.0 if k == 0 else 0.0)
            next_yaw = entries[(k + 1) % n][1] + (360.0 if k == n - 1 else 0.0)
            bounds[idx] = ((prev_yaw - yaw) / 2.0, (next_yaw - yaw) / 2.0)
    return bounds


def backproject_views_to_pcd(views: list, da3_result,
                             conf_lower_percentile: float = CONF_LOWER_PERCENTILE,
                             return_confidence: bool = False,
                             drop_mask=None):
    """
    Back-projects processed views into world space.
    Returns (all_pts, all_cols) combined, plus per_pano dicts
    {pano_id: pts} and {pano_id: colors}.

    `views` must be index-aligned with da3_result.prediction (i.e. the exact
    list DA3 was run on, not an arbitrary subset/reorder) — depth/pose/color
    are looked up positionally by enumerate(views).

    Each view only contributes points from its own angular wedge (see
    _wedge_bounds_per_pano) -- not its whole overlapping field of view.

    return_confidence: also return a 5th dict, {pano_id: confidences},
    DA3's own raw per-point confidence (same `1 + exp(x)` scale as the
    filter above) for every point THIS CALL ALREADY KEPT -- a point
    conf_lower_percentile dropped was never backprojected, so there is
    no confidence to hand back for it; this only ever describes points
    you already have. Lets a caller that kept more than it needs right
    now (a high conf_lower_percentile) trim further later by its own
    threshold, without asking DA3 to run again.

    drop_mask: optional callable, given every view's image path in one
    list, returning one boolean array per view (True = drop that pixel),
    at any resolution. For removing things like cars and people, decided by
    the caller's own model; this package stays model-agnostic.
    """
    all_points = []
    all_colors = []
    all_conf = [] if return_confidence else None
    per_pano_pts: dict[int, list] = {}
    per_pano_cols: dict[int, list] = {}
    per_pano_conf: dict[int, list] = {} if return_confidence else None

    pred = da3_result.prediction
    if pred is None:
        return (None, None, {}, {}) + (({},) if return_confidence else ())

    wedge_bounds = _wedge_bounds_per_pano(views)
    masks = drop_mask([v.path for v in views]) if drop_mask else None

    for i, v in enumerate(views):
        # 1. Geometry from DA3
        K = pred.intrinsics[i]
        depth = pred.depth[i]
        # Use the extrinsics we already snapped in DA3Model
        w2c = pred.extrinsics[i]
        conf = pred.conf[i] if pred.conf is not None else None

        h, w = depth.shape
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        pix = np.stack([us, vs, np.ones_like(us)], axis=-1).reshape(-1, 3)

        # Per-pixel yaw offset from this view's own optical axis (verified:
        # atan2(ray_x, ray_z) on K_inv @ pixel recovers the same yaw
        # convention extract_views_for_da3's THETA uses -- 0 at image
        # center, +/-HFOV/2 at the left/right edges), used to keep only
        # this view's own wedge.
        K_inv = np.linalg.inv(K)
        rays_full = (K_inv @ pix.T).T
        yaw_offset = np.degrees(np.arctan2(rays_full[:, 0], rays_full[:, 2])).reshape(h, w)
        lo, hi = wedge_bounds.get(i, (-180.0, 180.0))
        in_wedge = (yaw_offset >= lo) & (yaw_offset < hi)

        valid = np.isfinite(depth) & (depth > 0) & in_wedge
        if conf is not None:
            lower = np.percentile(conf, conf_lower_percentile)
            upper = np.percentile(conf, CONF_UPPER_PERCENTILE)
            conf_thr = min(max(CONF_ABS_FLOOR, lower), upper)
            valid &= conf >= conf_thr
        if masks is not None:
            m = masks[i]
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            valid &= ~m

        vidx = np.flatnonzero(valid.reshape(-1))
        if len(vidx) == 0:
            continue
        if return_confidence:
            v_conf = (conf.reshape(-1)[vidx] if conf is not None
                     else np.full(len(vidx), np.nan, dtype=np.float32))
            per_pano_conf.setdefault(v.pano_id, []).append(v_conf)

        # 2. Backproject to Camera Space
        rays = rays_full[vidx]
        pts_cam = rays * depth.flatten()[vidx][:, None]

        # 3. Transform to World Space (using C2W)
        w2c_homo = np.eye(4)
        w2c_homo[:3, :4] = w2c[:3, :4]
        c2w = np.linalg.inv(w2c_homo)

        pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]
        all_points.append(pts_world)
        per_pano_pts.setdefault(v.pano_id, []).append(pts_world)

        # 4. Colors
        if v.path and os.path.exists(v.path):
            img_bgr = cv2.imread(v.path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                if img_rgb.shape[:2] != (h, w):
                    img_rgb = cv2.resize(img_rgb, (w, h))
                cols = img_rgb.reshape(-1, 3)[vidx] / 255.0
                all_colors.append(cols)
                per_pano_cols.setdefault(v.pano_id, []).append(cols)

    if not all_points:
        return (None, None, {}, {}) + (({},) if return_confidence else ())
    consolidated_pts = {pid: np.concatenate(pts, axis=0) for pid, pts in per_pano_pts.items()}
    consolidated_cols = {pid: np.concatenate(cols, axis=0) for pid, cols in per_pano_cols.items()}
    out = (
        np.concatenate(all_points, axis=0),
        np.concatenate(all_colors, axis=0) if all_colors else None,
        consolidated_pts,
        consolidated_cols,
    )
    if return_confidence:
        consolidated_conf = {pid: np.concatenate(c, axis=0) for pid, c in per_pano_conf.items()}
        out += (consolidated_conf,)
    return out
