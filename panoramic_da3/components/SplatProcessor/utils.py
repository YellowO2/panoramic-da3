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


def _view_axes_per_pano(views: list) -> dict:
    """For each view: its rotation within its pano, and the optical axes
    (in that pano's frame) of every view of the same pano present here.
    A view keeps only the pixels whose direction is nearer its own axis
    than any other view's -- so every direction comes from exactly one view.

    Adjacent views overlap a lot (90 degree slices 20-30 degrees apart), so
    without this the same stretch of wall is contributed by 2-3 views whose
    depths don't quite agree, doubling it up; each view's own share is also
    its least distorted, central part. Only views actually present count: a
    view dropped by the consensus filter leaves its share to its neighbours.
    Around the horizon this is the old yaw wedge exactly; it also splits
    tilted rings from the horizon ring.

    Returns {view_index: (R_local, axes (n x 3), own row in axes)}."""
    from panoramic_da3.datatype import view_rotation
    by_pano = {}
    for i, v in enumerate(views):
        by_pano.setdefault(v.pano_id, []).append(i)
    out = {}
    for idx in by_pano.values():
        rots = [view_rotation(views[i]) for i in idx]
        axes = np.array([r[:, 2] for r in rots])
        for k, i in enumerate(idx):
            out[i] = (rots[k], axes, k)
    return out


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

    Each view only contributes points in its own share of directions (see
    _view_axes_per_pano) -- not its whole overlapping field of view.

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

    view_axes = _view_axes_per_pano(views)
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

        K_inv = np.linalg.inv(K)
        rays_full = (K_inv @ pix.T).T
        R_local, axes, own = view_axes[i]
        dirs = rays_full @ R_local.T
        in_wedge = (np.argmax(dirs @ axes.T, axis=1) == own).reshape(h, w)

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
