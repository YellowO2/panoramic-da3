import os
import time

import numpy as np
import torch

from panoramic_da3.components.DepthMapGenerator.DA3Model import DA3Model
from panoramic_da3.components.ViewExtractor.ViewExtractor import HFOV, extract_views_for_da3
from panoramic_da3.components.SplatProcessor.utils import (
    CONF_LOWER_PERCENTILE, backproject_views_to_pcd,
)
from panoramic_da3.components.Saver.Saver import Saver


def save_da3_pointcloud(points: np.ndarray, colors: np.ndarray, path: str) -> str:
    """Thin public wrapper over components.Saver -- for callers that need to
    save a raw point cloud without reaching into this package's internal
    components.* modules directly."""
    Saver.save_point_cloud(points, path, colors=colors)
    return path


def run_da3(
    target_depth_path: str,
    support_paths: list[str],
    cfg,
    views_base: str,
    da3: "DA3Model | None" = None,
    dist_thresh: float = 0.2,
    angle_thresh: float = 1,
    step_degrees: int = 20,
    conf_lower_percentile: float = CONF_LOWER_PERCENTILE,
    return_confidence: bool = False,
    drop_mask=None,
    hfov: float = HFOV,
    ring_pitches=(),
):
    """THE core primitive this package exposes: run DA3 jointly on a list
    of panos (target_depth_path plus any support_paths -- for a plain
    N-pano batch with no distinguished "target", just pass the whole list
    as target_depth_path=paths[0], support_paths=paths[1:], order doesn't
    matter to DA3 itself). Slices each pano into views, runs DA3's joint
    multi-view pose+depth inference, and backprojects to world-space
    points/colors. This package makes no decisions about what counts as a
    "good" result, an "edge," or a "rating" -- that's the caller's own
    domain logic (thresholds, pass/fail, which candidate wins), kept
    entirely out of this package on purpose so it stays a thin, general
    DA3 wrapper.

    cfg: anything with a `.da3_model` attribute (the model path/repo id) --
    a bare object, a caller's own richer config, whatever. This package
    doesn't define its own config class since the only thing it needs is
    that one path.

    Returns (filtered_views, da3_result, merged_pts, merged_cols,
    per_pano_pts, per_pano_cols):
      - filtered_views: views that survived DA3's consensus filter.
      - da3_result: DA3Result (pano_poses, pano_keep_counts,
        pano_avg_deviation, prediction) -- see DA3Model.py. Callers read
        pano_poses[pano_id]["center"/"rotation"] for a pose,
        pano_keep_counts[pano_id] for (kept, total) view counts, and
        pano_avg_deviation[pano_id] for consensus quality, all keyed by
        os.path.basename(path).
      - merged_pts, merged_cols: all panos' backprojected points/colors
        combined, in this call's own arbitrary local frame.
      - per_pano_pts, per_pano_cols: {os.path.basename(path): points/
        colors} -- the same points split by which pano they came from,
        for a caller that only wants one pano's own slice (e.g. to avoid
        re-adding points for an already-anchored pano).

    da3: reuse an already-loaded DA3Model instead of loading (and deleting)
    a fresh one -- for a caller making several of these calls in one GPU
    session, which would otherwise reload the model each time. Default
    None preserves the original behavior (load, use, delete) for every
    other caller.

    step_degrees: yaw spacing between slices (default 20 -- matches
    extract_views_for_da3's own default, i.e. 18 slices/pano at 90 HFOV,
    ~78% overlap between neighbors). Coarser values (e.g. 45 -> 8
    slices/pano) trade per-pano slice redundancy for a lower image count at
    the same viewpoint coverage -- exposed for experimenting with that
    tradeoff, not used by default anywhere.

    conf_lower_percentile: a pixel below this percentile of ITS OWN VIEW's
    confidence is dropped before backprojection, whatever its raw value --
    default 40 matches DA3's own reference export. Lower keeps more of a
    view's weaker pixels; a caller that wants to decide later what to keep
    rather than have DA3 decide now can push this down (0 keeps everything
    the absolute floor and wedge/depth validity checks allow).

    return_confidence: attach da3_result.pano_point_confidence --
    {pano_id: per-point confidence array}, index-aligned with
    per_pano_pts[pano_id] -- so a caller can trim further later by its
    own threshold without a second DA3 call. Only ever covers points
    already kept; does not recover anything conf_lower_percentile
    dropped. Off by default: this package's other two callers
    (panoramic-to-3dgs, da3-baseline-test) unpack run_da3's return by
    fixed position, so the attribute is added to da3_result rather than
    as a new return value, and stays empty unless asked for.

    drop_mask: pixels to leave out of the points, e.g. cars and people --
    see backproject_views_to_pcd. DA3 itself still sees the whole view.

    hfov, ring_pitches: how wide each slice is, and extra tilted rings --
    see extract_views_for_da3. The defaults are the horizon ring alone.
    """
    t_extract0 = time.monotonic()
    all_views = []
    for i, path in enumerate([target_depth_path, *support_paths]):
        da3_dir = os.path.join(views_base, f"views_pano_{i}_da3")
        os.makedirs(da3_dir, exist_ok=True)
        all_views.extend(extract_views_for_da3(path, da3_dir, prefix=f"pano_{i}_", pano_id=os.path.basename(path), step_degrees=step_degrees,
                                               hfov=hfov, ring_pitches=ring_pitches))
    t_extract = time.monotonic() - t_extract0

    owns_da3 = da3 is None
    if owns_da3:
        da3 = DA3Model(cfg.da3_model)
    t_infer0 = time.monotonic()
    filtered_views, da3_result = da3.process_views(all_views, dist_thresh=dist_thresh, angle_thresh=angle_thresh)
    t_infer = time.monotonic() - t_infer0
    t_backproject0 = time.monotonic()
    backprojected = backproject_views_to_pcd(
        filtered_views, da3_result, conf_lower_percentile=conf_lower_percentile,
        return_confidence=return_confidence, drop_mask=drop_mask,
    )
    merged_pts, merged_cols, per_pano_pts, per_pano_cols = backprojected[:4]
    da3_result.pano_point_confidence = backprojected[4] if return_confidence else {}
    t_backproject = time.monotonic() - t_backproject0
    print(f"[timing] run_da3: {len(all_views)} view(s) extracted in {t_extract:.2f}s, "
          f"DA3 inference in {t_infer:.2f}s, backproject in {t_backproject:.2f}s")
    if owns_da3:
        del da3
        torch.cuda.empty_cache()
    return filtered_views, da3_result, merged_pts, merged_cols, per_pano_pts, per_pano_cols
