"""
Runs DA3 on panoramas and returns one world-space point cloud.

This package makes no decisions about what counts as a "good" result, an
"edge," or a "rating" -- that's the caller's own domain logic, kept out on
purpose so it stays a thin, general DA3 wrapper. Reading the panoramas and
writing the points are the caller's too: see Loader and Saver.
"""

from panoramic_da3.components.DA3Model import DA3Model
from panoramic_da3.components.SplatProcessor import SplatProcessor
from panoramic_da3.components.ViewExtractor.main import ViewExtractor


def panoramic_da3(panoramas, cfg, da3=None):
    """THE core primitive this package exposes: run DA3 jointly on a batch of
    panoramas, {pano_id: BGR image}. Slices each into views, runs DA3's joint
    multi-view pose+depth inference, and backprojects to world-space points.
    Order doesn't matter to DA3 itself.

    cfg: a `Settings`.

    da3: reuse an already-loaded DA3Model instead of loading (and deleting)
    a fresh one -- for a caller making several of these calls in one GPU
    session, which would otherwise reload the model each time.

    Returns (point_clouds, da3_result):
      - point_clouds: {pano_id: PointCloud}, in this call's own arbitrary
        local frame. PointCloud.merge(point_clouds.values()) joins them.
      - da3_result: DA3Result -- each pano's pose, keep count and consensus
        quality, and the views that survived DA3's consensus filter.
    """
    owns_da3 = da3 is None
    if owns_da3:
        da3 = DA3Model(cfg.da3_model)
    views = ViewExtractor(panoramas, cfg)
    da3_result = da3.process_views(views, cfg.dist_thresh, cfg.angle_thresh)
    point_clouds = SplatProcessor(
        da3_result, cfg.conf_lower_percentile, cfg.conf_floor, cfg.return_confidence, cfg.drop_mask)
    if owns_da3:
        da3.release()
    return point_clouds, da3_result
