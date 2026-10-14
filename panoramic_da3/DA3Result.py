"""
What DA3Model found for each pano.
"""


class DA3Result:
    def __init__(self, pano_poses, prediction, pano_keep_counts=None, pano_avg_deviation=None, views=None):
        self.pano_poses = pano_poses # pano_id -> {center, rotation}
        self.prediction = prediction # Filtered DA3 Prediction object
        self.pano_avg_deviation = pano_avg_deviation or {} # pano_id -> avg dist deviation (m) among its own KEPT views only -- a single wild outlier gets filtered out anyway, so it says nothing about quality; the kept views still not agreeing well with each other on average is what actually flags a bad pairing. inf if zero views were kept.
        self.pano_keep_counts = pano_keep_counts or {} # pano_id -> (kept, total)
        self.views = views or [] # the views that survived the consensus filter, index-aligned with prediction
