# panoramic-da3

Runs [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) on a batch of panoramic images. Outputs a world-space point cloud. 

## Install

```bash
pip install git+https://github.com/YellowO2/panoramic-da3.git
```

## Usage

```python
from panoramic_da3 import Loader, PointCloud, Saver, Settings, panoramic_da3

panoramas = Loader(["pano_0.jpg", "pano_1.jpg", "pano_2.jpg"])
point_clouds, da3_result = panoramic_da3(panoramas, Settings(da3_model="depth-anything/DA3NESTED-GIANT-LARGE"))

cloud = PointCloud.merge(point_clouds.values())
Saver(cloud.points, "scene.ply", colors=cloud.colours)
```

- `panoramas` is `{pano_id: BGR image}`; `Loader` keys them by file name.
- `point_clouds` is `{pano_id: PointCloud}`, each with `points`, `colours` and, with `Settings(return_confidence=True)`, `confidence`.
- `da3_result` holds each pano's pose (`pano_poses`), how many of its views DA3 kept (`pano_keep_counts`) and how well they agreed (`pano_avg_deviation`), and the kept `views`.
- `Settings` holds everything that tunes the run; see `panoramic_da3/Settings.py`.

## License

MIT.
