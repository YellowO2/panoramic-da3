# panoramic-da3

Runs [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) on a batch of panoramic images. Outputs a world-space point cloud. 

## Install

```bash
pip install git+https://github.com/YellowO2/panoramic-da3.git
```

## Usage

```python
from panoramic_da3 import run_da3

class Config:
    da3_model = "depth-anything/DA3NESTED-GIANT-LARGE"

filtered_views, da3_result, points, colors, per_pano_points, per_pano_colors = run_da3(
    target_depth_path="pano_0.jpg",
    support_paths=["pano_1.jpg", "pano_2.jpg"],
    cfg=Config(),
    views_base="/tmp/views",
)
```

`cfg` just needs a `.da3_model` attribute (the model path or HF repo id) -- pass any object that has one, including a caller's own richer config.

See `run_da3`'s own docstring in `panoramic_da3/pipeline.py` for the full return shape.

## License

MIT.
