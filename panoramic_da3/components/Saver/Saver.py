import os
import numpy as np
import cv2

class Saver:
    @staticmethod
    def colorize_depth(depth):
        """Turns raw depth (meters/units) into a colored image (JPG/PNG compatible)."""
        # Normalize to 0-255
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 0:
            normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            normalized = depth * 0

        # Apply a standard colormap (Magma or Jet)
        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        return colored

    @staticmethod
    def save_depth_image(depth, path):
        """Saves depth as a colored JPG/PNG."""
        colored = Saver.colorize_depth(depth)
        cv2.imwrite(path, colored)
        print(f"Saved depth image to: {path}")

    @staticmethod
    def _voxel_downsample(points: np.ndarray, colors: np.ndarray | None, voxel_size: float):
        """Bins points into a uniform 3D grid and averages each occupied
        cell's points/colors into one representative point -- same
        semantics as open3d's voxel_down_sample, in pure numpy.

        Deliberately NOT open3d: its voxel_down_sample/write_point_cloud
        spin up a persistent background thread pool (via its own native
        threading) the first time either is called, which never gets torn
        down for the rest of the process's life. On HF Spaces' ZeroGPU,
        that's a real fork-safety hazard: any LATER @spaces.GPU call forks
        a fresh worker process, and if that fork lands while this
        leftover thread pool exists, it can corrupt an unrelated native
        library's one-time init lock -- confirmed directly (real
        segfault, isolated down to exactly this call via a dedicated
        debug probe) as the cause of a crash on a second GPU call in this
        app. Pure numpy/manual I/O never creates that thread pool."""
        if len(points) == 0:
            return points, colors
        voxel_indices = np.floor(points / voxel_size).astype(np.int64)
        _, inverse, counts = np.unique(voxel_indices, axis=0, return_inverse=True, return_counts=True)
        inverse = np.asarray(inverse).reshape(-1)
        counts = counts.astype(np.float64)
        n_voxels = len(counts)

        def grouped_mean(values):
            out = np.empty((n_voxels, values.shape[1]), dtype=np.float64)
            for c in range(values.shape[1]):
                out[:, c] = np.bincount(inverse, weights=values[:, c], minlength=n_voxels)
            return out / counts[:, None]

        down_points = grouped_mean(points).astype(points.dtype)
        down_colors = grouped_mean(colors).astype(colors.dtype) if colors is not None else None
        return down_points, down_colors

    @staticmethod
    def _write_ply(path: str, points: np.ndarray, colors: np.ndarray | None):
        """Writes a binary little-endian PLY -- points (float32 xyz) plus
        optional colors (uint8 rgb, from normalized [0,1] floats). One
        structured-array memory layout, one write() call: fast even for
        hundreds of thousands of points, no per-row Python formatting."""
        n = len(points)
        if colors is not None:
            dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                               ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        else:
            dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])

        structured = np.empty(n, dtype=dtype)
        structured["x"] = points[:, 0]
        structured["y"] = points[:, 1]
        structured["z"] = points[:, 2]
        if colors is not None:
            cols255 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
            structured["red"] = cols255[:, 0]
            structured["green"] = cols255[:, 1]
            structured["blue"] = cols255[:, 2]

        header = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n}",
            "property float x", "property float y", "property float z",
        ]
        if colors is not None:
            header += ["property uchar red", "property uchar green", "property uchar blue"]
        header.append("end_header")

        with open(path, "wb") as f:
            f.write(("\n".join(header) + "\n").encode("ascii"))
            f.write(structured.tobytes())

    @staticmethod
    def save_point_cloud(points: np.ndarray, path: str, colors: np.ndarray = None, voxel_size: float = 0.03):
        """
        Saves a raw XYZ point cloud (N, 3) to a PLY file.
        Args:
            points: (N, 3) array of XYZ points.
            path: Output path.
            colors: (N, 3) normalized RGB colors OR (H, W, 3) image.
            voxel_size: bin points into a uniform 3D grid (meters) and keep
                one per occupied cell before writing -- catches
                near-duplicate points the per-view angular wedge trim
                doesn't (e.g. redundant coverage between adjacent panos
                along a walked path, not just between one pano's own view
                slices). Set to 0/None to skip.
        """
        points = np.asarray(points)
        if colors is not None:
            colors = np.asarray(colors)
            if colors.ndim == 3:
                # It's an image, need to reshape it (legacy support)
                # Assuming colors is BGR image from cv2
                img_rgb = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)
                flat_colors = img_rgb.reshape(-1, 3) / 255.0
                if len(flat_colors) == len(points):
                    colors = flat_colors
                else:
                    print("Warning: Image size doesn't match point count. Skipping colors.")
                    colors = None
            elif not (colors.ndim == 2 and colors.shape[0] == points.shape[0]):
                colors = None

        if voxel_size:
            before = len(points)
            points, colors = Saver._voxel_downsample(points, colors, voxel_size)
            print(f"Voxel downsample ({voxel_size}m): {before} -> {len(points)} points")

        Saver._write_ply(path, points, colors)
        print(f"Saved point cloud to: {path}")
