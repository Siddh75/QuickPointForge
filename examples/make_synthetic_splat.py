"""
Generate a synthetic "Gaussian splat" PLY file for testing the pipeline
without needing a real trained 3DGS scene.

Scene: a box room (walls/floor/ceiling) plus a sphere "obstacle" in the
middle -- dense enough to sanity-check beam binning and occlusion (the
sphere should occlude the far wall behind it for beams that hit it).
"""

import numpy as np
from plyfile import PlyData, PlyElement


def sample_box_room(half_size=10.0, n_per_wall=40000, seed=0):
    rng = np.random.default_rng(seed)
    pts = []

    def wall(fixed_axis, fixed_val, n):
        u = rng.uniform(-half_size, half_size, n)
        v = rng.uniform(-half_size, half_size, n)
        p = np.zeros((n, 3))
        axes = [0, 1, 2]
        axes.remove(fixed_axis)
        p[:, axes[0]] = u
        p[:, axes[1]] = v
        p[:, fixed_axis] = fixed_val
        return p

    pts.append(wall(0, -half_size, n_per_wall))   # -x wall
    pts.append(wall(0, half_size, n_per_wall))    # +x wall
    pts.append(wall(1, -half_size, n_per_wall))   # -y wall
    pts.append(wall(1, half_size, n_per_wall))    # +y wall
    pts.append(wall(2, -half_size, n_per_wall))   # floor
    pts.append(wall(2, half_size, n_per_wall))    # ceiling
    return np.concatenate(pts, axis=0)


def sample_sphere(center, radius, n=30000, seed=1):
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return center[None, :] + radius * vecs


def write_gaussian_ply(path, centers, opacity, scale, color):
    n = centers.shape[0]
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    vertex = np.empty(n, dtype=dtype)
    vertex["x"], vertex["y"], vertex["z"] = centers[:, 0], centers[:, 1], centers[:, 2]
    vertex["opacity"] = opacity
    log_scale = np.log(scale)
    vertex["scale_0"], vertex["scale_1"], vertex["scale_2"] = log_scale[:, 0], log_scale[:, 1], log_scale[:, 2]
    # store color as SH DC term matching the 0.5 + C0*f_dc convention used in io.py
    C0 = 0.28209479177387814
    f_dc = (color - 0.5) / C0
    vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]

    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=False).write(path)


if __name__ == "__main__":
    room = sample_box_room(half_size=10.0, n_per_wall=40000)
    sphere = sample_sphere(center=np.array([3.0, 0.0, 0.0]), radius=1.5, n=30000)
    centers = np.concatenate([room, sphere], axis=0)

    n = centers.shape[0]
    rng = np.random.default_rng(42)
    opacity = rng.uniform(0.6, 0.99, n)          # mostly solid, some low-opacity "floaters" mixed in
    floaters = rng.uniform(0, 1, n) < 0.02
    opacity[floaters] = rng.uniform(0.0, 0.15, floaters.sum())

    scale = np.column_stack([
        rng.uniform(0.01, 0.05, n),
        rng.uniform(0.01, 0.05, n),
        rng.uniform(0.01, 0.05, n),
    ])
    color = rng.uniform(0.3, 0.9, (n, 3))

    out_path = "/home/claude/splat2lidar/examples/synthetic_room.ply"
    write_gaussian_ply(out_path, centers, opacity, scale, color)
    print(f"Wrote {n} synthetic Gaussian points to {out_path}")
