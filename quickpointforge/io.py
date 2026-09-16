"""
Loading 3D Gaussian Splat .ply files.

Standard 3DGS (INRIA / gsplat / nerfstudio-exported) PLY files store, per
Gaussian, custom vertex properties beyond plain xyz:

    x, y, z                 -- center position
    opacity                 -- raw opacity (pre-sigmoid, usually)
    scale_0, scale_1, scale_2   -- log-scale of the 3 principal axes
    rot_0..rot_3             -- quaternion (unused here, kept for completeness)
    f_dc_0, f_dc_1, f_dc_2    -- SH DC term (proxy for base RGB)
    f_rest_*                 -- higher-order SH (ignored)

Generic point-cloud readers (Open3D's read_point_cloud, meshlab, etc.)
often silently drop everything except xyz(+normal+rgb), which is why we
parse the raw PLY ourselves with `plyfile`.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from plyfile import PlyData


@dataclass
class GaussianSplat:
    """Container for the fields of a loaded Gaussian splat we actually use."""

    centers: np.ndarray          # (N, 3) float64, world-frame xyz
    opacity: np.ndarray          # (N,) float64, sigmoid-activated in [0, 1]
    scale: Optional[np.ndarray]  # (N, 3) float64, exp-activated (linear units), or None
    color: Optional[np.ndarray]  # (N, 3) float64 in [0, 1] approx RGB from SH DC term, or None

    def __len__(self) -> int:
        return self.centers.shape[0]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sh_dc_to_rgb(f_dc: np.ndarray) -> np.ndarray:
    # Standard 3DGS convention: RGB = 0.5 + C0 * f_dc, with C0 = 0.28209479177387814
    C0 = 0.28209479177387814
    rgb = 0.5 + C0 * f_dc
    return np.clip(rgb, 0.0, 1.0)


def load_gaussian_ply(path: str) -> GaussianSplat:
    """
    Load a 3D Gaussian Splat .ply file and extract centers, opacity, scale,
    and an approximate RGB color.

    Falls back gracefully if optional fields (opacity, scale, color) are
    missing, e.g. if you point this at a *plain* point cloud PLY instead of
    a trained Gaussian splat -- in that case opacity defaults to all-ones
    and scale/color are left as None.
    """
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    names = vertex.data.dtype.names

    centers = np.stack(
        [np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])],
        axis=-1,
    ).astype(np.float64)

    if "opacity" in names:
        raw_opacity = np.asarray(vertex["opacity"]).astype(np.float64)
        # 3DGS stores opacity pre-sigmoid; heuristically detect and activate.
        if raw_opacity.min() < 0.0 or raw_opacity.max() > 1.0:
            opacity = _sigmoid(raw_opacity)
        else:
            opacity = raw_opacity
    else:
        opacity = np.ones(centers.shape[0], dtype=np.float64)

    scale = None
    scale_names = [f"scale_{i}" for i in range(3)]
    if all(n in names for n in scale_names):
        log_scale = np.stack([np.asarray(vertex[n]) for n in scale_names], axis=-1).astype(np.float64)
        scale = np.exp(log_scale)

    color = None
    dc_names = [f"f_dc_{i}" for i in range(3)]
    if all(n in names for n in dc_names):
        f_dc = np.stack([np.asarray(vertex[n]) for n in dc_names], axis=-1).astype(np.float64)
        color = _sh_dc_to_rgb(f_dc)
    elif all(n in names for n in ("red", "green", "blue")):
        color = np.stack(
            [np.asarray(vertex["red"]), np.asarray(vertex["green"]), np.asarray(vertex["blue"])],
            axis=-1,
        ).astype(np.float64) / 255.0

    return GaussianSplat(centers=centers, opacity=opacity, scale=scale, color=color)


def filter_splat(
    splat: GaussianSplat,
    min_opacity: float = 0.2,
    max_scale: Optional[float] = None,
    max_anisotropy_ratio: Optional[float] = None,
) -> GaussianSplat:
    """
    Drop low-opacity "floater" Gaussians, oversized background blobs, and
    (optionally) round/isotropic Gaussians before projecting to spherical
    coordinates.

    min_opacity: Gaussians with opacity below this are dropped. 0.2 is a
        common default in 3DGS densification/pruning heuristics.
    max_scale: if given, drop Gaussians whose largest principal scale
        exceeds this value (in the same linear units as your scene, e.g.
        meters). Leave as None if `splat.scale` is unavailable.
    max_anisotropy_ratio: if given, drop Gaussians whose
        (min_scale / max_scale) ratio exceeds this value. Well-converged
        Gaussians on a real surface tend to flatten into thin disks
        (low ratio); round, isotropic "blob" Gaussians (ratio near 1) are
        the ones most likely to be floating off-surface rather than
        marking a real one -- opacity alone doesn't catch these, since a
        round blob can still be fully opaque. A typical starting value is
        around 0.3-0.5; leave as None to skip this filter (e.g. if
        `splat.scale` is unavailable).
    """
    mask = splat.opacity >= min_opacity
    if max_scale is not None and splat.scale is not None:
        mask &= splat.scale.max(axis=1) <= max_scale
    if max_anisotropy_ratio is not None and splat.scale is not None:
        ratio = splat.scale.min(axis=1) / splat.scale.max(axis=1)
        mask &= ratio <= max_anisotropy_ratio

    return GaussianSplat(
        centers=splat.centers[mask],
        opacity=splat.opacity[mask],
        scale=splat.scale[mask] if splat.scale is not None else None,
        color=splat.color[mask] if splat.color is not None else None,
    )
