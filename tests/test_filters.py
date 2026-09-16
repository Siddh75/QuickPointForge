import numpy as np

from quickpointforge.io import GaussianSplat, filter_splat


def test_opacity_filter_drops_low_opacity():
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    opacity = np.array([0.9, 0.05])
    splat = GaussianSplat(centers=centers, opacity=opacity, scale=None, color=None)

    out = filter_splat(splat, min_opacity=0.2)

    assert len(out) == 1
    assert np.allclose(out.centers[0], [0.0, 0.0, 0.0])


def test_anisotropy_filter_keeps_flat_drops_round():
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    opacity = np.array([0.9, 0.9])
    scale = np.array([
        [0.5, 0.5, 0.02],   # flat disk, ratio 0.04 -- surface-like
        [0.5, 0.48, 0.45],  # near-round blob, ratio ~0.9 -- likely a floater
    ])
    splat = GaussianSplat(centers=centers, opacity=opacity, scale=scale, color=None)

    out = filter_splat(splat, min_opacity=0.2, max_anisotropy_ratio=0.3)

    assert len(out) == 1
    assert np.allclose(out.centers[0], [0.0, 0.0, 0.0])


def test_max_scale_filter_drops_oversized_blobs():
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    opacity = np.array([0.9, 0.9])
    scale = np.array([
        [0.05, 0.05, 0.05],
        [5.0, 5.0, 5.0],   # oversized background blob
    ])
    splat = GaussianSplat(centers=centers, opacity=opacity, scale=scale, color=None)

    out = filter_splat(splat, min_opacity=0.2, max_scale=1.0)

    assert len(out) == 1
    assert np.allclose(out.centers[0], [0.0, 0.0, 0.0])


def test_filters_compose():
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    opacity = np.array([0.9, 0.9, 0.05])
    scale = np.array([
        [0.5, 0.5, 0.02],
        [0.5, 0.48, 0.45],
        [0.5, 0.5, 0.02],
    ])
    splat = GaussianSplat(centers=centers, opacity=opacity, scale=scale, color=None)

    out = filter_splat(splat, min_opacity=0.2, max_anisotropy_ratio=0.3)

    # index 1 fails anisotropy, index 2 fails opacity -- only index 0 survives
    assert len(out) == 1
    assert np.allclose(out.centers[0], [0.0, 0.0, 0.0])
