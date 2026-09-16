# splat2lidar

Simulate a LiDAR-style point cloud directly from **3D Gaussian Splat centers**
— no ray-casting, no alpha-blended rasterization. Treats the splat's
Gaussian means as if they were a dense photogrammetry point cloud, then
reuses the standard point-cloud-to-range-image binning approach from LiDAR
SLAM preprocessing to emulate a real sensor's beam pattern.

## Why this exists

Every published splat→LiDAR method (SplatAD, LiDAR-GS, LiDAR-RT, GS-LiDAR)
does proper ray-splat intersection or covariance-aware alpha-blended
rasterization in spherical/range-view space. That's more physically
correct (soft returns at occlusion boundaries, opacity-aware blending) but
computationally heavier. This project explores the cheap end of that
tradeoff: **can you get a usable synthetic LiDAR scan just by binning raw
Gaussian centers?**

Short answer: often yes for interior/bulk geometry, with known weaknesses
at silhouette edges and in sparsely-sampled regions of the splat. See
"Known limitations" below.

## How it works

1. **Load** a 3DGS-format `.ply` (centers, opacity, scale, SH-DC color).
2. **Filter** low-opacity "floater" Gaussians (and optionally oversized
   background blobs) before projecting.
3. **Transform** centers into the sensor's local frame given a pose.
4. **Project** to spherical coordinates: range, azimuth, elevation
   (sensor-frame convention: x-forward, y-left, z-up).
5. **Bin** each point into a `(beam_index, azimuth_bin)` cell using the
   target sensor's real fixed beam elevation angles and azimuth
   resolution.
6. **Resolve occlusion** by keeping only the *nearest-range* point in each
   cell — the cheap substitute for a proper first-return ray-surface
   intersection or z-buffer.

This is an `O(N log N)` sort (`np.lexsort` + `np.unique`), not
`O(num_rays × num_splats)` ray-surface intersection — no BVH, no
ray-triangle tests.

## Quickstart

```bash
pip install -r requirements.txt

cd examples
python make_synthetic_splat.py   # generates a synthetic room+sphere "splat"
python run_demo.py               # simulates a VLP-16 scan, saves simulated_scan.pcd
```

To use with a real trained 3DGS scene:

```python
from splat2lidar import load_gaussian_ply, VELODYNE_HDL64E_APPROX, simulate_lidar_scan
from splat2lidar.io import filter_splat
from splat2lidar.export import save_scan
import numpy as np

splat = load_gaussian_ply("point_cloud.ply")     # standard 3DGS export
splat = filter_splat(splat, min_opacity=0.2)

scan = simulate_lidar_scan(
    splat,
    sensor=VELODYNE_HDL64E_APPROX,
    sensor_position=np.array([0.0, 0.0, 1.8]),   # e.g. sensor mounted 1.8m up
    return_frame="world",
)

save_scan(scan, "simulated_scan.pcd")
print(scan.num_output_points, "points, hit rate", scan.hit_rate())
```

## Desktop app

A native desktop UI (Open3D's `gui`/`rendering` modules — one window, no
browser) for interactive use:

```bash
python app.py
```

Panel on the left, **two live 3D viewports** on the right:
- **Gaussian splat view** (left viewport): the raw/filtered splat point
  cloud, plus a coordinate-frame gizmo (red=X forward, green=Y left,
  blue=Z up) marking the sensor's current position and orientation. This
  updates live as you edit the pose fields, so you can see exactly where
  the sensor sits relative to the scene before running a scan.
- **Simulated LiDAR view** (right viewport): the binned scan result,
  colorable by range / intensity / beam index / RGB.

1. **Load** a `.ply` splat — shows Gaussian count and bounding box.
2. **Filter** with live sliders: min opacity, max anisotropy ratio
   (`min_scale/max_scale` — lower keeps only flat, surface-like
   Gaussians and drops round "floater" blobs), optional max scale.
   Click "Apply filters" to see the surviving count and update the splat
   view.
3. **Sensor**: pick a preset (VLP-16 / HDL-64E-approx / OS1-64-approx) or
   "Custom..." to enter your own beam count, FOV, azimuth resolution,
   and range limits.
4. **Pose**: sensor position (x/y/z) and orientation (yaw/pitch/roll) —
   the gizmo in the splat view moves live as you type.
5. **Simulate scan** — runs the binning algorithm and reports input/output
   point counts, hit rate, and range stats; populates the LiDAR view.
6. **Display**: color the LiDAR result by range, intensity (opacity
   proxy), beam index, or the splat's own RGB.
7. **Export** to `.pcd`/`.ply`/`.npy`.

Requires a real display (X11/Wayland/macOS/Windows) — it won't run in a
headless container/SSH session without a virtual framebuffer.

## Sensor presets

`splat2lidar.sensors` ships approximate beam tables for:
- `VELODYNE_VLP16` (16 beams, ±15°)
- `VELODYNE_HDL64E_APPROX` (64 beams, -24.8° to +2°, **uniform approximation**
  — the real unit has non-uniform per-beam spacing)
- `OUSTER_OS1_64_APPROX` (64 beams, ±22.5°, **uniform approximation**)

Use `generic_uniform_sensor(...)` to define your own, or build a
`SensorModel` directly from a real factory calibration file if you have
one and need per-beam accuracy.

## Known limitations (read before trusting the output)

- **Center-to-surface offset**: raw, unregularized 3DGS Gaussian centers
  don't sit exactly on the true surface — this is well documented (it's
  literally SuGaR's motivation). Expect worse accuracy on scenes trained
  without surface-alignment regularization (vanilla 3DGS) than on
  2DGS/SuGaR-regularized scenes.
- **Hard occlusion, no soft returns**: nearest-in-cell is a binary choice.
  Real LiDAR (and proper ray-splat rendering) gives graceful partial
  returns at silhouette edges; this method doesn't. Expect the largest
  error concentration at object boundaries, thin structures, and
  transparent/reflective surfaces.
- **No multi-return simulation**: only one point per cell, so no
  second/third returns through foliage or glass.
- **No physically grounded intensity**: `scan.intensity` is a proxy taken
  directly from Gaussian opacity, not a real reflectance value.
- **Density-dependent gaps**: splat density is uneven (denser near
  training camera viewpoints). Cells with no nearby Gaussian center within
  tolerance simply get no return — there's no fallback interpolation in
  v0.1. Check `scan.hit_rate()` per scene.

## Roadmap

- [ ] Fallback interpolation for empty cells (k-NN within angular cone)
- [ ] Range-dependent noise model to match a target sensor's real
      accuracy/precision spec
- [ ] Benchmark script: synthetic mesh → splat → {this method, proper
      ray-splat intersection} → Chamfer distance, to quantify the
      accuracy/speed tradeoff directly rather than estimating it
- [ ] Real factory beam-table loader (non-uniform calibration files)

## Project background

Built as a fast/cheap alternative to research pipelines like
[SplatAD](https://research.zenseact.com/publications/splatad/),
[LiDAR-GS](https://arxiv.org/abs/2410.05111), and
[LiDAR-RT](https://arxiv.org/pdf/2412.15199), which all preserve
covariance/opacity-aware alpha blending during LiDAR rendering. This
project deliberately trades that away for speed and simplicity, and is
meant as an engineering baseline to benchmark against, not a claim of
matching their fidelity.
