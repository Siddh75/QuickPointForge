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
at silhouette edges and in sparsely-sampled regions of the splat.

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

## Install

```bash
pip install -e ".[dev]"    # editable install + pytest, for development
# or: pip install -e .     # without dev/test dependencies
```

## Quickstart

A native desktop UI (Open3D's `gui`/`rendering` modules — one window, no
browser) for interactive use:

```bash
splat2lidar-gui          # after `pip install -e .`
# or: python -m splat2lidar.app
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
3. **Sensor**: pick a spinning or flash preset (see "Sensor presets"
   below) or "Custom..." to enter your own beam count, FOV, azimuth
   resolution, and range limits.
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

`splat2lidar.sensors` ships approximate beam tables for spinning sensors:
- `VELODYNE_VLP16` (16 beams, ±15°)
- `VELODYNE_VLP32C_APPROX` (32 beams, -25° to +15°, **uniform approximation**)
- `VELODYNE_HDL32E_APPROX` (32 beams, -30.67° to +10.67°, **uniform approximation**)
- `VELODYNE_HDL64E_APPROX` (64 beams, -24.8° to +2°, **uniform approximation**
  — the real unit has non-uniform per-beam spacing)
- `OUSTER_OS0_128_APPROX` (128 beams, ±45°, wide-FOV/short-range variant, **uniform approximation**)
- `OUSTER_OS1_64_APPROX` (64 beams, ±22.5°, **uniform approximation**)
- `OUSTER_OS2_128_APPROX` (128 beams, ±11.25°, narrow-FOV/long-range variant, **uniform approximation**)
- `HESAI_PANDAR64_APPROX` (64 beams, -25° to +15°, **uniform approximation**)

Use `generic_uniform_sensor(...)` to define your own spinning sensor, or
build a `SensorModel` directly from a real factory calibration file if you
have one and need per-beam accuracy.

For **flash** sensors (fixed rectangular FOV, single-shot, no 360°
wraparound), `SensorModel.azimuth_fov_deg` bounds the azimuth grid instead
of wrapping it. Two illustrative examples are provided (not tied to a
specific real product's datasheet):
- `FLASH_LIDAR_EXAMPLE` (64x64 grid, ±30°x±15° FOV, 60m range)
- `FLASH_LIDAR_NARROW_LONGRANGE_EXAMPLE` (96x48 grid, ±10°x±5° FOV, 150m range)

Use `generic_flash_sensor(...)` to define your own.

## Project background

Built as a fast/cheap alternative to research pipelines like
[SplatAD](https://research.zenseact.com/publications/splatad/),
[LiDAR-GS](https://arxiv.org/abs/2410.05111), and
[LiDAR-RT](https://arxiv.org/pdf/2412.15199), which all preserve
covariance/opacity-aware alpha blending during LiDAR rendering. This
project deliberately trades that away for speed and simplicity, and is
meant as an engineering baseline to benchmark against, not a claim of
matching their fidelity.
