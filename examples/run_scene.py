"""
Run the LiDAR simulation on a real trained 3DGS scene.

Usage:
    python run_scene.py "/path/to/scene.ply" [output_dir]

Sensor is placed at the scene's Gaussian-center centroid (a reasonable
stand-in for "somewhere in the middle of the room" when we don't know the
capture's up-axis/scale convention).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quickpointforge import (
    load_gaussian_ply,
    VELODYNE_VLP16,
    VELODYNE_HDL64E_APPROX,
    simulate_lidar_scan,
)
from quickpointforge.io import filter_splat
from quickpointforge.export import save_scan


def run(ply_path: str, out_dir: str) -> None:
    splat = load_gaussian_ply(ply_path)
    print(f"Loaded {len(splat)} raw Gaussian centers from {ply_path}")

    splat = filter_splat(splat, min_opacity=0.2)
    print(f"{len(splat)} remain after opacity filtering.")

    sensor_position = splat.centers.mean(axis=0)
    print(f"Sensor position (scene centroid): {sensor_position}")

    for sensor in (VELODYNE_VLP16, VELODYNE_HDL64E_APPROX):
        scan = simulate_lidar_scan(splat, sensor, sensor_position, return_frame="world")
        print(f"\nSensor: {sensor.name}")
        print(f"  beams={sensor.num_beams}, azimuth_bins={sensor.num_azimuth_bins}, "
              f"total_cells={sensor.num_beams * sensor.num_azimuth_bins}")
        print(f"Output LiDAR-style points: {scan.num_output_points}")
        print(f"Hit rate: {scan.hit_rate():.1%}")
        print(f"Range stats: min={scan.ranges.min():.2f}m max={scan.ranges.max():.2f}m "
              f"mean={scan.ranges.mean():.2f}m")

        out_path = os.path.join(out_dir, f"{sensor.name.replace(' ', '_')}_scan.pcd")
        save_scan(scan, out_path)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    ply_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(ply_path)
    run(ply_path, out_dir)
