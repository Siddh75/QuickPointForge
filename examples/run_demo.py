"""
End-to-end demo:
    1. Load the synthetic splat PLY (run make_synthetic_splat.py first).
    2. Filter low-opacity floaters.
    3. Simulate a VLP-16 scan from the room's center.
    4. Save the result and print summary stats.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quickpointforge import load_gaussian_ply, VELODYNE_VLP16, simulate_lidar_scan
from quickpointforge.io import filter_splat
from quickpointforge.export import save_scan


def main():
    ply_path = os.path.join(os.path.dirname(__file__), "synthetic_room.ply")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(
            f"{ply_path} not found -- run `python make_synthetic_splat.py` first."
        )

    splat = load_gaussian_ply(ply_path)
    print(f"Loaded {len(splat)} raw Gaussian centers.")

    splat = filter_splat(splat, min_opacity=0.2)
    print(f"{len(splat)} remain after opacity filtering.")

    sensor = VELODYNE_VLP16
    sensor_position = np.array([0.0, 0.0, 0.0])  # scanner at room center

    scan = simulate_lidar_scan(splat, sensor, sensor_position, return_frame="world")

    print(f"\nSensor: {sensor.name}")
    print(f"  beams={sensor.num_beams}, azimuth_bins={sensor.num_azimuth_bins}, "
          f"total_cells={sensor.num_beams * sensor.num_azimuth_bins}")
    print(f"Input points considered: {scan.num_input_points}")
    print(f"Output LiDAR-style points: {scan.num_output_points}")
    print(f"Hit rate: {scan.hit_rate():.1%}")
    print(f"Range stats: min={scan.ranges.min():.2f}m max={scan.ranges.max():.2f}m "
          f"mean={scan.ranges.mean():.2f}m")

    out_path = os.path.join(os.path.dirname(__file__), "simulated_scan.pcd")
    save_scan(scan, out_path)
    print(f"\nSaved simulated scan to {out_path}")


if __name__ == "__main__":
    main()
