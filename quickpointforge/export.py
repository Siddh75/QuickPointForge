"""Export a ScanResult to disk as .pcd / .ply / .npy."""

from typing import Optional

import numpy as np
import open3d as o3d

from .simulate import ScanResult


def save_scan(scan: ScanResult, path: str) -> None:
    """
    Save a ScanResult's points (+ color, if present) to disk.

    Format is inferred from the extension: .pcd, .ply, or .npy.
    For .npy, saves an (M, 7) array: [x, y, z, range, beam_index, azimuth_bin, intensity],
    or (M, 8) with a trailing timestamp column if `scan.timestamp` is set (e.g. a
    scan merged from several trajectory samples via `concatenate_scans`).
    Only .npy carries the timestamp column -- .pcd/.ply have no custom scalar
    field via Open3D's writer.
    """
    if path.endswith(".npy"):
        intensity = scan.intensity if scan.intensity is not None else np.zeros(scan.num_output_points)
        columns = [
            scan.points, scan.ranges, scan.beam_index.astype(np.float64),
            scan.azimuth_bin.astype(np.float64), intensity,
        ]
        if scan.timestamp is not None:
            columns.append(scan.timestamp)
        arr = np.column_stack(columns)
        np.save(path, arr)
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scan.points)
    if scan.color is not None:
        pcd.colors = o3d.utility.Vector3dVector(scan.color)
    o3d.io.write_point_cloud(path, pcd)
