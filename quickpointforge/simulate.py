"""
Core algorithm: spherical binning of Gaussian splat centers to simulate a
LiDAR scan, without ray-casting.

Pipeline per scan:
    1. Transform splat centers into the sensor frame (given sensor pose).
    2. Convert to spherical coordinates (range, azimuth, elevation).
       Sensor-frame convention: x-forward, y-left, z-up (REP-103 style).
    3. Drop points outside [min_range, max_range] or too far from any beam's
       elevation (beyond elevation_tolerance_deg).
    4. Assign each surviving point to a (beam_index, azimuth_bin) cell.
    5. Within each cell, keep only the point with the smallest range
       (approximates a first-return LiDAR without a real z-buffer / BVH).

This trades ray-casting's correct occlusion + alpha-blended soft returns
for an O(N log N) sort -- see README.md for the accuracy tradeoff this implies.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .io import GaussianSplat
from .sensors import SensorModel


@dataclass
class ScanResult:
    points: np.ndarray            # (M, 3) xyz in the SAME frame as `world_points_frame` arg
    ranges: np.ndarray            # (M,)
    beam_index: np.ndarray        # (M,) int
    azimuth_bin: np.ndarray       # (M,) int
    intensity: Optional[np.ndarray]   # (M,) in [0, 1], proxy from opacity, or None
    color: Optional[np.ndarray]       # (M, 3) in [0, 1], carried through from the splat, or None
    num_input_points: int
    num_output_points: int
    total_cells: int = 0
    timestamp: Optional[np.ndarray] = None  # (M,) time the scan producing each point was taken, or None

    def hit_rate(self) -> Optional[float]:
        """Fraction of the sensor's (beam, azimuth) cells that got a return, if known."""
        if not self.total_cells:
            return None
        return self.num_output_points / self.total_cells


def world_to_sensor(points_world: np.ndarray, sensor_position: np.ndarray, sensor_rotation: np.ndarray) -> np.ndarray:
    """
    Transform points from world frame to sensor frame.

    sensor_position: (3,) sensor origin in world coordinates.
    sensor_rotation: (3, 3) rotation matrix, columns = sensor axes expressed
        in world coordinates (i.e. world_point = R @ sensor_point + t).
    """
    return (points_world - sensor_position[None, :]) @ sensor_rotation  # R^T applied via right-multiply


def _cartesian_to_spherical(points_sensor: np.ndarray):
    x, y, z = points_sensor[:, 0], points_sensor[:, 1], points_sensor[:, 2]
    r = np.linalg.norm(points_sensor, axis=1)
    horiz = np.sqrt(x * x + y * y)
    elevation_deg = np.degrees(np.arctan2(z, horiz))
    # Raw azimuth in (-180, 180], 0 = forward (+x). Spinning sensors wrap
    # this to [0, 360) below; flash sensors use it as-is against their FOV.
    azimuth_deg = np.degrees(np.arctan2(y, x))
    return r, azimuth_deg, elevation_deg


def simulate_lidar_scan(
    splat: GaussianSplat,
    sensor: SensorModel,
    sensor_position: np.ndarray,
    sensor_rotation: Optional[np.ndarray] = None,
    return_frame: str = "sensor",
    timestamp: Optional[float] = None,
) -> ScanResult:
    """
    Simulate one LiDAR sweep by binning Gaussian centers into the sensor's
    beam/azimuth grid and keeping the nearest-range point per cell.

    sensor_position: (3,) sensor origin, world coordinates.
    sensor_rotation: (3, 3) rotation matrix (world <- sensor axes), identity
        if None (i.e. sensor axes aligned with world axes).
    return_frame: "sensor" (default) or "world" -- which frame the output
        `points` are expressed in. Range/beam/azimuth bookkeeping is
        always computed in the sensor frame regardless.
    timestamp: if given, stamps every output point with this time (e.g. a
        sample time along a Trajectory) -- see `concatenate_scans` for
        merging several timestamped scans into one exportable cloud.
    """
    if sensor_rotation is None:
        sensor_rotation = np.eye(3)

    points_sensor = world_to_sensor(splat.centers, np.asarray(sensor_position, dtype=np.float64), sensor_rotation)
    r, azimuth_deg, elevation_deg = _cartesian_to_spherical(points_sensor)

    in_range = (r >= sensor.min_range_m) & (r <= sensor.max_range_m)

    # Nearest-beam assignment + tolerance check: binary search against the
    # midpoints between sorted beams -- O(N log B) instead of an
    # (N, num_beams) distance matrix (GBs for 128-beam sensors).
    beam_order = np.argsort(sensor.beam_elevations_deg, kind="stable")
    beam_sorted = sensor.beam_elevations_deg[beam_order]
    nearest = np.searchsorted((beam_sorted[:-1] + beam_sorted[1:]) / 2.0, elevation_deg)
    beam_index = beam_order[nearest]
    within_tolerance = np.abs(elevation_deg - beam_sorted[nearest]) <= sensor.elevation_tolerance_deg

    if sensor.azimuth_fov_deg is None:
        azimuth_wrapped = np.mod(azimuth_deg, 360.0)
        azimuth_bin = np.mod(
            np.round(azimuth_wrapped / sensor.azimuth_resolution_deg).astype(np.int64),
            sensor.num_azimuth_bins,
        )
        in_fov = np.ones(azimuth_deg.shape, dtype=bool)
    else:
        fov_min, fov_max = sensor.azimuth_fov_deg
        in_fov = (azimuth_deg >= fov_min) & (azimuth_deg <= fov_max)
        azimuth_bin = np.clip(
            np.round((azimuth_deg - fov_min) / sensor.azimuth_resolution_deg).astype(np.int64),
            0, sensor.num_azimuth_bins - 1,
        )

    keep = in_range & within_tolerance & in_fov
    num_input = points_sensor.shape[0]

    total_cells = sensor.num_beams * sensor.num_azimuth_bins

    idx = np.nonzero(keep)[0]
    if idx.size == 0:
        empty = np.zeros((0, 3))
        return ScanResult(
            points=empty, ranges=np.zeros(0), beam_index=np.zeros(0, dtype=np.int64),
            azimuth_bin=np.zeros(0, dtype=np.int64), intensity=None, color=None,
            num_input_points=num_input, num_output_points=0, total_cells=total_cells,
            timestamp=np.zeros(0) if timestamp is not None else None,
        )

    r_k = r[idx]
    beam_k = beam_index[idx]
    az_k = azimuth_bin[idx]
    cell_key = beam_k.astype(np.int64) * sensor.num_azimuth_bins + az_k

    # Sort by (cell_key, range) ascending, then take the first row of each
    # group -> nearest-range point per (beam, azimuth) cell. Packing both into
    # one float key (range < max_range + 1 so cells can't overlap) lets a single
    # argsort replace a 2-key lexsort, ~2x faster.
    order = np.argsort(cell_key * (sensor.max_range_m + 1.0) + r_k)
    cell_key_sorted = cell_key[order]
    _, first_pos = np.unique(cell_key_sorted, return_index=True)
    winners = idx[order[first_pos]]

    points_sensor_out = points_sensor[winners]
    points_world_out = splat.centers[winners]

    intensity = splat.opacity[winners].copy() if splat.opacity is not None else None
    color = splat.color[winners].copy() if splat.color is not None else None

    out_points = points_sensor_out if return_frame == "sensor" else points_world_out

    return ScanResult(
        points=out_points,
        ranges=r[winners],
        beam_index=beam_index[winners],
        azimuth_bin=azimuth_bin[winners],
        intensity=intensity,
        color=color,
        num_input_points=num_input,
        num_output_points=len(winners),
        total_cells=total_cells,
        timestamp=np.full(len(winners), timestamp) if timestamp is not None else None,
    )


def concatenate_scans(scans: Sequence[ScanResult]) -> ScanResult:
    """Merge scans taken at different poses/times (e.g. samples along a
    Trajectory) into one ScanResult, for a single combined export."""
    def cat(field):
        parts = [getattr(s, field) for s in scans]
        return np.concatenate(parts) if all(p is not None for p in parts) else None

    return ScanResult(
        points=np.concatenate([s.points for s in scans]),
        ranges=np.concatenate([s.ranges for s in scans]),
        beam_index=np.concatenate([s.beam_index for s in scans]),
        azimuth_bin=np.concatenate([s.azimuth_bin for s in scans]),
        intensity=cat("intensity"),
        color=cat("color"),
        num_input_points=sum(s.num_input_points for s in scans),
        num_output_points=sum(s.num_output_points for s in scans),
        total_cells=sum(s.total_cells for s in scans),
        timestamp=cat("timestamp"),
    )
