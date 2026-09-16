"""
quickpointforge
================

Simulate a LiDAR-style scan directly from 3D Gaussian Splat *centers*,
without ray-casting or alpha-blended rasterization.

Core idea: treat Gaussian centers as if they were a dense photogrammetry
point cloud, project them into the sensor's spherical frame, bin them by
the target sensor's real beam layout, and keep the nearest-range point
per (beam, azimuth) bin -- the cheap substitute for a proper first-return
ray-splat intersection.

See README.md for the accuracy/speed tradeoff this implies.
"""

from .io import load_gaussian_ply, GaussianSplat
from .sensors import (
    SensorModel,
    VELODYNE_VLP16,
    VELODYNE_VLP32C_APPROX,
    VELODYNE_HDL32E_APPROX,
    VELODYNE_HDL64E_APPROX,
    OUSTER_OS0_128_APPROX,
    OUSTER_OS1_64_APPROX,
    OUSTER_OS2_128_APPROX,
    HESAI_PANDAR64_APPROX,
    FLASH_LIDAR_EXAMPLE,
    FLASH_LIDAR_NARROW_LONGRANGE_EXAMPLE,
    generic_uniform_sensor,
    generic_flash_sensor,
)
from .simulate import simulate_lidar_scan, ScanResult

__all__ = [
    "load_gaussian_ply",
    "GaussianSplat",
    "SensorModel",
    "VELODYNE_VLP16",
    "VELODYNE_VLP32C_APPROX",
    "VELODYNE_HDL32E_APPROX",
    "VELODYNE_HDL64E_APPROX",
    "OUSTER_OS0_128_APPROX",
    "OUSTER_OS1_64_APPROX",
    "OUSTER_OS2_128_APPROX",
    "HESAI_PANDAR64_APPROX",
    "FLASH_LIDAR_EXAMPLE",
    "FLASH_LIDAR_NARROW_LONGRANGE_EXAMPLE",
    "generic_uniform_sensor",
    "generic_flash_sensor",
    "simulate_lidar_scan",
    "ScanResult",
]

__version__ = "0.1.0"
