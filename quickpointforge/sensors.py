"""
Sensor beam layout models.

A SensorModel describes the fixed elevation angles ("channels") and
azimuth sampling resolution of a real spinning LiDAR. Real per-unit beam
tables (factory-calibrated) differ slightly from these idealized
uniform/near-uniform layouts -- if you have the actual calibration file
for your target sensor, load it and build a SensorModel with those exact
angles instead of using the presets below.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class SensorModel:
    name: str
    beam_elevations_deg: np.ndarray   # (num_beams,) fixed vertical angles, sensor frame
    azimuth_resolution_deg: float     # angular step between azimuth bins/columns
    max_range_m: float = 120.0
    min_range_m: float = 0.5
    elevation_tolerance_deg: float = field(default=None)  # None -> auto (half min spacing)
    # None -> spinning sensor, azimuth wraps over the full 360 deg circle.
    # (min_deg, max_deg) -> flash-style sensor: fixed rectangular FOV, no
    # wraparound, azimuth measured the same way as elevation (0 = forward).
    azimuth_fov_deg: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        self.beam_elevations_deg = np.asarray(self.beam_elevations_deg, dtype=np.float64)
        if self.elevation_tolerance_deg is None:
            if len(self.beam_elevations_deg) > 1:
                spacing = np.diff(np.sort(self.beam_elevations_deg))
                self.elevation_tolerance_deg = float(spacing.min()) / 2.0
            else:
                self.elevation_tolerance_deg = 1.0

    @property
    def num_beams(self) -> int:
        return len(self.beam_elevations_deg)

    @property
    def num_azimuth_bins(self) -> int:
        if self.azimuth_fov_deg is None:
            return int(round(360.0 / self.azimuth_resolution_deg))
        fov_min, fov_max = self.azimuth_fov_deg
        return int(round((fov_max - fov_min) / self.azimuth_resolution_deg)) + 1


def generic_uniform_sensor(
    name: str,
    num_beams: int,
    fov_min_deg: float,
    fov_max_deg: float,
    azimuth_resolution_deg: float,
    max_range_m: float = 120.0,
    min_range_m: float = 0.5,
) -> SensorModel:
    """Build a SensorModel with beams spaced uniformly across a vertical FOV."""
    elevations = np.linspace(fov_min_deg, fov_max_deg, num_beams)
    return SensorModel(
        name=name,
        beam_elevations_deg=elevations,
        azimuth_resolution_deg=azimuth_resolution_deg,
        max_range_m=max_range_m,
        min_range_m=min_range_m,
    )


def generic_flash_sensor(
    name: str,
    num_beams: int,
    fov_min_deg: float,
    fov_max_deg: float,
    num_azimuth_beams: int,
    azimuth_fov_min_deg: float,
    azimuth_fov_max_deg: float,
    max_range_m: float = 120.0,
    min_range_m: float = 0.5,
) -> SensorModel:
    """
    Build a SensorModel for a flash-style sensor: a fixed rectangular grid
    of beams over a limited (non-wrapping) FOV, all captured in one shot --
    as opposed to a spinning sensor sweeping a full 360 deg circle over time.
    """
    elevations = np.linspace(fov_min_deg, fov_max_deg, num_beams)
    azimuth_resolution_deg = (azimuth_fov_max_deg - azimuth_fov_min_deg) / max(num_azimuth_beams - 1, 1)
    return SensorModel(
        name=name,
        beam_elevations_deg=elevations,
        azimuth_resolution_deg=azimuth_resolution_deg,
        max_range_m=max_range_m,
        min_range_m=min_range_m,
        azimuth_fov_deg=(azimuth_fov_min_deg, azimuth_fov_max_deg),
    )


# --- Presets -----------------------------------------------------------
# These are approximations of real sensor datasheets, using uniform
# spacing where the real unit is close-to-uniform. For HDL-64E and OS1-64,
# real factory beam tables are non-uniform -- swap in the exact
# calibration angles if precision matters for your use case.

VELODYNE_VLP16 = generic_uniform_sensor(
    name="Velodyne VLP-16",
    num_beams=16,
    fov_min_deg=-15.0,
    fov_max_deg=15.0,
    azimuth_resolution_deg=0.2,   # ~0.1-0.4 deg depending on RPM; 0.2 is a common default
    max_range_m=100.0,
)

VELODYNE_HDL64E_APPROX = generic_uniform_sensor(
    name="Velodyne HDL-64E (uniform approximation)",
    num_beams=64,
    fov_min_deg=-24.8,
    fov_max_deg=2.0,
    azimuth_resolution_deg=0.17,  # ~0.08-0.35 deg depending on RPM
    max_range_m=120.0,
)

OUSTER_OS1_64_APPROX = generic_uniform_sensor(
    name="Ouster OS1-64 (uniform approximation)",
    num_beams=64,
    fov_min_deg=-22.5,
    fov_max_deg=22.5,
    azimuth_resolution_deg=0.35,  # 1024 or 2048 columns/rev typical; 0.35 deg ~ 1024 cols
    max_range_m=120.0,
)

VELODYNE_VLP32C_APPROX = generic_uniform_sensor(
    name="Velodyne VLP-32C (uniform approximation)",
    num_beams=32,
    fov_min_deg=-25.0,
    fov_max_deg=15.0,
    azimuth_resolution_deg=0.2,
    max_range_m=200.0,
)

VELODYNE_HDL32E_APPROX = generic_uniform_sensor(
    name="Velodyne HDL-32E (uniform approximation)",
    num_beams=32,
    fov_min_deg=-30.67,
    fov_max_deg=10.67,
    azimuth_resolution_deg=0.2,
    max_range_m=100.0,
)

OUSTER_OS0_128_APPROX = generic_uniform_sensor(
    name="Ouster OS0-128 (uniform approximation)",
    num_beams=128,
    fov_min_deg=-45.0,
    fov_max_deg=45.0,   # ultra-wide vertical FOV variant, short range
    azimuth_resolution_deg=0.35,
    max_range_m=50.0,
)

OUSTER_OS2_128_APPROX = generic_uniform_sensor(
    name="Ouster OS2-128 (uniform approximation)",
    num_beams=128,
    fov_min_deg=-11.25,
    fov_max_deg=11.25,  # narrow vertical FOV variant, long range
    azimuth_resolution_deg=0.18,
    max_range_m=240.0,
)

HESAI_PANDAR64_APPROX = generic_uniform_sensor(
    name="Hesai Pandar64 (uniform approximation)",
    num_beams=64,
    fov_min_deg=-25.0,
    fov_max_deg=15.0,
    azimuth_resolution_deg=0.2,
    max_range_m=200.0,
)

# Illustrative flash LiDAR (not tied to a specific real product's exact
# datasheet): a fixed 64x64 grid over a camera-like FOV, single-shot,
# no 360 deg wraparound. Swap the FOV/grid numbers for your target unit.
FLASH_LIDAR_EXAMPLE = generic_flash_sensor(
    name="Flash LiDAR (example, 60x30 FOV)",
    num_beams=64,
    fov_min_deg=-15.0,
    fov_max_deg=15.0,
    num_azimuth_beams=64,
    azimuth_fov_min_deg=-30.0,
    azimuth_fov_max_deg=30.0,
    max_range_m=60.0,
)

# Second illustrative flash example: narrower FOV, longer range -- the
# automotive-forward-looking end of the flash LiDAR tradeoff space.
FLASH_LIDAR_NARROW_LONGRANGE_EXAMPLE = generic_flash_sensor(
    name="Flash LiDAR (example, 20x10 FOV, long range)",
    num_beams=48,
    fov_min_deg=-5.0,
    fov_max_deg=5.0,
    num_azimuth_beams=96,
    azimuth_fov_min_deg=-10.0,
    azimuth_fov_max_deg=10.0,
    max_range_m=150.0,
)
