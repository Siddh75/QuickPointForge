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

import numpy as np


@dataclass
class SensorModel:
    name: str
    beam_elevations_deg: np.ndarray   # (num_beams,) fixed vertical angles, sensor frame
    azimuth_resolution_deg: float     # angular step between azimuth bins
    max_range_m: float = 120.0
    min_range_m: float = 0.5
    elevation_tolerance_deg: float = field(default=None)  # None -> auto (half min spacing)

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
        return int(round(360.0 / self.azimuth_resolution_deg))


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
