import numpy as np

from splat2lidar.io import GaussianSplat
from splat2lidar.sensors import generic_uniform_sensor, generic_flash_sensor
from splat2lidar.simulate import simulate_lidar_scan, world_to_sensor


def _flat_sensor(num_beams=1, azimuth_resolution_deg=90.0, max_range_m=100.0, min_range_m=0.1):
    """A trivial single-beam-at-0-degrees sensor with coarse azimuth bins,
    convenient for deterministic tests."""
    return generic_uniform_sensor(
        name="test",
        num_beams=num_beams,
        fov_min_deg=0.0,
        fov_max_deg=0.0,
        azimuth_resolution_deg=azimuth_resolution_deg,
        max_range_m=max_range_m,
        min_range_m=min_range_m,
    )


def test_nearest_point_wins_in_occluded_cell():
    # Two points along +x (same beam/azimuth cell): near one at x=2, far one at x=5.
    centers = np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    opacity = np.ones(2)
    splat = GaussianSplat(centers=centers, opacity=opacity, scale=None, color=None)

    sensor = _flat_sensor()
    scan = simulate_lidar_scan(splat, sensor, sensor_position=np.array([0.0, 0.0, 0.0]))

    assert scan.num_output_points == 1
    assert np.isclose(scan.ranges[0], 2.0)


def test_points_in_different_azimuth_bins_both_survive():
    # +x and -x land in different azimuth bins with 90-deg resolution.
    centers = np.array([[3.0, 0.0, 0.0], [-4.0, 0.0, 0.0]])
    opacity = np.ones(2)
    splat = GaussianSplat(centers=centers, opacity=opacity, scale=None, color=None)

    sensor = _flat_sensor()
    scan = simulate_lidar_scan(splat, sensor, sensor_position=np.array([0.0, 0.0, 0.0]))

    assert scan.num_output_points == 2
    assert set(np.round(scan.ranges, 3)) == {3.0, 4.0}


def test_range_limits_are_respected():
    centers = np.array([[0.05, 0.0, 0.0], [500.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    opacity = np.ones(3)
    splat = GaussianSplat(centers=centers, opacity=opacity, scale=None, color=None)

    sensor = _flat_sensor(min_range_m=0.5, max_range_m=100.0)
    scan = simulate_lidar_scan(splat, sensor, sensor_position=np.array([0.0, 0.0, 0.0]))

    # only the 10m point is within [0.5, 100] -- the 0.05m and 500m points are dropped
    assert scan.num_output_points == 1
    assert np.isclose(scan.ranges[0], 10.0)


def test_sensor_pose_translation_and_rotation_applied():
    # A point at world (5, 0, 0). Sensor sitting at world (5, 0, 0) offset
    # by +2 in x, facing +x still -- point should appear at sensor-relative x = -2.
    centers = np.array([[5.0, 0.0, 0.0]])
    opacity = np.ones(1)
    splat = GaussianSplat(centers=centers, opacity=opacity, scale=None, color=None)

    sensor_position = np.array([7.0, 0.0, 0.0])
    points_sensor = world_to_sensor(centers, sensor_position, np.eye(3))
    assert np.allclose(points_sensor, [[-2.0, 0.0, 0.0]])


def test_empty_result_when_nothing_in_tolerance():
    # A point far outside the single beam's elevation tolerance (90 deg up).
    centers = np.array([[0.0, 0.0, 10.0]])
    opacity = np.ones(1)
    splat = GaussianSplat(centers=centers, opacity=opacity, scale=None, color=None)

    sensor = _flat_sensor()
    scan = simulate_lidar_scan(splat, sensor, sensor_position=np.array([0.0, 0.0, 0.0]))

    assert scan.num_output_points == 0
    assert scan.points.shape == (0, 3)


def test_hit_rate_matches_total_cells():
    centers = np.array([[3.0, 0.0, 0.0]])
    opacity = np.ones(1)
    splat = GaussianSplat(centers=centers, opacity=opacity, scale=None, color=None)

    sensor = _flat_sensor(azimuth_resolution_deg=90.0)  # 4 azimuth bins, 1 beam -> 4 cells
    scan = simulate_lidar_scan(splat, sensor, sensor_position=np.array([0.0, 0.0, 0.0]))

    assert scan.total_cells == 4
    assert np.isclose(scan.hit_rate(), 1 / 4)


def _flash_sensor():
    """Single-beam flash sensor with a +/-30 deg azimuth FOV -- no wraparound."""
    return generic_flash_sensor(
        name="test-flash",
        num_beams=1,
        fov_min_deg=0.0,
        fov_max_deg=0.0,
        num_azimuth_beams=3,
        azimuth_fov_min_deg=-30.0,
        azimuth_fov_max_deg=30.0,
        max_range_m=100.0,
        min_range_m=0.1,
    )


def test_flash_sensor_drops_points_outside_fov():
    # Forward point (0 deg) is inside the +/-30 deg FOV; the behind point
    # (180 deg) would wrap into a spinning sensor's grid but must be
    # dropped outright for a flash sensor.
    centers = np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    opacity = np.ones(2)
    splat = GaussianSplat(centers=centers, opacity=opacity, scale=None, color=None)

    sensor = _flash_sensor()
    scan = simulate_lidar_scan(splat, sensor, sensor_position=np.array([0.0, 0.0, 0.0]))

    assert scan.num_output_points == 1
    assert np.isclose(scan.ranges[0], 5.0)


def test_flash_sensor_num_azimuth_bins_matches_grid():
    sensor = _flash_sensor()
    assert sensor.num_azimuth_bins == 3
