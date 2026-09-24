import numpy as np

from quickpointforge.pose import rotation_from_ypr, Trajectory


def test_identity_pose():
    R = rotation_from_ypr(0.0, 0.0, 0.0)
    assert np.allclose(R, np.eye(3), atol=1e-10)


def test_yaw_90_maps_x_to_y():
    R = rotation_from_ypr(90.0, 0.0, 0.0)
    v = R @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(v, [0.0, 1.0, 0.0], atol=1e-6)


def test_pitch_90_maps_x_to_negative_z():
    R = rotation_from_ypr(0.0, 90.0, 0.0)
    v = R @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(v, [0.0, 0.0, -1.0], atol=1e-6)


def test_rotation_matrix_is_orthonormal():
    R = rotation_from_ypr(37.0, -22.0, 8.0)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_trajectory_interpolates_position_and_clamps_ends():
    traj = Trajectory.from_keyframes([
        (0.0, [0.0, 0.0, 0.0], 0.0, 0.0, 0.0),
        (10.0, [10.0, 0.0, 0.0], 90.0, 0.0, 0.0),
    ])
    pos_mid, _ = traj.pose_at(5.0)
    assert np.allclose(pos_mid, [5.0, 0.0, 0.0])
    pos_before, _ = traj.pose_at(-5.0)
    assert np.allclose(pos_before, [0.0, 0.0, 0.0])
    pos_after, _ = traj.pose_at(50.0)
    assert np.allclose(pos_after, [10.0, 0.0, 0.0])


def test_trajectory_slerps_orientation_at_midpoint():
    traj = Trajectory.from_keyframes([
        (0.0, [0.0, 0.0, 0.0], 0.0, 0.0, 0.0),
        (10.0, [0.0, 0.0, 0.0], 90.0, 0.0, 0.0),
    ])
    _, rotation_mid = traj.pose_at(5.0)
    v = rotation_mid @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(v, [np.cos(np.radians(45.0)), np.sin(np.radians(45.0)), 0.0], atol=1e-6)


def test_trajectory_single_keyframe_returns_constant_pose():
    traj = Trajectory.from_keyframes([(0.0, [1.0, 2.0, 3.0], 30.0, 0.0, 0.0)])
    pos, rotation = traj.pose_at(999.0)
    assert np.allclose(pos, [1.0, 2.0, 3.0])
    assert np.allclose(rotation, rotation_from_ypr(30.0, 0.0, 0.0))
