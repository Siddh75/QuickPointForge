import numpy as np

from quickpointforge.pose import rotation_from_ypr


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
