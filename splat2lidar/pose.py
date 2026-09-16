"""Small pose-math helpers shared by the UI and any scripting use."""

import numpy as np


def rotation_from_ypr(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Build a world<-sensor rotation matrix from yaw/pitch/roll (degrees),
    intrinsic Z-Y-X convention (yaw about world Z, then pitch about the
    new Y, then roll about the new X) -- the common robotics convention.
    """
    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])

    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)

    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

    return rz @ ry @ rx
