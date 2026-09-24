"""Small pose-math helpers shared by the UI and any scripting use."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


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


def ypr_from_rotation(rotation: np.ndarray) -> Tuple[float, float, float]:
    """Inverse of rotation_from_ypr: (yaw_deg, pitch_deg, roll_deg) for a
    world<-sensor rotation matrix, e.g. to show an interpolated Trajectory
    pose in yaw/pitch/roll UI fields."""
    yaw, pitch, roll = Rotation.from_matrix(rotation).as_euler("ZYX", degrees=True)
    return float(yaw), float(pitch), float(roll)


# Keyframe: (t, position (3,), yaw_deg, pitch_deg, roll_deg)
Keyframe = Tuple[float, Sequence[float], float, float, float]


@dataclass
class Trajectory:
    """A sensor path as timestamped keyframes, sampled by linear position
    interpolation + quaternion slerp for orientation (no gimbal lock)."""

    times: np.ndarray       # (K,) strictly increasing
    positions: np.ndarray   # (K, 3)
    rotations: Rotation     # scipy Rotation stack, length K

    @classmethod
    def from_keyframes(cls, keyframes: Sequence[Keyframe]) -> "Trajectory":
        keyframes = sorted(keyframes, key=lambda k: k[0])
        times = np.array([k[0] for k in keyframes], dtype=np.float64)
        positions = np.array([k[1] for k in keyframes], dtype=np.float64)
        rotations = Rotation.from_matrix(
            np.stack([rotation_from_ypr(k[2], k[3], k[4]) for k in keyframes])
        )
        return cls(times=times, positions=positions, rotations=rotations)

    def pose_at(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolated (position, rotation_matrix) at time t, clamped to
        [times[0], times[-1]]."""
        t_clamped = float(np.clip(t, self.times[0], self.times[-1]))
        position = np.array([np.interp(t_clamped, self.times, self.positions[:, i]) for i in range(3)])
        if len(self.times) == 1:
            return position, self.rotations.as_matrix()[0]
        rotation = Slerp(self.times, self.rotations)(t_clamped).as_matrix()
        return position, rotation
