"""Leader-follower trajectory tracking controllers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from controllers.consensus import collision_avoidance


@dataclass
class Trajectory:
    """Continuous reference trajectory for the leader."""

    kind: str = "helix"
    radius: float = 6.0
    altitude: float = 4.0
    omega: float = 0.35

    def state(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        kind = self.kind.lower()
        if kind == "helix":
            position = np.array(
                [
                    self.radius * np.cos(self.omega * t),
                    self.radius * np.sin(self.omega * t),
                    self.altitude + 0.25 * t,
                ],
                dtype=float,
            )
            velocity = np.array(
                [
                    -self.radius * self.omega * np.sin(self.omega * t),
                    self.radius * self.omega * np.cos(self.omega * t),
                    0.25,
                ],
                dtype=float,
            )
            return position, velocity

        if kind == "lissajous":
            position = np.array(
                [
                    self.radius * np.sin(0.7 * self.omega * t + np.pi / 2.0),
                    self.radius * np.sin(self.omega * t),
                    self.altitude + 2.0 * np.sin(0.45 * self.omega * t),
                ],
                dtype=float,
            )
            velocity = np.array(
                [
                    self.radius * 0.7 * self.omega * np.cos(0.7 * self.omega * t + np.pi / 2.0),
                    self.radius * self.omega * np.cos(self.omega * t),
                    2.0 * 0.45 * self.omega * np.cos(0.45 * self.omega * t),
                ],
                dtype=float,
            )
            return position, velocity

        raise ValueError("Unsupported trajectory. Use helix or lissajous.")


class LeaderFollowerController:
    """Pinned consensus controller for one trajectory-tracking leader."""

    def __init__(
        self,
        laplacian: np.ndarray,
        offsets: np.ndarray,
        leader_id: int = 0,
        trajectory: Trajectory | None = None,
        formation_gain: float = 1.0,
        leader_gain: float = 1.5,
        safety_radius: float = 1.5,
        repulsion_gain: float = 1.2,
        max_command_speed: float = 6.0,
    ) -> None:
        self.laplacian = np.asarray(laplacian, dtype=float)
        self.offsets = np.asarray(offsets, dtype=float)
        self.leader_id = leader_id
        self.trajectory = trajectory or Trajectory()
        self.formation_gain = formation_gain
        self.leader_gain = leader_gain
        self.safety_radius = safety_radius
        self.repulsion_gain = repulsion_gain
        self.max_command_speed = max_command_speed
        self._validate()

    def commands(self, positions: np.ndarray, t: float) -> np.ndarray:
        positions = np.asarray(positions, dtype=float)
        leader_position, leader_velocity = self.trajectory.state(t)
        target_offsets = leader_position + self.offsets - self.offsets[self.leader_id]

        formation_error = positions - target_offsets
        commands = -self.formation_gain * (self.laplacian @ formation_error)

        leader_error = leader_position - positions[self.leader_id]
        commands[self.leader_id] = leader_velocity + self.leader_gain * leader_error

        followers = np.arange(positions.shape[0]) != self.leader_id
        commands[followers] += self.leader_gain * (target_offsets[followers] - positions[followers])
        commands += collision_avoidance(positions, self.safety_radius, self.repulsion_gain)
        return self._limit_rows(commands, self.max_command_speed)

    def _validate(self) -> None:
        if self.laplacian.ndim != 2 or self.laplacian.shape[0] != self.laplacian.shape[1]:
            raise ValueError("laplacian must be an N x N matrix")
        if self.offsets.shape != (self.laplacian.shape[0], 3):
            raise ValueError("offsets must be an N x 3 matrix")
        if not 0 <= self.leader_id < self.laplacian.shape[0]:
            raise ValueError("leader_id is out of range")

    @staticmethod
    def _limit_rows(vectors: np.ndarray, max_norm: float) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        scale = np.where((norms > max_norm) & (norms > 0.0), max_norm / norms, 1.0)
        return vectors * scale

