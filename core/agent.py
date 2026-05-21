"""Agent state and point-mass kinematics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


ArrayLike3 = np.ndarray | list[float] | tuple[float, float, float]


@dataclass
class Agent:
    """A lightweight quadrotor abstraction using double-integrator kinematics.

    The simulated vehicle state is intentionally minimal: position and velocity
    in an inertial Cartesian frame. Controllers may send either desired velocity
    vectors or acceleration vectors. The default update treats controller output
    as acceleration, which is the point-mass double-integrator model:

        p_dot = v
        v_dot = u

    Velocity-command mode is also available for first-order consensus laws.
    """

    agent_id: int
    position: ArrayLike3
    velocity: ArrayLike3 | None = None
    max_speed: float = 6.0
    max_acceleration: float = 8.0
    history: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        if self.velocity is None:
            self.velocity = np.zeros(3, dtype=float)
        else:
            self.velocity = np.asarray(self.velocity, dtype=float).reshape(3)
        self.history.append(self.position.copy())

    @property
    def state(self) -> np.ndarray:
        """Return a compact six-state vector [x, y, z, vx, vy, vz]."""

        return np.concatenate((self.position, self.velocity))

    def step_acceleration(self, acceleration: ArrayLike3, dt: float) -> None:
        """Advance state with semi-implicit Euler integration."""

        acceleration = np.asarray(acceleration, dtype=float).reshape(3)
        acceleration = self._limit_norm(acceleration, self.max_acceleration)
        self.velocity = self._limit_norm(self.velocity + acceleration * dt, self.max_speed)
        self.position = self.position + self.velocity * dt
        self.history.append(self.position.copy())

    def step_velocity_command(self, velocity_command: ArrayLike3, dt: float) -> None:
        """Advance state while tracking a commanded velocity directly."""

        velocity_command = np.asarray(velocity_command, dtype=float).reshape(3)
        self.velocity = self._limit_norm(velocity_command, self.max_speed)
        self.position = self.position + self.velocity * dt
        self.history.append(self.position.copy())

    @staticmethod
    def _limit_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if max_norm <= 0.0 or norm <= max_norm or norm == 0.0:
            return vector
        return vector * (max_norm / norm)


def stack_positions(agents: list[Agent]) -> np.ndarray:
    """Return agent positions as an N x 3 matrix."""

    return np.vstack([agent.position for agent in agents])


def stack_velocities(agents: list[Agent]) -> np.ndarray:
    """Return agent velocities as an N x 3 matrix."""

    return np.vstack([agent.velocity for agent in agents])

