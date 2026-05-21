"""Graph-Laplacian formation consensus and collision avoidance."""

from __future__ import annotations

import numpy as np


def formation_offsets(num_agents: int, shape: str = "circle", spacing: float = 3.0) -> np.ndarray:
    """Create an N x 3 geometric offset matrix for common formations."""

    if num_agents < 1:
        raise ValueError("num_agents must be at least 1")
    shape = shape.lower()

    if shape == "circle":
        angles = np.linspace(0.0, 2.0 * np.pi, num_agents, endpoint=False)
        radius = max(spacing, spacing * num_agents / (2.0 * np.pi))
        return np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.zeros(num_agents)))

    if shape == "square":
        side = int(np.ceil(np.sqrt(num_agents)))
        xs, ys = np.meshgrid(np.arange(side), np.arange(side))
        offsets = np.column_stack((xs.ravel()[:num_agents], ys.ravel()[:num_agents], np.zeros(num_agents)))
        offsets[:, :2] -= offsets[:, :2].mean(axis=0)
        return offsets * spacing

    if shape in {"v", "v-formation", "v_formation"}:
        offsets = np.zeros((num_agents, 3), dtype=float)
        for idx in range(1, num_agents):
            side = -1.0 if idx % 2 else 1.0
            rank = (idx + 1) // 2
            offsets[idx] = np.array([-rank * spacing, side * rank * spacing, 0.0])
        offsets -= offsets.mean(axis=0)
        return offsets

    if shape == "line":
        xs = (np.arange(num_agents) - (num_agents - 1) / 2.0) * spacing
        return np.column_stack((xs, np.zeros(num_agents), np.zeros(num_agents)))

    raise ValueError("Unsupported shape. Use circle, square, v-formation, or line.")


def collision_avoidance(
    positions: np.ndarray,
    safety_radius: float = 1.5,
    gain: float = 1.2,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute pairwise artificial-potential repulsion vectors.

    Agents closer than ``safety_radius`` receive equal and opposite velocity
    corrections. Magnitude scales nonlinearly with 1 / distance^2 and fades to
    zero at the safety boundary.
    """

    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be an N x 3 matrix")

    delta = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    mask = (distances > epsilon) & (distances < safety_radius)
    safe_distances = np.where(mask, distances, 1.0)
    directions = delta / safe_distances[:, :, None]
    magnitude = np.where(mask, gain * (1.0 / safe_distances**2 - 1.0 / safety_radius**2), 0.0)
    return np.sum(magnitude[:, :, None] * directions, axis=1)


class FormationConsensusController:
    """Formation-preserving consensus controller.

    Implements the requested law:

        dot(X) = -L * (X - C)

    where X is the N x 3 position matrix and C is the N x 3 structural offset
    matrix. Collision avoidance is added directly to the velocity command.
    """

    def __init__(
        self,
        laplacian: np.ndarray,
        offsets: np.ndarray,
        gain: float = 1.0,
        centroid_anchor_gain: float = 0.0,
        safety_radius: float = 1.5,
        repulsion_gain: float = 1.2,
        max_command_speed: float = 5.0,
    ) -> None:
        self.laplacian = np.asarray(laplacian, dtype=float)
        self.offsets = np.asarray(offsets, dtype=float)
        self.gain = gain
        self.centroid_anchor_gain = centroid_anchor_gain
        self.safety_radius = safety_radius
        self.repulsion_gain = repulsion_gain
        self.max_command_speed = max_command_speed
        self._validate_dimensions()

    def update_laplacian(self, laplacian: np.ndarray) -> None:
        self.laplacian = np.asarray(laplacian, dtype=float)
        self._validate_dimensions()

    def set_offsets(self, offsets: np.ndarray) -> None:
        self.offsets = np.asarray(offsets, dtype=float)
        self._validate_dimensions()

    def commands(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=float)
        formation_error = positions - self.offsets
        consensus = -self.gain * (self.laplacian @ formation_error)
        centroid_error = self.offsets.mean(axis=0) - positions.mean(axis=0)
        anchor = self.centroid_anchor_gain * centroid_error
        repulsion = collision_avoidance(positions, self.safety_radius, self.repulsion_gain)
        return self._limit_rows(consensus + anchor + repulsion, self.max_command_speed)

    def _validate_dimensions(self) -> None:
        if self.laplacian.ndim != 2 or self.laplacian.shape[0] != self.laplacian.shape[1]:
            raise ValueError("laplacian must be an N x N matrix")
        if self.offsets.shape != (self.laplacian.shape[0], 3):
            raise ValueError("offsets must be an N x 3 matrix matching the Laplacian")

    @staticmethod
    def _limit_rows(vectors: np.ndarray, max_norm: float) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        scale = np.where((norms > max_norm) & (norms > 0.0), max_norm / norms, 1.0)
        return vectors * scale
