"""Keyframe parser and waypoint interpolation for simulated drone shows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ShowKeyframe:
    """A structural swarm configuration at an absolute simulation time."""

    time: float
    positions: np.ndarray


def make_structure(name: str, num_agents: int, spacing: float = 3.0) -> np.ndarray:
    """Generate structural positions for named show keyframes."""

    if num_agents < 1:
        raise ValueError("num_agents must be at least 1")
    name = name.lower()

    if name == "grid":
        side = int(np.ceil(np.sqrt(num_agents)))
        xs, ys = np.meshgrid(np.arange(side), np.arange(side))
        positions = np.column_stack((xs.ravel()[:num_agents], ys.ravel()[:num_agents], np.zeros(num_agents)))
        positions[:, :2] -= positions[:, :2].mean(axis=0)
        positions *= spacing
        positions[:, 2] = 2.5
        return positions

    if name == "sphere":
        idx = np.arange(num_agents)
        phi = np.arccos(1.0 - 2.0 * (idx + 0.5) / num_agents)
        theta = np.pi * (1.0 + np.sqrt(5.0)) * idx
        radius = spacing * max(1.0, np.sqrt(num_agents) / 2.0)
        return np.column_stack(
            (
                radius * np.sin(phi) * np.cos(theta),
                radius * np.sin(phi) * np.sin(theta),
                4.0 + radius * np.cos(phi),
            )
        )

    if name == "cube":
        side = int(np.ceil(num_agents ** (1.0 / 3.0)))
        coords = np.array(np.meshgrid(np.arange(side), np.arange(side), np.arange(side))).T.reshape(-1, 3)
        positions = coords[:num_agents].astype(float)
        positions -= positions.mean(axis=0)
        positions *= spacing
        positions[:, 2] += 4.0
        return positions

    raise ValueError("Unsupported structure. Use grid, sphere, or cube.")


class ShowParser:
    """Interpolates time-stamped structural keyframes into smooth waypoints."""

    def __init__(self, keyframes: Iterable[ShowKeyframe]) -> None:
        self.keyframes = sorted(keyframes, key=lambda frame: frame.time)
        if len(self.keyframes) < 2:
            raise ValueError("At least two keyframes are required")
        self._validate()

    @classmethod
    def from_schedule(
        cls,
        schedule: Iterable[tuple[float, str]],
        num_agents: int,
        spacing: float = 3.0,
    ) -> "ShowParser":
        frames = [
            ShowKeyframe(time=float(time), positions=make_structure(name, num_agents, spacing))
            for time, name in schedule
        ]
        return cls(frames)

    def desired_positions(self, t: float) -> np.ndarray:
        """Return smoothly interpolated N x 3 positions at time t."""

        if t <= self.keyframes[0].time:
            return self.keyframes[0].positions.copy()
        if t >= self.keyframes[-1].time:
            return self.keyframes[-1].positions.copy()

        for start, end in zip(self.keyframes[:-1], self.keyframes[1:]):
            if start.time <= t <= end.time:
                tau = (t - start.time) / (end.time - start.time)
                smooth_tau = tau * tau * (3.0 - 2.0 * tau)
                return (1.0 - smooth_tau) * start.positions + smooth_tau * end.positions

        return self.keyframes[-1].positions.copy()

    @property
    def end_time(self) -> float:
        return self.keyframes[-1].time

    def _validate(self) -> None:
        shapes = {frame.positions.shape for frame in self.keyframes}
        if len(shapes) != 1:
            raise ValueError("All keyframes must have the same N x 3 shape")
        if next(iter(shapes))[1] != 3:
            raise ValueError("Keyframe positions must be N x 3 matrices")
        times = [frame.time for frame in self.keyframes]
        if len(set(times)) != len(times):
            raise ValueError("Keyframe times must be unique")

