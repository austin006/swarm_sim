"""Runtime entry point for the lightweight quadrotor swarm simulator."""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

from controllers.consensus import FormationConsensusController, formation_offsets
from controllers.leader_follower import LeaderFollowerController, Trajectory
from core.agent import Agent, stack_positions
from core.network import CommunicationNetwork
from utils.show_parser import ShowParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure-Python quadrotor swarm simulator")
    parser.add_argument("--mode", choices=["consensus", "leader-follower", "show"], default="consensus")
    parser.add_argument("--agents", type=int, default=12, help="Number of simulated agents")
    parser.add_argument("--topology", choices=["all-to-all", "ring", "line"], default="ring")
    parser.add_argument("--shape", choices=["circle", "square", "v-formation", "line"], default="circle")
    parser.add_argument("--trajectory", choices=["helix", "lissajous"], default="helix")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation step in seconds")
    parser.add_argument("--spacing", type=float, default=3.0, help="Formation spacing in meters")
    parser.add_argument("--safety-radius", type=float, default=1.5, help="Collision guard radius in meters")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for initial conditions")
    parser.add_argument("--no-animate", action="store_true", help="Run headless and print final positions")
    parser.add_argument("--save-plot", type=Path, help="Save a final 3D position plot to this path")
    parser.add_argument("--save-animation", type=Path, help="Save the full simulation as a GIF")
    parser.add_argument("--animation-fps", type=int, default=25, help="Saved animation frames per second")
    return parser.parse_args()


def initialize_agents(num_agents: int, seed: int) -> list[Agent]:
    rng = np.random.default_rng(seed)
    positions = rng.normal(loc=0.0, scale=4.0, size=(num_agents, 3))
    positions[:, 2] = rng.uniform(1.0, 5.0, size=num_agents)
    return [Agent(agent_id=i, position=positions[i]) for i in range(num_agents)]


def configure_axes(ax: plt.Axes, limit: float) -> None:
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(0.0, limit)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.view_init(elev=24.0, azim=42.0)


def set_equalish_limits(ax: plt.Axes, positions: np.ndarray, margin: float = 4.0) -> None:
    span = np.max(np.ptp(positions, axis=0)) + margin
    center = positions.mean(axis=0)
    half = max(span / 2.0, 8.0)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(max(0.0, center[2] - half), center[2] + half)


def save_final_plot(
    path: Path,
    positions: np.ndarray,
    title: str,
    target_positions: np.ndarray | None = None,
) -> None:
    """Write a final 3D swarm plot for headless container runs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        s=55,
        c=np.arange(positions.shape[0]),
        cmap="viridis",
        label="agents",
    )
    if target_positions is not None:
        ax.scatter(
            target_positions[:, 0],
            target_positions[:, 1],
            target_positions[:, 2],
            marker="x",
            s=36,
            c="black",
            alpha=0.55,
            label="targets",
        )
    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.view_init(elev=24.0, azim=42.0)
    set_equalish_limits(ax, positions if target_positions is None else np.vstack((positions, target_positions)))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_animation(
    path: Path,
    position_frames: list[np.ndarray],
    target_frames: list[np.ndarray | None],
    title: str,
    fps: int,
) -> None:
    """Write a GIF animation for headless container playback."""

    if not position_frames:
        raise ValueError("No animation frames were recorded")

    path.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1, fps)
    all_positions = np.vstack(
        [
            frame if target is None else np.vstack((frame, target))
            for frame, target in zip(position_frames, target_frames)
        ]
    )

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    first_positions = position_frames[0]
    scatter = ax.scatter(
        first_positions[:, 0],
        first_positions[:, 1],
        first_positions[:, 2],
        s=55,
        c=np.arange(first_positions.shape[0]),
        cmap="viridis",
    )
    target_scatter = ax.scatter([], [], [], marker="x", s=32, c="black", alpha=0.45)
    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.view_init(elev=24.0, azim=42.0)
    set_equalish_limits(ax, all_positions)

    def update(frame_index: int):
        positions = position_frames[frame_index]
        scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

        targets = target_frames[frame_index]
        if targets is None:
            target_scatter._offsets3d = ([], [], [])
        else:
            target_scatter._offsets3d = (targets[:, 0], targets[:, 1], targets[:, 2])

        ax.set_title(f"{title} | frame {frame_index + 1}/{len(position_frames)}")
        return scatter, target_scatter

    swarm_animation = animation.FuncAnimation(
        fig,
        update,
        frames=len(position_frames),
        interval=1000.0 / fps,
        blit=False,
    )
    writer = animation.PillowWriter(fps=fps)
    swarm_animation.save(path, writer=writer)
    plt.close(fig)


def build_controller(args: argparse.Namespace, network: CommunicationNetwork):
    offsets = formation_offsets(args.agents, args.shape, args.spacing)
    _, _, laplacian = network.matrices()

    if args.mode == "consensus":
        return FormationConsensusController(
            laplacian=laplacian,
            offsets=offsets,
            safety_radius=args.safety_radius,
        )

    if args.mode == "leader-follower":
        return LeaderFollowerController(
            laplacian=laplacian,
            offsets=offsets,
            trajectory=Trajectory(kind=args.trajectory),
            safety_radius=args.safety_radius,
        )

    show = ShowParser.from_schedule(
        [(0.0, "grid"), (5.0, "sphere"), (10.0, "cube"), (15.0, "grid")],
        num_agents=args.agents,
        spacing=args.spacing,
    )
    return FormationConsensusController(
        laplacian=laplacian,
        offsets=show.desired_positions(0.0),
        gain=1.2,
        centroid_anchor_gain=0.8,
        safety_radius=args.safety_radius,
    ), show


def run_simulation(args: argparse.Namespace) -> np.ndarray:
    if args.dt <= 0.0:
        raise ValueError("dt must be positive")
    if args.agents < 1:
        raise ValueError("agents must be at least 1")
    if args.animation_fps < 1:
        raise ValueError("animation_fps must be at least 1")

    agents = initialize_agents(args.agents, args.seed)
    network = CommunicationNetwork(args.agents, args.topology)
    controller_config = build_controller(args, network)
    show = None
    if isinstance(controller_config, tuple):
        controller, show = controller_config
    else:
        controller = controller_config
    final_targets = None

    fig = ax = scatter = target_scatter = None
    if not args.no_animate:
        plt.ion()
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        positions = stack_positions(agents)
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=55, c=np.arange(args.agents), cmap="viridis")
        target_scatter = ax.scatter([], [], [], marker="x", s=32, c="black", alpha=0.45)
        configure_axes(ax, limit=max(12.0, args.spacing * args.agents / 2.0))

    steps = int(np.ceil(args.duration / args.dt))
    render_stride = max(1, int(round((1.0 / args.animation_fps) / args.dt)))
    position_frames: list[np.ndarray] = []
    target_frames: list[np.ndarray | None] = []

    for step in range(steps):
        t = step * args.dt
        positions = stack_positions(agents)

        if args.mode == "leader-follower":
            commands = controller.commands(positions, t)
            leader_position, _ = controller.trajectory.state(t)
            target_positions = leader_position + controller.offsets - controller.offsets[controller.leader_id]
        elif args.mode == "show":
            target_positions = show.desired_positions(t)
            controller.set_offsets(target_positions)
            commands = controller.commands(positions)
        else:
            commands = controller.commands(positions)
            target_positions = controller.offsets
        final_targets = target_positions

        for agent, command in zip(agents, commands):
            agent.step_velocity_command(command, args.dt)

        should_record = args.save_animation is not None and step % render_stride == 0
        if should_record:
            position_frames.append(stack_positions(agents))
            target_frames.append(None if target_positions is None else target_positions.copy())

        if not args.no_animate and step % render_stride == 0:
            positions = stack_positions(agents)
            scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
            if target_positions is not None:
                target_scatter._offsets3d = (
                    target_positions[:, 0],
                    target_positions[:, 1],
                    target_positions[:, 2],
                )
            ax.set_title(f"{args.mode} | t={t:5.2f}s | topology={args.topology}")
            set_equalish_limits(ax, positions)
            fig.canvas.draw_idle()
            plt.pause(0.001)

    final_positions = stack_positions(agents)
    if args.save_plot is not None:
        save_final_plot(
            args.save_plot,
            final_positions,
            title=f"{args.mode} final state | topology={args.topology}",
            target_positions=final_targets,
        )
    if args.save_animation is not None:
        save_animation(
            args.save_animation,
            position_frames,
            target_frames,
            title=f"{args.mode} | topology={args.topology}",
            fps=args.animation_fps,
        )
    if not args.no_animate:
        plt.ioff()
        plt.show()
    return final_positions


def main() -> None:
    args = parse_args()
    final_positions = run_simulation(args)
    if args.no_animate:
        np.set_printoptions(precision=3, suppress=True)
        print(final_positions)


if __name__ == "__main__":
    main()
