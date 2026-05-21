# Lightweight Quadrotor Swarm Simulator

Pure-Python multi-agent swarm simulator for prototyping GNC control laws before moving into ROS 2/PX4/Gazebo.

## Stack

Only lightweight scientific Python packages are used:

- `numpy`
- `scipy`
- `networkx`
- `matplotlib`

## Run

```powershell
pip install -r requirements.txt
python main_simulation.py --mode consensus --agents 12 --topology ring --shape circle
python main_simulation.py --mode leader-follower --trajectory helix
python main_simulation.py --mode show
```

For a headless smoke test:

```powershell
python main_simulation.py --mode consensus --duration 1 --no-animate
```

Save a final visual snapshot, useful in headless Docker/VS Code container runs:

```powershell
python main_simulation.py --mode show --duration 10 --no-animate --save-plot outputs/show.png
```

Save the full simulation as a GIF:

```powershell
python main_simulation.py --mode show --duration 15 --no-animate --save-animation outputs/show.gif
```

The saved GIF camera pans at a constant slow speed by default: `6` degrees per second, so a full 360-degree orbit takes about one minute. Override the speed or force a total sweep if desired:

```powershell
python main_simulation.py --mode leader-follower --duration 15 --no-animate --save-animation outputs/leader_follower.gif --animation-rotation-rate 3
python main_simulation.py --mode leader-follower --duration 60 --no-animate --save-animation outputs/leader_follower_360.gif --animation-rotation 360
```

## Docker / VS Code Dev Container

This repository includes a `Dockerfile` and `.devcontainer/devcontainer.json` so Python does not need to be installed on the host.

### VS Code

1. Install Docker Desktop.
2. Install the VS Code extension `Dev Containers`.
3. Open this folder in VS Code.
4. Run `Dev Containers: Reopen in Container` from the Command Palette.
5. Use the built-in terminal inside the container:

```bash
python main_simulation.py --mode consensus --duration 1 --no-animate
python main_simulation.py --mode show --duration 10 --no-animate --save-plot outputs/show.png
python main_simulation.py --mode show --duration 15 --no-animate --save-animation outputs/show.gif
```

The container uses `MPLBACKEND=Agg`, so saved plots and GIF animations work without GUI forwarding. In VS Code, you can also run `Terminal > Run Task... > Watch: show GIF`.

### Plain Docker

From the repository root:

```powershell
docker build -t swarm-sim .
docker run --rm -v ${PWD}:/workspace/swarm_sim swarm-sim python main_simulation.py --mode consensus --duration 1 --no-animate
docker run --rm -v ${PWD}:/workspace/swarm_sim swarm-sim python main_simulation.py --mode show --duration 10 --no-animate --save-plot outputs/show.png
docker run --rm -v ${PWD}:/workspace/swarm_sim swarm-sim python main_simulation.py --mode show --duration 15 --no-animate --save-animation outputs/show.gif
```

## Modes

- `consensus`: graph-Laplacian formation consensus, `dot(X) = -L * (X - C)`.
- `leader-follower`: leader tracks a helix or Lissajous trajectory while followers are pinned to the leader-relative formation.
- `show`: keyframed structural morphing through grid, sphere, and cube formations.

All modes include active artificial-potential collision avoidance when agents breach the configured safety radius.
