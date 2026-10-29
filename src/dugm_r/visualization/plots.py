"""Visual checks for the public procedural 2D protocol."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import torch

from dugm_r.io import config_file, load_config, record_artifact, write_json
from dugm_r.nominal.train import make_env
from dugm_r.recovery.rvf import load_checkpoint
from dugm_r.simulation_2d.scene import BoxObstacle, CircleObstacle, Scene2D


def _draw_scene(ax, scene: Scene2D, *, trajectory: np.ndarray | None = None, modes: list[str] | None = None) -> None:
    ax.set_xlim(*scene.bounds_x_m)
    ax.set_ylim(*scene.bounds_y_m)
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    for obstacle in scene.static_obstacles:
        if isinstance(obstacle, CircleObstacle):
            ax.add_patch(Circle(obstacle.center, obstacle.radius_m, color="#555555", alpha=0.7))
        elif isinstance(obstacle, BoxObstacle):
            ax.add_patch(Rectangle((obstacle.center[0]-obstacle.size_m[0]/2, obstacle.center[1]-obstacle.size_m[1]/2), *obstacle.size_m, color="#555555", alpha=0.7))
    for obstacle in scene.dynamic_obstacles(0):
        if isinstance(obstacle, CircleObstacle):
            ax.add_patch(Circle(obstacle.center, obstacle.radius_m, color="#e99b3d", alpha=0.45))
    ax.plot(*scene.start_xy, marker="o", color="#218a49", markersize=7, label="start")
    ax.plot(*scene.goal_xy, marker="*", color="#bc3243", markersize=10, label="goal")
    if trajectory is not None and len(trajectory):
        xy = np.asarray(trajectory, dtype=float)
        if modes is None:
            ax.plot(xy[:, 0], xy[:, 1], color="#2661a8", linewidth=2, label="nominal")
        else:
            for i in range(1, len(xy)):
                mode = modes[min(i-1, len(modes)-1)] if modes else "nominal"
                ax.plot(xy[i-1:i+1, 0], xy[i-1:i+1, 1], color="#c23462" if mode == "recovery" else "#2661a8", linewidth=2)
            ax.plot([], [], color="#2661a8", label="nominal")
            ax.plot([], [], color="#c23462", label="recovery")
    ax.legend(loc="upper right", fontsize=8)


def save_trajectory(scene: Scene2D, trajectory: np.ndarray, modes: list[str], risks: list[float], path: Path, *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    _draw_scene(axes[0], scene, trajectory=trajectory, modes=modes)
    axes[0].set_title(title)
    axes[1].plot(risks, color="#7030a0", label="uncalibrated risk score")
    for index, mode in enumerate(modes):
        if mode == "recovery":
            axes[1].axvspan(index, index+1, color="#c23462", alpha=0.15)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("control step")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_curriculum(output: Path, *, config_name: str = "nominal_train_2d.yaml", seed: int = 2000000) -> list[Path]:
    cfg, cfg_path = load_config(config_name)
    curriculum = config_file(cfg["curriculum"], parent=cfg_path.parent)
    profile = config_file(cfg["robot_profile"], parent=cfg_path.parent)
    paths = []
    for index, stage in enumerate(cfg["stages"]):
        level = int(stage["level"])
        env = make_env(curriculum, profile, level=level, seed=seed + index, noise=False)
        try:
            env.reset(seed=seed + index)
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            _draw_scene(axes[0], env.scene)
            axes[0].set_title(f"{stage['id']} · example level {level}")
            mix = {int(key): float(value) for key, value in stage["level_mix"].items()}
            axes[1].bar([str(key) for key in mix], list(mix.values()), color="#2661a8")
            axes[1].set_ylim(0, 1)
            axes[1].set_xlabel("procedural level")
            axes[1].set_ylabel("sampling weight")
            axes[1].set_title("training scene mixture")
            fig.tight_layout()
            path = output / f"{stage['id']}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=140)
            plt.close(fig)
            paths.append(path)
        finally:
            env.close()
    write_json(output / "curriculum_manifest.json", {"protocol": "procedural_2d", "stages": [stage["id"] for stage in cfg["stages"]], "seed": seed, "files": [p.name for p in paths]})
    return paths


def render_observation(output: Path, *, config_name: str = "nominal_train_2d.yaml", level: int = 8, seed: int = 2000001, steps: int = 3) -> tuple[Path, Path]:
    cfg, cfg_path = load_config(config_name)
    env = make_env(config_file(cfg["curriculum"], parent=cfg_path.parent), config_file(cfg["robot_profile"], parent=cfg_path.parent), level=level, seed=seed, noise=False)
    try:
        obs, _ = env.reset(seed=seed)
        for _ in range(steps):
            obs, _, done, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
            if done or truncated:
                break
        grid = obs["dugm"].copy()
        static = env.last_static_cost.copy()
        dynamic = env.last_dynamic_cost.copy()
        state = obs["state"].copy()
    finally:
        env.close()
    output.mkdir(parents=True, exist_ok=True)
    arrays = output / "dugm_observation.npz"
    np.savez_compressed(arrays, dugm=grid, state=state, static_cost=static, dynamic_cost=dynamic)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, array, title in zip(axes, (grid[0], grid[1], static, dynamic), ("U_t · observed unsafe", "C_t · policy cost", "S_t · static endpoint", "D_t · dynamic prediction")):
        image = ax.imshow(array, vmin=0, vmax=1, origin="upper", cmap="magma")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(image, ax=axes, fraction=0.02)
    fig.savefig(output / "dugm_observation.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    record_artifact(output, "observation_arrays", arrays)
    record_artifact(output, "observation_plot", output / "dugm_observation.png")
    return arrays, output / "dugm_observation.png"


def render_rvf(rvf_checkpoint: Path, output: Path, *, level: int = 8, seed: int = 3000800,
               state_override: list[float] | None = None,
               config_name: str | Path = "nominal_train_2d.yaml") -> Path:
    cfg, cfg_path = load_config(config_name)
    env = make_env(config_file(cfg["curriculum"], parent=cfg_path.parent), config_file(cfg["robot_profile"], parent=cfg_path.parent), level=level, seed=seed, noise=False)
    try:
        obs, _ = env.reset(seed=seed)
    finally:
        env.close()
    if state_override is not None:
        state = np.asarray(state_override, dtype=np.float32)
        if state.shape != (4,) or not np.all(np.isfinite(state)) or np.any(state < -1) or np.any(state > 1):
            raise ValueError("--state must have four finite normalized values in [-1,1].")
        obs["state"] = state
    model, _ = load_checkpoint(rvf_checkpoint)
    model.eval()
    bearings = np.linspace(-1, 1, 21)
    linear = np.linspace(-1, 1, 21)
    scores = np.zeros((len(linear), len(bearings)), dtype=np.float32)
    with torch.no_grad():
        grid = torch.as_tensor(obs["dugm"], dtype=torch.float32).unsqueeze(0)
        for i, v in enumerate(linear):
            for j, bearing in enumerate(bearings):
                state = obs["state"].copy()
                state[1] = bearing
                state[2] = v
                scores[i, j] = model.score(grid, torch.as_tensor(state).unsqueeze(0)).item()
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(scores, origin="lower", extent=[-1,1,-1,1], vmin=0, vmax=1, cmap="magma")
    ax.set_xlabel("normalized goal bearing")
    ax.set_ylabel("normalized previous linear command")
    ax.set_title("RVF H40 · uncalibrated risk score\nfixed DUGM grid and other state values")
    fig.colorbar(image, ax=ax)
    path = output / "rvf_h40_state_sensitivity.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    write_json(output / "rvf_h40_state_sensitivity.json", {"level": level, "seed": seed, "fixed_state": obs["state"].tolist(), "bearing": bearings.tolist(), "previous_linear": linear.tolist(), "risk_score": scores.tolist()})
    record_artifact(output, "rvf_heatmap", path)
    return path
