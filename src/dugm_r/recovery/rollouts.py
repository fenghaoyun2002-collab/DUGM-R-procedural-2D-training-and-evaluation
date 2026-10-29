"""Collect deterministic nominal 2D episodes and future collision labels."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from dugm_r.io import config_file, load_config, require_empty_output, sha256, write_json
from dugm_r.nominal.features import DugmFeaturesExtractor  # needed by PPO checkpoint loader
from dugm_r.nominal.train import make_env

SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}


def future_collision_labels(collisions: np.ndarray, horizon: int) -> np.ndarray:
    labels = np.zeros(len(collisions), dtype=np.uint8)
    for index in np.flatnonzero(collisions):
        labels[max(0, int(index) - horizon):int(index) + 1] = 1
    return labels


def _split_code(index: int, *, smoke: bool) -> int:
    if smoke and index < 3:
        return index
    bucket = (index * 37 + 11) % 20
    return 0 if bucket < 14 else 1 if bucket < 17 else 2


def _stack_episodes(episodes: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([episode[key] for episode in episodes], axis=0) for key in episodes[0]}


def collect_rollouts(config_name: str | Path, nominal_checkpoint: Path, output: Path, *, smoke: bool = False) -> Path:
    cfg, cfg_path = load_config(config_name)
    if cfg.get("method") != "dugm_nominal":
        raise ValueError("Expected dugm_nominal rollout configuration.")
    if not nominal_checkpoint.is_file():
        raise FileNotFoundError(nominal_checkpoint)
    curriculum = config_file(cfg["curriculum"], parent=cfg_path.parent)
    profile = config_file(cfg["robot_profile"], parent=cfg_path.parent)
    require_empty_output(output)
    model = PPO.load(nominal_checkpoint, device="cpu")
    rng = np.random.default_rng(int(cfg["seed"]))
    levels = np.asarray([int(key) for key in cfg["level_mix"]], dtype=np.int32)
    weights = np.asarray(list(cfg["level_mix"].values()), dtype=np.float64)
    weights /= weights.sum()
    total = 3 if smoke else int(cfg["episodes"])
    per_shard = 3 if smoke else int(cfg["episodes_per_shard"])
    max_steps = 8 if smoke else int(cfg["max_steps_per_episode"])
    horizon = int(cfg["label_horizon"])
    pending: list[dict[str, np.ndarray]] = []
    manifest: dict[str, Any] = {
        "protocol": "procedural_2d_rollouts", "method": "dugm_nominal", "episodes": total,
        "horizon": horizon, "nominal_sha256": sha256(nominal_checkpoint),
        "config_sha256": sha256(cfg_path), "curriculum_sha256": sha256(curriculum),
        "split_policy": "episode", "split_fraction": cfg["split"], "level_mix": cfg["level_mix"],
        "scene_seed_namespace": "4000000+episode_id",
        "shards": [],
    }
    for episode_index in range(total):
        level = int(rng.choice(levels, p=weights))
        seed = 4_000_000 + episode_index
        env = make_env(curriculum, profile, level=level, seed=seed, noise=bool(cfg.get("observation_noise", False)))
        grids: list[np.ndarray] = []
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        collisions: list[bool] = []
        try:
            obs, _ = env.reset(seed=seed)
            for _ in range(max_steps):
                grids.append(np.asarray(obs["dugm"], dtype=np.float32).copy())
                states.append(np.asarray(obs["state"], dtype=np.float32).copy())
                action, _ = model.predict(obs, deterministic=bool(cfg.get("deterministic_policy", True)))
                action = np.asarray(action, dtype=np.float32).reshape(2)
                actions.append(action.copy())
                obs, _, terminal, truncated, info = env.step(action)
                collisions.append(bool(info.get("collision", False)))
                if terminal or truncated:
                    break
        finally:
            env.close()
        n = len(grids)
        episode = {
            "dugm": np.stack(grids).astype(np.float32),
            "state": np.stack(states).astype(np.float32),
            "action": np.stack(actions).astype(np.float32),
            "collision": np.asarray(collisions, dtype=np.uint8),
            "label_h40": future_collision_labels(np.asarray(collisions), horizon),
            "episode_id": np.full(n, episode_index, dtype=np.int32),
            "step_index": np.arange(n, dtype=np.int32),
            "level": np.full(n, level, dtype=np.int16),
            "split": np.full(n, _split_code(episode_index, smoke=smoke), dtype=np.uint8),
        }
        pending.append(episode)
        if len(pending) >= per_shard or episode_index == total - 1:
            index = len(manifest["shards"])
            path = output / f"rollouts_{index:05d}.npz"
            arrays = _stack_episodes(pending)
            np.savez_compressed(path, **arrays)
            manifest["shards"].append({"file": path.name, "sha256": sha256(path), "episodes": len(pending), "steps": len(arrays["dugm"])})
            pending.clear()
            write_json(output / "rollouts_manifest.json", manifest)
    return output / "rollouts_manifest.json"
