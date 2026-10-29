"""Stagewise nominal DUGM PPO training on procedural 2D scenes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from dugm_r.io import config_file, load_config, record_artifact, require_empty_output, sha256, write_json
from dugm_r.nominal.policy import ppo_from_config
from dugm_r.simulation_2d.env import Dugm2DEnv


class LevelMixEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, mix: dict[int | str, float], *, seed: int) -> None:
        super().__init__(env)
        self.levels = np.array([int(key) for key in mix], dtype=np.int32)
        weights = np.array(list(mix.values()), dtype=np.float64)
        if len(weights) == 0 or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("level_mix must have positive total weight.")
        self.weights = weights / weights.sum()
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        level = int(self.rng.choice(self.levels, p=self.weights))
        scene_seed = int(self.rng.integers(0, 1_000_000))
        return self.env.reset(seed=scene_seed, options={**(options or {}), "level": level})


def make_env(curriculum: Path, profile: Path, *, level: int, seed: int, noise: bool = True) -> Dugm2DEnv:
    return Dugm2DEnv(
        curriculum_path=curriculum, robot_profile_path=profile,
        level=level, seed=seed, observation_noise=noise,
    )


def evaluate_model(model: Any, curriculum: Path, profile: Path, *, scenes: list[tuple[int, int]], max_steps: int | None = None) -> dict[str, Any]:
    counts = {"success": 0, "collision": 0, "timeout": 0}
    for level, seed in scenes:
        env = make_env(curriculum, profile, level=level, seed=seed, noise=False)
        try:
            obs, _ = env.reset(seed=seed)
            for _ in range(min(env.max_steps, max_steps or env.max_steps)):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    key = "collision" if info.get("collision") else "success" if info.get("success") else "timeout"
                    counts[key] += 1
                    break
            else:
                counts["timeout"] += 1
        finally:
            env.close()
    total = max(1, sum(counts.values()))
    return {**counts, "episodes": len(scenes), "success_rate": counts["success"] / total, "collision_rate": counts["collision"] / total}


def train_nominal(config_name: str | Path, output: Path, *, smoke: bool = False) -> Path:
    cfg, cfg_path = load_config(config_name)
    if cfg.get("method") != "dugm_nominal":
        raise ValueError("Expected dugm_nominal training configuration.")
    curriculum = config_file(cfg["curriculum"], parent=cfg_path.parent)
    profile = config_file(cfg["robot_profile"], parent=cfg_path.parent)
    require_empty_output(output)
    seed = int(cfg["seed"])
    stage_records: list[dict[str, Any]] = []
    model = None
    best_key = (-1.0, -1.0)
    stages = cfg["stages"][:1] if smoke else cfg["stages"]
    for stage_index, stage in enumerate(stages):
        n_envs = 1 if smoke else int(cfg.get("num_envs", 1))
        level = int(stage["level"])
        mix = {int(k): float(v) for k, v in stage["level_mix"].items()}
        def factory(worker: int):
            def build():
                env = make_env(curriculum, profile, level=level, seed=seed + worker + stage_index * 1000, noise=bool(cfg.get("observation_noise", True)))
                return LevelMixEnv(env, mix, seed=seed + worker + stage_index * 1000)
            return build
        envs = [factory(worker) for worker in range(n_envs)]
        vec = DummyVecEnv(envs) if smoke or n_envs == 1 else SubprocVecEnv(envs, start_method="fork")
        try:
            if model is None:
                ppo = dict(cfg["ppo"])
                ppo.update(stage.get("ppo_overrides", {}))
                if smoke:
                    ppo.update({"n_steps": 16, "batch_size": 16, "n_epochs": 1})
                model = ppo_from_config(vec, ppo, seed=seed, device="cpu" if smoke else str(cfg.get("device", "cpu")))
            else:
                model.set_env(vec)
                for key, value in stage.get("ppo_overrides", {}).items():
                    if key in {"ent_coef", "target_kl"}:
                        setattr(model, key, value)
            steps = 16 if smoke else int(stage["steps"])
            model.learn(total_timesteps=steps, reset_num_timesteps=False, progress_bar=False)
            validation = cfg["validation"]
            scenes = [(int(l), int(s)) for l in validation["levels"] for s in validation["seeds"]]
            if smoke:
                scenes = scenes[:1]
            metrics = evaluate_model(model, curriculum, profile, scenes=scenes, max_steps=8 if smoke else None)
            stage_path = output / f"{stage['id']}.zip"
            model.save(stage_path)
            key = (float(metrics["success_rate"]), -float(metrics["collision_rate"]))
            if key > best_key or not (output / "nominal_best.zip").exists():
                model.save(output / "nominal_best.zip")
                best_key = key
            stage_records.append({"id": stage["id"], "level_mix": mix, "steps": steps, "validation": metrics, "checkpoint": stage_path.name})
            write_json(output / "training_history.json", {"method": "dugm_nominal", "stages": stage_records})
        finally:
            vec.close()
    best = output / "nominal_best.zip"
    record_artifact(output, "nominal", best, metadata={"method": "dugm_nominal", "config_sha256": sha256(cfg_path), "curriculum_sha256": sha256(curriculum)})
    return best
