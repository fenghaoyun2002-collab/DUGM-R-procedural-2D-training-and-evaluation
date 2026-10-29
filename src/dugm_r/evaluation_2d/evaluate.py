"""Fixed-seed evaluation on the public procedural 2D test protocol."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from dugm_r.io import config_file, load_config, sha256, write_json
from dugm_r.nominal.features import DugmFeaturesExtractor  # checkpoint class registration
from dugm_r.nominal.train import make_env
from dugm_r.recovery.rvf import load_checkpoint
from dugm_r.recovery.switch import DugmRPolicy
from dugm_r.visualization.plots import save_trajectory


def evaluate_2d(method: str, nominal_checkpoint: Path, output: Path, *, recovery_checkpoint: Path | None = None,
                rvf_checkpoint: Path | None = None, thresholds_path: Path | None = None,
                config_name: str | Path = "eval_2d.yaml", render: bool = False, smoke: bool = False) -> Path:
    if method not in {"dugm_nominal", "dugm_r"}:
        raise ValueError("method must be dugm_nominal or dugm_r.")
    cfg, cfg_path = load_config(config_name)
    curriculum = config_file(cfg["curriculum"], parent=cfg_path.parent)
    profile = config_file(cfg["robot_profile"], parent=cfg_path.parent)
    if not nominal_checkpoint.is_file():
        raise FileNotFoundError(nominal_checkpoint)
    nominal = PPO.load(nominal_checkpoint, device="cpu")
    assets = {"nominal": sha256(nominal_checkpoint)}
    policy = None
    if method == "dugm_r":
        if not all(p is not None and p.is_file() for p in (recovery_checkpoint, rvf_checkpoint, thresholds_path)):
            raise FileNotFoundError("DUGM-R evaluation requires recovery, RVF and threshold files.")
        recovery = PPO.load(recovery_checkpoint, device="cpu")
        rvf, _ = load_checkpoint(rvf_checkpoint)
        thresholds = json.loads(thresholds_path.read_text())
        switch_cfg, _ = load_config("recovery_train_2d.yaml")
        policy = DugmRPolicy(nominal, recovery, rvf, thresholds,
                             min_recovery_dwell_steps=int(switch_cfg["switching"]["min_recovery_dwell_steps"]),
                             return_dwell_steps=int(switch_cfg["switching"]["return_dwell_steps"]))
        assets.update({"recovery": sha256(recovery_checkpoint), "rvf": sha256(rvf_checkpoint), "thresholds": sha256(thresholds_path)})
    scenes = cfg["scenes"][:1] if smoke else cfg["scenes"]
    records = []
    root = output / "evaluation_2d" / method
    for spec in scenes:
        scene_id, level, seed = str(spec["scene_id"]), int(spec["level"]), int(spec["seed"])
        env = make_env(curriculum, profile, level=level, seed=seed, noise=bool(cfg.get("observation_noise", False)))
        try:
            obs, _ = env.reset(seed=seed)
            if policy is not None:
                policy.reset()
            trajectory = [env.pose[:2].copy()]
            modes: list[str] = []
            scores: list[float] = []
            events = []
            outcome = "timeout"
            for step in range(min(env.max_steps, 8) if smoke else env.max_steps):
                if policy is None:
                    action, _ = nominal.predict(obs, deterministic=True)
                    mode, score, triggered, released = "nominal", float("nan"), False, False
                else:
                    decision = policy.predict(obs, deterministic=True)
                    action, mode, score, triggered, released = decision.action, decision.mode, decision.risk_score, decision.triggered, decision.released
                obs, reward, terminated, truncated, info = env.step(action)
                trajectory.append(env.pose[:2].copy())
                modes.append(mode)
                scores.append(score)
                if triggered or released:
                    events.append({"step": step, "triggered": triggered, "released": released, "risk_score": score})
                if terminated or truncated:
                    outcome = "collision" if info.get("collision") else "success" if info.get("success") else "timeout"
                    break
            record = {"scene_id": scene_id, "level": level, "seed": seed, "method": method, "outcome": outcome,
                      "steps": len(modes), "trajectory_xy": np.asarray(trajectory).tolist(), "policy_modes": modes,
                      "risk_score": [None if np.isnan(x) else x for x in scores], "events": events}
            scene_dir = root / scene_id
            write_json(scene_dir / "episode.json", record)
            if render:
                save_trajectory(env.scene, np.asarray(trajectory), modes, scores, scene_dir / "trajectory.png", title=f"{method} · {scene_id}")
            records.append({"scene_id": scene_id, "outcome": outcome, "steps": len(modes)})
        finally:
            env.close()
    counts = {name: sum(row["outcome"] == name for row in records) for name in ("success", "collision", "timeout")}
    summary = root / "summary.json"
    write_json(summary, {"protocol": cfg["protocol"], "method": method, "asset_sha256": assets,
                         "curriculum_sha256": sha256(curriculum), "scenes": records, "counts": counts,
                         "success_rate": counts["success"] / max(1, len(records))})
    write_json(root / "manifest.json", {
        "protocol": cfg["protocol"], "method": method, "config_sha256": sha256(cfg_path),
        "curriculum_sha256": sha256(curriculum), "asset_sha256": assets,
        "scenes": [{"scene_id": item["scene_id"], "level": item["level"], "seed": item["seed"]} for item in scenes],
        "render": render, "render_dpi": 140 if render else None,
        "summary_sha256": sha256(summary),
    })
    return summary
