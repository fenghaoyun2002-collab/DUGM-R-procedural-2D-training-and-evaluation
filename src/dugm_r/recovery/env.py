"""Online recovery PPO environment: nominal autorun, RVF trigger, recovery action."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from dugm_r.io import config_file, load_config
from dugm_r.nominal.features import DugmFeaturesExtractor  # checkpoint class registration
from dugm_r.nominal.train import make_env
from dugm_r.recovery.rvf import load_checkpoint, risk_score
from dugm_r.simulation_2d.lidar import min_clearance


class RecoveryTrainingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, *, config_name: str | Path, nominal_checkpoint: Path, rvf_checkpoint: Path,
                 thresholds_path: Path, seed: int, smoke: bool = False) -> None:
        super().__init__()
        cfg, cfg_path = load_config(config_name)
        self.cfg = cfg
        self.curriculum = config_file(cfg["curriculum"], parent=cfg_path.parent)
        self.profile = config_file(cfg["robot_profile"], parent=cfg_path.parent)
        self.nominal = PPO.load(nominal_checkpoint, device="cpu")
        self.rvf, _ = load_checkpoint(rvf_checkpoint, device="cpu")
        import json
        thresholds = json.loads(thresholds_path.read_text())
        self.epsilon_on = float(thresholds["epsilon_on"])
        self.epsilon_return = float(thresholds["epsilon_return"])
        if self.epsilon_return > self.epsilon_on:
            raise ValueError("Recovery threshold must not exceed trigger threshold.")
        switch = cfg["switching"]
        self.min_recovery_dwell_steps = int(switch["min_recovery_dwell_steps"])
        self.return_dwell_steps = int(switch["return_dwell_steps"])
        self.max_task_steps = 4 if smoke else int(switch["max_task_steps_per_event"])
        self.max_recovery_steps = 8 if smoke else int(switch["max_recovery_steps_per_episode"])
        self.episode_max_steps = 12 if smoke else int(switch["episode_max_steps_override"])
        self.reward_cfg = cfg["reward"]
        self.rng = np.random.default_rng(seed)
        mix = cfg["level_mix"]
        self.levels = np.array([int(key) for key in mix], dtype=np.int32)
        self.weights = np.array(list(mix.values()), dtype=np.float64)
        self.weights /= self.weights.sum()
        self.env = None
        self.obs: dict[str, np.ndarray] | None = None
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.recovery_steps = 0
        self.episode_recovery_steps = 0
        self.return_dwell = 0
        self.handoff_count = 0
        self.episode_nominal_steps = 0
        self.observation_space = self.nominal.observation_space
        self.action_space = self.nominal.action_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for attempt in range(3):
            if self.env is not None:
                self.env.close()
            level = int(self.rng.choice(self.levels, p=self.weights))
            scene_seed = int(self.rng.integers(1_000_000, 2_000_000))
            self.env = make_env(self.curriculum, self.profile, level=level, seed=scene_seed, noise=bool(self.cfg.get("observation_noise", False)))
            self.obs, _ = self.env.reset(seed=scene_seed)
            self.env.max_steps = self.episode_max_steps
            self.previous_action[:] = 0
            self.recovery_steps = 0
            self.episode_recovery_steps = 0
            self.return_dwell = 0
            self.handoff_count = 0
            self.episode_nominal_steps = 0
            event = self._autorun_nominal()
            if not event["terminal"]:
                return self._copy_obs(), {"risk_score": event["risk_score"], "reset_trigger": event["trigger"], "policy_mode": "recovery"}
        return self._copy_obs(), {"risk_score": self._risk(), "reset_trigger": "forced_after_terminal", "policy_mode": "recovery"}

    def step(self, action: Any):
        if self.env is None or self.obs is None:
            raise RuntimeError("Reset the recovery environment before stepping.")
        action = np.asarray(action, dtype=np.float32).reshape(2)
        risk_before = self._risk()
        self.obs, _, terminated, truncated, info = self.env.step(action)
        self.recovery_steps += 1
        self.episode_recovery_steps += 1
        risk_after = self._risk()
        terminal = bool(terminated or truncated)
        released = False
        task_steps = 0
        if not terminal and self.recovery_steps >= self.min_recovery_dwell_steps:
            self.return_dwell = self.return_dwell + 1 if risk_after <= self.epsilon_return else 0
            if self.return_dwell >= self.return_dwell_steps:
                released = True
                self.handoff_count += 1
                event = self._autorun_nominal()
                task_steps = int(event["steps"])
                terminal = bool(event["terminal"])
                if terminal:
                    info = event["info"]
                risk_after = float(event["risk_score"])
                self.recovery_steps = 0
                self.return_dwell = 0
        else:
            self.return_dwell = 0
        forced_timeout = self.episode_recovery_steps >= self.max_recovery_steps and not terminal
        terminal = terminal or forced_timeout
        collision = bool(info.get("collision", False))
        success = bool(info.get("success", False))
        timeout = bool(info.get("timeout", False)) or forced_timeout
        clearance = self._clearance()
        command = info.get("command", {})
        linear = float(command.get("linear_mps", 0.0)) if isinstance(command, dict) else 0.0
        reward_terms = self._reward_terms(risk_before, risk_after, action, linear, clearance, success and terminal, collision and terminal, timeout and terminal, task_steps)
        self.previous_action = action.copy()
        out_info = {
            **info, **reward_terms, "policy_mode": "recovery", "risk_before": risk_before,
            "risk_score": risk_after, "return_to_safe": released, "handoff_count": self.handoff_count,
            "episode_nominal_steps": self.episode_nominal_steps, "recovery_steps": self.recovery_steps,
            "episode_recovery_steps": self.episode_recovery_steps,
            "collision": collision, "success": success, "timeout": timeout,
        }
        return self._copy_obs(), reward_terms["reward_total"], bool(terminal and not timeout), bool(terminal and timeout), out_info

    def _autorun_nominal(self) -> dict[str, Any]:
        final_info: dict[str, Any] = {}
        for steps in range(self.max_task_steps):
            score = self._risk()
            if score >= self.epsilon_on:
                return {"terminal": False, "trigger": "risk", "risk_score": score, "steps": steps, "info": final_info}
            action, _ = self.nominal.predict(self.obs, deterministic=True)
            self.obs, _, terminated, truncated, final_info = self.env.step(action)
            self.episode_nominal_steps += 1
            if terminated or truncated:
                return {"terminal": True, "trigger": "terminal", "risk_score": self._risk(), "steps": steps + 1, "info": final_info}
        return {"terminal": False, "trigger": "task_autorun_limit", "risk_score": self._risk(), "steps": self.max_task_steps, "info": final_info}

    def _risk(self) -> float:
        return risk_score(self.rvf, self.obs)

    def _clearance(self) -> float:
        scene = self.env._require_scene()
        return min_clearance(scene, point_xy=self.env.pose[:2], radius_m=self.env.robot_radius_m, time_s=self.env.step_count * self.env.dt_s)

    def _reward_terms(self, before: float, after: float, action: np.ndarray, linear: float,
                      clearance: float, success: bool, collision: bool, timeout: bool, task_steps: int) -> dict[str, float]:
        c = self.reward_cfg
        close = max(0.0, float(c["close_clearance_threshold_m"]) - clearance) / max(1e-6, float(c["close_clearance_threshold_m"]))
        terms = {
            "reward_risk_delta": float(c["risk_delta_weight"]) * (before - after),
            "reward_final_success": float(c["final_success_bonus"]) if success else 0.0,
            "reward_final_collision": -float(c["final_collision_penalty"]) if collision else 0.0,
            "reward_final_timeout": -float(c["final_timeout_penalty"]) if timeout else 0.0,
            "reward_handoff_failure": -float(c["handoff_failure_penalty"]) if self.handoff_count and (collision or timeout) else 0.0,
            "reward_recovery_step": -float(c["recovery_step_penalty"]),
            "reward_task_autorun_step": -float(c["task_autorun_step_penalty"]) * task_steps,
            "reward_clearance_penalty": -float(c["close_clearance_penalty_max"]) * close * close,
            "reward_action_l2_penalty": -float(c["action_l2_penalty"]) * float(np.sum(action * action)),
            "reward_action_delta_l2_penalty": -float(c["action_delta_l2_penalty"]) * float(np.sum((action - self.previous_action) ** 2)),
            "reward_positive_linear_penalty": -float(c["positive_linear_penalty"]) * linear if clearance < float(c["positive_linear_clearance_threshold_m"]) and linear > 0 else 0.0,
        }
        terms["reward_total"] = float(sum(terms.values()))
        return terms

    def _copy_obs(self) -> dict[str, np.ndarray]:
        return {key: value.astype(np.float32, copy=True) for key, value in self.obs.items()}

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
