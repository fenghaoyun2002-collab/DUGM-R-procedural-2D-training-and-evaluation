from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from dugm_r.simulation_2d.scene import Scene2D, _dynamic_actor_state


@dataclass(frozen=True)
class DynamicProximityRewardConfig:
    enabled: bool = False
    horizon_s: float = 4.0
    prediction_dt_s: float = 0.1
    safe_distance_m: float = 1.0
    max_penalty: float = 0.1

    @classmethod
    def from_mapping(cls, mapping: Any) -> "DynamicProximityRewardConfig":
        if mapping is None:
            return cls()
        if not isinstance(mapping, dict):
            raise ValueError("reward.dynamic_proximity must be a mapping when provided.")
        config = cls(
            enabled=bool(mapping.get("enabled", False)),
            horizon_s=float(mapping.get("horizon_s", 4.0)),
            prediction_dt_s=float(mapping.get("prediction_dt_s", 0.1)),
            safe_distance_m=float(mapping.get("safe_distance_m", 1.0)),
            max_penalty=float(mapping.get("max_penalty", 0.1)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.horizon_s < 0.0:
            raise ValueError("reward.dynamic_proximity.horizon_s must be non-negative.")
        if self.prediction_dt_s <= 0.0:
            raise ValueError("reward.dynamic_proximity.prediction_dt_s must be positive.")
        if self.safe_distance_m <= 0.0:
            raise ValueError("reward.dynamic_proximity.safe_distance_m must be positive.")
        if self.max_penalty < 0.0:
            raise ValueError("reward.dynamic_proximity.max_penalty must be non-negative.")


@dataclass(frozen=True)
class DynamicProximityRewardResult:
    reward: float
    min_distance_m: float
    risk: float
    dynamic_actor_count: int
    sample_count: int
    critical_actor_index: int
    critical_actor_kind: str
    diagnostics: dict[str, Any]


def compute_dynamic_proximity_reward(
    *,
    scene: Scene2D,
    robot_pose_xyyaw: np.ndarray,
    time_s: float,
    command: np.ndarray,
    robot_radius_m: float,
    collision_clearance_margin_m: float,
    config: DynamicProximityRewardConfig,
) -> DynamicProximityRewardResult:
    pose = np.asarray(robot_pose_xyyaw, dtype=np.float64).reshape(3)
    executable_command = np.asarray(command, dtype=np.float64).reshape(2)
    actor_count = int(len(scene.dynamic_actors))

    if not np.all(np.isfinite(pose)):
        raise ValueError("robot_pose_xyyaw must contain finite values.")
    if not np.all(np.isfinite(executable_command)):
        raise ValueError("command must contain finite values.")
    if float(robot_radius_m) < 0.0:
        raise ValueError("robot_radius_m must be non-negative.")

    if not config.enabled:
        return _result(
            config=config,
            command=executable_command,
            reward=0.0,
            min_distance_m=float("inf"),
            risk=0.0,
            actor_count=actor_count,
            sample_count=0,
            critical_actor_index=-1,
            critical_actor_kind="",
        )

    denominator = float(config.safe_distance_m) - float(collision_clearance_margin_m)
    if denominator <= 0.0:
        raise ValueError(
            "reward.dynamic_proximity.safe_distance_m must be greater than "
            "episode.collision_clearance_margin_m."
        )

    sample_times = prediction_sample_times(config.horizon_s, config.prediction_dt_s)
    if actor_count == 0:
        return _result(
            config=config,
            command=executable_command,
            reward=0.0,
            min_distance_m=float("inf"),
            risk=0.0,
            actor_count=0,
            sample_count=int(sample_times.size),
            critical_actor_index=-1,
            critical_actor_kind="",
        )

    robot_xy = rollout_unicycle_xy(pose, executable_command, sample_times)
    min_distance = float("inf")
    critical_actor_index = -1
    critical_actor_kind = ""

    for actor_index, actor in enumerate(scene.dynamic_actors):
        center_0, velocity_0 = _dynamic_actor_state(actor, float(time_s), scene.bounds_x_m, scene.bounds_y_m)
        center = np.asarray(center_0, dtype=np.float64)
        velocity = np.asarray(velocity_0, dtype=np.float64)
        actor_centers = center.reshape(1, 2) + sample_times.reshape(-1, 1) * velocity.reshape(1, 2)
        actor_clearances = _actor_clearances_over_time(
            actor=actor,
            robot_xy=robot_xy,
            actor_centers=actor_centers,
            velocity=velocity,
            robot_radius_m=float(robot_radius_m),
        )
        if actor_clearances.size == 0:
            continue
        actor_min = float(np.min(actor_clearances))
        if actor_min < min_distance:
            min_distance = actor_min
            critical_actor_index = int(actor_index)
            critical_actor_kind = str(actor.kind)

    risk = float(np.clip((float(config.safe_distance_m) - min_distance) / denominator, 0.0, 1.0))
    reward = -float(config.max_penalty) * risk
    return _result(
        config=config,
        command=executable_command,
        reward=reward,
        min_distance_m=min_distance,
        risk=risk,
        actor_count=actor_count,
        sample_count=int(sample_times.size),
        critical_actor_index=critical_actor_index,
        critical_actor_kind=critical_actor_kind,
    )


def prediction_sample_times(horizon_s: float, prediction_dt_s: float) -> np.ndarray:
    horizon = float(horizon_s)
    dt = float(prediction_dt_s)
    if horizon < 0.0:
        raise ValueError("horizon_s must be non-negative.")
    if dt <= 0.0:
        raise ValueError("prediction_dt_s must be positive.")
    if horizon <= 0.0:
        return np.asarray([0.0], dtype=np.float64)

    step_count = int(math.floor(horizon / dt + 1e-9))
    samples = [float(index) * dt for index in range(step_count + 1)]
    if not math.isclose(samples[-1], horizon, rel_tol=1e-9, abs_tol=1e-9):
        samples.append(horizon)
    else:
        samples[-1] = horizon
    return np.asarray(samples, dtype=np.float64)


def rollout_unicycle_xy(
    robot_pose_xyyaw: np.ndarray,
    command: np.ndarray,
    sample_times_s: np.ndarray,
    *,
    omega_epsilon: float = 1e-9,
) -> np.ndarray:
    pose = np.asarray(robot_pose_xyyaw, dtype=np.float64).reshape(3)
    executable_command = np.asarray(command, dtype=np.float64).reshape(2)
    tau = np.asarray(sample_times_s, dtype=np.float64).reshape(-1)
    x0, y0, theta0 = float(pose[0]), float(pose[1]), float(pose[2])
    v_mps, omega_radps = float(executable_command[0]), float(executable_command[1])

    if abs(omega_radps) < float(omega_epsilon):
        x = x0 + v_mps * math.cos(theta0) * tau
        y = y0 + v_mps * math.sin(theta0) * tau
    else:
        theta = theta0 + omega_radps * tau
        radius = v_mps / omega_radps
        x = x0 + radius * (np.sin(theta) - math.sin(theta0))
        y = y0 - radius * (np.cos(theta) - math.cos(theta0))
    return np.column_stack([x, y]).astype(np.float64, copy=False)


def _actor_clearances_over_time(
    *,
    actor: Any,
    robot_xy: np.ndarray,
    actor_centers: np.ndarray,
    velocity: np.ndarray,
    robot_radius_m: float,
) -> np.ndarray:
    if actor.kind == "cart":
        size = actor.size_m if actor.size_m is not None else (0.9, 0.55)
        return _box_clearances(robot_xy, actor_centers, size_m=size, robot_radius_m=robot_radius_m)
    if actor.kind == "person":
        motion = np.asarray(velocity, dtype=np.float64)
        if np.linalg.norm(motion) <= 1e-9:
            motion = np.asarray(actor.velocity, dtype=np.float64)
        heading = math.atan2(float(motion[1]), float(motion[0])) if np.linalg.norm(motion) > 1e-9 else 0.0
        side = np.asarray([-math.sin(heading), math.cos(heading)], dtype=np.float64)
        separation = float(actor.foot_separation_m or 0.34)
        radius = float(actor.foot_radius_m or 0.14)
        left_centers = actor_centers - 0.5 * separation * side.reshape(1, 2)
        right_centers = actor_centers + 0.5 * separation * side.reshape(1, 2)
        return np.minimum(
            _circle_clearances(robot_xy, left_centers, radius_m=radius, robot_radius_m=robot_radius_m),
            _circle_clearances(robot_xy, right_centers, radius_m=radius, robot_radius_m=robot_radius_m),
        )
    radius = float(actor.radius_m or 0.3)
    return _circle_clearances(robot_xy, actor_centers, radius_m=radius, robot_radius_m=robot_radius_m)


def _circle_clearances(
    robot_xy: np.ndarray,
    centers: np.ndarray,
    *,
    radius_m: float,
    robot_radius_m: float,
) -> np.ndarray:
    return np.linalg.norm(np.asarray(robot_xy, dtype=np.float64) - np.asarray(centers, dtype=np.float64), axis=1) - (
        float(radius_m) + float(robot_radius_m)
    )


def _box_clearances(
    robot_xy: np.ndarray,
    centers: np.ndarray,
    *,
    size_m: tuple[float, float],
    robot_radius_m: float,
) -> np.ndarray:
    point = np.asarray(robot_xy, dtype=np.float64)
    center = np.asarray(centers, dtype=np.float64)
    half = 0.5 * np.asarray(size_m, dtype=np.float64).reshape(1, 2)
    delta = np.abs(point - center) - half
    outside = np.linalg.norm(np.maximum(delta, 0.0), axis=1)
    inside = np.minimum(np.maximum(delta[:, 0], delta[:, 1]), 0.0)
    return outside + inside - float(robot_radius_m)


def _result(
    *,
    config: DynamicProximityRewardConfig,
    command: np.ndarray,
    reward: float,
    min_distance_m: float,
    risk: float,
    actor_count: int,
    sample_count: int,
    critical_actor_index: int,
    critical_actor_kind: str,
) -> DynamicProximityRewardResult:
    bounded_risk = float(np.clip(risk, 0.0, 1.0))
    bounded_reward = float(np.clip(reward, -float(config.max_penalty), 0.0))
    diagnostics: dict[str, Any] = {
        "dynamic_proximity_enabled": bool(config.enabled),
        "dynamic_proximity_horizon_s": float(config.horizon_s),
        "dynamic_proximity_prediction_dt_s": float(config.prediction_dt_s),
        "dynamic_proximity_safe_distance_m": float(config.safe_distance_m),
        "dynamic_proximity_max_penalty": float(config.max_penalty),
        "dynamic_proximity_dynamic_actor_count": int(actor_count),
        "dynamic_proximity_sample_count": int(sample_count),
        "predicted_dynamic_min_distance_m": float(min_distance_m),
        "predicted_dynamic_risk": bounded_risk,
        "reward_dynamic_proximity": bounded_reward,
        "selected_linear_velocity_mps": float(command[0]),
        "selected_angular_velocity_radps": float(command[1]),
        "dynamic_proximity_critical_actor_index": int(critical_actor_index),
        "dynamic_proximity_critical_actor_kind": str(critical_actor_kind),
    }
    return DynamicProximityRewardResult(
        reward=bounded_reward,
        min_distance_m=float(min_distance_m),
        risk=bounded_risk,
        dynamic_actor_count=int(actor_count),
        sample_count=int(sample_count),
        critical_actor_index=int(critical_actor_index),
        critical_actor_kind=str(critical_actor_kind),
        diagnostics=diagnostics,
    )
