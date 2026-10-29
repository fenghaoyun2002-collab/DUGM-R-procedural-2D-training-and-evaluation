from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ROBOT_PROFILE_PATH = Path(__file__).resolve().parents[1] / "configs" / "robot_profile_2d.yaml"


@dataclass(frozen=True)
class RobotGeometryProfile:
    footprint_radius_m: float
    collision_radius_m: float
    mass_kg: float
    body_length_m: float
    body_width_m: float
    body_height_m: float
    wheel_radius_m: float
    wheel_separation_m: float


@dataclass(frozen=True)
class RobotActionProfile:
    linear_velocity_bounds_mps: tuple[float, float]
    angular_velocity_bounds_radps: tuple[float, float]
    linear_accel_limit_mps2: float
    angular_accel_limit_radps2: float
    apply_wheel_speed_limits: bool
    wheel_max_radps: float


@dataclass(frozen=True)
class RobotDynamicsProfile:
    control_dt_s: float
    integration_substeps: int


@dataclass(frozen=True)
class RobotLidarProfile:
    num_rays: int
    fov_rad: float
    min_range_m: float
    max_range_m: float
    update_rate_hz: float


@dataclass(frozen=True)
class RobotLidarNoiseProfile:
    enabled: bool
    range_bias_std_m: float
    range_noise_std_m: float
    range_noise_ratio: float
    max_abs_noise_m: float
    max_abs_noise_ratio: float
    dropout_prob: float
    false_near_prob: float
    max_range_spike_prob: float


@dataclass(frozen=True)
class RobotProfile:
    name: str
    geometry: RobotGeometryProfile
    action: RobotActionProfile
    dynamics: RobotDynamicsProfile
    lidar: RobotLidarProfile
    lidar_noise: RobotLidarNoiseProfile


def load_robot_profile(path: str | Path | None = None) -> RobotProfile:
    profile_path = Path(path) if path is not None else DEFAULT_ROBOT_PROFILE_PATH
    with profile_path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}

    geometry = dict(data.get("geometry", {}))
    body = dict(geometry.get("body_size_m", {}))
    action = dict(data.get("action", {}))
    dynamics = dict(data.get("dynamics", {}))
    lidar = dict(data.get("lidar", {}))
    noise = dict(dict(data.get("noise", {})).get("lidar", {}))

    return RobotProfile(
        name=str(data.get("name", profile_path.stem)),
        geometry=RobotGeometryProfile(
            footprint_radius_m=float(geometry.get("footprint_radius_m", geometry.get("collision_radius_m", 0.089))),
            collision_radius_m=float(geometry.get("collision_radius_m", geometry.get("footprint_radius_m", 0.089))),
            mass_kg=float(geometry.get("mass_kg", 1.0)),
            body_length_m=float(body.get("length", 0.138)),
            body_width_m=float(body.get("width", 0.178)),
            body_height_m=float(body.get("height", 0.192)),
            wheel_radius_m=float(geometry.get("wheel_radius_m", 0.033)),
            wheel_separation_m=float(geometry.get("wheel_separation_m", 0.160)),
        ),
        action=RobotActionProfile(
            linear_velocity_bounds_mps=_float_pair(action.get("linear_velocity_bounds_mps", [0.0, 0.22])),
            angular_velocity_bounds_radps=_float_pair(action.get("angular_velocity_bounds_radps", [-2.84, 2.84])),
            linear_accel_limit_mps2=float(action.get("linear_accel_limit_mps2", 0.60)),
            angular_accel_limit_radps2=float(action.get("angular_accel_limit_radps2", 7.50)),
            apply_wheel_speed_limits=bool(action.get("apply_wheel_speed_limits", False)),
            wheel_max_radps=float(action.get("wheel_max_radps", 6.388)),
        ),
        dynamics=RobotDynamicsProfile(
            control_dt_s=float(dynamics.get("control_dt_s", 0.1)),
            integration_substeps=max(1, int(dynamics.get("integration_substeps", 2))),
        ),
        lidar=RobotLidarProfile(
            num_rays=int(lidar.get("num_rays", 360)),
            fov_rad=float(lidar.get("fov_rad", 6.283185307179586)),
            min_range_m=float(lidar.get("min_range_m", 0.16)),
            max_range_m=float(lidar.get("max_range_m", 8.0)),
            update_rate_hz=float(lidar.get("update_rate_hz", 10.7)),
        ),
        lidar_noise=RobotLidarNoiseProfile(
            enabled=bool(noise.get("enabled", True)),
            range_bias_std_m=float(noise.get("range_bias_std_m", 0.0)),
            range_noise_std_m=float(noise.get("range_noise_std_m", 0.003)),
            range_noise_ratio=float(noise.get("range_noise_ratio", 0.010)),
            max_abs_noise_m=float(noise.get("max_abs_noise_m", 0.010)),
            max_abs_noise_ratio=float(noise.get("max_abs_noise_ratio", 0.030)),
            dropout_prob=float(noise.get("dropout_prob", 0.005)),
            false_near_prob=float(noise.get("false_near_prob", 0.0)),
            max_range_spike_prob=float(noise.get("max_range_spike_prob", 0.0)),
        ),
    )


def _float_pair(value: Any) -> tuple[float, float]:
    return (float(value[0]), float(value[1]))
