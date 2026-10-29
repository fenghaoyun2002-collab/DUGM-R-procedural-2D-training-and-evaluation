from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any


_NATIVE_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _configure_import_time_native_threads() -> None:
    for key in _NATIVE_THREAD_ENV:
        os.environ.setdefault(key, "1")


_configure_import_time_native_threads()

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from dugm_r.simulation_2d.lidar import (
    build_static_obstacle_scan_cache,
    collides,
    min_clearance,
    scan_scene,
    scan_static_obstacle_cache,
)
from dugm_r.simulation_2d.reward import (
    DynamicProximityRewardConfig,
    compute_dynamic_proximity_reward,
)
from dugm_r.simulation_2d.path import (
    PathGuidance,
    PathGuidanceConfig,
    build_path_guidance,
    query_cost_to_go,
    select_local_waypoint,
)
from dugm_r.simulation_2d.profile import load_robot_profile
from dugm_r.simulation_2d.scene import CurriculumSceneGenerator, Scene2D, load_curriculum_config
from dugm_r.dugm.costmap import (
    estimated_dynamic_track_states,
    occupied_costmap_diagnostics,
    resolve_static_endpoint_inflation_radius,
    resolve_velocity_cost_core_inflation_radius,
    stack_occupied_costmap,
    visible_endpoint_cost_from_ranges,
    visible_dynamic_actor_states,
    visible_static_endpoint_cost_from_ranges,
)
from dugm_r.dugm.scan_grid import GridEncoder
from dugm_r.dugm.prediction import (
    EstimatedVelocityCostmapRasterizer,
    GtVelocityCostmapRasterizer,
    LinearVelocityRolloutCostmapRasterizer,
    UncertaintyAwareEstimatedVelocityCostmapRasterizer,
)
from dugm_r.dugm.estimator import DEFAULT_CONFIG_PATH as DEFAULT_DUGM_ESTIMATOR_CONFIG
from dugm_r.dugm.estimator import DugmEstimator


class Dugm2DEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        level: int = 0,
        curriculum_path: str | Path | None = None,
        seed: int | None = None,
        observation_noise: bool = True,
        robot_profile_path: str | Path | None = None,
        fixed_scene_set_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.config = load_curriculum_config(curriculum_path)
        if fixed_scene_set_path is not None:
            levels = dict(self.config.get("levels", {}))
            level_config = dict(levels.get(int(level), {}))
            level_config["fixed_scene_set_path"] = str(Path(fixed_scene_set_path).resolve())
            level_config.pop("fixed_scenes", None)
            levels[int(level)] = level_config
            self.config["levels"] = levels
        self.robot_profile = load_robot_profile(robot_profile_path)
        self.generator = CurriculumSceneGenerator(self.config, robot_radius_m=self.robot_profile.geometry.collision_radius_m)
        self.level = int(level)
        self.observation_noise = bool(observation_noise)
        self.rng = np.random.default_rng(seed)
        self.scene: Scene2D | None = None
        self.pose = np.zeros(3, dtype=np.float32)
        self.command = np.zeros(2, dtype=np.float32)
        self.initial_goal_distance_m = 1.0
        self.previous_goal_distance_m = 0.0
        self.step_count = 0
        self.path_guidance: PathGuidance | None = None
        self.previous_geodesic_cost_m: float | None = None
        self.current_waypoint_xy: np.ndarray | None = None
        self.previous_waypoint_heading_potential: float | None = None
        self.smoothed_waypoint_bearing_rad: float | None = None
        self.path_guidance_status = "disabled"

        episode = dict(self.config.get("episode", {}))
        self.robot_radius_m = float(self.robot_profile.geometry.collision_radius_m)
        self.num_rays = int(self.robot_profile.lidar.num_rays)
        self.fov_rad = float(self.robot_profile.lidar.fov_rad)
        self.min_range_m = float(self.robot_profile.lidar.min_range_m)
        self.max_range_m = float(self.robot_profile.lidar.max_range_m)
        self.lidar_noise = self.robot_profile.lidar_noise
        self.dt_s = float(episode.get("dt_s", self.robot_profile.dynamics.control_dt_s))
        self.default_max_steps = int(episode.get("max_steps", 500))
        if self.default_max_steps <= 0:
            raise ValueError("episode.max_steps must be positive.")
        self.max_steps = self.default_max_steps
        self.goal_radius_m = float(episode.get("goal_radius_m", episode.get("goal_tolerance_m", 0.35)))
        self.goal_success_distance_m = self.goal_radius_m
        self.collision_clearance_margin_m = float(episode.get("collision_clearance_margin_m", 0.0))
        self.initial_heading_noise_rad = float(episode.get("initial_heading_noise_rad", 0.0))
        self.linear_bounds_mps = self.robot_profile.action.linear_velocity_bounds_mps
        self.angular_bounds_radps = self.robot_profile.action.angular_velocity_bounds_radps
        self.linear_accel_limit_mps2 = float(self.robot_profile.action.linear_accel_limit_mps2)
        self.angular_accel_limit_radps2 = float(self.robot_profile.action.angular_accel_limit_radps2)
        self.apply_wheel_speed_limits = bool(self.robot_profile.action.apply_wheel_speed_limits)
        self.wheel_max_radps = float(self.robot_profile.action.wheel_max_radps)
        self.wheel_radius_m = float(self.robot_profile.geometry.wheel_radius_m)
        self.wheel_separation_m = float(self.robot_profile.geometry.wheel_separation_m)
        self.integration_substeps = int(self.robot_profile.dynamics.integration_substeps)
        self.reward_config = dict(self.config.get("reward", {}))
        self.reward_mode = str(self.reward_config.get("mode", "legacy")).lower()
        self.dynamic_proximity_config = DynamicProximityRewardConfig.from_mapping(
            self.reward_config.get("dynamic_proximity")
        )
        dynamic_proximity_config_mapping = (
            dict(self.reward_config.get("dynamic_proximity", {}))
            if isinstance(self.reward_config.get("dynamic_proximity"), dict)
            else {}
        )
        self.ground_truth_used_for_reward_shaping = bool(
            dynamic_proximity_config_mapping.get(
                "ground_truth_used_for_reward_shaping",
                self.dynamic_proximity_config.enabled,
            )
        )
        self.path_guidance_config = dict(self.config.get("path_guidance", {}))
        observation_config = dict(self.config.get("observation", {}))
        self.observation_type = str(observation_config.get("type", "dugm")).lower()
        if self.observation_type != "dugm":
            raise ValueError("The public 2D environment requires observation.type: dugm.")
        lidar_input = dict(observation_config.get("lidar_input", {}))
        if int(lidar_input.get("num_rays", self.num_rays)) != self.num_rays:
            raise ValueError("Observation and robot-profile LiDAR ray counts must match.")
        if str(lidar_input.get("downsample", "none")) != "none":
            raise ValueError("DUGM 2D consumes the raw scan; lidar_input.downsample must be 'none'.")
        self.angle_min_rad = float(lidar_input.get("angle_min_rad", -math.pi))
        self.angle_increment_rad = self.fov_rad / float(self.num_rays)
        self.grid_encoder = GridEncoder.from_config(observation_config)
        self.observation_source = str(observation_config.get("source", self.observation_type))
        self.visible_occupancy_cost_enabled = True
        default_dynamic_velocity_source = "estimated"
        self.dynamic_velocity_source = str(
            observation_config.get("dynamic_velocity_source", default_dynamic_velocity_source)
        ).lower()
        self.ground_truth_allowed_as_estimator_input = bool(
            observation_config.get("ground_truth_allowed_as_estimator_input", False)
        )
        self.ground_truth_used_for_observation = bool(
            observation_config.get(
                "ground_truth_used_for_observation",
                self.observation_source == "visible_occupancy_cost",
            )
        )
        if self.observation_source != "estimated_dugm" or self.dynamic_velocity_source != "estimated":
            raise ValueError("The public DUGM observation must use estimated tracks from LiDAR.")
        if self.ground_truth_used_for_observation or self.ground_truth_allowed_as_estimator_input:
            raise ValueError("Simulator actor truth cannot enter the public DUGM observation.")
        self.velocity_cost_config = dict(observation_config.get("velocity_cost", {}))
        self.velocity_cost_visibility_mode = str(self.velocity_cost_config.get("visibility_mode", "visible")).lower()
        if self.velocity_cost_visibility_mode not in {"visible", "all"}:
            raise ValueError("observation.velocity_cost.visibility_mode must be 'visible' or 'all'.")
        self.velocity_cost_visibility_range_tolerance_m = float(
            self.velocity_cost_config.get("visibility_range_tolerance_m", 0.35)
        )
        if self.velocity_cost_visibility_range_tolerance_m < 0.0:
            raise ValueError("observation.velocity_cost.visibility_range_tolerance_m must be non-negative.")
        self.visible_static_endpoint_inflation_radius_m = resolve_static_endpoint_inflation_radius(
            self.velocity_cost_config
        )
        self.velocity_cost_dynamic_cost_mode = str(
            self.velocity_cost_config.get("dynamic_cost_mode", "deterministic")
        ).lower()
        if self.velocity_cost_dynamic_cost_mode in {"trajectory", "fixed_trajectory"}:
            self.velocity_cost_dynamic_cost_mode = "deterministic"
        if self.velocity_cost_dynamic_cost_mode not in {
            "deterministic",
            "linear_rollout",
            "uncertainty_gaussian",
        }:
            raise ValueError(
                "observation.velocity_cost.dynamic_cost_mode must be one of "
                "'deterministic', 'linear_rollout', or 'uncertainty_gaussian'."
            )
        self.visible_static_visibility_range_tolerance_m = float(
            self.velocity_cost_config.get(
                "static_visibility_range_tolerance_m",
                self.velocity_cost_visibility_range_tolerance_m,
            )
        )
        if self.visible_static_visibility_range_tolerance_m < 0.0:
            raise ValueError("observation.velocity_cost.static_visibility_range_tolerance_m must be non-negative.")
        self.velocity_cost_rasterizer: GtVelocityCostmapRasterizer | EstimatedVelocityCostmapRasterizer | None = None
        self.estimated_velocity_estimator: DugmEstimator | None = None
        self.estimated_velocity_estimator_config_path: str | None = None
        self.estimated_velocity_estimator_config_hash: str | None = None
        self.estimated_velocity_valid_max_std_mps = float(
            self.velocity_cost_config.get("estimated_velocity_valid_max_std_mps", 0.30)
        )
        self.estimated_track_default_radius_m = float(
            self.velocity_cost_config.get("estimated_track_default_radius_m", 0.25)
        )
        self.estimated_track_min_radius_m = float(self.velocity_cost_config.get("estimated_track_min_radius_m", 0.10))
        self.estimated_track_max_radius_m = float(self.velocity_cost_config.get("estimated_track_max_radius_m", 0.80))
        self.estimated_track_require_current_measurement = bool(
            self.velocity_cost_config.get("estimated_track_require_current_measurement", True)
        )
        self.estimated_track_suppress_maneuver_tracks = bool(
            self.velocity_cost_config.get("estimated_track_suppress_maneuver_tracks", True)
        )
        if self.estimated_velocity_valid_max_std_mps < 0.0:
            raise ValueError("observation.velocity_cost.estimated_velocity_valid_max_std_mps must be non-negative.")
        if (
            self.estimated_track_default_radius_m < 0.0
            or self.estimated_track_min_radius_m < 0.0
            or self.estimated_track_max_radius_m < 0.0
        ):
            raise ValueError("observation.velocity_cost estimated track radii must be non-negative.")
        if self.estimated_track_max_radius_m < self.estimated_track_min_radius_m:
            raise ValueError(
                "observation.velocity_cost.estimated_track_max_radius_m must be >= estimated_track_min_radius_m."
            )
        if self.visible_occupancy_cost_enabled:
            if self.observation_source != "estimated_dugm":
                raise ValueError("observation.source must be estimated_dugm.")
            if self.ground_truth_allowed_as_estimator_input:
                raise ValueError("Ground truth must not be allowed as estimator input for occupied_costmap observations.")
            if self.observation_source == "visible_occupancy_cost":
                if self.velocity_cost_dynamic_cost_mode != "deterministic":
                    raise ValueError("Ground-truth visible_occupancy_cost only supports deterministic dynamic_cost_mode.")
                if self.dynamic_velocity_source != "ground_truth":
                    raise ValueError("visible_occupancy_cost must set dynamic_velocity_source: ground_truth.")
                if not self.ground_truth_used_for_observation:
                    raise ValueError(
                        "Oracle visible_occupancy_cost observations must set ground_truth_used_for_observation: true."
                    )
            else:
                if self.dynamic_velocity_source != "estimated":
                    raise ValueError(
                        "estimated_visible_occupancy_cost must set dynamic_velocity_source: estimated."
                    )
                if self.ground_truth_used_for_observation:
                    raise ValueError(
                        "estimated_visible_occupancy_cost must set ground_truth_used_for_observation: false."
                    )
                estimator_config_path = _resolve_config_path(
                    observation_config.get("estimator_config", DEFAULT_DUGM_ESTIMATOR_CONFIG),
                    base_dir=Path(str(self.config.get("_config_dir", "."))),
                )
                self.estimated_velocity_estimator = DugmEstimator(
                    config_path=estimator_config_path,
                    implementation=str(observation_config.get("estimator_implementation", "optimized")),
                    diagnostics_mode=str(observation_config.get("estimator_diagnostics_mode", "summary")),
                )
                self.estimated_velocity_estimator_config_path = str(estimator_config_path)
                self.estimated_velocity_estimator_config_hash = self.estimated_velocity_estimator.config_hash
                self._validate_estimated_velocity_estimator_contract()
            use_uncertainty_gaussian = (
                self.dynamic_velocity_source == "estimated"
                and self.velocity_cost_dynamic_cost_mode == "uncertainty_gaussian"
            )
            if use_uncertainty_gaussian:
                fallback_acceleration_std = self._estimated_velocity_fallback_process_acceleration_std_mps2()
                self.velocity_cost_rasterizer = UncertaintyAwareEstimatedVelocityCostmapRasterizer(
                    self.grid_encoder,
                    prediction_horizon_s=float(self.velocity_cost_config.get("prediction_horizon_s", 4.0)),
                    prediction_dt_s=float(self.velocity_cost_config.get("prediction_dt_s", 0.2)),
                    sigma_shape_m=float(self.velocity_cost_config.get("sigma_shape_m", 0.30)),
                    covariance_epsilon=float(self.velocity_cost_config.get("covariance_epsilon", 1.0e-4)),
                    mahalanobis_cutoff_sq=float(self.velocity_cost_config.get("mahalanobis_cutoff_sq", 16.0)),
                    covariance_eigenvalue_min_m2=float(
                        self.velocity_cost_config.get("covariance_eigenvalue_min_m2", 1.0e-4)
                    ),
                    covariance_eigenvalue_max_m2=float(
                        self.velocity_cost_config.get("covariance_eigenvalue_max_m2", 9.0)
                    ),
                    process_noise_source=str(
                        self.velocity_cost_config.get(
                            "process_noise_source",
                            "track_last_process_acceleration_std_mps2",
                        )
                    ),
                    fallback_process_acceleration_std_mps2=fallback_acceleration_std,
                    uncertainty_frame=str(self.velocity_cost_config.get("uncertainty_frame", "robot_xy")),
                    human_speed_max_mps=_optional_float(self.velocity_cost_config.get("human_speed_max_mps")),
                    clip_prediction_speed_to_max=bool(
                        self.velocity_cost_config.get("clip_prediction_speed_to_max", True)
                    ),
                    longitudinal_process_acceleration_scale=float(
                        self.velocity_cost_config.get("longitudinal_process_acceleration_scale", 1.0)
                    ),
                    lateral_process_acceleration_scale=float(
                        self.velocity_cost_config.get("lateral_process_acceleration_scale", 1.0)
                    ),
                    longitudinal_velocity_covariance_scale=float(
                        self.velocity_cost_config.get("longitudinal_velocity_covariance_scale", 1.0)
                    ),
                    lateral_velocity_covariance_scale=float(
                        self.velocity_cost_config.get("lateral_velocity_covariance_scale", 1.0)
                    ),
                    covariance_eigenvalue_max_longitudinal_m2=_optional_float(
                        self.velocity_cost_config.get("covariance_eigenvalue_max_longitudinal_m2")
                    ),
                    covariance_eigenvalue_max_lateral_m2=_optional_float(
                        self.velocity_cost_config.get("covariance_eigenvalue_max_lateral_m2")
                    ),
                    low_speed_isotropic_threshold_mps=float(
                        self.velocity_cost_config.get("low_speed_isotropic_threshold_mps", 0.03)
                    ),
                    temporal_decay=bool(self.velocity_cost_config.get("temporal_decay", False)),
                    combine_obstacles=str(self.velocity_cost_config.get("combine_obstacles", "max")),
                    stationary_speed_epsilon_mps=float(
                        self.velocity_cost_config.get("stationary_speed_epsilon_mps", 0.03)
                    ),
                )
            elif self.dynamic_velocity_source == "estimated" and self.velocity_cost_dynamic_cost_mode == "linear_rollout":
                self.velocity_cost_rasterizer = LinearVelocityRolloutCostmapRasterizer(
                    self.grid_encoder,
                    prediction_horizon_s=float(self.velocity_cost_config.get("prediction_horizon_s", 4.0)),
                    prediction_dt_s=float(self.velocity_cost_config.get("prediction_dt_s", 0.2)),
                    human_speed_max_mps=_optional_float(self.velocity_cost_config.get("human_speed_max_mps")),
                    clip_prediction_speed_to_max=bool(
                        self.velocity_cost_config.get("clip_prediction_speed_to_max", True)
                    ),
                    stationary_speed_epsilon_mps=float(
                        self.velocity_cost_config.get("stationary_speed_epsilon_mps", 0.03)
                    ),
                )
            else:
                rasterizer_class: type[GtVelocityCostmapRasterizer] | type[EstimatedVelocityCostmapRasterizer] = (
                    GtVelocityCostmapRasterizer
                    if self.dynamic_velocity_source == "ground_truth"
                    else EstimatedVelocityCostmapRasterizer
                )
                self.velocity_cost_rasterizer = rasterizer_class(
                    self.grid_encoder,
                    prediction_horizon_s=float(self.velocity_cost_config.get("prediction_horizon_s", 5.0)),
                    trajectory_dt_s=float(self.velocity_cost_config.get("trajectory_dt_s", 0.25)),
                    core_inflation_radius_m=resolve_velocity_cost_core_inflation_radius(
                        self.velocity_cost_config,
                        robot_radius_m=self.robot_radius_m,
                        collision_clearance_margin_m=self.collision_clearance_margin_m,
                    ),
                    uncertainty_radius_m=float(self.velocity_cost_config.get("uncertainty_radius_m", 0.50)),
                    stationary_speed_epsilon_mps=float(
                        self.velocity_cost_config.get("stationary_speed_epsilon_mps", 0.03)
                    ),
                )
            grid_channels = [
                str(value)
                for value in observation_config.get("channels", ("observed_unsafe", "visible_occupancy_cost"))
            ]
            if grid_channels != ["observed_unsafe", "visible_occupancy_cost"]:
                raise ValueError("occupied_costmap channels must be ['observed_unsafe', 'visible_occupancy_cost'].")
        else:
            grid_config = dict(observation_config.get("grid", {}))
            grid_channels = [str(value) for value in grid_config.get("channels", ("occupancy", "observed"))]
        expected_shape = tuple(int(value) for value in observation_config.get("shape", self.grid_encoder.output_shape))
        if expected_shape != tuple(self.grid_encoder.output_shape):
            raise ValueError(
                f"Configured grid shape {expected_shape} does not match encoder output {self.grid_encoder.output_shape}."
            )
        self.observation_layout = {
            "type": "dict",
            "keys": ["dugm", "state"],
            "dugm_shape": list(self.grid_encoder.output_shape),
            "dugm_dtype": "float32",
            "dugm_channels": grid_channels,
            "cnn_channel_order": grid_channels,
            "channel_layout": "channels_first",
            "state_shape": [4],
            "state_dtype": "float32",
            "state_order": ["goal_distance", "goal_bearing", "previous_linear_command", "previous_angular_command"],
            "raw_lidar_num_rays": int(self.num_rays),
            "history_len": 1,
            "observation_type": self.observation_type,
            "observation_source": self.observation_source,
            "visible_occupancy_cost_enabled": self.visible_occupancy_cost_enabled,
            "dynamic_velocity_source": self.dynamic_velocity_source,
            "estimated_velocity_estimator_enabled": self.estimated_velocity_estimator is not None,
            "estimator_config": self.estimated_velocity_estimator_config_path,
            "estimator_config_hash": self.estimated_velocity_estimator_config_hash,
            "ground_truth_allowed_as_estimator_input": self.ground_truth_allowed_as_estimator_input,
            "ground_truth_used_for_observation": self.ground_truth_used_for_observation,
            "ground_truth_used_for_reward_shaping": self.ground_truth_used_for_reward_shaping,
            "velocity_cost": self._velocity_cost_layout(),
            "scan_pose_source": "scan_scene_pose",
        }
        self._observation_cache_step: int | None = None
        self._observation_cache: dict[str, np.ndarray] | None = None
        self.scan_generation_count = 0
        self._episode_index = 0
        self.last_raw_scan = np.full(self.num_rays, self.max_range_m, dtype=np.float32)
        self.last_raw_valid_mask = np.ones(self.num_rays, dtype=bool)
        self.last_base_grid = np.zeros(self.grid_encoder.output_shape, dtype=np.float32)
        self.last_grid = np.zeros(self.grid_encoder.output_shape, dtype=np.float32)
        self.last_estimated_velocity_tracked_grid: np.ndarray | None = None
        self.last_velocity_cost_diagnostics: dict[str, Any] = {}
        self.last_static_cost = np.zeros(self.grid_encoder.output_shape[1:], dtype=np.float32)
        self.last_dynamic_cost = np.zeros(self.grid_encoder.output_shape[1:], dtype=np.float32)
        self.last_visible_static_scan_source = "disabled"
        self._visible_static_scan_cache = None
        self._last_requested_wheels_radps = np.zeros(2, dtype=np.float32)
        self._last_executed_wheels_radps = np.zeros(2, dtype=np.float32)
        self._last_wheel_speed_limited = False

        self.action_space = spaces.Box(
            low=np.asarray([-1.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0], dtype=np.float32),
        )
        self.observation_space = spaces.Dict(
            {
                "dugm": spaces.Box(low=0.0, high=1.0, shape=self.grid_encoder.output_shape, dtype=np.float32),
                "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        level = int(options.get("level", self.level))
        level_config = dict(dict(self.config.get("levels", {})).get(level, {}))
        self.max_steps = int(level_config.get("max_steps", self.default_max_steps))
        if self.max_steps <= 0:
            raise ValueError(f"levels[{level}].max_steps must be positive.")
        scene_seed = int(seed) if seed is not None and self._uses_fixed_scenes(level) else int(self.rng.integers(0, np.iinfo(np.int32).max))
        self.scene = self.generator.sample(level, seed=scene_seed)
        heading = math.atan2(self.scene.goal_xy[1] - self.scene.start_xy[1], self.scene.goal_xy[0] - self.scene.start_xy[0])
        fixed_heading_error = self.scene.metadata.get("initial_heading_error_rad")
        heading_error = float(fixed_heading_error) if fixed_heading_error is not None else 0.0
        initial_heading_noise_rad = float(level_config.get("initial_heading_noise_rad", self.initial_heading_noise_rad))
        if fixed_heading_error is None and initial_heading_noise_rad > 0.0:
            heading_error = float(self.rng.uniform(-initial_heading_noise_rad, initial_heading_noise_rad))
        heading = _wrap_angle(heading + heading_error)
        self.scene.metadata["initial_heading_error_rad"] = float(heading_error)
        self.pose = np.asarray([self.scene.start_xy[0], self.scene.start_xy[1], heading], dtype=np.float32)
        self.command = np.zeros(2, dtype=np.float32)
        self._last_requested_wheels_radps = np.zeros(2, dtype=np.float32)
        self._last_executed_wheels_radps = np.zeros(2, dtype=np.float32)
        self._last_wheel_speed_limited = False
        self.initial_goal_distance_m = max(1e-6, self._goal_distance())
        self.previous_goal_distance_m = self.initial_goal_distance_m
        self.step_count = 0
        self._observation_cache_step = None
        self._observation_cache = None
        self.scan_generation_count = 0
        self.last_velocity_cost_diagnostics = {}
        self.last_static_cost.fill(0.0)
        self.last_dynamic_cost.fill(0.0)
        self.last_estimated_velocity_tracked_grid = None
        self.last_visible_static_scan_source = "disabled"
        self._visible_static_scan_cache = (
            build_static_obstacle_scan_cache(self.scene)
            if self.visible_occupancy_cost_enabled and self.dynamic_velocity_source == "ground_truth"
            else None
        )
        self._episode_index += 1
        if self.estimated_velocity_estimator is not None:
            self.estimated_velocity_estimator.reset(reason="episode_reset")
        self._initialize_path_guidance()
        waypoint_distance, waypoint_bearing = self._waypoint_metrics()
        return self._observation(), {
            "scene": self.scene.to_dict(),
            "robot_profile": self.robot_profile.name,
            "observation_keys": list(self.observation_space.spaces),
            "observation_layout": dict(self.observation_layout),
            "initial_heading_error_rad": float(heading_error),
            "initial_goal_distance_m": float(self.initial_goal_distance_m),
            "episode_max_steps": int(self.max_steps),
            "path_guidance_status": self.path_guidance_status,
            "initial_geodesic_cost_to_go_m": self.previous_geodesic_cost_m,
            "waypoint_xy": self.current_waypoint_xy.astype(float).tolist() if self.current_waypoint_xy is not None else None,
            "waypoint_distance_m": waypoint_distance,
            "waypoint_bearing_rad": waypoint_bearing,
            "dynamic_velocity_source": self.dynamic_velocity_source,
            "ground_truth_used_for_observation": self.ground_truth_used_for_observation,
            "ground_truth_used_for_reward_shaping": self.ground_truth_used_for_reward_shaping,
            "estimator_config_hash": self.estimated_velocity_estimator_config_hash,
            "estimator_reset_reason": (
                self.estimated_velocity_estimator.sequence_guard.last_reset_reason
                if self.estimated_velocity_estimator is not None
                else None
            ),
        }

    def step(self, action):
        scene = self._require_scene()
        old_time_s = self.step_count * self.dt_s
        old_pose = self.pose.astype(np.float32).copy()
        old_distance = self._goal_distance()
        old_clearance = min_clearance(scene, point_xy=self.pose[:2], radius_m=self.robot_radius_m, time_s=old_time_s)
        previous_command = self.command.astype(np.float32).copy()
        command = self._apply_command_dynamics(np.asarray(action, dtype=np.float32))
        dynamic_proximity_result = compute_dynamic_proximity_reward(
            scene=scene,
            robot_pose_xyyaw=old_pose,
            time_s=old_time_s,
            command=command,
            robot_radius_m=self.robot_radius_m,
            collision_clearance_margin_m=self.collision_clearance_margin_m,
            config=self.dynamic_proximity_config,
        )
        command_delta = command - previous_command
        v_mps = float(command[0])
        w_radps = float(command[1])
        sub_dt = self.dt_s / max(1, self.integration_substeps)
        for _ in range(max(1, self.integration_substeps)):
            self.pose[2] = _wrap_angle(float(self.pose[2]) + w_radps * sub_dt)
            self.pose[0] += v_mps * math.cos(float(self.pose[2])) * sub_dt
            self.pose[1] += v_mps * math.sin(float(self.pose[2])) * sub_dt
        self.step_count += 1
        time_s = self.step_count * self.dt_s

        new_distance = self._goal_distance()
        progress = old_distance - new_distance
        bearing = self._goal_bearing()
        heading_alignment = 0.5 * (math.cos(bearing) + 1.0)
        stall_speed_epsilon_mps = self._reward_weight("stall_speed_epsilon_mps", 0.05)
        goal_proximity_scale_m = max(1e-6, self._reward_weight("goal_proximity_scale_m", 1.0))
        goal_proximity_delta = math.exp(-new_distance / goal_proximity_scale_m) - math.exp(-old_distance / goal_proximity_scale_m)
        clearance = min_clearance(scene, point_xy=self.pose[:2], radius_m=self.robot_radius_m, time_s=time_s)
        near_obstacle_penalty = self._near_obstacle_penalty(clearance)
        obstacle_approach_penalty, obstacle_closing_norm = self._obstacle_approach_penalty(old_clearance, clearance)
        normalized_linear_delta = abs(float(command_delta[0])) / max(1e-6, self.linear_accel_limit_mps2 * self.dt_s)
        normalized_angular_delta = abs(float(command_delta[1])) / max(1e-6, self.angular_accel_limit_radps2 * self.dt_s)
        smooth_linear_penalty = self._reward_weight("smooth_linear_accel_weight", 0.0) * normalized_linear_delta
        smooth_angular_penalty = self._reward_weight("smooth_angular_accel_weight", 0.0) * normalized_angular_delta
        angular_velocity_free_radps = max(0.0, self._reward_weight("angular_velocity_free_radps", 1.2))
        angular_penalty = self._reward_weight("angular_velocity_weight", 0.5) * max(0.0, abs(w_radps) - angular_velocity_free_radps)
        heading_reward = self._heading_reward(
            bearing,
            heading_alignment,
            progress,
            v_mps,
            stall_speed_epsilon_mps,
        )
        near_goal_speed_penalty = self._near_goal_speed_penalty(new_distance, v_mps)
        near_goal_heading_alignment_reward = self._near_goal_heading_alignment_reward(new_distance, heading_alignment)
        near_goal_low_speed_reward = self._near_goal_low_speed_reward(new_distance, v_mps)
        low_speed = bool(v_mps <= stall_speed_epsilon_mps)
        stall_penalty = self._reward_weight("stall_penalty", 0.0) if low_speed else 0.0
        reward_terms = {
            "progress_to_goal": float(self._reward_weight("progress_weight", 1.0) * progress),
            "goal_proximity_delta_reward": float(self._reward_weight("goal_proximity_delta_weight", 0.0) * goal_proximity_delta),
            "heading_alignment_reward": float(heading_reward),
            "near_goal_heading_alignment_reward": float(near_goal_heading_alignment_reward),
            "near_goal_low_speed_reward": float(near_goal_low_speed_reward),
            "success_bonus": 0.0,
            "collision_penalty": 0.0,
            "timeout_penalty": 0.0,
            "time_penalty": -float(self._reward_weight("time_penalty_per_step", 0.0)),
            "near_obstacle_penalty": -float(near_obstacle_penalty),
            "obstacle_approach_penalty": -float(obstacle_approach_penalty),
            "near_goal_speed_penalty": -float(near_goal_speed_penalty),
            "angular_velocity_penalty": -float(angular_penalty),
            "stall_penalty": -float(stall_penalty),
            "smooth_linear_accel_penalty": -float(smooth_linear_penalty),
            "smooth_angular_accel_penalty": -float(smooth_angular_penalty),
        }
        if self.dynamic_proximity_config.enabled:
            reward_terms["dynamic_proximity_penalty"] = float(dynamic_proximity_result.reward)
        guidance_info = self._waypoint_reward_terms()
        if self.reward_mode == "waypoint_geodesic":
            for key in (
                "progress_to_goal",
                "goal_proximity_delta_reward",
                "heading_alignment_reward",
                "near_goal_heading_alignment_reward",
                "near_goal_low_speed_reward",
                "near_goal_speed_penalty",
            ):
                reward_terms[key] = 0.0
            reward_terms.update(
                {
                    "geodesic_progress_reward": float(guidance_info["geodesic_progress_reward"]),
                    "waypoint_heading_reward": float(guidance_info["waypoint_heading_reward"]),
                    "heading_potential_reward": float(guidance_info["heading_potential_reward"]),
                }
            )

        terminated = False
        truncated = False
        success = new_distance <= self.goal_success_distance_m
        collision = (
            collides(scene, point_xy=self.pose[:2], radius_m=self.robot_radius_m, time_s=time_s)
            or clearance <= self.collision_clearance_margin_m
        )
        if success:
            reward_terms["success_bonus"] = self._reward_weight("success_bonus", 5.0)
            terminated = True
        if collision:
            reward_terms["collision_penalty"] = self._reward_weight("collision_penalty", -5.0)
            terminated = True
        if self.step_count >= self.max_steps:
            reward_terms["timeout_penalty"] = self._reward_weight("timeout_penalty", -1.0)
            truncated = True

        reward = float(sum(reward_terms.values()))
        self.previous_goal_distance_m = new_distance
        info = {
            "success": bool(success),
            "collision": bool(collision),
            # The PPO-Lagrangian CMDP cost is deliberately separate from V9
            # task reward.  It reuses this exact collision decision, including
            # footprint contact and the configured clearance-margin collision.
            "cost": float(collision),
            "timeout": bool(truncated and not terminated),
            "episode_max_steps": int(self.max_steps),
            "goal_distance_m": float(new_distance),
            "goal_bearing_rad": float(bearing),
            "goal_radius_m": float(self.goal_radius_m),
            "goal_success_distance_m": float(self.goal_success_distance_m),
            "collision_clearance_margin_m": float(self.collision_clearance_margin_m),
            "previous_minimum_clearance_m": float(old_clearance),
            "minimum_clearance_m": float(clearance),
            "obstacle_closing_norm": float(obstacle_closing_norm),
            "low_speed": bool(low_speed),
            "stall_speed_epsilon_mps": float(stall_speed_epsilon_mps),
            "low_speed_timeout_rate_threshold": float(np.clip(self._reward_weight("low_speed_timeout_rate_threshold", 0.5), 0.0, 1.0)),
            "command": {"linear_mps": v_mps, "angular_radps": w_radps},
            "command_delta": {"linear_mps": float(command_delta[0]), "angular_radps": float(command_delta[1])},
            "wheel_speed_limit": {
                "enabled": bool(self.apply_wheel_speed_limits),
                "limited": bool(self._last_wheel_speed_limited),
                "wheel_max_radps": float(self.wheel_max_radps),
                "wheel_radius_m": float(self.wheel_radius_m),
                "wheel_separation_m": float(self.wheel_separation_m),
            },
            "requested_wheels_radps": {
                "left": float(self._last_requested_wheels_radps[0]),
                "right": float(self._last_requested_wheels_radps[1]),
            },
            "executed_wheels_radps": {
                "left": float(self._last_executed_wheels_radps[0]),
                "right": float(self._last_executed_wheels_radps[1]),
            },
            "normalized_command_delta": {
                "linear": float(normalized_linear_delta),
                "angular": float(normalized_angular_delta),
            },
            "reward_terms": reward_terms,
            "geodesic_cost_to_go_m": guidance_info["geodesic_cost_to_go_m"],
            "geodesic_progress_m": guidance_info["geodesic_progress_m"],
            "waypoint_xy": guidance_info["waypoint_xy"],
            "waypoint_distance_m": guidance_info["waypoint_distance_m"],
            "waypoint_bearing_rad": guidance_info["waypoint_bearing_rad"],
            "path_guidance_status": self.path_guidance_status,
            "dynamic_velocity_source": self.dynamic_velocity_source,
            "ground_truth_used_for_observation": self.ground_truth_used_for_observation,
            "ground_truth_used_for_reward_shaping": self.ground_truth_used_for_reward_shaping,
        }
        info.update(dynamic_proximity_result.diagnostics)
        observation = self._observation()
        if self.last_velocity_cost_diagnostics:
            info.update(self.last_velocity_cost_diagnostics)
        return observation, reward, terminated, truncated, info

    def _observation(self) -> dict[str, np.ndarray]:
        if self._observation_cache_step == int(self.step_count) and self._observation_cache is not None:
            return {key: value.copy() for key, value in self._observation_cache.items()}
        scene = self._require_scene()
        noise_rng = self.rng if self.observation_noise else None
        ranges = scan_scene(
            scene,
            origin_xy=self.pose[:2],
            yaw_rad=float(self.pose[2]),
            time_s=self.step_count * self.dt_s,
            num_rays=self.num_rays,
            fov_rad=self.fov_rad,
            min_range_m=self.min_range_m,
            max_range_m=self.max_range_m,
            range_bias_std_m=self.lidar_noise.range_bias_std_m if self.observation_noise and self.lidar_noise.enabled else 0.0,
            noise_std_m=self.lidar_noise.range_noise_std_m if self.observation_noise and self.lidar_noise.enabled else 0.0,
            range_noise_ratio=self.lidar_noise.range_noise_ratio if self.observation_noise and self.lidar_noise.enabled else 0.0,
            max_abs_noise_m=self.lidar_noise.max_abs_noise_m if self.observation_noise and self.lidar_noise.enabled else 0.0,
            max_abs_noise_ratio=self.lidar_noise.max_abs_noise_ratio if self.observation_noise and self.lidar_noise.enabled else 0.0,
            dropout_prob=self.lidar_noise.dropout_prob if self.observation_noise and self.lidar_noise.enabled else 0.0,
            false_near_prob=self.lidar_noise.false_near_prob if self.observation_noise and self.lidar_noise.enabled else 0.0,
            max_range_spike_prob=self.lidar_noise.max_range_spike_prob if self.observation_noise and self.lidar_noise.enabled else 0.0,
            rng=noise_rng,
        )
        self.scan_generation_count += 1
        self.last_raw_scan = ranges.astype(np.float32, copy=True)
        valid_mask = np.isfinite(ranges) & (ranges >= self.min_range_m)
        self.last_raw_valid_mask = valid_mask.astype(bool, copy=True)
        distance = self._goal_distance()
        bearing = self._goal_bearing()
        timestamp_s = float(self.step_count * self.dt_s)
        base_grid = self.grid_encoder.encode(
            ranges,
            valid_mask=valid_mask,
            angle_min_rad=self.angle_min_rad,
            angle_increment_rad=self.angle_increment_rad,
            min_range_m=self.min_range_m,
            max_range_m=self.max_range_m,
        )
        self.last_base_grid = base_grid.astype(np.float32, copy=True)
        if self.visible_occupancy_cost_enabled:
            grid = self._occupied_costmap_observation(
                base_grid,
                scene=scene,
                timestamp_s=timestamp_s,
                ranges=ranges,
                valid_mask=valid_mask,
            )
        else:
            self.last_velocity_cost_diagnostics = {}
            grid = base_grid
        goal_obs = np.asarray(
            [
                np.clip(distance / self.initial_goal_distance_m, 0.0, 1.0),
                np.clip(bearing / math.pi, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        velocity_obs = np.asarray(
            [
                _normalize_symmetric(float(self.command[0]), self.linear_bounds_mps),
                _normalize_symmetric(float(self.command[1]), self.angular_bounds_radps),
            ],
            dtype=np.float32,
        )
        state = np.concatenate([goal_obs, velocity_obs], dtype=np.float32)
        observation = {"dugm": grid, "state": state}
        self.last_grid = grid.astype(np.float32, copy=True)
        self._observation_cache_step = int(self.step_count)
        self._observation_cache = {key: value.copy() for key, value in observation.items()}
        return observation

    def _occupied_costmap_observation(
        self,
        base_grid: np.ndarray,
        *,
        scene: Scene2D,
        timestamp_s: float,
        ranges: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        if self.velocity_cost_rasterizer is None:
            raise RuntimeError("occupied_costmap observation is enabled without a velocity cost rasterizer.")
        if self.dynamic_velocity_source == "estimated":
            dynamic_cost, diagnostics = self._estimated_dynamic_velocity_cost(
                base_grid=base_grid,
                ranges=ranges,
                valid_mask=valid_mask,
                timestamp_s=float(timestamp_s),
            )
            static_cost = self._visible_endpoint_cost_from_current_scan(ranges=ranges, valid_mask=valid_mask)
        else:
            dynamic_cost, diagnostics = self._ground_truth_dynamic_velocity_cost(
                scene,
                timestamp_s=float(timestamp_s),
                ranges=ranges,
            )
            static_cost = self._visible_static_endpoint_cost(scene, timestamp_s=float(timestamp_s), ranges=ranges)
        visible_occupancy_cost = np.maximum(static_cost, dynamic_cost).astype(np.float32, copy=False)
        self.last_static_cost = np.asarray(static_cost, dtype=np.float32).copy()
        self.last_dynamic_cost = np.asarray(dynamic_cost, dtype=np.float32).copy()
        diagnostics.update(
            {
                "dynamic_velocity_source": self.dynamic_velocity_source,
                "ground_truth_used_for_observation": self.ground_truth_used_for_observation,
                "ground_truth_used_for_reward_shaping": self.ground_truth_used_for_reward_shaping,
                "visible_static_scan_source": self.last_visible_static_scan_source,
                "visible_static_scan_cache_enabled": self._visible_static_scan_cache is not None,
            }
        )
        diagnostics.update(
            occupied_costmap_diagnostics(
                static_cost=static_cost,
                visible_occupancy_cost=visible_occupancy_cost,
                static_inflation_radius_m=self.visible_static_endpoint_inflation_radius_m,
                static_visibility_range_tolerance_m=self.visible_static_visibility_range_tolerance_m,
            )
        )
        self.last_velocity_cost_diagnostics = diagnostics
        return stack_occupied_costmap(base_grid, visible_occupancy_cost)

    def _ground_truth_dynamic_velocity_cost(
        self,
        scene: Scene2D,
        *,
        timestamp_s: float,
        ranges: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not isinstance(self.velocity_cost_rasterizer, GtVelocityCostmapRasterizer):
            raise RuntimeError("GT dynamic velocity source is enabled without a GT velocity rasterizer.")
        actor_states = visible_dynamic_actor_states(
            scene,
            timestamp_s=float(timestamp_s),
            ranges=ranges,
            robot_pose_xyyaw=self.pose,
            num_rays=self.num_rays,
            angle_min_rad=self.angle_min_rad,
            angle_increment_rad=self.angle_increment_rad,
            min_range_m=self.min_range_m,
            max_range_m=self.max_range_m,
            visibility_mode=self.velocity_cost_visibility_mode,
            visibility_range_tolerance_m=self.velocity_cost_visibility_range_tolerance_m,
        )
        dynamic_cost = self.velocity_cost_rasterizer.rasterize(actor_states, robot_pose_xyyaw=self.pose)
        diagnostics = dict(self.velocity_cost_rasterizer.last_diagnostics)
        diagnostics.update(
            {
                "gt_velocity_cost_total_dynamic_actor_count": int(len(scene.dynamic_actors)),
                "gt_velocity_cost_visibility_mode": self.velocity_cost_visibility_mode,
            }
        )
        return dynamic_cost, diagnostics

    def _estimated_dynamic_velocity_cost(
        self,
        *,
        base_grid: np.ndarray,
        ranges: np.ndarray,
        valid_mask: np.ndarray,
        timestamp_s: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.estimated_velocity_estimator is None:
            raise RuntimeError("estimated dynamic velocity source is enabled without a tracked grid estimator.")
        if not isinstance(self.velocity_cost_rasterizer, EstimatedVelocityCostmapRasterizer):
            raise RuntimeError("estimated dynamic velocity source is enabled without an estimated velocity rasterizer.")
        estimator_tracked_grid = self.estimated_velocity_estimator.update(
            ranges,
            valid_mask,
            scan_pose_xyyaw=self.pose,
            scan_timestamp_s=float(timestamp_s),
            sample_id=(int(self._episode_index), int(self.step_count)),
            base_grid=base_grid,
        )
        self.last_estimated_velocity_tracked_grid = np.asarray(estimator_tracked_grid, dtype=np.float32).copy()
        track_states, track_selection_diagnostics = estimated_dynamic_track_states(
            self.estimated_velocity_estimator.tracker.tracks,
            velocity_valid_max_std_mps=self.estimated_velocity_valid_max_std_mps,
            default_radius_m=self.estimated_track_default_radius_m,
            min_radius_m=self.estimated_track_min_radius_m,
            max_radius_m=self.estimated_track_max_radius_m,
            require_current_measurement=self.estimated_track_require_current_measurement,
            suppress_maneuver_tracks=self.estimated_track_suppress_maneuver_tracks,
            fallback_process_acceleration_std_mps2=self._estimated_velocity_fallback_process_acceleration_std_mps2(),
            return_diagnostics=True,
        )
        dynamic_cost = self.velocity_cost_rasterizer.rasterize(track_states)
        estimator_diagnostics = dict(self.estimated_velocity_estimator.last_diagnostics or {})
        diagnostics = dict(self.velocity_cost_rasterizer.last_diagnostics)
        diagnostics.update(track_selection_diagnostics)
        diagnostics.update(
            {
                "estimator_config_hash": self.estimated_velocity_estimator_config_hash,
                "estimator_sequence_action": estimator_diagnostics.get("sequence_action"),
                "estimator_reset_reason": estimator_diagnostics.get("reset_reason"),
                "estimator_duplicate_sample": bool(estimator_diagnostics.get("duplicate", False)),
                "estimator_base_grid_source": estimator_diagnostics.get("base_grid_source"),
                "active_track_count": int(estimator_diagnostics.get("active_track_count", 0)),
                "accepted_cluster_count": int(estimator_diagnostics.get("accepted_cluster_count", 0)),
                "estimator_valid_hit_count": int(estimator_diagnostics.get("valid_hit_count", 0)),
                "tracked_grid_encoding_ms": float(estimator_diagnostics.get("encoding_ms", 0.0)),
                "estimated_velocity_cost_track_ids": [int(state.track_id) for state in track_states],
                "estimated_velocity_cost_velocity_valid_max_std_mps": float(
                    self.estimated_velocity_valid_max_std_mps
                ),
                "estimated_velocity_cost_track_radius_range_m": [
                    float(self.estimated_track_min_radius_m),
                    float(self.estimated_track_max_radius_m),
                ],
                "estimated_velocity_cost_track_radius_source": "latest_cluster_diameter_clipped",
            }
        )
        return dynamic_cost, diagnostics

    def _estimated_velocity_fallback_process_acceleration_std_mps2(self) -> float:
        if self.estimated_velocity_estimator is None:
            return 0.0
        value = float(
            self.estimated_velocity_estimator.tracker.tracking_config.get("process_acceleration_std_mps2", 0.0)
        )
        return value if np.isfinite(value) and value >= 0.0 else 0.0

    def _visible_endpoint_cost_from_current_scan(
        self,
        *,
        ranges: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        self.last_visible_static_scan_source = "current_raw_scan_endpoints_no_ground_truth_filter"
        return visible_endpoint_cost_from_ranges(
            self.grid_encoder,
            ranges=ranges,
            valid_mask=valid_mask,
            angle_min_rad=self.angle_min_rad,
            angle_increment_rad=self.angle_increment_rad,
            min_range_m=self.min_range_m,
            max_range_m=self.max_range_m,
            inflation_radius_m=self.visible_static_endpoint_inflation_radius_m,
        )

    def _visible_static_endpoint_cost(
        self,
        scene: Scene2D,
        *,
        timestamp_s: float,
        ranges: np.ndarray,
    ) -> np.ndarray:
        if not scene.dynamic_actors and not self._lidar_noise_active():
            self.last_visible_static_scan_source = "current_scan_no_dynamic_no_noise"
            static_ranges = np.asarray(ranges, dtype=np.float32).reshape(self.num_rays)
        elif self._visible_static_scan_cache is not None:
            self.last_visible_static_scan_source = "static_obstacle_scan_cache"
            static_ranges = scan_static_obstacle_cache(
                self._visible_static_scan_cache,
                origin_xy=self.pose[:2],
                yaw_rad=float(self.pose[2]),
                num_rays=self.num_rays,
                fov_rad=self.fov_rad,
                min_range_m=self.min_range_m,
                max_range_m=self.max_range_m,
            )
        else:
            self.last_visible_static_scan_source = "static_scene_fallback"
            static_scene = Scene2D(
                level=scene.level,
                level_name=scene.level_name,
                bounds_x_m=scene.bounds_x_m,
                bounds_y_m=scene.bounds_y_m,
                start_xy=scene.start_xy,
                goal_xy=scene.goal_xy,
                static_obstacles=scene.static_obstacles,
                dynamic_actors=(),
                floor_regions=scene.floor_regions,
                metadata=scene.metadata,
            )
            static_ranges = scan_scene(
                static_scene,
                origin_xy=self.pose[:2],
                yaw_rad=float(self.pose[2]),
                time_s=float(timestamp_s),
                num_rays=self.num_rays,
                fov_rad=self.fov_rad,
                min_range_m=self.min_range_m,
                max_range_m=self.max_range_m,
            )
        return visible_static_endpoint_cost_from_ranges(
            self.grid_encoder,
            static_ranges=static_ranges,
            measured_ranges=ranges,
            angle_min_rad=self.angle_min_rad,
            angle_increment_rad=self.angle_increment_rad,
            min_range_m=self.min_range_m,
            max_range_m=self.max_range_m,
            visibility_range_tolerance_m=self.visible_static_visibility_range_tolerance_m,
            inflation_radius_m=self.visible_static_endpoint_inflation_radius_m,
        )

    def _lidar_noise_active(self) -> bool:
        return bool(
            self.observation_noise
            and self.lidar_noise.enabled
            and (
                self.lidar_noise.range_bias_std_m > 0.0
                or self.lidar_noise.range_noise_std_m > 0.0
                or self.lidar_noise.range_noise_ratio > 0.0
                or self.lidar_noise.dropout_prob > 0.0
                or self.lidar_noise.false_near_prob > 0.0
                or self.lidar_noise.max_range_spike_prob > 0.0
            )
        )

    def _velocity_cost_layout(self) -> dict[str, Any]:
        if self.velocity_cost_rasterizer is None:
            return {"enabled": False}
        rasterizer = self.velocity_cost_rasterizer
        dynamic_cost_mode = str(getattr(rasterizer, "dynamic_cost_mode", "deterministic"))
        layout = {
            "enabled": True,
            "channel": "visible_occupancy_cost",
            "dynamic_velocity_source": self.dynamic_velocity_source,
            "dynamic_cost_mode": dynamic_cost_mode,
            "prediction_horizon_s": float(rasterizer.prediction_horizon_s),
            "visibility_mode": self.velocity_cost_visibility_mode,
            "visibility_range_tolerance_m": float(self.velocity_cost_visibility_range_tolerance_m),
            "static_endpoint_inflation_radius_m": float(self.visible_static_endpoint_inflation_radius_m),
            "static_visibility_range_tolerance_m": float(self.visible_static_visibility_range_tolerance_m),
            "static_endpoint_inflation_semantics": "cell_overlap_binary_dilation_sets_occupancy",
            "aggregation": "cellwise_max",
        }
        if isinstance(rasterizer, UncertaintyAwareEstimatedVelocityCostmapRasterizer):
            layout.update(
                {
                    "prediction_dt_s": float(rasterizer.prediction_dt_s),
                    "sigma_shape_m": float(rasterizer.sigma_shape_m),
                    "covariance_epsilon": float(rasterizer.covariance_epsilon),
                    "mahalanobis_cutoff_sq": float(rasterizer.mahalanobis_cutoff_sq),
                    "covariance_eigenvalue_min_m2": float(rasterizer.covariance_eigenvalue_min_m2),
                    "covariance_eigenvalue_max_m2": float(rasterizer.covariance_eigenvalue_max_m2),
                    "process_noise_source": rasterizer.process_noise_source,
                    "uncertainty_frame": rasterizer.uncertainty_frame,
                    "human_speed_max_mps": rasterizer.human_speed_max_mps,
                    "clip_prediction_speed_to_max": bool(rasterizer.clip_prediction_speed_to_max),
                    "longitudinal_process_acceleration_scale": float(
                        rasterizer.longitudinal_process_acceleration_scale
                    ),
                    "lateral_process_acceleration_scale": float(rasterizer.lateral_process_acceleration_scale),
                    "longitudinal_velocity_covariance_scale": float(
                        rasterizer.longitudinal_velocity_covariance_scale
                    ),
                    "lateral_velocity_covariance_scale": float(rasterizer.lateral_velocity_covariance_scale),
                    "covariance_eigenvalue_max_longitudinal_m2": float(
                        rasterizer.covariance_eigenvalue_max_longitudinal_m2
                    ),
                    "covariance_eigenvalue_max_lateral_m2": float(
                        rasterizer.covariance_eigenvalue_max_lateral_m2
                    ),
                    "low_speed_isotropic_threshold_mps": float(rasterizer.low_speed_isotropic_threshold_mps),
                    "temporal_decay": bool(rasterizer.temporal_decay),
                    "combine_obstacles": rasterizer.combine_obstacles,
                }
            )
        elif isinstance(rasterizer, LinearVelocityRolloutCostmapRasterizer):
            layout.update(
                {
                    "prediction_dt_s": float(rasterizer.prediction_dt_s),
                    "stationary_speed_epsilon_mps": float(rasterizer.stationary_speed_epsilon_mps),
                    "human_speed_max_mps": rasterizer.human_speed_max_mps,
                    "clip_prediction_speed_to_max": bool(rasterizer.clip_prediction_speed_to_max),
                    "mean_prediction": "constant_velocity_with_optional_speed_clip",
                    "uses_track_covariance": False,
                    "uses_process_noise": False,
                    "uses_track_radius": False,
                    "rasterization": "binary_centerline_cells",
                }
            )
        else:
            layout.update(
                {
                    "trajectory_dt_s": float(rasterizer.trajectory_dt_s),
                    "stationary_speed_epsilon_mps": float(rasterizer.stationary_speed_epsilon_mps),
                    "core_inflation_radius_m": float(rasterizer.core_inflation_radius_m),
                    "uncertainty_radius_m": float(rasterizer.uncertainty_radius_m),
                }
            )
        if self.dynamic_velocity_source == "estimated":
            layout.update(
                {
                    "estimator_config": self.estimated_velocity_estimator_config_path,
                    "estimator_config_hash": self.estimated_velocity_estimator_config_hash,
                    "estimator_input_fields": [
                        "raw_scan",
                        "valid_mask",
                        "scan_pose_xyyaw",
                        "scan_timestamp_s",
                        "sample_id",
                    ],
                    "track_velocity_gate": {
                        "valid_max_std_mps": float(self.estimated_velocity_valid_max_std_mps),
                        "require_current_measurement": bool(self.estimated_track_require_current_measurement),
                        "suppress_maneuver_tracks": bool(self.estimated_track_suppress_maneuver_tracks),
                    },
                    "estimated_track_radius": {
                        "source": "latest_cluster_diameter_clipped",
                        "default_m": float(self.estimated_track_default_radius_m),
                        "min_m": float(self.estimated_track_min_radius_m),
                        "max_m": float(self.estimated_track_max_radius_m),
                    },
                    "ground_truth_inputs": [],
                }
            )
        else:
            layout["ground_truth_inputs"] = [
                "dynamic_actor_center",
                "dynamic_actor_velocity",
                "dynamic_actor_proxy_radius",
                "visible_static_obstacle_endpoints",
            ]
        return layout

    def _validate_estimated_velocity_estimator_contract(self) -> None:
        estimator = self.estimated_velocity_estimator
        if estimator is None:
            return
        geometry = estimator.geometry
        if int(geometry.num_rays) != int(self.num_rays):
            raise ValueError("Estimated velocity estimator LiDAR ray count must match the environment.")
        if not math.isclose(float(geometry.angle_min_rad), float(self.angle_min_rad), abs_tol=1e-9):
            raise ValueError("Estimated velocity estimator angle_min_rad must match the environment.")
        if not math.isclose(float(geometry.angle_increment_rad), float(self.angle_increment_rad), abs_tol=1e-12):
            raise ValueError("Estimated velocity estimator angle_increment_rad must match the environment.")
        if (int(geometry.height_cells), int(geometry.width_cells)) != tuple(self.grid_encoder.output_shape[1:]):
            raise ValueError("Estimated velocity estimator grid shape must match the occupied costmap grid.")
        if not math.isclose(float(geometry.height_m), float(self.grid_encoder.geometry.height_m), abs_tol=1e-9):
            raise ValueError("Estimated velocity estimator grid height must match the occupied costmap grid.")
        if not math.isclose(float(geometry.width_m), float(self.grid_encoder.geometry.width_m), abs_tol=1e-9):
            raise ValueError("Estimated velocity estimator grid width must match the occupied costmap grid.")
        if not math.isclose(float(geometry.resolution_m), float(self.grid_encoder.geometry.resolution_m), abs_tol=1e-12):
            raise ValueError("Estimated velocity estimator grid resolution must match the occupied costmap grid.")

    def _apply_command_dynamics(self, action: np.ndarray) -> np.ndarray:
        target = self._action_to_command(action)
        max_delta = np.asarray(
            [
                self.linear_accel_limit_mps2 * self.dt_s,
                self.angular_accel_limit_radps2 * self.dt_s,
            ],
            dtype=np.float32,
        )
        delta = np.clip(target - self.command, -max_delta, max_delta)
        command_low = np.asarray([self.linear_bounds_mps[0], self.angular_bounds_radps[0]], dtype=np.float32)
        command_high = np.asarray([self.linear_bounds_mps[1], self.angular_bounds_radps[1]], dtype=np.float32)
        command = np.clip(self.command + delta, command_low, command_high).astype(np.float32)
        requested_wheels = self._body_to_wheels(command)
        executed_wheels = requested_wheels
        wheel_speed_limited = False
        if self.apply_wheel_speed_limits:
            executed_wheels, wheel_speed_limited = self._limit_wheel_speed(requested_wheels)
            command = np.clip(self._wheels_to_body(executed_wheels), command_low, command_high).astype(np.float32)
        self._last_requested_wheels_radps = requested_wheels.astype(np.float32, copy=True)
        self._last_executed_wheels_radps = executed_wheels.astype(np.float32, copy=True)
        self._last_wheel_speed_limited = bool(wheel_speed_limited)
        self.command = command
        return self.command

    def _action_to_command(self, action: np.ndarray) -> np.ndarray:
        normalized = np.clip(action.reshape(-1)[:2], self.action_space.low, self.action_space.high).astype(np.float32)
        linear = _scale_unit_interval(float(normalized[0]), self.linear_bounds_mps)
        angular = _scale_unit_interval(float(normalized[1]), self.angular_bounds_radps)
        return np.asarray([linear, angular], dtype=np.float32)

    def _body_to_wheels(self, command: np.ndarray) -> np.ndarray:
        radius = max(float(self.wheel_radius_m), 1e-9)
        half_track = 0.5 * float(self.wheel_separation_m)
        linear = float(command[0])
        angular = float(command[1])
        return np.asarray(
            [
                (linear - angular * half_track) / radius,
                (linear + angular * half_track) / radius,
            ],
            dtype=np.float32,
        )

    def _wheels_to_body(self, wheels: np.ndarray) -> np.ndarray:
        radius = max(float(self.wheel_radius_m), 1e-9)
        track = max(float(self.wheel_separation_m), 1e-9)
        left = float(wheels[0])
        right = float(wheels[1])
        return np.asarray(
            [
                radius * (right + left) * 0.5,
                radius * (right - left) / track,
            ],
            dtype=np.float32,
        )

    def _limit_wheel_speed(self, wheels: np.ndarray) -> tuple[np.ndarray, bool]:
        max_abs = float(np.max(np.abs(wheels)))
        limit = abs(float(self.wheel_max_radps))
        if limit <= 0.0:
            return np.zeros(2, dtype=np.float32), bool(max_abs > 1e-12)
        if max_abs <= limit or max_abs <= 1e-12:
            return wheels.astype(np.float32, copy=True), False
        scale = limit / max_abs
        return (wheels * scale).astype(np.float32), True

    def _reward_weight(self, key: str, default: float) -> float:
        return float(self.reward_config.get(key, default))

    def _near_obstacle_penalty(self, clearance_m: float) -> float:
        weight = self._reward_weight("near_obstacle_weight", 5.0)
        if weight <= 0.0:
            return 0.0
        safe_clearance = max(0.0, self._reward_weight("near_obstacle_clearance_m", 0.6))
        mode = str(self.reward_config.get("near_obstacle_penalty_mode", "linear_distance")).lower()
        if mode in {"normalized_linear", "linear_normalized"}:
            denominator = max(1e-6, safe_clearance - self.collision_clearance_margin_m)
            danger = float(np.clip((safe_clearance - float(clearance_m)) / denominator, 0.0, 1.0))
            return float(weight * danger)
        return float(weight * max(0.0, safe_clearance - float(clearance_m)))

    def _obstacle_approach_penalty(self, old_clearance_m: float, new_clearance_m: float) -> tuple[float, float]:
        weight = self._reward_weight("obstacle_approach_weight", 0.0)
        max_step_distance = max(1e-6, float(self.linear_bounds_mps[1]) * self.dt_s)
        closing_m = max(0.0, float(old_clearance_m) - float(new_clearance_m))
        closing_norm = float(np.clip(closing_m / max_step_distance, 0.0, 1.0))
        if weight <= 0.0:
            return 0.0, closing_norm
        return float(weight * closing_norm), closing_norm

    def _heading_reward(
        self,
        bearing_rad: float,
        heading_alignment: float,
        progress_m: float,
        linear_speed_mps: float,
        stall_speed_epsilon_mps: float,
    ) -> float:
        weight = self._reward_weight("heading_alignment_weight", 0.0)
        if weight <= 0.0:
            return 0.0
        mode = str(self.reward_config.get("heading_reward_mode", "cosine_positive")).lower()
        if mode in {"angle_threshold_linear", "linear_angle_threshold", "drl_vo"}:
            threshold_rad = max(0.0, self._reward_weight("heading_angle_threshold_rad", math.pi / 6.0))
            return float(weight * (threshold_rad - abs(_wrap_angle(bearing_rad))))
        if linear_speed_mps <= stall_speed_epsilon_mps:
            return 0.0
        reward = weight * np.clip(heading_alignment, 0.0, 1.0)
        if mode in {"cosine_positive_progress", "progress_weighted_cosine_positive"}:
            max_progress_m = max(1e-6, float(self.linear_bounds_mps[1]) * self.dt_s)
            reward *= np.clip(progress_m / max_progress_m, 0.0, 1.0)
        return float(reward)

    def _near_goal_speed_penalty(self, distance_m: float, linear_speed_mps: float) -> float:
        weight = self._reward_weight("near_goal_speed_weight", 0.0)
        if weight <= 0.0:
            return 0.0
        slowdown_distance = max(
            self.goal_success_distance_m + 1e-6,
            self._reward_weight("near_goal_slowdown_distance_m", 1.2),
        )
        if distance_m >= slowdown_distance:
            return 0.0
        max_linear_mps = max(1e-6, float(self.linear_bounds_mps[1]))
        min_speed = float(np.clip(self._reward_weight("near_goal_min_speed_mps", 0.035), 0.0, max_linear_mps))
        ramp = float(np.clip((distance_m - self.goal_success_distance_m) / (slowdown_distance - self.goal_success_distance_m), 0.0, 1.0))
        target_speed = min_speed + ramp * (max_linear_mps - min_speed)
        close_factor = 1.0 - ramp
        overspeed = max(0.0, float(linear_speed_mps) - target_speed)
        return float(weight * close_factor * overspeed / max_linear_mps)

    def _near_goal_heading_alignment_reward(self, distance_m: float, heading_alignment: float) -> float:
        weight = self._reward_weight("near_goal_heading_alignment_weight", 0.0)
        if weight <= 0.0:
            return 0.0
        distance = max(
            self.goal_success_distance_m + 1e-6,
            self._reward_weight("near_goal_heading_alignment_distance_m", 1.0),
        )
        if distance_m >= distance:
            return 0.0
        close_factor = 1.0 - float(np.clip((distance_m - self.goal_success_distance_m) / (distance - self.goal_success_distance_m), 0.0, 1.0))
        return float(weight * close_factor * np.clip(heading_alignment, 0.0, 1.0))

    def _near_goal_low_speed_reward(self, distance_m: float, linear_speed_mps: float) -> float:
        weight = self._reward_weight("near_goal_low_speed_reward_weight", 0.0)
        if weight <= 0.0:
            return 0.0
        distance = max(
            self.goal_success_distance_m + 1e-6,
            self._reward_weight("near_goal_low_speed_distance_m", 0.5),
        )
        if distance_m >= distance:
            return 0.0
        speed_threshold = max(1e-6, self._reward_weight("near_goal_low_speed_threshold_mps", 0.1))
        if linear_speed_mps > speed_threshold:
            return 0.0
        stall_speed_epsilon_mps = self._reward_weight("stall_speed_epsilon_mps", 0.05)
        if linear_speed_mps <= stall_speed_epsilon_mps:
            return 0.0
        close_factor = 1.0 - float(np.clip((distance_m - self.goal_success_distance_m) / (distance - self.goal_success_distance_m), 0.0, 1.0))
        return float(weight * close_factor)

    def _initialize_path_guidance(self) -> None:
        self.path_guidance = None
        self.previous_geodesic_cost_m = None
        self.current_waypoint_xy = None
        self.previous_waypoint_heading_potential = None
        self.smoothed_waypoint_bearing_rad = None
        if self.reward_mode != "waypoint_geodesic" or not bool(self.path_guidance_config.get("enabled", True)):
            self.path_guidance_status = "disabled"
            return
        scene = self._require_scene()
        planner_safety_margin = max(
            0.0,
            float(self.path_guidance_config.get("planner_safety_margin_m", 0.05)),
        )
        config = PathGuidanceConfig(
            resolution_m=float(self.path_guidance_config.get("grid_resolution_m", 0.25)),
            clearance_margin_m=self.collision_clearance_margin_m + planner_safety_margin,
            waypoint_min_distance_m=float(self.path_guidance_config.get("waypoint_min_distance_m", 1.0)),
            waypoint_nominal_distance_m=float(self.path_guidance_config.get("waypoint_nominal_distance_m", 1.5)),
            waypoint_max_distance_m=float(self.path_guidance_config.get("waypoint_max_distance_m", 2.0)),
            dense_clearance_threshold_m=float(self.path_guidance_config.get("dense_clearance_threshold_m", 0.75)),
            sparse_clearance_threshold_m=float(self.path_guidance_config.get("sparse_clearance_threshold_m", 1.50)),
            finite_search_radius_cells=int(self.path_guidance_config.get("finite_search_radius_cells", 3)),
        )
        try:
            self.path_guidance = build_path_guidance(
                bounds_x_m=scene.bounds_x_m,
                bounds_y_m=scene.bounds_y_m,
                floor_regions=scene.floor_regions,
                static_obstacles=scene.static_obstacles,
                start_xy=scene.start_xy,
                goal_xy=scene.goal_xy,
                robot_radius_m=self.robot_radius_m,
                config=config,
            )
        except Exception:
            self.path_guidance_status = "planner_error"
            return
        if self.path_guidance is None:
            self.path_guidance_status = "unreachable"
            return
        self.previous_geodesic_cost_m = query_cost_to_go(self.path_guidance, self.pose[:2])
        if self.previous_geodesic_cost_m is None:
            self.path_guidance_status = "query_failed"
            return
        self.current_waypoint_xy = self._select_waypoint()
        if self.current_waypoint_xy is None:
            self.path_guidance_status = "waypoint_unavailable"
            return
        raw_bearing = self._bearing_to(self.current_waypoint_xy)
        self.smoothed_waypoint_bearing_rad = raw_bearing
        self.previous_waypoint_heading_potential = self._heading_potential(raw_bearing)
        self.path_guidance_status = "ok"

    def _waypoint_reward_terms(self) -> dict[str, Any]:
        empty = {
            "geodesic_progress_reward": 0.0,
            "waypoint_heading_reward": 0.0,
            "heading_potential_reward": 0.0,
            "geodesic_cost_to_go_m": None,
            "geodesic_progress_m": 0.0,
            "waypoint_xy": None,
            "waypoint_distance_m": None,
            "waypoint_bearing_rad": None,
        }
        if self.reward_mode != "waypoint_geodesic" or self.path_guidance is None:
            return empty

        new_cost = query_cost_to_go(self.path_guidance, self.pose[:2])
        old_cost = self.previous_geodesic_cost_m
        if new_cost is None or old_cost is None:
            self.path_guidance_status = "query_failed"
            self.previous_geodesic_cost_m = new_cost
            self.previous_waypoint_heading_potential = None
            return {**empty, "geodesic_cost_to_go_m": new_cost}

        raw_progress = float(old_cost - new_cost)
        soft_progress = self._soft_geodesic_progress(raw_progress)
        geodesic_reward = self._reward_weight("geodesic_progress_weight", 5.0) * soft_progress

        self.current_waypoint_xy = self._select_waypoint()
        waypoint_distance, raw_bearing = self._waypoint_metrics()
        waypoint_heading_reward = 0.0
        heading_potential_reward = 0.0
        if raw_bearing is not None:
            smoothed_bearing = self._smooth_waypoint_bearing(raw_bearing)
            expected_progress = max(
                1e-6,
                self._reward_weight(
                    "expected_step_progress_m",
                    float(self.linear_bounds_mps[1]) * self.dt_s,
                ),
            )
            progress_gate = float(np.clip(raw_progress / expected_progress, 0.0, 1.0))
            alignment = max(0.0, math.cos(smoothed_bearing))
            waypoint_heading_reward = (
                self._reward_weight("waypoint_heading_weight", 0.03)
                * progress_gate
                * alignment
            )
            potential = self._heading_potential(smoothed_bearing)
            if self.previous_waypoint_heading_potential is not None:
                potential_gamma = self._reward_weight("heading_potential_gamma", 0.9995)
                heading_potential_reward = self._reward_weight(
                    "waypoint_heading_potential_weight",
                    0.01,
                ) * (potential_gamma * potential - self.previous_waypoint_heading_potential)
            self.previous_waypoint_heading_potential = potential
            self.path_guidance_status = "ok"
        else:
            self.previous_waypoint_heading_potential = None
            self.path_guidance_status = "waypoint_unavailable"

        self.previous_geodesic_cost_m = new_cost
        return {
            "geodesic_progress_reward": float(geodesic_reward),
            "waypoint_heading_reward": float(waypoint_heading_reward),
            "heading_potential_reward": float(heading_potential_reward),
            "geodesic_cost_to_go_m": float(new_cost),
            "geodesic_progress_m": float(raw_progress),
            "waypoint_xy": self.current_waypoint_xy.astype(float).tolist() if self.current_waypoint_xy is not None else None,
            "waypoint_distance_m": waypoint_distance,
            "waypoint_bearing_rad": raw_bearing,
        }

    def _select_waypoint(self) -> np.ndarray | None:
        if self.path_guidance is None:
            return None
        scene = self._require_scene()
        waypoint = select_local_waypoint(
            self.path_guidance,
            point_xy=self.pose[:2],
            goal_xy=scene.goal_xy,
        )
        return np.asarray(waypoint, dtype=np.float64) if waypoint is not None else None

    def _soft_geodesic_progress(self, raw_progress_m: float) -> float:
        deadband = max(0.0, self._reward_weight("geodesic_progress_deadband_m", 0.002))
        if abs(float(raw_progress_m)) < deadband:
            return 0.0
        positive_clip = max(0.0, self._reward_weight("geodesic_progress_positive_clip_m", 0.03))
        negative_clip = max(0.0, self._reward_weight("geodesic_progress_negative_clip_m", 0.01))
        return float(np.clip(float(raw_progress_m), -negative_clip, positive_clip))

    def _waypoint_metrics(self) -> tuple[float | None, float | None]:
        if self.current_waypoint_xy is None:
            return None, None
        distance = float(np.linalg.norm(self.current_waypoint_xy - self.pose[:2]))
        return distance, self._bearing_to(self.current_waypoint_xy)

    def _bearing_to(self, point_xy: np.ndarray) -> float:
        angle = math.atan2(float(point_xy[1] - self.pose[1]), float(point_xy[0] - self.pose[0]))
        return _wrap_angle(angle - float(self.pose[2]))

    def _smooth_waypoint_bearing(self, raw_bearing_rad: float) -> float:
        alpha = float(np.clip(self._reward_weight("waypoint_heading_smoothing_alpha", 0.25), 0.0, 1.0))
        if self.smoothed_waypoint_bearing_rad is None:
            self.smoothed_waypoint_bearing_rad = float(raw_bearing_rad)
        else:
            delta = _wrap_angle(float(raw_bearing_rad) - self.smoothed_waypoint_bearing_rad)
            self.smoothed_waypoint_bearing_rad = _wrap_angle(self.smoothed_waypoint_bearing_rad + alpha * delta)
        return self.smoothed_waypoint_bearing_rad

    @staticmethod
    def _heading_potential(bearing_rad: float) -> float:
        return float(0.5 * (math.cos(float(bearing_rad)) + 1.0))

    def _goal_distance(self) -> float:
        scene = self._require_scene()
        return float(np.linalg.norm(np.asarray(scene.goal_xy, dtype=np.float32) - self.pose[:2]))

    def _goal_bearing(self) -> float:
        scene = self._require_scene()
        goal_angle = math.atan2(float(scene.goal_xy[1] - self.pose[1]), float(scene.goal_xy[0] - self.pose[0]))
        return _wrap_angle(goal_angle - float(self.pose[2]))

    def _require_scene(self) -> Scene2D:
        if self.scene is None:
            raise RuntimeError("Environment must be reset before use.")
        return self.scene

    def _uses_fixed_scenes(self, level: int) -> bool:
        level_config = dict(self.config.get("levels", {})).get(int(level), {})
        return bool(dict(level_config).get("fixed_scenes", ()) or dict(level_config).get("fixed_scene_set_path"))


def _wrap_angle(angle_rad: float) -> float:
    return float(math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad))))


def _scale_unit_interval(value: float, bounds: tuple[float, float]) -> float:
    low, high = float(bounds[0]), float(bounds[1])
    return low + 0.5 * (float(value) + 1.0) * (high - low)


def _normalize_symmetric(value: float, bounds: tuple[float, float]) -> float:
    low, high = float(bounds[0]), float(bounds[1])
    if high <= low:
        return 0.0
    normalized = 2.0 * (float(value) - low) / (high - low) - 1.0
    return float(np.clip(normalized, -1.0, 1.0))


def _resolve_config_path(path: str | Path, *, base_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    relative_to_config = Path(base_dir) / candidate
    if relative_to_config.exists():
        return relative_to_config
    return candidate


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _normalize_lidar_ranges(
    ranges: np.ndarray,
    *,
    min_range_m: float,
    max_range_m: float,
    mode: str,
) -> np.ndarray:
    clipped = np.clip(np.asarray(ranges, dtype=np.float32), float(min_range_m), float(max_range_m))
    if mode in {"legacy", "legacy_max", "legacy_max_symmetric"}:
        return np.clip(clipped / float(max_range_m), 0.0, 1.0) * 2.0 - 1.0

    denominator = max(1e-6, float(max_range_m) - float(min_range_m))
    unit = np.clip((clipped - float(min_range_m)) / denominator, 0.0, 1.0)
    if mode in {"minmax_unit", "unit", "unit01", "0_1"}:
        return unit.astype(np.float32, copy=False)
    if mode in {"minmax", "minmax_symmetric", "symmetric"}:
        return unit * 2.0 - 1.0
    raise ValueError(
        "Unsupported observation.lidar_normalization "
        f"{mode!r}; expected legacy_max_symmetric, minmax_symmetric, or minmax_unit."
    )
