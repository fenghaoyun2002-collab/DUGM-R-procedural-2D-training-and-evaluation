from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np

from dugm_r.dugm.scan_grid import GridEncoder


@dataclass(frozen=True)
class GtDynamicActorState:
    center_world: np.ndarray
    velocity_world: np.ndarray
    radius_m: float
    semantic_kind: str = "dynamic"


@dataclass(frozen=True)
class EstimatedDynamicTrackState:
    center_robot: np.ndarray
    velocity_robot: np.ndarray
    radius_m: float
    track_id: int
    velocity_std_mps: float
    state_xyvxvy_robot: np.ndarray | None = None
    covariance_xyvxvy_robot: np.ndarray | None = None
    current_measurement_available: bool = True
    in_maneuver_mode: bool = False
    miss_count: int = 0
    last_process_acceleration_std_mps2: float = 0.0

    def __post_init__(self) -> None:
        center = np.asarray(self.center_robot, dtype=np.float64).reshape(2)
        velocity = np.asarray(self.velocity_robot, dtype=np.float64).reshape(2)
        state = (
            np.concatenate((center, velocity)).astype(np.float64, copy=False)
            if self.state_xyvxvy_robot is None
            else np.asarray(self.state_xyvxvy_robot, dtype=np.float64)
        )
        covariance = (
            np.diag([0.0, 0.0, float(self.velocity_std_mps) ** 2, float(self.velocity_std_mps) ** 2]).astype(
                np.float64,
                copy=False,
            )
            if self.covariance_xyvxvy_robot is None
            else np.asarray(self.covariance_xyvxvy_robot, dtype=np.float64)
        )
        object.__setattr__(self, "center_robot", center)
        object.__setattr__(self, "velocity_robot", velocity)
        object.__setattr__(self, "state_xyvxvy_robot", state)
        object.__setattr__(self, "covariance_xyvxvy_robot", covariance)


class GtVelocityCostmapRasterizer:
    channels = ("gt_velocity_cost",)

    def __init__(
        self,
        grid_encoder: GridEncoder,
        *,
        prediction_horizon_s: float = 5.0,
        trajectory_dt_s: float = 0.25,
        core_inflation_radius_m: float = 0.25,
        uncertainty_radius_m: float = 0.50,
        stationary_speed_epsilon_mps: float = 0.03,
    ) -> None:
        self.grid_encoder = grid_encoder
        self.prediction_horizon_s = float(prediction_horizon_s)
        self.trajectory_dt_s = float(trajectory_dt_s)
        self.core_inflation_radius_m = float(core_inflation_radius_m)
        self.uncertainty_radius_m = float(uncertainty_radius_m)
        self.stationary_speed_epsilon_mps = float(stationary_speed_epsilon_mps)
        if self.prediction_horizon_s < 0.0:
            raise ValueError("prediction_horizon_s must be non-negative.")
        if self.trajectory_dt_s <= 0.0:
            raise ValueError("trajectory_dt_s must be positive.")
        if self.core_inflation_radius_m < 0.0:
            raise ValueError("core_inflation_radius_m must be non-negative.")
        if self.uncertainty_radius_m < 0.0:
            raise ValueError("uncertainty_radius_m must be non-negative.")
        if self.stationary_speed_epsilon_mps < 0.0:
            raise ValueError("stationary_speed_epsilon_mps must be non-negative.")
        self._grid_x_m, self._grid_y_m = self._grid_cell_centers()
        self.last_diagnostics: dict[str, Any] = {}

    @property
    def output_shape(self) -> tuple[int, int]:
        geometry = self.grid_encoder.geometry
        return geometry.height_cells, geometry.width_cells

    def rasterize(
        self,
        actor_states: Iterable[GtDynamicActorState],
        *,
        robot_pose_xyyaw: np.ndarray,
    ) -> np.ndarray:
        pose = np.asarray(robot_pose_xyyaw, dtype=np.float64).reshape(3)
        cost = np.zeros(self.output_shape, dtype=np.float32)
        actor_count = 0
        moving_actor_count = 0
        for actor_state in actor_states:
            actor_count += 1
            center_robot = world_point_to_robot(np.asarray(actor_state.center_world, dtype=np.float64), pose)
            velocity_robot = world_vector_to_robot(np.asarray(actor_state.velocity_world, dtype=np.float64), pose)
            speed_mps = float(np.linalg.norm(velocity_robot))
            if speed_mps > self.stationary_speed_epsilon_mps:
                moving_actor_count += 1
            core_radius_m = max(0.0, float(actor_state.radius_m)) + self.core_inflation_radius_m
            trajectory = self._trajectory_samples(center_robot, velocity_robot)
            cost = np.maximum(cost, self._cost_from_trajectory(trajectory, core_radius_m))
        cost = np.clip(cost, 0.0, 1.0).astype(np.float32, copy=False)
        self.assert_invariants(cost)
        self.last_diagnostics = {
            "gt_velocity_cost_actor_count": int(actor_count),
            "gt_velocity_cost_moving_actor_count": int(moving_actor_count),
            "gt_velocity_cost_nonzero_cell_count": int(np.sum(cost > 0.0)),
            "gt_velocity_cost_core_cell_count": int(np.sum(cost >= 1.0)),
            "gt_velocity_cost_max": float(np.max(cost)) if cost.size else 0.0,
            "gt_velocity_cost_sum": float(np.sum(cost)),
        }
        return cost

    def _trajectory_samples(self, center_robot: np.ndarray, velocity_robot: np.ndarray) -> np.ndarray:
        center = np.asarray(center_robot, dtype=np.float64).reshape(2)
        velocity = np.asarray(velocity_robot, dtype=np.float64).reshape(2)
        if float(np.linalg.norm(velocity)) <= self.stationary_speed_epsilon_mps or self.prediction_horizon_s <= 0.0:
            return center.reshape(1, 2)
        steps = max(1, int(math.ceil(self.prediction_horizon_s / self.trajectory_dt_s)))
        times = np.linspace(0.0, self.prediction_horizon_s, steps + 1, dtype=np.float64)
        return center.reshape(1, 2) + times.reshape(-1, 1) * velocity.reshape(1, 2)

    def _cost_from_trajectory(self, trajectory_xy: np.ndarray, core_radius_m: float) -> np.ndarray:
        trajectory = np.asarray(trajectory_xy, dtype=np.float64).reshape(-1, 2)
        if trajectory.size == 0:
            return np.zeros(self.output_shape, dtype=np.float32)
        dx = self._grid_x_m.reshape(1, *self.output_shape) - trajectory[:, 0].reshape(-1, 1, 1)
        dy = self._grid_y_m.reshape(1, *self.output_shape) - trajectory[:, 1].reshape(-1, 1, 1)
        distance_to_path = np.sqrt(np.min(dx * dx + dy * dy, axis=0))
        signed_distance = distance_to_path - max(0.0, float(core_radius_m))
        if self.uncertainty_radius_m <= 0.0:
            return (signed_distance <= 0.0).astype(np.float32)
        cost = np.where(
            signed_distance <= 0.0,
            1.0,
            np.maximum(0.0, 1.0 - signed_distance / self.uncertainty_radius_m),
        )
        return np.asarray(cost, dtype=np.float32)

    def _grid_cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        geometry = self.grid_encoder.geometry
        rows = np.arange(geometry.height_cells, dtype=np.float64)
        cols = np.arange(geometry.width_cells, dtype=np.float64)
        x_indices = geometry.height_cells - 1 - rows
        y_indices = geometry.width_cells - 1 - cols
        x_m = -0.5 * geometry.height_m + (x_indices + 0.5) * geometry.resolution_m
        y_m = -0.5 * geometry.width_m + (y_indices + 0.5) * geometry.resolution_m
        return np.meshgrid(x_m, y_m, indexing="ij")

    @staticmethod
    def assert_invariants(costmap: np.ndarray) -> None:
        cost = np.asarray(costmap, dtype=np.float32)
        if cost.ndim != 2 or cost.dtype != np.float32:
            raise AssertionError("GT velocity cost map must be a 2D float32 grid.")
        if not np.all(np.isfinite(cost)):
            raise AssertionError("GT velocity cost map contains NaN or Inf.")
        if np.min(cost) < 0.0 or np.max(cost) > 1.0:
            raise AssertionError("GT velocity cost map must stay in [0, 1].")


def world_vector_to_robot(vector_world: np.ndarray, robot_pose_xyyaw: np.ndarray) -> np.ndarray:
    yaw = float(robot_pose_xyyaw[2])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float64) @ np.asarray(vector_world, dtype=np.float64)


def world_point_to_robot(point_world: np.ndarray, robot_pose_xyyaw: np.ndarray) -> np.ndarray:
    return world_vector_to_robot(np.asarray(point_world, dtype=np.float64) - robot_pose_xyyaw[:2], robot_pose_xyyaw)


class EstimatedVelocityCostmapRasterizer(GtVelocityCostmapRasterizer):
    channels = ("estimated_velocity_cost",)
    dynamic_cost_mode = "deterministic"

    def rasterize(self, track_states: Iterable[EstimatedDynamicTrackState]) -> np.ndarray:
        cost = np.zeros(self.output_shape, dtype=np.float32)
        track_count = 0
        moving_track_count = 0
        max_velocity_std_mps = 0.0
        for track_state in track_states:
            track_count += 1
            center_robot = np.asarray(track_state.center_robot, dtype=np.float64).reshape(2)
            velocity_robot = np.asarray(track_state.velocity_robot, dtype=np.float64).reshape(2)
            speed_mps = float(np.linalg.norm(velocity_robot))
            if speed_mps > self.stationary_speed_epsilon_mps:
                moving_track_count += 1
            max_velocity_std_mps = max(max_velocity_std_mps, float(track_state.velocity_std_mps))
            core_radius_m = max(0.0, float(track_state.radius_m)) + self.core_inflation_radius_m
            trajectory = self._trajectory_samples(center_robot, velocity_robot)
            cost = np.maximum(cost, self._cost_from_trajectory(trajectory, core_radius_m))
        cost = np.clip(cost, 0.0, 1.0).astype(np.float32, copy=False)
        self.assert_invariants(cost)
        self.last_diagnostics = {
            "estimated_velocity_cost_mode": self.dynamic_cost_mode,
            "estimated_velocity_cost_track_count": int(track_count),
            "estimated_velocity_cost_moving_track_count": int(moving_track_count),
            "estimated_velocity_cost_nonzero_cell_count": int(np.sum(cost > 0.0)),
            "estimated_velocity_cost_core_cell_count": int(np.sum(cost >= 1.0)),
            "estimated_velocity_cost_max": float(np.max(cost)) if cost.size else 0.0,
            "estimated_velocity_cost_sum": float(np.sum(cost)),
            "estimated_velocity_cost_max_velocity_std_mps": float(max_velocity_std_mps),
            "estimated_velocity_cost_prediction_horizon_s": float(self.prediction_horizon_s),
            "estimated_velocity_cost_prediction_dt_s": float(self.trajectory_dt_s),
            "estimated_dynamic_trajectory_cost_nonzero_cell_count": int(np.sum(cost > 0.0)),
            "estimated_dynamic_trajectory_cost_core_cell_count": int(np.sum(cost >= 1.0)),
        }
        return cost


class _MeanVelocityRolloutBase(EstimatedVelocityCostmapRasterizer):
    """Shared mean-velocity prediction and diagnostics for no-diffusion rollout."""

    def __init__(
        self,
        grid_encoder: GridEncoder,
        *,
        prediction_horizon_s: float = 4.0,
        prediction_dt_s: float = 0.2,
        human_speed_max_mps: float | None = None,
        clip_prediction_speed_to_max: bool = True,
        stationary_speed_epsilon_mps: float = 0.03,
    ) -> None:
        super().__init__(
            grid_encoder,
            prediction_horizon_s=prediction_horizon_s,
            trajectory_dt_s=prediction_dt_s,
            core_inflation_radius_m=0.0,
            uncertainty_radius_m=0.0,
            stationary_speed_epsilon_mps=stationary_speed_epsilon_mps,
        )
        self.prediction_dt_s = float(prediction_dt_s)
        self.human_speed_max_mps = None if human_speed_max_mps is None else float(human_speed_max_mps)
        self.clip_prediction_speed_to_max = bool(clip_prediction_speed_to_max)
        if self.prediction_dt_s <= 0.0:
            raise ValueError("prediction_dt_s must be positive.")
        if self.human_speed_max_mps is not None and self.human_speed_max_mps <= 0.0:
            raise ValueError("human_speed_max_mps must be positive when configured.")
        self.prediction_times_s = _prediction_times(
            prediction_horizon_s=self.prediction_horizon_s,
            prediction_dt_s=self.prediction_dt_s,
        )

    def _diagnostics(
        self,
        *,
        cost: np.ndarray,
        track_count: int,
        moving_track_count: int,
        skipped_invalid_state_count: int,
        prediction_speed_clipped_count: int,
        max_velocity_std_mps: float,
    ) -> dict[str, Any]:
        return {
            "estimated_velocity_cost_mode": self.dynamic_cost_mode,
            "estimated_velocity_cost_track_count": int(track_count),
            "estimated_velocity_cost_moving_track_count": int(moving_track_count),
            "estimated_velocity_cost_skipped_invalid_state_count": int(skipped_invalid_state_count),
            "estimated_velocity_cost_prediction_speed_clipped_count": int(prediction_speed_clipped_count),
            "estimated_velocity_cost_nonzero_cell_count": int(np.sum(cost > 0.0)),
            "estimated_velocity_cost_core_cell_count": int(np.sum(cost >= 1.0)),
            "estimated_velocity_cost_max": float(np.max(cost)) if cost.size else 0.0,
            "estimated_velocity_cost_sum": float(np.sum(cost)),
            "estimated_velocity_cost_max_velocity_std_mps": float(max_velocity_std_mps),
            "estimated_velocity_cost_prediction_horizon_s": float(self.prediction_horizon_s),
            "estimated_velocity_cost_prediction_dt_s": float(self.prediction_dt_s),
            "estimated_velocity_cost_uses_track_covariance": False,
            "estimated_velocity_cost_uses_process_noise": False,
            "estimated_velocity_cost_uses_track_radius": False,
            "estimated_dynamic_trajectory_cost_nonzero_cell_count": int(np.sum(cost > 0.0)),
            "estimated_dynamic_trajectory_cost_core_cell_count": int(np.sum(cost >= 1.0)),
        }

    def _prediction_mean(self, center_robot: np.ndarray, velocity_robot: np.ndarray, tau: float) -> np.ndarray:
        center = np.asarray(center_robot, dtype=np.float64).reshape(2)
        return center + float(tau) * self._bounded_prediction_velocity(velocity_robot)

    def _bounded_prediction_velocity(self, velocity_robot: np.ndarray) -> np.ndarray:
        velocity = np.asarray(velocity_robot, dtype=np.float64).reshape(2)
        speed = float(np.linalg.norm(velocity))
        if (
            self.clip_prediction_speed_to_max
            and self.human_speed_max_mps is not None
            and speed > self.human_speed_max_mps
        ):
            return velocity * (self.human_speed_max_mps / max(speed, 1.0e-12))
        return velocity

    def _prediction_speed_was_clipped(self, velocity_robot: np.ndarray) -> bool:
        return bool(
            self.clip_prediction_speed_to_max
            and self.human_speed_max_mps is not None
            and float(np.linalg.norm(np.asarray(velocity_robot, dtype=np.float64).reshape(2)))
            > self.human_speed_max_mps
        )

def _prediction_times(*, prediction_horizon_s: float, prediction_dt_s: float) -> tuple[float, ...]:
    if prediction_horizon_s <= 0.0:
        return (0.0,)
    times: list[float] = [0.0]
    tau = float(prediction_dt_s)
    while tau < prediction_horizon_s - 1.0e-12:
        times.append(float(tau))
        tau += float(prediction_dt_s)
    if not math.isclose(times[-1], prediction_horizon_s, abs_tol=1.0e-12):
        times.append(float(prediction_horizon_s))
    return tuple(times)


class LinearVelocityRolloutCostmapRasterizer(_MeanVelocityRolloutBase):
    """Rasterize only the constant-velocity mean centerline as binary grid cells."""

    dynamic_cost_mode = "linear_rollout"

    def __init__(
        self,
        grid_encoder: GridEncoder,
        *,
        prediction_horizon_s: float = 4.0,
        prediction_dt_s: float = 0.2,
        human_speed_max_mps: float | None = None,
        clip_prediction_speed_to_max: bool = True,
        stationary_speed_epsilon_mps: float = 0.03,
    ) -> None:
        super().__init__(
            grid_encoder,
            prediction_horizon_s=prediction_horizon_s,
            prediction_dt_s=prediction_dt_s,
            human_speed_max_mps=human_speed_max_mps,
            clip_prediction_speed_to_max=clip_prediction_speed_to_max,
            stationary_speed_epsilon_mps=stationary_speed_epsilon_mps,
        )

    def rasterize(self, track_states: Iterable[EstimatedDynamicTrackState]) -> np.ndarray:
        cost = np.zeros(self.output_shape, dtype=np.float32)
        track_count = 0
        moving_track_count = 0
        skipped_invalid_state_count = 0
        prediction_speed_clipped_count = 0
        max_velocity_std_mps = 0.0

        for track_state in track_states:
            center = np.asarray(track_state.center_robot, dtype=np.float64).reshape(-1)
            velocity = np.asarray(track_state.velocity_robot, dtype=np.float64).reshape(-1)
            if center.size != 2 or velocity.size != 2 or not np.all(np.isfinite(np.concatenate((center, velocity)))):
                skipped_invalid_state_count += 1
                continue
            track_count += 1
            speed_mps = float(np.linalg.norm(velocity))
            if speed_mps > self.stationary_speed_epsilon_mps:
                moving_track_count += 1
            max_velocity_std_mps = max(max_velocity_std_mps, float(track_state.velocity_std_mps))
            if self._prediction_speed_was_clipped(velocity):
                prediction_speed_clipped_count += 1
            self._rasterize_centerline(cost, center, self._prediction_mean(center, velocity, self.prediction_horizon_s))

        self.assert_invariants(cost)
        self.last_diagnostics = self._diagnostics(
            cost=cost,
            track_count=track_count,
            moving_track_count=moving_track_count,
            skipped_invalid_state_count=skipped_invalid_state_count,
            prediction_speed_clipped_count=prediction_speed_clipped_count,
            max_velocity_std_mps=max_velocity_std_mps,
        )
        self.last_diagnostics["estimated_velocity_cost_rasterization"] = "binary_centerline_cells"
        return cost

    def _rasterize_centerline(self, output: np.ndarray, start_xy: np.ndarray, end_xy: np.ndarray) -> None:
        start = np.asarray(start_xy, dtype=np.float64).reshape(2)
        end = np.asarray(end_xy, dtype=np.float64).reshape(2)
        distance = float(np.linalg.norm(end - start))
        resolution = self.grid_encoder.geometry.resolution_m
        sample_count = max(1, int(math.ceil(distance / max(0.5 * resolution, 1.0e-12))))
        fractions = np.linspace(0.0, 1.0, sample_count + 1, dtype=np.float64)
        samples = start.reshape(1, 2) + fractions.reshape(-1, 1) * (end - start).reshape(1, 2)
        for point in samples:
            cell = self.grid_encoder.metric_to_grid(float(point[0]), float(point[1]))
            if cell is not None:
                output[cell] = 1.0


class UncertaintyAwareEstimatedVelocityCostmapRasterizer(EstimatedVelocityCostmapRasterizer):
    channels = ("estimated_velocity_cost",)
    dynamic_cost_mode = "uncertainty_gaussian"

    def __init__(
        self,
        grid_encoder: GridEncoder,
        *,
        prediction_horizon_s: float = 4.0,
        prediction_dt_s: float = 0.2,
        sigma_shape_m: float = 0.30,
        covariance_epsilon: float = 1.0e-4,
        mahalanobis_cutoff_sq: float = 16.0,
        covariance_eigenvalue_min_m2: float = 1.0e-4,
        covariance_eigenvalue_max_m2: float = 9.0,
        process_noise_source: str = "track_last_process_acceleration_std_mps2",
        fallback_process_acceleration_std_mps2: float = 0.0,
        uncertainty_frame: str = "robot_xy",
        human_speed_max_mps: float | None = None,
        clip_prediction_speed_to_max: bool = True,
        longitudinal_process_acceleration_scale: float = 1.0,
        lateral_process_acceleration_scale: float = 1.0,
        longitudinal_velocity_covariance_scale: float = 1.0,
        lateral_velocity_covariance_scale: float = 1.0,
        covariance_eigenvalue_max_longitudinal_m2: float | None = None,
        covariance_eigenvalue_max_lateral_m2: float | None = None,
        low_speed_isotropic_threshold_mps: float = 0.03,
        temporal_decay: bool = False,
        combine_obstacles: str = "max",
        stationary_speed_epsilon_mps: float = 0.03,
    ) -> None:
        super().__init__(
            grid_encoder,
            prediction_horizon_s=prediction_horizon_s,
            trajectory_dt_s=prediction_dt_s,
            core_inflation_radius_m=0.0,
            uncertainty_radius_m=0.0,
            stationary_speed_epsilon_mps=stationary_speed_epsilon_mps,
        )
        self.prediction_dt_s = float(prediction_dt_s)
        self.sigma_shape_m = float(sigma_shape_m)
        self.covariance_epsilon = float(covariance_epsilon)
        self.mahalanobis_cutoff_sq = float(mahalanobis_cutoff_sq)
        self.covariance_eigenvalue_min_m2 = float(covariance_eigenvalue_min_m2)
        self.covariance_eigenvalue_max_m2 = float(covariance_eigenvalue_max_m2)
        self.process_noise_source = str(process_noise_source)
        self.fallback_process_acceleration_std_mps2 = float(fallback_process_acceleration_std_mps2)
        self.uncertainty_frame = str(uncertainty_frame).lower()
        self.human_speed_max_mps = None if human_speed_max_mps is None else float(human_speed_max_mps)
        self.clip_prediction_speed_to_max = bool(clip_prediction_speed_to_max)
        self.longitudinal_process_acceleration_scale = float(longitudinal_process_acceleration_scale)
        self.lateral_process_acceleration_scale = float(lateral_process_acceleration_scale)
        self.longitudinal_velocity_covariance_scale = float(longitudinal_velocity_covariance_scale)
        self.lateral_velocity_covariance_scale = float(lateral_velocity_covariance_scale)
        self.covariance_eigenvalue_max_longitudinal_m2 = (
            self.covariance_eigenvalue_max_m2
            if covariance_eigenvalue_max_longitudinal_m2 is None
            else float(covariance_eigenvalue_max_longitudinal_m2)
        )
        self.covariance_eigenvalue_max_lateral_m2 = (
            self.covariance_eigenvalue_max_m2
            if covariance_eigenvalue_max_lateral_m2 is None
            else float(covariance_eigenvalue_max_lateral_m2)
        )
        self.low_speed_isotropic_threshold_mps = float(low_speed_isotropic_threshold_mps)
        self.temporal_decay = bool(temporal_decay)
        self.combine_obstacles = str(combine_obstacles).lower()
        if self.prediction_dt_s <= 0.0:
            raise ValueError("prediction_dt_s must be positive.")
        if self.sigma_shape_m < 0.0:
            raise ValueError("sigma_shape_m must be non-negative.")
        if self.covariance_epsilon < 0.0:
            raise ValueError("covariance_epsilon must be non-negative.")
        if self.mahalanobis_cutoff_sq <= 0.0:
            raise ValueError("mahalanobis_cutoff_sq must be positive.")
        if self.covariance_eigenvalue_min_m2 <= 0.0:
            raise ValueError("covariance_eigenvalue_min_m2 must be positive.")
        if self.covariance_eigenvalue_max_m2 < self.covariance_eigenvalue_min_m2:
            raise ValueError("covariance_eigenvalue_max_m2 must be >= covariance_eigenvalue_min_m2.")
        if self.fallback_process_acceleration_std_mps2 < 0.0:
            raise ValueError("fallback_process_acceleration_std_mps2 must be non-negative.")
        if self.process_noise_source not in {"track_last_process_acceleration_std_mps2"}:
            raise ValueError("process_noise_source must be 'track_last_process_acceleration_std_mps2'.")
        if self.uncertainty_frame not in {"robot_xy", "velocity_aligned_bounded_speed"}:
            raise ValueError("uncertainty_frame must be 'robot_xy' or 'velocity_aligned_bounded_speed'.")
        if self.human_speed_max_mps is not None and self.human_speed_max_mps <= 0.0:
            raise ValueError("human_speed_max_mps must be positive when configured.")
        for name, value in {
            "longitudinal_process_acceleration_scale": self.longitudinal_process_acceleration_scale,
            "lateral_process_acceleration_scale": self.lateral_process_acceleration_scale,
            "longitudinal_velocity_covariance_scale": self.longitudinal_velocity_covariance_scale,
            "lateral_velocity_covariance_scale": self.lateral_velocity_covariance_scale,
            "low_speed_isotropic_threshold_mps": self.low_speed_isotropic_threshold_mps,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if self.covariance_eigenvalue_max_longitudinal_m2 < self.covariance_eigenvalue_min_m2:
            raise ValueError("covariance_eigenvalue_max_longitudinal_m2 must be >= covariance_eigenvalue_min_m2.")
        if self.covariance_eigenvalue_max_lateral_m2 < self.covariance_eigenvalue_min_m2:
            raise ValueError("covariance_eigenvalue_max_lateral_m2 must be >= covariance_eigenvalue_min_m2.")
        if self.temporal_decay:
            raise ValueError("uncertainty_gaussian currently requires temporal_decay: false.")
        if self.combine_obstacles != "max":
            raise ValueError("uncertainty_gaussian currently requires combine_obstacles: max.")
        self.prediction_times_s = self._prediction_times()

    def rasterize(self, track_states: Iterable[EstimatedDynamicTrackState]) -> np.ndarray:
        cost = np.zeros(self.output_shape, dtype=np.float32)
        track_count = 0
        moving_track_count = 0
        skipped_invalid_state_count = 0
        invalid_covariance_count = 0
        skipped_invalid_covariance_count = 0
        fallback_covariance_count = 0
        prediction_speed_clipped_count = 0
        max_velocity_std_mps = 0.0
        max_position_std_m = 0.0
        max_effective_cov_eigenvalue_m2 = 0.0

        for track_state in track_states:
            state = np.asarray(track_state.state_xyvxvy_robot, dtype=np.float64).reshape(-1)
            if state.size < 4 or not np.all(np.isfinite(state[:4])):
                skipped_invalid_state_count += 1
                continue
            covariance, covariance_was_invalid, fallback_used = self._track_covariance_4x4(track_state)
            if covariance_was_invalid:
                invalid_covariance_count += 1
            if fallback_used:
                fallback_covariance_count += 1
            if covariance is None:
                skipped_invalid_covariance_count += 1
                continue

            track_count += 1
            velocity_robot = state[2:4]
            speed_mps = float(np.linalg.norm(velocity_robot))
            if speed_mps > self.stationary_speed_epsilon_mps:
                moving_track_count += 1
            max_velocity_std_mps = max(max_velocity_std_mps, float(track_state.velocity_std_mps))

            sigma_a = self._track_process_acceleration_std(track_state)
            if self._prediction_speed_was_clipped(velocity_robot):
                prediction_speed_clipped_count += 1
            velocity_aligned_context = None
            if (
                self.uncertainty_frame == "velocity_aligned_bounded_speed"
                and speed_mps > self.low_speed_isotropic_threshold_mps
            ):
                velocity_aligned_context = self._velocity_aligned_covariance_context(
                    covariance,
                    velocity_robot,
                )
            track_cost = np.zeros(self.output_shape, dtype=np.float32)
            for tau in self.prediction_times_s:
                mu = self._prediction_mean(state[:2], velocity_robot, float(tau))
                if velocity_aligned_context is not None:
                    sigma_p, sigma_p_eigenvalues, sigma_eff, effective_eigenvalues = (
                        self._velocity_aligned_covariance_pair(
                            velocity_aligned_context,
                            float(tau),
                            sigma_a,
                        )
                    )
                else:
                    sigma_p = self._predicted_position_covariance_for_track(
                        covariance,
                        float(tau),
                        sigma_a,
                        velocity_robot,
                    )
                    sigma_p_regularized = self._regularize_covariance_2d(
                        sigma_p + self.covariance_epsilon * np.eye(2, dtype=np.float64)
                    )
                    if sigma_p_regularized is None:
                        continue
                    sigma_p, sigma_p_eigenvalues = sigma_p_regularized

                    sigma_eff = self._effective_position_covariance_for_track(
                        covariance,
                        float(tau),
                        sigma_a,
                        velocity_robot,
                    )
                    regularized = self._regularize_covariance_2d(sigma_eff)
                    if regularized is None:
                        continue
                    sigma_eff, effective_eigenvalues = regularized
                max_position_std_m = max(max_position_std_m, float(np.sqrt(np.max(sigma_p_eigenvalues))))
                max_effective_cov_eigenvalue_m2 = max(
                    max_effective_cov_eigenvalue_m2,
                    float(np.max(effective_eigenvalues)),
                )
                self._apply_gaussian_cost(track_cost, mu, sigma_eff)

            current_cell = self.grid_encoder.metric_to_grid(float(state[0]), float(state[1]))
            if current_cell is not None:
                track_cost[current_cell] = 1.0
            cost = np.maximum(cost, track_cost)

        cost = np.clip(cost, 0.0, 1.0).astype(np.float32, copy=False)
        self.assert_invariants(cost)
        self.last_diagnostics = {
            "estimated_velocity_cost_mode": self.dynamic_cost_mode,
            "estimated_velocity_cost_track_count": int(track_count),
            "estimated_velocity_cost_moving_track_count": int(moving_track_count),
            "estimated_velocity_cost_skipped_invalid_state_count": int(skipped_invalid_state_count),
            "estimated_velocity_cost_invalid_covariance_count": int(invalid_covariance_count),
            "estimated_velocity_cost_skipped_invalid_covariance_count": int(skipped_invalid_covariance_count),
            "estimated_velocity_cost_fallback_covariance_count": int(fallback_covariance_count),
            "estimated_velocity_cost_prediction_speed_clipped_count": int(prediction_speed_clipped_count),
            "estimated_velocity_cost_nonzero_cell_count": int(np.sum(cost > 0.0)),
            "estimated_velocity_cost_core_cell_count": int(np.sum(cost >= 1.0)),
            "estimated_velocity_cost_max": float(np.max(cost)) if cost.size else 0.0,
            "estimated_velocity_cost_sum": float(np.sum(cost)),
            "estimated_velocity_cost_max_velocity_std_mps": float(max_velocity_std_mps),
            "estimated_velocity_cost_max_position_std_m": float(max_position_std_m),
            "estimated_velocity_cost_max_effective_cov_eigenvalue_m2": float(max_effective_cov_eigenvalue_m2),
            "estimated_velocity_cost_uncertainty_frame": self.uncertainty_frame,
            "estimated_velocity_cost_prediction_horizon_s": float(self.prediction_horizon_s),
            "estimated_velocity_cost_prediction_dt_s": float(self.prediction_dt_s),
            "estimated_dynamic_trajectory_cost_nonzero_cell_count": int(np.sum(cost > 0.0)),
            "estimated_dynamic_trajectory_cost_core_cell_count": int(np.sum(cost >= 1.0)),
        }
        return cost

    def _prediction_times(self) -> tuple[float, ...]:
        if self.prediction_horizon_s <= 0.0:
            return (0.0,)
        times: list[float] = [0.0]
        tau = self.prediction_dt_s
        while tau < self.prediction_horizon_s - 1.0e-12:
            times.append(float(tau))
            tau += self.prediction_dt_s
        if not math.isclose(times[-1], self.prediction_horizon_s, abs_tol=1.0e-12):
            times.append(float(self.prediction_horizon_s))
        return tuple(times)

    def _track_covariance_4x4(
        self,
        track_state: EstimatedDynamicTrackState,
    ) -> tuple[np.ndarray | None, bool, bool]:
        raw_covariance = np.asarray(track_state.covariance_xyvxvy_robot, dtype=np.float64)
        if raw_covariance.shape == (4, 4) and np.all(np.isfinite(raw_covariance)):
            return 0.5 * (raw_covariance + raw_covariance.T), False, False
        fallback = self._fallback_covariance_4x4(track_state)
        return fallback, True, fallback is not None

    def _fallback_covariance_4x4(self, track_state: EstimatedDynamicTrackState) -> np.ndarray | None:
        velocity_std = float(track_state.velocity_std_mps)
        if not np.isfinite(velocity_std) or velocity_std < 0.0:
            velocity_std = 0.30
        position_std = max(
            self.grid_encoder.geometry.resolution_m,
            self.sigma_shape_m,
            float(track_state.radius_m) if np.isfinite(float(track_state.radius_m)) else 0.0,
        )
        if not np.isfinite(position_std) or position_std <= 0.0:
            return None
        return np.diag(
            [
                position_std**2,
                position_std**2,
                max(velocity_std, 1.0e-3) ** 2,
                max(velocity_std, 1.0e-3) ** 2,
            ]
        ).astype(np.float64, copy=False)

    def _track_process_acceleration_std(self, track_state: EstimatedDynamicTrackState) -> float:
        value = float(track_state.last_process_acceleration_std_mps2)
        if np.isfinite(value) and value >= 0.0:
            return value
        return float(self.fallback_process_acceleration_std_mps2)

    def _prediction_mean(self, position_robot: np.ndarray, velocity_robot: np.ndarray, tau: float) -> np.ndarray:
        position = np.asarray(position_robot, dtype=np.float64).reshape(2)
        velocity = np.asarray(velocity_robot, dtype=np.float64).reshape(2)
        speed = float(np.linalg.norm(velocity))
        if (
            self.uncertainty_frame == "velocity_aligned_bounded_speed"
            and self.clip_prediction_speed_to_max
            and self.human_speed_max_mps is not None
            and speed > self.human_speed_max_mps
        ):
            velocity = velocity * (self.human_speed_max_mps / max(speed, 1.0e-12))
        return position + float(tau) * velocity

    def _prediction_speed_was_clipped(self, velocity_robot: np.ndarray) -> bool:
        if (
            self.uncertainty_frame != "velocity_aligned_bounded_speed"
            or not self.clip_prediction_speed_to_max
            or self.human_speed_max_mps is None
        ):
            return False
        return float(np.linalg.norm(np.asarray(velocity_robot, dtype=np.float64).reshape(2))) > self.human_speed_max_mps

    def _predicted_position_covariance_for_track(
        self,
        covariance_xyvxvy: np.ndarray,
        tau: float,
        sigma_a: float,
        velocity_robot: np.ndarray,
    ) -> np.ndarray:
        if self.uncertainty_frame != "velocity_aligned_bounded_speed":
            return self._predicted_position_covariance(covariance_xyvxvy, tau, sigma_a)
        return self._velocity_aligned_position_covariance(
            covariance_xyvxvy,
            tau,
            sigma_a,
            velocity_robot,
            include_shape=False,
        )

    def _effective_position_covariance_for_track(
        self,
        covariance_xyvxvy: np.ndarray,
        tau: float,
        sigma_a: float,
        velocity_robot: np.ndarray,
    ) -> np.ndarray:
        if self.uncertainty_frame != "velocity_aligned_bounded_speed":
            return self._predicted_position_covariance(covariance_xyvxvy, tau, sigma_a) + (
                self.sigma_shape_m**2 + self.covariance_epsilon
            ) * np.eye(2, dtype=np.float64)
        return self._velocity_aligned_position_covariance(
            covariance_xyvxvy,
            tau,
            sigma_a,
            velocity_robot,
            include_shape=True,
        )

    @staticmethod
    def _predicted_position_covariance(covariance_xyvxvy: np.ndarray, tau: float, sigma_a: float) -> np.ndarray:
        transition = np.asarray(
            [
                [1.0, 0.0, tau, 0.0],
                [0.0, 1.0, 0.0, tau],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        acceleration_map = np.asarray(
            [
                [0.5 * tau * tau, 0.0],
                [0.0, 0.5 * tau * tau],
                [tau, 0.0],
                [0.0, tau],
            ],
            dtype=np.float64,
        )
        process_noise = float(sigma_a) ** 2 * (acceleration_map @ acceleration_map.T)
        predicted = transition @ covariance_xyvxvy @ transition.T + process_noise
        predicted = 0.5 * (predicted + predicted.T)
        return predicted[:2, :2]

    def _velocity_aligned_position_covariance(
        self,
        covariance_xyvxvy: np.ndarray,
        tau: float,
        sigma_a: float,
        velocity_robot: np.ndarray,
        *,
        include_shape: bool,
    ) -> np.ndarray:
        speed = float(np.linalg.norm(np.asarray(velocity_robot, dtype=np.float64).reshape(2)))
        if speed <= self.low_speed_isotropic_threshold_mps:
            covariance = self._predicted_position_covariance(
                covariance_xyvxvy,
                tau,
                float(sigma_a) * self.lateral_process_acceleration_scale,
            )
            if include_shape:
                covariance = covariance + (self.sigma_shape_m**2 + self.covariance_epsilon) * np.eye(
                    2,
                    dtype=np.float64,
                )
            return self._cap_isotropic_speed_bounded_covariance(covariance, tau)

        basis = self._velocity_frame_basis(velocity_robot)
        to_local = basis.T
        transform = np.zeros((4, 4), dtype=np.float64)
        transform[:2, :2] = to_local
        transform[2:, 2:] = to_local
        local_covariance = transform @ np.asarray(covariance_xyvxvy, dtype=np.float64).reshape(4, 4) @ transform.T
        velocity_scale = np.diag(
            [
                1.0,
                1.0,
                self.longitudinal_velocity_covariance_scale,
                self.lateral_velocity_covariance_scale,
            ]
        )
        local_covariance = velocity_scale @ local_covariance @ velocity_scale
        transition = np.asarray(
            [
                [1.0, 0.0, tau, 0.0],
                [0.0, 1.0, 0.0, tau],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        predicted_local = transition @ local_covariance @ transition.T
        position_covariance = 0.5 * (predicted_local[:2, :2] + predicted_local[:2, :2].T)
        tau_sq = float(tau) * float(tau)
        tau_fourth = tau_sq * tau_sq
        position_covariance[0, 0] += (
            0.25
            * tau_fourth
            * (float(sigma_a) * self.longitudinal_process_acceleration_scale) ** 2
        )
        position_covariance[1, 1] += (
            0.25
            * tau_fourth
            * (float(sigma_a) * self.lateral_process_acceleration_scale) ** 2
        )
        shape_variance = self.sigma_shape_m**2 + self.covariance_epsilon if include_shape else 0.0
        var_longitudinal = float(position_covariance[0, 0]) + shape_variance
        var_lateral = float(position_covariance[1, 1]) + shape_variance
        current_var_longitudinal = max(float(local_covariance[0, 0]) + shape_variance, self.covariance_eigenvalue_min_m2)
        current_var_lateral = max(float(local_covariance[1, 1]) + shape_variance, self.covariance_eigenvalue_min_m2)
        var_longitudinal = self._clamp_axis_variance(
            var_longitudinal,
            axis_max_m2=self._longitudinal_variance_cap(speed, tau, current_var_longitudinal),
        )
        var_lateral = self._clamp_axis_variance(
            var_lateral,
            axis_max_m2=self._lateral_variance_cap(speed, tau, current_var_lateral),
        )
        local = np.diag([var_longitudinal, var_lateral]).astype(np.float64, copy=False)
        covariance = basis @ local @ basis.T
        return 0.5 * (covariance + covariance.T)

    def _velocity_aligned_covariance_context(
        self,
        covariance_xyvxvy: np.ndarray,
        velocity_robot: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Cache track-constant transforms used by every prediction time."""

        velocity = np.asarray(velocity_robot, dtype=np.float64).reshape(2)
        speed = float(np.linalg.norm(velocity))
        basis = self._velocity_frame_basis(velocity)
        to_local = basis.T
        transform = np.zeros((4, 4), dtype=np.float64)
        transform[:2, :2] = to_local
        transform[2:, 2:] = to_local
        local_covariance = transform @ np.asarray(covariance_xyvxvy, dtype=np.float64).reshape(4, 4) @ transform.T
        velocity_scale = np.diag(
            [
                1.0,
                1.0,
                self.longitudinal_velocity_covariance_scale,
                self.lateral_velocity_covariance_scale,
            ]
        )
        local_covariance = velocity_scale @ local_covariance @ velocity_scale
        return speed, basis, local_covariance

    def _velocity_aligned_covariance_pair(
        self,
        context: tuple[float, np.ndarray, np.ndarray],
        tau: float,
        sigma_a: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return position/effective covariance without repeating 4x4 algebra.

        The existing velocity-aligned formulation discards predicted XY
        cross-covariance in the velocity frame before rotating the two axis
        variances back to robot XY.  Computing those two diagonal terms
        directly is algebraically equivalent and avoids two 4x4 transition
        products plus two 2x2 eigendecompositions per prediction time.
        """

        speed, basis, local_covariance = context
        tau_value = float(tau)
        tau_sq = tau_value * tau_value
        tau_fourth = tau_sq * tau_sq
        base_longitudinal = (
            float(local_covariance[0, 0])
            + 2.0 * tau_value * float(local_covariance[0, 2])
            + tau_sq * float(local_covariance[2, 2])
            + 0.25
            * tau_fourth
            * (float(sigma_a) * self.longitudinal_process_acceleration_scale) ** 2
        )
        base_lateral = (
            float(local_covariance[1, 1])
            + 2.0 * tau_value * float(local_covariance[1, 3])
            + tau_sq * float(local_covariance[3, 3])
            + 0.25
            * tau_fourth
            * (float(sigma_a) * self.lateral_process_acceleration_scale) ** 2
        )

        position_variances = np.clip(
            self._velocity_aligned_axis_variances(
                base_longitudinal,
                base_lateral,
                speed=speed,
                tau=tau_value,
                current_longitudinal=float(local_covariance[0, 0]),
                current_lateral=float(local_covariance[1, 1]),
                shape_variance=0.0,
            )
            + self.covariance_epsilon,
            self.covariance_eigenvalue_min_m2,
            self.covariance_eigenvalue_max_m2,
        )
        shape_variance = self.sigma_shape_m**2 + self.covariance_epsilon
        effective_variances = self._velocity_aligned_axis_variances(
            base_longitudinal,
            base_lateral,
            speed=speed,
            tau=tau_value,
            current_longitudinal=float(local_covariance[0, 0]),
            current_lateral=float(local_covariance[1, 1]),
            shape_variance=shape_variance,
        )
        position_covariance = basis @ np.diag(position_variances) @ basis.T
        effective_covariance = basis @ np.diag(effective_variances) @ basis.T
        return (
            0.5 * (position_covariance + position_covariance.T),
            position_variances,
            0.5 * (effective_covariance + effective_covariance.T),
            effective_variances,
        )

    def _velocity_aligned_axis_variances(
        self,
        base_longitudinal: float,
        base_lateral: float,
        *,
        speed: float,
        tau: float,
        current_longitudinal: float,
        current_lateral: float,
        shape_variance: float,
    ) -> np.ndarray:
        current_var_longitudinal = max(
            float(current_longitudinal) + float(shape_variance),
            self.covariance_eigenvalue_min_m2,
        )
        current_var_lateral = max(
            float(current_lateral) + float(shape_variance),
            self.covariance_eigenvalue_min_m2,
        )
        var_longitudinal = self._clamp_axis_variance(
            float(base_longitudinal) + float(shape_variance),
            axis_max_m2=self._longitudinal_variance_cap(speed, tau, current_var_longitudinal),
        )
        var_lateral = self._clamp_axis_variance(
            float(base_lateral) + float(shape_variance),
            axis_max_m2=self._lateral_variance_cap(speed, tau, current_var_lateral),
        )
        return np.clip(
            np.asarray([var_longitudinal, var_lateral], dtype=np.float64),
            self.covariance_eigenvalue_min_m2,
            self.covariance_eigenvalue_max_m2,
        )

    def _cap_isotropic_speed_bounded_covariance(self, covariance_xy: np.ndarray, tau: float) -> np.ndarray:
        regularized = self._regularize_covariance_2d(covariance_xy)
        if regularized is None:
            return np.eye(2, dtype=np.float64) * self.covariance_eigenvalue_min_m2
        covariance, _ = regularized
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        cap = self.covariance_eigenvalue_max_lateral_m2
        if self.human_speed_max_mps is not None and tau > 0.0:
            cutoff_sigma = math.sqrt(self.mahalanobis_cutoff_sq)
            speed_cap = (self.human_speed_max_mps * float(tau) / cutoff_sigma) ** 2
            cap = min(cap, max(float(np.max(eigenvalues)), speed_cap))
        capped = np.clip(eigenvalues, self.covariance_eigenvalue_min_m2, cap)
        capped_covariance = eigenvectors @ np.diag(capped) @ eigenvectors.T
        return 0.5 * (capped_covariance + capped_covariance.T)

    def _clamp_axis_variance(self, variance_m2: float, *, axis_max_m2: float) -> float:
        value = float(variance_m2)
        if not np.isfinite(value):
            value = self.covariance_eigenvalue_min_m2
        return float(np.clip(value, self.covariance_eigenvalue_min_m2, max(axis_max_m2, self.covariance_eigenvalue_min_m2)))

    def _longitudinal_variance_cap(self, speed_mps: float, tau: float, current_variance_m2: float) -> float:
        cap = self.covariance_eigenvalue_max_longitudinal_m2
        if self.human_speed_max_mps is None or tau <= 0.0:
            return cap
        speed = min(max(0.0, float(speed_mps)), self.human_speed_max_mps)
        speed_margin_mps = max(speed, self.human_speed_max_mps - speed)
        bounded_std = speed_margin_mps * float(tau) / math.sqrt(self.mahalanobis_cutoff_sq)
        return min(cap, max(float(current_variance_m2), bounded_std**2))

    def _lateral_variance_cap(self, speed_mps: float, tau: float, current_variance_m2: float) -> float:
        cap = self.covariance_eigenvalue_max_lateral_m2
        if self.human_speed_max_mps is None or tau <= 0.0:
            return cap
        speed = min(max(0.0, float(speed_mps)), self.human_speed_max_mps)
        lateral_speed_margin = math.sqrt(max(0.0, self.human_speed_max_mps**2 - speed**2))
        bounded_std = lateral_speed_margin * float(tau) / math.sqrt(self.mahalanobis_cutoff_sq)
        return min(cap, max(float(current_variance_m2), bounded_std**2))

    @staticmethod
    def _velocity_frame_basis(velocity_robot: np.ndarray) -> np.ndarray:
        velocity = np.asarray(velocity_robot, dtype=np.float64).reshape(2)
        speed = max(float(np.linalg.norm(velocity)), 1.0e-12)
        longitudinal = velocity / speed
        lateral = np.asarray([-longitudinal[1], longitudinal[0]], dtype=np.float64)
        return np.column_stack((longitudinal, lateral)).astype(np.float64, copy=False)

    def _regularize_covariance_2d(self, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        cov = np.asarray(covariance, dtype=np.float64)
        if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
            return None
        cov = 0.5 * (cov + cov.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return None
        clamped = np.clip(
            eigenvalues,
            self.covariance_eigenvalue_min_m2,
            self.covariance_eigenvalue_max_m2,
        )
        regularized = eigenvectors @ np.diag(clamped) @ eigenvectors.T
        return 0.5 * (regularized + regularized.T), clamped

    def _gaussian_cost(self, mean_xy: np.ndarray, covariance_xy: np.ndarray) -> np.ndarray:
        output = np.zeros(self.output_shape, dtype=np.float32)
        self._apply_gaussian_cost(output, mean_xy, covariance_xy)
        return output

    def _apply_gaussian_cost(self, output: np.ndarray, mean_xy: np.ndarray, covariance_xy: np.ndarray) -> None:
        mean = np.asarray(mean_xy, dtype=np.float64).reshape(2)
        row_slice, col_slice = self._gaussian_support_slices(mean, covariance_xy)
        if row_slice.stop <= row_slice.start or col_slice.stop <= col_slice.start:
            return
        covariance = np.asarray(covariance_xy, dtype=np.float64).reshape(2, 2)
        determinant = float(covariance[0, 0] * covariance[1, 1] - covariance[0, 1] * covariance[1, 0])
        if np.isfinite(determinant) and determinant > 0.0:
            inv_covariance = np.asarray(
                [
                    [covariance[1, 1], -covariance[0, 1]],
                    [-covariance[1, 0], covariance[0, 0]],
                ],
                dtype=np.float64,
            ) / determinant
        else:
            inv_covariance = np.linalg.inv(covariance)
        dx = self._grid_x_m[row_slice, col_slice] - float(mean[0])
        dy = self._grid_y_m[row_slice, col_slice] - float(mean[1])
        d2 = (
            inv_covariance[0, 0] * dx * dx
            + 2.0 * inv_covariance[0, 1] * dx * dy
            + inv_covariance[1, 1] * dy * dy
        )
        local = np.exp(-0.5 * d2)
        local = np.where(d2 <= self.mahalanobis_cutoff_sq, local, 0.0)
        target = output[row_slice, col_slice]
        np.maximum(target, np.asarray(local, dtype=np.float32), out=target)

    def _gaussian_support_slices(self, mean_xy: np.ndarray, covariance_xy: np.ndarray) -> tuple[slice, slice]:
        covariance = np.asarray(covariance_xy, dtype=np.float64)
        cutoff = max(0.0, float(self.mahalanobis_cutoff_sq))
        radius_x = math.sqrt(max(0.0, cutoff * float(covariance[0, 0])))
        radius_y = math.sqrt(max(0.0, cutoff * float(covariance[1, 1])))
        x_values = self._grid_x_m[:, 0]
        y_values = self._grid_y_m[0, :]
        row_indices = np.flatnonzero(
            (x_values >= float(mean_xy[0]) - radius_x) & (x_values <= float(mean_xy[0]) + radius_x)
        )
        col_indices = np.flatnonzero(
            (y_values >= float(mean_xy[1]) - radius_y) & (y_values <= float(mean_xy[1]) + radius_y)
        )
        if row_indices.size == 0 or col_indices.size == 0:
            return slice(0, 0), slice(0, 0)
        return slice(int(row_indices[0]), int(row_indices[-1]) + 1), slice(int(col_indices[0]), int(col_indices[-1]) + 1)
