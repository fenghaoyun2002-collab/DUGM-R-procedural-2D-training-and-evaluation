from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, Sequence

import numpy as np

from dugm_r.simulation_2d.scene import (
    BoxObstacle,
    CircleObstacle,
    Scene2D,
    _dynamic_actor_primitives,
    _dynamic_actor_proxy_radius,
    _dynamic_actor_state,
)
from dugm_r.dugm.scan_grid import GridEncoder
from dugm_r.dugm.prediction import (
    EstimatedDynamicTrackState,
    GtDynamicActorState,
)
from dugm_r.dugm.track import SEMANTIC_DYNAMIC


def observed_unsafe_from_grid(grid: np.ndarray) -> np.ndarray:
    array = np.asarray(grid, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] < 2:
        raise ValueError(f"Expected at least occupancy/observed grid channels, got shape {array.shape}.")
    occupancy = array[0] > 0.5
    observed = array[1] > 0.5
    return np.where(occupancy | ~observed, 1.0, 0.0).astype(np.float32)


def stack_occupied_costmap(base_grid: np.ndarray, visible_occupancy_cost: np.ndarray) -> np.ndarray:
    base = np.asarray(base_grid, dtype=np.float32)
    cost = np.asarray(visible_occupancy_cost, dtype=np.float32)
    if base.ndim != 3 or base.shape[0] < 2:
        raise ValueError(f"base_grid must have shape (2, H, W), got {base.shape}.")
    if cost.shape != tuple(base.shape[1:]):
        raise ValueError(f"visible_occupancy_cost shape {cost.shape} does not match grid spatial shape {base.shape[1:]}.")
    return np.stack(
        (
            observed_unsafe_from_grid(base),
            np.clip(cost, 0.0, 1.0).astype(np.float32, copy=False),
        ),
        axis=0,
    ).astype(np.float32, copy=False)


def visible_static_endpoint_cost_from_ranges(
    grid_encoder: GridEncoder,
    *,
    static_ranges: np.ndarray,
    measured_ranges: np.ndarray,
    angle_min_rad: float,
    angle_increment_rad: float,
    min_range_m: float,
    max_range_m: float,
    visibility_range_tolerance_m: float,
    inflation_radius_m: float,
) -> np.ndarray:
    static = np.asarray(static_ranges, dtype=np.float32).reshape(grid_encoder.geometry.num_rays)
    measured = np.asarray(measured_ranges, dtype=np.float32).reshape(grid_encoder.geometry.num_rays)
    tolerance = float(visibility_range_tolerance_m)
    if tolerance < 0.0:
        raise ValueError("visibility_range_tolerance_m must be non-negative.")
    visible_static_hit = (
        np.isfinite(static)
        & np.isfinite(measured)
        & (static >= float(min_range_m))
        & (static < float(max_range_m))
        & (np.abs(static - measured) <= tolerance)
    )
    static_occupancy = grid_encoder.rasterize_occupied_endpoints(
        static,
        valid_mask=visible_static_hit,
        angle_min_rad=float(angle_min_rad),
        angle_increment_rad=float(angle_increment_rad),
        min_range_m=float(min_range_m),
        max_range_m=float(max_range_m),
    )
    return inflate_occupied_cells(
        static_occupancy,
        radius_m=float(inflation_radius_m),
        resolution_m=grid_encoder.geometry.resolution_m,
    )


def visible_endpoint_cost_from_ranges(
    grid_encoder: GridEncoder,
    *,
    ranges: np.ndarray,
    valid_mask: np.ndarray,
    angle_min_rad: float,
    angle_increment_rad: float,
    min_range_m: float,
    max_range_m: float,
    inflation_radius_m: float,
) -> np.ndarray:
    scan_ranges = np.asarray(ranges, dtype=np.float32).reshape(grid_encoder.geometry.num_rays)
    scan_valid = np.asarray(valid_mask, dtype=bool).reshape(grid_encoder.geometry.num_rays)
    endpoint_hits = (
        scan_valid
        & np.isfinite(scan_ranges)
        & (scan_ranges >= float(min_range_m))
        & (scan_ranges < float(max_range_m))
    )
    endpoint_occupancy = grid_encoder.rasterize_occupied_endpoints(
        scan_ranges,
        valid_mask=endpoint_hits,
        angle_min_rad=float(angle_min_rad),
        angle_increment_rad=float(angle_increment_rad),
        min_range_m=float(min_range_m),
        max_range_m=float(max_range_m),
    )
    return inflate_occupied_cells(
        endpoint_occupancy,
        radius_m=float(inflation_radius_m),
        resolution_m=grid_encoder.geometry.resolution_m,
    )


def visible_dynamic_actor_states(
    scene: Scene2D,
    *,
    timestamp_s: float,
    ranges: np.ndarray,
    robot_pose_xyyaw: np.ndarray,
    num_rays: int,
    angle_min_rad: float,
    angle_increment_rad: float,
    min_range_m: float,
    max_range_m: float,
    visibility_mode: str = "visible",
    visibility_range_tolerance_m: float = 0.35,
) -> tuple[GtDynamicActorState, ...]:
    mode = str(visibility_mode).lower()
    if mode not in {"visible", "all"}:
        raise ValueError("visibility_mode must be 'visible' or 'all'.")
    tolerance = float(visibility_range_tolerance_m)
    if tolerance < 0.0:
        raise ValueError("visibility_range_tolerance_m must be non-negative.")

    states: list[GtDynamicActorState] = []
    primitives_by_actor: list[tuple[CircleObstacle | BoxObstacle, ...]] = []
    for actor in scene.dynamic_actors:
        center, velocity = _dynamic_actor_state(actor, timestamp_s, scene.bounds_x_m, scene.bounds_y_m)
        primitives = _dynamic_actor_primitives(actor, center=center, velocity=velocity)
        primitives_by_actor.append(primitives)
        states.append(
            GtDynamicActorState(
                center_world=np.asarray(center, dtype=np.float64),
                velocity_world=np.asarray(velocity, dtype=np.float64),
                radius_m=float(_dynamic_actor_proxy_radius(actor)),
                semantic_kind=str(actor.kind),
            )
        )
    if mode == "visible":
        visible_mask = _dynamic_actor_visibility_mask(
            primitives_by_actor,
            ranges=ranges,
            robot_pose_xyyaw=robot_pose_xyyaw,
            num_rays=num_rays,
            angle_min_rad=angle_min_rad,
            angle_increment_rad=angle_increment_rad,
            min_range_m=min_range_m,
            max_range_m=max_range_m,
            visibility_range_tolerance_m=tolerance,
        )
        states = [state for state, visible in zip(states, visible_mask) if bool(visible)]
    return tuple(states)


def estimated_dynamic_track_states(
    tracks: Sequence[Any],
    *,
    velocity_valid_max_std_mps: float,
    default_radius_m: float,
    min_radius_m: float,
    max_radius_m: float,
    require_current_measurement: bool = True,
    suppress_maneuver_tracks: bool = True,
    fallback_process_acceleration_std_mps2: float = 0.0,
    return_diagnostics: bool = False,
) -> tuple[EstimatedDynamicTrackState, ...] | tuple[tuple[EstimatedDynamicTrackState, ...], dict[str, Any]]:
    if velocity_valid_max_std_mps < 0.0:
        raise ValueError("velocity_valid_max_std_mps must be non-negative.")
    if default_radius_m < 0.0 or min_radius_m < 0.0 or max_radius_m < 0.0:
        raise ValueError("Estimated track radii must be non-negative.")
    if max_radius_m < min_radius_m:
        raise ValueError("max_radius_m must be greater than or equal to min_radius_m.")
    if fallback_process_acceleration_std_mps2 < 0.0:
        raise ValueError("fallback_process_acceleration_std_mps2 must be non-negative.")

    diagnostics: dict[str, Any] = {
        "estimated_velocity_cost_candidate_track_count": int(len(tracks)),
        "estimated_velocity_cost_skipped_unconfirmed_count": 0,
        "estimated_velocity_cost_skipped_non_dynamic_count": 0,
        "estimated_velocity_cost_skipped_missing_measurement_count": 0,
        "estimated_velocity_cost_skipped_maneuver_count": 0,
        "estimated_velocity_cost_skipped_velocity_std_count": 0,
        "estimated_velocity_cost_skipped_invalid_state_count": 0,
        "estimated_velocity_cost_skipped_invalid_covariance_count": 0,
        "estimated_velocity_cost_fallback_covariance_count": 0,
    }
    states: list[EstimatedDynamicTrackState] = []
    for track in tracks:
        if not bool(getattr(track, "confirmed", False)):
            diagnostics["estimated_velocity_cost_skipped_unconfirmed_count"] += 1
            continue
        if getattr(track, "semantic_state", None) != SEMANTIC_DYNAMIC:
            diagnostics["estimated_velocity_cost_skipped_non_dynamic_count"] += 1
            continue
        if require_current_measurement and not bool(getattr(track, "current_measurement_available", False)):
            diagnostics["estimated_velocity_cost_skipped_missing_measurement_count"] += 1
            continue
        if suppress_maneuver_tracks and bool(getattr(track, "in_maneuver_mode", False)):
            diagnostics["estimated_velocity_cost_skipped_maneuver_count"] += 1
            continue
        velocity_std_mps = float(getattr(track, "velocity_std_mps", math.inf))
        if not np.isfinite(velocity_std_mps) or velocity_std_mps > float(velocity_valid_max_std_mps):
            diagnostics["estimated_velocity_cost_skipped_velocity_std_count"] += 1
            continue
        state = np.asarray(getattr(track, "state", ()), dtype=np.float64).reshape(-1)
        if state.size < 4 or not np.all(np.isfinite(state[:4])):
            diagnostics["estimated_velocity_cost_skipped_invalid_state_count"] += 1
            continue
        covariance, fallback_used = _estimated_track_covariance_xyvxvy(
            track,
            default_radius_m=float(default_radius_m),
            velocity_std_mps=velocity_std_mps,
        )
        if covariance is None:
            diagnostics["estimated_velocity_cost_skipped_invalid_covariance_count"] += 1
            continue
        if fallback_used:
            diagnostics["estimated_velocity_cost_fallback_covariance_count"] += 1
        last_process_acceleration_std_mps2 = float(
            getattr(track, "last_process_acceleration_std_mps2", float(fallback_process_acceleration_std_mps2))
        )
        if not np.isfinite(last_process_acceleration_std_mps2) or last_process_acceleration_std_mps2 < 0.0:
            last_process_acceleration_std_mps2 = float(fallback_process_acceleration_std_mps2)
        states.append(
            EstimatedDynamicTrackState(
                center_robot=state[:2].astype(np.float64, copy=True),
                velocity_robot=state[2:4].astype(np.float64, copy=True),
                radius_m=_estimated_track_radius_m(
                    track,
                    default_radius_m=float(default_radius_m),
                    min_radius_m=float(min_radius_m),
                    max_radius_m=float(max_radius_m),
                ),
                track_id=int(getattr(track, "track_id", -1)),
                velocity_std_mps=velocity_std_mps,
                state_xyvxvy_robot=state[:4].astype(np.float64, copy=True),
                covariance_xyvxvy_robot=covariance,
                current_measurement_available=bool(getattr(track, "current_measurement_available", False)),
                in_maneuver_mode=bool(getattr(track, "in_maneuver_mode", False)),
                miss_count=int(getattr(track, "miss_count", 0)),
                last_process_acceleration_std_mps2=float(last_process_acceleration_std_mps2),
            )
        )
    result = tuple(states)
    if return_diagnostics:
        return result, diagnostics
    return result


def _estimated_track_covariance_xyvxvy(
    track: Any,
    *,
    default_radius_m: float,
    velocity_std_mps: float,
) -> tuple[np.ndarray | None, bool]:
    covariance = np.asarray(getattr(track, "covariance", ()), dtype=np.float64)
    if covariance.shape == (4, 4) and np.all(np.isfinite(covariance)):
        return 0.5 * (covariance + covariance.T), False

    position_std_m = max(float(default_radius_m), 1.0e-3)
    velocity_std = max(float(velocity_std_mps), 1.0e-3)
    if not np.isfinite(position_std_m) or not np.isfinite(velocity_std):
        return None, False
    fallback = np.diag([position_std_m**2, position_std_m**2, velocity_std**2, velocity_std**2]).astype(
        np.float64,
        copy=False,
    )
    return fallback, True


def _estimated_track_radius_m(
    track: Any,
    *,
    default_radius_m: float,
    min_radius_m: float,
    max_radius_m: float,
) -> float:
    cluster = getattr(track, "latest_cluster", None)
    diameter = float(getattr(cluster, "diameter_m", math.nan)) if cluster is not None else math.nan
    radius = 0.5 * diameter if np.isfinite(diameter) and diameter > 0.0 else float(default_radius_m)
    return float(np.clip(radius, float(min_radius_m), float(max_radius_m)))


def dynamic_actor_visible(
    primitives: tuple[CircleObstacle | BoxObstacle, ...],
    *,
    ranges: np.ndarray,
    robot_pose_xyyaw: np.ndarray,
    num_rays: int,
    angle_min_rad: float,
    angle_increment_rad: float,
    min_range_m: float,
    max_range_m: float,
    visibility_range_tolerance_m: float,
) -> bool:
    measured_ranges = np.asarray(ranges, dtype=np.float64).reshape(-1)
    if measured_ranges.shape != (int(num_rays),):
        return False
    origin = np.asarray(robot_pose_xyyaw[:2], dtype=np.float64).reshape(2)
    yaw = float(robot_pose_xyyaw[2])
    for beam_id, measured in enumerate(measured_ranges):
        if not np.isfinite(measured) or measured >= float(max_range_m):
            continue
        angle = yaw + float(angle_min_rad) + beam_id * float(angle_increment_rad)
        direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
        expected = min(
            (
                _raycast_primitive_distance(
                    origin,
                    direction,
                    primitive,
                    max_range_m=float(max_range_m),
                )
                for primitive in primitives
            ),
            default=math.inf,
        )
        if (
            np.isfinite(expected)
            and expected >= float(min_range_m)
            and abs(float(measured) - float(expected)) <= float(visibility_range_tolerance_m)
        ):
            return True
    return False


def _dynamic_actor_visibility_mask(
    primitives_by_actor: Sequence[tuple[CircleObstacle | BoxObstacle, ...]],
    *,
    ranges: np.ndarray,
    robot_pose_xyyaw: np.ndarray,
    num_rays: int,
    angle_min_rad: float,
    angle_increment_rad: float,
    min_range_m: float,
    max_range_m: float,
    visibility_range_tolerance_m: float,
) -> np.ndarray:
    actor_count = len(primitives_by_actor)
    visible = np.zeros(actor_count, dtype=bool)
    if actor_count == 0:
        return visible
    measured_ranges = np.asarray(ranges, dtype=np.float64).reshape(-1)
    if measured_ranges.shape != (int(num_rays),):
        return visible

    usable_beams = np.isfinite(measured_ranges) & (measured_ranges < float(max_range_m))
    beam_ids = np.flatnonzero(usable_beams)
    if beam_ids.size == 0:
        return visible

    pose = np.asarray(robot_pose_xyyaw, dtype=np.float64).reshape(3)
    origin = pose[:2]
    yaw = float(pose[2])
    angles = yaw + float(angle_min_rad) + beam_ids.astype(np.float64) * float(angle_increment_rad)
    directions = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float64, copy=False)
    expected_by_actor = np.full((beam_ids.size, actor_count), np.inf, dtype=np.float64)

    circle_centers: list[tuple[float, float]] = []
    circle_radii: list[float] = []
    circle_actor_ids: list[int] = []
    box_centers: list[tuple[float, float]] = []
    box_sizes: list[tuple[float, float]] = []
    box_actor_ids: list[int] = []
    for actor_id, primitives in enumerate(primitives_by_actor):
        for primitive in primitives:
            if isinstance(primitive, CircleObstacle):
                circle_centers.append(primitive.center)
                circle_radii.append(float(primitive.radius_m))
                circle_actor_ids.append(actor_id)
            else:
                box_centers.append(primitive.center)
                box_sizes.append(primitive.size_m)
                box_actor_ids.append(actor_id)

    if circle_centers:
        circle_hits = _raycast_circles_batch(
            origin,
            directions,
            np.asarray(circle_centers, dtype=np.float64).reshape(-1, 2),
            np.asarray(circle_radii, dtype=np.float64),
            float(max_range_m),
        )
        _minimum_hits_by_actor(expected_by_actor, circle_hits, np.asarray(circle_actor_ids, dtype=np.int64))
    if box_centers:
        box_hits = _raycast_aabbs_batch(
            origin,
            directions,
            np.asarray(box_centers, dtype=np.float64).reshape(-1, 2),
            np.asarray(box_sizes, dtype=np.float64).reshape(-1, 2),
            float(max_range_m),
        )
        _minimum_hits_by_actor(expected_by_actor, box_hits, np.asarray(box_actor_ids, dtype=np.int64))

    measured = measured_ranges[beam_ids].reshape(-1, 1)
    visible_hits = (
        np.isfinite(expected_by_actor)
        & (expected_by_actor >= float(min_range_m))
        & (np.abs(measured - expected_by_actor) <= float(visibility_range_tolerance_m))
    )
    return np.any(visible_hits, axis=0)


def _minimum_hits_by_actor(expected_by_actor: np.ndarray, hits: np.ndarray, actor_ids: np.ndarray) -> None:
    if hits.size == 0:
        return
    rows = np.repeat(np.arange(hits.shape[0], dtype=np.int64), hits.shape[1])
    cols = np.tile(actor_ids.astype(np.int64, copy=False), hits.shape[0])
    np.minimum.at(expected_by_actor, (rows, cols), hits.reshape(-1))


def _raycast_circles_batch(
    origin_xy: np.ndarray,
    directions_xy: np.ndarray,
    centers_xy: np.ndarray,
    radii_m: np.ndarray,
    max_range_m: float,
) -> np.ndarray:
    if centers_xy.size == 0:
        return np.empty((directions_xy.shape[0], 0), dtype=np.float64)
    origin = np.asarray(origin_xy, dtype=np.float64).reshape(2)
    directions = np.asarray(directions_xy, dtype=np.float64).reshape(-1, 2)
    centers = np.asarray(centers_xy, dtype=np.float64).reshape(-1, 2)
    radii = np.asarray(radii_m, dtype=np.float64).reshape(-1)
    oc = origin.reshape(1, 2) - centers
    b = 2.0 * np.einsum("bd,cd->bc", directions, oc, optimize=True)
    c = np.sum(oc * oc, axis=1) - radii * radii
    disc = b * b - 4.0 * c.reshape(1, -1)
    valid_disc = disc >= 0.0
    root = np.sqrt(np.maximum(disc, 0.0))
    t1 = (-b - root) * 0.5
    t2 = (-b + root) * 0.5
    t1 = np.where(valid_disc & (t1 >= 0.0) & (t1 <= float(max_range_m)), t1, np.inf)
    t2 = np.where(valid_disc & (t2 >= 0.0) & (t2 <= float(max_range_m)), t2, np.inf)
    return np.minimum(t1, t2)


def _raycast_aabbs_batch(
    origin_xy: np.ndarray,
    directions_xy: np.ndarray,
    centers_xy: np.ndarray,
    sizes_m: np.ndarray,
    max_range_m: float,
) -> np.ndarray:
    if centers_xy.size == 0:
        return np.empty((directions_xy.shape[0], 0), dtype=np.float64)
    origin = np.asarray(origin_xy, dtype=np.float64).reshape(2)
    directions = np.asarray(directions_xy, dtype=np.float64).reshape(-1, 2)
    centers = np.asarray(centers_xy, dtype=np.float64).reshape(-1, 2)
    half = 0.5 * np.asarray(sizes_m, dtype=np.float64).reshape(-1, 2)
    bounds_min = centers - half
    bounds_max = centers + half
    valid = np.ones((directions.shape[0], centers.shape[0]), dtype=bool)
    near_values: list[np.ndarray] = []
    far_values: list[np.ndarray] = []
    for axis in range(2):
        direction = directions[:, axis].reshape(-1, 1)
        origin_axis = float(origin[axis])
        slab_min = bounds_min[:, axis].reshape(1, -1)
        slab_max = bounds_max[:, axis].reshape(1, -1)
        parallel = np.abs(direction) < 1.0e-12
        inside_parallel_slab = (origin_axis >= slab_min) & (origin_axis <= slab_max)
        valid &= (~parallel) | inside_parallel_slab

        t1 = np.divide(slab_min - origin_axis, direction, out=np.zeros_like(valid, dtype=np.float64), where=~parallel)
        t2 = np.divide(slab_max - origin_axis, direction, out=np.zeros_like(valid, dtype=np.float64), where=~parallel)
        near = np.minimum(t1, t2)
        far = np.maximum(t1, t2)
        near_values.append(np.where(parallel, -np.inf, near))
        far_values.append(np.where(parallel, np.inf, far))

    t_min = np.maximum.reduce(near_values)
    t_max = np.minimum.reduce(far_values)
    hit_distance = np.where(t_min >= 0.0, t_min, t_max)
    hit = (
        valid
        & (t_min <= t_max)
        & (t_max >= 0.0)
        & (hit_distance >= 0.0)
        & (hit_distance <= float(max_range_m))
    )
    return np.where(hit, hit_distance, np.inf)


def occupied_costmap_diagnostics(
    *,
    static_cost: np.ndarray,
    visible_occupancy_cost: np.ndarray,
    static_inflation_radius_m: float,
    static_visibility_range_tolerance_m: float,
) -> dict[str, Any]:
    static = np.asarray(static_cost, dtype=np.float32)
    visible = np.asarray(visible_occupancy_cost, dtype=np.float32)
    return {
        "visible_static_cost_inflation_radius_m": float(static_inflation_radius_m),
        "visible_static_cost_visibility_range_tolerance_m": float(static_visibility_range_tolerance_m),
        "visible_static_cost_nonzero_cell_count": int(np.sum(static > 0.0)),
        "visible_static_cost_core_cell_count": int(np.sum(static >= 1.0)),
        "visible_static_cost_max": float(np.max(static)) if static.size else 0.0,
        "visible_occupancy_cost_nonzero_cell_count": int(np.sum(visible > 0.0)),
        "visible_occupancy_cost_core_cell_count": int(np.sum(visible >= 1.0)),
        "visible_occupancy_cost_max": float(np.max(visible)) if visible.size else 0.0,
        "visible_occupancy_cost_static_dynamic_aggregation": "cellwise_max",
    }


def resolve_velocity_cost_core_inflation_radius(
    velocity_cost_config: dict[str, Any],
    *,
    robot_radius_m: float,
    collision_clearance_margin_m: float,
) -> float:
    raw_radius = velocity_cost_config.get("core_inflation_radius_m", "auto")
    if raw_radius is None:
        return 0.0
    if isinstance(raw_radius, bool):
        raise ValueError("observation.velocity_cost.core_inflation_radius_m must be a number or supported string.")
    if isinstance(raw_radius, str):
        key = raw_radius.strip().lower()
        if key in {"", "none", "disabled"}:
            return 0.0
        if key in {"auto", "robot_collision_clearance", "robot_radius_plus_collision_clearance_margin"}:
            return float(robot_radius_m) + max(0.0, float(collision_clearance_margin_m))
        try:
            radius = float(key)
        except ValueError as exc:
            raise ValueError(
                "observation.velocity_cost.core_inflation_radius_m must be numeric, "
                "'auto', or 'robot_radius_plus_collision_clearance_margin'."
            ) from exc
    else:
        radius = float(raw_radius)
    if radius < 0.0:
        raise ValueError("observation.velocity_cost.core_inflation_radius_m must be non-negative.")
    return radius


def resolve_static_endpoint_inflation_radius(velocity_cost_config: dict[str, Any]) -> float:
    raw_radius = velocity_cost_config.get("static_endpoint_inflation_radius_m", 0.0)
    if raw_radius is None:
        return 0.0
    if isinstance(raw_radius, bool):
        raise ValueError("observation.velocity_cost.static_endpoint_inflation_radius_m must be numeric.")
    if isinstance(raw_radius, str):
        key = raw_radius.strip().lower()
        if key in {"", "none", "disabled"}:
            return 0.0
        try:
            radius = float(key)
        except ValueError as exc:
            raise ValueError("observation.velocity_cost.static_endpoint_inflation_radius_m must be numeric.") from exc
    else:
        radius = float(raw_radius)
    if radius < 0.0:
        raise ValueError("observation.velocity_cost.static_endpoint_inflation_radius_m must be non-negative.")
    return radius


def inflate_occupied_cells(occupancy: np.ndarray, *, radius_m: float, resolution_m: float) -> np.ndarray:
    """Conservatively mark any grid cell touched by the inflated occupied layer."""

    grid = np.asarray(occupancy, dtype=np.float32)
    if grid.ndim != 2:
        raise ValueError(f"occupancy must be a 2D grid, got shape {grid.shape}.")
    radius = float(radius_m)
    resolution = float(resolution_m)
    if radius < 0.0:
        raise ValueError("radius_m must be non-negative.")
    if resolution <= 0.0:
        raise ValueError("resolution_m must be positive.")
    occupied = grid > 0.5
    if radius <= 0.0 or not np.any(occupied):
        return occupied.astype(np.float32)

    rows, cols = np.nonzero(occupied)
    inflated = occupied.copy()
    for row_offset, col_offset in _cell_overlap_inflation_offsets(radius, resolution):
        target_rows = rows + row_offset
        target_cols = cols + col_offset
        inside = (
            (0 <= target_rows)
            & (target_rows < inflated.shape[0])
            & (0 <= target_cols)
            & (target_cols < inflated.shape[1])
        )
        inflated[target_rows[inside], target_cols[inside]] = True
    return inflated.astype(np.float32)


@lru_cache(maxsize=32)
def _cell_overlap_inflation_offsets(radius_m: float, resolution_m: float) -> tuple[tuple[int, int], ...]:
    radius = float(radius_m)
    resolution = float(resolution_m)
    max_offset = int(math.floor(radius / resolution + 1.0e-12)) + 1
    offsets: list[tuple[int, int]] = []
    for row_offset in range(-max_offset, max_offset + 1):
        for col_offset in range(-max_offset, max_offset + 1):
            row_gap = max(0.0, (abs(row_offset) - 1) * resolution)
            col_gap = max(0.0, (abs(col_offset) - 1) * resolution)
            if math.hypot(row_gap, col_gap) <= radius + 1.0e-12:
                offsets.append((row_offset, col_offset))
    return tuple(offsets)


def _raycast_primitive_distance(
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
    primitive: CircleObstacle | BoxObstacle,
    *,
    max_range_m: float,
) -> float:
    if isinstance(primitive, CircleObstacle):
        return _raycast_circle_distance(
            origin_xy,
            direction_xy,
            center_xy=np.asarray(primitive.center, dtype=np.float64),
            radius_m=float(primitive.radius_m),
            max_range_m=max_range_m,
        )
    return _raycast_aabb_distance(
        origin_xy,
        direction_xy,
        center_xy=np.asarray(primitive.center, dtype=np.float64),
        size_m=np.asarray(primitive.size_m, dtype=np.float64),
        max_range_m=max_range_m,
    )


def _raycast_circle_distance(
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
    *,
    center_xy: np.ndarray,
    radius_m: float,
    max_range_m: float,
) -> float:
    origin = np.asarray(origin_xy, dtype=np.float64).reshape(2)
    direction = np.asarray(direction_xy, dtype=np.float64).reshape(2)
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    oc = origin - center
    b = 2.0 * float(direction @ oc)
    c = float(oc @ oc) - float(radius_m) * float(radius_m)
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return math.inf
    root = math.sqrt(max(0.0, disc))
    candidates = [(-b - root) * 0.5, (-b + root) * 0.5]
    valid = [item for item in candidates if 0.0 <= item <= float(max_range_m)]
    return float(min(valid)) if valid else math.inf


def _raycast_aabb_distance(
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
    *,
    center_xy: np.ndarray,
    size_m: np.ndarray,
    max_range_m: float,
) -> float:
    origin = np.asarray(origin_xy, dtype=np.float64).reshape(2)
    direction = np.asarray(direction_xy, dtype=np.float64).reshape(2)
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    half = 0.5 * np.asarray(size_m, dtype=np.float64).reshape(2)
    bounds_min = center - half
    bounds_max = center + half
    t_min = -math.inf
    t_max = math.inf
    for axis in range(2):
        component = float(direction[axis])
        if abs(component) < 1.0e-12:
            if origin[axis] < bounds_min[axis] or origin[axis] > bounds_max[axis]:
                return math.inf
            continue
        inv = 1.0 / component
        t1 = float((bounds_min[axis] - origin[axis]) * inv)
        t2 = float((bounds_max[axis] - origin[axis]) * inv)
        t_near, t_far = min(t1, t2), max(t1, t2)
        t_min = max(t_min, t_near)
        t_max = min(t_max, t_far)
        if t_min > t_max:
            return math.inf
    if t_max < 0.0:
        return math.inf
    hit = t_min if t_min >= 0.0 else t_max
    return float(hit) if 0.0 <= hit <= float(max_range_m) else math.inf
