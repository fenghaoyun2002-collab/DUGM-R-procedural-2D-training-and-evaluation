from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Sequence

import numpy as np

from dugm_r.simulation_2d.scene import BoxObstacle, CircleObstacle, PrimitiveObstacle, Scene2D


_EPS = 1e-9


@dataclass(frozen=True)
class StaticObstacleScanCache:
    bounds_x_m: tuple[float, float]
    bounds_y_m: tuple[float, float]
    circle_centers: np.ndarray
    circle_radii: np.ndarray
    box_centers: np.ndarray
    box_sizes: np.ndarray


def build_static_obstacle_scan_cache(scene: Scene2D) -> StaticObstacleScanCache:
    circles, boxes = _split_obstacles(scene.static_obstacles)
    circle_centers, circle_radii = circles
    box_centers, box_sizes = boxes
    return StaticObstacleScanCache(
        bounds_x_m=scene.bounds_x_m,
        bounds_y_m=scene.bounds_y_m,
        circle_centers=circle_centers,
        circle_radii=circle_radii,
        box_centers=box_centers,
        box_sizes=box_sizes,
    )


def scan_static_obstacle_cache(
    cache: StaticObstacleScanCache,
    *,
    origin_xy: tuple[float, float] | np.ndarray,
    yaw_rad: float,
    num_rays: int = 360,
    fov_rad: float = 2.0 * math.pi,
    min_range_m: float = 0.16,
    max_range_m: float = 8.0,
) -> np.ndarray:
    origin = np.asarray(origin_xy, dtype=np.float64).reshape(2)
    directions = _ray_directions(int(num_rays), float(fov_rad), float(yaw_rad))
    ranges = _scan_obstacle_batches(
        origin=origin,
        directions=directions,
        bounds_x_m=cache.bounds_x_m,
        bounds_y_m=cache.bounds_y_m,
        circle_centers=cache.circle_centers,
        circle_radii=cache.circle_radii,
        box_centers=cache.box_centers,
        box_sizes=cache.box_sizes,
        max_range_m=float(max_range_m),
    )
    return np.clip(ranges, float(min_range_m), float(max_range_m)).astype(np.float32, copy=False)


def scan_scene(
    scene: Scene2D,
    *,
    origin_xy: tuple[float, float] | np.ndarray,
    yaw_rad: float,
    time_s: float = 0.0,
    num_rays: int = 360,
    fov_rad: float = 2.0 * math.pi,
    min_range_m: float = 0.16,
    max_range_m: float = 8.0,
    range_bias_std_m: float = 0.0,
    noise_std_m: float = 0.0,
    range_noise_ratio: float = 0.0,
    max_abs_noise_m: float = 0.0,
    max_abs_noise_ratio: float = 0.0,
    dropout_prob: float = 0.0,
    false_near_prob: float = 0.0,
    max_range_spike_prob: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    origin = np.asarray(origin_xy, dtype=np.float64).reshape(2)
    directions = _ray_directions(int(num_rays), float(fov_rad), float(yaw_rad))
    ranges = _scan_scene_geometry(
        scene,
        origin=origin,
        directions=directions,
        time_s=float(time_s),
        max_range_m=float(max_range_m),
    ).astype(np.float32, copy=False)
    if rng is not None and range_bias_std_m > 0.0:
        ranges += float(rng.normal(0.0, float(range_bias_std_m)))
    if rng is not None and (noise_std_m > 0.0 or range_noise_ratio > 0.0):
        std = float(noise_std_m) + ranges * float(range_noise_ratio)
        noise = rng.normal(0.0, std, size=ranges.shape).astype(np.float32)
        max_abs = float(max_abs_noise_m) + ranges * float(max_abs_noise_ratio)
        if np.any(max_abs > 0.0):
            noise = np.clip(noise, -max_abs, max_abs)
        ranges += noise
    if rng is not None and dropout_prob > 0.0:
        mask = rng.random(ranges.shape) < float(dropout_prob)
        ranges[mask] = float(max_range_m)
    if rng is not None and false_near_prob > 0.0:
        mask = rng.random(ranges.shape) < float(false_near_prob)
        ranges[mask] = rng.uniform(float(min_range_m), min(1.0, float(max_range_m)), size=int(np.sum(mask))).astype(np.float32)
    if rng is not None and max_range_spike_prob > 0.0:
        mask = rng.random(ranges.shape) < float(max_range_spike_prob)
        ranges[mask] = float(max_range_m)
    return np.clip(ranges, float(min_range_m), float(max_range_m)).astype(np.float32, copy=False)


def raycast_scene(
    scene: Scene2D,
    *,
    origin_xy: tuple[float, float] | np.ndarray,
    angle_rad: float,
    time_s: float,
    max_range_m: float,
) -> float:
    origin = np.asarray(origin_xy, dtype=np.float64).reshape(2)
    directions = np.asarray([[math.cos(float(angle_rad)), math.sin(float(angle_rad))]], dtype=np.float64)
    return float(
        _scan_scene_geometry(
            scene,
            origin=origin,
            directions=directions,
            time_s=float(time_s),
            max_range_m=float(max_range_m),
        )[0]
    )


def collides(
    scene: Scene2D,
    *,
    point_xy: tuple[float, float] | np.ndarray,
    radius_m: float,
    time_s: float,
) -> bool:
    point = np.asarray(point_xy, dtype=np.float64).reshape(2)
    radius = float(radius_m)
    if (
        point[0] - radius < scene.bounds_x_m[0]
        or point[0] + radius > scene.bounds_x_m[1]
        or point[1] - radius < scene.bounds_y_m[0]
        or point[1] + radius > scene.bounds_y_m[1]
    ):
        return True
    obstacle_clearances = _clearances_to_obstacles(point, scene.obstacles_at(time_s))
    return bool(obstacle_clearances.size > 0 and np.any(obstacle_clearances <= radius))


def min_clearance(
    scene: Scene2D,
    *,
    point_xy: tuple[float, float] | np.ndarray,
    radius_m: float,
    time_s: float,
) -> float:
    point = np.asarray(point_xy, dtype=np.float64).reshape(2)
    bounds_clearances = np.asarray(
        [
            point[0] - scene.bounds_x_m[0],
            scene.bounds_x_m[1] - point[0],
            point[1] - scene.bounds_y_m[0],
            scene.bounds_y_m[1] - point[1],
        ],
        dtype=np.float64,
    )
    obstacle_clearances = _clearances_to_obstacles(point, scene.obstacles_at(time_s))
    if obstacle_clearances.size:
        clearances = np.concatenate([bounds_clearances, obstacle_clearances])
    else:
        clearances = bounds_clearances
    return float(np.min(clearances) - float(radius_m))


def clearance_to_obstacle(point_xy: tuple[float, float] | np.ndarray, obstacle: PrimitiveObstacle) -> float:
    point = np.asarray(point_xy, dtype=np.float64).reshape(2)
    if isinstance(obstacle, CircleObstacle):
        center = np.asarray(obstacle.center, dtype=np.float64)
        return float(np.linalg.norm(point - center) - float(obstacle.radius_m))
    center = np.asarray(obstacle.center, dtype=np.float64)
    half = 0.5 * np.asarray(obstacle.size_m, dtype=np.float64)
    delta = np.abs(point - center) - half
    outside = float(np.linalg.norm(np.maximum(delta, 0.0)))
    inside = float(min(max(delta[0], delta[1]), 0.0))
    return outside + inside


def _scan_scene_geometry(
    scene: Scene2D,
    *,
    origin: np.ndarray,
    directions: np.ndarray,
    time_s: float,
    max_range_m: float,
) -> np.ndarray:
    circles, boxes = _split_obstacles(scene.obstacles_at(time_s))
    circle_centers, circle_radii = circles
    box_centers, box_sizes = boxes
    return _scan_obstacle_batches(
        origin=origin,
        directions=directions,
        bounds_x_m=scene.bounds_x_m,
        bounds_y_m=scene.bounds_y_m,
        circle_centers=circle_centers,
        circle_radii=circle_radii,
        box_centers=box_centers,
        box_sizes=box_sizes,
        max_range_m=float(max_range_m),
    )


def _scan_obstacle_batches(
    *,
    origin: np.ndarray,
    directions: np.ndarray,
    bounds_x_m: tuple[float, float],
    bounds_y_m: tuple[float, float],
    circle_centers: np.ndarray,
    circle_radii: np.ndarray,
    box_centers: np.ndarray,
    box_sizes: np.ndarray,
    max_range_m: float,
) -> np.ndarray:
    ranges = _raycast_bounds_batch(
        origin,
        directions,
        bounds_x_m,
        bounds_y_m,
        float(max_range_m),
    )
    if circle_centers.size:
        circle_hits = _raycast_circles_batch(origin, directions, circle_centers, circle_radii, float(max_range_m))
        ranges = np.minimum(ranges, np.min(circle_hits, axis=1))
    if box_centers.size:
        box_hits = _raycast_aabbs_batch(origin, directions, box_centers, box_sizes, float(max_range_m))
        ranges = np.minimum(ranges, np.min(box_hits, axis=1))
    return np.minimum(ranges, float(max_range_m))


@lru_cache(maxsize=32)
def _ray_offsets(num_rays: int, fov_rad: float) -> np.ndarray:
    return np.linspace(-0.5 * float(fov_rad), 0.5 * float(fov_rad), int(num_rays), endpoint=False, dtype=np.float64)


def _ray_directions(num_rays: int, fov_rad: float, yaw_rad: float) -> np.ndarray:
    angles = float(yaw_rad) + _ray_offsets(int(num_rays), float(fov_rad))
    return np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float64, copy=False)


def _split_obstacles(
    obstacles: Sequence[PrimitiveObstacle],
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    circle_centers: list[tuple[float, float]] = []
    circle_radii: list[float] = []
    box_centers: list[tuple[float, float]] = []
    box_sizes: list[tuple[float, float]] = []
    for obstacle in obstacles:
        if isinstance(obstacle, CircleObstacle):
            circle_centers.append(obstacle.center)
            circle_radii.append(float(obstacle.radius_m))
        else:
            box_centers.append(obstacle.center)
            box_sizes.append(obstacle.size_m)
    circles = (
        np.asarray(circle_centers, dtype=np.float64).reshape(-1, 2),
        np.asarray(circle_radii, dtype=np.float64),
    )
    boxes = (
        np.asarray(box_centers, dtype=np.float64).reshape(-1, 2),
        np.asarray(box_sizes, dtype=np.float64).reshape(-1, 2),
    )
    return circles, boxes


def _raycast_bounds_batch(
    origin: np.ndarray,
    directions: np.ndarray,
    bounds_x: tuple[float, float],
    bounds_y: tuple[float, float],
    max_range_m: float,
) -> np.ndarray:
    ox, oy = float(origin[0]), float(origin[1])
    dx = directions[:, 0]
    dy = directions[:, 1]
    xmin, xmax = float(bounds_x[0]), float(bounds_x[1])
    ymin, ymax = float(bounds_y[0]), float(bounds_y[1])
    inf = np.full(directions.shape[0], np.inf, dtype=np.float64)
    candidates: list[np.ndarray] = []

    for x in (xmin, xmax):
        valid_direction = np.abs(dx) >= _EPS
        t = np.divide(float(x) - ox, dx, out=inf.copy(), where=valid_direction)
        y = oy + t * dy
        valid_hit = valid_direction & (t >= 0.0) & (t <= max_range_m) & (y >= ymin - _EPS) & (y <= ymax + _EPS)
        candidates.append(np.where(valid_hit, t, np.inf))

    for y in (ymin, ymax):
        valid_direction = np.abs(dy) >= _EPS
        t = np.divide(float(y) - oy, dy, out=inf.copy(), where=valid_direction)
        x = ox + t * dx
        valid_hit = valid_direction & (t >= 0.0) & (t <= max_range_m) & (x >= xmin - _EPS) & (x <= xmax + _EPS)
        candidates.append(np.where(valid_hit, t, np.inf))

    best = np.min(np.stack(candidates, axis=1), axis=1)
    return np.where(np.isfinite(best), best, float(max_range_m))


def _raycast_circles_batch(
    origin: np.ndarray,
    directions: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    max_range_m: float,
) -> np.ndarray:
    oc = origin.reshape(1, 2) - centers.reshape(-1, 2)
    b = 2.0 * np.einsum("rd,cd->rc", directions, oc, optimize=True)
    c = np.sum(oc * oc, axis=1) - radii * radii
    disc = b * b - 4.0 * c.reshape(1, -1)
    valid_disc = disc >= 0.0
    root = np.sqrt(np.maximum(disc, 0.0))
    t1 = (-b - root) / 2.0
    t2 = (-b + root) / 2.0
    t1 = np.where(valid_disc & (t1 >= 0.0) & (t1 <= max_range_m), t1, np.inf)
    t2 = np.where(valid_disc & (t2 >= 0.0) & (t2 <= max_range_m), t2, np.inf)
    return np.minimum(t1, t2)


def _raycast_aabbs_batch(
    origin: np.ndarray,
    directions: np.ndarray,
    centers: np.ndarray,
    sizes: np.ndarray,
    max_range_m: float,
) -> np.ndarray:
    half = 0.5 * sizes
    min_xy = centers - half
    max_xy = centers + half
    near_values: list[np.ndarray] = []
    far_values: list[np.ndarray] = []
    valid = np.ones((directions.shape[0], centers.shape[0]), dtype=bool)

    for axis in range(2):
        direction = directions[:, axis].reshape(-1, 1)
        origin_axis = float(origin[axis])
        slab_min = min_xy[:, axis].reshape(1, -1)
        slab_max = max_xy[:, axis].reshape(1, -1)
        parallel = np.abs(direction) < _EPS
        inside_parallel_slab = (origin_axis >= slab_min) & (origin_axis <= slab_max)
        valid &= (~parallel) | inside_parallel_slab

        t1 = np.divide(slab_min - origin_axis, direction, out=np.zeros_like(valid, dtype=np.float64), where=~parallel)
        t2 = np.divide(slab_max - origin_axis, direction, out=np.zeros_like(valid, dtype=np.float64), where=~parallel)
        near = np.minimum(t1, t2)
        far = np.maximum(t1, t2)
        near = np.where(parallel, -np.inf, near)
        far = np.where(parallel, np.inf, far)
        near_values.append(near)
        far_values.append(far)

    t_near = np.maximum.reduce([near_values[0], near_values[1], np.zeros_like(near_values[0])])
    t_far = np.minimum.reduce([far_values[0], far_values[1], np.full_like(far_values[0], float(max_range_m))])
    hit = valid & (t_far >= t_near) & (t_near >= 0.0) & (t_near <= max_range_m)
    return np.where(hit, t_near, np.inf)


def _clearances_to_obstacles(point: np.ndarray, obstacles: Sequence[PrimitiveObstacle]) -> np.ndarray:
    circles, boxes = _split_obstacles(obstacles)
    pieces: list[np.ndarray] = []
    circle_centers, circle_radii = circles
    if circle_centers.size:
        pieces.append(np.linalg.norm(circle_centers - point.reshape(1, 2), axis=1) - circle_radii)

    box_centers, box_sizes = boxes
    if box_centers.size:
        half = 0.5 * box_sizes
        delta = np.abs(point.reshape(1, 2) - box_centers) - half
        outside = np.linalg.norm(np.maximum(delta, 0.0), axis=1)
        inside = np.minimum(np.maximum(delta[:, 0], delta[:, 1]), 0.0)
        pieces.append(outside + inside)

    if not pieces:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(pieces)
