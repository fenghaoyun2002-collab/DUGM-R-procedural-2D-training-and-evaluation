from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import numpy as np

from dugm_r.simulation_2d.scene import (
    BoxObstacle,
    CircleObstacle,
    PrimitiveObstacle,
    RectRegion,
)


_NEIGHBORS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


@dataclass(frozen=True)
class GridSpec:
    bounds_x_m: tuple[float, float]
    bounds_y_m: tuple[float, float]
    resolution_m: float
    width: int
    height: int


@dataclass(frozen=True)
class PathGuidanceConfig:
    resolution_m: float = 0.25
    clearance_margin_m: float = 0.15
    waypoint_min_distance_m: float = 1.0
    waypoint_nominal_distance_m: float = 1.5
    waypoint_max_distance_m: float = 2.0
    dense_clearance_threshold_m: float = 0.75
    sparse_clearance_threshold_m: float = 1.50
    finite_search_radius_cells: int = 3


@dataclass
class PathGuidance:
    grid: GridSpec
    blocked: np.ndarray
    clearance_m: np.ndarray
    cost_to_go_m: np.ndarray
    goal_cell: tuple[int, int]
    path_cells: tuple[tuple[int, int], ...]
    path_xy: np.ndarray
    static_obstacles: tuple[PrimitiveObstacle, ...]
    floor_regions: tuple[RectRegion, ...]
    inflated_radius_m: float
    config: PathGuidanceConfig
    visited_cells: int


def build_path_guidance(
    *,
    bounds_x_m: tuple[float, float],
    bounds_y_m: tuple[float, float],
    floor_regions: Iterable[RectRegion] | None,
    static_obstacles: Iterable[PrimitiveObstacle],
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    robot_radius_m: float,
    config: PathGuidanceConfig,
) -> PathGuidance | None:
    resolution = max(1e-3, float(config.resolution_m))
    width = int(math.ceil((bounds_x_m[1] - bounds_x_m[0]) / resolution))
    height = int(math.ceil((bounds_y_m[1] - bounds_y_m[0]) / resolution))
    if width <= 0 or height <= 0:
        return None
    grid = GridSpec(
        bounds_x_m=(float(bounds_x_m[0]), float(bounds_x_m[1])),
        bounds_y_m=(float(bounds_y_m[0]), float(bounds_y_m[1])),
        resolution_m=resolution,
        width=width,
        height=height,
    )
    obstacles = tuple(static_obstacles)
    regions = tuple(floor_regions or ())
    inflated_radius = float(robot_radius_m) + max(0.0, float(config.clearance_margin_m))
    blocked = np.zeros((height, width), dtype=np.bool_)
    clearance = np.full((height, width), np.inf, dtype=np.float64)
    for y in range(height):
        for x in range(width):
            point = cell_to_world(grid, (x, y))
            shell_clearance = _free_space_clearance(
                point,
                bounds_x_m=grid.bounds_x_m,
                bounds_y_m=grid.bounds_y_m,
                floor_regions=regions,
                obstacles=obstacles,
            )
            clearance[y, x] = shell_clearance
            blocked[y, x] = shell_clearance <= inflated_radius

    start_clearance = _free_space_clearance(
        start_xy,
        bounds_x_m=grid.bounds_x_m,
        bounds_y_m=grid.bounds_y_m,
        floor_regions=regions,
        obstacles=obstacles,
    )
    goal_clearance = _free_space_clearance(
        goal_xy,
        bounds_x_m=grid.bounds_x_m,
        bounds_y_m=grid.bounds_y_m,
        floor_regions=regions,
        obstacles=obstacles,
    )
    if start_clearance <= inflated_radius or goal_clearance <= inflated_radius:
        return None
    start_cell = _reachable_anchor_cell(
        grid,
        blocked,
        point_xy=start_xy,
        bounds_x_m=grid.bounds_x_m,
        bounds_y_m=grid.bounds_y_m,
        floor_regions=regions,
        obstacles=obstacles,
        inflated_radius_m=inflated_radius,
    )
    goal_cell = _reachable_anchor_cell(
        grid,
        blocked,
        point_xy=goal_xy,
        bounds_x_m=grid.bounds_x_m,
        bounds_y_m=grid.bounds_y_m,
        floor_regions=regions,
        obstacles=obstacles,
        inflated_radius_m=inflated_radius,
    )
    if start_cell is None or goal_cell is None:
        return None

    costs = np.full((height, width), np.inf, dtype=np.float64)
    costs[goal_cell[1], goal_cell[0]] = 0.0
    queue: list[tuple[float, int, int]] = [(0.0, goal_cell[0], goal_cell[1])]
    visited = 0
    while queue:
        cost, x, y = heapq.heappop(queue)
        if cost > costs[y, x] + 1e-12:
            continue
        visited += 1
        cell = (x, y)
        for neighbor, step_cost in _iter_free_neighbors(cell, blocked, resolution):
            nx, ny = neighbor
            candidate_cost = cost + step_cost
            if candidate_cost + 1e-12 >= costs[ny, nx]:
                continue
            costs[ny, nx] = candidate_cost
            heapq.heappush(queue, (candidate_cost, nx, ny))

    if not np.isfinite(costs[start_cell[1], start_cell[0]]):
        return None
    path_cells = _extract_descending_path(start_cell, goal_cell, blocked, costs, resolution)
    if not path_cells:
        return None
    path_xy = np.asarray([cell_to_world(grid, cell) for cell in path_cells], dtype=np.float64)
    return PathGuidance(
        grid=grid,
        blocked=blocked,
        clearance_m=clearance,
        cost_to_go_m=costs,
        goal_cell=goal_cell,
        path_cells=path_cells,
        path_xy=path_xy,
        static_obstacles=obstacles,
        floor_regions=regions,
        inflated_radius_m=inflated_radius,
        config=config,
        visited_cells=visited,
    )


def query_cost_to_go(guidance: PathGuidance, point_xy: tuple[float, float] | np.ndarray) -> float | None:
    grid = guidance.grid
    px, py = float(point_xy[0]), float(point_xy[1])
    gx = (px - grid.bounds_x_m[0]) / grid.resolution_m - 0.5
    gy = (py - grid.bounds_y_m[0]) / grid.resolution_m - 0.5
    x0 = math.floor(gx)
    y0 = math.floor(gy)
    tx = gx - x0
    ty = gy - y0
    weighted_cost = 0.0
    weight_sum = 0.0
    for x, y, weight in (
        (x0, y0, (1.0 - tx) * (1.0 - ty)),
        (x0 + 1, y0, tx * (1.0 - ty)),
        (x0, y0 + 1, (1.0 - tx) * ty),
        (x0 + 1, y0 + 1, tx * ty),
    ):
        if 0 <= x < grid.width and 0 <= y < grid.height:
            value = float(guidance.cost_to_go_m[y, x])
            if np.isfinite(value) and weight > 0.0:
                weighted_cost += weight * value
                weight_sum += weight
    if weight_sum > 1e-12:
        return float(weighted_cost / weight_sum)

    center = world_to_cell(grid, (px, py))
    if center is None:
        return None
    nearest = _nearest_finite_cell(
        guidance.cost_to_go_m,
        center,
        max_radius=max(0, int(guidance.config.finite_search_radius_cells)),
    )
    if nearest is None:
        return None
    return float(guidance.cost_to_go_m[nearest[1], nearest[0]])


def select_local_waypoint(
    guidance: PathGuidance,
    *,
    point_xy: tuple[float, float] | np.ndarray,
    goal_xy: tuple[float, float] | np.ndarray,
) -> tuple[float, float] | None:
    point = np.asarray(point_xy, dtype=np.float64)
    goal = np.asarray(goal_xy, dtype=np.float64)
    if float(np.linalg.norm(goal - point)) <= guidance.config.waypoint_min_distance_m:
        goal_pair = (float(goal[0]), float(goal[1]))
        return goal_pair if segment_is_reachable(guidance, start_xy=point, end_xy=goal_pair) else None

    start_cell = world_to_cell(guidance.grid, point)
    if start_cell is None:
        return None
    if not np.isfinite(guidance.cost_to_go_m[start_cell[1], start_cell[0]]):
        start_cell = _nearest_finite_cell(
            guidance.cost_to_go_m,
            start_cell,
            max_radius=max(0, int(guidance.config.finite_search_radius_cells)),
        )
    if start_cell is None:
        return None
    local_path = _extract_descending_path(
        start_cell,
        guidance.goal_cell,
        guidance.blocked,
        guidance.cost_to_go_m,
        guidance.grid.resolution_m,
    )
    if not local_path:
        return None

    local_clearance = max(
        0.0,
        float(guidance.clearance_m[start_cell[1], start_cell[0]]) - guidance.inflated_radius_m,
    )
    if local_clearance <= guidance.config.dense_clearance_threshold_m:
        lookahead = guidance.config.waypoint_min_distance_m
    elif local_clearance >= guidance.config.sparse_clearance_threshold_m:
        lookahead = guidance.config.waypoint_max_distance_m
    else:
        lookahead = guidance.config.waypoint_nominal_distance_m
    lookahead = float(
        np.clip(
            lookahead,
            guidance.config.waypoint_min_distance_m,
            guidance.config.waypoint_max_distance_m,
        )
    )

    candidates: list[tuple[float, tuple[float, float]]] = []
    fallback: tuple[float, float] | None = None
    for cell in local_path[1:]:
        candidate = cell_to_world(guidance.grid, cell)
        distance = float(np.linalg.norm(np.asarray(candidate, dtype=np.float64) - point))
        if distance > guidance.config.waypoint_max_distance_m + 1e-9:
            break
        if not segment_is_reachable(guidance, start_xy=point, end_xy=candidate):
            continue
        fallback = candidate
        if distance >= guidance.config.waypoint_min_distance_m:
            candidates.append((abs(distance - lookahead), candidate))
    if candidates:
        candidates.sort(key=lambda item: (item[0], -float(np.linalg.norm(np.asarray(item[1]) - point))))
        return candidates[0][1]
    return fallback


def segment_is_reachable(
    guidance: PathGuidance,
    *,
    start_xy: tuple[float, float] | np.ndarray,
    end_xy: tuple[float, float] | np.ndarray,
) -> bool:
    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    distance = float(np.linalg.norm(end - start))
    samples = max(1, int(math.ceil(distance / max(1e-3, 0.5 * guidance.grid.resolution_m))))
    for index in range(samples + 1):
        point = start + (index / samples) * (end - start)
        clearance = _free_space_clearance(
            (float(point[0]), float(point[1])),
            bounds_x_m=guidance.grid.bounds_x_m,
            bounds_y_m=guidance.grid.bounds_y_m,
            floor_regions=guidance.floor_regions,
            obstacles=guidance.static_obstacles,
        )
        if clearance <= guidance.inflated_radius_m:
            return False
    return True


def world_to_cell(grid: GridSpec, point_xy: tuple[float, float] | np.ndarray) -> tuple[int, int] | None:
    x = int(math.floor((float(point_xy[0]) - grid.bounds_x_m[0]) / grid.resolution_m))
    y = int(math.floor((float(point_xy[1]) - grid.bounds_y_m[0]) / grid.resolution_m))
    if not (0 <= x < grid.width and 0 <= y < grid.height):
        return None
    return x, y


def cell_to_world(grid: GridSpec, cell: tuple[int, int]) -> tuple[float, float]:
    return (
        grid.bounds_x_m[0] + (cell[0] + 0.5) * grid.resolution_m,
        grid.bounds_y_m[0] + (cell[1] + 0.5) * grid.resolution_m,
    )


def _extract_descending_path(
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
    blocked: np.ndarray,
    costs: np.ndarray,
    resolution: float,
) -> tuple[tuple[int, int], ...]:
    if not np.isfinite(costs[start_cell[1], start_cell[0]]):
        return ()
    path = [start_cell]
    current = start_cell
    max_steps = int(blocked.size)
    for _ in range(max_steps):
        if current == goal_cell:
            return tuple(path)
        current_cost = float(costs[current[1], current[0]])
        candidates = [
            (float(costs[cell[1], cell[0]]), cell)
            for cell, _step_cost in _iter_free_neighbors(current, blocked, resolution)
            if np.isfinite(costs[cell[1], cell[0]]) and float(costs[cell[1], cell[0]]) < current_cost - 1e-9
        ]
        if not candidates:
            return ()
        candidates.sort(key=lambda item: (item[0], item[1][1], item[1][0]))
        current = candidates[0][1]
        path.append(current)
    return ()


def _iter_free_neighbors(
    cell: tuple[int, int],
    blocked: np.ndarray,
    resolution: float,
):
    width = blocked.shape[1]
    height = blocked.shape[0]
    x, y = cell
    for dx, dy in _NEIGHBORS:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < width and 0 <= ny < height):
            continue
        if blocked[ny, nx]:
            continue
        if dx and dy and (blocked[y, nx] or blocked[ny, x]):
            continue
        step = math.sqrt(2.0) * resolution if dx and dy else resolution
        yield (nx, ny), step


def _nearest_finite_cell(
    costs: np.ndarray,
    center: tuple[int, int],
    *,
    max_radius: int,
) -> tuple[int, int] | None:
    width = costs.shape[1]
    height = costs.shape[0]
    cx, cy = center
    best: tuple[float, tuple[int, int]] | None = None
    for radius in range(max_radius + 1):
        for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
            for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
                if max(abs(x - cx), abs(y - cy)) != radius:
                    continue
                if not np.isfinite(costs[y, x]):
                    continue
                candidate = (float((x - cx) ** 2 + (y - cy) ** 2), (x, y))
                if best is None or candidate < best:
                    best = candidate
        if best is not None:
            return best[1]
    return None


def _is_blocked(blocked: np.ndarray, cell: tuple[int, int]) -> bool:
    return bool(blocked[cell[1], cell[0]])


def _reachable_anchor_cell(
    grid: GridSpec,
    blocked: np.ndarray,
    *,
    point_xy: tuple[float, float],
    bounds_x_m: tuple[float, float],
    bounds_y_m: tuple[float, float],
    floor_regions: tuple[RectRegion, ...],
    obstacles: tuple[PrimitiveObstacle, ...],
    inflated_radius_m: float,
) -> tuple[int, int] | None:
    center = world_to_cell(grid, point_xy)
    if center is None:
        return None
    candidates: list[tuple[float, tuple[int, int]]] = []
    max_radius = 3
    for radius in range(max_radius + 1):
        for y in range(max(0, center[1] - radius), min(grid.height, center[1] + radius + 1)):
            for x in range(max(0, center[0] - radius), min(grid.width, center[0] + radius + 1)):
                if max(abs(x - center[0]), abs(y - center[1])) != radius or blocked[y, x]:
                    continue
                cell = (x, y)
                world = cell_to_world(grid, cell)
                distance = float(np.linalg.norm(np.asarray(world) - np.asarray(point_xy)))
                candidates.append((distance, cell))
        if candidates:
            break
    candidates.sort(key=lambda item: (item[0], item[1][1], item[1][0]))
    for _distance, cell in candidates:
        if _segment_has_clearance(
            point_xy,
            cell_to_world(grid, cell),
            sample_spacing_m=0.5 * grid.resolution_m,
            bounds_x_m=bounds_x_m,
            bounds_y_m=bounds_y_m,
            floor_regions=floor_regions,
            obstacles=obstacles,
            inflated_radius_m=inflated_radius_m,
        ):
            return cell
    return None


def _free_space_clearance(
    point: tuple[float, float],
    *,
    bounds_x_m: tuple[float, float],
    bounds_y_m: tuple[float, float],
    floor_regions: tuple[RectRegion, ...],
    obstacles: tuple[PrimitiveObstacle, ...],
) -> float:
    px, py = float(point[0]), float(point[1])
    if floor_regions and not _inside_any_region(point, floor_regions):
        return -float("inf")
    boundary_clearance = min(
        px - bounds_x_m[0],
        bounds_x_m[1] - px,
        py - bounds_y_m[0],
        bounds_y_m[1] - py,
    )
    obstacle_clearance = min((_clearance_to_obstacle(point, item) for item in obstacles), default=float("inf"))
    return float(min(boundary_clearance, obstacle_clearance))


def _segment_has_clearance(
    start_xy: tuple[float, float] | np.ndarray,
    end_xy: tuple[float, float] | np.ndarray,
    *,
    sample_spacing_m: float,
    bounds_x_m: tuple[float, float],
    bounds_y_m: tuple[float, float],
    floor_regions: tuple[RectRegion, ...],
    obstacles: tuple[PrimitiveObstacle, ...],
    inflated_radius_m: float,
) -> bool:
    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    distance = float(np.linalg.norm(end - start))
    samples = max(1, int(math.ceil(distance / max(1e-3, sample_spacing_m))))
    for index in range(samples + 1):
        point = start + (index / samples) * (end - start)
        if _free_space_clearance(
            (float(point[0]), float(point[1])),
            bounds_x_m=bounds_x_m,
            bounds_y_m=bounds_y_m,
            floor_regions=floor_regions,
            obstacles=obstacles,
        ) <= inflated_radius_m:
            return False
    return True


def _inside_any_region(point: tuple[float, float], regions: tuple[RectRegion, ...]) -> bool:
    px, py = float(point[0]), float(point[1])
    for region in regions:
        half_x = 0.5 * region.size_m[0]
        half_y = 0.5 * region.size_m[1]
        if (
            region.center[0] - half_x <= px <= region.center[0] + half_x
            and region.center[1] - half_y <= py <= region.center[1] + half_y
        ):
            return True
    return False


def _clearance_to_obstacle(point: tuple[float, float], obstacle: PrimitiveObstacle) -> float:
    px, py = float(point[0]), float(point[1])
    cx, cy = obstacle.center
    if isinstance(obstacle, CircleObstacle):
        return float(math.hypot(px - cx, py - cy) - obstacle.radius_m)
    if not isinstance(obstacle, BoxObstacle):
        raise TypeError(f"Unsupported obstacle type: {type(obstacle)!r}")
    half_x = 0.5 * obstacle.size_m[0]
    half_y = 0.5 * obstacle.size_m[1]
    dx = abs(px - cx) - half_x
    dy = abs(py - cy) - half_y
    outside = math.hypot(max(dx, 0.0), max(dy, 0.0))
    inside = min(max(dx, dy), 0.0)
    return float(outside + inside)
