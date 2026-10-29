from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from dugm_r.simulation_2d.path import PathGuidanceConfig, build_path_guidance, query_cost_to_go
from dugm_r.simulation_2d.scene import PrimitiveObstacle, RectRegion


@dataclass(frozen=True)
class ReachabilityResult:
    reachable: bool
    path_length_m: float | None
    straight_distance_m: float
    grid_resolution_m: float
    visited_cells: int
    planner_status: str = "ok"

    @property
    def detour_ratio(self) -> float | None:
        if self.path_length_m is None or self.straight_distance_m <= 1e-9:
            return None
        return float(self.path_length_m / self.straight_distance_m)


def validate_reachable(
    *,
    bounds_x_m: tuple[float, float],
    bounds_y_m: tuple[float, float],
    obstacles: Iterable[PrimitiveObstacle],
    floor_regions: Iterable[RectRegion] | None = None,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    robot_radius_m: float,
    resolution_m: float = 0.25,
    clearance_margin_m: float = 0.08,
) -> ReachabilityResult:
    straight_distance = float(np.linalg.norm(np.asarray(goal_xy, dtype=np.float64) - np.asarray(start_xy, dtype=np.float64)))
    guidance = build_path_guidance(
        bounds_x_m=bounds_x_m,
        bounds_y_m=bounds_y_m,
        floor_regions=floor_regions,
        static_obstacles=obstacles,
        start_xy=start_xy,
        goal_xy=goal_xy,
        robot_radius_m=robot_radius_m,
        config=PathGuidanceConfig(
            resolution_m=float(resolution_m),
            clearance_margin_m=float(clearance_margin_m),
        ),
    )
    if guidance is None:
        return ReachabilityResult(
            False,
            None,
            straight_distance,
            float(resolution_m),
            0,
            planner_status="unreachable",
        )
    path_length = query_cost_to_go(guidance, start_xy)
    return ReachabilityResult(
        path_length is not None,
        path_length,
        straight_distance,
        guidance.grid.resolution_m,
        guidance.visited_cells,
        planner_status="ok" if path_length is not None else "query_failed",
    )
