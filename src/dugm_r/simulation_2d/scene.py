from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


DEFAULT_CURRICULUM_PATH = Path(__file__).resolve().parents[1] / "configs" / "curriculum_2d.yaml"


@dataclass(frozen=True)
class CircleObstacle:
    center: tuple[float, float]
    radius_m: float
    kind: str = "circle"


@dataclass(frozen=True)
class BoxObstacle:
    center: tuple[float, float]
    size_m: tuple[float, float]
    kind: str = "box"


@dataclass(frozen=True)
class DynamicMotionSegment:
    start_time_s: float
    end_time_s: float
    start_center: tuple[float, float]
    velocity: tuple[float, float]


@dataclass(frozen=True)
class RectRegion:
    center: tuple[float, float]
    size_m: tuple[float, float]
    kind: str
    name: str = ""

    def sample(self, rng: np.random.Generator, *, margin_m: float = 0.0) -> tuple[float, float]:
        half_x = max(0.05, 0.5 * self.size_m[0] - float(margin_m))
        half_y = max(0.05, 0.5 * self.size_m[1] - float(margin_m))
        return (
            float(rng.uniform(self.center[0] - half_x, self.center[0] + half_x)),
            float(rng.uniform(self.center[1] - half_y, self.center[1] + half_y)),
        )


@dataclass(frozen=True)
class DynamicActor:
    kind: str
    center: tuple[float, float]
    velocity: tuple[float, float]
    radius_m: float | None = None
    size_m: tuple[float, float] | None = None
    foot_radius_m: float | None = None
    foot_separation_m: float | None = None
    trajectory: tuple[DynamicMotionSegment, ...] | None = None


PrimitiveObstacle = CircleObstacle | BoxObstacle


@dataclass(frozen=True)
class FloorPlan:
    variant: str
    regions: tuple[RectRegion, ...]
    walls: tuple[PrimitiveObstacle, ...]
    dynamic_lanes: tuple[RectRegion, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    start_regions: tuple[RectRegion, ...] = field(default_factory=tuple)
    goal_regions: tuple[RectRegion, ...] = field(default_factory=tuple)
    fixed_obstacles_by_id: dict[str, PrimitiveObstacle] = field(default_factory=dict)


@dataclass(frozen=True)
class Scene2D:
    level: int
    level_name: str
    bounds_x_m: tuple[float, float]
    bounds_y_m: tuple[float, float]
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    static_obstacles: tuple[PrimitiveObstacle, ...] = field(default_factory=tuple)
    dynamic_actors: tuple[DynamicActor, ...] = field(default_factory=tuple)
    floor_regions: tuple[RectRegion, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def dynamic_obstacles(self, time_s: float) -> tuple[PrimitiveObstacle, ...]:
        primitives: list[PrimitiveObstacle] = []
        for actor in self.dynamic_actors:
            center, velocity = _dynamic_actor_state(actor, time_s, self.bounds_x_m, self.bounds_y_m)
            primitives.extend(_dynamic_actor_primitives(actor, center=center, velocity=velocity))
        return tuple(primitives)

    def obstacles_at(self, time_s: float = 0.0) -> tuple[PrimitiveObstacle, ...]:
        return self.static_obstacles + self.dynamic_obstacles(time_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "level_name": self.level_name,
            "bounds_x_m": list(self.bounds_x_m),
            "bounds_y_m": list(self.bounds_y_m),
            "start_xy": list(self.start_xy),
            "goal_xy": list(self.goal_xy),
            "floor_regions": [
                {"name": item.name, "kind": item.kind, "center": list(item.center), "size_m": list(item.size_m)}
                for item in self.floor_regions
            ],
            "static_obstacles": [_primitive_to_dict(item) for item in self.static_obstacles],
            "dynamic_actors": [
                {
                    "kind": item.kind,
                    "center": list(item.center),
                    "velocity": list(item.velocity),
                    "radius_m": item.radius_m,
                    "size_m": list(item.size_m) if item.size_m is not None else None,
                    "foot_radius_m": item.foot_radius_m,
                    "foot_separation_m": item.foot_separation_m,
                }
                for item in self.dynamic_actors
            ],
            "metadata": dict(self.metadata),
        }


def load_curriculum_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = (Path(path) if path is not None else DEFAULT_CURRICULUM_PATH).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    extends = data.pop("extends", None)
    if extends is not None:
        base_path = Path(str(extends))
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        data = _deep_merge(load_curriculum_config(base_path), data)
    data["_config_path"] = str(config_path)
    data["_config_dir"] = str(config_path.parent)
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def load_fixed_scene_set(path: str | Path, *, base_dir: str | Path | None = None) -> dict[str, Any]:
    scene_set_path = Path(path)
    if not scene_set_path.is_absolute() and base_dir is not None:
        scene_set_path = Path(base_dir) / scene_set_path
    scene_set_path = scene_set_path.resolve()
    with scene_set_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if isinstance(data, list):
        data = {"scenes": data}
    data["_scene_set_path"] = str(scene_set_path)
    data["_scene_set_dir"] = str(scene_set_path.parent)
    return data


class CurriculumSceneGenerator:
    def __init__(self, config: dict[str, Any] | None = None, *, robot_radius_m: float | None = None) -> None:
        self.config = config if config is not None else load_curriculum_config()
        world = dict(self.config.get("world", {}))
        bounds = dict(world.get("bounds_m", {}))
        self.bounds_x_m = _float_pair(bounds.get("x", [-15.0, 15.0]))
        self.bounds_y_m = _float_pair(bounds.get("y", [-15.0, 15.0]))
        self.robot_radius_m = float(robot_radius_m if robot_radius_m is not None else world.get("robot_radius_m", 0.089))
        self.start_goal_distance_m = _float_pair(world.get("start_goal_distance_m", [5.0, 14.0]))
        self.start_goal_clearance_m = float(world.get("start_goal_clearance_m", 0.7))
        self.start_clearance_buckets_m = tuple(
            _float_pair(item) for item in world.get("start_clearance_buckets_m", ())
        )
        raw_bucket_weights = tuple(float(item) for item in world.get("start_clearance_bucket_weights", ()))
        if self.start_clearance_buckets_m:
            if raw_bucket_weights and len(raw_bucket_weights) != len(self.start_clearance_buckets_m):
                raise ValueError("start_clearance_bucket_weights must match start_clearance_buckets_m.")
            weights = raw_bucket_weights or tuple(1.0 for _ in self.start_clearance_buckets_m)
            weight_array = np.asarray(weights, dtype=np.float64)
            if np.any(weight_array < 0.0) or float(np.sum(weight_array)) <= 0.0:
                raise ValueError("start_clearance_bucket_weights must contain a positive total weight.")
            self.start_clearance_bucket_weights = tuple((weight_array / np.sum(weight_array)).tolist())
        else:
            self.start_clearance_bucket_weights = ()
        self.goal_clearance_m = _float_pair(world.get("goal_clearance_m", [self.start_goal_clearance_m] * 2))
        self.config_dir = Path(str(self.config.get("_config_dir", DEFAULT_CURRICULUM_PATH.parent)))
        self.levels = {int(key): self._resolve_level_config(dict(value)) for key, value in dict(self.config.get("levels", {})).items()}
        indoor = dict(self.config.get("indoor_generation", {}))
        self.validator_config = dict(indoor.get("validator", {}))
        self.max_scene_attempts = int(indoor.get("max_scene_attempts", 80))

    @classmethod
    def from_yaml(cls, path: str | Path | None = None, *, robot_radius_m: float | None = None) -> "CurriculumSceneGenerator":
        return cls(load_curriculum_config(path), robot_radius_m=robot_radius_m)

    def sample(self, level: int, *, seed: int | None = None) -> Scene2D:
        if int(level) not in self.levels:
            raise ValueError(f"Unknown curriculum level {level!r}; available levels: {sorted(self.levels)}")
        rng = np.random.default_rng(seed)
        level_config = self.levels[int(level)]
        bounds_x_m, bounds_y_m = self._level_bounds(level_config)
        fixed_scenes = list(level_config.get("fixed_scenes", []))
        if fixed_scenes:
            return self._sample_fixed_scene(
                int(level),
                level_config,
                fixed_scenes,
                seed=seed,
                rng=rng,
                bounds_x_m=bounds_x_m,
                bounds_y_m=bounds_y_m,
            )
        last_error: Exception | None = None
        for _ in range(self.max_scene_attempts):
            try:
                floor_plan = self._sample_floor_plan(level_config, rng)
                static_obstacles: list[PrimitiveObstacle] = list(floor_plan.walls)
                density_field = _ValueNoiseField(rng, bounds_x_m, bounds_y_m)
                static_obstacle_metadata = self._sample_semantic_static_obstacles(
                    level_config,
                    rng,
                    density_field,
                    floor_plan,
                    static_obstacles,
                    bounds_x_m=bounds_x_m,
                    bounds_y_m=bounds_y_m,
                )
                start_xy, goal_xy, clearance_metadata = self._sample_start_goal(
                    rng,
                    tuple(static_obstacles),
                    floor_plan,
                    level_config,
                    bounds_x_m=bounds_x_m,
                    bounds_y_m=bounds_y_m,
                )
                zone_static_metadata = self._sample_authored_static_obstacle_zones(
                    level_config,
                    rng,
                    floor_plan,
                    static_obstacles,
                    protected_endpoints={"start": start_xy, "goal": goal_xy},
                    bounds_x_m=bounds_x_m,
                    bounds_y_m=bounds_y_m,
                )
                static_obstacle_metadata.update(zone_static_metadata)
                if clearance_metadata.get("start_goal_sampling_mode") == "fixed_obstacle_edge_annulus":
                    start_nearest_clearance = _point_clearance(start_xy, tuple(static_obstacles))
                    goal_nearest_clearance = _point_clearance(goal_xy, tuple(static_obstacles))
                    clearance_metadata.update(
                        {
                            "sampled_start_nearest_edge_clearance_m": float(start_nearest_clearance),
                            "sampled_goal_nearest_edge_clearance_m": float(goal_nearest_clearance),
                            "sampled_start_clearance_m": float(start_nearest_clearance - self.robot_radius_m),
                            "sampled_goal_clearance_m": float(goal_nearest_clearance - self.robot_radius_m),
                        }
                    )
                path_obstacle_metadata = self._sample_path_static_obstacles(
                    level_config,
                    rng,
                    floor_plan,
                    static_obstacles,
                    start_xy,
                    goal_xy,
                    bounds_x_m=bounds_x_m,
                    bounds_y_m=bounds_y_m,
                )
                if path_obstacle_metadata["path_static_obstacle_count"] > 0:
                    clearance_metadata = self._fixed_scene_clearance_metadata(
                        start_xy,
                        goal_xy,
                        tuple(static_obstacles),
                        bounds_x_m=bounds_x_m,
                        bounds_y_m=bounds_y_m,
                    )
                dynamic_actors = self._sample_dynamic_actors(
                    level_config,
                    rng,
                    floor_plan,
                    tuple(static_obstacles),
                    start_xy,
                    goal_xy,
                    bounds_x_m=bounds_x_m,
                    bounds_y_m=bounds_y_m,
                )
                scene = Scene2D(
                    level=int(level),
                    level_name=str(level_config.get("name", f"level_{level}")),
                    bounds_x_m=bounds_x_m,
                    bounds_y_m=bounds_y_m,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                    static_obstacles=tuple(static_obstacles),
                    dynamic_actors=tuple(dynamic_actors),
                    floor_regions=floor_plan.regions,
                    metadata={
                        "floorplan_variant": floor_plan.variant,
                        "wall_variant": floor_plan.variant,
                        "density_seeded": True,
                        "require_turning_route": bool(level_config.get("require_turning_route", False)),
                        "dynamic_actor_count": len(dynamic_actors),
                        "dynamic_actor_kinds": sorted({actor.kind for actor in dynamic_actors}),
                        **static_obstacle_metadata,
                        **path_obstacle_metadata,
                        **clearance_metadata,
                        **floor_plan.metadata,
                    },
                )
                reachability = self._validate_scene(scene)
                if not reachability["reachable"]:
                    continue
                if _turning_route_detour_too_short(level_config, floor_plan, reachability):
                    continue
                scene.metadata.update(reachability)
                return scene
            except Exception as exc:  # pragma: no cover - defensive rejection path
                last_error = exc
                continue
        if last_error is not None:
            raise RuntimeError(f"Failed to sample valid indoor curriculum scene: {last_error}") from last_error
        raise RuntimeError("Failed to sample valid indoor curriculum scene.")

    def _level_bounds(self, level_config: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
        level_world = dict(level_config.get("world", {}))
        bounds = dict(level_world.get("bounds_m", {}))
        return (
            _float_pair(bounds.get("x", self.bounds_x_m)),
            _float_pair(bounds.get("y", self.bounds_y_m)),
        )

    def _resolve_level_config(self, level_config: dict[str, Any]) -> dict[str, Any]:
        definition_path = level_config.get("floorplan_definition_path")
        if definition_path is not None:
            path = Path(str(definition_path))
            if not path.is_absolute():
                path = self.config_dir / path
            path = path.resolve()
            with path.open("r", encoding="utf-8") as handle:
                definition = yaml.safe_load(handle) or {}
            if not isinstance(definition, dict):
                raise ValueError(f"Floorplan definition {definition_path!r} must be a mapping.")
            level_config["floorplan_definition"] = definition
            level_config["floorplan_definition_path"] = str(path)
        scene_set_path = level_config.get("fixed_scene_set_path") or level_config.get("fixed_scene_set")
        if scene_set_path is None:
            return level_config
        scene_set = load_fixed_scene_set(scene_set_path, base_dir=self.config_dir)
        scenes = list(scene_set.get("scenes", []))
        if not scenes:
            raise ValueError(f"Fixed scene set {scene_set_path!r} has no scenes.")
        resolved = dict(level_config)
        resolved["fixed_scenes"] = scenes
        resolved["fixed_scene_set_id"] = str(scene_set.get("scene_set_id", scene_set.get("id", Path(str(scene_set_path)).stem)))
        resolved["fixed_scene_set_path"] = str(scene_set["_scene_set_path"])
        return resolved

    def _sample_fixed_scene(
        self,
        level: int,
        level_config: dict[str, Any],
        fixed_scenes: list[Any],
        *,
        seed: int | None,
        rng: np.random.Generator,
        bounds_x_m: tuple[float, float],
        bounds_y_m: tuple[float, float],
    ) -> Scene2D:
        if seed is not None:
            index = int(seed) % len(fixed_scenes)
        else:
            index = int(rng.integers(0, len(fixed_scenes)))
        item = dict(fixed_scenes[index])
        variant = str(item.get("floorplan", level_config.get("floorplans", ["straight_corridor"])[0]))
        corridor_width = float(item.get("corridor_width_m", _uniform_range(rng, level_config.get("corridor_width_m", [2.2, 3.4]))))
        wall_thickness = float(item.get("wall_thickness_m", level_config.get("wall_thickness_m", 0.28)))
        floor_plan = self._fixed_floor_plan(
            variant,
            item,
            level_config=level_config,
            corridor_width=corridor_width,
            wall_thickness=wall_thickness,
        )
        static_obstacles = list(floor_plan.walls)
        static_obstacles.extend(_parse_static_obstacles(item.get("static_obstacles", ())))
        dynamic_actors = tuple(_parse_dynamic_actors(item.get("dynamic_actors", ())))
        dynamic_actors = tuple(
            self._build_dynamic_motion(
                level_config,
                dynamic_actors,
                tuple(static_obstacles),
                bounds_x_m=bounds_x_m,
                bounds_y_m=bounds_y_m,
            )
        )
        scene_id = str(item.get("scene_id", item.get("id", f"fixed_{index:02d}")))
        scene = Scene2D(
            level=int(level),
            level_name=str(level_config.get("name", f"level_{level}")),
            bounds_x_m=bounds_x_m,
            bounds_y_m=bounds_y_m,
            start_xy=_float_pair(item["start_xy"]),
            goal_xy=_float_pair(item["goal_xy"]),
            static_obstacles=tuple(static_obstacles),
            dynamic_actors=dynamic_actors,
            floor_regions=floor_plan.regions,
            metadata={
                "fixed_scene": True,
                "fixed_scene_set_id": str(level_config.get("fixed_scene_set_id", "")),
                "fixed_scene_set_path": str(level_config.get("fixed_scene_set_path", "")),
                "fixed_scene_index": index,
                "fixed_scene_id": scene_id,
                "scene_id": scene_id,
                "floorplan_variant": floor_plan.variant,
                "wall_variant": floor_plan.variant,
                **self._fixed_scene_clearance_metadata(
                    _float_pair(item["start_xy"]),
                    _float_pair(item["goal_xy"]),
                    tuple(static_obstacles),
                    bounds_x_m=bounds_x_m,
                    bounds_y_m=bounds_y_m,
                ),
                **_scene_item_metadata(item),
                **floor_plan.metadata,
            },
        )
        reachability = self._validate_scene(scene)
        guidance_enabled = bool(dict(self.config.get("path_guidance", {})).get("enabled", False))
        if not reachability["reachable"] and not guidance_enabled:
            raise RuntimeError(f"Fixed scene {scene.metadata['fixed_scene_id']!r} is not reachable.")
        scene.metadata.update(reachability)
        return scene

    def _fixed_floor_plan(
        self,
        variant: str,
        item: dict[str, Any],
        *,
        level_config: dict[str, Any],
        corridor_width: float,
        wall_thickness: float,
    ) -> FloorPlan:
        if variant == "authored_rectilinear":
            return self._authored_rectilinear(level_config)
        if variant == "open_room":
            room = RectRegion(center=(0.0, 0.0), size_m=_float_pair(item.get("room_size_m", [14.0, 10.0])), kind="open_room")
            return FloorPlan(
                variant=variant,
                regions=(room,),
                walls=tuple(_boundary_walls((room,), wall_thickness)),
                dynamic_lanes=(room,),
            )
        if variant == "straight_corridor":
            return self._straight_corridor(corridor_width, wall_thickness)
        if variant == "l_corridor":
            return self._l_corridor(corridor_width, wall_thickness)
        if variant == "t_junction":
            return self._t_junction(corridor_width, wall_thickness)
        if variant == "side_rooms":
            return self._side_rooms(corridor_width, wall_thickness)
        raise ValueError(f"Unsupported fixed scene floorplan variant: {variant!r}")

    def _sample_floor_plan(self, level_config: dict[str, Any], rng: np.random.Generator) -> FloorPlan:
        variants = list(level_config.get("floorplans", ["open_room"]))
        variant = str(rng.choice(variants))
        corridor_width = _uniform_range(rng, level_config.get("corridor_width_m", [2.2, 3.4]))
        wall_thickness = float(level_config.get("wall_thickness_m", 0.28))
        if variant == "authored_rectilinear":
            return self._authored_rectilinear(level_config)
        if variant == "open_room":
            region = RectRegion(center=(0.0, 0.0), size_m=(25.0, 19.0), kind="open_room")
            return FloorPlan(
                variant=variant,
                regions=(region,),
                walls=tuple(_boundary_walls((region,), wall_thickness)),
                dynamic_lanes=(region,),
            )
        if variant == "straight_corridor":
            return self._straight_corridor(corridor_width, wall_thickness)
        if variant == "l_corridor":
            return self._l_corridor(corridor_width, wall_thickness)
        if variant == "t_junction":
            return self._t_junction(corridor_width, wall_thickness)
        if variant == "side_rooms":
            return self._side_rooms(corridor_width, wall_thickness)
        raise ValueError(f"Unsupported indoor floorplan variant: {variant!r}")

    def _authored_rectilinear(self, level_config: dict[str, Any]) -> FloorPlan:
        definition = dict(level_config.get("floorplan_definition", {}))
        definition_id = str(definition.get("floorplan_id", "authored_rectilinear"))
        if not bool(definition.get("ready_for_sampling", False)):
            status = str(definition.get("status", "geometry_pending"))
            raise RuntimeError(
                f"Floorplan {definition_id!r} is a non-training scaffold ({status}); "
                "surveyed walkable regions and walls must be confirmed before sampling."
            )

        transform = dict(definition.get("transform", {}))
        translation = _float_pair(transform.get("translation_xy_m", [0.0, 0.0]))
        rotation_deg = float(transform.get("rotation_deg", 0.0))
        quarter_turns = int(round(rotation_deg / 90.0))
        if not math.isclose(rotation_deg, 90.0 * quarter_turns, abs_tol=1e-6):
            raise ValueError("authored_rectilinear transform.rotation_deg must be a multiple of 90 degrees.")
        quarter_turns %= 4

        regions: list[RectRegion] = []
        dynamic_lanes: list[RectRegion] = []
        start_regions: list[RectRegion] = []
        goal_regions: list[RectRegion] = []
        for index, raw_region in enumerate(definition.get("walkable_regions", ())):
            item = dict(raw_region)
            center = _transform_rectilinear_point(
                _float_pair(item["center_xy_m"]),
                translation=translation,
                quarter_turns=quarter_turns,
            )
            size = _transform_rectilinear_size(_float_pair(item["size_xy_m"]), quarter_turns=quarter_turns)
            region = RectRegion(
                center=center,
                size_m=size,
                kind=str(item.get("kind", "walkable")),
                name=str(item.get("id", f"region_{index:02d}")),
            )
            sampling = dict(item.get("sampling", {}))
            regions.append(region)
            if bool(sampling.get("dynamic_actor", True)):
                dynamic_lanes.append(region)
            if bool(sampling.get("start", True)):
                start_regions.append(region)
            if bool(sampling.get("goal", True)):
                goal_regions.append(region)
        if not regions:
            raise ValueError(f"Floorplan {definition_id!r} must define at least one walkable region.")

        parameters = dict(definition.get("parameters", {}))
        default_thickness = float(
            parameters.get("wall_thickness_m", level_config.get("wall_thickness_m", 0.28))
        )
        walls = list(_boundary_walls(tuple(regions), default_thickness)) if bool(definition.get("auto_boundary_walls", False)) else []
        for index, raw_wall in enumerate(definition.get("wall_segments", ())):
            item = dict(raw_wall)
            start = _transform_rectilinear_point(
                _float_pair(item["start_xy_m"]),
                translation=translation,
                quarter_turns=quarter_turns,
            )
            end = _transform_rectilinear_point(
                _float_pair(item["end_xy_m"]),
                translation=translation,
                quarter_turns=quarter_turns,
            )
            thickness = float(item.get("thickness_m", default_thickness))
            walls.append(
                _axis_aligned_wall_segment(
                    start,
                    end,
                    thickness_m=thickness,
                    kind=str(item.get("kind", "wall")),
                    name=str(item.get("id", f"wall_{index:02d}")),
                )
            )
        fixed_obstacle_count = 0
        fixed_obstacles_by_id: dict[str, PrimitiveObstacle] = {}
        for index, raw_obstacle in enumerate(definition.get("fixed_obstacles", ())):
            item = dict(raw_obstacle)
            obstacle_id = str(item.get("id", f"fixed_obstacle_{index:02d}"))
            if obstacle_id in fixed_obstacles_by_id:
                raise ValueError(f"Duplicate fixed floorplan obstacle id {obstacle_id!r}.")
            shape = str(item.get("shape", "box")).lower()
            center = _transform_rectilinear_point(
                _float_pair(item["center_xy_m"]),
                translation=translation,
                quarter_turns=quarter_turns,
            )
            if shape == "box":
                size = _transform_rectilinear_size(
                    _float_pair(item["size_xy_m"]),
                    quarter_turns=quarter_turns,
                )
                obstacle: PrimitiveObstacle = BoxObstacle(
                    center=center,
                    size_m=size,
                    kind=str(item.get("kind", "fixed_obstacle")),
                )
            elif shape == "circle":
                radius_m = float(item["radius_m"])
                if radius_m <= 0.0:
                    raise ValueError(
                        f"Fixed floorplan obstacle {item.get('id', index)!r} must have a positive radius_m."
                    )
                obstacle = CircleObstacle(
                    center=center,
                    radius_m=radius_m,
                    kind=str(item.get("kind", "fixed_obstacle")),
                )
            else:
                raise ValueError(
                    f"Fixed floorplan obstacle {item.get('id', index)!r} uses unsupported shape {shape!r}; "
                    "authored_rectilinear fixed obstacles support box and circle."
                )
            walls.append(obstacle)
            fixed_obstacles_by_id[obstacle_id] = obstacle
            fixed_obstacle_count += 1
        if not walls:
            raise ValueError(
                f"Floorplan {definition_id!r} has no walls; enable auto_boundary_walls or add wall_segments."
            )

        return FloorPlan(
            variant="authored_rectilinear",
            regions=tuple(regions),
            walls=tuple(_dedupe_walls(walls)),
            dynamic_lanes=tuple(dynamic_lanes or regions),
            metadata={
                "floorplan_definition_id": definition_id,
                "floorplan_definition_path": str(level_config.get("floorplan_definition_path", "")),
                "floorplan_definition_status": str(definition.get("status", "ready")),
                "floorplan_translation_xy_m": list(translation),
                "floorplan_rotation_deg": 90.0 * quarter_turns,
                "floorplan_fixed_obstacle_count": fixed_obstacle_count,
            },
            start_regions=tuple(start_regions or regions),
            goal_regions=tuple(goal_regions or regions),
            fixed_obstacles_by_id=fixed_obstacles_by_id,
        )

    def _straight_corridor(self, width: float, thickness: float) -> FloorPlan:
        corridor = RectRegion(center=(0.0, 0.0), size_m=(24.0, width), kind="corridor")
        walls = tuple(_boundary_walls((corridor,), thickness))
        return FloorPlan("straight_corridor", (corridor,), walls, (corridor,), {"corridor_width_m": width})

    def _l_corridor(self, width: float, thickness: float) -> FloorPlan:
        h = RectRegion(center=(-5.2, 0.0), size_m=(15.5, width), kind="corridor")
        v = RectRegion(center=(0.0, 5.3), size_m=(width, 15.5), kind="corridor")
        walls = tuple(_boundary_walls((h, v), thickness))
        return FloorPlan("l_corridor", (h, v), walls, (h, v), {"corridor_width_m": width})

    def _t_junction(self, width: float, thickness: float) -> FloorPlan:
        h = RectRegion(center=(0.0, 0.0), size_m=(24.0, width), kind="corridor")
        v = RectRegion(center=(0.0, 5.7), size_m=(width, 12.4), kind="corridor")
        walls = tuple(_boundary_walls((h, v), thickness))
        return FloorPlan("t_junction", (h, v), walls, (h, v), {"corridor_width_m": width})

    def _side_rooms(self, width: float, thickness: float) -> FloorPlan:
        corridor = RectRegion(center=(0.0, 0.0), size_m=(24.0, width), kind="corridor")
        door_width = 1.55
        connector_depth = 0.9
        room_size = (5.3, 5.1)
        top_connector = RectRegion(
            center=(-6.4, 0.5 * width + 0.5 * connector_depth),
            size_m=(door_width, connector_depth),
            kind="door",
        )
        bottom_connector = RectRegion(
            center=(5.8, -0.5 * width - 0.5 * connector_depth),
            size_m=(door_width, connector_depth),
            kind="door",
        )
        room_a = RectRegion(
            center=(top_connector.center[0], top_connector.center[1] + 0.5 * connector_depth + 0.5 * room_size[1]),
            size_m=room_size,
            kind="side_room",
        )
        room_b = RectRegion(
            center=(bottom_connector.center[0], bottom_connector.center[1] - 0.5 * connector_depth - 0.5 * room_size[1]),
            size_m=(5.5, 5.1),
            kind="side_room",
        )
        regions = (corridor, top_connector, room_a, bottom_connector, room_b)
        walls = tuple(_boundary_walls(regions, thickness))
        dynamic_lanes = (corridor, room_a, room_b)
        return FloorPlan("side_rooms", regions, walls, dynamic_lanes, {"corridor_width_m": width, "door_width_m": door_width})

    def _sample_semantic_static_obstacles(
        self,
        level_config: dict[str, Any],
        rng: np.random.Generator,
        density_field: "_ValueNoiseField",
        floor_plan: FloorPlan,
        out: list[PrimitiveObstacle],
        *,
        bounds_x_m: tuple[float, float],
        bounds_y_m: tuple[float, float],
    ) -> dict[str, Any]:
        box_count = _randint_range(rng, level_config.get("static_boxes", [0, 0]))
        circle_count = _randint_range(rng, level_config.get("static_circles", [0, 0]))
        box_size_range = _float_pair(level_config.get("box_size_m", [0.5, 1.4]))
        circle_radius_range = _float_pair(level_config.get("circle_radius_m", [0.2, 0.55]))
        clutter_bias = float(level_config.get("noise_clutter_bias", 0.45))

        for _ in range(box_count):
            size = _semantic_box_size(rng, box_size_range)
            obstacle = self._sample_semantic_box(
                rng,
                density_field,
                floor_plan,
                tuple(out),
                size,
                clutter_bias,
                bounds_x_m=bounds_x_m,
                bounds_y_m=bounds_y_m,
            )
            if obstacle is not None:
                out.append(obstacle)
        for _ in range(circle_count):
            radius = float(rng.uniform(circle_radius_range[0], circle_radius_range[1]))
            obstacle = self._sample_semantic_circle(
                rng,
                density_field,
                floor_plan,
                tuple(out),
                radius,
                clutter_bias,
                bounds_x_m=bounds_x_m,
                bounds_y_m=bounds_y_m,
            )
            if obstacle is not None:
                out.append(obstacle)

        return {
            "semantic_static_box_count": int(box_count),
            "semantic_static_circle_count": int(circle_count),
        }

    def _sample_authored_static_obstacle_zones(
        self,
        level_config: dict[str, Any],
        rng: np.random.Generator,
        floor_plan: FloorPlan,
        out: list[PrimitiveObstacle],
        *,
        protected_endpoints: dict[str, tuple[float, float]],
        bounds_x_m: tuple[float, float],
        bounds_y_m: tuple[float, float],
    ) -> dict[str, Any]:
        definition = dict(level_config.get("floorplan_definition", {}))
        zone_specs = list(definition.get("static_obstacle_zones", ()))
        if not zone_specs:
            return {
                "zone_static_obstacle_count": 0,
                "zone_static_obstacle_counts_by_region": {},
            }

        region_by_id = {region.name: region for region in floor_plan.regions}
        counts_by_region: dict[str, int] = {}
        total_count = 0
        for raw_zone in zone_specs:
            zone = dict(raw_zone)
            zone_id = str(zone.get("id", "wall_attached_static"))
            region_ids = [str(value) for value in zone.get("region_ids", ())]
            if not region_ids:
                raise ValueError(f"Static obstacle zone {zone_id!r} must list region_ids.")
            shape = str(zone.get("shape", "circle")).lower()
            if shape != "circle":
                raise ValueError(
                    f"Static obstacle zone {zone_id!r} uses unsupported shape {shape!r}; "
                    "wall-attached authored zones currently support circles only."
                )
            radius_m = float(zone.get("radius_m", 0.25))
            if radius_m <= 0.0:
                raise ValueError(f"Static obstacle zone {zone_id!r} must use a positive radius_m.")
            placement = str(zone.get("placement", "tangent")).lower()
            if placement != "tangent":
                raise ValueError(f"Static obstacle zone {zone_id!r} only supports tangent placement.")
            allowed_attachment_kinds = {
                str(value) for value in zone.get("allowed_attachment_kinds", ["wall", "fixed_obstacle"])
            }
            attachment_anchors = tuple(
                obstacle for obstacle in floor_plan.walls if obstacle.kind in allowed_attachment_kinds
            )
            if not attachment_anchors:
                raise ValueError(
                    f"Static obstacle zone {zone_id!r} has no anchors matching "
                    f"{sorted(allowed_attachment_kinds)}."
                )
            surface_gap_m = float(zone.get("surface_gap_m", 0.0))
            if surface_gap_m < 0.0:
                raise ValueError(f"Static obstacle zone {zone_id!r} must use a non-negative surface_gap_m.")
            endpoint_clearance_m = float(zone.get("min_endpoint_edge_clearance_m", 0.0))
            if endpoint_clearance_m < 0.0:
                raise ValueError(
                    f"Static obstacle zone {zone_id!r} must use a non-negative min_endpoint_edge_clearance_m."
                )
            protected_roles = tuple(
                str(value) for value in zone.get("protected_endpoint_roles", ("start",))
            )
            unknown_roles = set(protected_roles).difference(protected_endpoints)
            if unknown_roles:
                raise ValueError(
                    f"Static obstacle zone {zone_id!r} references unknown protected endpoint roles: "
                    f"{sorted(unknown_roles)}"
                )
            protected_points = tuple(protected_endpoints[role] for role in protected_roles)
            count_range = zone.get("count_per_region", zone.get("count", [0, 0]))
            max_attempts = int(zone.get("max_attempts_per_obstacle", 2400))
            require_exact = bool(zone.get("require_exact_count_per_region", True))
            if max_attempts <= 0:
                raise ValueError(f"Static obstacle zone {zone_id!r} must use positive max_attempts_per_obstacle.")

            for region_id in region_ids:
                if region_id not in region_by_id:
                    raise ValueError(
                        f"Static obstacle zone {zone_id!r} references unknown region {region_id!r}."
                    )
                region = region_by_id[region_id]
                target_count = _randint_range(rng, count_range)
                placed_count = 0
                for _ in range(target_count):
                    candidate = self._sample_wall_attached_circle(
                        rng,
                        region,
                        attachment_anchors,
                        tuple(out),
                        radius_m=radius_m,
                        surface_gap_m=surface_gap_m,
                        protected_points=protected_points,
                        min_endpoint_edge_clearance_m=endpoint_clearance_m,
                        max_attempts=max_attempts,
                        bounds_x_m=bounds_x_m,
                        bounds_y_m=bounds_y_m,
                    )
                    if candidate is None:
                        if require_exact:
                            raise RuntimeError(
                                f"Failed to place all {target_count} wall-attached circles "
                                f"in region {region_id!r} for zone {zone_id!r}."
                            )
                        break
                    out.append(candidate)
                    placed_count += 1
                counts_by_region[region_id] = counts_by_region.get(region_id, 0) + placed_count
                total_count += placed_count

        return {
            "zone_static_obstacle_count": int(total_count),
            "zone_static_obstacle_counts_by_region": counts_by_region,
            "zone_static_obstacle_attachment": "tangent_to_authored_wall_or_fixed_obstacle",
            "zone_static_min_endpoint_edge_clearance_m": endpoint_clearance_m,
            "zone_static_protected_endpoint_roles": list(protected_roles),
        }

    def _sample_wall_attached_circle(
        self,
        rng: np.random.Generator,
        region: RectRegion,
        attachment_anchors: tuple[PrimitiveObstacle, ...],
        existing: tuple[PrimitiveObstacle, ...],
        *,
        radius_m: float,
        surface_gap_m: float,
        protected_points: tuple[tuple[float, float], ...],
        min_endpoint_edge_clearance_m: float,
        max_attempts: int,
        bounds_x_m: tuple[float, float],
        bounds_y_m: tuple[float, float],
    ) -> CircleObstacle | None:
        for _ in range(max_attempts):
            anchor = attachment_anchors[int(rng.integers(0, len(attachment_anchors)))]
            center = _sample_tangent_circle_center(rng, anchor, radius_m, surface_gap_m=surface_gap_m)
            candidate = CircleObstacle(
                center=center,
                radius_m=radius_m,
                kind="wall_attached_random_circle",
            )
            if not _circle_inside_region(candidate, region):
                continue
            if not _inside_bounds(center, bounds_x_m, bounds_y_m, radius_m):
                continue
            if any(
                _clearance_to_primitive(point, candidate) < min_endpoint_edge_clearance_m - 1e-9
                for point in protected_points
            ):
                continue
            collision = False
            for obstacle in existing:
                clearance = _primitive_clearance(candidate, obstacle)
                if obstacle is anchor:
                    if abs(clearance - surface_gap_m) > 1e-6:
                        collision = True
                        break
                elif clearance < -1e-8:
                    collision = True
                    break
            if not collision:
                return candidate
        return None

    def _sample_semantic_box(
        self,
        rng: np.random.Generator,
        density_field: "_ValueNoiseField",
        floor_plan: FloorPlan,
        existing: tuple[PrimitiveObstacle, ...],
        size: tuple[float, float],
        clutter_bias: float,
        *,
        bounds_x_m: tuple[float, float],
        bounds_y_m: tuple[float, float],
    ) -> BoxObstacle | None:
        for _ in range(180):
            region = _choose_region(rng, floor_plan.regions)
            point = _sample_semantic_point(rng, region, size, wall_aligned=True)
            if rng.random() > clutter_bias + 0.45 * density_field.value(point):
                continue
            candidate = BoxObstacle(center=point, size_m=size, kind="cabinet" if region.kind == "side_room" else "box")
            if _inside_bounds(point, bounds_x_m, bounds_y_m, 0.8) and _separated(candidate, existing, margin=0.35):
                return candidate
        return None

    def _sample_semantic_circle(
        self,
        rng: np.random.Generator,
        density_field: "_ValueNoiseField",
        floor_plan: FloorPlan,
        existing: tuple[PrimitiveObstacle, ...],
        radius: float,
        clutter_bias: float,
        *,
        bounds_x_m: tuple[float, float],
        bounds_y_m: tuple[float, float],
    ) -> CircleObstacle | None:
        for _ in range(180):
            region = _choose_region(rng, floor_plan.regions)
            point = region.sample(rng, margin_m=radius + 0.45)
            if rng.random() > clutter_bias + 0.45 * density_field.value(point):
                continue
            candidate = CircleObstacle(center=point, radius_m=radius, kind="pillar" if region.kind == "corridor" else "round_obstacle")
            if _inside_bounds(point, bounds_x_m, bounds_y_m, 0.8) and _separated(candidate, existing, margin=0.3):
                return candidate
        return None

    def _sample_path_static_obstacles(
        self,
        level_config: dict[str, Any],
        rng: np.random.Generator,
        floor_plan: FloorPlan,
        out: list[PrimitiveObstacle],
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        *,
        bounds_x_m: tuple[float, float],
        bounds_y_m: tuple[float, float],
    ) -> dict[str, Any]:
        count_range = level_config.get("path_static_obstacles", [0, 0])
        target_count = _randint_range(rng, count_range)
        if target_count <= 0:
            return {"path_static_obstacle_count": 0, "path_static_obstacle_target_count": 0}
        size_range = _float_pair(level_config.get("path_obstacle_size_m", level_config.get("box_size_m", [0.45, 1.20])))
        radius_range = _float_pair(level_config.get("path_obstacle_radius_m", level_config.get("circle_radius_m", [0.18, 0.40])))
        t_range = _float_pair(level_config.get("path_obstacle_t_m", [0.35, 0.70]))
        lateral_range = _float_pair(level_config.get("path_obstacle_lateral_offset_m", [-0.45, 0.45]))
        lateral_abs_range = _optional_float_pair(level_config.get("path_obstacle_lateral_abs_offset_m"))
        obstacle_shape = str(level_config.get("path_obstacle_shape", "mixed")).lower()
        start = np.asarray(start_xy, dtype=np.float64)
        goal = np.asarray(goal_xy, dtype=np.float64)
        delta = goal - start
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-6:
            return {"path_static_obstacle_count": 0, "path_static_obstacle_target_count": int(target_count)}
        direction = delta / distance
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        added: list[PrimitiveObstacle] = []
        min_anchor_distance = float(level_config.get("path_obstacle_min_anchor_distance_m", 1.2))
        for _ in range(int(target_count)):
            obstacle = self._sample_one_path_static_obstacle(
                rng,
                floor_plan,
                tuple(out),
                start,
                goal,
                direction,
                normal,
                distance,
                size_range=size_range,
                radius_range=radius_range,
                t_range=t_range,
                lateral_range=lateral_range,
                lateral_abs_range=lateral_abs_range,
                min_anchor_distance=min_anchor_distance,
                obstacle_shape=obstacle_shape,
                bounds_x_m=bounds_x_m,
                bounds_y_m=bounds_y_m,
            )
            if obstacle is not None:
                out.append(obstacle)
                added.append(obstacle)
        return {
            "path_static_obstacle_count": int(len(added)),
            "path_static_obstacle_target_count": int(target_count),
            "path_static_obstacle_centers": [[float(item.center[0]), float(item.center[1])] for item in added],
        }

    def _sample_one_path_static_obstacle(
        self,
        rng: np.random.Generator,
        floor_plan: FloorPlan,
        existing: tuple[PrimitiveObstacle, ...],
        start: np.ndarray,
        goal: np.ndarray,
        direction: np.ndarray,
        normal: np.ndarray,
        distance: float,
        *,
        size_range: tuple[float, float],
        radius_range: tuple[float, float],
        t_range: tuple[float, float],
        lateral_range: tuple[float, float],
        lateral_abs_range: tuple[float, float] | None,
        min_anchor_distance: float,
        obstacle_shape: str,
        bounds_x_m: tuple[float, float],
        bounds_y_m: tuple[float, float],
    ) -> PrimitiveObstacle | None:
        if obstacle_shape not in {"mixed", "circle", "pillar", "box"}:
            raise ValueError(f"Unsupported path_obstacle_shape {obstacle_shape!r}; expected mixed, circle, pillar, or box.")
        for _ in range(240):
            t = float(rng.uniform(max(0.05, t_range[0]), min(0.95, t_range[1])))
            if lateral_abs_range is None:
                lateral = float(rng.uniform(lateral_range[0], lateral_range[1]))
            else:
                lateral_abs = float(rng.uniform(max(0.0, lateral_abs_range[0]), max(0.0, lateral_abs_range[1])))
                lateral = lateral_abs * float(rng.choice([-1.0, 1.0]))
            center_np = start + t * distance * direction + lateral * normal
            center = (float(center_np[0]), float(center_np[1]))
            if not _inside_any_region(center, floor_plan.regions):
                continue
            if min(float(np.linalg.norm(center_np - start)), float(np.linalg.norm(center_np - goal))) < min_anchor_distance:
                continue
            use_box = obstacle_shape == "box" or (obstacle_shape == "mixed" and rng.random() < 0.55)
            if use_box:
                long_edge = float(rng.uniform(size_range[0], size_range[1]))
                short_edge = float(rng.uniform(max(0.25, 0.45 * size_range[0]), max(0.35, 0.65 * long_edge)))
                size = (long_edge, short_edge) if rng.random() < 0.5 else (short_edge, long_edge)
                candidate: PrimitiveObstacle = BoxObstacle(center=center, size_m=size, kind="path_box")
                margin = 0.35
            else:
                radius = float(rng.uniform(radius_range[0], radius_range[1]))
                candidate = CircleObstacle(center=center, radius_m=radius, kind="path_pillar")
                margin = 0.30
            if not _inside_bounds(center, bounds_x_m, bounds_y_m, 0.8):
                continue
            if not _separated(candidate, existing, margin=margin):
                continue
            if _clearance_to_primitive(tuple(start.tolist()), candidate) < min_anchor_distance:
                continue
            if _clearance_to_primitive(tuple(goal.tolist()), candidate) < min_anchor_distance:
                continue
            return candidate
        return None

    def _sample_start_goal(
        self,
        rng: np.random.Generator,
        obstacles: tuple[PrimitiveObstacle, ...],
        floor_plan: FloorPlan,
        level_config: dict[str, Any],
        *,
        bounds_x_m: tuple[float, float],
        bounds_y_m: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float], dict[str, Any]]:
        require_turning_route = bool(level_config.get("require_turning_route", False))
        start_goal_distance_m = _float_pair(level_config.get("start_goal_distance_m", self.start_goal_distance_m))
        start_regions = floor_plan.start_regions or floor_plan.regions
        goal_regions = floor_plan.goal_regions or floor_plan.regions
        definition = dict(level_config.get("floorplan_definition", {}))
        authored_sampling = dict(definition.get("start_goal_sampling", level_config.get("start_goal_sampling", {})))
        if authored_sampling:
            mode = str(authored_sampling.get("mode", "")).lower()
            if mode != "fixed_obstacle_edge_annulus":
                raise ValueError(f"Unsupported authored start_goal_sampling mode {mode!r}.")
            return self._sample_fixed_obstacle_edge_start_goal(
                rng,
                obstacles,
                floor_plan,
                authored_sampling,
                start_goal_distance_m=start_goal_distance_m,
                start_regions=start_regions,
                goal_regions=goal_regions,
                bounds_x_m=bounds_x_m,
                bounds_y_m=bounds_y_m,
            )
        if self.start_clearance_buckets_m:
            bucket_index = int(rng.choice(len(self.start_clearance_buckets_m), p=self.start_clearance_bucket_weights))
            start_range = self.start_clearance_buckets_m[bucket_index]
            goal_range = self.goal_clearance_m
            minimum_margin = self.robot_radius_m + min(start_range[0], goal_range[0])
            for _ in range(2400):
                start_region = _choose_region(rng, start_regions)
                goal_region = _choose_region(rng, goal_regions)
                if not _turning_route_region_pair_allowed(floor_plan, start_region, goal_region, require_turning_route):
                    continue
                start = start_region.sample(rng, margin_m=minimum_margin)
                goal = goal_region.sample(rng, margin_m=minimum_margin)
                distance = float(np.linalg.norm(np.asarray(goal) - np.asarray(start)))
                if not (start_goal_distance_m[0] <= distance <= start_goal_distance_m[1]):
                    continue
                start_clearance = self._robot_shell_clearance(
                    start,
                    obstacles,
                    bounds_x_m=bounds_x_m,
                    bounds_y_m=bounds_y_m,
                )
                goal_clearance = self._robot_shell_clearance(
                    goal,
                    obstacles,
                    bounds_x_m=bounds_x_m,
                    bounds_y_m=bounds_y_m,
                )
                if not (start_range[0] <= start_clearance <= start_range[1]):
                    continue
                if not (goal_range[0] <= goal_clearance <= goal_range[1]):
                    continue
                return start, goal, {
                    "sampled_start_clearance_m": float(start_clearance),
                    "sampled_goal_clearance_m": float(goal_clearance),
                    "start_clearance_bucket": _clearance_bucket_name(bucket_index, start_range),
                    "goal_clearance_bucket": _clearance_range_name(goal_range),
                }
            raise RuntimeError("Failed to sample start/goal in configured clearance buckets.")

        point_margin = self.robot_radius_m + self.start_goal_clearance_m
        for _ in range(700):
            start_region = _choose_region(rng, start_regions)
            goal_region = _choose_region(rng, goal_regions)
            if not _turning_route_region_pair_allowed(floor_plan, start_region, goal_region, require_turning_route):
                continue
            start = start_region.sample(rng, margin_m=point_margin)
            goal = goal_region.sample(rng, margin_m=point_margin)
            distance = float(np.linalg.norm(np.asarray(goal) - np.asarray(start)))
            if not (start_goal_distance_m[0] <= distance <= start_goal_distance_m[1]):
                continue
            if _point_clearance(start, obstacles) < point_margin or _point_clearance(goal, obstacles) < point_margin:
                continue
            return start, goal, {
                "sampled_start_clearance_m": float(
                    self._robot_shell_clearance(
                        start,
                        obstacles,
                        bounds_x_m=bounds_x_m,
                        bounds_y_m=bounds_y_m,
                    )
                ),
                "sampled_goal_clearance_m": float(
                    self._robot_shell_clearance(
                        goal,
                        obstacles,
                        bounds_x_m=bounds_x_m,
                        bounds_y_m=bounds_y_m,
                    )
                ),
                "start_clearance_bucket": "legacy",
                "goal_clearance_bucket": "legacy",
            }
        raise RuntimeError("Failed to sample non-overlapping indoor start and goal.")

    def _sample_fixed_obstacle_edge_start_goal(
        self,
        rng: np.random.Generator,
        obstacles: tuple[PrimitiveObstacle, ...],
        floor_plan: FloorPlan,
        sampling: dict[str, Any],
        *,
        start_goal_distance_m: tuple[float, float],
        start_regions: tuple[RectRegion, ...],
        goal_regions: tuple[RectRegion, ...],
        bounds_x_m: tuple[float, float],
        bounds_y_m: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float], dict[str, Any]]:
        edge_clearance_m = _float_pair(sampling.get("edge_clearance_m", [0.2, 0.5]))
        if edge_clearance_m[0] < 0.0 or edge_clearance_m[1] < edge_clearance_m[0]:
            raise ValueError("start_goal_sampling.edge_clearance_m must be a non-negative ordered range.")
        eligible_ids = tuple(str(item) for item in sampling.get("eligible_obstacle_ids", ()))
        if not eligible_ids:
            raise ValueError("start_goal_sampling.eligible_obstacle_ids must not be empty.")
        missing_ids = tuple(item for item in eligible_ids if item not in floor_plan.fixed_obstacles_by_id)
        if missing_ids:
            raise ValueError(f"Unknown start/goal anchor obstacle ids: {missing_ids!r}.")

        forbidden_pairs: set[frozenset[str]] = set()
        for raw_pair in sampling.get("forbidden_obstacle_pairs", ()):
            pair = tuple(str(item) for item in raw_pair)
            if len(pair) != 2:
                raise ValueError("Each forbidden_obstacle_pairs entry must contain exactly two obstacle ids.")
            if pair[0] not in floor_plan.fixed_obstacles_by_id or pair[1] not in floor_plan.fixed_obstacles_by_id:
                raise ValueError(f"Unknown obstacle id in forbidden pair {pair!r}.")
            forbidden_pairs.add(frozenset(pair))

        max_attempts = int(sampling.get("max_attempts", 4000))
        boundary_margin = self.robot_radius_m
        for _ in range(max_attempts):
            start_anchor_id = eligible_ids[int(rng.integers(0, len(eligible_ids)))]
            goal_anchor_id = eligible_ids[int(rng.integers(0, len(eligible_ids)))]
            if frozenset((start_anchor_id, goal_anchor_id)) in forbidden_pairs:
                continue
            start_anchor = floor_plan.fixed_obstacles_by_id[start_anchor_id]
            goal_anchor = floor_plan.fixed_obstacles_by_id[goal_anchor_id]
            start = _sample_point_at_primitive_edge_clearance(rng, start_anchor, edge_clearance_m)
            goal = _sample_point_at_primitive_edge_clearance(rng, goal_anchor, edge_clearance_m)
            if not _inside_bounds(start, bounds_x_m, bounds_y_m, boundary_margin):
                continue
            if not _inside_bounds(goal, bounds_x_m, bounds_y_m, boundary_margin):
                continue
            if not _inside_any_region(start, start_regions) or not _inside_any_region(goal, goal_regions):
                continue

            start_nearest_clearance = _point_clearance(start, obstacles)
            goal_nearest_clearance = _point_clearance(goal, obstacles)
            if not (edge_clearance_m[0] <= start_nearest_clearance <= edge_clearance_m[1]):
                continue
            if not (edge_clearance_m[0] <= goal_nearest_clearance <= edge_clearance_m[1]):
                continue
            distance = float(np.linalg.norm(np.asarray(goal) - np.asarray(start)))
            if not (start_goal_distance_m[0] <= distance <= start_goal_distance_m[1]):
                continue

            start_anchor_clearance = _clearance_to_primitive(start, start_anchor)
            goal_anchor_clearance = _clearance_to_primitive(goal, goal_anchor)
            return start, goal, {
                "start_goal_sampling_mode": "fixed_obstacle_edge_annulus",
                "endpoint_clearance_reference": "robot_center_to_obstacle_edge",
                "start_anchor_obstacle_id": start_anchor_id,
                "goal_anchor_obstacle_id": goal_anchor_id,
                "sampled_start_anchor_edge_clearance_m": float(start_anchor_clearance),
                "sampled_goal_anchor_edge_clearance_m": float(goal_anchor_clearance),
                "sampled_start_nearest_edge_clearance_m": float(start_nearest_clearance),
                "sampled_goal_nearest_edge_clearance_m": float(goal_nearest_clearance),
                "sampled_start_clearance_m": float(start_nearest_clearance - self.robot_radius_m),
                "sampled_goal_clearance_m": float(goal_nearest_clearance - self.robot_radius_m),
                "start_clearance_bucket": _clearance_range_name(edge_clearance_m),
                "goal_clearance_bucket": _clearance_range_name(edge_clearance_m),
            }
        raise RuntimeError("Failed to sample start/goal in fixed-obstacle edge annuli.")

    def _robot_shell_clearance(
        self,
        point: tuple[float, float],
        obstacles: tuple[PrimitiveObstacle, ...],
        *,
        bounds_x_m: tuple[float, float],
        bounds_y_m: tuple[float, float],
    ) -> float:
        boundary_clearance = min(
            float(point[0]) - bounds_x_m[0],
            bounds_x_m[1] - float(point[0]),
            float(point[1]) - bounds_y_m[0],
            bounds_y_m[1] - float(point[1]),
        )
        center_clearance = min(boundary_clearance, _point_clearance(point, obstacles))
        return float(center_clearance - self.robot_radius_m)

    def _fixed_scene_clearance_metadata(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        obstacles: tuple[PrimitiveObstacle, ...],
        *,
        bounds_x_m: tuple[float, float],
        bounds_y_m: tuple[float, float],
    ) -> dict[str, Any]:
        start_clearance = self._robot_shell_clearance(
            start_xy,
            obstacles,
            bounds_x_m=bounds_x_m,
            bounds_y_m=bounds_y_m,
        )
        goal_clearance = self._robot_shell_clearance(
            goal_xy,
            obstacles,
            bounds_x_m=bounds_x_m,
            bounds_y_m=bounds_y_m,
        )
        return {
            "sampled_start_clearance_m": float(start_clearance),
            "sampled_goal_clearance_m": float(goal_clearance),
            "start_clearance_bucket": _matching_clearance_bucket(
                start_clearance,
                self.start_clearance_buckets_m,
            ),
            "goal_clearance_bucket": _clearance_range_name(self.goal_clearance_m),
        }

    def _sample_dynamic_actors(
        self,
        level_config: dict[str, Any],
        rng: np.random.Generator,
        floor_plan: FloorPlan,
        static_obstacles: tuple[PrimitiveObstacle, ...],
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        *,
        bounds_x_m: tuple[float, float],
        bounds_y_m: tuple[float, float],
    ) -> list[DynamicActor]:
        definition = dict(level_config.get("floorplan_definition", {}))
        sampling = dict(definition.get("dynamic_actor_sampling", {}))
        effective_config = dict(level_config)
        if sampling:
            if "count" in sampling:
                effective_config["dynamic_actors"] = sampling["count"]
            if "speed_mps" in sampling:
                effective_config["dynamic_speed_mps"] = sampling["speed_mps"]
            if "kinds" in sampling:
                effective_config["dynamic_actor_kinds"] = sampling["kinds"]
            if "require_exact_count" in sampling:
                effective_config["dynamic_actor_require_exact_count"] = sampling["require_exact_count"]

        count = _randint_range(rng, effective_config.get("dynamic_actors", [0, 0]))
        speed_range = _float_pair(effective_config.get("dynamic_speed_mps", [0.1, 0.3]))
        allowed_kinds = tuple(str(value).lower() for value in effective_config.get("dynamic_actor_kinds", ["person", "person", "cart"]))
        if not allowed_kinds or any(kind not in {"person", "cart"} for kind in allowed_kinds):
            raise ValueError("dynamic_actor_kinds must be a non-empty list containing only 'person' or 'cart'.")
        requested_region_ids = tuple(str(value) for value in sampling.get("region_ids", ()))
        if requested_region_ids:
            requested = set(requested_region_ids)
            dynamic_lanes = tuple(lane for lane in floor_plan.dynamic_lanes if lane.name in requested)
            missing = requested.difference(lane.name for lane in dynamic_lanes)
            if missing:
                raise ValueError(f"Dynamic actor sampling references unavailable regions: {sorted(missing)}")
        else:
            dynamic_lanes = floor_plan.dynamic_lanes
        if count > 0 and not dynamic_lanes:
            raise ValueError("Dynamic actor sampling requires at least one dynamic lane.")

        actors: list[DynamicActor] = []
        blocked_points = (start_xy, goal_xy)
        for _ in range(count):
            kind = str(rng.choice(allowed_kinds))
            radius = 0.45 if kind == "cart" else 0.35
            margin = radius + 0.18
            placed = False
            for _attempt in range(160):
                lane = _choose_region(rng, dynamic_lanes)
                center = lane.sample(rng, margin_m=margin)
                if kind == "cart" and lane.kind == "corridor":
                    center = _push_to_corridor_side(rng, center, lane, side_margin=0.45)
                if any(np.linalg.norm(np.asarray(center) - np.asarray(point)) < 1.4 for point in blocked_points):
                    continue
                if _point_clearance(center, static_obstacles) < margin:
                    continue
                if not _center_separated_from_dynamic_actors(center, radius, tuple(actors), margin=0.35):
                    continue
                speed = float(rng.uniform(speed_range[0], speed_range[1]))
                heading = _sample_dynamic_heading(rng, lane, effective_config, center=center, start_xy=start_xy, goal_xy=goal_xy)
                velocity = (speed * math.cos(heading), speed * math.sin(heading))
                if kind == "cart":
                    size = _float_pair(effective_config.get("cart_size_m", [0.9, 0.55]))
                    actors.append(DynamicActor(kind="cart", center=center, velocity=velocity, size_m=size))
                else:
                    actors.append(
                        DynamicActor(
                            kind="person",
                            center=center,
                            velocity=velocity,
                            foot_radius_m=float(effective_config.get("person_foot_radius_m", 0.14)),
                            foot_separation_m=float(effective_config.get("person_foot_separation_m", 0.34)),
                        )
                    )
                placed = True
                break
            if placed:
                continue
            fallback = self._sample_fallback_person_actor(
                rng,
                effective_config,
                dynamic_lanes,
                static_obstacles,
                blocked_points,
                speed_range,
                tuple(actors),
                start_xy,
                goal_xy,
            )
            if fallback is not None:
                actors.append(fallback)
        if bool(effective_config.get("dynamic_actor_require_exact_count", False)) and len(actors) != count:
            raise RuntimeError(f"Required exactly {count} dynamic actors, but placed {len(actors)}.")
        return self._build_dynamic_motion(
            effective_config,
            tuple(actors),
            static_obstacles,
            bounds_x_m=bounds_x_m,
            bounds_y_m=bounds_y_m,
        )

    def _sample_fallback_person_actor(
        self,
        rng: np.random.Generator,
        level_config: dict[str, Any],
        dynamic_lanes: tuple[RectRegion, ...],
        static_obstacles: tuple[PrimitiveObstacle, ...],
        blocked_points: tuple[tuple[float, float], ...],
        speed_range: tuple[float, float],
        existing_actors: tuple[DynamicActor, ...],
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
    ) -> DynamicActor | None:
        margin = 0.42
        for _ in range(220):
            lane = _choose_region(rng, dynamic_lanes)
            center = lane.sample(rng, margin_m=margin)
            if any(np.linalg.norm(np.asarray(center) - np.asarray(point)) < 1.2 for point in blocked_points):
                continue
            if _point_clearance(center, static_obstacles) < margin:
                continue
            if not _center_separated_from_dynamic_actors(center, 0.35, existing_actors, margin=0.35):
                continue
            speed = float(rng.uniform(speed_range[0], speed_range[1]))
            heading = _sample_dynamic_heading(rng, lane, level_config, center=center, start_xy=start_xy, goal_xy=goal_xy)
            return DynamicActor(
                kind="person",
                center=center,
                velocity=(speed * math.cos(heading), speed * math.sin(heading)),
                foot_radius_m=0.14,
                foot_separation_m=0.34,
            )
        return None

    def _build_dynamic_motion(
        self,
        level_config: dict[str, Any],
        actors: tuple[DynamicActor, ...] | list[DynamicActor],
        static_obstacles: tuple[PrimitiveObstacle, ...],
        *,
        bounds_x_m: tuple[float, float],
        bounds_y_m: tuple[float, float],
    ) -> list[DynamicActor]:
        actors = tuple(actors)
        if not actors or not bool(level_config.get("dynamic_reflection_motion", False)):
            return list(actors)
        episode = dict(self.config.get("episode", {}))
        episode_max_steps = int(level_config.get("max_steps", episode.get("max_steps", 1500)))
        horizon_s = float(
            level_config.get(
                "dynamic_motion_horizon_s",
                float(episode_max_steps) * float(episode.get("dt_s", 0.1)),
            )
        )
        step_s = float(level_config.get("dynamic_reflection_step_s", episode.get("dt_s", 0.1)))
        pause_s = float(level_config.get("dynamic_reflection_pause_s", 1.0))
        boundary_margin_m = float(level_config.get("dynamic_reflection_boundary_margin_m", 0.15))
        collision_margin_m = float(level_config.get("dynamic_reflection_collision_margin_m", 0.05))
        trajectories = _simulate_reflective_dynamic_trajectories(
            actors,
            static_obstacles=static_obstacles,
            bounds_x=bounds_x_m,
            bounds_y=bounds_y_m,
            horizon_s=horizon_s,
            step_s=step_s,
            pause_s=pause_s,
            boundary_margin_m=boundary_margin_m,
            collision_margin_m=collision_margin_m,
        )
        return [
            DynamicActor(
                kind=actor.kind,
                center=actor.center,
                velocity=actor.velocity,
                radius_m=actor.radius_m,
                size_m=actor.size_m,
                foot_radius_m=actor.foot_radius_m,
                foot_separation_m=actor.foot_separation_m,
                trajectory=tuple(trajectories[index]),
            )
            for index, actor in enumerate(actors)
        ]

    def _validate_scene(self, scene: Scene2D) -> dict[str, Any]:
        from dugm_r.simulation_2d.validator import validate_reachable

        episode = dict(self.config.get("episode", {}))
        validator_clearance_margin_m = float(self.validator_config.get("clearance_margin_m", 0.08))
        collision_clearance_margin_m = float(episode.get("collision_clearance_margin_m", 0.0))
        guidance_config = dict(self.config.get("path_guidance", {}))
        planner_safety_margin_m = (
            float(guidance_config.get("planner_safety_margin_m", 0.0))
            if bool(guidance_config.get("enabled", False))
            else 0.0
        )
        reachability_clearance_margin_m = max(
            validator_clearance_margin_m,
            collision_clearance_margin_m + planner_safety_margin_m,
        )
        result = validate_reachable(
            bounds_x_m=scene.bounds_x_m,
            bounds_y_m=scene.bounds_y_m,
            obstacles=scene.static_obstacles,
            floor_regions=scene.floor_regions,
            start_xy=scene.start_xy,
            goal_xy=scene.goal_xy,
            robot_radius_m=self.robot_radius_m,
            resolution_m=float(self.validator_config.get("grid_resolution_m", 0.25)),
            clearance_margin_m=reachability_clearance_margin_m,
        )
        max_detour = float(self.validator_config.get("max_detour_ratio", 3.5))
        reachable = result.reachable and (result.detour_ratio is None or result.detour_ratio <= max_detour)
        return {
            "reachable": bool(reachable),
            "path_length_m": result.path_length_m,
            "straight_distance_m": result.straight_distance_m,
            "detour_ratio": result.detour_ratio,
            "reachability_grid_resolution_m": result.grid_resolution_m,
            "reachability_clearance_margin_m": reachability_clearance_margin_m,
            "validator_clearance_margin_m": validator_clearance_margin_m,
            "collision_clearance_margin_m": collision_clearance_margin_m,
            "planner_safety_margin_m": planner_safety_margin_m,
            "reachability_visited_cells": result.visited_cells,
            "planner_status": result.planner_status,
        }


def _scene_item_metadata(item: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "id",
        "scene_id",
        "start_xy",
        "goal_xy",
        "room_size_m",
        "corridor_width_m",
        "wall_thickness_m",
        "static_obstacles",
        "dynamic_actors",
    }
    metadata = {str(key): value for key, value in item.items() if key not in excluded}
    tags = metadata.pop("tags", None)
    if isinstance(tags, dict):
        metadata.update(tags)
    return metadata


def _parse_static_obstacles(items: Iterable[Any]) -> list[PrimitiveObstacle]:
    obstacles: list[PrimitiveObstacle] = []
    for raw in items or ():
        item = dict(raw)
        kind = str(item.get("kind", "box"))
        if "radius_m" in item:
            obstacles.append(
                CircleObstacle(
                    center=_float_pair(item["center"]),
                    radius_m=float(item["radius_m"]),
                    kind=kind,
                )
            )
        else:
            obstacles.append(
                BoxObstacle(
                    center=_float_pair(item["center"]),
                    size_m=_float_pair(item.get("size_m", [0.5, 0.5])),
                    kind=kind,
                )
            )
    return obstacles


def _parse_dynamic_actors(items: Iterable[Any]) -> list[DynamicActor]:
    actors: list[DynamicActor] = []
    for raw in items or ():
        item = dict(raw)
        size = item.get("size_m")
        actors.append(
            DynamicActor(
                kind=str(item.get("kind", "person")),
                center=_float_pair(item["center"]),
                velocity=_float_pair(item.get("velocity", [0.0, 0.0])),
                radius_m=float(item["radius_m"]) if item.get("radius_m") is not None else None,
                size_m=_float_pair(size) if size is not None else None,
                foot_radius_m=float(item["foot_radius_m"]) if item.get("foot_radius_m") is not None else None,
                foot_separation_m=float(item["foot_separation_m"]) if item.get("foot_separation_m") is not None else None,
            )
        )
    return actors


class _ValueNoiseField:
    def __init__(
        self,
        rng: np.random.Generator,
        bounds_x_m: tuple[float, float],
        bounds_y_m: tuple[float, float],
        *,
        cells_x: int = 6,
        cells_y: int = 6,
    ) -> None:
        self.bounds_x_m = bounds_x_m
        self.bounds_y_m = bounds_y_m
        self.values = rng.random((cells_y + 1, cells_x + 1))

    def value(self, point: tuple[float, float]) -> float:
        x = (float(point[0]) - self.bounds_x_m[0]) / (self.bounds_x_m[1] - self.bounds_x_m[0])
        y = (float(point[1]) - self.bounds_y_m[0]) / (self.bounds_y_m[1] - self.bounds_y_m[0])
        x = float(np.clip(x, 0.0, 0.999))
        y = float(np.clip(y, 0.0, 0.999))
        gx = x * (self.values.shape[1] - 1)
        gy = y * (self.values.shape[0] - 1)
        ix = int(math.floor(gx))
        iy = int(math.floor(gy))
        tx = _smoothstep(gx - ix)
        ty = _smoothstep(gy - iy)
        v00 = self.values[iy, ix]
        v10 = self.values[iy, ix + 1]
        v01 = self.values[iy + 1, ix]
        v11 = self.values[iy + 1, ix + 1]
        return float((1 - ty) * ((1 - tx) * v00 + tx * v10) + ty * ((1 - tx) * v01 + tx * v11))


def _boundary_walls(
    regions: tuple[RectRegion, ...],
    thickness: float,
    *,
    step_m: float = 0.25,
) -> list[BoxObstacle]:
    """Build closed physical walls around the exposed boundary of a rectangle union."""

    walls: list[BoxObstacle] = []
    eps = 1e-3
    for region in regions:
        cx, cy = region.center
        sx, sy = region.size_m
        left = cx - 0.5 * sx
        right = cx + 0.5 * sx
        bottom = cy - 0.5 * sy
        top = cy + 0.5 * sy
        walls.extend(_edge_walls(regions, axis="x", fixed=left, low=bottom, high=top, normal=(-1.0, 0.0), thickness=thickness, step_m=step_m, eps=eps))
        walls.extend(_edge_walls(regions, axis="x", fixed=right, low=bottom, high=top, normal=(1.0, 0.0), thickness=thickness, step_m=step_m, eps=eps))
        walls.extend(_edge_walls(regions, axis="y", fixed=bottom, low=left, high=right, normal=(0.0, -1.0), thickness=thickness, step_m=step_m, eps=eps))
        walls.extend(_edge_walls(regions, axis="y", fixed=top, low=left, high=right, normal=(0.0, 1.0), thickness=thickness, step_m=step_m, eps=eps))
    return _dedupe_walls(walls)


def _edge_walls(
    regions: tuple[RectRegion, ...],
    *,
    axis: str,
    fixed: float,
    low: float,
    high: float,
    normal: tuple[float, float],
    thickness: float,
    step_m: float,
    eps: float,
) -> list[BoxObstacle]:
    span = float(high) - float(low)
    count = max(1, int(math.ceil(span / float(step_m))))
    edges = np.linspace(float(low), float(high), count + 1)
    exposed: list[tuple[float, float]] = []
    for start, end in zip(edges[:-1], edges[1:]):
        mid = 0.5 * (float(start) + float(end))
        if axis == "x":
            outside = (float(fixed) + normal[0] * eps, mid)
        else:
            outside = (mid, float(fixed) + normal[1] * eps)
        if not _inside_any_region(outside, regions):
            exposed.append((float(start), float(end)))
    merged = _merge_intervals(exposed, tolerance=1e-6)
    walls: list[BoxObstacle] = []
    for start, end in merged:
        length = max(0.01, float(end) - float(start))
        mid = 0.5 * (float(start) + float(end))
        if axis == "x":
            center = (float(fixed) + normal[0] * 0.5 * float(thickness), mid)
            size = (float(thickness), length)
        else:
            center = (mid, float(fixed) + normal[1] * 0.5 * float(thickness))
            size = (length, float(thickness))
        walls.append(BoxObstacle(center=center, size_m=size, kind="wall"))
    return walls


def _inside_any_region(point: tuple[float, float], regions: Iterable[RectRegion]) -> bool:
    px, py = float(point[0]), float(point[1])
    for region in regions:
        half_x = 0.5 * region.size_m[0]
        half_y = 0.5 * region.size_m[1]
        if (
            region.center[0] - half_x - 1e-9 <= px <= region.center[0] + half_x + 1e-9
            and region.center[1] - half_y - 1e-9 <= py <= region.center[1] + half_y + 1e-9
        ):
            return True
    return False


def _circle_inside_region(circle: CircleObstacle, region: RectRegion) -> bool:
    half_x = 0.5 * float(region.size_m[0]) - float(circle.radius_m)
    half_y = 0.5 * float(region.size_m[1]) - float(circle.radius_m)
    if half_x < 0.0 or half_y < 0.0:
        return False
    return bool(
        abs(float(circle.center[0]) - float(region.center[0])) <= half_x + 1e-9
        and abs(float(circle.center[1]) - float(region.center[1])) <= half_y + 1e-9
    )


def _sample_tangent_circle_center(
    rng: np.random.Generator,
    anchor: PrimitiveObstacle,
    radius_m: float,
    *,
    surface_gap_m: float = 0.0,
) -> tuple[float, float]:
    radius = float(radius_m)
    offset = radius + float(surface_gap_m)
    if isinstance(anchor, CircleObstacle):
        angle = float(rng.uniform(-math.pi, math.pi))
        distance = float(anchor.radius_m) + offset
        return (
            float(anchor.center[0] + distance * math.cos(angle)),
            float(anchor.center[1] + distance * math.sin(angle)),
        )

    half_x = 0.5 * float(anchor.size_m[0])
    half_y = 0.5 * float(anchor.size_m[1])
    # Weight faces by their usable length so long wall faces receive samples
    # in proportion to the available attachment surface.
    face_weights = np.asarray([half_y, half_y, half_x, half_x], dtype=np.float64)
    face = int(rng.choice(4, p=face_weights / np.sum(face_weights)))
    if face == 0:
        return (
            float(anchor.center[0] - half_x - offset),
            float(rng.uniform(anchor.center[1] - half_y, anchor.center[1] + half_y)),
        )
    if face == 1:
        return (
            float(anchor.center[0] + half_x + offset),
            float(rng.uniform(anchor.center[1] - half_y, anchor.center[1] + half_y)),
        )
    if face == 2:
        return (
            float(rng.uniform(anchor.center[0] - half_x, anchor.center[0] + half_x)),
            float(anchor.center[1] - half_y - offset),
        )
    return (
        float(rng.uniform(anchor.center[0] - half_x, anchor.center[0] + half_x)),
        float(anchor.center[1] + half_y + offset),
    )


def _merge_intervals(intervals: list[tuple[float, float]], *, tolerance: float) -> list[tuple[float, float]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals)
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if float(start) <= float(prev_end) + float(tolerance):
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _dedupe_walls(walls: list[PrimitiveObstacle]) -> list[PrimitiveObstacle]:
    seen: set[tuple[str, int, int, int, int]] = set()
    unique: list[PrimitiveObstacle] = []
    for wall in walls:
        if isinstance(wall, CircleObstacle):
            key = (
                "circle",
                int(round(wall.center[0] * 1000.0)),
                int(round(wall.center[1] * 1000.0)),
                int(round(wall.radius_m * 1000.0)),
                0,
            )
        else:
            key = (
                "box",
                int(round(wall.center[0] * 1000.0)),
                int(round(wall.center[1] * 1000.0)),
                int(round(wall.size_m[0] * 1000.0)),
                int(round(wall.size_m[1] * 1000.0)),
            )
        if key in seen:
            continue
        seen.add(key)
        unique.append(wall)
    return unique


def _transform_rectilinear_point(
    point: tuple[float, float],
    *,
    translation: tuple[float, float],
    quarter_turns: int,
) -> tuple[float, float]:
    x, y = float(point[0]), float(point[1])
    for _ in range(int(quarter_turns) % 4):
        x, y = -y, x
    return (x + float(translation[0]), y + float(translation[1]))


def _transform_rectilinear_size(
    size: tuple[float, float],
    *,
    quarter_turns: int,
) -> tuple[float, float]:
    sx, sy = float(size[0]), float(size[1])
    return (sy, sx) if int(quarter_turns) % 2 else (sx, sy)


def _axis_aligned_wall_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    thickness_m: float,
    kind: str,
    name: str,
) -> BoxObstacle:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    if math.isclose(dx, 0.0, abs_tol=1e-6) == math.isclose(dy, 0.0, abs_tol=1e-6):
        raise ValueError(f"Wall segment {name!r} must be a non-zero horizontal or vertical line.")
    thickness = float(thickness_m)
    if thickness <= 0.0:
        raise ValueError(f"Wall segment {name!r} thickness must be positive.")
    center = (0.5 * (float(start[0]) + float(end[0])), 0.5 * (float(start[1]) + float(end[1])))
    size = (abs(dx), thickness) if math.isclose(dy, 0.0, abs_tol=1e-6) else (thickness, abs(dy))
    return BoxObstacle(center=center, size_m=size, kind=kind)


def _float_pair(value: Any) -> tuple[float, float]:
    if value is None:
        raise ValueError("Expected a two-value range, got None.")
    return (float(value[0]), float(value[1]))


def _optional_float_pair(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    return _float_pair(value)


def _uniform_range(rng: np.random.Generator, value: Any) -> float:
    low, high = _float_pair(value)
    return float(rng.uniform(low, high))


def _randint_range(rng: np.random.Generator, value: Any) -> int:
    low, high = int(value[0]), int(value[1])
    return int(rng.integers(low, high + 1))


def _choose_region(rng: np.random.Generator, regions: tuple[RectRegion, ...]) -> RectRegion:
    areas = np.asarray([item.size_m[0] * item.size_m[1] for item in regions], dtype=np.float64)
    probs = areas / np.sum(areas)
    return regions[int(rng.choice(len(regions), p=probs))]


def _turning_route_region_pair_allowed(
    floor_plan: FloorPlan,
    start_region: RectRegion,
    goal_region: RectRegion,
    require_turning_route: bool,
) -> bool:
    if not require_turning_route:
        return True
    if floor_plan.variant not in {"l_corridor", "t_junction"}:
        return True
    if start_region.kind != "corridor" or goal_region.kind != "corridor":
        return False
    return start_region != goal_region


def _turning_route_detour_too_short(
    level_config: dict[str, Any],
    floor_plan: FloorPlan,
    reachability: dict[str, Any],
) -> bool:
    if not bool(level_config.get("require_turning_route", False)):
        return False
    if floor_plan.variant not in {"l_corridor", "t_junction"}:
        return False
    minimum_detour = float(level_config.get("turning_route_min_detour_ratio", 1.05))
    detour_ratio = reachability.get("detour_ratio")
    return detour_ratio is not None and float(detour_ratio) < minimum_detour


def _semantic_box_size(rng: np.random.Generator, size_range: tuple[float, float]) -> tuple[float, float]:
    long_edge = float(rng.uniform(size_range[0], size_range[1]))
    short_edge = float(rng.uniform(max(0.25, 0.45 * size_range[0]), max(0.35, 0.65 * long_edge)))
    if rng.random() < 0.5:
        return (long_edge, short_edge)
    return (short_edge, long_edge)


def _sample_semantic_point(
    rng: np.random.Generator,
    region: RectRegion,
    size: tuple[float, float],
    *,
    wall_aligned: bool,
) -> tuple[float, float]:
    if not wall_aligned or region.kind == "corridor" and rng.random() < 0.35:
        return region.sample(rng, margin_m=max(size) * 0.5 + 0.35)
    half_x = 0.5 * region.size_m[0]
    half_y = 0.5 * region.size_m[1]
    inset_x = 0.5 * size[0] + 0.35
    inset_y = 0.5 * size[1] + 0.35
    if half_x > half_y:
        x = float(rng.uniform(region.center[0] - half_x + inset_x, region.center[0] + half_x - inset_x))
        y = region.center[1] + rng.choice([-1.0, 1.0]) * max(0.0, half_y - inset_y)
    else:
        x = region.center[0] + rng.choice([-1.0, 1.0]) * max(0.0, half_x - inset_x)
        y = float(rng.uniform(region.center[1] - half_y + inset_y, region.center[1] + half_y - inset_y))
    return (float(x), float(y))


def _push_to_corridor_side(
    rng: np.random.Generator,
    point: tuple[float, float],
    lane: RectRegion,
    *,
    side_margin: float,
) -> tuple[float, float]:
    if lane.size_m[0] >= lane.size_m[1]:
        side_y = lane.center[1] + rng.choice([-1.0, 1.0]) * max(0.0, 0.5 * lane.size_m[1] - side_margin)
        return (point[0], float(side_y))
    side_x = lane.center[0] + rng.choice([-1.0, 1.0]) * max(0.0, 0.5 * lane.size_m[0] - side_margin)
    return (float(side_x), point[1])


def _lane_heading(rng: np.random.Generator, lane: RectRegion) -> float:
    heading = 0.0 if lane.size_m[0] >= lane.size_m[1] else math.pi / 2.0
    if rng.random() < 0.5:
        heading += math.pi
    return float(heading)


def _sample_dynamic_heading(
    rng: np.random.Generator,
    lane: RectRegion,
    level_config: dict[str, Any],
    *,
    center: tuple[float, float],
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
) -> float:
    mode = str(level_config.get("dynamic_heading_mode", "lane"))
    lane_heading = _lane_heading(rng, lane)
    if mode == "random":
        return float(rng.uniform(-math.pi, math.pi))
    if mode != "mixed":
        if rng.random() < float(level_config.get("dynamic_cross_heading_prob", 0.18)):
            lane_heading += float(rng.choice([-1.0, 1.0])) * math.pi / 2.0
        return float(lane_heading)

    choices = [
        ("lane", float(level_config.get("dynamic_heading_lane_weight", 0.45))),
        ("cross", float(level_config.get("dynamic_heading_cross_weight", 0.25))),
        ("diagonal", float(level_config.get("dynamic_heading_diagonal_weight", 0.15))),
        ("path_cross", float(level_config.get("dynamic_heading_path_cross_weight", 0.10))),
        ("random", float(level_config.get("dynamic_heading_random_weight", 0.05))),
    ]
    weights = np.asarray([max(0.0, item[1]) for item in choices], dtype=np.float64)
    if float(np.sum(weights)) <= 1e-9:
        return float(lane_heading)
    selected = choices[int(rng.choice(len(choices), p=weights / np.sum(weights)))][0]
    if selected == "lane":
        return float(lane_heading)
    if selected == "cross":
        return float(lane_heading + float(rng.choice([-1.0, 1.0])) * math.pi / 2.0)
    if selected == "diagonal":
        return float(lane_heading + float(rng.choice([-1.0, 1.0])) * math.pi / 4.0)
    if selected == "path_cross":
        path = np.asarray(goal_xy, dtype=np.float64) - np.asarray(start_xy, dtype=np.float64)
        if np.linalg.norm(path) <= 1e-9:
            path = np.asarray(goal_xy, dtype=np.float64) - np.asarray(center, dtype=np.float64)
        path_heading = math.atan2(float(path[1]), float(path[0])) if np.linalg.norm(path) > 1e-9 else lane_heading
        jitter = float(rng.normal(0.0, float(level_config.get("dynamic_heading_jitter_rad", 0.20))))
        return float(path_heading + float(rng.choice([-1.0, 1.0])) * math.pi / 2.0 + jitter)
    return float(rng.uniform(-math.pi, math.pi))


def _center_separated_from_dynamic_actors(
    center: tuple[float, float],
    radius_m: float,
    actors: tuple[DynamicActor, ...],
    *,
    margin: float,
) -> bool:
    point = np.asarray(center, dtype=np.float64)
    for actor in actors:
        other = np.asarray(actor.center, dtype=np.float64)
        if np.linalg.norm(point - other) < float(radius_m) + _dynamic_actor_proxy_radius(actor) + float(margin):
            return False
    return True


def _dynamic_actor_proxy_radius(actor: DynamicActor) -> float:
    if actor.kind == "cart":
        size = actor.size_m if actor.size_m is not None else (0.9, 0.55)
        return float(0.5 * math.hypot(float(size[0]), float(size[1])))
    if actor.kind == "person":
        return float(0.5 * float(actor.foot_separation_m or 0.34) + float(actor.foot_radius_m or 0.14))
    return float(actor.radius_m or 0.3)


def _dynamic_actor_state(
    actor: DynamicActor,
    time_s: float,
    bounds_x: tuple[float, float],
    bounds_y: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    if actor.trajectory:
        return _trajectory_state(actor.trajectory, time_s)
    return (
        _bounced_position(actor.center, actor.velocity, time_s, bounds_x, bounds_y, margin=0.7),
        actor.velocity,
    )


def _trajectory_state(
    trajectory: tuple[DynamicMotionSegment, ...],
    time_s: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if not trajectory:
        return ((0.0, 0.0), (0.0, 0.0))
    t = float(time_s)
    for segment in trajectory:
        if segment.start_time_s <= t <= segment.end_time_s + 1e-9:
            dt = max(0.0, t - float(segment.start_time_s))
            center = np.asarray(segment.start_center, dtype=np.float64) + dt * np.asarray(segment.velocity, dtype=np.float64)
            return (float(center[0]), float(center[1])), segment.velocity
    segment = trajectory[-1]
    dt = max(0.0, float(segment.end_time_s) - float(segment.start_time_s))
    center = np.asarray(segment.start_center, dtype=np.float64) + dt * np.asarray(segment.velocity, dtype=np.float64)
    return (float(center[0]), float(center[1])), segment.velocity


def _dynamic_actor_primitives(
    actor: DynamicActor,
    *,
    center: tuple[float, float],
    velocity: tuple[float, float],
) -> tuple[PrimitiveObstacle, ...]:
    if actor.kind == "cart":
        size = actor.size_m if actor.size_m is not None else (0.9, 0.55)
        return (BoxObstacle(center=center, size_m=size, kind="cart"),)
    if actor.kind == "person":
        motion = np.asarray(velocity, dtype=np.float64)
        if np.linalg.norm(motion) <= 1e-9:
            motion = np.asarray(actor.velocity, dtype=np.float64)
        heading = math.atan2(float(motion[1]), float(motion[0])) if np.linalg.norm(motion) > 1e-9 else 0.0
        side = np.asarray([-math.sin(heading), math.cos(heading)], dtype=np.float64)
        separation = float(actor.foot_separation_m or 0.34)
        radius = float(actor.foot_radius_m or 0.14)
        base = np.asarray(center, dtype=np.float64)
        primitives: list[PrimitiveObstacle] = []
        for sign in (-0.5, 0.5):
            foot = base + sign * separation * side
            primitives.append(CircleObstacle(center=(float(foot[0]), float(foot[1])), radius_m=radius, kind="person_foot"))
        return tuple(primitives)
    radius = float(actor.radius_m or 0.3)
    return (CircleObstacle(center=center, radius_m=radius, kind=actor.kind),)


def _inside_bounds(
    point: tuple[float, float],
    bounds_x: tuple[float, float],
    bounds_y: tuple[float, float],
    margin: float,
) -> bool:
    return (
        bounds_x[0] + margin <= float(point[0]) <= bounds_x[1] - margin
        and bounds_y[0] + margin <= float(point[1]) <= bounds_y[1] - margin
    )


def _point_clearance(point: tuple[float, float], obstacles: Iterable[PrimitiveObstacle]) -> float:
    clearances = [_clearance_to_primitive(point, item) for item in obstacles]
    return min(clearances) if clearances else float("inf")


def _sample_point_at_primitive_edge_clearance(
    rng: np.random.Generator,
    obstacle: PrimitiveObstacle,
    clearance_range_m: tuple[float, float],
) -> tuple[float, float]:
    clearance = float(rng.uniform(clearance_range_m[0], clearance_range_m[1]))
    if isinstance(obstacle, CircleObstacle):
        angle = float(rng.uniform(-math.pi, math.pi))
        radius = float(obstacle.radius_m) + clearance
        return (
            float(obstacle.center[0] + radius * math.cos(angle)),
            float(obstacle.center[1] + radius * math.sin(angle)),
        )

    half_x = 0.5 * float(obstacle.size_m[0])
    half_y = 0.5 * float(obstacle.size_m[1])
    side_weights = np.asarray([half_y, half_y, half_x, half_x], dtype=np.float64)
    side_weights /= np.sum(side_weights)
    side = int(rng.choice(4, p=side_weights))
    cx, cy = obstacle.center
    if side == 0:
        return (float(cx - half_x - clearance), float(rng.uniform(cy - half_y, cy + half_y)))
    if side == 1:
        return (float(cx + half_x + clearance), float(rng.uniform(cy - half_y, cy + half_y)))
    if side == 2:
        return (float(rng.uniform(cx - half_x, cx + half_x)), float(cy - half_y - clearance))
    return (float(rng.uniform(cx - half_x, cx + half_x)), float(cy + half_y + clearance))


def _clearance_bucket_name(index: int, value_range: tuple[float, float]) -> str:
    labels = ("near_wall", "medium", "normal", "center")
    if 0 <= int(index) < len(labels):
        return labels[int(index)]
    return _clearance_range_name(value_range)


def _clearance_range_name(value_range: tuple[float, float]) -> str:
    return f"{float(value_range[0]):.2f}-{float(value_range[1]):.2f}m"


def _matching_clearance_bucket(
    clearance_m: float,
    buckets: tuple[tuple[float, float], ...],
) -> str:
    for index, value_range in enumerate(buckets):
        if value_range[0] <= float(clearance_m) <= value_range[1]:
            return _clearance_bucket_name(index, value_range)
    return "outside_configured_buckets" if buckets else "legacy"


def _clearance_to_primitive(point: tuple[float, float], obstacle: PrimitiveObstacle) -> float:
    px, py = float(point[0]), float(point[1])
    cx, cy = obstacle.center
    if isinstance(obstacle, CircleObstacle):
        return float(math.hypot(px - cx, py - cy) - obstacle.radius_m)
    half_x = 0.5 * obstacle.size_m[0]
    half_y = 0.5 * obstacle.size_m[1]
    dx = abs(px - cx) - half_x
    dy = abs(py - cy) - half_y
    outside = math.hypot(max(dx, 0.0), max(dy, 0.0))
    inside = min(max(dx, dy), 0.0)
    return float(outside + inside)


def _bounding_radius(obstacle: PrimitiveObstacle) -> float:
    if isinstance(obstacle, CircleObstacle):
        return float(obstacle.radius_m)
    return float(0.5 * math.hypot(obstacle.size_m[0], obstacle.size_m[1]))


def _separated(candidate: PrimitiveObstacle, obstacles: Iterable[PrimitiveObstacle], *, margin: float) -> bool:
    for obstacle in obstacles:
        if _primitive_clearance(candidate, obstacle) < float(margin):
            return False
    return True


def _primitive_clearance(a: PrimitiveObstacle, b: PrimitiveObstacle) -> float:
    if isinstance(a, CircleObstacle) and isinstance(b, CircleObstacle):
        return float(np.linalg.norm(np.asarray(a.center) - np.asarray(b.center)) - a.radius_m - b.radius_m)
    if isinstance(a, CircleObstacle) and isinstance(b, BoxObstacle):
        return float(_clearance_to_primitive(a.center, b) - a.radius_m)
    if isinstance(a, BoxObstacle) and isinstance(b, CircleObstacle):
        return float(_clearance_to_primitive(b.center, a) - b.radius_m)
    if isinstance(a, BoxObstacle) and isinstance(b, BoxObstacle):
        dx = abs(float(a.center[0] - b.center[0])) - 0.5 * (a.size_m[0] + b.size_m[0])
        dy = abs(float(a.center[1] - b.center[1])) - 0.5 * (a.size_m[1] + b.size_m[1])
        outside = math.hypot(max(dx, 0.0), max(dy, 0.0))
        inside = min(max(dx, dy), 0.0)
        return float(outside + inside)
    raise TypeError(f"Unsupported primitive pair: {type(a)!r}, {type(b)!r}")


def _simulate_reflective_dynamic_trajectories(
    actors: tuple[DynamicActor, ...],
    *,
    static_obstacles: tuple[PrimitiveObstacle, ...],
    bounds_x: tuple[float, float],
    bounds_y: tuple[float, float],
    horizon_s: float,
    step_s: float,
    pause_s: float,
    boundary_margin_m: float,
    collision_margin_m: float,
) -> list[list[DynamicMotionSegment]]:
    count = len(actors)
    if count == 0:
        return []
    horizon = max(0.0, float(horizon_s))
    step = max(1e-3, float(step_s))
    pause = max(0.0, float(pause_s))
    positions = np.asarray([actor.center for actor in actors], dtype=np.float64)
    resume_velocities = np.asarray([actor.velocity for actor in actors], dtype=np.float64)
    proxy_radii = np.asarray([_dynamic_actor_proxy_radius(actor) + max(0.0, float(collision_margin_m)) for actor in actors], dtype=np.float64)
    pause_until = np.zeros(count, dtype=np.float64)
    segment_start_time = np.zeros(count, dtype=np.float64)
    segment_start_center = positions.copy()
    segment_velocity = resume_velocities.copy()
    segments: list[list[DynamicMotionSegment]] = [[] for _ in range(count)]
    t = 0.0
    while t < horizon - 1e-9:
        next_t = min(horizon, t + step)
        dt = next_t - t
        actual_velocities = np.zeros_like(resume_velocities)
        for index in range(count):
            if t + 1e-9 >= pause_until[index]:
                actual_velocities[index] = resume_velocities[index]
            if not np.allclose(actual_velocities[index], segment_velocity[index], atol=1e-9):
                _append_motion_segment(segments[index], segment_start_time[index], t, segment_start_center[index], segment_velocity[index])
                segment_start_time[index] = t
                segment_start_center[index] = positions[index].copy()
                segment_velocity[index] = actual_velocities[index].copy()

        proposed = positions + actual_velocities * dt
        collision_normals: list[list[np.ndarray]] = [[] for _ in range(count)]
        for index in range(count):
            if np.linalg.norm(actual_velocities[index]) <= 1e-9:
                continue
            normal = _motion_collision_normal(
                proposed[index],
                proxy_radii[index],
                static_obstacles=static_obstacles,
                bounds_x=bounds_x,
                bounds_y=bounds_y,
                boundary_margin_m=boundary_margin_m,
            )
            if normal is not None:
                collision_normals[index].append(normal)

        for left in range(count):
            for right in range(left + 1, count):
                delta = proposed[left] - proposed[right]
                distance = float(np.linalg.norm(delta))
                if distance > proxy_radii[left] + proxy_radii[right]:
                    continue
                normal = _normalize_vector(delta)
                if normal is None:
                    normal = _normalize_vector(positions[left] - positions[right])
                if normal is None:
                    normal = np.asarray([1.0, 0.0], dtype=np.float64)
                if np.linalg.norm(actual_velocities[left]) > 1e-9:
                    collision_normals[left].append(normal)
                if np.linalg.norm(actual_velocities[right]) > 1e-9:
                    collision_normals[right].append(-normal)

        for index in range(count):
            if collision_normals[index] and np.linalg.norm(actual_velocities[index]) > 1e-9:
                _append_motion_segment(segments[index], segment_start_time[index], t, segment_start_center[index], segment_velocity[index])
                normal = _normalize_vector(np.sum(np.asarray(collision_normals[index]), axis=0))
                if normal is None:
                    fallback = _normalize_vector(actual_velocities[index])
                    normal = -fallback if fallback is not None else np.asarray([1.0, 0.0], dtype=np.float64)
                reflected = _reflect_velocity(actual_velocities[index], normal)
                speed = float(np.linalg.norm(actual_velocities[index]))
                reflected_normalized = _normalize_vector(reflected)
                if reflected_normalized is None:
                    fallback = _normalize_vector(actual_velocities[index])
                    reflected_normalized = -fallback if fallback is not None else np.asarray([1.0, 0.0], dtype=np.float64)
                resume_velocities[index] = reflected_normalized * speed
                pause_until[index] = max(pause_until[index], t + pause)
                positions[index] = _clamp_motion_position(positions[index] + normal * min(0.05, max(0.01, 0.2 * proxy_radii[index])), proxy_radii[index], bounds_x, bounds_y, boundary_margin_m)
                segment_start_time[index] = t
                segment_start_center[index] = positions[index].copy()
                segment_velocity[index] = np.zeros(2, dtype=np.float64)
            else:
                positions[index] = proposed[index]
        t = next_t

    for index in range(count):
        _append_motion_segment(segments[index], segment_start_time[index], horizon, segment_start_center[index], segment_velocity[index])
        if not segments[index]:
            segments[index].append(
                DynamicMotionSegment(
                    start_time_s=0.0,
                    end_time_s=horizon,
                    start_center=(float(positions[index][0]), float(positions[index][1])),
                    velocity=(0.0, 0.0),
                )
            )
    return segments


def _append_motion_segment(
    segments: list[DynamicMotionSegment],
    start_time_s: float,
    end_time_s: float,
    start_center: np.ndarray,
    velocity: np.ndarray,
) -> None:
    if float(end_time_s) <= float(start_time_s) + 1e-9:
        return
    segments.append(
        DynamicMotionSegment(
            start_time_s=float(start_time_s),
            end_time_s=float(end_time_s),
            start_center=(float(start_center[0]), float(start_center[1])),
            velocity=(float(velocity[0]), float(velocity[1])),
        )
    )


def _motion_collision_normal(
    point: np.ndarray,
    radius_m: float,
    *,
    static_obstacles: tuple[PrimitiveObstacle, ...],
    bounds_x: tuple[float, float],
    bounds_y: tuple[float, float],
    boundary_margin_m: float,
) -> np.ndarray | None:
    radius = float(radius_m)
    margin = max(0.0, float(boundary_margin_m))
    normals: list[np.ndarray] = []
    if float(point[0]) - radius < bounds_x[0] + margin:
        normals.append(np.asarray([1.0, 0.0], dtype=np.float64))
    if float(point[0]) + radius > bounds_x[1] - margin:
        normals.append(np.asarray([-1.0, 0.0], dtype=np.float64))
    if float(point[1]) - radius < bounds_y[0] + margin:
        normals.append(np.asarray([0.0, 1.0], dtype=np.float64))
    if float(point[1]) + radius > bounds_y[1] - margin:
        normals.append(np.asarray([0.0, -1.0], dtype=np.float64))
    point_pair = (float(point[0]), float(point[1]))
    for obstacle in static_obstacles:
        if _clearance_to_primitive(point_pair, obstacle) <= radius:
            normal = _collision_normal_from_obstacle(point, obstacle)
            if normal is not None:
                normals.append(normal)
    if not normals:
        return None
    return _normalize_vector(np.sum(np.asarray(normals), axis=0))


def _collision_normal_from_obstacle(point: np.ndarray, obstacle: PrimitiveObstacle) -> np.ndarray | None:
    center = np.asarray(obstacle.center, dtype=np.float64)
    if isinstance(obstacle, CircleObstacle):
        return _normalize_vector(point - center)
    half = 0.5 * np.asarray(obstacle.size_m, dtype=np.float64)
    delta = point - center
    closest = center + np.clip(delta, -half, half)
    outside = point - closest
    normal = _normalize_vector(outside)
    if normal is not None:
        return normal
    distances = np.asarray(
        [
            half[0] - delta[0],
            half[0] + delta[0],
            half[1] - delta[1],
            half[1] + delta[1],
        ],
        dtype=np.float64,
    )
    index = int(np.argmin(distances))
    if index == 0:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    if index == 1:
        return np.asarray([-1.0, 0.0], dtype=np.float64)
    if index == 2:
        return np.asarray([0.0, 1.0], dtype=np.float64)
    return np.asarray([0.0, -1.0], dtype=np.float64)


def _reflect_velocity(velocity: np.ndarray, normal: np.ndarray) -> np.ndarray:
    unit = _normalize_vector(normal)
    if unit is None:
        return -np.asarray(velocity, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    return v - 2.0 * float(np.dot(v, unit)) * unit


def _normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return None
    return np.asarray(vector, dtype=np.float64) / norm


def _clamp_motion_position(
    point: np.ndarray,
    radius_m: float,
    bounds_x: tuple[float, float],
    bounds_y: tuple[float, float],
    boundary_margin_m: float,
) -> np.ndarray:
    radius = float(radius_m) + max(0.0, float(boundary_margin_m))
    return np.asarray(
        [
            float(np.clip(point[0], bounds_x[0] + radius, bounds_x[1] - radius)),
            float(np.clip(point[1], bounds_y[0] + radius, bounds_y[1] - radius)),
        ],
        dtype=np.float64,
    )


def _bounced_position(
    center: tuple[float, float],
    velocity: tuple[float, float],
    time_s: float,
    bounds_x: tuple[float, float],
    bounds_y: tuple[float, float],
    *,
    margin: float,
) -> tuple[float, float]:
    raw = np.asarray(center, dtype=np.float64) + float(time_s) * np.asarray(velocity, dtype=np.float64)
    return (
        _reflect_interval(float(raw[0]), bounds_x[0] + margin, bounds_x[1] - margin),
        _reflect_interval(float(raw[1]), bounds_y[0] + margin, bounds_y[1] - margin),
    )


def _reflect_interval(value: float, low: float, high: float) -> float:
    span = float(high) - float(low)
    if span <= 0.0:
        return float(low)
    shifted = (float(value) - float(low)) % (2.0 * span)
    if shifted > span:
        shifted = 2.0 * span - shifted
    return float(low + shifted)


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _primitive_to_dict(obstacle: PrimitiveObstacle) -> dict[str, Any]:
    data: dict[str, Any] = {"kind": obstacle.kind, "center": list(obstacle.center)}
    if isinstance(obstacle, CircleObstacle):
        data["radius_m"] = obstacle.radius_m
    else:
        data["size_m"] = list(obstacle.size_m)
    return data
