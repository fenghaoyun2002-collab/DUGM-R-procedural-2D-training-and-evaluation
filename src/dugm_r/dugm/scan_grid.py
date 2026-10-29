from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


_EPS = 1e-10


@dataclass(frozen=True)
class GridGeometry:
    """Geometry and beam contract for a robot-centred occupancy grid."""

    width_m: float = 16.0
    height_m: float = 16.0
    resolution_m: float = 0.25
    num_rays: int = 360
    angle_min_rad: float = -math.pi
    angle_increment_rad: float = 2.0 * math.pi / 360.0

    def __post_init__(self) -> None:
        if self.width_m <= 0.0 or self.height_m <= 0.0 or self.resolution_m <= 0.0:
            raise ValueError("grid dimensions and resolution must be positive.")
        if self.num_rays <= 0:
            raise ValueError("num_rays must be positive.")
        width_cells = self.width_m / self.resolution_m
        height_cells = self.height_m / self.resolution_m
        if not math.isclose(width_cells, round(width_cells), abs_tol=1e-9):
            raise ValueError("width_m must be an integer multiple of resolution_m.")
        if not math.isclose(height_cells, round(height_cells), abs_tol=1e-9):
            raise ValueError("height_m must be an integer multiple of resolution_m.")

    @property
    def width_cells(self) -> int:
        return int(round(self.width_m / self.resolution_m))

    @property
    def height_cells(self) -> int:
        return int(round(self.height_m / self.resolution_m))


class GridEncoder:
    """Encode one raw planar scan as ``[occupancy, observed]``.

    The implementation is deliberately independent of Gym, SB3, simulators, and
    robot pose. Traversal geometry is precomputed once for the configured beam
    contract. Robot-local ``+x`` is forward (image top) and ``+y`` is left
    (image left).
    """

    def __init__(self, geometry: GridGeometry | None = None, *, endpoint_rasterization: str = "optimized") -> None:
        self.geometry = geometry or GridGeometry()
        if endpoint_rasterization not in {"reference", "optimized"}:
            raise ValueError("endpoint_rasterization must be 'reference' or 'optimized'.")
        self.endpoint_rasterization = endpoint_rasterization
        self._path_beam_ids, self._path_flat_cells, self._path_entry_distances = self._precompute_paths()

    @classmethod
    def from_config(cls, observation_config: dict) -> "GridEncoder":
        lidar = dict(observation_config.get("lidar_input", {}))
        grid = dict(observation_config.get("grid", {}))
        static_encoder = dict(observation_config.get("static_grid_encoder", {}))
        num_rays = int(lidar.get("num_rays", 360))
        angle_min = float(lidar.get("angle_min_rad", -math.pi))
        fov = float(lidar.get("fov_rad", 2.0 * math.pi))
        return cls(
            GridGeometry(
                width_m=float(grid.get("width_m", 16.0)),
                height_m=float(grid.get("height_m", 16.0)),
                resolution_m=float(grid.get("resolution_m", 0.25)),
                num_rays=num_rays,
                angle_min_rad=angle_min,
                angle_increment_rad=fov / float(num_rays),
            ),
            endpoint_rasterization=str(
                static_encoder.get(
                    "endpoint_rasterization",
                    grid.get("endpoint_rasterization", observation_config.get("endpoint_rasterization", "optimized")),
                )
            ),
        )

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return (2, self.geometry.height_cells, self.geometry.width_cells)

    def encode(
        self,
        ranges_m,
        *,
        valid_mask=None,
        angle_min_rad: float,
        angle_increment_rad: float,
        min_range_m: float,
        max_range_m: float,
    ) -> np.ndarray:
        ranges = np.asarray(ranges_m, dtype=np.float64)
        expected_shape = (self.geometry.num_rays,)
        if ranges.shape != expected_shape:
            raise ValueError(f"ranges_m must have shape {expected_shape}, got {ranges.shape}.")
        if valid_mask is None:
            valid = np.ones(expected_shape, dtype=bool)
        else:
            valid = np.asarray(valid_mask, dtype=bool)
            if valid.shape != expected_shape:
                raise ValueError(f"valid_mask must have shape {expected_shape}, got {valid.shape}.")
        if min_range_m < 0.0 or max_range_m <= min_range_m:
            raise ValueError("Expected 0 <= min_range_m < max_range_m.")
        if not math.isclose(float(angle_min_rad), self.geometry.angle_min_rad, abs_tol=1e-9):
            raise ValueError("angle_min_rad does not match the encoder beam contract.")
        if not math.isclose(float(angle_increment_rad), self.geometry.angle_increment_rad, abs_tol=1e-12):
            raise ValueError("angle_increment_rad does not match the encoder beam contract.")

        usable = valid & np.isfinite(ranges) & (ranges >= float(min_range_m))
        cleaned = np.where(usable, np.clip(ranges, float(min_range_m), float(max_range_m)), float(max_range_m))
        hit = usable & (cleaned < float(max_range_m))

        flat_size = self.geometry.height_cells * self.geometry.width_cells
        observed_flat = np.zeros(flat_size, dtype=np.float32)
        traversal_limit = cleaned[self._path_beam_ids]
        traversed = self._path_entry_distances <= traversal_limit + 1e-9
        observed_flat[self._path_flat_cells[traversed]] = 1.0

        occupancy_flat = np.zeros(flat_size, dtype=np.float32)
        hit_ids = np.flatnonzero(hit)
        if hit_ids.size:
            angles = self.geometry.angle_min_rad + hit_ids.astype(np.float64) * self.geometry.angle_increment_rad
            endpoints = np.column_stack((cleaned[hit_ids] * np.cos(angles), cleaned[hit_ids] * np.sin(angles)))
            if self.endpoint_rasterization == "reference":
                self._rasterize_endpoints_reference(occupancy_flat, endpoints)
            else:
                self._rasterize_endpoints_optimized(occupancy_flat, endpoints)

        observed_flat[occupancy_flat > 0.0] = 1.0
        return np.stack(
            [
                occupancy_flat.reshape(self.geometry.height_cells, self.geometry.width_cells),
                observed_flat.reshape(self.geometry.height_cells, self.geometry.width_cells),
            ],
            axis=0,
        ).astype(np.float32, copy=False)

    def rasterize_occupied_endpoints(
        self,
        ranges_m,
        *,
        valid_mask=None,
        angle_min_rad: float,
        angle_increment_rad: float,
        min_range_m: float,
        max_range_m: float,
    ) -> np.ndarray:
        """Rasterize only occupied hit endpoints using the encoder beam contract."""

        ranges = np.asarray(ranges_m, dtype=np.float64)
        expected_shape = (self.geometry.num_rays,)
        if ranges.shape != expected_shape:
            raise ValueError(f"ranges_m must have shape {expected_shape}, got {ranges.shape}.")
        if valid_mask is None:
            valid = np.ones(expected_shape, dtype=bool)
        else:
            valid = np.asarray(valid_mask, dtype=bool)
            if valid.shape != expected_shape:
                raise ValueError(f"valid_mask must have shape {expected_shape}, got {valid.shape}.")
        if min_range_m < 0.0 or max_range_m <= min_range_m:
            raise ValueError("Expected 0 <= min_range_m < max_range_m.")
        if not math.isclose(float(angle_min_rad), self.geometry.angle_min_rad, abs_tol=1e-9):
            raise ValueError("angle_min_rad does not match the encoder beam contract.")
        if not math.isclose(float(angle_increment_rad), self.geometry.angle_increment_rad, abs_tol=1e-12):
            raise ValueError("angle_increment_rad does not match the encoder beam contract.")

        usable = valid & np.isfinite(ranges) & (ranges >= float(min_range_m))
        cleaned = np.where(usable, np.clip(ranges, float(min_range_m), float(max_range_m)), float(max_range_m))
        hit = usable & (cleaned < float(max_range_m))

        flat_size = self.geometry.height_cells * self.geometry.width_cells
        occupancy_flat = np.zeros(flat_size, dtype=np.float32)
        hit_ids = np.flatnonzero(hit)
        if hit_ids.size:
            angles = self.geometry.angle_min_rad + hit_ids.astype(np.float64) * self.geometry.angle_increment_rad
            endpoints = np.column_stack((cleaned[hit_ids] * np.cos(angles), cleaned[hit_ids] * np.sin(angles)))
            if self.endpoint_rasterization == "reference":
                self._rasterize_endpoints_reference(occupancy_flat, endpoints)
            else:
                self._rasterize_endpoints_optimized(occupancy_flat, endpoints)
        return occupancy_flat.reshape(self.geometry.height_cells, self.geometry.width_cells).astype(np.float32, copy=False)

    def _rasterize_endpoints_reference(self, occupancy_flat: np.ndarray, endpoints_xy: np.ndarray) -> None:
        for x_m, y_m in np.asarray(endpoints_xy, dtype=np.float64).reshape(-1, 2):
            cell = self.metric_to_grid(float(x_m), float(y_m))
            if cell is not None:
                occupancy_flat[cell[0] * self.geometry.width_cells + cell[1]] = 1.0

    def _rasterize_endpoints_optimized(self, occupancy_flat: np.ndarray, endpoints_xy: np.ndarray) -> None:
        endpoints = np.asarray(endpoints_xy, dtype=np.float64).reshape(-1, 2)
        if endpoints.size == 0:
            return
        x_m, y_m = endpoints[:, 0], endpoints[:, 1]
        half_x = 0.5 * self.geometry.height_m
        half_y = 0.5 * self.geometry.width_m
        inside = (-half_x <= x_m) & (x_m < half_x) & (-half_y <= y_m) & (y_m < half_y)
        if not np.any(inside):
            return
        x_index = np.floor((x_m[inside] + half_x) / self.geometry.resolution_m).astype(np.int64)
        y_index = np.floor((y_m[inside] + half_y) / self.geometry.resolution_m).astype(np.int64)
        rows = self.geometry.height_cells - 1 - x_index
        cols = self.geometry.width_cells - 1 - y_index
        occupancy_flat[rows * self.geometry.width_cells + cols] = 1.0

    def metric_to_grid(self, x_forward_m: float, y_left_m: float) -> tuple[int, int] | None:
        """Map robot-frame metric coordinates to image ``(row, column)``."""

        half_x = 0.5 * self.geometry.height_m
        half_y = 0.5 * self.geometry.width_m
        if not (-half_x <= x_forward_m < half_x and -half_y <= y_left_m < half_y):
            return None
        x_index = int(math.floor((x_forward_m + half_x) / self.geometry.resolution_m))
        y_index = int(math.floor((y_left_m + half_y) / self.geometry.resolution_m))
        row = self.geometry.height_cells - 1 - x_index
        col = self.geometry.width_cells - 1 - y_index
        return row, col

    def grid_cell_center(self, row: int, col: int) -> tuple[float, float]:
        if not (0 <= row < self.geometry.height_cells and 0 <= col < self.geometry.width_cells):
            raise IndexError((row, col))
        x_index = self.geometry.height_cells - 1 - int(row)
        y_index = self.geometry.width_cells - 1 - int(col)
        half_x = 0.5 * self.geometry.height_m
        half_y = 0.5 * self.geometry.width_m
        return (
            -half_x + (x_index + 0.5) * self.geometry.resolution_m,
            -half_y + (y_index + 0.5) * self.geometry.resolution_m,
        )

    def _precompute_paths(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        beam_ids: list[int] = []
        flat_cells: list[int] = []
        entry_distances: list[float] = []
        for beam_id in range(self.geometry.num_rays):
            angle = self.geometry.angle_min_rad + beam_id * self.geometry.angle_increment_rad
            for row, col, entry_distance in self._traverse_ray(angle):
                beam_ids.append(beam_id)
                flat_cells.append(row * self.geometry.width_cells + col)
                entry_distances.append(entry_distance)
        return (
            np.asarray(beam_ids, dtype=np.int32),
            np.asarray(flat_cells, dtype=np.int32),
            np.asarray(entry_distances, dtype=np.float64),
        )

    def _traverse_ray(self, angle_rad: float) -> list[tuple[int, int, float]]:
        direction_x = math.cos(angle_rad)
        direction_y = math.sin(angle_rad)
        resolution = self.geometry.resolution_m
        half_x = 0.5 * self.geometry.height_m
        half_y = 0.5 * self.geometry.width_m
        cells_x = self.geometry.height_cells
        cells_y = self.geometry.width_cells

        probe = min(1e-8, resolution * 1e-6)
        index_x = int(math.floor((probe * direction_x + half_x) / resolution))
        index_y = int(math.floor((probe * direction_y + half_y) / resolution))
        step_x = 1 if direction_x > _EPS else -1 if direction_x < -_EPS else 0
        step_y = 1 if direction_y > _EPS else -1 if direction_y < -_EPS else 0
        delta_x = resolution / abs(direction_x) if step_x else math.inf
        delta_y = resolution / abs(direction_y) if step_y else math.inf

        if step_x > 0:
            next_x = -half_x + (index_x + 1) * resolution
            max_x = next_x / direction_x
        elif step_x < 0:
            next_x = -half_x + index_x * resolution
            max_x = next_x / direction_x
        else:
            max_x = math.inf
        if step_y > 0:
            next_y = -half_y + (index_y + 1) * resolution
            max_y = next_y / direction_y
        elif step_y < 0:
            next_y = -half_y + index_y * resolution
            max_y = next_y / direction_y
        else:
            max_y = math.inf

        entry_distance = 0.0
        path: list[tuple[int, int, float]] = []
        while 0 <= index_x < cells_x and 0 <= index_y < cells_y:
            row = cells_x - 1 - index_x
            col = cells_y - 1 - index_y
            path.append((row, col, max(0.0, float(entry_distance))))
            if math.isclose(max_x, max_y, rel_tol=0.0, abs_tol=1e-12):
                entry_distance = max_x
                index_x += step_x
                index_y += step_y
                max_x += delta_x
                max_y += delta_y
            elif max_x < max_y:
                entry_distance = max_x
                index_x += step_x
                max_x += delta_x
            else:
                entry_distance = max_y
                index_y += step_y
                max_y += delta_y
        return path
