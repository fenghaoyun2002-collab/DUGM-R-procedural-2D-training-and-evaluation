from __future__ import annotations

import numpy as np

from dugm_r.dugm.track import KalmanTrack, SEMANTIC_DYNAMIC
from dugm_r.dugm.tracker_grid import GridEncoder


class TrackedGridRasterizer:
    channels = ("occupancy", "observed", "dynamic_mask", "velocity_x", "velocity_y", "velocity_valid")

    def __init__(self, grid_encoder: GridEncoder, velocity_config: dict) -> None:
        self.grid_encoder = grid_encoder
        self.velocity_normalization_max_mps = float(velocity_config["normalization_max_mps"])
        self.velocity_valid_max_std_mps = float(velocity_config["valid_max_std_mps"])
        if self.velocity_normalization_max_mps <= 0.0:
            raise ValueError("normalization_max_mps must be positive.")

    def rasterize(self, base_grid: np.ndarray, tracks: list[KalmanTrack]) -> np.ndarray:
        base = np.asarray(base_grid, dtype=np.float32)
        if base.shape != self.grid_encoder.output_shape:
            raise ValueError(f"base_grid must have shape {self.grid_encoder.output_shape}.")
        result = np.zeros((6, *base.shape[1:]), dtype=np.float32)
        result[:2] = base
        for track in tracks:
            if (
                track.semantic_state != SEMANTIC_DYNAMIC
                or not track.current_measurement_available
                or track.latest_cluster is None
            ):
                continue
            cells: set[tuple[int, int]] = set()
            for point in track.latest_cluster.points_xy:
                cell = self.grid_encoder.metric_to_grid(float(point[0]), float(point[1]))
                if cell is not None and result[0][cell] > 0.5:
                    cells.add(cell)
            if not cells:
                continue
            velocity_is_valid = (
                not track.in_maneuver_mode
                and track.velocity_std_mps <= self.velocity_valid_max_std_mps
            )
            normalized = np.clip(track.state[2:] / self.velocity_normalization_max_mps, -1.0, 1.0)
            for cell in cells:
                result[2][cell] = 1.0
                if velocity_is_valid:
                    result[3][cell] = np.float32(normalized[0])
                    result[4][cell] = np.float32(normalized[1])
                    result[5][cell] = 1.0
        self.assert_invariants(result)
        return result

    @staticmethod
    def assert_invariants(tracked_grid: np.ndarray) -> None:
        if tracked_grid.shape[0] != 6 or tracked_grid.dtype != np.float32:
            raise AssertionError("tracked grid must be float32 with six channels.")
        if not np.all(np.isfinite(tracked_grid)):
            raise AssertionError("tracked grid contains NaN or Inf.")
        occupancy, observed, dynamic, velocity_x, velocity_y, velocity_valid = tracked_grid
        if not np.all((0.0 <= occupancy) & (occupancy <= observed) & (observed <= 1.0)):
            raise AssertionError("occupancy/observed subset invariant failed.")
        if not np.all((0.0 <= dynamic) & (dynamic <= occupancy)):
            raise AssertionError("dynamic_mask must be a subset of occupancy.")
        if not np.all((0.0 <= velocity_valid) & (velocity_valid <= dynamic)):
            raise AssertionError("velocity_valid must be a subset of dynamic_mask.")
        invalid = velocity_valid == 0.0
        if np.any(velocity_x[invalid] != 0.0) or np.any(velocity_y[invalid] != 0.0):
            raise AssertionError("Invalid velocity cells must contain zero velocity.")
        if np.min(velocity_x) < -1.0 or np.max(velocity_x) > 1.0 or np.min(velocity_y) < -1.0 or np.max(velocity_y) > 1.0:
            raise AssertionError("Velocity channels are outside [-1, 1].")
