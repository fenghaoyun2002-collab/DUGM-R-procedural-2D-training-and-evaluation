from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreprocessedScan:
    cleaned_ranges_m: np.ndarray
    usable_mask: np.ndarray
    hit_mask: np.ndarray
    endpoints_xy: np.ndarray
    beam_indices: np.ndarray


class LidarPreprocessor:
    def __init__(self, *, num_rays: int, angle_min_rad: float, angle_increment_rad: float) -> None:
        self.num_rays = int(num_rays)
        self.angle_min_rad = float(angle_min_rad)
        self.angle_increment_rad = float(angle_increment_rad)
        self.beam_angles = self.angle_min_rad + np.arange(self.num_rays, dtype=np.float64) * self.angle_increment_rad
        self.beam_unit_vectors = np.column_stack((np.cos(self.beam_angles), np.sin(self.beam_angles)))

    def preprocess(self, ranges_m, valid_mask, *, min_range_m: float, max_range_m: float) -> PreprocessedScan:
        ranges = np.asarray(ranges_m, dtype=np.float64)
        valid = np.asarray(valid_mask, dtype=bool)
        expected = (self.num_rays,)
        if ranges.shape != expected or valid.shape != expected:
            raise ValueError(f"Scan and mask must have shape {expected}.")
        usable = valid & np.isfinite(ranges) & (ranges >= float(min_range_m))
        cleaned = np.where(usable, np.clip(ranges, min_range_m, max_range_m), max_range_m)
        hit = usable & (cleaned < float(max_range_m))
        beam_indices = np.flatnonzero(hit).astype(np.int32)
        endpoints = cleaned[beam_indices, None] * self.beam_unit_vectors[beam_indices]
        return PreprocessedScan(cleaned, usable, hit, endpoints, beam_indices)
