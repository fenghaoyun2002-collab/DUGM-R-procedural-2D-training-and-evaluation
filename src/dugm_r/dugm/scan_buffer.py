from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable
import math

import numpy as np


def wrap_angle(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class ScanSample:
    ranges_m: np.ndarray
    valid_mask: np.ndarray
    scan_pose_xyyaw: np.ndarray
    scan_timestamp_s: float
    sample_id: Hashable

    @classmethod
    def create(cls, ranges_m, valid_mask, scan_pose_xyyaw, scan_timestamp_s, sample_id) -> "ScanSample":
        ranges = np.asarray(ranges_m, dtype=np.float32).reshape(-1).copy()
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1).copy()
        pose = np.asarray(scan_pose_xyyaw, dtype=np.float64).reshape(-1).copy()
        if ranges.size == 0 or valid.shape != ranges.shape:
            raise ValueError("ranges_m and valid_mask must be non-empty and have identical shapes.")
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            raise ValueError("scan_pose_xyyaw must contain three finite values.")
        stamp = float(scan_timestamp_s)
        if not math.isfinite(stamp):
            raise ValueError("scan_timestamp_s must be finite.")
        if sample_id is None:
            raise ValueError("sample_id is required by the V1 contract.")
        ranges.setflags(write=False)
        valid.setflags(write=False)
        pose.setflags(write=False)
        return cls(ranges, valid, pose, stamp, sample_id)


class ScanSequenceGuard:
    """Classify samples without mutating perception state more than once."""

    def __init__(
        self,
        *,
        max_linear_speed_mps: float,
        max_angular_speed_radps: float,
        translation_tolerance_m: float,
        rotation_tolerance_rad: float,
        timestamp_epsilon_s: float = 1e-9,
    ) -> None:
        self.max_linear_speed_mps = float(max_linear_speed_mps)
        self.max_angular_speed_radps = float(max_angular_speed_radps)
        self.translation_tolerance_m = float(translation_tolerance_m)
        self.rotation_tolerance_rad = float(rotation_tolerance_rad)
        self.timestamp_epsilon_s = float(timestamp_epsilon_s)
        self.last_sample: ScanSample | None = None
        self.last_reset_reason: str | None = None
        self._seen_sample_ids: set[Hashable] = set()

    def reset(self, *, reason: str = "episode_reset") -> None:
        self.last_sample = None
        self.last_reset_reason = str(reason)
        self._seen_sample_ids.clear()

    def classify(self, sample: ScanSample) -> str:
        previous = self.last_sample
        if previous is None:
            self.last_sample = sample
            self._seen_sample_ids = {sample.sample_id}
            self.last_reset_reason = "first_sample"
            return "reset"
        dt = sample.scan_timestamp_s - previous.scan_timestamp_s
        if abs(dt) <= self.timestamp_epsilon_s:
            return "duplicate"
        if dt < -self.timestamp_epsilon_s:
            self.last_sample = sample
            self._seen_sample_ids = {sample.sample_id}
            self.last_reset_reason = "timestamp_regression"
            return "reset"
        if sample.sample_id in self._seen_sample_ids:
            return "duplicate"
        translation = float(np.linalg.norm(sample.scan_pose_xyyaw[:2] - previous.scan_pose_xyyaw[:2]))
        rotation = abs(wrap_angle(float(sample.scan_pose_xyyaw[2] - previous.scan_pose_xyyaw[2])))
        allowed_translation = self.max_linear_speed_mps * dt + self.translation_tolerance_m
        allowed_rotation = self.max_angular_speed_radps * dt + self.rotation_tolerance_rad
        self.last_sample = sample
        if translation > allowed_translation or rotation > allowed_rotation:
            self._seen_sample_ids = {sample.sample_id}
            self.last_reset_reason = "pose_discontinuity"
            return "reset"
        self._seen_sample_ids.add(sample.sample_id)
        self.last_reset_reason = None
        return "advance"
