from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time


_NATIVE_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _configure_import_time_native_threads() -> None:
    for key in _NATIVE_THREAD_ENV:
        os.environ.setdefault(key, "1")


_configure_import_time_native_threads()

import numpy as np
import yaml

from dugm_r.dugm.cluster import OrderedClusterExtractor
from dugm_r.dugm.tracker_rasterizer import TrackedGridRasterizer
from dugm_r.dugm.preprocessor import LidarPreprocessor
from dugm_r.dugm.tracker import MultiObjectTracker
from dugm_r.dugm.tracker_grid import GridEncoder, GridGeometry
from dugm_r.dugm.scan_buffer import ScanSample, ScanSequenceGuard


FROZEN_V1_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "dugm_estimator.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "dugm_estimator.yaml"
EXPECTED_CHANNELS = ["occupancy", "observed", "dynamic_mask", "velocity_x", "velocity_y", "velocity_valid"]


def load_estimator_config(path: str | Path | None = None) -> dict:
    config_path = Path(path or DEFAULT_CONFIG_PATH).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    validate_estimator_config(config)
    config["_config_path"] = str(config_path)
    return config


def validate_estimator_config(config: dict) -> None:
    if config.get("algorithm") != "dugm_tracker" or int(config.get("version", 0)) != 1:
        raise ValueError("Expected a DUGM tracker version 1 configuration.")
    input_config, grid, tracking = config["input"], config["grid"], config["tracking"]
    if int(input_config["num_rays"]) != 360:
        raise ValueError("The frozen input contract requires 360 rays.")
    if list(grid["shape"]) != [64, 64] or list(grid["channels"]) != EXPECTED_CHANNELS:
        raise ValueError("The frozen six-channel 64x64 observation contract was changed.")
    if grid.get("layout") != "channels_first" or grid.get("dtype") != "float32":
        raise ValueError("The frozen contract requires channels-first float32 output.")
    if tracking.get("max_tracks") is not None:
        raise ValueError("The versioned estimator must not impose a fixed track cap.")
    maneuver = tracking.get("maneuver_mode")
    if not isinstance(maneuver, dict):
        raise ValueError("Versioned tracking requires a maneuver_mode configuration.")
    normal_acceleration_std = float(tracking["process_acceleration_std_mps2"])
    maneuver_acceleration_std = float(maneuver["process_acceleration_std_mps2"])
    if normal_acceleration_std <= 0.0 or maneuver_acceleration_std <= normal_acceleration_std:
        raise ValueError("Maneuver acceleration std must exceed the positive normal-mode value.")
    trigger_nis = float(maneuver["trigger_nis_chi2"])
    if trigger_nis <= 0.0 or trigger_nis > float(tracking["association_gate_chi2"]):
        raise ValueError("Maneuver NIS trigger must be positive and no larger than the association gate.")
    persistent_nis = float(maneuver["persistent_innovation_nis"])
    if persistent_nis <= 0.0 or persistent_nis >= trigger_nis:
        raise ValueError("Persistent innovation threshold must be positive and below the immediate NIS trigger.")
    if int(maneuver["persistent_innovation_frames"]) < 2:
        raise ValueError("Persistent innovation must require at least two frames.")
    if float(maneuver["association_ambiguity_margin_nis"]) < 0.0:
        raise ValueError("Maneuver association ambiguity margin must be non-negative.")
    if not 3 <= int(maneuver["exit_stable_frames"]) <= 5:
        raise ValueError("Maneuver mode must require 3 to 5 stable frames before exit.")
    if not isinstance(maneuver["trigger_on_confirmed_miss"], bool):
        raise ValueError("trigger_on_confirmed_miss must be boolean.")


def estimator_config_hash(config: dict) -> str:
    canonical = {key: value for key, value in config.items() if not str(key).startswith("_")}
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DugmEstimator:
    """Stateful versioned estimator; GT labels are absent from this API."""

    def __init__(
        self,
        config: dict | None = None,
        *,
        config_path: str | Path | None = None,
        implementation: str = "optimized",
        diagnostics_mode: str = "full",
    ) -> None:
        if implementation not in {"reference", "optimized"}:
            raise ValueError("implementation must be 'reference' or 'optimized'.")
        if diagnostics_mode not in {"full", "summary"}:
            raise ValueError("diagnostics_mode must be 'full' or 'summary'.")
        self.implementation = implementation
        self.diagnostics_mode = diagnostics_mode
        self.config = deepcopy(config) if config is not None else load_estimator_config(config_path)
        validate_estimator_config(self.config)
        input_config, grid = self.config["input"], self.config["grid"]
        num_rays = int(input_config["num_rays"])
        angle_increment = float(input_config["fov_rad"]) / num_rays
        self.geometry = GridGeometry(
            width_m=float(grid["width_m"]),
            height_m=float(grid["height_m"]),
            resolution_m=float(grid["resolution_m"]),
            num_rays=num_rays,
            angle_min_rad=float(input_config["angle_min_rad"]),
            angle_increment_rad=angle_increment,
        )
        self.grid_encoder = GridEncoder(self.geometry, endpoint_rasterization=implementation)
        self.preprocessor = LidarPreprocessor(
            num_rays=num_rays,
            angle_min_rad=self.geometry.angle_min_rad,
            angle_increment_rad=self.geometry.angle_increment_rad,
        )
        self.cluster_extractor = OrderedClusterExtractor(
            self.config["clustering"],
            beam_unit_vectors=self.preprocessor.beam_unit_vectors,
            angle_increment_rad=angle_increment,
            implementation=implementation,
        )
        self.tracker = MultiObjectTracker(
            self.config["tracking"],
            self.config["dynamic_semantics"],
            implementation=implementation,
        )
        self.rasterizer = TrackedGridRasterizer(self.grid_encoder, self.config["velocity"])
        self.sequence_guard = ScanSequenceGuard(
            max_linear_speed_mps=float(input_config["max_linear_speed_mps"]),
            max_angular_speed_radps=float(input_config["max_angular_speed_radps"]),
            translation_tolerance_m=float(input_config["pose_translation_tolerance_m"]),
            rotation_tolerance_rad=float(input_config["pose_rotation_tolerance_rad"]),
            timestamp_epsilon_s=float(input_config["timestamp_epsilon_s"]),
        )
        self.config_hash = estimator_config_hash(self.config)
        self.last_tracked_grid: np.ndarray | None = None
        self.last_diagnostics: dict | None = None
        self.last_preprocessed_scan = None
        self.last_cluster_result = None
        self.last_tracker_result = None
        self.unique_update_count = 0

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return (6, self.geometry.height_cells, self.geometry.width_cells)

    def reset(self, *, reason: str = "episode_reset") -> None:
        self.tracker.reset()
        self.sequence_guard.reset(reason=reason)
        self.last_tracked_grid = None
        self.last_diagnostics = None
        self.last_preprocessed_scan = None
        self.last_cluster_result = None
        self.last_tracker_result = None
        self.unique_update_count = 0

    def update(
        self,
        ranges_m,
        valid_mask,
        *,
        scan_pose_xyyaw,
        scan_timestamp_s: float,
        sample_id,
        base_grid=None,
    ) -> np.ndarray:
        started = time.perf_counter()
        sample = ScanSample.create(ranges_m, valid_mask, scan_pose_xyyaw, scan_timestamp_s, sample_id)
        validated_at = time.perf_counter()
        if sample.ranges_m.shape != (self.geometry.num_rays,):
            raise ValueError(f"Expected {(self.geometry.num_rays,)} scan, got {sample.ranges_m.shape}.")
        previous = self.sequence_guard.last_sample
        sequence_action = self.sequence_guard.classify(sample)
        if sequence_action == "duplicate":
            if self.last_tracked_grid is None:
                raise RuntimeError("Duplicate sample encountered before estimator initialization.")
            self.last_diagnostics = {**(self.last_diagnostics or {}), "sequence_action": "duplicate", "duplicate": True}
            return self.last_tracked_grid.copy()
        if sequence_action == "reset":
            self.tracker.reset()

        input_config = self.config["input"]
        preprocessed = self.preprocessor.preprocess(
            sample.ranges_m,
            sample.valid_mask,
            min_range_m=float(input_config["min_range_m"]),
            max_range_m=float(input_config["max_range_m"]),
        )
        preprocessed_at = time.perf_counter()
        clusters = self.cluster_extractor.extract(preprocessed)
        clustered_at = time.perf_counter()
        if sequence_action == "advance":
            if previous is None:
                raise RuntimeError("Sequence guard advanced without a previous sample.")
            tracker_result = self.tracker.step(
                clusters.clusters,
                previous_pose_xyyaw=previous.scan_pose_xyyaw,
                current_pose_xyyaw=sample.scan_pose_xyyaw,
                previous_timestamp_s=previous.scan_timestamp_s,
                current_timestamp_s=sample.scan_timestamp_s,
            )
        else:
            tracker_result = self.tracker.initialize(clusters.clusters, timestamp_s=sample.scan_timestamp_s)
        tracked_at = time.perf_counter()
        if base_grid is None:
            resolved_base_grid = self.grid_encoder.encode(
                sample.ranges_m,
                valid_mask=sample.valid_mask,
                angle_min_rad=self.geometry.angle_min_rad,
                angle_increment_rad=self.geometry.angle_increment_rad,
                min_range_m=float(input_config["min_range_m"]),
                max_range_m=float(input_config["max_range_m"]),
            )
            base_grid_source = "encoded"
        else:
            resolved_base_grid = np.asarray(base_grid, dtype=np.float32)
            if resolved_base_grid.shape != self.grid_encoder.output_shape:
                raise ValueError(
                    f"base_grid must have shape {self.grid_encoder.output_shape}, got {resolved_base_grid.shape}."
                )
            base_grid_source = "provided"
        encoded_at = time.perf_counter()
        tracked_grid = self.rasterizer.rasterize(resolved_base_grid, self.tracker.tracks)
        rasterized_at = time.perf_counter()
        self.last_preprocessed_scan = preprocessed
        self.last_cluster_result = clusters
        self.last_tracker_result = tracker_result
        self.unique_update_count += 1
        self.last_tracked_grid = tracked_grid.copy()
        diagnostics = {
            "sample_id": sample.sample_id,
            "timestamp_s": sample.scan_timestamp_s,
            "sequence_action": sequence_action,
            "reset_reason": self.sequence_guard.last_reset_reason,
            "duplicate": False,
            "valid_hit_count": int(np.sum(preprocessed.hit_mask)),
            "raw_segment_count": clusters.raw_segment_count,
            "accepted_cluster_count": len(clusters.clusters),
            "cluster_rejection_counts": dict(clusters.rejection_counts),
            "cluster_merge_count": clusters.merge_count,
            "active_track_count": len(self.tracker.tracks),
            "match_count": len(tracker_result.association.matches),
            "gated_pair_count": tracker_result.association.gated_pair_count,
            "diagnostics_mode": self.diagnostics_mode,
            "base_grid_source": base_grid_source,
            "timing_ms": {
                "input_validation": (validated_at - started) * 1000.0,
                "range_cleaning_and_cartesian": (preprocessed_at - validated_at) * 1000.0,
                "clustering": (clustered_at - preprocessed_at) * 1000.0,
                "tracking_prediction_association_update": (tracked_at - clustered_at) * 1000.0,
                "occupancy_observed_encoding": (encoded_at - tracked_at) * 1000.0,
                "dynamic_rasterization_and_invariants": (rasterized_at - encoded_at) * 1000.0,
            },
            "encoding_ms": (rasterized_at - started) * 1000.0,
        }
        if self.diagnostics_mode == "full":
            diagnostics.update(
                {
                    "clusters": [
                        {
                            "detection_id": cluster.detection_id,
                            "centroid_xy": cluster.centroid_xy.astype(float).tolist(),
                            "point_count": cluster.point_count,
                            "diameter_m": cluster.diameter_m,
                            "width_m": cluster.width_m,
                            "aspect_ratio": cluster.aspect_ratio,
                        }
                        for cluster in clusters.clusters
                    ],
                    "created_track_ids": list(tracker_result.created_track_ids),
                    "deleted_track_ids": list(tracker_result.deleted_track_ids),
                    "ambiguous_track_indices": list(tracker_result.association.ambiguous_track_indices),
                    "ambiguous_detection_indices": list(tracker_result.association.ambiguous_detection_indices),
                    "matched_nis": list(tracker_result.matched_nis),
                    "maneuver_triggered_track_ids": list(tracker_result.maneuver_triggered_track_ids),
                    "maneuver_exited_track_ids": list(tracker_result.maneuver_exited_track_ids),
                    "tracks": self.tracker.snapshots(),
                }
            )
        self.last_diagnostics = diagnostics
        return tracked_grid

    def resolved_config(self) -> dict:
        return {
            **{key: deepcopy(value) for key, value in self.config.items() if not str(key).startswith("_")},
            "config_sha256": self.config_hash,
        }
