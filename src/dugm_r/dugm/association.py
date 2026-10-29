from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from dugm_r.dugm.cluster import Cluster
from dugm_r.dugm.track import KalmanTrack


@dataclass(frozen=True)
class AssociationResult:
    matches: tuple[tuple[int, int], ...]
    unmatched_track_indices: tuple[int, ...]
    unmatched_detection_indices: tuple[int, ...]
    mahalanobis_cost: np.ndarray
    euclidean_distance_m: np.ndarray
    gated_pair_count: int
    ambiguous_track_indices: tuple[int, ...]
    ambiguous_detection_indices: tuple[int, ...]


def associate(
    tracks: list[KalmanTrack],
    detections: tuple[Cluster, ...],
    *,
    gate_chi2: float,
    measurement_position_std_floor_m: float,
    ambiguity_margin_nis: float = 0.0,
    implementation: str = "reference",
) -> AssociationResult:
    track_count, detection_count = len(tracks), len(detections)
    if implementation == "optimized":
        costs, euclidean = _cost_matrices_optimized(
            tracks,
            detections,
            gate_chi2=float(gate_chi2),
            measurement_position_std_floor_m=float(measurement_position_std_floor_m),
        )
    elif implementation == "reference":
        costs, euclidean = _cost_matrices_reference(
            tracks,
            detections,
            gate_chi2=float(gate_chi2),
            measurement_position_std_floor_m=float(measurement_position_std_floor_m),
        )
    else:
        raise ValueError("implementation must be 'reference' or 'optimized'.")
    if track_count == 0 or detection_count == 0 or not np.any(np.isfinite(costs)):
        matches: list[tuple[int, int]] = []
    else:
        finite_fill = np.where(np.isfinite(costs), costs, float(gate_chi2) + 1e6)
        rows, cols = linear_sum_assignment(finite_fill)
        matches = [(int(row), int(col)) for row, col in zip(rows, cols, strict=True) if np.isfinite(costs[row, col])]
    matched_tracks = {row for row, _ in matches}
    matched_detections = {col for _, col in matches}
    ambiguous_tracks, ambiguous_detections = _find_close_competitions(costs, float(ambiguity_margin_nis))
    return AssociationResult(
        matches=tuple(matches),
        unmatched_track_indices=tuple(index for index in range(track_count) if index not in matched_tracks),
        unmatched_detection_indices=tuple(index for index in range(detection_count) if index not in matched_detections),
        mahalanobis_cost=costs,
        euclidean_distance_m=euclidean,
        gated_pair_count=int(np.sum(~np.isfinite(costs))),
        ambiguous_track_indices=ambiguous_tracks,
        ambiguous_detection_indices=ambiguous_detections,
    )


def _cost_matrices_reference(
    tracks: list[KalmanTrack],
    detections: tuple[Cluster, ...],
    *,
    gate_chi2: float,
    measurement_position_std_floor_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    costs = np.full((len(tracks), len(detections)), np.inf, dtype=np.float64)
    euclidean = np.full_like(costs, np.inf)
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            _residual, _covariance, nis = track.innovation(detection, measurement_position_std_floor_m)
            euclidean[track_index, detection_index] = np.linalg.norm(detection.centroid_xy - track.state[:2])
            if nis <= gate_chi2:
                costs[track_index, detection_index] = nis
    return costs, euclidean


def _cost_matrices_optimized(
    tracks: list[KalmanTrack],
    detections: tuple[Cluster, ...],
    *,
    gate_chi2: float,
    measurement_position_std_floor_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    track_count, detection_count = len(tracks), len(detections)
    if track_count == 0 or detection_count == 0:
        empty = np.full((track_count, detection_count), np.inf, dtype=np.float64)
        return empty, empty.copy()
    track_positions = np.stack([track.state[:2] for track in tracks], axis=0)
    detection_positions = np.stack([detection.centroid_xy for detection in detections], axis=0)
    residual = detection_positions[None, :, :] - track_positions[:, None, :]
    euclidean = np.sqrt(np.sum(residual * residual, axis=2))
    track_covariance = np.stack([track.covariance[:2, :2] for track in tracks], axis=0)
    floor_variance = measurement_position_std_floor_m**2
    detection_covariance = np.stack(
        [
            detection.covariance_xy / max(1, detection.point_count) + np.eye(2, dtype=np.float64) * floor_variance
            for detection in detections
        ],
        axis=0,
    )
    innovation = track_covariance[:, None, :, :] + detection_covariance[None, :, :, :]
    s00, s01 = innovation[:, :, 0, 0], innovation[:, :, 0, 1]
    s10, s11 = innovation[:, :, 1, 0], innovation[:, :, 1, 1]
    determinant = s00 * s11 - s01 * s10
    r0, r1 = residual[:, :, 0], residual[:, :, 1]
    numerator = r0 * r0 * s11 - r0 * r1 * (s01 + s10) + r1 * r1 * s00
    nis = numerator / determinant
    return np.where(nis <= gate_chi2, nis, np.inf), euclidean


def _find_close_competitions(costs: np.ndarray, margin_nis: float) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Report close candidates without changing the Hungarian assignment cost."""

    if margin_nis <= 0.0 or costs.size == 0:
        return (), ()
    ambiguous_tracks: set[int] = set()
    ambiguous_detections: set[int] = set()
    for track_index, row in enumerate(costs):
        finite_indices = np.flatnonzero(np.isfinite(row))
        if finite_indices.size < 2:
            continue
        ordered = finite_indices[np.argsort(row[finite_indices])]
        if float(row[ordered[1]] - row[ordered[0]]) <= margin_nis:
            ambiguous_tracks.add(track_index)
            ambiguous_detections.update(int(index) for index in ordered if row[index] <= row[ordered[0]] + margin_nis)
    for detection_index, column in enumerate(costs.T):
        finite_indices = np.flatnonzero(np.isfinite(column))
        if finite_indices.size < 2:
            continue
        ordered = finite_indices[np.argsort(column[finite_indices])]
        if float(column[ordered[1]] - column[ordered[0]]) <= margin_nis:
            ambiguous_detections.add(detection_index)
            ambiguous_tracks.update(int(index) for index in ordered if column[index] <= column[ordered[0]] + margin_nis)
    return tuple(sorted(ambiguous_tracks)), tuple(sorted(ambiguous_detections))
