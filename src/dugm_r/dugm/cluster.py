from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
from scipy.spatial.distance import pdist

from dugm_r.dugm.preprocessor import PreprocessedScan


@dataclass(frozen=True)
class Cluster:
    detection_id: int
    centroid_xy: np.ndarray
    covariance_xy: np.ndarray
    diameter_m: float
    width_m: float
    aspect_ratio: float
    beam_indices: np.ndarray
    points_xy: np.ndarray
    footprint_offsets_xy: np.ndarray

    @property
    def point_count(self) -> int:
        return int(self.points_xy.shape[0])


@dataclass(frozen=True)
class ClusterExtractionResult:
    clusters: tuple[Cluster, ...]
    rejected_segments: tuple["RejectedSegment", ...]
    rejection_counts: dict[str, int]
    raw_segment_count: int
    merge_count: int


@dataclass(frozen=True)
class RejectedSegment:
    beam_indices: np.ndarray
    points_xy: np.ndarray
    reason: str


class OrderedClusterExtractor:
    """Linear adjacent-point segmentation with explicit 359/0 handling."""

    def __init__(
        self,
        config: dict,
        *,
        beam_unit_vectors: np.ndarray,
        angle_increment_rad: float,
        implementation: str = "optimized",
    ) -> None:
        if implementation not in {"reference", "optimized"}:
            raise ValueError("implementation must be 'reference' or 'optimized'.")
        self.implementation = implementation
        self.config = dict(config)
        self.beam_unit_vectors = np.asarray(beam_unit_vectors, dtype=np.float64)
        self.num_rays = int(self.beam_unit_vectors.shape[0])
        self.angle_increment_rad = float(angle_increment_rad)
        self.epsilon_base_m = float(config["epsilon_base_m"])
        self.epsilon_range_scale = float(config["epsilon_range_scale"])
        self.min_points = int(config["min_points"])
        self.max_diameter_m = float(config["max_diameter_m"])
        self.wraparound = bool(config.get("wraparound", True))
        self.wall_rejection = bool(config.get("wall_rejection", True))
        self.wall_min_diameter_m = float(config.get("wall_min_diameter_m", 0.8))
        self.wall_min_aspect_ratio = float(config.get("wall_min_aspect_ratio", 8.0))
        self.component_merge = bool(config.get("person_component_merge", True))
        self.merge_distance_m = float(config.get("component_merge_max_centroid_distance_m", 0.6))
        self.merge_max_diameter_m = float(config.get("component_merge_max_diameter_m", 0.85))

    def extract(self, scan: PreprocessedScan) -> ClusterExtractionResult:
        segments = self._segments(scan)
        rejected = {"min_points": 0, "max_diameter": 0, "wall_like": 0}
        rejected_segments: list[RejectedSegment] = []
        accepted: list[Cluster] = []
        for segment in segments:
            cluster, reason = self._make_cluster(segment, scan, detection_id=len(accepted))
            if reason is not None:
                rejected[reason] += 1
                points = scan.cleaned_ranges_m[segment, None] * self.beam_unit_vectors[segment]
                rejected_segments.append(RejectedSegment(segment.copy(), points, reason))
            elif cluster is not None:
                accepted.append(cluster)
        merged, merge_count = self._merge_components(accepted)
        return ClusterExtractionResult(
            tuple(replace(cluster, detection_id=index) for index, cluster in enumerate(merged)),
            tuple(rejected_segments),
            rejected,
            len(segments),
            merge_count,
        )

    def _segments(self, scan: PreprocessedScan) -> list[np.ndarray]:
        if self.implementation == "optimized":
            return self._segments_optimized(scan)
        return self._segments_reference(scan)

    def _segments_reference(self, scan: PreprocessedScan) -> list[np.ndarray]:
        segments: list[list[int]] = []
        current: list[int] = []
        for beam_id in range(self.num_rays):
            if not scan.hit_mask[beam_id]:
                if current:
                    segments.append(current)
                    current = []
                continue
            if current and self._adjacent(current[-1], beam_id, scan.cleaned_ranges_m):
                current.append(beam_id)
            else:
                if current:
                    segments.append(current)
                current = [beam_id]
        if current:
            segments.append(current)
        if (
            self.wraparound
            and len(segments) > 1
            and segments[0][0] == 0
            and segments[-1][-1] == self.num_rays - 1
            and self._adjacent(self.num_rays - 1, 0, scan.cleaned_ranges_m)
        ):
            segments[0] = segments[-1] + segments[0]
            segments.pop()
        return [np.asarray(segment, dtype=np.int32) for segment in segments]

    def _segments_optimized(self, scan: PreprocessedScan) -> list[np.ndarray]:
        """Vectorized linear segmentation with the same wraparound rule."""

        hit_ids = np.flatnonzero(scan.hit_mask).astype(np.int32)
        if hit_ids.size == 0:
            return []
        if hit_ids.size == 1:
            return [hit_ids]
        ranges = scan.cleaned_ranges_m[hit_ids]
        points = ranges[:, None] * self.beam_unit_vectors[hit_ids]
        distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
        epsilons = self.epsilon_base_m + self.epsilon_range_scale * np.minimum(ranges[1:], ranges[:-1]) * abs(
            self.angle_increment_rad
        )
        adjacent_beams = hit_ids[1:] == hit_ids[:-1] + 1
        split_indices = np.flatnonzero(~(adjacent_beams & (distances <= epsilons))) + 1
        segments = [segment.astype(np.int32, copy=False) for segment in np.split(hit_ids, split_indices)]
        if (
            self.wraparound
            and len(segments) > 1
            and segments[0][0] == 0
            and segments[-1][-1] == self.num_rays - 1
            and self._adjacent(self.num_rays - 1, 0, scan.cleaned_ranges_m)
        ):
            segments[0] = np.concatenate((segments[-1], segments[0])).astype(np.int32, copy=False)
            segments.pop()
        return segments

    def _adjacent(self, left: int, right: int, ranges: np.ndarray) -> bool:
        r_left = float(ranges[left])
        r_right = float(ranges[right])
        p_left = r_left * self.beam_unit_vectors[left]
        p_right = r_right * self.beam_unit_vectors[right]
        distance = float(np.linalg.norm(p_left - p_right))
        epsilon = self.epsilon_base_m + self.epsilon_range_scale * min(r_left, r_right) * abs(self.angle_increment_rad)
        return distance <= epsilon

    def _make_cluster(
        self, beam_indices: np.ndarray, scan: PreprocessedScan, *, detection_id: int
    ) -> tuple[Cluster | None, str | None]:
        if beam_indices.size < self.min_points:
            return None, "min_points"
        points = scan.cleaned_ranges_m[beam_indices, None] * self.beam_unit_vectors[beam_indices]
        diameter = self._diameter(points)
        if diameter > self.max_diameter_m:
            return None, "max_diameter"
        centroid, centered, covariance, extents = self._cluster_geometry(points)
        length = float(max(extents[0], diameter))
        width = float(max(extents[1], 1e-6))
        aspect_ratio = length / width
        if self.wall_rejection and diameter >= self.wall_min_diameter_m and aspect_ratio >= self.wall_min_aspect_ratio:
            return None, "wall_like"
        return (
            Cluster(
                detection_id=detection_id,
                centroid_xy=centroid.astype(np.float64),
                covariance_xy=covariance,
                diameter_m=diameter,
                width_m=width,
                aspect_ratio=aspect_ratio,
                beam_indices=beam_indices.copy(),
                points_xy=points.astype(np.float64),
                footprint_offsets_xy=centered.astype(np.float64),
            ),
            None,
        )

    def _merge_components(self, clusters: list[Cluster]) -> tuple[list[Cluster], int]:
        if not self.component_merge or len(clusters) < 2:
            return clusters, 0
        active = list(clusters)
        merge_count = 0
        changed = True
        while changed:
            changed = False
            for left_index, right_index in combinations(range(len(active)), 2):
                left, right = active[left_index], active[right_index]
                if np.linalg.norm(left.centroid_xy - right.centroid_xy) > self.merge_distance_m:
                    continue
                points = np.concatenate((left.points_xy, right.points_xy), axis=0)
                if self._diameter(points) > self.merge_max_diameter_m:
                    continue
                beams = np.concatenate((left.beam_indices, right.beam_indices))
                # Recompute through the same geometry representation without
                # applying wall rejection a second time.
                centroid, centered, covariance, extents = self._cluster_geometry(points)
                diameter = self._diameter(points)
                merged = Cluster(
                    detection_id=left.detection_id,
                    centroid_xy=centroid,
                    covariance_xy=np.asarray(covariance).reshape(2, 2),
                    diameter_m=diameter,
                    width_m=float(max(extents[1], 1e-6)),
                    aspect_ratio=float(max(extents[0], diameter) / max(extents[1], 1e-6)),
                    beam_indices=np.sort(beams),
                    points_xy=points,
                    footprint_offsets_xy=centered,
                )
                active = [item for index, item in enumerate(active) if index not in (left_index, right_index)]
                active.append(merged)
                merge_count += 1
                changed = True
                break
            if changed:
                continue
        active.sort(key=lambda item: int(np.min(item.beam_indices)))
        return active, merge_count

    def _diameter(self, points: np.ndarray) -> float:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(points) < 2:
            return 0.0
        if self.implementation == "optimized" and len(points) >= 8:
            return float(np.max(pdist(points, metric="euclidean")))
        return _diameter(points)

    def _cluster_geometry(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        if len(points) < 2:
            return centroid, centered, np.zeros((2, 2), dtype=np.float64), np.zeros(2, dtype=np.float64)
        if self.implementation == "reference":
            covariance = np.asarray(np.cov(points.T, bias=False), dtype=np.float64).reshape(2, 2)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            projected = centered @ eigenvectors[:, np.argsort(eigenvalues)[::-1]]
            return centroid, centered, covariance, np.ptp(projected, axis=0)

        covariance = (centered.T @ centered) / float(len(points) - 1)
        # Closed-form principal axes avoid one LAPACK dispatch per noisy scan
        # fragment. Axis signs do not affect the projected extents.
        angle = 0.5 * np.arctan2(2.0 * covariance[0, 1], covariance[0, 0] - covariance[1, 1])
        cosine, sine = np.cos(angle), np.sin(angle)
        major = centered[:, 0] * cosine + centered[:, 1] * sine
        minor = -centered[:, 0] * sine + centered[:, 1] * cosine
        extents = np.asarray([np.ptp(major), np.ptp(minor)], dtype=np.float64)
        return centroid, centered, covariance, extents


def _diameter(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        return 0.0
    deltas = points[:, None, :] - points[None, :, :]
    return float(np.sqrt(np.max(np.sum(deltas * deltas, axis=2))))
