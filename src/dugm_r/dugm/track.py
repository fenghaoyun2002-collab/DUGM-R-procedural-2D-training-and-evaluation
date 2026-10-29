from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dugm_r.dugm.cluster import Cluster


SEMANTIC_TENTATIVE = "tentative"
SEMANTIC_STATIC_OR_UNKNOWN = "confirmed_static_or_unknown"
SEMANTIC_DYNAMIC = "confirmed_dynamic"
MOTION_MODE_NORMAL = "normal"
MOTION_MODE_MANEUVER = "maneuver"


def rotation_old_to_current(previous_pose_xyyaw: np.ndarray, current_pose_xyyaw: np.ndarray) -> np.ndarray:
    delta = float(previous_pose_xyyaw[2] - current_pose_xyyaw[2])
    cosine, sine = np.cos(delta), np.sin(delta)
    return np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def world_point_to_robot(point_world_xy: np.ndarray, pose_xyyaw: np.ndarray) -> np.ndarray:
    cosine, sine = np.cos(float(pose_xyyaw[2])), np.sin(float(pose_xyyaw[2]))
    world_to_robot = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    return world_to_robot @ (np.asarray(point_world_xy, dtype=np.float64) - pose_xyyaw[:2])


@dataclass
class KalmanTrack:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    age: int
    hit_count: int
    consecutive_hits: int
    miss_count: int
    confirmed: bool
    semantic_state: str
    dynamic_evidence_count: int
    current_measurement_available: bool
    last_measurement_timestamp: float
    latest_cluster: Cluster | None
    motion_mode: str
    maneuver_stable_count: int
    elevated_nis_count: int
    last_maneuver_trigger: str | None
    maneuver_entry_count: int
    last_process_acceleration_std_mps2: float

    @classmethod
    def from_detection(cls, track_id: int, detection: Cluster, timestamp_s: float, config: dict) -> "KalmanTrack":
        position_std = float(config["initial_position_std_m"])
        velocity_std = float(config["initial_velocity_std_mps"])
        state = np.asarray([*detection.centroid_xy, 0.0, 0.0], dtype=np.float64)
        covariance = np.diag([position_std**2, position_std**2, velocity_std**2, velocity_std**2])
        return cls(
            track_id=int(track_id),
            state=state,
            covariance=covariance,
            age=1,
            hit_count=1,
            consecutive_hits=1,
            miss_count=0,
            confirmed=int(config["min_hits"]) <= 1,
            semantic_state=SEMANTIC_TENTATIVE,
            dynamic_evidence_count=0,
            current_measurement_available=True,
            last_measurement_timestamp=float(timestamp_s),
            latest_cluster=detection,
            motion_mode=MOTION_MODE_NORMAL,
            maneuver_stable_count=0,
            elevated_nis_count=0,
            last_maneuver_trigger=None,
            maneuver_entry_count=0,
            last_process_acceleration_std_mps2=float(config.get("process_acceleration_std_mps2", 0.0)),
        )

    def predict(
        self,
        *,
        previous_pose_xyyaw: np.ndarray,
        current_pose_xyyaw: np.ndarray,
        dt_s: float,
        acceleration_std_mps2: float,
    ) -> None:
        dt = float(dt_s)
        rotation = rotation_old_to_current(previous_pose_xyyaw, current_pose_xyyaw)
        previous_origin_in_current = world_point_to_robot(previous_pose_xyyaw[:2], current_pose_xyyaw)
        transition = np.zeros((4, 4), dtype=np.float64)
        transition[:2, :2] = rotation
        transition[:2, 2:] = dt * rotation
        transition[2:, 2:] = rotation
        self.state = transition @ self.state
        self.state[:2] += previous_origin_in_current
        acceleration_map = np.vstack((0.5 * dt * dt * rotation, dt * rotation))
        process_noise = (float(acceleration_std_mps2) ** 2) * (acceleration_map @ acceleration_map.T)
        self.covariance = transition @ self.covariance @ transition.T + process_noise
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.age += 1
        self.current_measurement_available = False
        self.last_process_acceleration_std_mps2 = float(acceleration_std_mps2)

    def measurement_covariance(self, detection: Cluster, position_std_floor_m: float) -> np.ndarray:
        cluster_mean_cov = detection.covariance_xy / max(1, detection.point_count)
        return cluster_mean_cov + np.eye(2, dtype=np.float64) * float(position_std_floor_m) ** 2

    def innovation(self, detection: Cluster, position_std_floor_m: float) -> tuple[np.ndarray, np.ndarray, float]:
        residual = detection.centroid_xy - self.state[:2]
        innovation_cov = self.covariance[:2, :2] + self.measurement_covariance(detection, position_std_floor_m)
        nis = float(residual.T @ np.linalg.solve(innovation_cov, residual))
        return residual, innovation_cov, nis

    def update(self, detection: Cluster, *, timestamp_s: float, position_std_floor_m: float) -> float:
        residual, innovation_cov, nis = self.innovation(detection, position_std_floor_m)
        gain = self.covariance[:, :2] @ np.linalg.inv(innovation_cov)
        self.state = self.state + gain @ residual
        identity = np.eye(4, dtype=np.float64)
        measurement_matrix = np.zeros((2, 4), dtype=np.float64)
        measurement_matrix[:, :2] = np.eye(2)
        measurement_cov = self.measurement_covariance(detection, position_std_floor_m)
        # Joseph form keeps the reference covariance symmetric and PSD.
        residual_map = identity - gain @ measurement_matrix
        self.covariance = residual_map @ self.covariance @ residual_map.T + gain @ measurement_cov @ gain.T
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.hit_count += 1
        self.consecutive_hits += 1
        self.miss_count = 0
        self.current_measurement_available = True
        self.last_measurement_timestamp = float(timestamp_s)
        self.latest_cluster = detection
        return nis

    def mark_missed(self) -> None:
        self.miss_count += 1
        self.consecutive_hits = 0
        self.current_measurement_available = False
        self.latest_cluster = None
        self.elevated_nis_count = 0

    def enter_maneuver_mode(self, trigger: str) -> bool:
        """Enter/refresh maneuver mode and return whether this is a new entry."""

        entered = self.motion_mode != MOTION_MODE_MANEUVER
        self.motion_mode = MOTION_MODE_MANEUVER
        self.maneuver_stable_count = 0
        self.last_maneuver_trigger = str(trigger)
        if entered:
            self.maneuver_entry_count += 1
        return entered

    def hold_maneuver_mode(self) -> None:
        if self.motion_mode == MOTION_MODE_MANEUVER:
            self.maneuver_stable_count = 0

    def record_stable_maneuver_frame(self, *, exit_stable_frames: int) -> bool:
        """Count one clear measured frame and return whether normal mode resumed."""

        if self.motion_mode != MOTION_MODE_MANEUVER:
            return False
        self.maneuver_stable_count += 1
        if self.maneuver_stable_count < int(exit_stable_frames):
            return False
        self.motion_mode = MOTION_MODE_NORMAL
        self.maneuver_stable_count = 0
        return True

    @property
    def in_maneuver_mode(self) -> bool:
        return self.motion_mode == MOTION_MODE_MANEUVER

    @property
    def velocity_std_mps(self) -> float:
        return float(np.sqrt(max(0.0, np.max(np.diag(self.covariance)[2:]))))
