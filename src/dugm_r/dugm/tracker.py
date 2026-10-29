from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dugm_r.dugm.cluster import Cluster
from dugm_r.dugm.association import AssociationResult, associate
from dugm_r.dugm.classifier import update_dynamic_semantic
from dugm_r.dugm.track import KalmanTrack


@dataclass(frozen=True)
class TrackerStepResult:
    association: AssociationResult
    deleted_track_ids: tuple[int, ...]
    created_track_ids: tuple[int, ...]
    matched_nis: tuple[float, ...]
    maneuver_triggered_track_ids: tuple[int, ...]
    maneuver_exited_track_ids: tuple[int, ...]


class MultiObjectTracker:
    def __init__(self, tracking_config: dict, dynamic_config: dict, *, implementation: str = "optimized") -> None:
        self.tracking_config = dict(tracking_config)
        self.dynamic_config = dict(dynamic_config)
        if self.tracking_config.get("max_tracks") is not None:
            raise ValueError("DUGM tracker forbids a fixed max_tracks cap.")
        self.maneuver_config = dict(self.tracking_config["maneuver_mode"])
        self.implementation = implementation
        self.tracks: list[KalmanTrack] = []
        self.next_track_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.next_track_id = 1

    def initialize(self, detections: tuple[Cluster, ...], *, timestamp_s: float) -> TrackerStepResult:
        empty = associate(
            [], detections,
            gate_chi2=float(self.tracking_config["association_gate_chi2"]),
            measurement_position_std_floor_m=float(self.tracking_config["measurement_position_std_floor_m"]),
            ambiguity_margin_nis=float(self.maneuver_config["association_ambiguity_margin_nis"]),
            implementation=self.implementation,
        )
        created = tuple(self._create(detection, timestamp_s=timestamp_s).track_id for detection in detections)
        for track in self.tracks:
            self._update_lifecycle(track)
        return TrackerStepResult(empty, (), created, (), (), ())

    def step(
        self,
        detections: tuple[Cluster, ...],
        *,
        previous_pose_xyyaw: np.ndarray,
        current_pose_xyyaw: np.ndarray,
        previous_timestamp_s: float,
        current_timestamp_s: float,
    ) -> TrackerStepResult:
        dt = float(current_timestamp_s - previous_timestamp_s)
        if dt <= 0.0:
            raise ValueError("Tracker step requires a strictly increasing timestamp.")
        for track in self.tracks:
            acceleration_std = (
                float(self.maneuver_config["process_acceleration_std_mps2"])
                if track.in_maneuver_mode
                else float(self.tracking_config["process_acceleration_std_mps2"])
            )
            track.predict(
                previous_pose_xyyaw=previous_pose_xyyaw,
                current_pose_xyyaw=current_pose_xyyaw,
                dt_s=dt,
                acceleration_std_mps2=acceleration_std,
            )
        association = associate(
            self.tracks,
            detections,
            gate_chi2=float(self.tracking_config["association_gate_chi2"]),
            measurement_position_std_floor_m=float(self.tracking_config["measurement_position_std_floor_m"]),
            ambiguity_margin_nis=float(self.maneuver_config["association_ambiguity_margin_nis"]),
            implementation=self.implementation,
        )
        matched_nis: list[float] = []
        nis_by_track_index: dict[int, float] = {}
        for track_index, detection_index in association.matches:
            nis = self.tracks[track_index].update(
                detections[detection_index],
                timestamp_s=current_timestamp_s,
                position_std_floor_m=float(self.tracking_config["measurement_position_std_floor_m"]),
            )
            matched_nis.append(nis)
            nis_by_track_index[track_index] = nis
        for track_index in association.unmatched_track_indices:
            self.tracks[track_index].mark_missed()
        maneuver_triggered_ids: list[int] = []
        maneuver_exited_ids: list[int] = []
        ambiguous_indices = set(association.ambiguous_track_indices)
        unmatched_indices = set(association.unmatched_track_indices)
        for track_index, track in enumerate(self.tracks):
            trigger_reasons: list[str] = []
            nis = nis_by_track_index.get(track_index)
            elevated_innovation = False
            if nis is not None:
                elevated_innovation = nis >= float(self.maneuver_config["persistent_innovation_nis"])
                track.elevated_nis_count = track.elevated_nis_count + 1 if elevated_innovation else 0
            if nis is not None and nis >= float(self.maneuver_config["trigger_nis_chi2"]):
                trigger_reasons.append("high_nis")
            if track.elevated_nis_count >= int(self.maneuver_config["persistent_innovation_frames"]):
                trigger_reasons.append("persistent_innovation")
            if track_index in ambiguous_indices:
                trigger_reasons.append("association_ambiguity")
            if (
                track_index in unmatched_indices
                and track.confirmed
                and bool(self.maneuver_config["trigger_on_confirmed_miss"])
            ):
                trigger_reasons.append("confirmed_miss")
            if trigger_reasons:
                track.enter_maneuver_mode("+".join(trigger_reasons))
                maneuver_triggered_ids.append(track.track_id)
            elif nis is not None:
                if elevated_innovation:
                    track.hold_maneuver_mode()
                elif track.record_stable_maneuver_frame(
                    exit_stable_frames=int(self.maneuver_config["exit_stable_frames"])
                ):
                    maneuver_exited_ids.append(track.track_id)
            else:
                track.hold_maneuver_mode()
        created_ids = tuple(
            self._create(detections[detection_index], timestamp_s=current_timestamp_s).track_id
            for detection_index in association.unmatched_detection_indices
        )
        for track in self.tracks:
            self._update_lifecycle(track)
        deleted_ids = tuple(track.track_id for track in self.tracks if self._should_delete(track))
        self.tracks = [track for track in self.tracks if track.track_id not in set(deleted_ids)]
        return TrackerStepResult(
            association,
            deleted_ids,
            created_ids,
            tuple(matched_nis),
            tuple(maneuver_triggered_ids),
            tuple(maneuver_exited_ids),
        )

    def _create(self, detection: Cluster, *, timestamp_s: float) -> KalmanTrack:
        track = KalmanTrack.from_detection(
            self.next_track_id,
            detection,
            timestamp_s,
            self.tracking_config,
        )
        self.next_track_id += 1
        self.tracks.append(track)
        return track

    def _update_lifecycle(self, track: KalmanTrack) -> None:
        if not track.confirmed and track.hit_count >= int(self.tracking_config["min_hits"]):
            track.confirmed = True
        update_dynamic_semantic(track, self.dynamic_config)

    def _should_delete(self, track: KalmanTrack) -> bool:
        limit = (
            int(self.tracking_config["max_misses_confirmed"])
            if track.confirmed
            else int(self.tracking_config["max_misses_tentative"])
        )
        return track.miss_count > limit

    def snapshots(self) -> list[dict]:
        return [
            {
                "track_id": track.track_id,
                "state": track.state.astype(float).tolist(),
                "covariance": track.covariance.astype(float).tolist(),
                "age": track.age,
                "hit_count": track.hit_count,
                "consecutive_hits": track.consecutive_hits,
                "miss_count": track.miss_count,
                "confirmed": track.confirmed,
                "semantic_state": track.semantic_state,
                "dynamic_evidence_count": track.dynamic_evidence_count,
                "current_measurement_available": track.current_measurement_available,
                "velocity_std_mps": track.velocity_std_mps,
                "motion_mode": track.motion_mode,
                "maneuver_stable_count": track.maneuver_stable_count,
                "elevated_nis_count": track.elevated_nis_count,
                "last_maneuver_trigger": track.last_maneuver_trigger,
                "maneuver_entry_count": track.maneuver_entry_count,
                "last_process_acceleration_std_mps2": track.last_process_acceleration_std_mps2,
            }
            for track in self.tracks
        ]
