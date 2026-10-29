from __future__ import annotations

import numpy as np

from dugm_r.dugm.track import (
    KalmanTrack,
    SEMANTIC_DYNAMIC,
    SEMANTIC_STATIC_OR_UNKNOWN,
    SEMANTIC_TENTATIVE,
)


def update_dynamic_semantic(track: KalmanTrack, config: dict) -> None:
    """Apply V1 sticky dynamic semantics after a matched measurement."""

    if not track.confirmed:
        track.semantic_state = SEMANTIC_TENTATIVE
        track.dynamic_evidence_count = 0
        return
    if track.semantic_state == SEMANTIC_DYNAMIC:
        return
    track.semantic_state = SEMANTIC_STATIC_OR_UNKNOWN
    speed = float(np.linalg.norm(track.state[2:]))
    covariance_ok = track.velocity_std_mps <= float(config["require_velocity_std_below_mps"])
    if track.current_measurement_available and speed >= float(config["entry_speed_mps"]) and covariance_ok:
        track.dynamic_evidence_count += 1
    else:
        track.dynamic_evidence_count = 0
    if (
        track.age >= int(config["entry_min_age"])
        and track.dynamic_evidence_count >= int(config["entry_min_consecutive"])
    ):
        track.semantic_state = SEMANTIC_DYNAMIC
