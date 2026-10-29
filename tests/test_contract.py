from __future__ import annotations

import numpy as np

from dugm_r.recovery.rollouts import future_collision_labels
from dugm_r.recovery.switch import HysteresisSwitch
from dugm_r.simulation_2d.env import Dugm2DEnv


def test_observation_is_sensor_derived_continuous_and_deterministic():
    first = Dugm2DEnv(level=8, seed=123, observation_noise=False)
    second = Dugm2DEnv(level=8, seed=123, observation_noise=False)
    try:
        a, _ = first.reset(seed=123)
        b, _ = second.reset(seed=123)
        assert set(a) == {"dugm", "state"}
        assert first.observation_space.contains(a)
        assert a["dugm"].shape == (2, 64, 64)
        assert a["state"].shape == (4,)
        assert a["dugm"].dtype == a["state"].dtype == np.float32
        assert first.ground_truth_used_for_observation is False
        assert first.ground_truth_allowed_as_estimator_input is False
        np.testing.assert_array_equal(a["dugm"], b["dugm"])
        for _ in range(6):
            a, _, _, _, _ = first.step(np.array([0.2, 0.1], dtype=np.float32))
            b, _, _, _, _ = second.step(np.array([0.2, 0.1], dtype=np.float32))
            np.testing.assert_array_equal(a["dugm"], b["dugm"])
            np.testing.assert_array_equal(a["state"], b["state"])
        assert np.any((a["dugm"][1] > 0) & (a["dugm"][1] < 1))
        np.testing.assert_array_equal(a["dugm"][1], np.maximum(first.last_static_cost, first.last_dynamic_cost))
        reset_obs, _ = first.reset(seed=123)
        fresh_obs, _ = second.reset(seed=123)
        np.testing.assert_array_equal(first.command, np.zeros(2, dtype=np.float32))
        np.testing.assert_array_equal(reset_obs["dugm"], fresh_obs["dugm"])
    finally:
        first.close()
        second.close()


def test_h40_labels_do_not_cross_episode_boundary():
    a = future_collision_labels(np.array([0, 0, 1], dtype=bool), 2)
    b = future_collision_labels(np.array([0, 0, 0], dtype=bool), 2)
    np.testing.assert_array_equal(a, [1, 1, 1])
    np.testing.assert_array_equal(b, [0, 0, 0])


def test_hysteresis_requires_dwell_and_release_steps():
    switch = HysteresisSwitch(epsilon_on=0.8, epsilon_return=0.4,
                              min_recovery_dwell_steps=2, return_dwell_steps=2)
    assert switch.update(0.7) == ("nominal", False, False)
    assert switch.update(0.9) == ("recovery", True, False)
    assert switch.update(0.3) == ("recovery", False, False)
    assert switch.update(0.3) == ("recovery", False, False)
    assert switch.update(0.3) == ("nominal", False, True)
    switch.reset()
    assert switch.update(0.7) == ("nominal", False, False)
