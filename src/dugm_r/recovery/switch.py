"""Risk-triggered nominal/recovery policy switching with hysteresis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dugm_r.recovery.rvf import RiskValueNet, risk_score


@dataclass(frozen=True)
class SwitchingDecision:
    mode: str
    risk_score: float
    action: np.ndarray
    triggered: bool
    released: bool


class HysteresisSwitch:
    def __init__(self, *, epsilon_on: float, epsilon_return: float,
                 min_recovery_dwell_steps: int = 4, return_dwell_steps: int = 3) -> None:
        if not 0 <= epsilon_return <= epsilon_on <= 1:
            raise ValueError("Expected 0 <= epsilon_return <= epsilon_on <= 1.")
        if min_recovery_dwell_steps < 1 or return_dwell_steps < 1:
            raise ValueError("Dwell steps must be positive.")
        self.epsilon_on = float(epsilon_on)
        self.epsilon_return = float(epsilon_return)
        self.min_recovery_dwell_steps = int(min_recovery_dwell_steps)
        self.return_dwell_steps = int(return_dwell_steps)
        self.reset()

    def reset(self) -> None:
        self.recovery_active = False
        self.recovery_step_count = 0
        self.return_dwell_count = 0

    def update(self, score: float) -> tuple[str, bool, bool]:
        triggered = released = False
        if self.recovery_active:
            if self.recovery_step_count >= self.min_recovery_dwell_steps and score <= self.epsilon_return:
                self.return_dwell_count += 1
            else:
                self.return_dwell_count = 0
            if self.return_dwell_count >= self.return_dwell_steps:
                self.recovery_active = False
                self.recovery_step_count = 0
                self.return_dwell_count = 0
                released = True
        elif score >= self.epsilon_on:
            self.recovery_active = True
            self.recovery_step_count = 0
            self.return_dwell_count = 0
            triggered = True
        if self.recovery_active:
            self.recovery_step_count += 1
        return ("recovery" if self.recovery_active else "nominal", triggered, released)


class DugmRPolicy:
    def __init__(self, nominal: Any, recovery: Any, rvf: RiskValueNet,
                 thresholds: dict[str, float], *, device: str = "cpu",
                 min_recovery_dwell_steps: int = 4, return_dwell_steps: int = 3) -> None:
        self.nominal = nominal
        self.recovery = recovery
        self.rvf = rvf
        self.device = device
        self.switch = HysteresisSwitch(
            epsilon_on=float(thresholds["epsilon_on"]),
            epsilon_return=float(thresholds["epsilon_return"]),
            min_recovery_dwell_steps=min_recovery_dwell_steps,
            return_dwell_steps=return_dwell_steps,
        )

    def reset(self) -> None:
        self.switch.reset()

    def predict(self, observation: dict[str, np.ndarray], *, deterministic: bool = True) -> SwitchingDecision:
        score = risk_score(self.rvf, observation, device=self.device)
        mode, triggered, released = self.switch.update(score)
        model = self.recovery if mode == "recovery" else self.nominal
        action, _ = model.predict(observation, deterministic=deterministic)
        return SwitchingDecision(mode, score, np.asarray(action, dtype=np.float32).reshape(2), triggered, released)
