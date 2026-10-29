"""PPO configuration shared by nominal and recovery training."""
from __future__ import annotations

from typing import Any

from stable_baselines3 import PPO

from dugm_r.nominal.features import DugmFeaturesExtractor


def ppo_from_config(env: Any, settings: dict[str, Any], *, seed: int, device: str = "cpu") -> PPO:
    values = dict(settings)
    values.pop("total_timesteps", None)
    values.pop("n_envs", None)
    extractor = dict(values.pop("features_extractor_kwargs", {}))
    kwargs = dict(values.pop("policy_kwargs", {}))
    kwargs["features_extractor_class"] = DugmFeaturesExtractor
    kwargs["features_extractor_kwargs"] = extractor
    values["policy_kwargs"] = kwargs
    return PPO("MultiInputPolicy", env, seed=int(seed), device=device, verbose=0, **values)
