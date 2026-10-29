"""Recovery PPO training with frozen nominal policy and RVF."""
from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from dugm_r.io import load_config, record_artifact, require_empty_output, sha256, write_json
from dugm_r.nominal.policy import ppo_from_config
from dugm_r.recovery.env import RecoveryTrainingEnv


def train_recovery(config_name: str | Path, nominal_checkpoint: Path, rvf_checkpoint: Path,
                   thresholds_path: Path, output: Path, *, smoke: bool = False,
                   warm_start_checkpoint: Path | None = None) -> Path:
    cfg, cfg_path = load_config(config_name)
    if cfg.get("method") != "dugm_r":
        raise ValueError("Expected DUGM-R recovery configuration.")
    for path in (nominal_checkpoint, rvf_checkpoint, thresholds_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if warm_start_checkpoint is not None and not warm_start_checkpoint.is_file():
        raise FileNotFoundError(warm_start_checkpoint)
    before = {"nominal": sha256(nominal_checkpoint), "rvf": sha256(rvf_checkpoint), "thresholds": sha256(thresholds_path)}
    require_empty_output(output)
    ppo_cfg = dict(cfg["ppo"])
    n_envs = 1 if smoke else int(ppo_cfg["n_envs"])
    def factory(index: int):
        return lambda: RecoveryTrainingEnv(
            config_name=config_name, nominal_checkpoint=nominal_checkpoint, rvf_checkpoint=rvf_checkpoint,
            thresholds_path=thresholds_path, seed=int(cfg["seed"]) + index, smoke=smoke,
        )
    factories = [factory(i) for i in range(n_envs)]
    vec = DummyVecEnv(factories) if smoke or n_envs == 1 else SubprocVecEnv(factories, start_method="fork")
    try:
        if smoke:
            ppo_cfg.update({"n_steps": 16, "batch_size": 16, "n_epochs": 1})
        if warm_start_checkpoint is None:
            model = ppo_from_config(vec, ppo_cfg, seed=int(cfg["seed"]), device="cpu")
        else:
            from stable_baselines3 import PPO
            from dugm_r.nominal.features import DugmFeaturesExtractor  # checkpoint class registration
            overrides = {key: value for key, value in ppo_cfg.items()
                         if key not in {"total_timesteps", "n_envs", "policy_kwargs", "features_extractor_kwargs"}}
            model = PPO.load(warm_start_checkpoint, env=vec, device="cpu", **overrides)
        steps = 16 if smoke else int(cfg["ppo"]["total_timesteps"])
        model.learn(total_timesteps=steps, reset_num_timesteps=warm_start_checkpoint is None, progress_bar=False)
        checkpoint = output / "recovery_best.zip"
        model.save(checkpoint)
    finally:
        vec.close()
    after = {"nominal": sha256(nominal_checkpoint), "rvf": sha256(rvf_checkpoint), "thresholds": sha256(thresholds_path)}
    if before != after:
        raise RuntimeError("Frozen nominal, RVF, or threshold artifact changed during recovery training.")
    write_json(output / "recovery_training.json", {"method": "dugm_r", "steps": steps, "frozen_hashes": before,
                                                  "checkpoint_selection": "final_training_step",
                                                  "warm_start_sha256": sha256(warm_start_checkpoint) if warm_start_checkpoint is not None else None})
    record_artifact(output, "recovery", checkpoint, metadata={"config_sha256": sha256(cfg_path), "frozen_hashes": before,
                                                               "warm_start_sha256": sha256(warm_start_checkpoint) if warm_start_checkpoint is not None else None})
    return checkpoint
