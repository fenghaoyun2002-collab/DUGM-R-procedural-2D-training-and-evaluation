"""Train H40 RVF from nominal rollouts and select a validation threshold."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from dugm_r.io import load_config, record_artifact, require_empty_output, sha256, write_json
from dugm_r.recovery.calibration import binary_metrics, calibrate_threshold
from dugm_r.recovery.rvf import build_model, save_checkpoint


def _shard_paths(manifest_path: Path) -> tuple[list[Path], dict[str, Any]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    return [manifest_path.parent / row["file"] for row in manifest["shards"]], manifest


def _counts(paths: list[Path], split: int) -> tuple[int, int]:
    positives = total = 0
    for path in paths:
        with np.load(path) as data:
            mask = data["split"] == split
            labels = data["label_h40"][mask]
            positives += int(labels.sum())
            total += len(labels)
    return positives, total


def _score_split(model: nn.Module, paths: list[Path], *, split: int, device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    episodes_all: list[np.ndarray] = []
    steps_all: list[np.ndarray] = []
    collisions_all: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for path in paths:
            with np.load(path) as data:
                mask = data["split"] == split
                indices = np.flatnonzero(mask)
                labels_all.append(data["label_h40"][indices].astype(bool))
                episodes_all.append(data["episode_id"][indices])
                steps_all.append(data["step_index"][indices])
                collisions_all.append(data["collision"][indices].astype(bool))
                grid = data["dugm"]
                state = data["state"]
                for batch in np.array_split(indices, max(1, int(np.ceil(len(indices) / batch_size)))):
                    if len(batch) == 0:
                        continue
                    g = torch.as_tensor(grid[batch], dtype=torch.float32, device=device)
                    s = torch.as_tensor(state[batch], dtype=torch.float32, device=device)
                    scores_all.append(model.score(g, s).cpu().numpy().astype(np.float32))
    def cat(items, dtype):
        return np.concatenate(items).astype(dtype) if items else np.empty(0, dtype=dtype)
    return cat(labels_all, bool), cat(scores_all, np.float32), cat(episodes_all, np.int32), cat(steps_all, np.int32), cat(collisions_all, bool)


def train_rvf(config_name: str | Path, rollout_manifest: Path, output: Path, *, smoke: bool = False) -> tuple[Path, Path]:
    cfg, cfg_path = load_config(config_name)
    if int(cfg["horizon"]) != 40:
        raise ValueError("This DUGM-R release uses H40 labels.")
    paths, rollout_info = _shard_paths(rollout_manifest)
    if int(rollout_info["horizon"]) != 40:
        raise ValueError("Rollout labels do not match H40 RVF configuration.")
    if not paths:
        raise ValueError("No rollout shards found.")
    require_empty_output(output)
    torch.manual_seed(int(cfg["seed"]))
    torch.set_num_threads(1)
    device = "cpu" if smoke else str(cfg["training"].get("device", "cpu"))
    model = build_model(cfg["model"]).to(device)
    positive_count, train_count = _counts(paths, 0)
    if train_count == 0:
        raise ValueError("No train split samples.")
    weight = max(1.0, (train_count - positive_count) / positive_count) if positive_count else 1.0
    configured_weight = cfg["training"].get("positive_weight", "auto")
    if configured_weight not in ("auto", None):
        weight = float(configured_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    epochs = 1 if smoke else int(cfg["training"]["epochs"])
    batch_size = min(16, int(cfg["training"]["batch_size"])) if smoke else int(cfg["training"]["batch_size"])
    rng = np.random.default_rng(int(cfg["seed"]))
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for shard_index in rng.permutation(len(paths)):
            with np.load(paths[int(shard_index)]) as data:
                indices = np.flatnonzero(data["split"] == 0)
                rng.shuffle(indices)
                grid, state, labels = data["dugm"], data["state"], data["label_h40"]
                for start in range(0, len(indices), batch_size):
                    batch = indices[start:start + batch_size]
                    g = torch.as_tensor(grid[batch], dtype=torch.float32, device=device)
                    s = torch.as_tensor(state[batch], dtype=torch.float32, device=device)
                    y = torch.as_tensor(labels[batch], dtype=torch.float32, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(g, s), y)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)) if losses else None})
    checkpoint = output / "rvf_h40.pt"
    save_checkpoint(checkpoint, model, cfg["model"], {"horizon": 40, "positive_weight": weight, "rollout_manifest_sha256": sha256(rollout_manifest)})
    val_labels, val_scores, episode_ids, step_indices, collisions = _score_split(model, paths, split=1, device=device, batch_size=batch_size)
    tcfg = cfg["thresholds"]
    if len(val_labels):
        calibration = calibrate_threshold(
            val_labels, val_scores, episode_ids=episode_ids, step_index=step_indices, collision=collisions,
            calibration_mode=str(tcfg["calibration_mode"]), recall_target=float(tcfg["dangerous_recall_target"]),
            early_warning_recall_target=float(tcfg["early_warning_recall_target"]),
            max_false_positive_rate=float(tcfg["max_false_positive_rate"]),
            max_false_trigger_episode_rate=float(tcfg["max_false_trigger_episode_rate"]),
            min_mean_warning_time_s=float(tcfg["min_mean_warning_time_s"]),
            warning_time_target_s=float(tcfg["warning_steps_target"]) * float(tcfg["step_time_s"]),
            step_time_s=float(tcfg["step_time_s"]),
            epsilon_return_fraction=float(tcfg["epsilon_return_fraction"]),
            max_thresholds=int(tcfg["max_sweep_thresholds"]),
        )
    else:
        calibration = {"epsilon_on": 1.0, "epsilon_return": 0.5, "feasible": False, "selection_reason": "no_validation_samples"}
    thresholds = output / "rvf_h40_thresholds.json"
    write_json(thresholds, calibration)
    test_labels, test_scores, _, _, _ = _score_split(model, paths, split=2, device=device, batch_size=batch_size)
    write_json(output / "rvf_training.json", {
        "history": history, "train_samples": train_count, "train_positive_count": positive_count,
        "validation": binary_metrics(val_labels, val_scores, threshold=float(calibration["epsilon_on"])),
        "test": binary_metrics(test_labels, test_scores, threshold=float(calibration["epsilon_on"])),
    })
    record_artifact(output, "rvf", checkpoint, metadata={"horizon": 40, "config_sha256": sha256(cfg_path)})
    record_artifact(output, "thresholds", thresholds)
    return checkpoint, thresholds
