from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dugm_r.cli import main
from dugm_r.io import load_config


def test_full_pipeline_from_empty_outputs(tmp_path: Path):
    root = tmp_path / "run"
    main(["smoke", "--output", str(root)])
    for name in ("nominal/nominal_best.zip", "rollouts/rollouts_manifest.json", "rvf/rvf_h40.pt",
                 "rvf/rvf_h40_thresholds.json", "recovery/recovery_best.zip",
                 "evaluation_2d/dugm_nominal/summary.json", "evaluation_2d/dugm_r/summary.json",
                 "visualization/observation/dugm_observation.npz", "visualization/rvf/rvf_h40_state_sensitivity.png"):
        assert (root / name).is_file(), name
    with np.load(root / "rollouts/rollouts_00000.npz") as data:
        assert data["dugm"].dtype == np.float32
        assert set(data["split"].tolist()) == {0, 1, 2}
    with np.load(root / "visualization/observation/dugm_observation.npz") as data:
        assert data["dugm"].shape == (2, 64, 64)
        np.testing.assert_array_equal(data["dugm"][1], np.maximum(data["static_cost"], data["dynamic_cost"]))
    nominal = json.loads((root / "nominal/manifest.json").read_text())
    json.loads((root / "rvf/rvf_h40_thresholds.json").read_text(),
               parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    recovery = json.loads((root / "recovery/recovery_training.json").read_text())
    assert nominal["artifacts"]["nominal"]["sha256"] == recovery["frozen_hashes"]["nominal"]
    assert recovery["frozen_hashes"]["rvf"] == json.loads((root / "rvf/manifest.json").read_text())["artifacts"]["rvf"]["sha256"]
    eval_cfg, _ = load_config("eval_2d.yaml")
    train_cfg, _ = load_config("nominal_train_2d.yaml")
    assert {s["seed"] for s in eval_cfg["scenes"]}.isdisjoint(train_cfg["validation"]["seeds"])
