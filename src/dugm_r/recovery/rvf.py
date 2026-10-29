"""Finite-horizon Risk Value Function on the actual DUGM policy observation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class RiskValueNet(nn.Module):
    def __init__(self, *, dugm_channels: int = 2, state_dim: int = 4,
                 dugm_features_dim: int = 128, state_features_dim: int = 32,
                 fusion_hidden_sizes: tuple[int, ...] = (128, 64), dropout: float = 0.05) -> None:
        super().__init__()
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(dugm_channels, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        )
        self.grid_projection = nn.Sequential(nn.Linear(64 * 4 * 4, dugm_features_dim), nn.ReLU())
        self.state_branch = nn.Sequential(nn.Linear(state_dim, state_features_dim), nn.ReLU())
        layers: list[nn.Module] = []
        current = dugm_features_dim + state_features_dim
        for hidden in fusion_hidden_sizes:
            layers.extend((nn.Linear(current, int(hidden)), nn.ReLU()))
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            current = int(hidden)
        layers.append(nn.Linear(current, 1))
        self.fusion = nn.Sequential(*layers)

    def forward(self, dugm: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        grid = self.grid_projection(self.grid_cnn(dugm.float()))
        task = self.state_branch(state.float())
        return self.fusion(torch.cat((grid, task), dim=1)).squeeze(-1)

    def score(self, dugm: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Uncalibrated risk score; weighted BCE does not make it a collision probability."""
        return torch.sigmoid(self.forward(dugm, state))


def build_model(config: dict[str, Any]) -> RiskValueNet:
    values = dict(config)
    values["fusion_hidden_sizes"] = tuple(values.get("fusion_hidden_sizes", (128, 64)))
    return RiskValueNet(**values)


def save_checkpoint(path: Path, model: RiskValueNet, model_config: dict[str, Any], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_config": model_config, "metadata": metadata}, path)


def load_checkpoint(path: Path, *, device: str = "cpu") -> tuple[RiskValueNet, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def risk_score(model: RiskValueNet, observation: dict[str, Any], *, device: str = "cpu") -> float:
    with torch.no_grad():
        grid = torch.as_tensor(observation["dugm"], dtype=torch.float32, device=device).unsqueeze(0)
        state = torch.as_tensor(observation["state"], dtype=torch.float32, device=device).unsqueeze(0)
        return float(model.score(grid, state).item())
