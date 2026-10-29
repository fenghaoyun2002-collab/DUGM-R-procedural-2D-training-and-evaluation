"""The dual branch feature extractor used by both PPO policies."""

from __future__ import annotations

import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DugmFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        dugm_features_dim: int = 256,
        state_hidden_dim: int = 32,
        state_features_dim: int = 64,
        fused_features_dim: int = 256,
    ) -> None:
        if set(observation_space.spaces) != {"dugm", "state"}:
            raise ValueError("DUGM policies require {'dugm', 'state'} observations.")
        grid = observation_space.spaces["dugm"]
        state = observation_space.spaces["state"]
        if grid.shape != (2, 64, 64) or state.shape != (4,):
            raise ValueError("Expected DUGM [2,64,64] and state [4].")
        super().__init__(observation_space, features_dim=fused_features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.Flatten(),
        )
        with torch.no_grad():
            flattened = int(self.cnn(torch.zeros(1, *grid.shape)).shape[1])
        self.dugm_projection = nn.Sequential(
            nn.Linear(flattened, dugm_features_dim), nn.LayerNorm(dugm_features_dim), nn.SiLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(4, state_hidden_dim), nn.SiLU(),
            nn.Linear(state_hidden_dim, state_features_dim), nn.LayerNorm(state_features_dim), nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(dugm_features_dim + state_features_dim, fused_features_dim),
            nn.LayerNorm(fused_features_dim), nn.SiLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        grid = self.dugm_projection(self.cnn(observations["dugm"].float()))
        state = self.state_encoder(observations["state"].float())
        return self.fusion(torch.cat((grid, state), dim=1))
