from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


class Generator(nn.Module):
    def __init__(self, latent_dim: int, data_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Critic(nn.Module):
    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        *,
        use_spectral_norm: bool,
    ):
        super().__init__()

        def linear(in_features: int, out_features: int) -> nn.Module:
            layer = nn.Linear(in_features, out_features)
            return spectral_norm(layer) if use_spectral_norm else layer

        self.net = nn.Sequential(
            linear(data_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)
