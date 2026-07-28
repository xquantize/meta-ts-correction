from __future__ import annotations

import torch
from torch import nn


class ResidualCorrectorV1(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 32, dropout: float = 0.0):
        super().__init__()
        if in_dim < 1:
            raise ValueError("in_dim must be >= 1")
        if hidden < 1:
            raise ValueError("hidden must be >= 1")
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
