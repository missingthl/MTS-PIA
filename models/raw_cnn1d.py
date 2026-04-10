from __future__ import annotations

import torch
from torch import nn


class RawCNN1D(nn.Module):
    """Minimal 1D CNN for fixed-length raw trial classification."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        hidden_channels: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        c1 = int(hidden_channels)
        c2 = int(hidden_channels * 2)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(c2, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.head(feat)
