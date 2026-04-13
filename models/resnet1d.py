from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePadConv1d(nn.Module):
    """Conv1d with explicit SAME padding for stride=1 kernels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            stride=1,
            padding=0,
            bias=bool(bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total_pad = max(0, int(self.kernel_size) - 1)
        pad_left = int(total_pad // 2)
        pad_right = int(total_pad - pad_left)
        if total_pad > 0:
            x = F.pad(x, (pad_left, pad_right))
        return self.conv(x)


class ResNet1DBlock(nn.Module):
    """Wang et al. style residual block with 8/5/3 receptive fields."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_sizes: tuple[int, int, int] = (7, 5, 3),
    ) -> None:
        super().__init__()
        k1, k2, k3 = (int(v) for v in kernel_sizes)

        self.conv1 = SamePadConv1d(in_channels, out_channels, kernel_size=k1, bias=False)
        self.bn1 = nn.BatchNorm1d(int(out_channels))
        self.conv2 = SamePadConv1d(out_channels, out_channels, kernel_size=k2, bias=False)
        self.bn2 = nn.BatchNorm1d(int(out_channels))
        self.conv3 = SamePadConv1d(out_channels, out_channels, kernel_size=k3, bias=False)
        self.bn3 = nn.BatchNorm1d(int(out_channels))
        self.relu = nn.ReLU(inplace=True)

        if int(in_channels) != int(out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv1d(int(in_channels), int(out_channels), kernel_size=1, bias=False),
                nn.BatchNorm1d(int(out_channels)),
            )
        else:
            self.shortcut = nn.BatchNorm1d(int(in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.relu(x + residual)
        return x


@dataclass
class ResNet1DBackboneOutputs:
    sequence_features: torch.Tensor
    latent: torch.Tensor


class ResNet1DBackbone(nn.Module):
    """Canonical TSC ResNet-1D backbone aligned to tsai's standard implementation."""

    def __init__(
        self,
        in_channels: int,
        *,
        block_channels: tuple[int, int, int] = (64, 128, 128),
        kernel_sizes: tuple[int, int, int] = (7, 5, 3),
    ) -> None:
        super().__init__()
        c1, c2, c3 = (int(v) for v in block_channels)
        self.block1 = ResNet1DBlock(int(in_channels), c1, kernel_sizes=kernel_sizes)
        self.block2 = ResNet1DBlock(c1, c2, kernel_sizes=kernel_sizes)
        self.block3 = ResNet1DBlock(c2, c3, kernel_sizes=kernel_sizes)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = int(c3)

    def forward_features(self, x: torch.Tensor) -> ResNet1DBackboneOutputs:
        if x.ndim != 3:
            raise ValueError(f"ResNet1DBackbone expects [B,C,T], got {tuple(x.shape)}")
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        latent = self.gap(x).squeeze(-1)
        return ResNet1DBackboneOutputs(sequence_features=x, latent=latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x).latent


class ResNet1DClassifier(nn.Module):
    """Backbone + GAP + linear head in the standard TSC style."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        block_channels: tuple[int, int, int] = (64, 128, 128),
        kernel_sizes: tuple[int, int, int] = (7, 5, 3),
    ) -> None:
        super().__init__()
        self.backbone = ResNet1DBackbone(
            int(in_channels),
            block_channels=block_channels,
            kernel_sizes=kernel_sizes,
        )
        self.classifier = nn.Linear(self.backbone.feature_dim, int(num_classes))

    @property
    def feature_dim(self) -> int:
        return int(self.backbone.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.backbone(x)
        return self.classifier(latent)
