from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SamePadConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool = False):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total_pad = max(0, self.kernel_size - 1)
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        if total_pad > 0:
            x = F.pad(x, (pad_left, pad_right))
        return self.conv(x)

class ResNet1DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: tuple[int, int, int] = (7, 5, 3)):
        super().__init__()
        k1, k2, k3 = kernel_sizes
        self.conv1 = SamePadConv1d(in_channels, out_channels, kernel_size=k1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = SamePadConv1d(out_channels, out_channels, kernel_size=k2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = SamePadConv1d(out_channels, out_channels, kernel_size=k3)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm1d(out_channels))
        else:
            self.shortcut = nn.BatchNorm1d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.relu(x + residual)
        return x

class ResNet1DBackbone(nn.Module):
    def __init__(self, in_channels: int, block_channels: tuple[int, int, int] = (64, 128, 128)):
        super().__init__()
        c1, c2, c3 = block_channels
        self.block1 = ResNet1DBlock(in_channels, c1)
        self.block2 = ResNet1DBlock(c1, c2)
        self.block3 = ResNet1DBlock(c2, c3)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = c3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.gap(x).squeeze(-1)

class ResNet1DClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.backbone = ResNet1DBackbone(in_channels)
        self.classifier = nn.Linear(self.backbone.feature_dim, num_classes)
        self.projection_head = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, self.backbone.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.backbone.feature_dim, 64),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def project(self, features: torch.Tensor) -> torch.Tensor:
        proj = self.projection_head(features)
        return F.normalize(proj, p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return self.classify(latent)
