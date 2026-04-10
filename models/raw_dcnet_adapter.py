from __future__ import annotations

import torch
from torch import nn

from runners.spatial_dcnet_torch import DCNetTorch


class RawDCNetAdapter(nn.Module):
    """Minimal adapter that feeds raw [B, C, T] sequences into DCNet."""

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        num_classes: int,
        *,
        allow_variable_length: bool = False,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        init: str = "default",
        classifier_type: str = "conv",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.seq_len = int(seq_len)
        self.allow_variable_length = bool(allow_variable_length)
        self.input_dim = int(in_channels) * int(seq_len)
        self.dcnet = DCNetTorch(
            input_dim=self.input_dim,
            num_classes=int(num_classes),
            bn_eps=float(bn_eps),
            bn_momentum=float(bn_momentum),
            init=str(init),
            classifier_type=str(classifier_type),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if x.ndim != 3:
            raise ValueError(f"RawDCNetAdapter expects [B,C,T], got {tuple(x.shape)}")
        b, c, t = x.shape
        if c != self.in_channels:
            raise ValueError(
                f"RawDCNetAdapter expected [B,{self.in_channels},T], "
                f"got {tuple(x.shape)}"
            )
        if (not self.allow_variable_length) and t != self.seq_len:
            raise ValueError(
                f"RawDCNetAdapter expected [B,{self.in_channels},{self.seq_len}], "
                f"got {tuple(x.shape)}"
            )
        if self.allow_variable_length and t != self.seq_len:
            if t > self.seq_len:
                x = x[..., : self.seq_len]
            else:
                pad = self.seq_len - int(t)
                x = nn.functional.pad(x, (0, pad))
        x_flat = x.reshape(b, self.input_dim)
        return self.dcnet(x_flat, return_features=return_features)


class RawDCNetTemporalAdapter(nn.Module):
    """Minimal temporal front-end before DCNet to avoid pure raw flattening."""

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        num_classes: int,
        *,
        proj_channels: int = 16,
        proj_bins: int = 8,
        allow_variable_length: bool = False,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        init: str = "default",
        classifier_type: str = "conv",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.seq_len = int(seq_len)
        self.allow_variable_length = bool(allow_variable_length)
        self.proj_channels = int(proj_channels)
        self.proj_bins = int(proj_bins)
        self.temporal = nn.Sequential(
            nn.Conv1d(self.in_channels, self.proj_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(self.proj_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(self.proj_bins),
        )
        self.input_dim = self.proj_channels * self.proj_bins
        self.dcnet = DCNetTorch(
            input_dim=self.input_dim,
            num_classes=int(num_classes),
            bn_eps=float(bn_eps),
            bn_momentum=float(bn_momentum),
            init=str(init),
            classifier_type=str(classifier_type),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if x.ndim != 3:
            raise ValueError(f"RawDCNetTemporalAdapter expects [B,C,T], got {tuple(x.shape)}")
        b, c, t = x.shape
        if c != self.in_channels:
            raise ValueError(
                f"RawDCNetTemporalAdapter expected [B,{self.in_channels},T], "
                f"got {tuple(x.shape)}"
            )
        if (not self.allow_variable_length) and t != self.seq_len:
            raise ValueError(
                f"RawDCNetTemporalAdapter expected [B,{self.in_channels},{self.seq_len}], "
                f"got {tuple(x.shape)}"
            )
        feat = self.temporal(x)
        x_flat = feat.reshape(b, self.input_dim)
        return self.dcnet(x_flat, return_features=return_features)
