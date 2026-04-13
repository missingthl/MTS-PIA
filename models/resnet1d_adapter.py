from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.resnet1d import ResNet1DClassifier


@dataclass
class ResNet1DForwardOutputs:
    sequence_features: torch.Tensor
    latent: torch.Tensor
    base_logit: torch.Tensor
    final_logit: torch.Tensor


class ResNet1DAdapter(nn.Module):
    """Standard ResNet-1D host that exposes latent/base-logit for E0/E1/E2."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        block_channels: tuple[int, int, int] = (64, 128, 128),
        kernel_sizes: tuple[int, int, int] = (7, 5, 3),
        base_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.base_model = base_model or ResNet1DClassifier(
            int(in_channels),
            int(num_classes),
            block_channels=block_channels,
            kernel_sizes=kernel_sizes,
        )
        self.feature_dim = int(self.base_model.feature_dim)

    def forward_features(self, x: torch.Tensor) -> ResNet1DForwardOutputs:
        if x.ndim != 3:
            raise ValueError(f"ResNet1DAdapter expects [B,C,T], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.in_channels:
            raise ValueError(
                f"ResNet1DAdapter expected channel dim {self.in_channels}, got {tuple(x.shape)}"
            )
        feats = self.base_model.backbone.forward_features(x.float())
        base_logit = self.base_model.classifier(feats.latent)
        return ResNet1DForwardOutputs(
            sequence_features=feats.sequence_features,
            latent=feats.latent,
            base_logit=base_logit,
            final_logit=base_logit,
        )

    def forward(self, x: torch.Tensor, *, return_features: bool = False):
        outputs = self.forward_features(x)
        if return_features:
            return outputs
        return outputs.final_logit
