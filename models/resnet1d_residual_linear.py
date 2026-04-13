from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.resnet1d_adapter import ResNet1DAdapter, ResNet1DForwardOutputs


@dataclass
class ResNet1DResidualLinearOutputs:
    sequence_features: torch.Tensor
    latent: torch.Tensor
    base_logit: torch.Tensor
    residual_logit: torch.Tensor
    final_logit: torch.Tensor
    beta: torch.Tensor


class ResNet1DResidualLinear(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        init_beta: float = 0.1,
        block_channels: tuple[int, int, int] = (64, 128, 128),
        kernel_sizes: tuple[int, int, int] = (7, 5, 3),
    ) -> None:
        super().__init__()
        self.adapter = ResNet1DAdapter(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            block_channels=block_channels,
            kernel_sizes=kernel_sizes,
        )
        self.residual_head = nn.Linear(self.adapter.feature_dim, self.adapter.num_classes)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

    def forward(self, x: torch.Tensor, *, return_features: bool = False):
        base: ResNet1DForwardOutputs = self.adapter(x, return_features=True)
        residual_logit = self.residual_head(base.latent)
        final_logit = base.base_logit + self.beta * residual_logit
        if return_features:
            return ResNet1DResidualLinearOutputs(
                sequence_features=base.sequence_features,
                latent=base.latent,
                base_logit=base.base_logit,
                residual_logit=residual_logit,
                final_logit=final_logit,
                beta=self.beta,
            )
        return final_logit
