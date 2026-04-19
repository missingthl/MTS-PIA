from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.patchtst_adapter import PatchTSTAdapter, PatchTSTForwardOutputs


@dataclass
class PatchTSTResidualLinearOutputs:
    sequence_features: torch.Tensor
    latent: torch.Tensor
    base_logit: torch.Tensor
    residual_logit: torch.Tensor
    final_logit: torch.Tensor
    beta: torch.Tensor


class PatchTSTResidualLinear(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        seq_len: int,
        num_classes: int,
        init_beta: float = 0.1,
        d_model: int = 128,
        d_ff: int = 256,
        e_layers: int = 3,
        n_heads: int = 8,
        factor: int = 1,
        dropout: float = 0.1,
        activation: str = "gelu",
        patch_len: int = 16,
        patch_stride: int = 8,
    ) -> None:
        super().__init__()
        self.adapter = PatchTSTAdapter(
            in_channels=int(in_channels),
            seq_len=int(seq_len),
            num_classes=int(num_classes),
            d_model=int(d_model),
            d_ff=int(d_ff),
            e_layers=int(e_layers),
            n_heads=int(n_heads),
            factor=int(factor),
            dropout=float(dropout),
            activation=str(activation),
            patch_len=int(patch_len),
            patch_stride=int(patch_stride),
        )
        self.residual_head = nn.Linear(self.adapter.feature_dim, self.adapter.num_classes)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

    def forward(self, x: torch.Tensor, *, return_features: bool = False):
        base: PatchTSTForwardOutputs = self.adapter(x, return_features=True)
        residual_logit = self.residual_head(base.latent)
        final_logit = base.base_logit + self.beta * residual_logit
        if return_features:
            return PatchTSTResidualLinearOutputs(
                sequence_features=base.sequence_features,
                latent=base.latent,
                base_logit=base.base_logit,
                residual_logit=residual_logit,
                final_logit=final_logit,
                beta=self.beta,
            )
        return final_logit
