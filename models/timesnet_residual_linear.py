from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.timesnet_adapter import TimesNetAdapter, TimesNetForwardOutputs


@dataclass
class TimesNetResidualLinearOutputs:
    sequence_features: torch.Tensor
    latent: torch.Tensor
    base_logit: torch.Tensor
    residual_logit: torch.Tensor
    final_logit: torch.Tensor
    beta: torch.Tensor


class TimesNetResidualLinear(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        seq_len: int,
        num_classes: int,
        init_beta: float = 0.1,
        d_model: int = 32,
        d_ff: int = 64,
        e_layers: int = 2,
        top_k: int = 3,
        num_kernels: int = 4,
        dropout: float = 0.1,
        embed: str = "fixed",
        freq: str = "h",
    ) -> None:
        super().__init__()
        self.adapter = TimesNetAdapter(
            in_channels=int(in_channels),
            seq_len=int(seq_len),
            num_classes=int(num_classes),
            d_model=int(d_model),
            d_ff=int(d_ff),
            e_layers=int(e_layers),
            top_k=int(top_k),
            num_kernels=int(num_kernels),
            dropout=float(dropout),
            embed=str(embed),
            freq=str(freq),
        )
        self.residual_head = nn.Linear(self.adapter.feature_dim, self.adapter.num_classes)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

    def forward(self, x: torch.Tensor, *, return_features: bool = False):
        base: TimesNetForwardOutputs = self.adapter(x, return_features=True)
        residual_logit = self.residual_head(base.latent)
        final_logit = base.base_logit + self.beta * residual_logit
        if return_features:
            return TimesNetResidualLinearOutputs(
                sequence_features=base.sequence_features,
                latent=base.latent,
                base_logit=base.base_logit,
                residual_logit=residual_logit,
                final_logit=final_logit,
                beta=self.beta,
            )
        return final_logit
