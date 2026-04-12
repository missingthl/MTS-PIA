from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.tensor_cspnet_adapter import TensorCSPNetAdapter, TensorCSPNetForwardOutputs


@dataclass
class TensorCSPNetResidualLinearOutputs:
    x_csp: torch.Tensor
    x_log: torch.Tensor
    latent: torch.Tensor
    base_logit: torch.Tensor
    residual_logit: torch.Tensor
    final_logit: torch.Tensor
    beta: torch.Tensor


class TensorCSPNetResidualLinear(nn.Module):
    def __init__(
        self,
        *,
        channel_num: int,
        mlp: bool = False,
        dataset: str = "BCIC",
        init_beta: float = 0.1,
    ) -> None:
        super().__init__()
        self.adapter = TensorCSPNetAdapter(
            channel_num=int(channel_num),
            mlp=bool(mlp),
            dataset=str(dataset),
        )
        self.residual_head = nn.Linear(
            self.adapter.feature_dim,
            self.adapter.num_classes,
        ).float()
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

    def forward(self, x: torch.Tensor, *, return_features: bool = False):
        base: TensorCSPNetForwardOutputs = self.adapter(x, return_features=True)
        residual_logit = self.residual_head(base.latent)
        final_logit = base.base_logit + self.beta * residual_logit
        if return_features:
            return TensorCSPNetResidualLinearOutputs(
                x_csp=base.x_csp,
                x_log=base.x_log,
                latent=base.latent,
                base_logit=base.base_logit,
                residual_logit=residual_logit,
                final_logit=final_logit,
                beta=self.beta,
            )
        return final_logit
