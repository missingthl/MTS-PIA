from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.tensor_cspnet_adapter import TensorCSPNetAdapter, TensorCSPNetForwardOutputs


class LocalClosedFormResidualHead(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        num_classes: int,
        prototypes_per_class: int,
        routing_temperature: float,
        ridge: float,
        prototype_init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.prototypes_per_class = int(prototypes_per_class)
        self.routing_temperature = max(float(routing_temperature), 1e-6)
        self.ridge = float(ridge)
        self.prototypes = nn.Parameter(
            torch.randn(
                self.num_classes,
                self.prototypes_per_class,
                self.feature_dim,
                dtype=torch.float32,
            )
            * float(prototype_init_scale)
        )

    def _squared_similarity(self, h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        diff = h[:, None, :] - p[None, :, :]
        return -torch.sum(diff * diff, dim=-1) / self.routing_temperature

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch_size = int(h.shape[0])
        eye = torch.eye(self.feature_dim, dtype=h.dtype, device=h.device)
        class_logits = []

        for class_idx in range(self.num_classes):
            same_proto = self.prototypes[class_idx]
            opp_proto = self.prototypes[
                torch.arange(self.num_classes, device=h.device) != class_idx
            ].reshape(-1, self.feature_dim)

            same_w = torch.softmax(self._squared_similarity(h, same_proto), dim=-1) * 0.5
            opp_w = torch.softmax(self._squared_similarity(h, opp_proto), dim=-1) * 0.5

            support = torch.cat([same_proto, opp_proto], dim=0)
            weights = torch.cat([same_w, opp_w], dim=1)
            targets = torch.cat(
                [
                    torch.ones(self.prototypes_per_class, dtype=h.dtype, device=h.device),
                    -torch.ones(opp_proto.shape[0], dtype=h.dtype, device=h.device),
                ],
                dim=0,
            )

            gram = torch.einsum("bn,nd,ne->bde", weights, support, support)
            rhs = torch.einsum("bn,nd,n->bd", weights, support, targets)
            solved = torch.linalg.solve(
                gram + self.ridge * eye.unsqueeze(0),
                rhs.unsqueeze(-1),
            ).squeeze(-1)
            class_logits.append(torch.sum(h * solved, dim=-1))

        return torch.stack(class_logits, dim=-1).reshape(batch_size, self.num_classes)


@dataclass
class TensorCSPNetLocalClosedFormOutputs:
    x_csp: torch.Tensor
    x_log: torch.Tensor
    latent: torch.Tensor
    base_logit: torch.Tensor
    local_closed_form_logit: torch.Tensor
    final_logit: torch.Tensor
    beta: torch.Tensor


class TensorCSPNetLocalClosedFormResidual(nn.Module):
    def __init__(
        self,
        *,
        channel_num: int,
        mlp: bool = False,
        dataset: str = "BCIC",
        prototypes_per_class: int = 4,
        routing_temperature: float = 1.0,
        ridge: float = 1e-2,
        init_beta: float = 0.1,
    ) -> None:
        super().__init__()
        self.adapter = TensorCSPNetAdapter(
            channel_num=int(channel_num),
            mlp=bool(mlp),
            dataset=str(dataset),
        )
        self.local_head = LocalClosedFormResidualHead(
            feature_dim=self.adapter.feature_dim,
            num_classes=self.adapter.num_classes,
            prototypes_per_class=int(prototypes_per_class),
            routing_temperature=float(routing_temperature),
            ridge=float(ridge),
        )
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

    def forward(self, x: torch.Tensor, *, return_features: bool = False):
        base: TensorCSPNetForwardOutputs = self.adapter(x, return_features=True)
        local_closed_form_logit = self.local_head(base.latent)
        final_logit = base.base_logit + self.beta * local_closed_form_logit
        if return_features:
            return TensorCSPNetLocalClosedFormOutputs(
                x_csp=base.x_csp,
                x_log=base.x_log,
                latent=base.latent,
                base_logit=base.base_logit,
                local_closed_form_logit=local_closed_form_logit,
                final_logit=final_logit,
                beta=self.beta,
            )
        return final_logit
