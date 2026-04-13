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
        support_mode: str = "same_opp_balanced",
        prototype_aggregation: str = "pooled",
        prototype_init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.prototypes_per_class = int(prototypes_per_class)
        self.routing_temperature = max(float(routing_temperature), 1e-6)
        self.ridge = float(ridge)
        self.support_mode = str(support_mode)
        self.prototype_aggregation = str(prototype_aggregation)
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

    def _resolve_support_masses(self) -> tuple[float, float]:
        if self.support_mode == "same_opp_balanced":
            return 0.5, 0.5
        if self.support_mode == "same_only":
            return 1.0, 0.0
        if self.support_mode == "same_opp_asym":
            return 0.75, 0.25
        raise ValueError(f"unsupported support mode: {self.support_mode}")

    def _solve_direction(
        self,
        h: torch.Tensor,
        *,
        same_proto: torch.Tensor,
        opp_proto: torch.Tensor,
        eye: torch.Tensor,
    ) -> torch.Tensor:
        same_mass, opp_mass = self._resolve_support_masses()
        batch_size = int(h.shape[0])

        if same_proto.shape[0] == 1:
            same_w = torch.full(
                (batch_size, 1),
                float(same_mass),
                dtype=h.dtype,
                device=h.device,
            )
        else:
            same_w = torch.softmax(self._squared_similarity(h, same_proto), dim=-1) * float(same_mass)

        if opp_proto.shape[0] > 0 and opp_mass > 0.0:
            opp_w = torch.softmax(self._squared_similarity(h, opp_proto), dim=-1) * float(opp_mass)
            support = torch.cat([same_proto, opp_proto], dim=0)
            weights = torch.cat([same_w, opp_w], dim=1)
            targets = torch.cat(
                [
                    torch.ones(same_proto.shape[0], dtype=h.dtype, device=h.device),
                    -torch.ones(opp_proto.shape[0], dtype=h.dtype, device=h.device),
                ],
                dim=0,
            )
        else:
            support = same_proto
            weights = same_w
            targets = torch.ones(same_proto.shape[0], dtype=h.dtype, device=h.device)

        gram = torch.einsum("bn,nd,ne->bde", weights, support, support)
        rhs = torch.einsum("bn,nd,n->bd", weights, support, targets)
        return torch.linalg.solve(
            gram + self.ridge * eye.unsqueeze(0),
            rhs.unsqueeze(-1),
        ).squeeze(-1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch_size = int(h.shape[0])
        eye = torch.eye(self.feature_dim, dtype=h.dtype, device=h.device)
        class_logits = []

        for class_idx in range(self.num_classes):
            same_proto = self.prototypes[class_idx]
            opp_proto = self.prototypes[
                torch.arange(self.num_classes, device=h.device) != class_idx
            ].reshape(-1, self.feature_dim)
            if self.prototype_aggregation == "pooled":
                solved = self._solve_direction(
                    h,
                    same_proto=same_proto,
                    opp_proto=opp_proto,
                    eye=eye,
                )
                class_logit = torch.sum(h * solved, dim=-1)
            elif self.prototype_aggregation == "committee_mean":
                member_logits = []
                for proto_idx in range(int(same_proto.shape[0])):
                    solved = self._solve_direction(
                        h,
                        same_proto=same_proto[proto_idx : proto_idx + 1],
                        opp_proto=opp_proto,
                        eye=eye,
                    )
                    member_logits.append(torch.sum(h * solved, dim=-1))
                class_logit = torch.stack(member_logits, dim=0).mean(dim=0)
            else:
                raise ValueError(f"unsupported prototype aggregation: {self.prototype_aggregation}")
            class_logits.append(class_logit)

        return torch.stack(class_logits, dim=-1).reshape(batch_size, self.num_classes)


@dataclass
class TensorCSPNetLocalClosedFormOutputs:
    x_csp: torch.Tensor
    x_log: torch.Tensor
    latent: torch.Tensor
    base_logit: torch.Tensor
    local_closed_form_logit: torch.Tensor
    readout_gate: torch.Tensor | None
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
        detach_local_input: bool = False,
        support_mode: str = "same_opp_balanced",
        prototype_aggregation: str = "pooled",
        readout_gate_mode: str = "none",
        spd_dtype: torch.dtype = torch.double,
    ) -> None:
        super().__init__()
        self.adapter = TensorCSPNetAdapter(
            channel_num=int(channel_num),
            mlp=bool(mlp),
            dataset=str(dataset),
            spd_dtype=spd_dtype,
        )
        self.local_head = LocalClosedFormResidualHead(
            feature_dim=self.adapter.feature_dim,
            num_classes=self.adapter.num_classes,
            prototypes_per_class=int(prototypes_per_class),
            routing_temperature=float(routing_temperature),
            ridge=float(ridge),
            support_mode=str(support_mode),
            prototype_aggregation=str(prototype_aggregation),
        )
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))
        self.detach_local_input = bool(detach_local_input)
        self.readout_gate_mode = str(readout_gate_mode)
        if self.readout_gate_mode == "consistency":
            self.readout_gate_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.readout_gate_bias = nn.Parameter(torch.zeros(self.adapter.num_classes, dtype=torch.float32))
        elif self.readout_gate_mode != "none":
            raise ValueError(f"unsupported readout gate mode: {self.readout_gate_mode}")

    def forward(
        self,
        x: torch.Tensor,
        *,
        fusion_alpha: float = 1.0,
        return_features: bool = False,
    ):
        base: TensorCSPNetForwardOutputs = self.adapter(x, return_features=True)
        local_input = base.latent.detach() if self.detach_local_input else base.latent
        local_closed_form_logit = self.local_head(local_input)
        alpha = torch.tensor(float(fusion_alpha), dtype=base.base_logit.dtype, device=base.base_logit.device)
        readout_gate = None
        if self.readout_gate_mode == "consistency":
            gate_bias = self.readout_gate_bias.to(dtype=base.base_logit.dtype, device=base.base_logit.device)
            gate_scale = self.readout_gate_scale.to(dtype=base.base_logit.dtype, device=base.base_logit.device)
            readout_gate = torch.sigmoid(gate_scale * base.base_logit * local_closed_form_logit + gate_bias)
        gated_local_logit = local_closed_form_logit if readout_gate is None else readout_gate * local_closed_form_logit
        final_logit = base.base_logit + alpha * self.beta * gated_local_logit
        if return_features:
            return TensorCSPNetLocalClosedFormOutputs(
                x_csp=base.x_csp,
                x_log=base.x_log,
                latent=base.latent,
                base_logit=base.base_logit,
                local_closed_form_logit=local_closed_form_logit,
                readout_gate=readout_gate,
                final_logit=final_logit,
                beta=self.beta,
            )
        return final_logit
