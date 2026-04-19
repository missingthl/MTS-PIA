from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.local_closed_form_residual_head import LocalClosedFormResidualHead
from models.patchtst_adapter import PatchTSTAdapter, PatchTSTForwardOutputs


@dataclass
class PatchTSTLocalClosedFormOutputs:
    sequence_features: torch.Tensor
    latent: torch.Tensor
    base_logit: torch.Tensor
    local_closed_form_logit: torch.Tensor
    readout_gate: torch.Tensor | None
    final_logit: torch.Tensor
    beta: torch.Tensor


class PatchTSTLocalClosedFormResidual(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        seq_len: int,
        num_classes: int,
        prototypes_per_class: int = 4,
        routing_temperature: float = 1.0,
        class_prior_temperature: float | None = None,
        subproto_temperature: float | None = None,
        ridge: float = 1e-2,
        ridge_mode: str = "fixed",
        ridge_trace_eps: float = 1e-8,
        solve_mode: str = "ridge_solve",
        pinv_rcond: float = 1e-4,
        input_norm_mode: str = "none",
        input_norm_eps: float = 1e-8,
        enable_probe: bool = False,
        init_beta: float = 0.1,
        detach_local_input: bool = False,
        support_mode: str = "same_only",
        prototype_aggregation: str = "pooled",
        prototype_geometry_mode: str = "flat",
        tangent_rank: int = 2,
        tangent_source: str = "subproto_offsets",
        readout_gate_mode: str = "none",
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
        self.local_head = LocalClosedFormResidualHead(
            feature_dim=self.adapter.feature_dim,
            num_classes=self.adapter.num_classes,
            prototypes_per_class=int(prototypes_per_class),
            routing_temperature=float(routing_temperature),
            class_prior_temperature=class_prior_temperature,
            subproto_temperature=subproto_temperature,
            ridge=float(ridge),
            ridge_mode=str(ridge_mode),
            ridge_trace_eps=float(ridge_trace_eps),
            solve_mode=str(solve_mode),
            pinv_rcond=float(pinv_rcond),
            input_norm_mode=str(input_norm_mode),
            input_norm_eps=float(input_norm_eps),
            enable_probe=bool(enable_probe),
            support_mode=str(support_mode),
            prototype_aggregation=str(prototype_aggregation),
            prototype_geometry_mode=str(prototype_geometry_mode),
            tangent_rank=int(tangent_rank),
            tangent_source=str(tangent_source),
        )
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))
        self.detach_local_input = bool(detach_local_input)
        self.readout_gate_mode = str(readout_gate_mode)
        if self.readout_gate_mode == "consistency":
            self.readout_gate_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.readout_gate_bias = nn.Parameter(torch.zeros(self.adapter.num_classes, dtype=torch.float32))
        elif self.readout_gate_mode != "none":
            raise ValueError(f"unsupported readout gate mode: {self.readout_gate_mode}")

    def forward(self, x: torch.Tensor, *, fusion_alpha: float = 1.0, return_features: bool = False):
        base: PatchTSTForwardOutputs = self.adapter(x, return_features=True)
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
            return PatchTSTLocalClosedFormOutputs(
                sequence_features=base.sequence_features,
                latent=base.latent,
                base_logit=base.base_logit,
                local_closed_form_logit=local_closed_form_logit,
                readout_gate=readout_gate,
                final_logit=final_logit,
                beta=self.beta,
            )
        return final_logit
