from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalClosedFormResidualHead(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        num_classes: int,
        prototypes_per_class: int,
        routing_temperature: float,
        ridge: float,
        ridge_mode: str = "fixed",
        ridge_trace_eps: float = 1e-8,
        solve_mode: str = "ridge_solve",
        pinv_rcond: float = 1e-4,
        input_norm_mode: str = "none",
        input_norm_eps: float = 1e-8,
        enable_probe: bool = False,
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
        self.ridge_mode = str(ridge_mode)
        self.ridge_trace_eps = max(float(ridge_trace_eps), 0.0)
        self.solve_mode = str(solve_mode)
        self.pinv_rcond = max(float(pinv_rcond), 0.0)
        self.input_norm_mode = str(input_norm_mode)
        self.input_norm_eps = max(float(input_norm_eps), 1e-12)
        self.enable_probe = bool(enable_probe)
        self.support_mode = str(support_mode)
        self.prototype_aggregation = str(prototype_aggregation)
        if self.ridge_mode not in {"fixed", "trace_adaptive"}:
            raise ValueError(f"unsupported ridge mode: {self.ridge_mode}")
        if self.solve_mode not in {"ridge_solve", "pinv", "dual_ridge", "dual_pinv"}:
            raise ValueError(f"unsupported solve mode: {self.solve_mode}")
        if self.input_norm_mode not in {"none", "l2_hypersphere"}:
            raise ValueError(f"unsupported input norm mode: {self.input_norm_mode}")
        self.prototypes = nn.Parameter(
            torch.randn(
                self.num_classes,
                self.prototypes_per_class,
                self.feature_dim,
                dtype=torch.float32,
            )
            * float(prototype_init_scale)
        )
        self._probe_rows: List[Dict[str, float | int | str]] = []
        self._probe_split = "unset"
        self._probe_epoch = -1

    def reset_probe(self) -> None:
        self._probe_rows = []
        self._probe_split = "unset"
        self._probe_epoch = -1

    def set_probe_context(self, *, split: str, epoch: int) -> None:
        self._probe_split = str(split)
        self._probe_epoch = int(epoch)

    def export_probe_rows(self) -> List[Dict[str, float | int | str]]:
        return list(self._probe_rows)

    def _record_probe_rows(
        self,
        *,
        gram: torch.Tensor,
        ridge_scale: torch.Tensor,
        solved: torch.Tensor,
        class_idx: int,
        committee_idx: int,
    ) -> None:
        if not self.enable_probe:
            return
        with torch.no_grad():
            dim = float(gram.shape[-1])
            mean_trace = torch.einsum("bii->b", gram) / dim
            eigvals = torch.linalg.eigvalsh(gram)
            eig_abs = eigvals.abs()
            cond_eps = max(self.ridge_trace_eps, 1e-12)
            cond_number = eig_abs.max(dim=-1).values / eig_abs.min(dim=-1).values.clamp_min(cond_eps)
            weight_norm = torch.linalg.vector_norm(solved, dim=-1)

            mean_trace_cpu = mean_trace.detach().cpu().tolist()
            cond_cpu = cond_number.detach().cpu().tolist()
            ridge_cpu = ridge_scale.detach().cpu().tolist()
            wnorm_cpu = weight_norm.detach().cpu().tolist()

            for sample_idx, (trace_i, cond_i, ridge_i, wnorm_i) in enumerate(
                zip(mean_trace_cpu, cond_cpu, ridge_cpu, wnorm_cpu)
            ):
                self._probe_rows.append(
                    {
                        "split": self._probe_split,
                        "epoch": int(self._probe_epoch),
                        "class_idx": int(class_idx),
                        "committee_idx": int(committee_idx),
                        "sample_idx_in_batch": int(sample_idx),
                        "ridge_mode": self.ridge_mode,
                        "solve_mode": self.solve_mode,
                        "support_mode": self.support_mode,
                        "prototype_aggregation": self.prototype_aggregation,
                        "input_norm_mode": self.input_norm_mode,
                        "mean_trace": float(trace_i),
                        "condition_number": float(cond_i),
                        "effective_ridge": float(ridge_i),
                        "weight_norm": float(wnorm_i),
                    }
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

    def _maybe_normalize_inputs(
        self,
        h: torch.Tensor,
        same_proto: torch.Tensor,
        opp_proto: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.input_norm_mode != "l2_hypersphere":
            return h, same_proto, opp_proto
        h = F.normalize(h, p=2, dim=-1, eps=self.input_norm_eps)
        same_proto = F.normalize(same_proto, p=2, dim=-1, eps=self.input_norm_eps)
        if opp_proto.numel() > 0:
            opp_proto = F.normalize(opp_proto, p=2, dim=-1, eps=self.input_norm_eps)
        return h, same_proto, opp_proto

    def _solve_direction(
        self,
        h: torch.Tensor,
        *,
        same_proto: torch.Tensor,
        opp_proto: torch.Tensor,
        eye: torch.Tensor,
        class_idx: int,
        committee_idx: int = -1,
    ) -> torch.Tensor:
        same_mass, opp_mass = self._resolve_support_masses()
        batch_size = int(h.shape[0])
        h, same_proto, opp_proto = self._maybe_normalize_inputs(h, same_proto, opp_proto)

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
        weight_sqrt = torch.sqrt(weights.clamp_min(0.0))
        weighted_support = weight_sqrt[:, :, None] * support.unsqueeze(0)
        weighted_targets = weight_sqrt * targets.unsqueeze(0)

        if self.solve_mode == "pinv":
            ridge_scale = torch.zeros(
                (gram.shape[0],),
                dtype=h.dtype,
                device=h.device,
            )
            solved = torch.matmul(
                torch.linalg.pinv(gram, rcond=self.pinv_rcond, hermitian=True),
                rhs.unsqueeze(-1),
            ).squeeze(-1)
        elif self.solve_mode == "dual_pinv":
            ridge_scale = torch.zeros(
                (gram.shape[0],),
                dtype=h.dtype,
                device=h.device,
            )
            dual_gram = torch.einsum("bnd,bmd->bnm", weighted_support, weighted_support)
            dual_solution = torch.matmul(
                torch.linalg.pinv(dual_gram, rcond=self.pinv_rcond, hermitian=True),
                weighted_targets.unsqueeze(-1),
            ).squeeze(-1)
            solved = torch.einsum("bnd,bn->bd", weighted_support, dual_solution)
        else:
            if self.ridge_mode == "trace_adaptive":
                dim = float(gram.shape[-1])
                mean_trace = torch.einsum("bii->b", gram) / dim
                ridge_scale = self.ridge * mean_trace.clamp_min(self.ridge_trace_eps)
            else:
                ridge_scale = torch.full(
                    (gram.shape[0],),
                    float(self.ridge),
                    dtype=h.dtype,
                    device=h.device,
                )
            if self.solve_mode == "dual_ridge":
                dual_gram = torch.einsum("bnd,bmd->bnm", weighted_support, weighted_support)
                dual_eye = torch.eye(
                    dual_gram.shape[-1],
                    dtype=h.dtype,
                    device=h.device,
                ).unsqueeze(0)
                dual_solution = torch.linalg.solve(
                    dual_gram + ridge_scale[:, None, None] * dual_eye,
                    weighted_targets.unsqueeze(-1),
                ).squeeze(-1)
                solved = torch.einsum("bnd,bn->bd", weighted_support, dual_solution)
            else:
                solved = torch.linalg.solve(
                    gram + ridge_scale[:, None, None] * eye.unsqueeze(0),
                    rhs.unsqueeze(-1),
                ).squeeze(-1)
        self._record_probe_rows(
            gram=gram,
            ridge_scale=ridge_scale,
            solved=solved,
            class_idx=int(class_idx),
            committee_idx=int(committee_idx),
        )
        return solved

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
                    class_idx=class_idx,
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
                        class_idx=class_idx,
                        committee_idx=proto_idx,
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
        ridge_mode: str = "fixed",
        ridge_trace_eps: float = 1e-8,
        solve_mode: str = "ridge_solve",
        pinv_rcond: float = 1e-4,
        input_norm_mode: str = "none",
        input_norm_eps: float = 1e-8,
        enable_probe: bool = False,
        init_beta: float = 0.1,
        detach_local_input: bool = False,
        support_mode: str = "same_opp_balanced",
        prototype_aggregation: str = "pooled",
        readout_gate_mode: str = "none",
        spd_dtype: torch.dtype = torch.double,
    ) -> None:
        super().__init__()
        from models.tensor_cspnet_adapter import TensorCSPNetAdapter

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
            ridge_mode=str(ridge_mode),
            ridge_trace_eps=float(ridge_trace_eps),
            solve_mode=str(solve_mode),
            pinv_rcond=float(pinv_rcond),
            input_norm_mode=str(input_norm_mode),
            input_norm_eps=float(input_norm_eps),
            enable_probe=bool(enable_probe),
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
        base = self.adapter(x, return_features=True)
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
