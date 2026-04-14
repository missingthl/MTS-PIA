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
        class_prior_temperature: float | None = None,
        subproto_temperature: float | None = None,
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
        prototype_geometry_mode: str = "flat",
        prototype_init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.prototypes_per_class = int(prototypes_per_class)
        self.routing_temperature = max(float(routing_temperature), 1e-6)
        self.class_prior_temperature = max(
            float(self.routing_temperature if class_prior_temperature is None else class_prior_temperature),
            1e-6,
        )
        self.subproto_temperature = max(
            float(self.routing_temperature if subproto_temperature is None else subproto_temperature),
            1e-6,
        )
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
        self.prototype_geometry_mode = str(prototype_geometry_mode)
        if self.ridge_mode not in {"fixed", "trace_adaptive"}:
            raise ValueError(f"unsupported ridge mode: {self.ridge_mode}")
        if self.solve_mode not in {"ridge_solve", "pinv", "dual_ridge", "dual_pinv"}:
            raise ValueError(f"unsupported solve mode: {self.solve_mode}")
        if self.input_norm_mode not in {"none", "l2_hypersphere"}:
            raise ValueError(f"unsupported input norm mode: {self.input_norm_mode}")
        if self.prototype_geometry_mode not in {"flat", "center_subproto", "center_only"}:
            raise ValueError(f"unsupported prototype geometry mode: {self.prototype_geometry_mode}")
        if self.prototype_geometry_mode in {"center_subproto", "center_only"} and self.prototype_aggregation != "pooled":
            raise ValueError(f"{self.prototype_geometry_mode} geometry currently supports pooled aggregation only")
        if self.prototype_geometry_mode == "flat":
            self.register_parameter(
                "prototypes",
                nn.Parameter(
                    torch.randn(
                        self.num_classes,
                        self.prototypes_per_class,
                        self.feature_dim,
                        dtype=torch.float32,
                    )
                    * float(prototype_init_scale)
                ),
            )
            self.register_parameter("center_params", None)
            self.register_parameter("sub_offsets", None)
        elif self.prototype_geometry_mode == "center_only":
            self.register_parameter("prototypes", None)
            self.register_parameter(
                "center_params",
                nn.Parameter(
                    torch.randn(
                        self.num_classes,
                        self.feature_dim,
                        dtype=torch.float32,
                    )
                    * float(prototype_init_scale)
                ),
            )
            self.register_parameter("sub_offsets", None)
        else:
            self.register_parameter("prototypes", None)
            self.register_parameter(
                "center_params",
                nn.Parameter(
                    torch.randn(
                        self.num_classes,
                        self.feature_dim,
                        dtype=torch.float32,
                    )
                    * float(prototype_init_scale)
                ),
            )
            self.register_parameter(
                "sub_offsets",
                nn.Parameter(
                    torch.empty(
                        self.num_classes,
                        self.prototypes_per_class,
                        self.feature_dim,
                        dtype=torch.float32,
                    )
                ),
            )
            self._initialize_center_subproto_offsets()
        self._probe_rows: List[Dict[str, float | int | str]] = []
        self._probe_split = "unset"
        self._probe_epoch = -1
        self._last_dataflow_summary: Dict[str, object] | None = None

    def _initialize_center_subproto_offsets(self) -> None:
        if self.prototype_geometry_mode != "center_subproto":
            return
        assert self.center_params is not None
        assert self.sub_offsets is not None
        if self.feature_dim < self.prototypes_per_class:
            raise ValueError(
                "center_subproto initialization requires feature_dim >= prototypes_per_class, "
                f"got feature_dim={self.feature_dim}, prototypes_per_class={self.prototypes_per_class}"
            )

        with torch.no_grad():
            centers = F.normalize(self.center_params.data, p=2, dim=-1, eps=self.input_norm_eps)
            # Use a moderate tangent-space spread: directions start distinguishable,
            # but still remain meaningfully anchored around the class center.
            offset_scale = 0.75
            for class_idx in range(self.num_classes):
                center = centers[class_idx]
                basis = torch.randn(
                    self.feature_dim,
                    self.prototypes_per_class,
                    dtype=self.sub_offsets.dtype,
                    device=self.sub_offsets.device,
                )
                # Remove the center direction so offsets start in the tangent space.
                basis = basis - center.unsqueeze(1) * torch.matmul(center.unsqueeze(0), basis)
                q, _ = torch.linalg.qr(basis, mode="reduced")
                offsets = q.transpose(0, 1) * float(offset_scale)
                self.sub_offsets.data[class_idx].copy_(offsets)

    def reset_probe(self) -> None:
        self._probe_rows = []
        self._probe_split = "unset"
        self._probe_epoch = -1

    def set_probe_context(self, *, split: str, epoch: int) -> None:
        self._probe_split = str(split)
        self._probe_epoch = int(epoch)

    def export_probe_rows(self) -> List[Dict[str, float | int | str]]:
        return list(self._probe_rows)

    def export_last_dataflow_summary(self) -> Dict[str, object] | None:
        if self._last_dataflow_summary is None:
            return None
        return dict(self._last_dataflow_summary)

    @staticmethod
    def _normalized_entropy(weights: torch.Tensor) -> torch.Tensor:
        if weights.numel() == 0 or weights.shape[-1] <= 1:
            return torch.zeros(weights.shape[0], dtype=weights.dtype, device=weights.device)
        probs = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
        denom = torch.log(torch.tensor(float(weights.shape[-1]), dtype=weights.dtype, device=weights.device))
        return entropy / denom.clamp_min(1e-12)

    def _cosine_similarity(self, h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        h_norm = F.normalize(h, p=2, dim=-1, eps=self.input_norm_eps)
        p_norm = F.normalize(p, p=2, dim=-1, eps=self.input_norm_eps)
        return torch.matmul(h_norm, p_norm.transpose(0, 1))

    def _get_class_centers(self) -> torch.Tensor:
        if self.prototype_geometry_mode == "flat":
            raise RuntimeError("class centers only exist for center_subproto geometry")
        assert self.center_params is not None
        return F.normalize(self.center_params, p=2, dim=-1, eps=self.input_norm_eps)

    def _get_sub_prototypes(self, class_centers: torch.Tensor) -> torch.Tensor:
        if self.prototype_geometry_mode in {"flat", "center_only"}:
            raise RuntimeError("sub prototypes only exist for center_subproto geometry")
        assert self.sub_offsets is not None
        return F.normalize(
            class_centers[:, None, :] + self.sub_offsets,
            p=2,
            dim=-1,
            eps=self.input_norm_eps,
        )

    def _summarize_subprototype_geometry(
        self,
        *,
        class_center: torch.Tensor,
        same_proto: torch.Tensor,
        same_cosine: torch.Tensor,
    ) -> Dict[str, float]:
        with torch.no_grad():
            metrics: Dict[str, float] = {}
            center_to_subproto_cos = torch.sum(class_center.unsqueeze(0) * same_proto, dim=-1)
            metrics["center_to_subproto_cos_mean"] = float(center_to_subproto_cos.mean().item())
            metrics["center_to_subproto_cos_std"] = float(center_to_subproto_cos.std(unbiased=False).item())

            if same_proto.shape[0] > 1:
                pairwise = torch.matmul(same_proto, same_proto.transpose(0, 1))
                mask = ~torch.eye(same_proto.shape[0], dtype=torch.bool, device=same_proto.device)
                off_diag = pairwise[mask]
                metrics["subproto_pairwise_cos_mean"] = float(off_diag.mean().item())
                metrics["subproto_pairwise_cos_std"] = float(off_diag.std(unbiased=False).item())
                pairwise_no_diag = pairwise.masked_fill(
                    torch.eye(same_proto.shape[0], dtype=torch.bool, device=same_proto.device),
                    float("-inf"),
                )
                metrics["subproto_nn_cos_mean"] = float(pairwise_no_diag.max(dim=-1).values.mean().item())

                centered = same_proto - same_proto.mean(dim=0, keepdim=True)
                cov = centered.transpose(0, 1) @ centered / float(max(1, same_proto.shape[0] - 1))
                eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
                metrics["subproto_cov_trace"] = float(eigvals.sum().item())
                metrics["subproto_cov_top_eig"] = float(eigvals.max().item())
                if float(eigvals.sum().item()) > 0.0:
                    eig_probs = eigvals / eigvals.sum().clamp_min(1e-12)
                    entropy = -(eig_probs * eig_probs.clamp_min(1e-12).log()).sum()
                    metrics["subproto_cov_effective_rank"] = float(torch.exp(entropy).item())
                else:
                    metrics["subproto_cov_effective_rank"] = 0.0
            else:
                metrics["subproto_pairwise_cos_mean"] = 1.0
                metrics["subproto_pairwise_cos_std"] = 0.0
                metrics["subproto_nn_cos_mean"] = 1.0
                metrics["subproto_cov_trace"] = 0.0
                metrics["subproto_cov_top_eig"] = 0.0
                metrics["subproto_cov_effective_rank"] = 0.0

            if same_cosine.shape[-1] > 1:
                top2 = torch.topk(same_cosine, k=2, dim=-1).values
                metrics["subproto_cos_top1_top2_gap_mean"] = float((top2[:, 0] - top2[:, 1]).mean().item())
            else:
                metrics["subproto_cos_top1_top2_gap_mean"] = 0.0
            metrics["subproto_cos_max_minus_mean_mean"] = float(
                (same_cosine.max(dim=-1).values - same_cosine.mean(dim=-1)).mean().item()
            )
            metrics["subproto_cos_var_mean"] = float(same_cosine.var(dim=-1, unbiased=False).mean().item())
            metrics["subproto_cos_top1_mean"] = float(same_cosine.max(dim=-1).values.mean().item())
            return metrics

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
        same_weights_override: torch.Tensor | None = None,
        opp_weights_override: torch.Tensor | None = None,
        extra_diag: Dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, Dict[str, object]]:
        same_mass, opp_mass = self._resolve_support_masses()
        batch_size = int(h.shape[0])
        h, same_proto, opp_proto = self._maybe_normalize_inputs(h, same_proto, opp_proto)

        if same_weights_override is not None:
            same_w = same_weights_override.to(dtype=h.dtype, device=h.device)
        elif same_proto.shape[0] == 1:
            same_w = torch.full(
                (batch_size, 1),
                float(same_mass),
                dtype=h.dtype,
                device=h.device,
            )
        else:
            same_w = torch.softmax(self._squared_similarity(h, same_proto), dim=-1) * float(same_mass)

        if opp_proto.shape[0] > 0 and opp_mass > 0.0:
            if opp_weights_override is not None:
                opp_w = opp_weights_override.to(dtype=h.dtype, device=h.device)
            else:
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
            opp_proto = opp_proto[:0]

        gram = torch.einsum("bn,nd,ne->bde", weights, support, support)
        rhs = torch.einsum("bn,nd,n->bd", weights, support, targets)
        weight_sqrt = torch.sqrt(weights.clamp_min(0.0))
        weighted_support = weight_sqrt[:, :, None] * support.unsqueeze(0)
        weighted_targets = weight_sqrt * targets.unsqueeze(0)
        solve_matrix_shape = list(gram.shape)
        solve_space_dim = int(gram.shape[-1])

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
            solve_matrix_shape = list(dual_gram.shape)
            solve_space_dim = int(dual_gram.shape[-1])
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
                solve_matrix_shape = list(dual_gram.shape)
                solve_space_dim = int(dual_gram.shape[-1])
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
        with torch.no_grad():
            diag: Dict[str, object] = {
                "class_idx": int(class_idx),
                "committee_idx": int(committee_idx),
                "batch_size": int(batch_size),
                "feature_dim": int(self.feature_dim),
                "same_proto_count": int(same_proto.shape[0]),
                "opp_proto_count": int(opp_proto.shape[0]),
                "support_count": int(support.shape[0]),
                "support_shape": [int(v) for v in support.shape],
                "positive_support_shape": [int(v) for v in same_proto.shape],
                "negative_support_shape": [int(v) for v in opp_proto.shape],
                "weights_shape": [int(v) for v in weights.shape],
                "targets_shape": [int(v) for v in targets.shape],
                "gram_shape": [int(v) for v in gram.shape],
                "rhs_shape": [int(v) for v in rhs.shape],
                "solve_matrix_shape": [int(v) for v in solve_matrix_shape],
                "solve_space_dim": int(solve_space_dim),
                "solved_shape": [int(v) for v in solved.shape],
                "same_weight_max_mean": float(same_w.max(dim=-1).values.mean().item()),
                "same_weight_entropy_mean": float(self._normalized_entropy(same_w).mean().item()),
                "same_weight_sum_mean": float(same_w.sum(dim=-1).mean().item()),
                "gram_trace_mean": float(torch.einsum("bii->b", gram).mean().item()),
                "rhs_norm_mean": float(torch.linalg.vector_norm(rhs, dim=-1).mean().item()),
                "weight_norm_mean": float(torch.linalg.vector_norm(solved, dim=-1).mean().item()),
                "effective_ridge_mean": float(ridge_scale.mean().item()),
            }
            if opp_proto.shape[0] > 0 and opp_mass > 0.0:
                diag["opp_weight_max_mean"] = float(opp_w.max(dim=-1).values.mean().item())
                diag["opp_weight_entropy_mean"] = float(self._normalized_entropy(opp_w).mean().item())
                diag["opp_weight_sum_mean"] = float(opp_w.sum(dim=-1).mean().item())
            else:
                diag["opp_weight_max_mean"] = 0.0
                diag["opp_weight_entropy_mean"] = 0.0
                diag["opp_weight_sum_mean"] = 0.0
            if extra_diag is not None:
                diag.update(extra_diag)
        return solved, diag

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch_size = int(h.shape[0])
        eye = torch.eye(self.feature_dim, dtype=h.dtype, device=h.device)
        class_logits = []
        class_summaries: List[Dict[str, object]] = []
        class_centers_shape: List[int] | None = None
        sub_prototypes_shape: List[int] | None = None
        class_prior_shape: List[int] | None = None

        if self.prototype_geometry_mode == "center_subproto":
            class_centers = self._get_class_centers().to(dtype=h.dtype, device=h.device)
            sub_prototypes = self._get_sub_prototypes(class_centers).to(dtype=h.dtype, device=h.device)
            class_prior = torch.softmax(
                self._cosine_similarity(h, class_centers) / self.class_prior_temperature,
                dim=-1,
            )
            class_centers_shape = [int(v) for v in class_centers.shape]
            sub_prototypes_shape = [int(v) for v in sub_prototypes.shape]
            class_prior_shape = [int(v) for v in class_prior.shape]
            same_mass, opp_mass = self._resolve_support_masses()
            for class_idx in range(self.num_classes):
                same_proto = sub_prototypes[class_idx]
                same_cosine = self._cosine_similarity(h, same_proto)
                same_routing = torch.softmax(same_cosine / self.subproto_temperature, dim=-1)
                same_weights = same_routing * float(same_mass)
                mask = torch.arange(self.num_classes, device=h.device) != class_idx
                opp_proto = class_centers[mask]
                if opp_proto.shape[0] > 0 and opp_mass > 0.0:
                    opp_prior = class_prior[:, mask]
                    opp_weights = (
                        opp_prior / opp_prior.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    ) * float(opp_mass)
                else:
                    opp_weights = None
                subproto_geom_metrics = self._summarize_subprototype_geometry(
                    class_center=class_centers[class_idx],
                    same_proto=same_proto,
                    same_cosine=same_cosine,
                )
                solved, diag = self._solve_direction(
                    h,
                    same_proto=same_proto,
                    opp_proto=opp_proto,
                    eye=eye,
                    class_idx=class_idx,
                    same_weights_override=same_weights,
                    opp_weights_override=opp_weights,
                    extra_diag={
                        "class_prior_shape": [int(v) for v in class_prior.shape],
                        "subproto_routing_shape": [int(v) for v in same_routing.shape],
                        "class_centers_shape": [int(v) for v in class_centers.shape],
                        "sub_prototypes_shape": [int(v) for v in sub_prototypes.shape],
                        "class_prior_max_mean": float(class_prior.max(dim=-1).values.mean().item()),
                        "class_prior_entropy_mean": float(self._normalized_entropy(class_prior).mean().item()),
                        "class_prior_temperature": float(self.class_prior_temperature),
                        "subproto_temperature": float(self.subproto_temperature),
                        "subproto_weight_max_mean": float(same_routing.max(dim=-1).values.mean().item()),
                        "subproto_weight_entropy_mean": float(self._normalized_entropy(same_routing).mean().item()),
                        "same_support_norm_mean": float(torch.linalg.vector_norm(same_proto, dim=-1).mean().item()),
                        "opp_center_weight_entropy_mean": 0.0
                        if opp_weights is None
                        else float(self._normalized_entropy(opp_weights / float(max(opp_mass, 1e-12))).mean().item()),
                        **subproto_geom_metrics,
                    },
                )
                class_logit = torch.sum(h * solved, dim=-1)
                diag["aggregation_mode"] = "center_subproto_weighted_support"
                class_summaries.append(diag)
                class_logits.append(class_logit)
        elif self.prototype_geometry_mode == "center_only":
            class_centers = self._get_class_centers().to(dtype=h.dtype, device=h.device)
            class_prior = torch.softmax(
                self._cosine_similarity(h, class_centers) / self.class_prior_temperature,
                dim=-1,
            )
            class_centers_shape = [int(v) for v in class_centers.shape]
            class_prior_shape = [int(v) for v in class_prior.shape]
            for class_idx in range(self.num_classes):
                same_proto = class_centers[class_idx : class_idx + 1]
                same_cosine = self._cosine_similarity(h, same_proto)
                same_routing = torch.ones(
                    (batch_size, 1),
                    dtype=h.dtype,
                    device=h.device,
                )
                solved, diag = self._solve_direction(
                    h,
                    same_proto=same_proto,
                    opp_proto=class_centers[:0],
                    eye=eye,
                    class_idx=class_idx,
                    same_weights_override=same_routing,
                    opp_weights_override=None,
                    extra_diag={
                        "class_prior_shape": [int(v) for v in class_prior.shape],
                        "subproto_routing_shape": [int(v) for v in same_routing.shape],
                        "class_centers_shape": [int(v) for v in class_centers.shape],
                        "sub_prototypes_shape": None,
                        "class_prior_max_mean": float(class_prior.max(dim=-1).values.mean().item()),
                        "class_prior_entropy_mean": float(self._normalized_entropy(class_prior).mean().item()),
                        "class_prior_temperature": float(self.class_prior_temperature),
                        "subproto_temperature": float(self.subproto_temperature),
                        "subproto_weight_max_mean": 1.0,
                        "subproto_weight_entropy_mean": 0.0,
                        "same_support_norm_mean": float(torch.linalg.vector_norm(same_proto, dim=-1).mean().item()),
                        "opp_center_weight_entropy_mean": 0.0,
                        "center_to_subproto_cos_mean": 1.0,
                        "center_to_subproto_cos_std": 0.0,
                        "subproto_pairwise_cos_mean": 1.0,
                        "subproto_pairwise_cos_std": 0.0,
                        "subproto_nn_cos_mean": 1.0,
                        "subproto_cov_trace": 0.0,
                        "subproto_cov_top_eig": 0.0,
                        "subproto_cov_effective_rank": 0.0,
                        "subproto_cos_top1_top2_gap_mean": 0.0,
                        "subproto_cos_max_minus_mean_mean": 0.0,
                        "subproto_cos_var_mean": 0.0,
                        "subproto_cos_top1_mean": float(same_cosine.max(dim=-1).values.mean().item()),
                    },
                )
                class_logit = torch.sum(h * solved, dim=-1)
                diag["aggregation_mode"] = "center_only_support"
                class_summaries.append(diag)
                class_logits.append(class_logit)
        else:
            for class_idx in range(self.num_classes):
                assert self.prototypes is not None
                same_proto = self.prototypes[class_idx]
                opp_proto = self.prototypes[
                    torch.arange(self.num_classes, device=h.device) != class_idx
                ].reshape(-1, self.feature_dim)
                if self.prototype_aggregation == "pooled":
                    solved, diag = self._solve_direction(
                        h,
                        same_proto=same_proto,
                        opp_proto=opp_proto,
                        eye=eye,
                        class_idx=class_idx,
                    )
                    class_logit = torch.sum(h * solved, dim=-1)
                    diag["aggregation_mode"] = "pooled"
                    class_summaries.append(diag)
                elif self.prototype_aggregation == "committee_mean":
                    member_logits = []
                    member_summaries: List[Dict[str, object]] = []
                    for proto_idx in range(int(same_proto.shape[0])):
                        solved, diag = self._solve_direction(
                            h,
                            same_proto=same_proto[proto_idx : proto_idx + 1],
                            opp_proto=opp_proto,
                            eye=eye,
                            class_idx=class_idx,
                            committee_idx=proto_idx,
                        )
                        member_logits.append(torch.sum(h * solved, dim=-1))
                        member_summaries.append(diag)
                    class_logit = torch.stack(member_logits, dim=0).mean(dim=0)
                    class_summaries.append(
                        {
                            "class_idx": int(class_idx),
                            "aggregation_mode": "committee_mean",
                            "committee_size": int(len(member_summaries)),
                            "members": member_summaries,
                        }
                    )
                else:
                    raise ValueError(f"unsupported prototype aggregation: {self.prototype_aggregation}")
                class_logits.append(class_logit)
        local_logits = torch.stack(class_logits, dim=-1).reshape(batch_size, self.num_classes)
        self._last_dataflow_summary = {
            "split": str(self._probe_split),
            "epoch": int(self._probe_epoch),
            "latent_shape": [int(v) for v in h.shape],
            "prototype_geometry_mode": self.prototype_geometry_mode,
            "prototypes_shape": None if self.prototypes is None else [int(v) for v in self.prototypes.shape],
            "class_centers_shape": class_centers_shape,
            "sub_prototypes_shape": sub_prototypes_shape,
            "class_prior_shape": class_prior_shape,
            "local_logit_shape": [int(v) for v in local_logits.shape],
            "feature_dim": int(self.feature_dim),
            "num_classes": int(self.num_classes),
            "prototypes_per_class": int(self.prototypes_per_class),
            "support_mode": self.support_mode,
            "prototype_aggregation": self.prototype_aggregation,
            "routing_temperature": float(self.routing_temperature),
            "class_prior_temperature": float(self.class_prior_temperature),
            "subproto_temperature": float(self.subproto_temperature),
            "solve_mode": self.solve_mode,
            "ridge_mode": self.ridge_mode,
            "input_norm_mode": self.input_norm_mode,
            "class_summaries": class_summaries,
        }
        return local_logits


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
        prototype_geometry_mode: str = "flat",
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
            prototype_geometry_mode=str(prototype_geometry_mode),
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
