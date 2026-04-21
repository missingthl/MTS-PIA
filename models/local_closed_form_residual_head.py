from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

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
        tangent_rank: int = 2,
        tangent_source: str = "subproto_offsets",
        prob_tangent_version: str = "v1",
        rank_selection_mode: str = "mdl",
        posterior_mode: str = "gaussian_dimnorm",
        posterior_student_dof: float = 3.0,
        mdl_penalty_beta: float = 1.0,
        gaussian_refine_variant: str = "base",
        mdl_zero_rank_rescue_margin: float = 0.03,
        local_solver_competition_mode: str = "none",
        relative_solver_temperature: float = 1.0,
        abs_gate_activity_floor: float = 1e-6,
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
        self.tangent_rank = max(int(tangent_rank), 1)
        self.tangent_source = str(tangent_source)
        self.prob_tangent_version = str(prob_tangent_version)
        self.rank_selection_mode = str(rank_selection_mode)
        self.posterior_mode = str(posterior_mode)
        self.posterior_student_dof = max(float(posterior_student_dof), 1e-6)
        self.mdl_penalty_beta = max(float(mdl_penalty_beta), 1e-6)
        self.gaussian_refine_variant = str(gaussian_refine_variant)
        self.mdl_zero_rank_rescue_margin = max(float(mdl_zero_rank_rescue_margin), 0.0)
        self.local_solver_competition_mode = str(local_solver_competition_mode)
        self.relative_solver_temperature = max(float(relative_solver_temperature), 1e-6)
        self.abs_gate_activity_floor = max(float(abs_gate_activity_floor), 0.0)
        if self.ridge_mode not in {"fixed", "trace_adaptive"}:
            raise ValueError(f"unsupported ridge mode: {self.ridge_mode}")
        if self.solve_mode not in {"ridge_solve", "pinv", "dual_ridge", "dual_pinv"}:
            raise ValueError(f"unsupported solve mode: {self.solve_mode}")
        if self.input_norm_mode not in {"none", "l2_hypersphere"}:
            raise ValueError(f"unsupported input norm mode: {self.input_norm_mode}")
        if self.prototype_geometry_mode not in {"flat", "center_subproto", "center_only", "center_tangent", "center_prob_tangent"}:
            raise ValueError(f"unsupported prototype geometry mode: {self.prototype_geometry_mode}")
        if self.prototype_geometry_mode in {"center_subproto", "center_only", "center_tangent", "center_prob_tangent"} and self.prototype_aggregation != "pooled":
            raise ValueError(f"{self.prototype_geometry_mode} geometry currently supports pooled aggregation only")
        if self.tangent_source not in {"subproto_offsets"}:
            raise ValueError(f"unsupported tangent source: {self.tangent_source}")
        if self.prob_tangent_version not in {"v1", "v2", "v3"}:
            raise ValueError(f"unsupported prob tangent version: {self.prob_tangent_version}")
        if self.rank_selection_mode not in {"mdl", "bic"}:
            raise ValueError(f"unsupported rank selection mode: {self.rank_selection_mode}")
        if self.posterior_mode not in {"gaussian_dimnorm", "student_t"}:
            raise ValueError(f"unsupported posterior mode: {self.posterior_mode}")
        if self.gaussian_refine_variant not in {"base", "trace_floor", "trace_floor_mdl_margin"}:
            raise ValueError(f"unsupported gaussian refine variant: {self.gaussian_refine_variant}")
        if self.local_solver_competition_mode not in {"none", "relcomp"}:
            raise ValueError(f"unsupported local solver competition mode: {self.local_solver_competition_mode}")
        if self.prototype_geometry_mode in {"center_tangent", "center_prob_tangent"} and self.support_mode != "same_only":
            raise ValueError(f"{self.prototype_geometry_mode} geometry currently supports same_only support mode only")
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
        self._last_batch_routing_payload: Dict[str, object] | None = None

    def _initialize_center_subproto_offsets(self) -> None:
        if self.prototype_geometry_mode not in {"center_subproto", "center_tangent", "center_prob_tangent"}:
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
        self._last_batch_routing_payload = None

    def set_probe_context(self, *, split: str, epoch: int) -> None:
        self._probe_split = str(split)
        self._probe_epoch = int(epoch)

    def export_probe_rows(self) -> List[Dict[str, float | int | str]]:
        return list(self._probe_rows)

    def export_last_dataflow_summary(self) -> Dict[str, object] | None:
        if self._last_dataflow_summary is None:
            return None
        return dict(self._last_dataflow_summary)

    def export_last_batch_routing_payload(self) -> Dict[str, object] | None:
        if self._last_batch_routing_payload is None:
            return None
        return dict(self._last_batch_routing_payload)

    def export_learned_prototype_geometry_summary(self) -> Dict[str, object] | None:
        if self.prototype_geometry_mode == "flat":
            return None
        with torch.no_grad():
            if self.prototype_geometry_mode == "center_only":
                return {
                    "prototype_geometry_mode": "center_only",
                    "center_to_subproto_cos_mean": 1.0,
                    "center_to_subproto_cos_std": 0.0,
                    "subproto_pairwise_cos_mean": 1.0,
                    "subproto_pairwise_cos_std": 0.0,
                    "subproto_nn_cos_mean": 1.0,
                    "subproto_cov_trace": 0.0,
                    "subproto_cov_top_eig": 0.0,
                    "subproto_cov_effective_rank": 1.0,
                    "effective_support_count": 1,
                }

            class_centers = self._get_class_centers()
            sub_prototypes = self._get_sub_prototypes(class_centers)
            metric_keys = [
                "center_to_subproto_cos_mean",
                "center_to_subproto_cos_std",
                "subproto_pairwise_cos_mean",
                "subproto_pairwise_cos_std",
                "subproto_nn_cos_mean",
                "subproto_cov_trace",
                "subproto_cov_top_eig",
                "subproto_cov_effective_rank",
            ]
            values = {k: [] for k in metric_keys}
            for class_idx in range(self.num_classes):
                metrics = self._summarize_subprototype_geometry(
                    class_center=class_centers[class_idx],
                    same_proto=sub_prototypes[class_idx],
                    same_cosine=self._cosine_similarity(
                        class_centers[class_idx : class_idx + 1],
                        sub_prototypes[class_idx],
                    ),
                )
                for key in metric_keys:
                    values[key].append(float(metrics[key]))
            return {
                "prototype_geometry_mode": str(self.prototype_geometry_mode),
                **{key: float(sum(vals) / max(1, len(vals))) for key, vals in values.items()},
                "effective_support_count": int(self.prototypes_per_class),
            }

    def export_tangent_probe_payload(self) -> Dict[str, object] | None:
        if self.prototype_geometry_mode not in {"center_subproto", "center_tangent", "center_prob_tangent"}:
            return None
        with torch.no_grad():
            class_centers = self._get_class_centers()
            sub_prototypes = self._get_sub_prototypes(class_centers)
            class_rows: List[Dict[str, object]] = []
            rank95_values: List[int] = []
            effective_ranks: List[float] = []
            spectral_gaps: List[float] = []
            top1_energy_ratios: List[float] = []
            actual_basis_ranks: List[int] = []
            for class_idx in range(self.num_classes):
                tangent_payload = self._compute_tangent_support(
                    class_center=class_centers[class_idx],
                    same_proto=sub_prototypes[class_idx],
                    sub_offsets=None if self.sub_offsets is None else self.sub_offsets[class_idx],
                )
                class_rows.append(
                    {
                        "class_idx": int(class_idx),
                        "prototype_geometry_mode": str(self.prototype_geometry_mode),
                        "tangent_source": str(self.tangent_source),
                        "requested_tangent_rank": int(self.tangent_rank),
                        "actual_tangent_rank": int(tangent_payload["actual_rank"]),
                        "rank95": int(tangent_payload["rank95"]),
                        "effective_rank": float(tangent_payload["effective_rank"]),
                        "top1_energy_ratio": float(tangent_payload["top1_energy_ratio"]),
                        "top1_top2_spectral_gap": float(tangent_payload["top1_top2_spectral_gap"]),
                        "singular_values": [float(v) for v in tangent_payload["singular_values"]],
                        "energy_probs": [float(v) for v in tangent_payload["energy_probs"]],
                        "cumulative_energy": [float(v) for v in tangent_payload["cumulative_energy"]],
                    }
                )
                rank95_values.append(int(tangent_payload["rank95"]))
                effective_ranks.append(float(tangent_payload["effective_rank"]))
                spectral_gaps.append(float(tangent_payload["top1_top2_spectral_gap"]))
                top1_energy_ratios.append(float(tangent_payload["top1_energy_ratio"]))
                actual_basis_ranks.append(int(tangent_payload["actual_rank"]))

            observed = sorted({max(1, int(v)) for v in rank95_values})
            if not observed:
                recommended = [1]
            elif len(observed) <= 3:
                recommended = observed
            else:
                recommended = sorted({observed[0], observed[len(observed) // 2], observed[-1]})
            recommended = [int(min(self.prototypes_per_class, max(1, v))) for v in recommended]
            recommended = sorted({int(v) for v in recommended})
            summary = {
                "prototype_geometry_mode": str(self.prototype_geometry_mode),
                "tangent_source": str(self.tangent_source),
                "requested_tangent_rank": int(self.tangent_rank),
                "num_classes": int(self.num_classes),
                "prototypes_per_class": int(self.prototypes_per_class),
                "rank95_mean": float(sum(rank95_values) / max(1, len(rank95_values))),
                "effective_rank_mean": float(sum(effective_ranks) / max(1, len(effective_ranks))),
                "top1_energy_ratio_mean": float(sum(top1_energy_ratios) / max(1, len(top1_energy_ratios))),
                "top1_top2_spectral_gap_mean": float(sum(spectral_gaps) / max(1, len(spectral_gaps))),
                "actual_tangent_rank_mean": float(sum(actual_basis_ranks) / max(1, len(actual_basis_ranks))),
                "recommended_candidate_ranks": recommended,
                "rank95_values": rank95_values,
            }
            return {
                "summary": summary,
                "class_rows": class_rows,
            }

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
            raise RuntimeError("class centers only exist for center-based prototype geometry")
        assert self.center_params is not None
        return torch.nan_to_num(
            F.normalize(self.center_params, p=2, dim=-1, eps=self.input_norm_eps),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _get_sub_prototypes(self, class_centers: torch.Tensor) -> torch.Tensor:
        if self.prototype_geometry_mode in {"flat", "center_only"}:
            raise RuntimeError("sub prototypes only exist for center_subproto/center_tangent geometry")
        assert self.sub_offsets is not None
        return torch.nan_to_num(
            F.normalize(
                class_centers[:, None, :] + self.sub_offsets,
                p=2,
                dim=-1,
                eps=self.input_norm_eps,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _scale_aware_eps(self, reference: torch.Tensor | float, *, multiplier: float = 1e-6) -> float:
        if isinstance(reference, torch.Tensor):
            ref_tensor = torch.nan_to_num(reference.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            if ref_tensor.numel() == 0:
                scale = 1.0
            else:
                scale = float(ref_tensor.abs().max().item())
        else:
            scale = abs(float(reference))
        scale = max(scale, 1.0)
        return max(float(self.input_norm_eps), float(multiplier) * scale, 1e-12)

    @staticmethod
    def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
        return 0.5 * (matrix + matrix.transpose(-1, -2))

    def _compute_ledoit_wolf_shrunk_covariance(
        self,
        samples: torch.Tensor,
    ) -> Dict[str, object]:
        x = torch.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        if x.dim() != 2:
            raise ValueError(f"expected 2D samples, got shape={list(x.shape)}")
        n_samples = int(x.shape[0])
        dim = int(x.shape[1])
        if n_samples <= 0:
            raise ValueError("Ledoit-Wolf covariance requires at least one sample")

        # `subproto_offsets` are already defined relative to the class center, so
        # we treat them as an already-centered local cloud instead of removing
        # their sample mean once more. This preserves the intended 4-direction
        # local structure rather than collapsing it to rank <= 3.
        centered = x
        x2 = centered.square()
        emp_cov = self._symmetrize(centered.transpose(0, 1) @ centered / float(max(1, n_samples)))
        emp_cov_trace = x2.sum(dim=0) / float(max(1, n_samples))
        mu = float(emp_cov_trace.sum().item()) / float(max(1, dim))

        beta_ = float((x2.transpose(0, 1) @ x2).sum().item())
        delta_raw = float(((centered.transpose(0, 1) @ centered).square()).sum().item())
        delta_scaled = delta_raw / float(max(1, n_samples**2))
        beta = (beta_ / float(max(1, n_samples)) - delta_scaled) / float(max(1, dim * n_samples))
        delta = (delta_scaled - 2.0 * mu * float(emp_cov_trace.sum().item()) + float(dim) * (mu**2)) / float(max(1, dim))
        delta = max(delta, self._scale_aware_eps(emp_cov))
        beta = min(max(beta, 0.0), delta)
        shrinkage = 0.0 if beta <= 0.0 else float(beta / delta)
        shrinkage = float(min(max(shrinkage, 0.0), 1.0))

        target = torch.eye(dim, dtype=emp_cov.dtype, device=emp_cov.device) * float(mu)
        shrunk_cov = self._symmetrize((1.0 - shrinkage) * emp_cov + shrinkage * target)
        return {
            "centered_samples": centered,
            "empirical_cov": emp_cov,
            "shrunk_cov": shrunk_cov,
            "shrinkage_alpha": float(shrinkage),
            "mu": float(mu),
        }

    @staticmethod
    def _effective_rank_from_eigvals(eigvals: torch.Tensor) -> float:
        if eigvals.numel() == 0:
            return 0.0
        total = eigvals.sum().clamp_min(1e-12)
        probs = eigvals / total
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
        return float(torch.exp(entropy).item())

    def _ppca_parameter_count(self, *, rank: int, dim: int) -> int:
        rank = int(max(0, rank))
        dim = int(max(1, dim))
        return int(rank * (dim + 1) - rank * (rank - 1) / 2 + 1)

    def _posterior_stat_summary(self, tensor: torch.Tensor) -> Dict[str, float]:
        values = tensor.reshape(-1)
        if values.numel() == 0:
            return {"mean": 0.0, "std": 0.0, "q10": 0.0, "q50": 0.0, "q90": 0.0}
        values = values.to(dtype=torch.float32)
        return {
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
            "q10": float(torch.quantile(values, 0.10).item()),
            "q50": float(torch.quantile(values, 0.50).item()),
            "q90": float(torch.quantile(values, 0.90).item()),
        }

    def _solver_top1_occupancy_entropy(
        self,
        *,
        top1_index: torch.Tensor,
        active_mask: torch.Tensor,
        solver_count: int,
    ) -> float:
        solver_count = int(max(1, solver_count))
        if solver_count <= 1:
            return 0.0
        active_index = top1_index[active_mask]
        if active_index.numel() == 0:
            return 0.0
        counts = torch.bincount(active_index.to(dtype=torch.int64), minlength=solver_count).to(dtype=torch.float32)
        probs = counts / counts.sum().clamp_min(1.0)
        positive = probs > 0
        if not bool(positive.any()):
            return 0.0
        entropy = -(probs[positive] * probs[positive].log()).sum()
        denom = max(math.log(float(solver_count)), 1e-12)
        return float((entropy / denom).item())

    def _compose_relative_solver_competition(
        self,
        *,
        solved_bank: torch.Tensor,
        delta_bank: torch.Tensor,
        abs_gate: torch.Tensor,
    ) -> Dict[str, torch.Tensor | float | int]:
        batch_size = int(delta_bank.shape[0])
        solver_count = int(delta_bank.shape[1]) if delta_bank.dim() == 2 else 0
        if solver_count <= 0:
            raise ValueError("solver bank must contain at least one solver")

        if solver_count == 1:
            raw_weights = torch.ones((batch_size, 1), dtype=delta_bank.dtype, device=delta_bank.device)
            active_mask = torch.zeros((batch_size,), dtype=torch.bool, device=delta_bank.device)
        else:
            raw_weights = torch.softmax(delta_bank / self.relative_solver_temperature, dim=-1)
            active_mask = abs_gate > float(self.abs_gate_activity_floor)

        default_weights = torch.zeros_like(raw_weights)
        default_weights[:, 0] = 1.0
        mix_weights = raw_weights if solver_count == 1 else torch.where(active_mask.unsqueeze(-1), raw_weights, default_weights)
        solved_mix = torch.einsum("bs,bsd->bd", mix_weights, solved_bank)
        delta_mix = torch.sum(mix_weights * delta_bank, dim=-1)
        delta_final = abs_gate * delta_mix

        if solver_count == 1:
            top1_index = torch.zeros((batch_size,), dtype=torch.int64, device=delta_bank.device)
            top1_weight = torch.zeros((batch_size,), dtype=delta_bank.dtype, device=delta_bank.device)
            rel_entropy = torch.zeros((batch_size,), dtype=delta_bank.dtype, device=delta_bank.device)
            rel_margin = torch.zeros((batch_size,), dtype=delta_bank.dtype, device=delta_bank.device)
        else:
            top2 = torch.topk(raw_weights, k=2, dim=-1).values
            raw_top1_weight, raw_top1_index = raw_weights.max(dim=-1)
            top1_index = torch.where(
                active_mask,
                raw_top1_index.to(dtype=torch.int64),
                torch.zeros_like(raw_top1_index, dtype=torch.int64),
            )
            top1_weight = torch.where(
                active_mask,
                raw_top1_weight,
                torch.zeros_like(raw_top1_weight),
            )
            rel_entropy = torch.where(
                active_mask,
                self._normalized_entropy(raw_weights),
                torch.zeros((batch_size,), dtype=delta_bank.dtype, device=delta_bank.device),
            )
            rel_margin = torch.where(
                active_mask,
                top2[:, 0] - top2[:, 1],
                torch.zeros((batch_size,), dtype=delta_bank.dtype, device=delta_bank.device),
            )

        solved_mix_norm = torch.linalg.vector_norm(solved_mix, dim=-1)
        occupancy_entropy = self._solver_top1_occupancy_entropy(
            top1_index=top1_index,
            active_mask=active_mask,
            solver_count=solver_count,
        )
        return {
            "solver_count": int(solver_count),
            "raw_weights": raw_weights,
            "mix_weights": mix_weights,
            "solved_mix": solved_mix,
            "delta_mix": delta_mix,
            "delta_final": delta_final,
            "top1_index": top1_index,
            "top1_weight": top1_weight,
            "entropy": rel_entropy,
            "margin": rel_margin,
            "active_mask": active_mask.to(dtype=delta_bank.dtype),
            "active_rate": float(active_mask.to(dtype=torch.float32).mean().item()),
            "solved_mix_norm": solved_mix_norm,
            "occupancy_entropy": float(occupancy_entropy),
        }

    def _select_prob_tangent_rank(
        self,
        eigvals: torch.Tensor,
        *,
        sample_count: int,
        max_rank: int | None = None,
    ) -> Dict[str, object]:
        dim = int(eigvals.numel())
        max_rank = int(min(4, self.prototypes_per_class, dim) if max_rank is None else min(max_rank, dim))
        scale_eps = self._scale_aware_eps(eigvals)
        sample_count = max(int(sample_count), 2)
        log_n = math.log(float(sample_count))
        rank_rows: List[Dict[str, float | int]] = []
        best_rank = 0
        best_sigma2 = scale_eps
        best_score: float | None = None
        mode = str(self.rank_selection_mode)
        beta = float(self.mdl_penalty_beta)

        for rank in range(max_rank + 1):
            if rank < dim:
                discarded = eigvals[rank:]
                sigma2 = float(discarded.mean().item())
            else:
                discarded = eigvals[:0]
                sigma2 = scale_eps
            sigma2 = max(sigma2, scale_eps)
            selected = eigvals[:rank]
            selected_logdet = float(torch.log(selected.clamp_min(scale_eps)).sum().item()) if rank > 0 else 0.0
            residual_dim = max(dim - rank, 1)
            neg_log_likelihood = 0.5 * float(sample_count) * (
                float(dim) * math.log(2.0 * math.pi)
                + selected_logdet
                + float(residual_dim) * math.log(sigma2)
                + float(dim)
            )
            param_count = self._ppca_parameter_count(rank=rank, dim=dim)
            bic_score = 2.0 * neg_log_likelihood + float(param_count) * log_n
            mdl_penalty = 0.5 * beta * float(param_count) * log_n + 0.5 * beta * float(rank) * math.log(float(max(2, dim)))
            mdl_score = neg_log_likelihood + mdl_penalty
            score = mdl_score if mode == "mdl" else bic_score
            rank_rows.append(
                {
                    "rank": int(rank),
                    "sigma2": float(sigma2),
                    "selected_logdet": float(selected_logdet),
                    "neg_log_likelihood": float(neg_log_likelihood),
                    "penalty_term": float(mdl_penalty),
                    "beta_scaled_penalty": float(mdl_penalty),
                    "bic_score": float(bic_score),
                    "mdl_score": float(mdl_score),
                    "param_count": int(param_count),
                    "residual_dim": int(residual_dim),
                    "selected_eig_sum": float(selected.sum().item()) if rank > 0 else 0.0,
                }
            )
            if best_score is None or float(score) < float(best_score):
                best_rank = int(rank)
                best_sigma2 = float(sigma2)
                best_score = float(score)

        return {
            "selected_rank": int(best_rank),
            "sigma2": float(best_sigma2),
            "rank_rows": rank_rows,
        }

    def _compute_prob_tangent_support(
        self,
        *,
        class_center: torch.Tensor,
        same_proto: torch.Tensor,
        sub_offsets: torch.Tensor | None = None,
    ) -> Dict[str, object]:
        if self.tangent_source == "subproto_offsets" and sub_offsets is not None:
            deviations = sub_offsets.to(dtype=same_proto.dtype, device=same_proto.device)
        else:
            deviations = same_proto - class_center.unsqueeze(0)
        deviations = deviations - torch.sum(deviations * class_center.unsqueeze(0), dim=-1, keepdim=True) * class_center.unsqueeze(0)
        deviations = torch.nan_to_num(deviations, nan=0.0, posinf=0.0, neginf=0.0)
        if deviations.shape[0] == 0 or float(torch.linalg.vector_norm(deviations).item()) <= 0.0:
            support = class_center.unsqueeze(0)
            empty_tensor = deviations.new_zeros((0,))
            return {
                "support": support,
                "basis": deviations[:0],
                "selected_rank": 0,
                "rank95": 0,
                "effective_rank": 0.0,
                "shrinkage_alpha": 0.0,
                "sigma2": self._scale_aware_eps(class_center),
                "selected_eigvals_tensor": empty_tensor,
                "selected_eigvals": [],
                "energy_probs": [],
                "cumulative_energy": [],
                "rank_rows": [],
                "signal_variance": empty_tensor,
                "trace_per_dim": self._scale_aware_eps(class_center),
                "rank0_score": None,
                "rank1_score": None,
                "rank01_relative_gap": None,
                "zero_rank_rescued": False,
            }

        lw_payload = self._compute_ledoit_wolf_shrunk_covariance(deviations)
        centered = lw_payload["centered_samples"]
        sample_count = int(centered.shape[0])
        dual_cov = self._symmetrize(centered @ centered.transpose(0, 1) / float(max(1, sample_count)))
        dual_eigvals, dual_eigvecs = torch.linalg.eigh(dual_cov)
        dual_eigvals = torch.flip(dual_eigvals.clamp_min(0.0), dims=[0])
        dual_eigvecs = torch.flip(dual_eigvecs, dims=[1])
        basis_rows: List[torch.Tensor] = []
        empirical_eigvals: List[torch.Tensor] = []
        eig_eps = self._scale_aware_eps(dual_eigvals)
        for eigval, dual_vec in zip(dual_eigvals.unbind(dim=0), dual_eigvecs.transpose(0, 1).unbind(dim=0)):
            eigval_value = float(eigval.item())
            if eigval_value <= eig_eps:
                continue
            basis_vec = torch.matmul(centered.transpose(0, 1), dual_vec) / math.sqrt(float(sample_count) * eigval_value)
            basis_vec = torch.nan_to_num(
                F.normalize(basis_vec.unsqueeze(0), p=2, dim=-1, eps=self.input_norm_eps).squeeze(0),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            basis_rows.append(basis_vec)
            empirical_eigvals.append(eigval)

        if basis_rows:
            empirical_eigvals_tensor = torch.stack(empirical_eigvals, dim=0)
            basis_all = torch.stack(basis_rows, dim=0)
        else:
            empirical_eigvals_tensor = deviations.new_zeros((0,))
            basis_all = deviations[:0]

        dim = int(deviations.shape[-1])
        shrinkage_alpha = float(lw_payload["shrinkage_alpha"])
        mu = float(lw_payload["mu"])
        shrunk_nonzero = (1.0 - shrinkage_alpha) * empirical_eigvals_tensor + shrinkage_alpha * float(mu)
        null_count = max(0, dim - int(shrunk_nonzero.numel()))
        if null_count > 0:
            shrunk_tail = deviations.new_full((null_count,), float(shrinkage_alpha * mu))
            full_shrunk_eigvals = torch.cat([shrunk_nonzero, shrunk_tail], dim=0)
        else:
            full_shrunk_eigvals = shrunk_nonzero

        total_energy = full_shrunk_eigvals.sum().clamp_min(1e-12)
        energy_probs = full_shrunk_eigvals / total_energy
        cumulative_energy = torch.cumsum(energy_probs, dim=0)
        rank95 = int(torch.nonzero(cumulative_energy >= 0.95, as_tuple=False)[0, 0].item() + 1) if full_shrunk_eigvals.numel() > 0 else 0
        effective_rank = self._effective_rank_from_eigvals(full_shrunk_eigvals)
        max_rank = int(min(4, self.prototypes_per_class, basis_all.shape[0]))
        trace_per_dim = float(full_shrunk_eigvals.mean().item()) if full_shrunk_eigvals.numel() > 0 else self._scale_aware_eps(class_center)
        rank0_score: float | None = None
        rank1_score: float | None = None
        rank01_relative_gap: float | None = None
        zero_rank_rescued = False

        if self.prob_tangent_version == "v1":
            selected_rank = int(max_rank)
            rank_rows: List[Dict[str, float | int]] = []
            if selected_rank < int(full_shrunk_eigvals.numel()):
                sigma2 = float(full_shrunk_eigvals[selected_rank:].mean().item())
            else:
                sigma2 = self._scale_aware_eps(full_shrunk_eigvals)
        else:
            rank_eigvals = shrunk_nonzero[:max(1, max_rank)] if shrunk_nonzero.numel() > 0 else full_shrunk_eigvals[:1]
            rank_payload = self._select_prob_tangent_rank(
                rank_eigvals,
                sample_count=int(deviations.shape[0]),
                max_rank=max_rank,
            )
            selected_rank = int(rank_payload["selected_rank"])
            rank_rows = list(rank_payload["rank_rows"])
            score_key = "mdl_score" if self.rank_selection_mode == "mdl" else "bic_score"
            row0 = next((row for row in rank_rows if int(row["rank"]) == 0), None)
            row1 = next((row for row in rank_rows if int(row["rank"]) == 1), None)
            if row0 is not None:
                rank0_score = float(row0.get(score_key, row0.get("mdl_score", 0.0)))
            if row1 is not None:
                rank1_score = float(row1.get(score_key, row1.get("mdl_score", 0.0)))
            if rank0_score is not None and rank1_score is not None:
                rank01_relative_gap = float((rank1_score - rank0_score) / max(abs(rank0_score), 1e-6))
            if (
                self.gaussian_refine_variant == "trace_floor_mdl_margin"
                and self.rank_selection_mode == "mdl"
                and selected_rank == 0
                and rank01_relative_gap is not None
                and rank01_relative_gap <= float(self.mdl_zero_rank_rescue_margin)
            ):
                selected_rank = 1
                zero_rank_rescued = True
            if selected_rank < int(full_shrunk_eigvals.numel()):
                sigma2 = float(full_shrunk_eigvals[selected_rank:].mean().item())
            else:
                sigma2 = self._scale_aware_eps(full_shrunk_eigvals)
        sigma2 = max(float(sigma2), self._scale_aware_eps(full_shrunk_eigvals))

        basis = basis_all[:selected_rank]
        selected_eigvals = shrunk_nonzero[:selected_rank]
        if self.prob_tangent_version == "v3":
            signal_variance = (selected_eigvals - float(sigma2)).clamp_min(self._scale_aware_eps(selected_eigvals))
        else:
            signal_variance = selected_eigvals.clamp_min(self._scale_aware_eps(selected_eigvals))

        if selected_rank > 0:
            tangent_points = class_center.unsqueeze(0) + torch.sqrt(signal_variance).unsqueeze(-1) * basis
            tangent_points = torch.nan_to_num(
                F.normalize(tangent_points, p=2, dim=-1, eps=self.input_norm_eps),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            support = torch.cat([class_center.unsqueeze(0), tangent_points], dim=0)
        else:
            support = class_center.unsqueeze(0)

        return {
            "support": support,
            "basis": basis,
            "selected_rank": int(selected_rank),
            "rank95": int(rank95),
            "effective_rank": float(effective_rank),
            "shrinkage_alpha": float(shrinkage_alpha),
            "sigma2": float(sigma2),
            "selected_eigvals_tensor": selected_eigvals,
            "selected_eigvals": selected_eigvals.detach().cpu().tolist(),
            "energy_probs": energy_probs.detach().cpu().tolist(),
            "cumulative_energy": cumulative_energy.detach().cpu().tolist(),
            "rank_rows": rank_rows,
            "signal_variance": signal_variance,
            "trace_per_dim": float(trace_per_dim),
            "rank0_score": None if rank0_score is None else float(rank0_score),
            "rank1_score": None if rank1_score is None else float(rank1_score),
            "rank01_relative_gap": None if rank01_relative_gap is None else float(rank01_relative_gap),
            "zero_rank_rescued": bool(zero_rank_rescued),
        }

    def _compute_prob_tangent_posterior(
        self,
        *,
        h: torch.Tensor,
        class_center: torch.Tensor,
        basis: torch.Tensor,
        selected_eigvals: torch.Tensor,
        sigma2: float,
        trace_per_dim: float,
    ) -> Dict[str, torch.Tensor]:
        sigma2_floor = max(float(sigma2), self._scale_aware_eps(selected_eigvals))
        centered = h - class_center.unsqueeze(0)
        feature_dim = max(1, int(h.shape[-1]))
        trace_floor = max(0.5 * float(trace_per_dim), 0.0)
        if self.gaussian_refine_variant in {"trace_floor", "trace_floor_mdl_margin"}:
            sigma2_used_scalar = max(sigma2_floor, trace_floor, self._scale_aware_eps(centered))
        else:
            sigma2_used_scalar = max(sigma2_floor, self._scale_aware_eps(centered))
        if basis.numel() == 0 or selected_eigvals.numel() == 0:
            residual = centered
            residual_dim = max(1, int(h.shape[-1]))
            residual_energy = torch.sum(residual * residual, dim=-1)
            residual_energy_per_dim = residual_energy / float(residual_dim)
            if self.posterior_mode == "student_t":
                sigma2_eff = residual_energy.new_full(
                    residual_energy.shape,
                    max(float(self.posterior_student_dof) * sigma2_used_scalar * float(feature_dim), self._scale_aware_eps(residual_energy)),
                )
                base = 1.0 + residual_energy / sigma2_eff.clamp_min(self._scale_aware_eps(residual_energy))
                log_confidence = -0.5 * float(self.posterior_student_dof + residual_dim) * torch.log(base.clamp_min(1.0))
            else:
                sigma2_eff = residual_energy.new_full(
                    residual_energy.shape,
                    max(sigma2_used_scalar * float(residual_dim) * float(feature_dim), self._scale_aware_eps(residual_energy)),
                )
                log_confidence = -residual_energy / (2.0 * sigma2_eff.clamp_min(self._scale_aware_eps(residual_energy)))
            confidence = torch.exp(log_confidence.clamp(min=-60.0, max=0.0))
            sigma2_used = residual_energy.new_full(residual_energy.shape, float(sigma2_used_scalar))
            return {
                "confidence": confidence.clamp(0.0, 1.0),
                "log_confidence": log_confidence,
                "residual_energy": residual_energy,
                "residual_energy_per_dim": residual_energy_per_dim,
                "sigma2_eff": sigma2_eff,
                "sigma2_used": sigma2_used,
                "residual": residual,
            }

        coeff = torch.matmul(centered, basis.transpose(0, 1))
        shrink = ((selected_eigvals - sigma2_floor) / selected_eigvals.clamp_min(self._scale_aware_eps(selected_eigvals))).clamp(0.0, 1.0)
        projected = torch.matmul(coeff * shrink.unsqueeze(0), basis)
        residual = centered - projected
        residual_dim = max(1, int(h.shape[-1] - basis.shape[0]))
        residual_energy = torch.sum(residual * residual, dim=-1)
        residual_energy_per_dim = residual_energy / float(residual_dim)
        if self.posterior_mode == "student_t":
            sigma2_eff = residual_energy.new_full(
                residual_energy.shape,
                max(float(self.posterior_student_dof) * sigma2_used_scalar * float(feature_dim), self._scale_aware_eps(residual_energy)),
            )
            base = 1.0 + residual_energy / sigma2_eff.clamp_min(self._scale_aware_eps(residual_energy))
            log_confidence = -0.5 * float(self.posterior_student_dof + residual_dim) * torch.log(base.clamp_min(1.0))
        else:
            sigma2_eff = residual_energy.new_full(
                residual_energy.shape,
                max(sigma2_used_scalar * float(residual_dim) * float(feature_dim), self._scale_aware_eps(residual_energy)),
            )
            log_confidence = -residual_energy / (2.0 * sigma2_eff.clamp_min(self._scale_aware_eps(residual_energy)))
        confidence = torch.exp(log_confidence.clamp(min=-60.0, max=0.0))
        sigma2_used = residual_energy.new_full(residual_energy.shape, float(sigma2_used_scalar))
        return {
            "confidence": confidence.clamp(0.0, 1.0),
            "log_confidence": log_confidence,
            "residual_energy": residual_energy,
            "residual_energy_per_dim": residual_energy_per_dim,
            "sigma2_eff": sigma2_eff,
            "sigma2_used": sigma2_used,
            "residual": residual,
        }

    def _compute_tangent_support(
        self,
        *,
        class_center: torch.Tensor,
        same_proto: torch.Tensor,
        sub_offsets: torch.Tensor | None = None,
    ) -> Dict[str, object]:
        if self.tangent_source == "subproto_offsets" and sub_offsets is not None:
            deviations = sub_offsets.to(dtype=same_proto.dtype, device=same_proto.device)
        else:
            deviations = same_proto - class_center.unsqueeze(0)
        deviations = deviations - torch.sum(deviations * class_center.unsqueeze(0), dim=-1, keepdim=True) * class_center.unsqueeze(0)
        deviations = torch.nan_to_num(deviations, nan=0.0, posinf=0.0, neginf=0.0)
        if deviations.shape[0] == 0 or float(torch.linalg.vector_norm(deviations).item()) <= 0.0:
            basis = deviations[:0]
            singular_values = deviations.new_zeros((0,))
        else:
            try:
                _, singular_values, vh = torch.linalg.svd(deviations, full_matrices=False)
            except RuntimeError:
                deviations_cpu = deviations.detach().cpu().to(dtype=torch.float64)
                _, singular_values_cpu, vh_cpu = torch.linalg.svd(deviations_cpu, full_matrices=False)
                singular_values = singular_values_cpu.to(dtype=deviations.dtype, device=deviations.device)
                vh = vh_cpu.to(dtype=deviations.dtype, device=deviations.device)
            max_rank = int(min(self.tangent_rank, vh.shape[0]))
            basis = vh[:max_rank]
            if basis.numel() > 0:
                basis = F.normalize(basis, p=2, dim=-1, eps=self.input_norm_eps)
        if singular_values.numel() > 0:
            energy = singular_values.square()
            energy_probs = energy / energy.sum().clamp_min(1e-12)
            cumulative_energy = torch.cumsum(energy_probs, dim=0)
            rank95 = int(torch.nonzero(cumulative_energy >= 0.95, as_tuple=False)[0, 0].item() + 1)
            entropy = -(energy_probs * energy_probs.clamp_min(1e-12).log()).sum()
            effective_rank = float(torch.exp(entropy).item())
            top1_energy_ratio = float(energy_probs[0].item())
            if energy_probs.numel() > 1:
                top1_top2_gap = float((energy_probs[0] - energy_probs[1]).item())
            else:
                top1_top2_gap = float(top1_energy_ratio)
        else:
            energy_probs = deviations.new_zeros((0,))
            cumulative_energy = deviations.new_zeros((0,))
            rank95 = 0
            effective_rank = 0.0
            top1_energy_ratio = 0.0
            top1_top2_gap = 0.0
        support = torch.cat([class_center.unsqueeze(0), basis], dim=0)
        return {
            "support": support,
            "basis": basis,
            "actual_rank": int(basis.shape[0]),
            "rank95": int(rank95),
            "effective_rank": float(effective_rank),
            "top1_energy_ratio": float(top1_energy_ratio),
            "top1_top2_spectral_gap": float(top1_top2_gap),
            "singular_values": singular_values.detach().cpu().tolist(),
            "energy_probs": energy_probs.detach().cpu().tolist(),
            "cumulative_energy": cumulative_energy.detach().cpu().tolist(),
        }

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
                cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
                try:
                    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
                except RuntimeError:
                    eigvals = torch.linalg.eigvalsh(
                        cov.detach().cpu().to(dtype=torch.float64)
                    ).to(dtype=cov.dtype, device=cov.device).clamp_min(0.0)
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
        h: torch.Tensor,
        gram: torch.Tensor | None,
        ridge_scale: torch.Tensor,
        solved: torch.Tensor,
        class_idx: int,
        committee_idx: int,
    ) -> None:
        if not self.enable_probe:
            return
        with torch.no_grad():
            batch_size = h.shape[0]
            mean_trace_cpu = [0.0] * batch_size
            cond_cpu = [0.0] * batch_size

            if gram is not None:
                dim = float(gram.shape[-1])
                mean_trace = torch.einsum("bii->b", gram) / dim
                mean_trace_cpu = mean_trace.detach().cpu().tolist()
                
                # Skip expensive eigenvalue decomposition for high-dimensional features
                if int(gram.shape[-1]) <= 1024:
                    # abs() helps with numerical stability for condition number
                    eig_abs = torch.linalg.eigvalsh(gram).abs()
                    cond_eps = max(self.ridge_trace_eps, 1e-12)
                    cond_number = eig_abs.max(dim=-1).values / eig_abs.min(dim=-1).values.clamp_min(cond_eps)
                    cond_cpu = cond_number.detach().cpu().tolist()
                else:
                    cond_cpu = [-1.0] * batch_size

            weight_norm = torch.linalg.vector_norm(solved, dim=-1)
            ridge_cpu = ridge_scale.detach().cpu().tolist()
            wnorm_cpu = weight_norm.detach().cpu().tolist()

            for sample_idx, (trace_i, cond_i, ridge_i, wnorm_i) in enumerate(
                zip(mean_trace_cpu, cond_cpu, ridge_cpu, wnorm_cpu)
            ):
                self._probe_rows.append(
                    {
                        "split": str(self._probe_split),
                        "epoch": int(self._probe_epoch),
                        "class_idx": int(class_idx),
                        "committee_idx": int(committee_idx),
                        "sample_idx_in_batch": int(sample_idx),
                        "ridge_mode": str(self.ridge_mode),
                        "solve_mode": str(self.solve_mode),
                        "support_mode": str(self.support_mode),
                        "prototype_aggregation": str(self.prototype_aggregation),
                        "input_norm_mode": str(self.input_norm_mode),
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

        if self.solve_mode == "pinv":
            gram = torch.einsum("bn,nd,ne->bde", weights, support, support)
            rhs = torch.einsum("bn,nd,n->bd", weights, support, targets)
            ridge_scale = torch.zeros(
                (h.shape[0],),
                dtype=h.dtype,
                device=h.device,
            )
            gram_safe = torch.nan_to_num(gram, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                gram_pinv = torch.linalg.pinv(gram_safe, rcond=self.pinv_rcond, hermitian=True)
            except RuntimeError:
                try:
                    gram_pinv = torch.linalg.pinv(gram_safe, rcond=self.pinv_rcond, hermitian=False)
                except RuntimeError:
                    gram_pinv = torch.linalg.pinv(
                        gram_safe.detach().cpu().to(dtype=torch.float64),
                        rcond=self.pinv_rcond,
                        hermitian=False,
                    ).to(dtype=h.dtype, device=h.device)
            solved = torch.matmul(
                gram_pinv,
                rhs.unsqueeze(-1),
            ).squeeze(-1)
            solve_matrix_shape = list(gram.shape)
            solve_space_dim = int(gram.shape[-1])
        elif self.solve_mode == "dual_pinv":
            gram = None 
            rhs = None
            ridge_scale = torch.zeros(
                (h.shape[0],),
                dtype=h.dtype,
                device=h.device,
            )
            weight_sqrt = torch.sqrt(weights.clamp_min(0.0))
            weighted_support = weight_sqrt[:, :, None] * support.unsqueeze(0)
            weighted_targets = weight_sqrt * targets.unsqueeze(0)
            
            dual_gram = torch.einsum("bnd,bmd->bnm", weighted_support, weighted_support)
            dual_gram_safe = torch.nan_to_num(dual_gram, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                dual_gram_pinv = torch.linalg.pinv(dual_gram_safe, rcond=self.pinv_rcond, hermitian=True)
            except RuntimeError:
                try:
                    dual_gram_pinv = torch.linalg.pinv(dual_gram_safe, rcond=self.pinv_rcond, hermitian=False)
                except RuntimeError:
                    dual_gram_pinv = torch.linalg.pinv(
                        dual_gram_safe.detach().cpu().to(dtype=torch.float64),
                        rcond=self.pinv_rcond,
                        hermitian=False,
                    ).to(dtype=h.dtype, device=h.device)
            dual_solution = torch.matmul(
                dual_gram_pinv,
                weighted_targets.unsqueeze(-1),
            ).squeeze(-1)
            solved = torch.einsum("bnd,bn->bd", weighted_support, dual_solution)
            solve_matrix_shape = list(dual_gram.shape)
            solve_space_dim = int(dual_gram.shape[-1])
        else:
            # Handle ridge modes
            if self.solve_mode == "dual_ridge":
                gram = None
                rhs = None
            else:
                gram = torch.einsum("bn,nd,ne->bde", weights, support, support)
                rhs = torch.einsum("bn,nd,n->bd", weights, support, targets)

            if gram is not None and self.ridge_mode == "trace_adaptive":
                dim = float(gram.shape[-1])
                mean_trace = torch.einsum("bii->b", gram) / dim
                ridge_scale = self.ridge * mean_trace.clamp_min(self.ridge_trace_eps)
            else:
                ridge_scale = torch.full(
                    (h.shape[0],),
                    float(self.ridge),
                    dtype=h.dtype,
                    device=h.device,
                )

            if self.solve_mode == "dual_ridge":
                weight_sqrt = torch.sqrt(weights.clamp_min(0.0))
                weighted_support = weight_sqrt[:, :, None] * support.unsqueeze(0)
                weighted_targets = weight_sqrt * targets.unsqueeze(0)
                
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
                eye = torch.eye(gram.shape[-1], dtype=h.dtype, device=h.device)
                solved = torch.linalg.solve(
                    gram + ridge_scale[:, None, None] * eye.unsqueeze(0),
                    rhs.unsqueeze(-1),
                ).squeeze(-1)
                solve_matrix_shape = list(gram.shape)
                solve_space_dim = int(gram.shape[-1])

        self._record_probe_rows(
            h=h,
            gram=gram,
            ridge_scale=ridge_scale,
            solved=torch.nan_to_num(solved, nan=0.0, posinf=0.0, neginf=0.0),
            class_idx=int(class_idx),
            committee_idx=int(committee_idx),
        )
        solved = torch.nan_to_num(solved, nan=0.0, posinf=0.0, neginf=0.0)
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
                "gram_shape": [int(v) for v in gram.shape] if gram is not None else [-1, -1, -1],
                "rhs_shape": [int(v) for v in rhs.shape] if rhs is not None else [-1, -1],
                "solve_matrix_shape": [int(v) for v in solve_matrix_shape],
                "solve_space_dim": int(solve_space_dim),
                "solved_shape": [int(v) for v in solved.shape],
                "same_weight_max_mean": float(same_w.max(dim=-1).values.mean().item()),
                "same_weight_entropy_mean": float(self._normalized_entropy(same_w).mean().item()),
                "same_weight_sum_mean": float(same_w.sum(dim=-1).mean().item()),
                "gram_trace_mean": float(torch.einsum("bii->b", gram).mean().item()) if gram is not None else -1.0,
                "rhs_norm_mean": float(torch.linalg.vector_norm(rhs, dim=-1).mean().item()) if rhs is not None else -1.0,
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
        class_routings: List[torch.Tensor] = []
        class_cosines: List[torch.Tensor] = []
        class_centers_shape: List[int] | None = None
        sub_prototypes_shape: List[int] | None = None
        class_prior_shape: List[int] | None = None
        tangent_basis_shape: List[int] | None = None
        prob_selected_rank_values: List[torch.Tensor] = []
        prob_shrinkage_alpha_values: List[torch.Tensor] = []
        prob_sigma2_values: List[torch.Tensor] = []
        prob_posterior_confidence_values: List[torch.Tensor] = []
        prob_posterior_log_confidence_values: List[torch.Tensor] = []
        prob_posterior_residual_energy_values: List[torch.Tensor] = []
        prob_posterior_residual_energy_per_dim_values: List[torch.Tensor] = []
        prob_posterior_sigma2_eff_values: List[torch.Tensor] = []
        prob_posterior_sigma2_used_values: List[torch.Tensor] = []
        prob_trace_per_dim_values: List[torch.Tensor] = []
        prob_rank0_score_values: List[torch.Tensor] = []
        prob_rank1_score_values: List[torch.Tensor] = []
        prob_rank01_relative_gap_values: List[torch.Tensor] = []
        prob_zero_rank_rescued_values: List[torch.Tensor] = []
        prob_abs_gate_values: List[torch.Tensor] = []
        prob_relative_top1_weight_values: List[torch.Tensor] = []
        prob_relative_top1_entropy_values: List[torch.Tensor] = []
        prob_relative_margin_values: List[torch.Tensor] = []
        prob_relative_active_values: List[torch.Tensor] = []
        prob_weighted_delta_norm_values: List[torch.Tensor] = []
        prob_solver_top1_index_values: List[torch.Tensor] = []

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
                class_routings.append(same_routing)
                class_cosines.append(same_cosine)
                class_logits.append(class_logit)
        elif self.prototype_geometry_mode == "center_tangent":
            class_centers = self._get_class_centers().to(dtype=h.dtype, device=h.device)
            sub_prototypes = self._get_sub_prototypes(class_centers).to(dtype=h.dtype, device=h.device)
            class_prior = torch.softmax(
                self._cosine_similarity(h, class_centers) / self.class_prior_temperature,
                dim=-1,
            )
            class_centers_shape = [int(v) for v in class_centers.shape]
            sub_prototypes_shape = [int(v) for v in sub_prototypes.shape]
            class_prior_shape = [int(v) for v in class_prior.shape]
            for class_idx in range(self.num_classes):
                tangent_payload = self._compute_tangent_support(
                    class_center=class_centers[class_idx],
                    same_proto=sub_prototypes[class_idx],
                    sub_offsets=None if self.sub_offsets is None else self.sub_offsets[class_idx],
                )
                same_proto = tangent_payload["support"].to(dtype=h.dtype, device=h.device)
                same_cosine = self._cosine_similarity(h, same_proto)
                same_routing = torch.softmax(same_cosine / self.routing_temperature, dim=-1)
                tangent_basis_shape = [int(v) for v in tangent_payload["basis"].shape]
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
                        "sub_prototypes_shape": [int(v) for v in sub_prototypes.shape],
                        "tangent_basis_shape": [int(v) for v in tangent_payload["basis"].shape],
                        "class_prior_max_mean": float(class_prior.max(dim=-1).values.mean().item()),
                        "class_prior_entropy_mean": float(self._normalized_entropy(class_prior).mean().item()),
                        "class_prior_temperature": float(self.class_prior_temperature),
                        "subproto_temperature": float(self.subproto_temperature),
                        "subproto_weight_max_mean": float(same_routing.max(dim=-1).values.mean().item()),
                        "subproto_weight_entropy_mean": float(self._normalized_entropy(same_routing).mean().item()),
                        "same_support_norm_mean": float(torch.linalg.vector_norm(same_proto, dim=-1).mean().item()),
                        "opp_center_weight_entropy_mean": 0.0,
                        "tangent_rank": int(tangent_payload["actual_rank"]),
                        "tangent_requested_rank": int(self.tangent_rank),
                        "tangent_source": str(self.tangent_source),
                        "tangent_rank95": int(tangent_payload["rank95"]),
                        "tangent_effective_rank": float(tangent_payload["effective_rank"]),
                        "tangent_top1_energy_ratio": float(tangent_payload["top1_energy_ratio"]),
                        "tangent_top1_top2_spectral_gap": float(tangent_payload["top1_top2_spectral_gap"]),
                        "center_to_subproto_cos_mean": 1.0,
                        "center_to_subproto_cos_std": 0.0,
                        "subproto_pairwise_cos_mean": 0.0,
                        "subproto_pairwise_cos_std": 0.0,
                        "subproto_nn_cos_mean": 0.0,
                        "subproto_cov_trace": 0.0,
                        "subproto_cov_top_eig": 0.0,
                        "subproto_cov_effective_rank": float(tangent_payload["effective_rank"]),
                        "subproto_cos_top1_top2_gap_mean": float(
                            (torch.topk(same_cosine, k=2, dim=-1).values[:, 0] - torch.topk(same_cosine, k=2, dim=-1).values[:, 1]).mean().item()
                        ) if same_cosine.shape[-1] > 1 else 0.0,
                        "subproto_cos_max_minus_mean_mean": float(
                            (same_cosine.max(dim=-1).values - same_cosine.mean(dim=-1)).mean().item()
                        ),
                        "subproto_cos_var_mean": float(same_cosine.var(dim=-1, unbiased=False).mean().item()),
                        "subproto_cos_top1_mean": float(same_cosine.max(dim=-1).values.mean().item()),
                    },
                )
                class_logit = torch.sum(h * solved, dim=-1)
                diag["aggregation_mode"] = "center_tangent_support"
                class_summaries.append(diag)
                class_routings.append(same_routing)
                class_cosines.append(same_cosine)
                class_logits.append(class_logit)
        elif self.prototype_geometry_mode == "center_prob_tangent":
            class_centers = self._get_class_centers().to(dtype=h.dtype, device=h.device)
            sub_prototypes = self._get_sub_prototypes(class_centers).to(dtype=h.dtype, device=h.device)
            class_prior = torch.softmax(
                self._cosine_similarity(h, class_centers) / self.class_prior_temperature,
                dim=-1,
            )
            class_centers_shape = [int(v) for v in class_centers.shape]
            sub_prototypes_shape = [int(v) for v in sub_prototypes.shape]
            class_prior_shape = [int(v) for v in class_prior.shape]
            for class_idx in range(self.num_classes):
                prob_payload = self._compute_prob_tangent_support(
                    class_center=class_centers[class_idx],
                    same_proto=sub_prototypes[class_idx],
                    sub_offsets=None if self.sub_offsets is None else self.sub_offsets[class_idx],
                )
                same_proto = prob_payload["support"].to(dtype=h.dtype, device=h.device)
                same_cosine = self._cosine_similarity(h, same_proto)
                same_routing = torch.softmax(same_cosine / self.routing_temperature, dim=-1)
                posterior_confidence = torch.ones(
                    (batch_size,),
                    dtype=h.dtype,
                    device=h.device,
                )
                if self.prob_tangent_version == "v3":
                    posterior_payload = self._compute_prob_tangent_posterior(
                        h=h,
                        class_center=class_centers[class_idx],
                        basis=prob_payload["basis"].to(dtype=h.dtype, device=h.device),
                        selected_eigvals=prob_payload["selected_eigvals_tensor"].to(dtype=h.dtype, device=h.device),
                        sigma2=float(prob_payload["sigma2"]),
                        trace_per_dim=float(prob_payload["trace_per_dim"]),
                    )
                    posterior_confidence = posterior_payload["confidence"].to(dtype=h.dtype, device=h.device)
                    posterior_log_confidence = posterior_payload["log_confidence"].to(dtype=h.dtype, device=h.device)
                    posterior_residual_energy = posterior_payload["residual_energy"].to(dtype=h.dtype, device=h.device)
                    posterior_residual_energy_per_dim = posterior_payload["residual_energy_per_dim"].to(dtype=h.dtype, device=h.device)
                    posterior_sigma2_eff = posterior_payload["sigma2_eff"].to(dtype=h.dtype, device=h.device)
                    posterior_sigma2_used = posterior_payload["sigma2_used"].to(dtype=h.dtype, device=h.device)
                    if self.local_solver_competition_mode == "none" and same_routing.shape[-1] > 1:
                        tangent_weights = same_routing[:, 1:] * posterior_confidence.unsqueeze(-1)
                        collapsed_mass = same_routing[:, 1:].sum(dim=-1, keepdim=True) - tangent_weights.sum(dim=-1, keepdim=True)
                        same_routing = torch.cat([same_routing[:, :1] + collapsed_mass, tangent_weights], dim=-1)
                    posterior_conf_stats = self._posterior_stat_summary(posterior_confidence)
                    posterior_log_conf_stats = self._posterior_stat_summary(posterior_log_confidence)
                    posterior_q_gap = float(posterior_conf_stats["q90"] - posterior_conf_stats["q10"])
                    posterior_q_ratio = float(
                        posterior_conf_stats["q90"] / max(posterior_conf_stats["q10"], 1e-12)
                    )
                else:
                    posterior_log_confidence = torch.zeros_like(posterior_confidence)
                    posterior_residual_energy = torch.zeros_like(posterior_confidence)
                    posterior_residual_energy_per_dim = torch.zeros_like(posterior_confidence)
                    posterior_sigma2_eff = torch.ones_like(posterior_confidence)
                    posterior_sigma2_used = torch.full_like(posterior_confidence, float(prob_payload["sigma2"]))
                    posterior_conf_stats = {"q10": 1.0, "q50": 1.0, "q90": 1.0, "mean": 1.0, "std": 0.0}
                    posterior_log_conf_stats = {"q10": 0.0, "q50": 0.0, "q90": 0.0, "mean": 0.0, "std": 0.0}
                    posterior_q_gap = 0.0
                    posterior_q_ratio = 1.0
                tangent_basis_shape = [int(v) for v in prob_payload["basis"].shape]
                common_diag = {
                    "class_prior_shape": [int(v) for v in class_prior.shape],
                    "subproto_routing_shape": [int(v) for v in same_routing.shape],
                    "class_centers_shape": [int(v) for v in class_centers.shape],
                    "sub_prototypes_shape": [int(v) for v in sub_prototypes.shape],
                    "tangent_basis_shape": [int(v) for v in prob_payload["basis"].shape],
                    "class_prior_max_mean": float(class_prior.max(dim=-1).values.mean().item()),
                    "class_prior_entropy_mean": float(self._normalized_entropy(class_prior).mean().item()),
                    "class_prior_temperature": float(self.class_prior_temperature),
                    "subproto_temperature": float(self.subproto_temperature),
                    "subproto_weight_max_mean": float(same_routing.max(dim=-1).values.mean().item()),
                    "subproto_weight_entropy_mean": float(self._normalized_entropy(same_routing).mean().item()),
                    "same_support_norm_mean": float(torch.linalg.vector_norm(same_proto, dim=-1).mean().item()),
                    "opp_center_weight_entropy_mean": 0.0,
                    "prob_tangent_version": str(self.prob_tangent_version),
                    "rank_selection_mode": str(self.rank_selection_mode),
                    "posterior_mode": str(self.posterior_mode),
                    "posterior_student_dof": float(self.posterior_student_dof),
                    "mdl_penalty_beta": float(self.mdl_penalty_beta),
                    "gaussian_refine_variant": str(self.gaussian_refine_variant),
                    "mdl_zero_rank_rescue_margin": float(self.mdl_zero_rank_rescue_margin),
                    "selected_rank": int(prob_payload["selected_rank"]),
                    "lw_shrinkage_alpha": float(prob_payload["shrinkage_alpha"]),
                    "ppca_sigma2": float(prob_payload["sigma2"]),
                    "trace_per_dim": float(prob_payload["trace_per_dim"]),
                    "rank0_score": prob_payload["rank0_score"],
                    "rank1_score": prob_payload["rank1_score"],
                    "rank01_relative_gap": prob_payload["rank01_relative_gap"],
                    "zero_rank_rescued": int(bool(prob_payload["zero_rank_rescued"])),
                    "posterior_confidence_mean": float(posterior_confidence.mean().item()),
                    "posterior_log_confidence_mean": float(posterior_log_conf_stats["mean"]),
                    "posterior_log_confidence_std": float(posterior_log_conf_stats["std"]),
                    "posterior_confidence_std": float(posterior_conf_stats["std"]),
                    "posterior_confidence_q10": float(posterior_conf_stats["q10"]),
                    "posterior_confidence_q50": float(posterior_conf_stats["q50"]),
                    "posterior_confidence_q90": float(posterior_conf_stats["q90"]),
                    "posterior_confidence_qgap": float(posterior_q_gap),
                    "posterior_confidence_qratio": float(posterior_q_ratio),
                    "posterior_residual_energy_mean": float(posterior_residual_energy.mean().item()),
                    "posterior_residual_energy_per_dim_mean": float(posterior_residual_energy_per_dim.mean().item()),
                    "posterior_sigma2_eff_mean": float(posterior_sigma2_eff.mean().item()),
                    "posterior_sigma2_used_mean": float(posterior_sigma2_used.mean().item()),
                    "k0_fallback": int(int(prob_payload["selected_rank"]) == 0),
                    "tangent_requested_rank": int(min(4, self.prototypes_per_class)),
                    "tangent_rank95": int(prob_payload["rank95"]),
                    "tangent_effective_rank": float(prob_payload["effective_rank"]),
                    "tangent_source": str(self.tangent_source),
                    "rank_rows": [dict(row) for row in prob_payload["rank_rows"]],
                    "center_to_subproto_cos_mean": 1.0,
                    "center_to_subproto_cos_std": 0.0,
                    "subproto_pairwise_cos_mean": 0.0,
                    "subproto_pairwise_cos_std": 0.0,
                    "subproto_nn_cos_mean": 0.0,
                    "subproto_cov_trace": 0.0,
                    "subproto_cov_top_eig": 0.0,
                    "subproto_cov_effective_rank": float(prob_payload["effective_rank"]),
                    "subproto_cos_top1_top2_gap_mean": float(
                        (torch.topk(same_cosine, k=2, dim=-1).values[:, 0] - torch.topk(same_cosine, k=2, dim=-1).values[:, 1]).mean().item()
                    ) if same_cosine.shape[-1] > 1 else 0.0,
                    "subproto_cos_max_minus_mean_mean": float(
                        (same_cosine.max(dim=-1).values - same_cosine.mean(dim=-1)).mean().item()
                    ),
                    "subproto_cos_var_mean": float(same_cosine.var(dim=-1, unbiased=False).mean().item()),
                    "subproto_cos_top1_mean": float(same_cosine.max(dim=-1).values.mean().item()),
                    "local_solver_competition_mode": str(self.local_solver_competition_mode),
                    "relative_solver_temperature": float(self.relative_solver_temperature),
                    "abs_gate_activity_floor": float(self.abs_gate_activity_floor),
                    "abs_gate_mean": float(posterior_confidence.mean().item()),
                    "abs_gate_q10": float(posterior_conf_stats["q10"]),
                    "abs_gate_q50": float(posterior_conf_stats["q50"]),
                    "abs_gate_q90": float(posterior_conf_stats["q90"]),
                }
                if self.local_solver_competition_mode == "relcomp":
                    solver_solved_bank: List[torch.Tensor] = []
                    solver_delta_bank: List[torch.Tensor] = []
                    solver_member_diags: List[Dict[str, object]] = []

                    solved_full, diag_full = self._solve_direction(
                        h,
                        same_proto=same_proto,
                        opp_proto=class_centers[:0],
                        eye=eye,
                        class_idx=class_idx,
                        committee_idx=0,
                        same_weights_override=same_routing,
                        opp_weights_override=None,
                        extra_diag={"solver_variant": "full", **common_diag},
                    )
                    solver_solved_bank.append(solved_full)
                    solver_delta_bank.append(torch.sum(h * solved_full, dim=-1))
                    solver_member_diags.append(diag_full)

                    tangent_count = int(max(0, same_proto.shape[0] - 1))
                    for tangent_idx in range(tangent_count):
                        if same_proto.shape[0] <= 2:
                            break
                        pair_support = torch.stack(
                            [same_proto[0], same_proto[tangent_idx + 1]],
                            dim=0,
                        )
                        pair_cosine = self._cosine_similarity(h, pair_support)
                        pair_routing = torch.softmax(pair_cosine / self.routing_temperature, dim=-1)
                        solved_pair, diag_pair = self._solve_direction(
                            h,
                            same_proto=pair_support,
                            opp_proto=class_centers[:0],
                            eye=eye,
                            class_idx=class_idx,
                            committee_idx=int(tangent_idx + 1),
                            same_weights_override=pair_routing,
                            opp_weights_override=None,
                            extra_diag={
                                "solver_variant": f"pair_{int(tangent_idx)}",
                                "solver_pair_tangent_index": int(tangent_idx),
                                **common_diag,
                            },
                        )
                        solver_solved_bank.append(solved_pair)
                        solver_delta_bank.append(torch.sum(h * solved_pair, dim=-1))
                        solver_member_diags.append(diag_pair)

                    solved_bank = torch.stack(solver_solved_bank, dim=1)
                    delta_bank = torch.stack(solver_delta_bank, dim=1)
                    competition = self._compose_relative_solver_competition(
                        solved_bank=solved_bank,
                        delta_bank=delta_bank,
                        abs_gate=posterior_confidence,
                    )
                    class_logit = competition["delta_final"]
                    diag = dict(diag_full)
                    diag.update(common_diag)
                    diag.update(
                        {
                            "aggregation_mode": "center_prob_tangent_relcomp_solver_bank",
                            "solver_bank_size": int(competition["solver_count"]),
                            "relative_top1_weight_mean": float(torch.as_tensor(competition["top1_weight"]).mean().item()),
                            "relative_top1_weight_entropy": float(torch.as_tensor(competition["entropy"]).mean().item()),
                            "relative_solver_margin_mean": float(torch.as_tensor(competition["margin"]).mean().item()),
                            "relative_competition_active_rate": float(competition["active_rate"]),
                            "local_solver_weighted_delta_norm_mean": float(torch.as_tensor(competition["solved_mix_norm"]).mean().item()),
                            "solver_top1_occupancy_entropy": float(competition["occupancy_entropy"]),
                            "solver_bank_members": solver_member_diags,
                        }
                    )
                    relative_top1_weight = torch.as_tensor(competition["top1_weight"]).to(dtype=h.dtype, device=h.device)
                    relative_entropy = torch.as_tensor(competition["entropy"]).to(dtype=h.dtype, device=h.device)
                    relative_margin = torch.as_tensor(competition["margin"]).to(dtype=h.dtype, device=h.device)
                    relative_active = torch.as_tensor(competition["active_mask"]).to(dtype=h.dtype, device=h.device)
                    weighted_delta_norm = torch.as_tensor(competition["solved_mix_norm"]).to(dtype=h.dtype, device=h.device)
                    solver_top1_index = torch.as_tensor(competition["top1_index"]).to(dtype=torch.int64, device=h.device)
                else:
                    solved, diag = self._solve_direction(
                        h,
                        same_proto=same_proto,
                        opp_proto=class_centers[:0],
                        eye=eye,
                        class_idx=class_idx,
                        same_weights_override=same_routing,
                        opp_weights_override=None,
                        extra_diag={"solver_variant": "full", **common_diag},
                    )
                    class_logit = torch.sum(h * solved, dim=-1)
                    diag.update(
                        {
                            "aggregation_mode": "center_prob_tangent_support",
                            "solver_bank_size": 1,
                            "relative_top1_weight_mean": 0.0,
                            "relative_top1_weight_entropy": 0.0,
                            "relative_solver_margin_mean": 0.0,
                            "relative_competition_active_rate": 0.0,
                            "local_solver_weighted_delta_norm_mean": float(torch.linalg.vector_norm(solved, dim=-1).mean().item()),
                            "solver_top1_occupancy_entropy": 0.0,
                        }
                    )
                    relative_top1_weight = torch.zeros((batch_size,), dtype=h.dtype, device=h.device)
                    relative_entropy = torch.zeros((batch_size,), dtype=h.dtype, device=h.device)
                    relative_margin = torch.zeros((batch_size,), dtype=h.dtype, device=h.device)
                    relative_active = torch.zeros((batch_size,), dtype=h.dtype, device=h.device)
                    weighted_delta_norm = torch.linalg.vector_norm(solved, dim=-1)
                    solver_top1_index = torch.zeros((batch_size,), dtype=torch.int64, device=h.device)
                class_summaries.append(diag)
                class_routings.append(same_routing)
                class_cosines.append(same_cosine)
                class_logits.append(class_logit)
                prob_selected_rank_values.append(
                    torch.full((batch_size,), float(prob_payload["selected_rank"]), dtype=h.dtype, device=h.device)
                )
                prob_shrinkage_alpha_values.append(
                    torch.full((batch_size,), float(prob_payload["shrinkage_alpha"]), dtype=h.dtype, device=h.device)
                )
                prob_sigma2_values.append(
                    torch.full((batch_size,), float(prob_payload["sigma2"]), dtype=h.dtype, device=h.device)
                )
                prob_trace_per_dim_values.append(
                    torch.full((batch_size,), float(prob_payload["trace_per_dim"]), dtype=h.dtype, device=h.device)
                )
                prob_rank0_score_values.append(
                    torch.full(
                        (batch_size,),
                        float(0.0 if prob_payload["rank0_score"] is None else prob_payload["rank0_score"]),
                        dtype=h.dtype,
                        device=h.device,
                    )
                )
                prob_rank1_score_values.append(
                    torch.full(
                        (batch_size,),
                        float(0.0 if prob_payload["rank1_score"] is None else prob_payload["rank1_score"]),
                        dtype=h.dtype,
                        device=h.device,
                    )
                )
                prob_rank01_relative_gap_values.append(
                    torch.full(
                        (batch_size,),
                        float(0.0 if prob_payload["rank01_relative_gap"] is None else prob_payload["rank01_relative_gap"]),
                        dtype=h.dtype,
                        device=h.device,
                    )
                )
                prob_zero_rank_rescued_values.append(
                    torch.full((batch_size,), float(bool(prob_payload["zero_rank_rescued"])), dtype=h.dtype, device=h.device)
                )
                prob_abs_gate_values.append(posterior_confidence)
                prob_relative_top1_weight_values.append(relative_top1_weight)
                prob_relative_top1_entropy_values.append(relative_entropy)
                prob_relative_margin_values.append(relative_margin)
                prob_relative_active_values.append(relative_active)
                prob_weighted_delta_norm_values.append(weighted_delta_norm)
                prob_solver_top1_index_values.append(solver_top1_index)
                if self.prob_tangent_version == "v3":
                    prob_posterior_confidence_values.append(posterior_confidence)
                    prob_posterior_log_confidence_values.append(posterior_log_confidence)
                    prob_posterior_residual_energy_values.append(posterior_residual_energy)
                    prob_posterior_residual_energy_per_dim_values.append(posterior_residual_energy_per_dim)
                    prob_posterior_sigma2_eff_values.append(posterior_sigma2_eff)
                    prob_posterior_sigma2_used_values.append(posterior_sigma2_used)
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
                class_routings.append(same_routing)
                class_cosines.append(same_cosine)
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
        self._last_batch_routing_payload = None
        if class_routings:
            max_width = max(int(t.shape[-1]) for t in class_routings)
            padded_routings: List[torch.Tensor] = []
            padded_cosines: List[torch.Tensor] = []
            for routing, cosine in zip(class_routings, class_cosines):
                pad_width = int(max_width - routing.shape[-1])
                if pad_width > 0:
                    routing = F.pad(routing, (0, pad_width), value=0.0)
                    cosine = F.pad(cosine, (0, pad_width), value=float("-inf"))
                padded_routings.append(routing)
                padded_cosines.append(cosine)
            all_routings = torch.stack(padded_routings, dim=1)  # [B, C, M]
            all_cosines = torch.stack(padded_cosines, dim=1)  # [B, C, M]
            local_pred = local_logits.argmax(dim=-1)
            batch_indices = torch.arange(batch_size, device=h.device)
            selected_routing = all_routings[batch_indices, local_pred]
            selected_cosines = all_cosines[batch_indices, local_pred]
            selected_top1_weight, selected_top1_index = selected_routing.max(dim=-1)
            if selected_cosines.shape[-1] > 1:
                top2 = torch.topk(selected_cosines, k=2, dim=-1).values
                selected_cos_gap = top2[:, 0] - top2[:, 1]
            else:
                selected_cos_gap = torch.zeros(
                    (batch_size,),
                    dtype=selected_cosines.dtype,
                    device=selected_cosines.device,
                )
            selected_cos_gap = torch.nan_to_num(selected_cos_gap, nan=0.0, posinf=0.0, neginf=0.0)
            selected_entropy = self._normalized_entropy(selected_routing)
            payload = {
                "routing_width": int(selected_routing.shape[-1]),
                "local_pred_class": local_pred.detach().cpu().tolist(),
                "top1_index": selected_top1_index.detach().cpu().tolist(),
                "top1_weight": selected_top1_weight.detach().cpu().tolist(),
                "routing_entropy": selected_entropy.detach().cpu().tolist(),
                "cos_top1_top2_gap": selected_cos_gap.detach().cpu().tolist(),
            }
            if prob_selected_rank_values:
                selected_rank_stack = torch.stack(prob_selected_rank_values, dim=1)
                shrinkage_stack = torch.stack(prob_shrinkage_alpha_values, dim=1)
                sigma2_stack = torch.stack(prob_sigma2_values, dim=1)
                payload.update(
                    {
                        "selected_rank": selected_rank_stack[batch_indices, local_pred].detach().cpu().tolist(),
                        "lw_shrinkage_alpha": shrinkage_stack[batch_indices, local_pred].detach().cpu().tolist(),
                        "ppca_sigma2": sigma2_stack[batch_indices, local_pred].detach().cpu().tolist(),
                        "trace_per_dim": torch.stack(prob_trace_per_dim_values, dim=1)[batch_indices, local_pred].detach().cpu().tolist(),
                        "rank0_score": torch.stack(prob_rank0_score_values, dim=1)[batch_indices, local_pred].detach().cpu().tolist(),
                        "rank1_score": torch.stack(prob_rank1_score_values, dim=1)[batch_indices, local_pred].detach().cpu().tolist(),
                        "rank01_relative_gap": torch.stack(prob_rank01_relative_gap_values, dim=1)[batch_indices, local_pred].detach().cpu().tolist(),
                        "zero_rank_rescued": torch.stack(prob_zero_rank_rescued_values, dim=1)[batch_indices, local_pred].detach().cpu().tolist(),
                    }
                )
                if prob_posterior_confidence_values:
                    posterior_stack = torch.stack(prob_posterior_confidence_values, dim=1)
                    payload["posterior_confidence"] = posterior_stack[batch_indices, local_pred].detach().cpu().tolist()
                    log_conf_stack = torch.stack(prob_posterior_log_confidence_values, dim=1)
                    residual_energy_stack = torch.stack(prob_posterior_residual_energy_values, dim=1)
                    residual_energy_per_dim_stack = torch.stack(prob_posterior_residual_energy_per_dim_values, dim=1)
                    sigma2_eff_stack = torch.stack(prob_posterior_sigma2_eff_values, dim=1)
                    sigma2_used_stack = torch.stack(prob_posterior_sigma2_used_values, dim=1)
                    payload["posterior_log_confidence"] = log_conf_stack[batch_indices, local_pred].detach().cpu().tolist()
                    payload["posterior_residual_energy"] = residual_energy_stack[batch_indices, local_pred].detach().cpu().tolist()
                    payload["posterior_residual_energy_per_dim"] = residual_energy_per_dim_stack[batch_indices, local_pred].detach().cpu().tolist()
                    payload["posterior_sigma2_eff"] = sigma2_eff_stack[batch_indices, local_pred].detach().cpu().tolist()
                    payload["posterior_sigma2_used"] = sigma2_used_stack[batch_indices, local_pred].detach().cpu().tolist()
                if prob_abs_gate_values:
                    abs_gate_stack = torch.stack(prob_abs_gate_values, dim=1)
                    relative_top1_weight_stack = torch.stack(prob_relative_top1_weight_values, dim=1)
                    relative_entropy_stack = torch.stack(prob_relative_top1_entropy_values, dim=1)
                    relative_margin_stack = torch.stack(prob_relative_margin_values, dim=1)
                    relative_active_stack = torch.stack(prob_relative_active_values, dim=1)
                    weighted_delta_norm_stack = torch.stack(prob_weighted_delta_norm_values, dim=1)
                    solver_top1_index_stack = torch.stack(prob_solver_top1_index_values, dim=1)
                    payload["abs_gate"] = abs_gate_stack[batch_indices, local_pred].detach().cpu().tolist()
                    payload["relative_top1_weight"] = relative_top1_weight_stack[batch_indices, local_pred].detach().cpu().tolist()
                    payload["relative_top1_weight_entropy"] = relative_entropy_stack[batch_indices, local_pred].detach().cpu().tolist()
                    payload["relative_solver_margin"] = relative_margin_stack[batch_indices, local_pred].detach().cpu().tolist()
                    payload["relative_competition_active"] = relative_active_stack[batch_indices, local_pred].detach().cpu().tolist()
                    payload["local_solver_weighted_delta_norm"] = weighted_delta_norm_stack[batch_indices, local_pred].detach().cpu().tolist()
                    payload["relative_solver_top1_index"] = solver_top1_index_stack[batch_indices, local_pred].detach().cpu().tolist()
            self._last_batch_routing_payload = payload
        self._last_dataflow_summary = {
            "split": str(self._probe_split),
            "epoch": int(self._probe_epoch),
            "latent_shape": [int(v) for v in h.shape],
            "prototype_geometry_mode": self.prototype_geometry_mode,
            "prototypes_shape": None if self.prototypes is None else [int(v) for v in self.prototypes.shape],
            "class_centers_shape": class_centers_shape,
            "sub_prototypes_shape": sub_prototypes_shape,
            "tangent_basis_shape": tangent_basis_shape,
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
            "tangent_rank": int(self.tangent_rank),
            "tangent_source": str(self.tangent_source),
            "prob_tangent_version": str(self.prob_tangent_version),
            "rank_selection_mode": str(self.rank_selection_mode),
            "posterior_mode": str(self.posterior_mode),
            "posterior_student_dof": float(self.posterior_student_dof),
            "mdl_penalty_beta": float(self.mdl_penalty_beta),
            "gaussian_refine_variant": str(self.gaussian_refine_variant),
            "mdl_zero_rank_rescue_margin": float(self.mdl_zero_rank_rescue_margin),
            "local_solver_competition_mode": str(self.local_solver_competition_mode),
            "relative_solver_temperature": float(self.relative_solver_temperature),
            "abs_gate_activity_floor": float(self.abs_gate_activity_floor),
            "solve_mode": self.solve_mode,
            "ridge_mode": self.ridge_mode,
            "input_norm_mode": self.input_norm_mode,
            "class_summaries": class_summaries,
        }
        geometry_summary = self.export_learned_prototype_geometry_summary()
        if geometry_summary is not None:
            self._last_dataflow_summary["learned_prototype_geometry"] = geometry_summary
        tangent_probe = self.export_tangent_probe_payload()
        if tangent_probe is not None:
            self._last_dataflow_summary["tangent_probe_summary"] = tangent_probe["summary"]
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
