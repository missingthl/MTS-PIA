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
        tangent_rank: int = 2,
        tangent_source: str = "subproto_offsets",
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
        if self.ridge_mode not in {"fixed", "trace_adaptive"}:
            raise ValueError(f"unsupported ridge mode: {self.ridge_mode}")
        if self.solve_mode not in {"ridge_solve", "pinv", "dual_ridge", "dual_pinv"}:
            raise ValueError(f"unsupported solve mode: {self.solve_mode}")
        if self.input_norm_mode not in {"none", "l2_hypersphere"}:
            raise ValueError(f"unsupported input norm mode: {self.input_norm_mode}")
        if self.prototype_geometry_mode not in {"flat", "center_subproto", "center_only", "center_tangent"}:
            raise ValueError(f"unsupported prototype geometry mode: {self.prototype_geometry_mode}")
        if self.prototype_geometry_mode in {"center_subproto", "center_only", "center_tangent"} and self.prototype_aggregation != "pooled":
            raise ValueError(f"{self.prototype_geometry_mode} geometry currently supports pooled aggregation only")
        if self.tangent_source not in {"subproto_offsets"}:
            raise ValueError(f"unsupported tangent source: {self.tangent_source}")
        if self.prototype_geometry_mode == "center_tangent" and self.support_mode != "same_only":
            raise ValueError("center_tangent geometry currently supports same_only support mode only")
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
        if self.prototype_geometry_mode not in {"center_subproto", "center_tangent"}:
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
        if self.prototype_geometry_mode not in {"center_subproto", "center_tangent"}:
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
            self._last_batch_routing_payload = {
                "routing_width": int(selected_routing.shape[-1]),
                "local_pred_class": local_pred.detach().cpu().tolist(),
                "top1_index": selected_top1_index.detach().cpu().tolist(),
                "top1_weight": selected_top1_weight.detach().cpu().tolist(),
                "routing_entropy": selected_entropy.detach().cpu().tolist(),
                "cos_top1_top2_gap": selected_cos_gap.detach().cpu().tolist(),
            }
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
