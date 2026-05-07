from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from core.bridge import bridge_single, logvec_to_spd
from core.pia import _build_spectral_structure_basis_from_zpia_bank

from .act_builder import _attach_feedback_scores_to_aug_out, _build_act_realized_augmentations
from .diagnostics import summarize_multitemplate_audit_rows as _summarize_multitemplate_audit_rows
from .direction_banks import build_direction_bank_for_args as _build_direction_bank_for_args
from .materialize import materialize_z_aug_out as _materialize_z_aug_out
from .state import TrialRecord
from .template_slots import build_top_response_template_slots as _build_top_response_template_slots


def _summarize_osf_audit_rows(audit_rows: List[Dict[str, object]]) -> Dict[str, float]:
    if not audit_rows:
        return {
            "osf_structure_overflow_rate": 0.0,
            "osf_alpha_eff_mean": 0.0,
            "osf_risk_scale_mean": 0.0,
            "osf_risk_zero_perp_rate": 0.0,
            "osf_risk_clipped_rate": 0.0,
        }
    df = pd.DataFrame(audit_rows)

    def mean_float(col: str, default: float = 0.0) -> float:
        if col not in df.columns:
            return default
        vals = pd.to_numeric(df[col], errors="coerce")
        return float(vals.mean()) if vals.notna().any() else default

    status = df.get("osf_risk_status")
    return {
        "osf_structure_overflow_rate": mean_float("osf_structure_overflow"),
        "osf_alpha_eff_mean": mean_float("osf_alpha_eff"),
        "osf_risk_scale_mean": mean_float("osf_risk_scale"),
        "osf_risk_zero_perp_rate": float((status == "zero_perp").mean()) if status is not None else 0.0,
        "osf_risk_clipped_rate": float((status == "clipped").mean()) if status is not None else 0.0,
    }


def _summarize_spectral_audit_rows(audit_rows: List[Dict[str, object]]) -> Dict[str, float]:
    if not audit_rows:
        return {
            "spectral_proj_norm_ratio_mean": 0.0,
            "spectral_perp_norm_ratio_mean": 0.0,
            "spectral_k_eff_mean": 0.0,
            "spectral_energy_ratio_eff_mean": 0.0,
        }
    df = pd.DataFrame(audit_rows)

    def mean_float(col: str, default: float = 0.0) -> float:
        if col not in df.columns:
            return default
        vals = pd.to_numeric(df[col], errors="coerce")
        return float(vals.mean()) if vals.notna().any() else default

    return {
        "spectral_proj_norm_ratio_mean": mean_float("spectral_proj_norm_ratio"),
        "spectral_perp_norm_ratio_mean": mean_float("spectral_perp_norm_ratio"),
        "spectral_k_eff_mean": mean_float("spectral_k_eff"),
        "spectral_energy_ratio_eff_mean": mean_float("spectral_energy_ratio_eff"),
    }


def _project_spectral_structure_out(*, U: torch.Tensor, spectral_basis: torch.Tensor) -> Dict[str, torch.Tensor]:
    if spectral_basis.numel() == 0:
        proj = torch.zeros_like(U)
    else:
        basis = spectral_basis.to(dtype=U.dtype, device=U.device)
        proj = (U @ basis.T) @ basis
    return {"proj": proj, "U_perp": U - proj}


def _apply_rc4_safe_governance(
    *,
    W: torch.Tensor,
    U: torch.Tensor,
    U_perp: torch.Tensor,
    r_shared: torch.Tensor,
    alpha: float,
    beta: float,
    kappa: float,
    eps: float = 1e-8,
    tol: float = 1e-6,
    disable_safe_step: bool = False,
) -> Dict[str, torch.Tensor]:
    norm_u = torch.norm(U, p=2, dim=-1, keepdim=True)
    norm_u_perp = torch.norm(U_perp, p=2, dim=-1, keepdim=True)
    has_perp = norm_u_perp > eps

    U_hat_perp = torch.where(has_perp, U_perp / (norm_u_perp + eps), torch.zeros_like(U_perp))
    U_restored = U_hat_perp * (float(kappa) * norm_u)
    norm_u_restored = torch.norm(U_restored, p=2, dim=-1, keepdim=True)

    delta_s_raw = float(alpha) * W
    norm_delta_s_raw = torch.norm(delta_s_raw, p=2, dim=-1, keepdim=True)
    alpha_base = torch.full_like(r_shared, float(alpha))
    alpha_eff = torch.where(
        norm_delta_s_raw > r_shared,
        alpha_base * (r_shared / (norm_delta_s_raw + eps)),
        alpha_base,
    )
    delta_s = alpha_eff * W
    norm_delta_s = torch.norm(delta_s, p=2, dim=-1, keepdim=True)

    r_risk = torch.sqrt(torch.clamp(r_shared * r_shared - norm_delta_s * norm_delta_s, min=0.0))
    delta_r_raw = float(beta) * U_restored
    norm_delta_r_raw = torch.norm(delta_r_raw, p=2, dim=-1, keepdim=True)
    risk_scale = torch.where(
        norm_delta_r_raw > eps,
        torch.minimum(torch.ones_like(norm_delta_r_raw), r_risk / (norm_delta_r_raw + eps)),
        torch.zeros_like(norm_delta_r_raw),
    )
    delta_r = delta_r_raw * risk_scale

    delta_z = delta_s + delta_r
    norm_final = torch.norm(delta_z, p=2, dim=-1, keepdim=True)
    if torch.any(norm_final > (r_shared + tol)):
        max_overshoot = torch.max(norm_final - r_shared).item()
        if bool(disable_safe_step):
            scale = r_shared / (norm_final + eps)
            delta_z = delta_z * scale
            norm_final = torch.norm(delta_z, p=2, dim=-1, keepdim=True)
        else:
            raise RuntimeError(
                f"RC-4 OSF final norm exceeded shared radius by {max_overshoot:.6e}; "
                "this indicates a bug in the structure-first risk-budget logic."
            )

    zero_perp_mask = (~has_perp).reshape(-1)
    restored_mask = (~zero_perp_mask) & (norm_u_restored.reshape(-1) > eps)
    clipped_mask = ((~zero_perp_mask) & ((norm_delta_r_raw.reshape(-1) - r_risk.reshape(-1)) > tol))
    safe_clip_rate = float(clipped_mask.float().mean().item())

    return {
        "delta_z": delta_z,
        "clip_mask": clipped_mask,
        "safe_clip_rate": safe_clip_rate,
        "r_shared": r_shared,
        "norm_final": norm_final,
        "U_perp": U_perp,
        "norm_u": norm_u,
        "norm_u_perp": norm_u_perp,
        "U_restored": U_restored,
        "norm_u_restored": norm_u_restored,
        "alpha_eff": alpha_eff,
        "delta_s_raw": delta_s_raw,
        "norm_delta_s_raw": norm_delta_s_raw,
        "delta_s": delta_s,
        "norm_delta_s": norm_delta_s,
        "r_risk": r_risk,
        "delta_r_raw": delta_r_raw,
        "norm_delta_r_raw": norm_delta_r_raw,
        "risk_scale": risk_scale,
        "delta_r": delta_r,
        "zero_perp_mask": zero_perp_mask,
        "restored_mask": restored_mask,
        "clipped_mask": clipped_mask,
        "structure_overflow_mask": (norm_delta_s_raw.reshape(-1) > (r_shared.reshape(-1) + tol)),
    }


def _project_rank1_structure_out(
    *,
    W: torch.Tensor,
    U: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    dot_uw = torch.sum(U * W, dim=-1, keepdim=True)
    norm_w_sq = torch.sum(W * W, dim=-1, keepdim=True) + eps
    proj = (dot_uw / norm_w_sq) * W
    U_perp = U - proj
    return {"proj": proj, "U_perp": U_perp}


def _clone_args_with_updates(args, **updates):
    cloned = argparse.Namespace(**vars(args))
    for key, value in updates.items():
        setattr(cloned, key, value)
    return cloned


def _build_rc4_fused_aug_out(
    *,
    args,
    seed: int,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    model_obj,
    batch_size: int,
    include_feedback_scores: bool,
    algo_label: str,
    fusion_mode: str = "rank1_osf",
) -> Dict[str, object]:
    aug_out_lraes = _build_act_realized_augmentations(
        args=_clone_args_with_updates(args, algo="lraes"),
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        train_recs=train_recs,
        mean_log=mean_log,
        algo_override="lraes",
        engine_id="lraes",
    )
    aug_out_zpia = _build_act_realized_augmentations(
        args=_clone_args_with_updates(args, algo="zpia"),
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        train_recs=train_recs,
        mean_log=mean_log,
        algo_override="zpia",
        engine_id="zpia",
    )

    if include_feedback_scores:
        _attach_feedback_scores_to_aug_out(
            aug_out=aug_out_lraes,
            model_obj=model_obj,
            device=args.device,
            batch_size=batch_size,
            feedback_margin_temperature=args.feedback_margin_temperature,
            engine_id="lraes",
        )
        _attach_feedback_scores_to_aug_out(
            aug_out=aug_out_zpia,
            model_obj=model_obj,
            device=args.device,
            batch_size=batch_size,
            feedback_margin_temperature=args.feedback_margin_temperature,
            engine_id="zpia",
        )

    if args.disable_safe_step:
        raise ValueError("orthogonal_fusion requires safe-step metadata; do not use --disable-safe-step.")
    if len(aug_out_lraes["tid_aug"]) != len(aug_out_zpia["tid_aug"]):
        raise ValueError("orthogonal_fusion requires aligned lraes/zpia candidate counts.")
    if len(aug_out_lraes.get("audit_rows", [])) != len(aug_out_zpia.get("audit_rows", [])):
        raise ValueError("orthogonal_fusion requires aligned lraes/zpia audit rows.")

    for idx, (tid_l, tid_z) in enumerate(zip(aug_out_lraes["tid_aug"], aug_out_zpia["tid_aug"])):
        if tid_l != tid_z:
            raise ValueError(
                f"orthogonal_fusion requires aligned candidate anchors; mismatch at index {idx}: {tid_l} != {tid_z}"
            )
    if not np.array_equal(aug_out_lraes["y_aug"], aug_out_zpia["y_aug"]):
        raise ValueError("orthogonal_fusion requires aligned candidate labels between lraes and zpia.")

    eta_l = aug_out_lraes.get("eta_safe", None)
    eta_z = aug_out_zpia.get("eta_safe", None)
    if eta_l is None or eta_z is None:
        raise ValueError("orthogonal_fusion requires finite eta_safe from both engines.")
    if abs(float(eta_l) - float(eta_z)) > 1e-12:
        raise ValueError(f"orthogonal_fusion requires shared eta_safe; got {eta_l} vs {eta_z}.")
    eta_safe = float(eta_l)

    z_o = torch.from_numpy(
        np.stack([aug_out_lraes["tid_to_rec"][tid].z for tid in aug_out_lraes["tid_aug"]])
    )
    z_z = torch.from_numpy(aug_out_zpia["z_aug"])
    z_l = torch.from_numpy(aug_out_lraes["z_aug"])

    W = z_z - z_o
    U = z_l - z_o

    shared_margin = []
    for idx, (row_l, row_z) in enumerate(zip(aug_out_lraes["audit_rows"], aug_out_zpia["audit_rows"])):
        margin_l = float(row_l.get("manifold_margin", np.nan))
        margin_z = float(row_z.get("manifold_margin", np.nan))
        if (not np.isfinite(margin_l)) or (not np.isfinite(margin_z)):
            raise ValueError(
                f"orthogonal_fusion requires finite manifold_margin on both engines; failed at candidate {idx}."
            )
        if margin_l < 0.0 or margin_z < 0.0:
            raise ValueError(
                f"orthogonal_fusion received negative manifold_margin at candidate {idx}: {margin_l}, {margin_z}"
            )
        shared_margin.append(min(margin_l, margin_z))
    r_shared_np = eta_safe * np.asarray(shared_margin, dtype=np.float64)
    r_shared = torch.from_numpy(r_shared_np).to(dtype=z_o.dtype).unsqueeze(-1)

    spectral_meta: Dict[str, object] = {}
    spectral_basis_np: Optional[np.ndarray] = None
    spectral_proj = None
    if fusion_mode == "rank1_osf":
        proj_out = _project_rank1_structure_out(W=W, U=U)
        rc4 = _apply_rc4_safe_governance(
            W=W,
            U=U,
            U_perp=proj_out["U_perp"],
            r_shared=r_shared,
            alpha=float(args.osf_alpha),
            beta=float(args.osf_beta),
            kappa=float(getattr(args, "osf_kappa", 1.0)),
            disable_safe_step=bool(getattr(args, "disable_safe_step", False)),
        )
        rc4["proj"] = proj_out["proj"]
        rc4["norm_proj"] = torch.norm(proj_out["proj"], p=2, dim=-1, keepdim=True)
        direction_bank_source = "orthogonal_fusion"
    elif fusion_mode == "spectral_osf":
        zpia_bank = np.asarray(aug_out_zpia.get("direction_bank"), dtype=np.float64)
        spectral_basis_np, spectral_meta = _build_spectral_structure_basis_from_zpia_bank(
            zpia_bank,
            energy_ratio=float(getattr(args, "spectral_osf_rho", 0.90)),
        )
        spectral_basis = torch.from_numpy(spectral_basis_np).to(dtype=z_o.dtype)
        spectral_proj = _project_spectral_structure_out(U=U, spectral_basis=spectral_basis)
        rc4 = _apply_rc4_safe_governance(
            W=W,
            U=U,
            U_perp=spectral_proj["U_perp"],
            r_shared=r_shared,
            alpha=float(args.osf_alpha),
            beta=float(args.osf_beta),
            kappa=float(getattr(args, "osf_kappa", 1.0)),
            disable_safe_step=bool(getattr(args, "disable_safe_step", False)),
        )
        rc4["proj"] = spectral_proj["proj"]
        rc4["norm_proj"] = torch.norm(spectral_proj["proj"], p=2, dim=-1, keepdim=True)
        direction_bank_source = "spectral_osf"
    else:
        raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
    z_f = z_o + rc4["delta_z"]

    fused_x_list: List[np.ndarray] = []
    fused_bridge_metrics: List[Dict[str, object]] = []
    fused_audit_rows: List[Dict[str, object]] = []
    for i in range(len(z_f)):
        src = aug_out_lraes["tid_to_rec"][aug_out_lraes["tid_aug"][i]]
        sigma_f = logvec_to_spd(z_f[i].numpy(), mean_log)
        xf, bridge_meta = bridge_single(
            torch.from_numpy(src.x_raw),
            torch.from_numpy(src.sigma_orig),
            torch.from_numpy(sigma_f),
        )
        xf_np = xf.numpy()
        fused_x_list.append(xf_np)
        fused_bridge_metrics.append(bridge_meta)

        base_row = dict(aug_out_lraes["audit_rows"][i])
        row_z = aug_out_zpia["audit_rows"][i]
        final_norm = float(rc4["norm_final"][i].item())
        shared_r = float(r_shared_np[i])
        if rc4["zero_perp_mask"][i].item():
            risk_status = "zero_perp"
        elif rc4["clipped_mask"][i].item():
            risk_status = "clipped"
        else:
            risk_status = "restored"

        base_row.update(
            {
                "algo": str(algo_label),
                "engine_id": "osf_fused",
                "direction_bank_source": direction_bank_source,
                "manifold_margin": float(shared_margin[i]),
                "safe_radius_ratio": float(final_norm / (shared_r + 1e-12)) if shared_r > 0.0 else 0.0,
                "transport_error_fro": float(bridge_meta.get("transport_error_fro", 0.0)),
                "transport_error_logeuc": float(bridge_meta.get("transport_error_logeuc", 0.0)),
                "bridge_cond_A": float(bridge_meta.get("bridge_cond_A", 0.0)),
                "metric_preservation_error": float(bridge_meta.get("metric_preservation_error", 0.0)),
                "osf_direction_id_lraes": int(base_row.get("direction_id", -1)),
                "osf_direction_id_zpia": int(row_z.get("direction_id", -1)),
                "osf_sign_lraes": float(base_row.get("sign", 0.0)),
                "osf_sign_zpia": float(row_z.get("sign", 0.0)),
                "osf_gamma_used_lraes": float(base_row.get("gamma_used", 0.0)),
                "osf_gamma_used_zpia": float(row_z.get("gamma_used", 0.0)),
                "direction_id_lraes": int(base_row.get("direction_id", -1)),
                "direction_id_zpia": int(row_z.get("direction_id", -1)),
                "gamma_lraes": float(base_row.get("gamma_used", 0.0)),
                "gamma_zpia": float(row_z.get("gamma_used", 0.0)),
                "manifold_margin_lraes": float(aug_out_lraes["audit_rows"][i].get("manifold_margin", np.nan)),
                "manifold_margin_zpia": float(row_z.get("manifold_margin", np.nan)),
                "osf_r_shared": shared_r,
                "osf_alpha_eff": float(rc4["alpha_eff"][i].item()),
                "osf_structure_norm_raw": float(rc4["norm_delta_s_raw"][i].item()),
                "osf_structure_norm_eff": float(rc4["norm_delta_s"][i].item()),
                "osf_risk_norm_restored": float(rc4["norm_u_restored"][i].item()),
                "osf_risk_budget": float(rc4["r_risk"][i].item()),
                "osf_risk_scale": float(rc4["risk_scale"][i].item()),
                "osf_final_norm": final_norm,
                "osf_structure_overflow": bool(rc4["structure_overflow_mask"][i].item()),
                "osf_risk_status": risk_status,
            }
        )
        if fusion_mode == "spectral_osf":
            norm_u = float(rc4["norm_u"][i].item())
            proj_norm = float(rc4["norm_proj"][i].item())
            perp_norm = float(rc4["norm_u_perp"][i].item())
            base_row.update(
                {
                    "spectral_k_eff": int(spectral_meta.get("spectral_k_eff", 0)),
                    "spectral_energy_ratio_eff": float(spectral_meta.get("spectral_energy_ratio_eff", 0.0)),
                    "spectral_proj_norm": proj_norm,
                    "spectral_perp_norm": perp_norm,
                    "spectral_proj_norm_ratio": float(proj_norm / (norm_u + 1e-12)),
                    "spectral_perp_norm_ratio": float(perp_norm / (norm_u + 1e-12)),
                }
            )
        fused_audit_rows.append(base_row)

    fused_x = np.stack(fused_x_list) if fused_x_list else None

    gamma_req_vals = [float(row.get("gamma_requested", 0.0)) for row in fused_audit_rows]
    gamma_used_vals = [float(row.get("gamma_used", 0.0)) for row in fused_audit_rows]
    if not gamma_req_vals:
        gamma_req_vals = [float(args.pia_gamma)] * len(fused_audit_rows)

    clip_flags = [float(1.0 if row.get("osf_risk_status") == "clipped" else 0.0) for row in fused_audit_rows]
    safe_ratios = [float(row.get("safe_radius_ratio", 0.0)) for row in fused_audit_rows]
    margins = [float(row.get("manifold_margin", 0.0)) for row in fused_audit_rows]
    fused_aug_out = {
        "effective_k": int(max(aug_out_lraes.get("effective_k", 0), aug_out_zpia.get("effective_k", 0))),
        "effective_k_lraes": int(aug_out_lraes.get("effective_k", 0)),
        "effective_k_zpia": int(aug_out_zpia.get("effective_k", 0)),
        "z_aug": z_f.numpy(),
        "y_aug": aug_out_lraes["y_aug"],
        "tid_aug": aug_out_lraes["tid_aug"],
        "aug_trials": [
            {"x": fused_x_list[i], "y": int(aug_out_lraes["y_aug"][i]), "tid": aug_out_lraes["tid_aug"][i]}
            for i in range(len(fused_x_list))
        ],
        "X_aug_raw": fused_x,
        "y_aug_np": aug_out_lraes["y_aug_np"],
        "tid_to_rec": aug_out_lraes["tid_to_rec"],
        "avg_bridge": pd.DataFrame(fused_bridge_metrics).mean().to_dict() if fused_bridge_metrics else {},
        "audit_rows": fused_audit_rows,
        "direction_bank_meta": {
            "bank_source": direction_bank_source,
            "engine_sources": [
                aug_out_lraes.get("direction_bank_meta", {}).get("bank_source", "lraes"),
                aug_out_zpia.get("direction_bank_meta", {}).get("bank_source", "zpia_telm2"),
            ],
            "lraes_meta": aug_out_lraes.get("direction_bank_meta", {}),
            "zpia_meta": aug_out_zpia.get("direction_bank_meta", {}),
            **spectral_meta,
        },
        "safe_radius_ratio_mean": float(np.mean(safe_ratios)) if safe_ratios else 1.0,
        "safe_clip_rate": float(np.mean(clip_flags)) if clip_flags else 0.0,
        "manifold_margin_mean": float(np.mean(margins)) if margins else 0.0,
        "gamma_requested_mean": float(np.mean(gamma_req_vals)) if gamma_req_vals else 0.0,
        "gamma_used_mean": float(np.mean(gamma_used_vals)) if gamma_used_vals else 0.0,
        "gamma_zero_rate": float(np.mean([1.0 if g < 1e-12 else 0.0 for g in gamma_used_vals])) if gamma_used_vals else 0.0,
        "eta_safe": eta_safe,
        "candidate_total_count": int(aug_out_lraes.get("candidate_total_count", len(fused_x_list))),
        "aug_total_count": int(aug_out_lraes.get("aug_total_count", len(fused_x_list))),
        "aug_dataset": aug_out_lraes.get("aug_dataset"),
    }
    if fused_aug_out["aug_dataset"] is not None:
        fused_aug_out["aug_dataset"]._z_cands = z_f.numpy().astype(np.float32)

    if include_feedback_scores:
        _attach_feedback_scores_to_aug_out(
            aug_out=fused_aug_out,
            model_obj=model_obj,
            device=args.device,
            batch_size=batch_size,
            feedback_margin_temperature=args.feedback_margin_temperature,
            engine_id="osf_fused",
        )

    return fused_aug_out


def _build_rc4_multiz_fused_aug_out(
    *,
    args,
    seed: int,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    model_obj=None,
    batch_size: Optional[int] = None,
    algo_label: str = "rc4_multiz_fused",
) -> Dict[str, object]:
    if args.disable_safe_step:
        raise ValueError("rc4_multiz_fused requires safe-step metadata; do not use --disable-safe-step.")

    aug_out_lraes = _build_act_realized_augmentations(
        args=_clone_args_with_updates(args, algo="lraes"),
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        train_recs=train_recs,
        mean_log=mean_log,
        algo_override="lraes",
        engine_id="lraes",
    )
    bank_out_zpia = _build_direction_bank_for_args(
        args=_clone_args_with_updates(args, algo="zpia"),
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        algo_override="zpia",
    )
    zpia_bank = np.asarray(bank_out_zpia["bank"], dtype=np.float64)
    zpia_meta = dict(bank_out_zpia["meta"])
    eta_safe = float(aug_out_lraes.get("eta_safe", 0.5))
    slots = _build_top_response_template_slots(
        args=args,
        X_train_z=X_train_z,
        y_train=y_train,
        train_recs=train_recs,
        zpia_bank=zpia_bank,
        seed=seed,
        candidate_rows=list(aug_out_lraes.get("audit_rows", [])),
        top1_only=False,
        eta_safe=eta_safe,
        direction_meta=zpia_meta,
    )
    if len(aug_out_lraes["tid_aug"]) != len(slots["tid_aug"]):
        raise ValueError("rc4_multiz_fused requires aligned LRAES and zPIA slot counts.")
    for idx, (tid_l, tid_z) in enumerate(zip(aug_out_lraes["tid_aug"], slots["tid_aug"])):
        if tid_l != tid_z:
            raise ValueError(f"rc4_multiz_fused slot tid mismatch at {idx}: {tid_l} != {tid_z}")
    if not np.array_equal(aug_out_lraes["y_aug"], slots["y_aug"]):
        raise ValueError("rc4_multiz_fused requires aligned LRAES and zPIA slot labels.")

    z_o = torch.from_numpy(
        np.stack([aug_out_lraes["tid_to_rec"][tid].z for tid in aug_out_lraes["tid_aug"]])
    )
    z_l = torch.from_numpy(aug_out_lraes["z_aug"])
    W = torch.from_numpy(slots["W_slots"]).to(dtype=z_o.dtype)
    U = z_l - z_o

    shared_margin: List[float] = []
    for idx, (row_l, row_z) in enumerate(zip(aug_out_lraes["audit_rows"], slots["audit_rows"])):
        margin_l = float(row_l.get("manifold_margin", np.nan))
        margin_z = float(row_z.get("manifold_margin", np.nan))
        if (not np.isfinite(margin_l)) or (not np.isfinite(margin_z)):
            raise ValueError(f"rc4_multiz_fused requires finite manifold_margin at slot {idx}.")
        shared_margin.append(min(margin_l, margin_z))
    r_shared_np = eta_safe * np.asarray(shared_margin, dtype=np.float64)
    r_shared = torch.from_numpy(r_shared_np).to(dtype=z_o.dtype).unsqueeze(-1)

    proj_out = _project_rank1_structure_out(W=W, U=U)
    rc4 = _apply_rc4_safe_governance(
        W=W,
        U=U,
        U_perp=proj_out["U_perp"],
        r_shared=r_shared,
        alpha=float(args.osf_alpha),
        beta=float(args.osf_beta),
        kappa=float(getattr(args, "osf_kappa", 1.0)),
    )
    rc4["proj"] = proj_out["proj"]
    rc4["norm_proj"] = torch.norm(proj_out["proj"], p=2, dim=-1, keepdim=True)
    z_f = z_o + rc4["delta_z"]

    fused_rows: List[Dict[str, object]] = []
    for i, (row_l, row_z) in enumerate(zip(aug_out_lraes["audit_rows"], slots["audit_rows"])):
        final_norm = float(rc4["norm_final"][i].item())
        shared_r = float(r_shared_np[i])
        if rc4["zero_perp_mask"][i].item():
            risk_status = "zero_perp"
        elif rc4["clipped_mask"][i].item():
            risk_status = "clipped"
        else:
            risk_status = "restored"
        norm_u = float(rc4["norm_u"][i].item())
        norm_u_perp = float(rc4["norm_u_perp"][i].item())
        row = dict(row_l)
        row.update(
            {
                "algo": algo_label,
                "engine_id": "multi_template_osf",
                "direction_bank_source": "multi_template_osf",
                "manifold_margin": float(shared_margin[i]),
                "safe_radius_ratio": float(final_norm / (shared_r + 1e-12)) if shared_r > 0.0 else 0.0,
                "lraes_direction_id": int(row_l.get("direction_id", -1)),
                "lraes_sign": float(row_l.get("sign", 0.0)),
                "lraes_gamma_used": float(row_l.get("gamma_used", 0.0)),
                "lraes_delta_norm": norm_u,
                "zpia_template_id": int(row_z.get("zpia_template_id", -1)),
                "zpia_template_sign": float(row_z.get("zpia_template_sign", 0.0)),
                "zpia_template_rank": int(row_z.get("zpia_template_rank", -1)),
                "zpia_template_response_abs": float(row_z.get("zpia_template_response_abs", 0.0)),
                "zpia_gamma_used": float(row_z.get("gamma_used", 0.0)),
                "zpia_delta_norm": float(row_z.get("zpia_delta_norm", 0.0)),
                "proj_on_template_norm": float(rc4["norm_proj"][i].item()),
                "u_perp_norm": norm_u_perp,
                "u_perp_norm_ratio": float(norm_u_perp / (norm_u + 1e-12)),
                "osf_r_shared": shared_r,
                "osf_alpha_eff": float(rc4["alpha_eff"][i].item()),
                "osf_structure_norm_raw": float(rc4["norm_delta_s_raw"][i].item()),
                "osf_structure_norm_eff": float(rc4["norm_delta_s"][i].item()),
                "osf_risk_norm_restored": float(rc4["norm_u_restored"][i].item()),
                "osf_risk_budget": float(rc4["r_risk"][i].item()),
                "osf_risk_scale": float(rc4["risk_scale"][i].item()),
                "osf_final_norm": final_norm,
                "osf_structure_overflow": bool(rc4["structure_overflow_mask"][i].item()),
                "osf_risk_status": risk_status,
            }
        )
        fused_rows.append(row)

    direction_meta = {
        "bank_source": "multi_template_osf",
        "engine_sources": [
            aug_out_lraes.get("direction_bank_meta", {}).get("bank_source", "lraes"),
            zpia_meta.get("bank_source", "zpia_telm2"),
        ],
        "lraes_meta": aug_out_lraes.get("direction_bank_meta", {}),
        "zpia_meta": zpia_meta,
        "template_selection": "anchor_top_abs_response",
        "template_slot_policy": "slot_preserving_dual_sign_fixed_budget",
        "multi_template_pairs": int(slots["multi_template_pairs"]),
    }
    return _materialize_z_aug_out(
        z_aug=z_f.numpy().astype(np.float32),
        y_aug=aug_out_lraes["y_aug"],
        tid_aug=aug_out_lraes["tid_aug"],
        audit_rows=fused_rows,
        train_recs=train_recs,
        mean_log=mean_log,
        direction_bank_meta=direction_meta,
        effective_k=int(max(aug_out_lraes.get("effective_k", 0), zpia_bank.shape[0])),
        eta_safe=eta_safe,
        algo_name=algo_label,
        engine_id="multi_template_osf",
        extra_meta={
            "effective_k_lraes": int(aug_out_lraes.get("effective_k", 0)),
            "effective_k_zpia": int(zpia_bank.shape[0]),
            "multi_template_pairs": int(slots["multi_template_pairs"]),
            "template_usage_entropy": float(slots["template_usage_entropy"]),
            "top_template_concentration": float(slots["top_template_concentration"]),
            **_summarize_osf_audit_rows(fused_rows),
            **_summarize_multitemplate_audit_rows(fused_rows),
        },
    )
