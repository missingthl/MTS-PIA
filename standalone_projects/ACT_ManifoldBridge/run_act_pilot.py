import os

if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from core.bridge import bridge_single, logvec_to_spd
from core.curriculum import active_direction_probs, build_curriculum_aug_candidates, estimate_local_manifold_margins
from core.pia import (
    FisherPIAConfig,
    LRAESConfig,
    _build_spectral_structure_basis_from_zpia_bank,
    build_lraes_direction_bank,
    build_pia_direction_bank,
    build_zpia_direction_bank,
)
from host_alignment_probe import compute_gradient_alignment
from utils.datasets import AEON_FIXED_SPLIT_SPECS, load_trials_for_dataset, make_trial_split
from utils.evaluators import (
    ManifoldAugDataset,
    TauScheduler,
    build_model,
    fit_eval_minirocket,
    fit_eval_patchtst,
    fit_eval_patchtst_weighted_aug_ce,
    fit_eval_resnet1d_adaptive_aug_ce,
    fit_eval_resnet1d_progressive_aug_ce,
    fit_eval_resnet1d,
    fit_eval_resnet1d_weighted_aug_ce,
    fit_eval_timesnet,
    fit_eval_timesnet_weighted_aug_ce,
)


@dataclass
class TrialRecord:
    tid: str
    y: int
    x_raw: np.ndarray
    sigma_orig: np.ndarray
    z: np.ndarray


def _build_trial_records(trials, spd_eps: float = 1e-4):
    if not trials:
        return [], None

    records = []
    log_covs = []
    for t in trials:
        x = torch.from_numpy(t.x).double()
        x = x - x.mean(dim=-1, keepdim=True)
        cov = (x @ x.transpose(-1, -2)) / (x.shape[-1] - 1)
        cov = cov + spd_eps * torch.eye(cov.shape[0], dtype=cov.dtype)
        vals, vecs = torch.linalg.eigh(cov)
        log_cov = vecs @ torch.diag_embed(torch.log(torch.clamp(vals, min=spd_eps))) @ vecs.transpose(-1, -2)
        log_covs.append(log_cov.numpy())
        records.append(
            {
                "tid": t.tid,
                "y": t.y,
                "x_raw": t.x,
                "sigma_orig": cov.numpy(),
                "log_cov": log_cov.numpy(),
            }
        )

    mean_log = np.mean(log_covs, axis=0)
    idx = np.triu_indices(mean_log.shape[0])
    final_records = []
    for record in records:
        z = (record["log_cov"] - mean_log)[idx]
        final_records.append(
            TrialRecord(
                tid=record["tid"],
                y=record["y"],
                x_raw=record["x_raw"],
                sigma_orig=record["sigma_orig"],
                z=z,
            )
        )
    return final_records, mean_log


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
        raise RuntimeError(
            f"RC-4 OSF final norm exceeded shared radius by {max_overshoot:.6e}; "
            "this indicates a bug in the structure-first risk-budget logic."
        )

    zero_perp_mask = (~has_perp).reshape(-1)
    clipped_mask = ((~zero_perp_mask) & ((norm_delta_r_raw.reshape(-1) - r_risk.reshape(-1)) > tol))
    restored_mask = (~zero_perp_mask) & (~clipped_mask)

    return {
        "delta_z": delta_z,
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
        "norm_final": norm_final,
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


def _project_spectral_structure_out(
    *,
    U: torch.Tensor,
    spectral_basis: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    del eps  # kept for interface symmetry with rank-1 projection helper
    proj = (U @ spectral_basis) @ spectral_basis.transpose(0, 1)
    U_perp = U - proj
    return {"proj": proj, "U_perp": U_perp}


def _compute_osf_rc4_fusion(
    *,
    W: torch.Tensor,
    U: torch.Tensor,
    r_shared: torch.Tensor,
    alpha: float,
    beta: float,
    kappa: float,
    eps: float = 1e-8,
    tol: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    proj_out = _project_rank1_structure_out(W=W, U=U, eps=eps)
    govern = _apply_rc4_safe_governance(
        W=W,
        U=U,
        U_perp=proj_out["U_perp"],
        r_shared=r_shared,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        eps=eps,
        tol=tol,
    )
    govern.update(
        {
            "proj": proj_out["proj"],
            "norm_proj": torch.norm(proj_out["proj"], p=2, dim=-1, keepdim=True),
        }
    )
    return govern


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
    out: Dict[str, float] = {}
    if "osf_structure_overflow" in df.columns:
        out["osf_structure_overflow_rate"] = float(pd.to_numeric(df["osf_structure_overflow"], errors="coerce").fillna(0.0).mean())
    if "osf_alpha_eff" in df.columns:
        out["osf_alpha_eff_mean"] = float(pd.to_numeric(df["osf_alpha_eff"], errors="coerce").dropna().mean())
    if "osf_risk_scale" in df.columns:
        out["osf_risk_scale_mean"] = float(pd.to_numeric(df["osf_risk_scale"], errors="coerce").dropna().mean())
    if "osf_risk_status" in df.columns:
        status = df["osf_risk_status"].astype(str)
        out["osf_risk_zero_perp_rate"] = float((status == "zero_perp").mean())
        out["osf_risk_clipped_rate"] = float((status == "clipped").mean())
    return {
        "osf_structure_overflow_rate": float(out.get("osf_structure_overflow_rate", 0.0)),
        "osf_alpha_eff_mean": float(out.get("osf_alpha_eff_mean", 0.0)),
        "osf_risk_scale_mean": float(out.get("osf_risk_scale_mean", 0.0)),
        "osf_risk_zero_perp_rate": float(out.get("osf_risk_zero_perp_rate", 0.0)),
        "osf_risk_clipped_rate": float(out.get("osf_risk_clipped_rate", 0.0)),
    }


def _summarize_spectral_audit_rows(audit_rows: List[Dict[str, object]]) -> Dict[str, float]:
    if not audit_rows:
        return {
            "spectral_proj_norm_ratio_mean": 0.0,
            "spectral_perp_norm_ratio_mean": 0.0,
            "spectral_zero_perp_rate": 0.0,
            "spectral_risk_clipped_rate": 0.0,
            "spectral_risk_scale_mean": 0.0,
            "spectral_structure_overflow_rate": 0.0,
            "spectral_alpha_eff_mean": 0.0,
        }

    df = pd.DataFrame(audit_rows)
    status = df["osf_risk_status"].astype(str) if "osf_risk_status" in df.columns else pd.Series([], dtype=str)
    return {
        "spectral_proj_norm_ratio_mean": float(
            pd.to_numeric(df.get("spectral_proj_norm_ratio"), errors="coerce").dropna().mean()
        ) if "spectral_proj_norm_ratio" in df.columns else 0.0,
        "spectral_perp_norm_ratio_mean": float(
            pd.to_numeric(df.get("spectral_perp_norm_ratio"), errors="coerce").dropna().mean()
        ) if "spectral_perp_norm_ratio" in df.columns else 0.0,
        "spectral_zero_perp_rate": float((status == "zero_perp").mean()) if not status.empty else 0.0,
        "spectral_risk_clipped_rate": float((status == "clipped").mean()) if not status.empty else 0.0,
        "spectral_risk_scale_mean": float(
            pd.to_numeric(df.get("osf_risk_scale"), errors="coerce").dropna().mean()
        ) if "osf_risk_scale" in df.columns else 0.0,
        "spectral_structure_overflow_rate": float(
            pd.to_numeric(df.get("osf_structure_overflow"), errors="coerce").fillna(0.0).mean()
        ) if "osf_structure_overflow" in df.columns else 0.0,
        "spectral_alpha_eff_mean": float(
            pd.to_numeric(df.get("osf_alpha_eff"), errors="coerce").dropna().mean()
        ) if "osf_alpha_eff" in df.columns else 0.0,
    }


def _summarize_multitemplate_audit_rows(audit_rows: List[Dict[str, object]]) -> Dict[str, float]:
    if not audit_rows:
        return {
            "template_usage_entropy": 0.0,
            "top_template_concentration": 0.0,
            "u_perp_norm_ratio_mean": 0.0,
            "u_perp_zero_rate": 0.0,
        }
    df = pd.DataFrame(audit_rows)
    out: Dict[str, float] = {}
    if "zpia_template_id" in df.columns:
        template_ids = pd.to_numeric(df["zpia_template_id"], errors="coerce").dropna().astype(int).tolist()
        out.update(_template_usage_stats(template_ids))
    if "u_perp_norm_ratio" in df.columns:
        ratios = pd.to_numeric(df["u_perp_norm_ratio"], errors="coerce").dropna()
        out["u_perp_norm_ratio_mean"] = float(ratios.mean()) if not ratios.empty else 0.0
        out["u_perp_zero_rate"] = float((ratios <= 1e-8).mean()) if not ratios.empty else 0.0
    return {
        "template_usage_entropy": float(out.get("template_usage_entropy", 0.0)),
        "top_template_concentration": float(out.get("top_template_concentration", 0.0)),
        "u_perp_norm_ratio_mean": float(out.get("u_perp_norm_ratio_mean", 0.0)),
        "u_perp_zero_rate": float(out.get("u_perp_zero_rate", 0.0)),
    }


def _summarize_progressive_audit_rows(audit_rows: List[Dict[str, object]]) -> Dict[str, float]:
    if not audit_rows:
        return {
            "progressive_useful_zpia_mean": 0.0,
            "progressive_useful_osf_mean": 0.0,
            "progressive_mode_zpia_rate": 0.0,
            "progressive_mode_osf_rate": 0.0,
            "progressive_mode_conservative_rate": 0.0,
        }
    df = pd.DataFrame(audit_rows)
    mode = df["mode"].astype(str) if "mode" in df.columns else pd.Series([], dtype=str)
    useful = pd.to_numeric(df.get("useful"), errors="coerce").fillna(0.0) if "useful" in df.columns else pd.Series([], dtype=float)
    zpia_mask = mode.isin(["zpia_top1", "conservative_zpia"]) if not mode.empty else pd.Series([], dtype=bool)
    osf_mask = mode == "weak_osf" if not mode.empty else pd.Series([], dtype=bool)
    return {
        "progressive_useful_zpia_mean": float(useful[zpia_mask].mean()) if len(useful) and zpia_mask.any() else 0.0,
        "progressive_useful_osf_mean": float(useful[osf_mask].mean()) if len(useful) and osf_mask.any() else 0.0,
        "progressive_mode_zpia_rate": float((mode == "zpia_top1").mean()) if not mode.empty else 0.0,
        "progressive_mode_osf_rate": float(osf_mask.mean()) if not mode.empty else 0.0,
        "progressive_mode_conservative_rate": float((mode == "conservative_zpia").mean()) if not mode.empty else 0.0,
    }


def _fit_host_model(
    *,
    args,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    return_model_obj: bool = False,
    loader_seed: Optional[int] = None,
) -> Dict[str, object]:
    kwargs = {
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "patience": patience,
        "device": args.device,
        "return_model_obj": return_model_obj,
    }
    if args.model == "resnet1d":
        kwargs["loader_seed"] = loader_seed
        return fit_eval_resnet1d(X_tr, y_tr, X_val_raw, y_val, X_test_raw, y_test, **kwargs)
    if args.model == "patchtst":
        kwargs["loader_seed"] = loader_seed
        return fit_eval_patchtst(X_tr, y_tr, X_val_raw, y_val, X_test_raw, y_test, **kwargs)
    if args.model == "timesnet":
        kwargs["loader_seed"] = loader_seed
        return fit_eval_timesnet(X_tr, y_tr, X_val_raw, y_val, X_test_raw, y_test, **kwargs)

    model = build_model(n_kernels=args.n_kernels, random_state=loader_seed or 42)
    return fit_eval_minirocket(model, X_tr, y_tr, X_test_raw, y_test)


def _fit_host_model_weighted_aug_ce(
    *,
    args,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_aug: Optional[np.ndarray],
    y_aug: Optional[np.ndarray],
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    loader_seed: Optional[int] = None,
    aug_dataset: Optional[ManifoldAugDataset] = None,
    tau_scheduler: Optional[TauScheduler] = None,
) -> Dict[str, object]:
    kwargs = {
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "patience": patience,
        "device": args.device,
        "feedback_margin_temperature": args.feedback_margin_temperature,
        "aug_loss_weight": args.aug_loss_weight,
        "loader_seed": loader_seed,
        # V2 params
        "aug_dataset": aug_dataset,
        "weight_mode": args.aug_weight_mode,
        "tau_scheduler": tau_scheduler,
        "steps_per_epoch": args.steps_per_epoch,
    }
    if args.model == "resnet1d":
        return fit_eval_resnet1d_weighted_aug_ce(
            X_tr, y_tr, X_aug, y_aug, X_val_raw, y_val, X_test_raw, y_test, **kwargs
        )
    if args.model == "patchtst":
        return fit_eval_patchtst_weighted_aug_ce(
            X_tr, y_tr, X_aug, y_aug, X_val_raw, y_val, X_test_raw, y_test, **kwargs
        )
    if args.model == "timesnet":
        return fit_eval_timesnet_weighted_aug_ce(
            X_tr, y_tr, X_aug, y_aug, X_val_raw, y_val, X_test_raw, y_test, **kwargs
        )
    raise ValueError("Weighted aug-CE training supports resnet1d, patchtst, and timesnet only.")


def _fit_host_model_adaptive_aug_ce(
    *,
    args,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_aug_lraes: Optional[np.ndarray],
    y_aug_lraes: Optional[np.ndarray],
    X_aug_zpia: Optional[np.ndarray],
    y_aug_zpia: Optional[np.ndarray],
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    loader_seed: Optional[int] = None,
    aug_dataset_lraes: Optional[ManifoldAugDataset] = None,
    aug_dataset_zpia: Optional[ManifoldAugDataset] = None,
    tau_scheduler: Optional[TauScheduler] = None,
) -> Dict[str, object]:
    if args.model != "resnet1d":
        raise ValueError("Adaptive router v1 currently supports resnet1d only.")
    return fit_eval_resnet1d_adaptive_aug_ce(
        X_tr, y_tr,
        X_aug_lraes, y_aug_lraes,
        X_aug_zpia, y_aug_zpia,
        X_val_raw, y_val, X_test_raw, y_test,
        epochs=epochs, lr=lr, batch_size=batch_size, patience=patience,
        device=args.device,
        feedback_margin_temperature=args.feedback_margin_temperature,
        aug_loss_weight=args.aug_loss_weight,
        router_temperature=args.router_temperature,
        router_min_prob=args.router_min_prob,
        router_smoothing=args.router_smoothing,
        loader_seed=loader_seed,
        aug_dataset_lraes=aug_dataset_lraes,
        aug_dataset_zpia=aug_dataset_zpia,
        weight_mode=getattr(args, "aug_weight_mode", "sigmoid"),
        tau_scheduler=tau_scheduler,
        steps_per_epoch=getattr(args, "steps_per_epoch", 0),
        lambda_consistency=getattr(args, "lambda_consistency", 0.0),
        consistency_mode=getattr(args, "consistency_mode", "mse"),
    )


def _score_aug_margins(
    *,
    model_obj,
    X_aug: Optional[np.ndarray],
    y_aug: Optional[np.ndarray],
    device: str,
    batch_size: int,
) -> np.ndarray:
    if model_obj is None or X_aug is None or y_aug is None or len(y_aug) == 0:
        return np.empty((0,), dtype=np.float64)
    use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    dev = torch.device(device if use_cuda else "cpu")
    model_obj.to(dev)
    model_obj.eval()
    margins: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(y_aug), int(batch_size)):
            bx = torch.from_numpy(X_aug[start : start + int(batch_size)]).float().to(dev)
            by = torch.from_numpy(y_aug[start : start + int(batch_size)]).long().to(dev)
            logits = model_obj(bx)
            true_logits = logits.gather(1, by.view(-1, 1)).squeeze(1)
            if logits.shape[1] <= 1:
                margin = true_logits
            else:
                masked = logits.clone()
                masked.scatter_(1, by.view(-1, 1), -torch.inf)
                other_logits = torch.max(masked, dim=1).values
                margin = true_logits - other_logits
            margins.append(margin.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(margins) if margins else np.empty((0,), dtype=np.float64)


def _attach_feedback_scores_to_aug_out(
    *,
    aug_out: Dict[str, object],
    model_obj,
    device: str,
    batch_size: int,
    feedback_margin_temperature: float,
    engine_id: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    margins = _score_aug_margins(
        model_obj=model_obj,
        X_aug=aug_out.get("X_aug_raw"),
        y_aug=aug_out.get("y_aug_np"),
        device=device,
        batch_size=batch_size,
    )
    scaled_margins = np.clip(margins / max(float(feedback_margin_temperature), 1e-6), -60.0, 60.0)
    weights = 1.0 / (1.0 + np.exp(-scaled_margins))
    for idx, row in enumerate(aug_out.get("audit_rows", [])):
        if engine_id is not None:
            row["engine_id"] = str(engine_id)
        row["margin_aug"] = float(margins[idx]) if idx < len(margins) else 0.0
        row["feedback_weight"] = float(weights[idx]) if idx < len(weights) else 0.0
    return {"margins": margins, "weights": weights}


def _clone_args_with_updates(args, **updates):
    cloned = argparse.Namespace(**vars(args))
    for key, value in updates.items():
        setattr(cloned, key, value)
    return cloned


def _build_direction_bank_for_args(
    *,
    args,
    seed: int,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    algo_override: Optional[str] = None,
) -> Dict[str, object]:
    algo_name = str(algo_override or args.algo)
    if algo_name == "lraes":
        direction_bank, direction_meta = build_lraes_direction_bank(
            X_train_z,
            y_train,
            k_dir=args.k_dir,
            fisher_cfg=FisherPIAConfig(),
            lraes_cfg=LRAESConfig(),
        )
    elif algo_name == "zpia":
        direction_bank, direction_meta = build_zpia_direction_bank(
            X_train_z,
            k_dir=args.k_dir,
            seed=seed,
            telm2_n_iters=args.telm2_n_iters,
            telm2_c_repr=args.telm2_c_repr,
            telm2_activation=args.telm2_activation,
            telm2_bias_update_mode=args.telm2_bias_update_mode,
        )
    else:
        direction_bank, direction_meta = build_pia_direction_bank(X_train_z, k_dir=args.k_dir, seed=seed)
    return {"bank": direction_bank, "meta": direction_meta}


def _resolve_multi_template_pairs(*, args, effective_k: int, top1_only: bool) -> int:
    if int(args.multiplier) <= 0:
        raise ValueError("multi-template pools require --multiplier > 0.")
    if int(args.multiplier) % 2 != 0:
        raise ValueError("multi-template pools require an even --multiplier for +/- template slots.")
    if top1_only:
        pairs = 1
    else:
        configured = int(getattr(args, "multi_template_pairs", 0))
        pairs = configured if configured > 0 else int(args.multiplier) // 2
        if 2 * pairs != int(args.multiplier):
            raise ValueError(
                "--multi-template-pairs must satisfy 2 * pairs == multiplier for zpia_multidir_pool "
                "and rc4_multiz_fused."
            )
    if pairs <= 0:
        raise ValueError("--multi-template-pairs must be positive.")
    if pairs > int(effective_k):
        raise ValueError(
            f"--multi-template-pairs={pairs} exceeds effective zPIA bank size {effective_k}."
        )
    return pairs


def _template_usage_stats(template_ids: List[int]) -> Dict[str, float]:
    if not template_ids:
        return {"template_usage_entropy": 0.0, "top_template_concentration": 0.0}
    _, counts = np.unique(np.asarray(template_ids, dtype=np.int64), return_counts=True)
    probs = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    return {
        "template_usage_entropy": entropy,
        "top_template_concentration": float(probs.max()) if probs.size else 0.0,
    }


def _build_top_response_template_slots(
    *,
    args,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    train_recs: List[TrialRecord],
    zpia_bank: np.ndarray,
    candidate_rows: Optional[List[Dict[str, object]]] = None,
    top1_only: bool = False,
    eta_safe: Optional[float] = 0.5,
) -> Dict[str, object]:
    zpia_bank = np.asarray(zpia_bank, dtype=np.float64)
    if zpia_bank.ndim != 2 or zpia_bank.shape[0] <= 0:
        raise ValueError("multi-template zPIA requires a non-empty 2D direction bank.")
    pairs = _resolve_multi_template_pairs(args=args, effective_k=zpia_bank.shape[0], top1_only=top1_only)

    tid_arr = np.asarray([record.tid for record in train_recs], dtype=object)
    y_arr = np.asarray(y_train, dtype=np.int64).ravel()
    tid_to_idx = {tid: i for i, tid in enumerate(tid_arr)}
    margins = estimate_local_manifold_margins(X_train_z, y_arr)

    def _top_ids_for_idx(idx: int) -> np.ndarray:
        responses = np.abs(np.asarray(X_train_z[idx], dtype=np.float64) @ zpia_bank.T)
        # Sort by response descending, then template id ascending for deterministic ties.
        order = np.lexsort((np.arange(zpia_bank.shape[0]), -responses))
        return order[:pairs]

    if candidate_rows is None:
        row_specs: List[Dict[str, object]] = []
        for tid in sorted(tid_to_idx):
            idx = int(tid_to_idx[tid])
            for candidate_order in range(int(args.multiplier)):
                row_specs.append(
                    {
                        "anchor_index": idx,
                        "tid": tid,
                        "class_id": int(y_arr[idx]),
                        "candidate_order": int(candidate_order),
                    }
                )
    else:
        row_specs = []
        for i, row in enumerate(candidate_rows):
            tid = row.get("tid")
            if tid not in tid_to_idx:
                raise ValueError(f"Unknown tid in candidate_rows at slot {i}: {tid}")
            idx = int(row.get("anchor_index", tid_to_idx[tid]))
            row_specs.append(
                {
                    "anchor_index": idx,
                    "tid": tid,
                    "class_id": int(row.get("class_id", y_arr[idx])),
                    "candidate_order": int(row.get("candidate_order", i % int(args.multiplier))),
                }
            )

    z_aug: List[np.ndarray] = []
    y_aug: List[int] = []
    tid_aug: List[object] = []
    w_slots: List[np.ndarray] = []
    slot_rows: List[Dict[str, object]] = []
    template_ids_used: List[int] = []
    eps = 1e-12
    pair_cycle = 2 * pairs
    for slot_idx, spec in enumerate(row_specs):
        idx = int(spec["anchor_index"])
        tid = spec["tid"]
        candidate_order = int(spec["candidate_order"])
        top_ids = _top_ids_for_idx(idx)
        pair_pos = (candidate_order % pair_cycle) // 2
        template_sign = 1.0 if (candidate_order % 2 == 0) else -1.0
        template_id = int(top_ids[pair_pos])
        direction = np.asarray(zpia_bank[template_id], dtype=np.float64)
        direction_norm = float(np.linalg.norm(direction))
        d_min = float(margins[idx])
        gamma_requested = float(args.pia_gamma)
        if eta_safe is None:
            gamma_used = gamma_requested
            safe_upper_bound = float("inf")
            safe_radius_ratio = 1.0
        else:
            safe_upper_bound = float(eta_safe) * d_min / (direction_norm + eps)
            gamma_used = min(gamma_requested, safe_upper_bound)
            safe_radius = float(eta_safe) * d_min
            safe_radius_ratio = float(abs(gamma_used) * direction_norm / (safe_radius + eps)) if safe_radius > 0 else 0.0
        W_i = template_sign * gamma_used * direction
        response_abs = float(abs(np.dot(np.asarray(X_train_z[idx], dtype=np.float64), direction)))

        z_aug.append((np.asarray(X_train_z[idx], dtype=np.float64) + W_i).astype(np.float32))
        y_aug.append(int(y_arr[idx]))
        tid_aug.append(tid)
        w_slots.append(W_i.astype(np.float32))
        template_ids_used.append(template_id)
        slot_rows.append(
            {
                "anchor_index": idx,
                "tid": tid,
                "class_id": int(y_arr[idx]),
                "candidate_order": candidate_order,
                "slot_index": int(slot_idx),
                "template_pair_count": int(pairs),
                "zpia_template_id": template_id,
                "zpia_template_sign": float(template_sign),
                "zpia_template_rank": int(pair_pos),
                "zpia_template_response_abs": response_abs,
                "direction_id": template_id,
                "sign": float(template_sign),
                "gamma_requested": gamma_requested,
                "gamma_used": float(gamma_used),
                "direction_norm": direction_norm,
                "safe_upper_bound": float(safe_upper_bound),
                "safe_radius_ratio": float(safe_radius_ratio),
                "manifold_margin": d_min,
                "zpia_delta_norm": float(np.linalg.norm(W_i)),
            }
        )

    usage_stats = _template_usage_stats(template_ids_used)
    return {
        "z_aug": np.stack(z_aug).astype(np.float32) if z_aug else np.empty((0, X_train_z.shape[1]), dtype=np.float32),
        "y_aug": np.asarray(y_aug, dtype=np.int64),
        "tid_aug": np.asarray(tid_aug, dtype=object),
        "W_slots": np.stack(w_slots).astype(np.float32) if w_slots else np.empty((0, X_train_z.shape[1]), dtype=np.float32),
        "audit_rows": slot_rows,
        "multi_template_pairs": int(pairs),
        "template_usage_entropy": usage_stats["template_usage_entropy"],
        "top_template_concentration": usage_stats["top_template_concentration"],
    }


def _materialize_z_aug_out(
    *,
    z_aug: np.ndarray,
    y_aug: np.ndarray,
    tid_aug: np.ndarray,
    audit_rows: List[Dict[str, object]],
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    direction_bank_meta: Dict[str, object],
    effective_k: int,
    eta_safe: Optional[float],
    algo_name: str,
    engine_id: str,
    extra_meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    tid_to_rec = {record.tid: record for record in train_recs}
    aug_trials: List[Dict[str, object]] = []
    bridge_metrics: List[Dict[str, object]] = []
    out_rows: List[Dict[str, object]] = []
    for i in range(len(z_aug)):
        src = tid_to_rec[tid_aug[i]]
        sigma_aug = logvec_to_spd(z_aug[i], mean_log)
        x_aug, bridge_meta = bridge_single(
            torch.from_numpy(src.x_raw),
            torch.from_numpy(src.sigma_orig),
            torch.from_numpy(sigma_aug),
        )
        aug_trials.append({"x": x_aug.numpy(), "y": int(y_aug[i]), "tid": tid_aug[i]})
        bridge_metrics.append(bridge_meta)
        row = dict(audit_rows[i]) if i < len(audit_rows) else {}
        row.update(
            {
                "algo": algo_name,
                "engine_id": engine_id,
                "direction_bank_source": direction_bank_meta.get("bank_source", algo_name),
                "transport_error_fro": float(bridge_meta.get("transport_error_fro", 0.0)),
                "transport_error_logeuc": float(bridge_meta.get("transport_error_logeuc", 0.0)),
                "bridge_cond_A": float(bridge_meta.get("bridge_cond_A", 0.0)),
                "metric_preservation_error": float(bridge_meta.get("metric_preservation_error", 0.0)),
            }
        )
        out_rows.append(row)

    X_aug_raw = np.stack([trial["x"] for trial in aug_trials]) if aug_trials else None
    y_aug_np = np.asarray([trial["y"] for trial in aug_trials], dtype=np.int64) if aug_trials else None
    avg_bridge = pd.DataFrame(bridge_metrics).mean().to_dict() if bridge_metrics else {}
    safe_ratios = [float(row.get("safe_radius_ratio", 0.0)) for row in out_rows]
    margins = [float(row.get("manifold_margin", 0.0)) for row in out_rows]
    out = {
        "effective_k": int(effective_k),
        "z_aug": z_aug,
        "y_aug": y_aug,
        "tid_aug": tid_aug,
        "aug_trials": aug_trials,
        "X_aug_raw": X_aug_raw,
        "y_aug_np": y_aug_np,
        "tid_to_rec": tid_to_rec,
        "avg_bridge": avg_bridge,
        "audit_rows": out_rows,
        "direction_bank_meta": direction_bank_meta,
        "safe_radius_ratio_mean": float(np.mean(safe_ratios)) if safe_ratios else 0.0,
        "manifold_margin_mean": float(np.mean(margins)) if margins else 0.0,
        "eta_safe": eta_safe,
        "candidate_total_count": int(len(z_aug)),
        "aug_total_count": int(len(z_aug)),
        "aug_dataset": None,
    }
    if extra_meta:
        out.update(extra_meta)
    return out


def _build_act_realized_augmentations(
    *,
    args,
    seed: int,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    algo_override: Optional[str] = None,
    engine_id: Optional[str] = None,
) -> Dict[str, object]:
    algo_name = str(algo_override or args.algo)
    bank_out = _build_direction_bank_for_args(
        args=args,
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        algo_override=algo_name,
    )
    direction_bank = bank_out["bank"]
    direction_meta = bank_out["meta"]

    effective_k = int(direction_bank.shape[0])
    print(
        f"Requested K: {args.k_dir} | Effective K: {effective_k} | "
        f"Source: {direction_meta.get('bank_source', algo_name)} | Classes: {len(np.unique(y_train))}"
    )

    gamma_budget = np.full((effective_k,), float(args.pia_gamma), dtype=np.float64)
    direction_probs = active_direction_probs(gamma_budget, freeze_eps=0.01)
    eta_safe = None if args.disable_safe_step else 0.5
    tid_train = np.asarray([record.tid for record in train_recs], dtype=object)

    z_aug, y_aug, tid_aug, _, _, aug_meta = build_curriculum_aug_candidates(
        X_train_z,
        y_train,
        tid_train,
        direction_bank=direction_bank,
        direction_probs=direction_probs,
        gamma_by_dir=gamma_budget,
        multiplier=args.multiplier,
        seed=seed + 42,
        eta_safe=eta_safe,
    )

    tid_to_rec = {record.tid: record for record in train_recs}
    aug_trials: List[Dict[str, object]] = []
    bridge_metrics: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    candidate_rows = list(aug_meta.get("candidate_rows", []))
    for i in range(len(z_aug)):
        src = tid_to_rec[tid_aug[i]]
        sigma_aug = logvec_to_spd(z_aug[i], mean_log)
        x_aug, bridge_meta = bridge_single(
            torch.from_numpy(src.x_raw),
            torch.from_numpy(src.sigma_orig),
            torch.from_numpy(sigma_aug),
        )
        aug_trials.append({"x": x_aug.numpy(), "y": int(y_aug[i]), "tid": tid_aug[i]})
        bridge_metrics.append(bridge_meta)
        audit = candidate_rows[i].copy() if i < len(candidate_rows) else {
            "anchor_index": -1,
            "tid": tid_aug[i],
            "class_id": int(y_aug[i]),
            "candidate_order": int(i),
            "direction_id": -1,
            "sign": 0.0,
            "gamma_used": 0.0,
            "safe_radius_ratio": 0.0,
        }
        audit.update(
            {
                "algo": algo_name,
                "engine_id": str(engine_id or algo_name),
                "direction_bank_source": direction_meta.get("bank_source", algo_name),
                "transport_error_fro": float(bridge_meta.get("transport_error_fro", 0.0)),
                "transport_error_logeuc": float(bridge_meta.get("transport_error_logeuc", 0.0)),
                "bridge_cond_A": float(bridge_meta.get("bridge_cond_A", 0.0)),
                "metric_preservation_error": float(bridge_meta.get("metric_preservation_error", 0.0)),
            }
        )
        audit_rows.append(audit)

    X_aug_raw = np.stack([trial["x"] for trial in aug_trials]) if aug_trials else None
    y_aug_np = np.asarray([trial["y"] for trial in aug_trials], dtype=np.int64) if aug_trials else None
    avg_bridge = pd.DataFrame(bridge_metrics).mean().to_dict() if bridge_metrics else {}

    # --- V2: build on-the-fly dataset (only when --onthefly-aug is set) ---
    aug_dataset_out: Optional[ManifoldAugDataset] = None
    if getattr(args, "onthefly_aug", False) and len(z_aug) > 0:
        anchor_x_raws = [tid_to_rec[tid_aug[i]].x_raw for i in range(len(z_aug))]
        anchor_sigma_origs = [tid_to_rec[tid_aug[i]].sigma_orig for i in range(len(z_aug))]
        aug_dataset_out = ManifoldAugDataset(
            anchor_x_raws=anchor_x_raws,
            anchor_sigma_origs=anchor_sigma_origs,
            z_cands=z_aug,
            y_cands=y_aug,
            mean_log=mean_log,
        )

    return {
        "effective_k": effective_k,
        "direction_bank": direction_bank,
        "z_aug": z_aug,
        "y_aug": y_aug,
        "tid_aug": tid_aug,
        "aug_trials": aug_trials,
        "X_aug_raw": X_aug_raw,
        "y_aug_np": y_aug_np,
        "tid_to_rec": tid_to_rec,
        "avg_bridge": avg_bridge,
        "audit_rows": audit_rows,
        "direction_bank_meta": direction_meta,
        "safe_radius_ratio_mean": aug_meta.get("safe_radius_ratio_mean", 1.0),
        "manifold_margin_mean": aug_meta.get("manifold_margin_mean", 0.0),
        "eta_safe": eta_safe,
        "candidate_total_count": int(aug_meta.get("aug_total_count", len(aug_trials))),
        "aug_total_count": int(aug_meta.get("aug_total_count", len(aug_trials))),
        # V2 on-the-fly dataset (None when --onthefly-aug not set)
        "aug_dataset": aug_dataset_out,
    }


def _build_zpia_template_pool_aug_out(
    *,
    args,
    seed: int,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    algo_label: str,
    top1_only: bool,
) -> Dict[str, object]:
    bank_out = _build_direction_bank_for_args(
        args=_clone_args_with_updates(args, algo="zpia"),
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        algo_override="zpia",
    )
    zpia_bank = np.asarray(bank_out["bank"], dtype=np.float64)
    zpia_meta = dict(bank_out["meta"])
    eta_safe = None if args.disable_safe_step else 0.5
    slots = _build_top_response_template_slots(
        args=args,
        X_train_z=X_train_z,
        y_train=y_train,
        train_recs=train_recs,
        zpia_bank=zpia_bank,
        candidate_rows=None,
        top1_only=top1_only,
        eta_safe=eta_safe,
    )
    direction_meta = {
        "bank_source": algo_label,
        "zpia_meta": zpia_meta,
        "template_selection": "anchor_top_abs_response",
        "template_slot_policy": "dual_sign_fixed_budget",
        "multi_template_pairs": int(slots["multi_template_pairs"]),
    }
    return _materialize_z_aug_out(
        z_aug=slots["z_aug"],
        y_aug=slots["y_aug"],
        tid_aug=slots["tid_aug"],
        audit_rows=slots["audit_rows"],
        train_recs=train_recs,
        mean_log=mean_log,
        direction_bank_meta=direction_meta,
        effective_k=int(zpia_bank.shape[0]),
        eta_safe=eta_safe,
        algo_name=algo_label,
        engine_id=algo_label,
        extra_meta={
            "effective_k_zpia": int(zpia_bank.shape[0]),
            "multi_template_pairs": int(slots["multi_template_pairs"]),
            "template_usage_entropy": float(slots["template_usage_entropy"]),
            "top_template_concentration": float(slots["top_template_concentration"]),
        },
    )


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
        "safe_radius_ratio_mean": float(
            np.mean([row["safe_radius_ratio"] for row in fused_audit_rows])
        ) if fused_audit_rows else 0.0,
        "manifold_margin_mean": float(np.mean(shared_margin)) if shared_margin else 0.0,
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
        candidate_rows=list(aug_out_lraes.get("audit_rows", [])),
        top1_only=False,
        eta_safe=eta_safe,
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


def _template_response_diagnostics(
    *,
    X_train_z: np.ndarray,
    zpia_bank: np.ndarray,
) -> Dict[str, np.ndarray]:
    responses = np.abs(np.asarray(X_train_z, dtype=np.float64) @ np.asarray(zpia_bank, dtype=np.float64).T)
    n, k = responses.shape
    top1_ids = np.zeros((n,), dtype=np.int64)
    top2_ids = np.zeros((n,), dtype=np.int64)
    r1 = np.zeros((n,), dtype=np.float64)
    r2 = np.zeros((n,), dtype=np.float64)
    for i in range(n):
        order = np.lexsort((np.arange(k), -responses[i]))
        top1_ids[i] = int(order[0])
        top2_ids[i] = int(order[1]) if k > 1 else int(order[0])
        r1[i] = float(responses[i, top1_ids[i]])
        r2[i] = float(responses[i, top2_ids[i]]) if k > 1 else 0.0
    confidence = (r1 - r2) / (r1 + 1e-12)
    return {
        "top1_ids": top1_ids,
        "top2_ids": top2_ids,
        "top1_response": r1,
        "top2_response": r2,
        "template_confidence": confidence,
    }


def _build_progressive_round_aug_out(
    *,
    args,
    seed: int,
    round_idx: int,
    model_obj,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_train_z: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    zpia_bank: np.ndarray,
    zpia_meta: Dict[str, object],
    state: Dict[str, object],
    batch_size: int,
) -> Dict[str, object]:
    per_round_multiplier = int(args.per_round_multiplier)
    if per_round_multiplier <= 0 or per_round_multiplier % 2 != 0:
        raise ValueError("--per-round-multiplier must be a positive even integer.")
    eta_safe = None if args.disable_safe_step else 0.5
    allow_osf = args.algo in {
        "progressive_zpia_osf",
        "random_progressive_osf",
        "progressive_zpia_osf_highdose",
    }

    p_osf_used = float(state.get("p_osf", 0.0))
    beta_used = float(state.get("beta", getattr(args, "progressive_osf_beta_init", 0.25)))
    gamma_scale_used = float(state.get("gamma_scale", 1.0))
    acceptance_ema = np.asarray(state.get("anchor_acceptance_ema"), dtype=np.float64)
    if acceptance_ema.shape[0] != len(y_train):
        acceptance_ema = np.ones((len(y_train),), dtype=np.float64)

    orig_margins = _score_aug_margins(
        model_obj=model_obj,
        X_aug=X_train_raw,
        y_aug=y_train,
        device=args.device,
        batch_size=batch_size,
    )
    if orig_margins.shape[0] != len(y_train):
        orig_margins = np.zeros((len(y_train),), dtype=np.float64)

    tpl_diag = _template_response_diagnostics(X_train_z=X_train_z, zpia_bank=zpia_bank)
    tpl_conf = tpl_diag["template_confidence"]
    trigger_q = float(getattr(args, "progressive_trigger_quantile", 0.25))
    tpl_thresh = float(np.quantile(tpl_conf, trigger_q)) if tpl_conf.size else 0.0
    margin_thresh = float(np.quantile(orig_margins, trigger_q)) if orig_margins.size else 0.0
    eligible_real = (tpl_conf <= tpl_thresh) | (orig_margins <= margin_thresh)

    rng = np.random.default_rng(int(seed) + 10007 * (int(round_idx) + 1))
    if args.algo == "random_progressive_osf":
        eligible = np.zeros_like(eligible_real, dtype=bool)
        n_eligible = int(eligible_real.sum())
        if n_eligible > 0:
            chosen = rng.choice(len(eligible), size=n_eligible, replace=False)
            eligible[chosen] = True
    else:
        eligible = eligible_real.astype(bool)

    osf_anchor = np.zeros_like(eligible, dtype=bool)
    if allow_osf and eligible.any() and p_osf_used > 0.0:
        osf_anchor[eligible] = rng.random(int(eligible.sum())) < p_osf_used

    conservative_threshold = 0.5
    enable_conservative = not bool(getattr(args, "progressive_disable_conservative", False))
    conservative_anchor = (acceptance_ema < conservative_threshold) if enable_conservative else np.zeros_like(eligible, dtype=bool)
    # If an anchor has repeatedly rejected aug samples, be conservative first.
    osf_anchor = osf_anchor & (~conservative_anchor)

    args_slot = _clone_args_with_updates(
        args,
        algo="zpia",
        multiplier=per_round_multiplier,
        multi_template_pairs=1,
        pia_gamma=float(args.pia_gamma) * gamma_scale_used,
    )
    lraes_out = None
    candidate_rows = None
    if allow_osf:
        args_lraes = _clone_args_with_updates(
            args,
            algo="lraes",
            multiplier=per_round_multiplier,
            pia_gamma=float(args.pia_gamma),
        )
        lraes_out = _build_act_realized_augmentations(
            args=args_lraes,
            seed=int(seed) + 211 * (int(round_idx) + 1),
            X_train_z=X_train_z,
            y_train=y_train,
            train_recs=train_recs,
            mean_log=mean_log,
            algo_override="lraes",
            engine_id="progressive_lraes",
        )
        candidate_rows = list(lraes_out.get("audit_rows", []))

    slots = _build_top_response_template_slots(
        args=args_slot,
        X_train_z=X_train_z,
        y_train=y_train,
        train_recs=train_recs,
        zpia_bank=zpia_bank,
        candidate_rows=candidate_rows,
        top1_only=True,
        eta_safe=eta_safe,
    )

    z_final = np.asarray(slots["z_aug"], dtype=np.float64).copy()
    W_slots = np.asarray(slots["W_slots"], dtype=np.float64).copy()
    slot_rows = [dict(row) for row in slots["audit_rows"]]
    modes: List[str] = []
    osf_slot_indices: List[int] = []
    for i, row in enumerate(slot_rows):
        anchor_idx = int(row.get("anchor_index", -1))
        if anchor_idx < 0:
            mode = "zpia_top1"
        elif bool(osf_anchor[anchor_idx]):
            mode = "weak_osf"
            osf_slot_indices.append(i)
        elif bool(conservative_anchor[anchor_idx]):
            mode = "conservative_zpia"
            # Conservative zPIA uses the same top-response template at half step.
            z_orig = np.asarray(X_train_z[anchor_idx], dtype=np.float64)
            W_slots[i] *= 0.5
            z_final[i] = z_orig + W_slots[i]
            row["gamma_used"] = float(row.get("gamma_used", 0.0)) * 0.5
            row["safe_radius_ratio"] = float(row.get("safe_radius_ratio", 0.0)) * 0.5
            row["zpia_delta_norm"] = float(np.linalg.norm(W_slots[i]))
        else:
            mode = "zpia_top1"
        modes.append(mode)

    if osf_slot_indices:
        if lraes_out is None:
            raise RuntimeError("OSF slots were selected but no LRAES candidate stream was built.")
        if len(lraes_out["tid_aug"]) != len(slots["tid_aug"]):
            raise ValueError("progressive OSF requires aligned LRAES and zPIA slot counts.")
        if not np.array_equal(lraes_out["tid_aug"], slots["tid_aug"]):
            raise ValueError("progressive OSF requires aligned LRAES and zPIA slot tids.")
        if not np.array_equal(lraes_out["y_aug"], slots["y_aug"]):
            raise ValueError("progressive OSF requires aligned LRAES and zPIA slot labels.")

        z_o = torch.from_numpy(np.stack([lraes_out["tid_to_rec"][tid].z for tid in lraes_out["tid_aug"]]))
        z_l = torch.from_numpy(np.asarray(lraes_out["z_aug"], dtype=np.float64))
        W = torch.from_numpy(W_slots).to(dtype=z_o.dtype)
        U = z_l - z_o
        shared_margin: List[float] = []
        for idx, (row_l, row_z) in enumerate(zip(lraes_out["audit_rows"], slot_rows)):
            margin_l = float(row_l.get("manifold_margin", np.nan))
            margin_z = float(row_z.get("manifold_margin", np.nan))
            if (not np.isfinite(margin_l)) or (not np.isfinite(margin_z)):
                raise ValueError(f"progressive OSF requires finite manifold_margin at slot {idx}.")
            shared_margin.append(min(margin_l, margin_z))
        r_shared_np = float(eta_safe) * np.asarray(shared_margin, dtype=np.float64)
        r_shared = torch.from_numpy(r_shared_np).to(dtype=z_o.dtype).unsqueeze(-1)
        proj_out = _project_rank1_structure_out(W=W, U=U)
        rc4 = _apply_rc4_safe_governance(
            W=W,
            U=U,
            U_perp=proj_out["U_perp"],
            r_shared=r_shared,
            alpha=float(args.osf_alpha),
            beta=beta_used,
            kappa=float(getattr(args, "osf_kappa", 1.0)),
        )
        rc4["norm_proj"] = torch.norm(proj_out["proj"], p=2, dim=-1, keepdim=True)
        z_osf = z_o + rc4["delta_z"]
        for i in osf_slot_indices:
            z_final[i] = z_osf[i].numpy()
            row = slot_rows[i]
            row_l = lraes_out["audit_rows"][i]
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
            row.update(
                {
                    "lraes_direction_id": int(row_l.get("direction_id", -1)),
                    "lraes_sign": float(row_l.get("sign", 0.0)),
                    "lraes_gamma_used": float(row_l.get("gamma_used", 0.0)),
                    "lraes_delta_norm": norm_u,
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
                    "safe_radius_ratio": float(final_norm / (shared_r + 1e-12)) if shared_r > 0.0 else 0.0,
                    "manifold_margin": float(shared_margin[i]),
                }
            )

    for i, row in enumerate(slot_rows):
        anchor_idx = int(row.get("anchor_index", -1))
        row.update(
            {
                "round": int(round_idx) + 1,
                "mode": modes[i],
                "eligible": bool(eligible[anchor_idx]) if 0 <= anchor_idx < len(eligible) else False,
                "template_id": int(row.get("zpia_template_id", -1)),
                "template_confidence": float(tpl_conf[anchor_idx]) if 0 <= anchor_idx < len(tpl_conf) else 0.0,
                "orig_margin": float(orig_margins[anchor_idx]) if 0 <= anchor_idx < len(orig_margins) else 0.0,
                "anchor_acceptance_ema": float(acceptance_ema[anchor_idx]) if 0 <= anchor_idx < len(acceptance_ema) else 1.0,
                "gamma_used": float(row.get("gamma_used", 0.0)),
                "beta_used": beta_used if modes[i] == "weak_osf" else 0.0,
                "osf_risk_status": str(row.get("osf_risk_status", "none")),
            }
        )
        shared_r_default = float(eta_safe) * float(row.get("manifold_margin", 0.0)) if eta_safe is not None else 0.0
        row.setdefault("lraes_direction_id", -1)
        row.setdefault("lraes_sign", 0.0)
        row.setdefault("lraes_gamma_used", 0.0)
        row.setdefault("lraes_delta_norm", 0.0)
        row.setdefault("proj_on_template_norm", 0.0)
        row.setdefault("u_perp_norm", 0.0)
        row.setdefault("u_perp_norm_ratio", 0.0)
        row.setdefault("osf_r_shared", shared_r_default)
        row.setdefault("osf_alpha_eff", 0.0)
        row.setdefault("osf_structure_norm_raw", float(row.get("zpia_delta_norm", 0.0)))
        row.setdefault("osf_structure_norm_eff", float(row.get("zpia_delta_norm", 0.0)))
        row.setdefault("osf_risk_norm_restored", 0.0)
        row.setdefault("osf_risk_budget", 0.0)
        row.setdefault("osf_risk_scale", 0.0)
        row.setdefault("osf_final_norm", float(row.get("zpia_delta_norm", 0.0)))
        row.setdefault("osf_structure_overflow", False)

    direction_meta = {
        "bank_source": str(args.algo),
        "zpia_meta": zpia_meta,
        "template_selection": "anchor_top_abs_response",
        "progressive_policy": str(args.algo),
    }
    aug_out = _materialize_z_aug_out(
        z_aug=z_final.astype(np.float32),
        y_aug=slots["y_aug"],
        tid_aug=slots["tid_aug"],
        audit_rows=slot_rows,
        train_recs=train_recs,
        mean_log=mean_log,
        direction_bank_meta=direction_meta,
        effective_k=int(zpia_bank.shape[0]),
        eta_safe=eta_safe,
        algo_name=str(args.algo),
        engine_id=str(args.algo),
        extra_meta={
            "effective_k_zpia": int(zpia_bank.shape[0]),
            "multi_template_pairs": 1,
        },
    )

    aug_margins = _score_aug_margins(
        model_obj=model_obj,
        X_aug=aug_out.get("X_aug_raw"),
        y_aug=aug_out.get("y_aug_np"),
        device=args.device,
        batch_size=batch_size,
    )
    margin_upper = float(getattr(args, "progressive_margin_upper", 5.0))
    accepted = aug_margins > 0.0
    useful = (aug_margins > 0.0) & (aug_margins < margin_upper)
    easy = aug_margins >= margin_upper
    for i, row in enumerate(aug_out.get("audit_rows", [])):
        m = float(aug_margins[i]) if i < len(aug_margins) else 0.0
        row["aug_margin"] = m
        row["accepted"] = bool(accepted[i]) if i < len(accepted) else False
        row["useful"] = bool(useful[i]) if i < len(useful) else False
        row["easy"] = bool(easy[i]) if i < len(easy) else False

    ema_decay = 0.7
    rows_df = pd.DataFrame(aug_out.get("audit_rows", []))
    if not rows_df.empty and "anchor_index" in rows_df.columns:
        for anchor_idx, group in rows_df.groupby("anchor_index"):
            idx = int(anchor_idx)
            if 0 <= idx < acceptance_ema.shape[0]:
                anchor_acc = pd.to_numeric(group["accepted"], errors="coerce").fillna(0.0).mean()
                acceptance_ema[idx] = ema_decay * acceptance_ema[idx] + (1.0 - ema_decay) * float(anchor_acc)
    state["anchor_acceptance_ema"] = acceptance_ema

    mode_arr = np.asarray(modes, dtype=object)
    osf_mask = mode_arr == "weak_osf"
    conservative_mask = mode_arr == "conservative_zpia"
    zpia_mask = mode_arr == "zpia_top1"
    zpia_family_mask = ~osf_mask
    def _mask_mean(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.mean(values[mask])) if values.size and np.any(mask) else 0.0

    acc_zpia = _mask_mean(accepted.astype(np.float64), zpia_family_mask)
    acc_osf = _mask_mean(accepted.astype(np.float64), osf_mask)
    useful_zpia = _mask_mean(useful.astype(np.float64), zpia_family_mask)
    useful_osf = _mask_mean(useful.astype(np.float64), osf_mask)
    easy_zpia = _mask_mean(easy.astype(np.float64), zpia_family_mask)
    easy_osf = _mask_mean(easy.astype(np.float64), osf_mask)

    if allow_osf and np.any(osf_mask):
        diff = useful_osf - useful_zpia
        eta_p = 0.2
        p_next = float(np.clip(p_osf_used + eta_p * diff, 0.0, float(args.progressive_osf_p_max)))
        if diff > 0.05:
            beta_next = beta_used * 1.1
        elif diff < -0.05:
            beta_next = beta_used * 0.8
        else:
            beta_next = beta_used
        state["p_osf"] = float(np.clip(p_next, 0.0, float(args.progressive_osf_p_max)))
        state["beta"] = float(np.clip(beta_next, float(args.progressive_osf_beta_min), float(args.progressive_osf_beta_max)))
    elif allow_osf:
        state["p_osf"] = float(np.clip(p_osf_used * 0.95, 0.0, float(args.progressive_osf_p_max)))

    gamma_next = gamma_scale_used
    if acc_zpia < 0.5:
        gamma_next *= 0.9
    elif easy_zpia > 0.7:
        gamma_next *= 1.05
    state["gamma_scale"] = float(np.clip(gamma_next, 0.5, 1.25))

    cumulative_count = int(state.get("cumulative_aug_count", 0)) + int(len(aug_out.get("y_aug_np", [])))
    state["cumulative_aug_count"] = cumulative_count

    eligible_anchor_rate = float(np.mean(eligible)) if eligible.size else 0.0
    osf_given_eligible_rate = float(np.mean(osf_anchor[eligible])) if eligible.any() else 0.0
    trace_row = {
        "round": int(round_idx) + 1,
        "p_osf": p_osf_used,
        "beta": beta_used,
        "gamma_scale": gamma_scale_used,
        "eligible_anchor_rate": eligible_anchor_rate,
        "osf_given_eligible_rate": osf_given_eligible_rate,
        "actual_osf_rate": float(np.mean(osf_mask)) if mode_arr.size else 0.0,
        "n_zpia": int(np.sum(zpia_mask)),
        "n_osf": int(np.sum(osf_mask)),
        "n_conservative": int(np.sum(conservative_mask)),
        "acc_zpia": acc_zpia,
        "acc_osf": acc_osf,
        "useful_zpia": useful_zpia,
        "useful_osf": useful_osf,
        "easy_zpia_rate": easy_zpia,
        "easy_osf_rate": easy_osf,
        "margin_orig_mean": float(np.mean(orig_margins)) if orig_margins.size else 0.0,
        "margin_aug_mean": float(np.mean(aug_margins)) if aug_margins.size else 0.0,
        "active_aug_count": 0,
        "cumulative_aug_count": cumulative_count,
        "aug_exposure_count": 0,
    }
    return {
        "X_aug": aug_out.get("X_aug_raw"),
        "y_aug": aug_out.get("y_aug_np"),
        "audit_rows": aug_out.get("audit_rows", []),
        "trace_row": trace_row,
        "aug_out": aug_out,
    }


def _run_analysis_probe(
    *,
    args,
    model_obj,
    tid_aug: np.ndarray,
    aug_trials: List[Dict[str, object]],
    tid_to_rec: Dict[str, TrialRecord],
) -> Dict[str, float]:
    alignment_metrics = {"host_geom_cosine_mean": 0.0, "host_conflict_rate": 0.0}
    if not args.theory_diagnostics or args.model == "minirocket" or model_obj is None or not aug_trials:
        return alignment_metrics

    print("Running theory diagnostics...")
    with torch.enable_grad():
        aligns = []
        probe_idx = np.random.choice(len(aug_trials), min(20, len(aug_trials)), replace=False)
        for idx in probe_idx:
            src = tid_to_rec[tid_aug[idx]]
            x_orig = torch.from_numpy(src.x_raw).unsqueeze(0).float()
            y_orig = torch.tensor([src.y]).long()
            x_aug = torch.from_numpy(aug_trials[idx]["x"]).unsqueeze(0).float()
            aligns.append(compute_gradient_alignment(model_obj, x_orig, y_orig, x_aug, device=args.device))

        if aligns:
            alignment_metrics["host_geom_cosine_mean"] = float(np.mean([probe["alignment_cosine"] for probe in aligns]))
            alignment_metrics["host_conflict_rate"] = float(np.mean([probe["is_conflict"] for probe in aligns]))
    return alignment_metrics


def _run_act_pipeline(
    *,
    args,
    seed: int,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    X_train_z: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
) -> Dict[str, object]:
    aug_out = _build_act_realized_augmentations(
        args=args,
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        train_recs=train_recs,
        mean_log=mean_log,
    )

    print("Fitting baseline...")
    res_base = _fit_host_model(
        args=args,
        X_tr=X_train_raw,
        y_tr=y_train,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        return_model_obj=args.theory_diagnostics,
        loader_seed=seed,
    )

    if aug_out["aug_trials"]:
        X_mix = np.concatenate([X_train_raw, aug_out["X_aug_raw"]], axis=0)
        y_mix = np.concatenate([y_train, aug_out["y_aug_np"]], axis=0)
    else:
        X_mix = X_train_raw
        y_mix = y_train

    alignment_metrics = _run_analysis_probe(
        args=args,
        model_obj=res_base.get("model_obj"),
        tid_aug=aug_out["tid_aug"],
        aug_trials=aug_out["aug_trials"],
        tid_to_rec=aug_out["tid_to_rec"],
    )

    print(f"Fitting ACT model ({len(X_mix)} samples)...")
    res_act = _fit_host_model(
        args=args,
        X_tr=X_mix,
        y_tr=y_mix,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        return_model_obj=False,
        loader_seed=seed,
    )

    return {
        "res_base": res_base,
        "res_act": res_act,
        "avg_bridge": aug_out["avg_bridge"],
        "safe_radius_ratio_mean": aug_out["safe_radius_ratio_mean"],
        "manifold_margin_mean": aug_out["manifold_margin_mean"],
        "host_geom_cosine_mean": alignment_metrics["host_geom_cosine_mean"],
        "host_conflict_rate": alignment_metrics["host_conflict_rate"],
        "candidate_total_count": aug_out["candidate_total_count"],
        "aug_total_count": aug_out["aug_total_count"],
        "effective_k": aug_out["effective_k"],
        "direction_bank_meta": aug_out.get("direction_bank_meta", {}),
        "audit_rows": aug_out.get("audit_rows", []),
        "viz_payload": {
            "Z_orig": X_train_z,
            "Z_aug": aug_out["z_aug"],
            "y_aug": aug_out["y_aug"],
            "X_aug_raw": np.stack([trial["x"] for trial in aug_out["aug_trials"][:20]]) if aug_out["aug_trials"] else None,
        },
    }


def _run_act_zpia_template_pool_pipeline(
    *,
    args,
    seed: int,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    X_train_z: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    algo_label: str,
    top1_only: bool,
) -> Dict[str, object]:
    print("Fitting baseline...")
    res_base = _fit_host_model(
        args=args,
        X_tr=X_train_raw,
        y_tr=y_train,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        return_model_obj=args.theory_diagnostics,
        loader_seed=seed,
    )

    aug_out = _build_zpia_template_pool_aug_out(
        args=args,
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        train_recs=train_recs,
        mean_log=mean_log,
        algo_label=algo_label,
        top1_only=top1_only,
    )

    if aug_out["aug_trials"]:
        X_mix = np.concatenate([X_train_raw, aug_out["X_aug_raw"]], axis=0)
        y_mix = np.concatenate([y_train, aug_out["y_aug_np"]], axis=0)
    else:
        X_mix = X_train_raw
        y_mix = y_train

    alignment_metrics = _run_analysis_probe(
        args=args,
        model_obj=res_base.get("model_obj"),
        tid_aug=aug_out["tid_aug"],
        aug_trials=aug_out["aug_trials"],
        tid_to_rec=aug_out["tid_to_rec"],
    )

    print(f"Fitting {algo_label} core model ({len(X_mix)} samples)...")
    res_act = _fit_host_model(
        args=args,
        X_tr=X_mix,
        y_tr=y_mix,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        return_model_obj=False,
        loader_seed=seed,
    )

    return {
        "res_base": res_base,
        "res_act": res_act,
        "avg_bridge": aug_out["avg_bridge"],
        "safe_radius_ratio_mean": aug_out["safe_radius_ratio_mean"],
        "manifold_margin_mean": aug_out["manifold_margin_mean"],
        "host_geom_cosine_mean": alignment_metrics["host_geom_cosine_mean"],
        "host_conflict_rate": alignment_metrics["host_conflict_rate"],
        "candidate_total_count": aug_out["candidate_total_count"],
        "aug_total_count": aug_out["aug_total_count"],
        "effective_k": aug_out["effective_k"],
        "effective_k_zpia": aug_out.get("effective_k_zpia", 0),
        "direction_bank_meta": aug_out.get("direction_bank_meta", {}),
        "audit_rows": aug_out.get("audit_rows", []),
        **_summarize_multitemplate_audit_rows(aug_out.get("audit_rows", [])),
        "multi_template_pairs": int(aug_out.get("multi_template_pairs", 0)),
        "viz_payload": {
            "Z_orig": X_train_z,
            "Z_aug": aug_out["z_aug"],
            "y_aug": aug_out["y_aug"],
            "X_aug_raw": np.stack([trial["x"] for trial in aug_out["aug_trials"][:20]])
            if aug_out["aug_trials"]
            else None,
        },
    }


def _run_act_progressive_pipeline(
    *,
    args,
    seed: int,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    X_train_z: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
) -> Dict[str, object]:
    print("Fitting baseline...")
    res_base = _fit_host_model(
        args=args,
        X_tr=X_train_raw,
        y_tr=y_train,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        return_model_obj=False,
        loader_seed=seed,
    )

    bank_out = _build_direction_bank_for_args(
        args=_clone_args_with_updates(args, algo="zpia"),
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        algo_override="zpia",
    )
    zpia_bank = np.asarray(bank_out["bank"], dtype=np.float64)
    zpia_meta = dict(bank_out["meta"])
    if zpia_bank.ndim != 2 or zpia_bank.shape[0] <= 0:
        raise ValueError(f"{args.algo} requires a non-empty zPIA direction bank.")

    state: Dict[str, object] = {
        "p_osf": 0.0
        if args.algo in {
            "progressive_zpia_only",
            "progressive_zpia_only_pure",
            "progressive_zpia_pure_cumulative",
            "progressive_zpia_exposure_matched",
        }
        else float(args.progressive_osf_p_init),
        "beta": float(args.progressive_osf_beta_init),
        "gamma_scale": 1.0,
        "anchor_acceptance_ema": np.ones((len(y_train),), dtype=np.float64),
        "cumulative_aug_count": 0,
    }

    def _round_callback(ctx: Dict[str, object]) -> Dict[str, object]:
        return _build_progressive_round_aug_out(
            args=args,
            seed=seed,
            round_idx=int(ctx["round"]),
            model_obj=ctx["model"],
            X_train_raw=X_train_raw,
            y_train=y_train,
            X_train_z=X_train_z,
            train_recs=train_recs,
            mean_log=mean_log,
            zpia_bank=zpia_bank,
            zpia_meta=zpia_meta,
            state=state,
            batch_size=batch_size,
        )

    print(
        f"Fitting {args.algo} progressive model "
        f"(warmup={args.progressive_warmup_epochs}, rounds={args.progressive_rounds}, "
        f"round_epochs={args.progressive_round_epochs})..."
    )
    res_act = fit_eval_resnet1d_progressive_aug_ce(
        X_train_raw,
        y_train,
        X_val_raw,
        y_val,
        X_test_raw,
        y_test,
        round_callback=_round_callback,
        rounds=int(args.progressive_rounds),
        warmup_epochs=int(args.progressive_warmup_epochs),
        round_epochs=int(args.progressive_round_epochs),
        pool_keep_rounds=int(args.progressive_pool_keep_rounds),
        lr=lr,
        batch_size=batch_size,
        device=args.device,
        loader_seed=seed,
    )

    audit_rows = list(res_act.get("progressive_audit_rows", []))
    trace_rows = list(res_act.get("progressive_trace", []))
    bridge_cols = [
        "transport_error_fro",
        "transport_error_logeuc",
        "bridge_cond_A",
        "metric_preservation_error",
    ]
    avg_bridge: Dict[str, float] = {}
    if audit_rows:
        df_audit = pd.DataFrame(audit_rows)
        for col in bridge_cols:
            if col in df_audit.columns:
                vals = pd.to_numeric(df_audit[col], errors="coerce").dropna()
                avg_bridge[col] = float(vals.mean()) if not vals.empty else 0.0
    progressive_summary = _summarize_progressive_audit_rows(audit_rows)

    return {
        "res_base": res_base,
        "res_act": res_act,
        "avg_bridge": avg_bridge,
        "safe_radius_ratio_mean": float(
            pd.to_numeric(pd.DataFrame(audit_rows).get("safe_radius_ratio"), errors="coerce").dropna().mean()
        ) if audit_rows and "safe_radius_ratio" in pd.DataFrame(audit_rows).columns else 1.0,
        "manifold_margin_mean": float(
            pd.to_numeric(pd.DataFrame(audit_rows).get("manifold_margin"), errors="coerce").dropna().mean()
        ) if audit_rows and "manifold_margin" in pd.DataFrame(audit_rows).columns else 0.0,
        "host_geom_cosine_mean": 0.0,
        "host_conflict_rate": 0.0,
        "candidate_total_count": int(len(audit_rows)),
        "aug_total_count": int(len(audit_rows)),
        "effective_k": int(zpia_bank.shape[0]),
        "effective_k_zpia": int(zpia_bank.shape[0]),
        "direction_bank_meta": {
            "bank_source": str(args.algo),
            "zpia_meta": zpia_meta,
            "template_selection": "anchor_top_abs_response",
            "progressive_policy": str(args.algo),
        },
        "audit_rows": audit_rows,
        "progressive_trace": trace_rows,
        "progressive_rounds": int(args.progressive_rounds),
        "per_round_multiplier": int(args.per_round_multiplier),
        "progressive_cumulative_aug_count": int(state.get("cumulative_aug_count", len(audit_rows))),
        "progressive_active_aug_count_mean": float(res_act.get("progressive_active_aug_count_mean", 0.0)),
        "progressive_aug_exposure_count": int(res_act.get("progressive_aug_exposure_count", 0)),
        "progressive_optimizer_steps": int(res_act.get("progressive_optimizer_steps", 0)),
        "progressive_osf_p_final": float(state.get("p_osf", 0.0)),
        "progressive_beta_final": float(state.get("beta", 0.0)),
        "progressive_gamma_scale_final": float(state.get("gamma_scale", 1.0)),
        **progressive_summary,
        **_summarize_osf_audit_rows(audit_rows),
        "viz_payload": {
            "Z_orig": X_train_z,
            "Z_aug": np.empty((0, X_train_z.shape[1]), dtype=np.float32),
            "y_aug": np.empty((0,), dtype=np.int64),
            "X_aug_raw": None,
        },
    }


def _run_mba_feedback_pipeline(
    *,
    args,
    seed: int,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    X_train_z: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
) -> Dict[str, object]:
    print("Fitting baseline...")
    res_base = _fit_host_model(
        args=args,
        X_tr=X_train_raw,
        y_tr=y_train,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        return_model_obj=True,
        loader_seed=seed,
    )

    aug_out = _build_act_realized_augmentations(
        args=args,
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        train_recs=train_recs,
        mean_log=mean_log,
    )
    score_out = _attach_feedback_scores_to_aug_out(
        aug_out=aug_out,
        model_obj=res_base.get("model_obj"),
        device=args.device,
        batch_size=batch_size,
        feedback_margin_temperature=args.feedback_margin_temperature,
    )
    margins = score_out["margins"]
    weights = score_out["weights"]

    # Build TauScheduler if requested
    _tau_sched: Optional[TauScheduler] = None
    if getattr(args, "aug_weight_mode", "sigmoid") != "sigmoid" or getattr(args, "tau_max", None) is not None:
        _tau_sched = TauScheduler(
            total_epochs=epochs,
            tau_max=getattr(args, "tau_max", 2.0),
            tau_min=getattr(args, "tau_min", 0.1),
            warmup_ratio=getattr(args, "tau_warmup_ratio", 0.3),
        )

    print(f"Fitting MBA feedback model ({len(y_train)} orig + {len(aug_out['y_aug_np']) if aug_out['y_aug_np'] is not None else 0} aug stream)...")
    res_act = _fit_host_model_weighted_aug_ce(
        args=args,
        X_tr=X_train_raw,
        y_tr=y_train,
        X_aug=aug_out["X_aug_raw"] if aug_out["aug_dataset"] is None else None,
        y_aug=aug_out["y_aug_np"] if aug_out["aug_dataset"] is None else None,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        loader_seed=seed,
        aug_dataset=aug_out["aug_dataset"],
        tau_scheduler=_tau_sched,
    )

    return {
        "res_base": res_base,
        "res_act": res_act,
        "avg_bridge": aug_out["avg_bridge"],
        "audit_rows": aug_out.get("audit_rows", []),
        "direction_bank_meta": aug_out.get("direction_bank_meta", {}),
        "safe_radius_ratio_mean": aug_out["safe_radius_ratio_mean"],
        "manifold_margin_mean": aug_out["manifold_margin_mean"],
        "host_geom_cosine_mean": 0.0,
        "host_conflict_rate": 0.0,
        "candidate_total_count": aug_out["candidate_total_count"],
        "aug_total_count": aug_out["aug_total_count"],
        "effective_k": aug_out["effective_k"],
        "feedback_weight_mean": float(np.mean(weights)) if weights.size else 0.0,
        "feedback_weight_std": float(np.std(weights)) if weights.size else 0.0,
        "last_aug_margin_mean": float(np.mean(margins)) if margins.size else 0.0,
        "viz_payload": {
            "Z_orig": X_train_z,
            "Z_aug": aug_out["z_aug"],
            "y_aug": aug_out["y_aug"],
            "X_aug_raw": np.stack([trial["x"] for trial in aug_out["aug_trials"][:20]]) if aug_out["aug_trials"] else None,
        },
}


def _run_act_fused_core_pipeline(
    *,
    args,
    seed: int,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    X_train_z: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    fusion_mode: str,
    algo_label: str,
) -> Dict[str, object]:
    print("Fitting baseline...")
    res_base = _fit_host_model(
        args=args,
        X_tr=X_train_raw,
        y_tr=y_train,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        return_model_obj=args.theory_diagnostics,
        loader_seed=seed,
    )

    fused_aug_out = _build_rc4_fused_aug_out(
        args=args,
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        train_recs=train_recs,
        mean_log=mean_log,
        model_obj=res_base.get("model_obj"),
        batch_size=batch_size,
        include_feedback_scores=False,
        algo_label=algo_label,
        fusion_mode=fusion_mode,
    )

    if fused_aug_out["aug_trials"]:
        X_mix = np.concatenate([X_train_raw, fused_aug_out["X_aug_raw"]], axis=0)
        y_mix = np.concatenate([y_train, fused_aug_out["y_aug_np"]], axis=0)
    else:
        X_mix = X_train_raw
        y_mix = y_train

    alignment_metrics = _run_analysis_probe(
        args=args,
        model_obj=res_base.get("model_obj"),
        tid_aug=fused_aug_out["tid_aug"],
        aug_trials=fused_aug_out["aug_trials"],
        tid_to_rec=fused_aug_out["tid_to_rec"],
    )

    print(f"Fitting {algo_label} core model ({len(X_mix)} samples)...")
    res_act = _fit_host_model(
        args=args,
        X_tr=X_mix,
        y_tr=y_mix,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        return_model_obj=False,
        loader_seed=seed,
    )

    return {
        "res_base": res_base,
        "res_act": res_act,
        "avg_bridge": fused_aug_out["avg_bridge"],
        "safe_radius_ratio_mean": fused_aug_out["safe_radius_ratio_mean"],
        "manifold_margin_mean": fused_aug_out["manifold_margin_mean"],
        "host_geom_cosine_mean": alignment_metrics["host_geom_cosine_mean"],
        "host_conflict_rate": alignment_metrics["host_conflict_rate"],
        "candidate_total_count": fused_aug_out["candidate_total_count"],
        "aug_total_count": fused_aug_out["aug_total_count"],
        "effective_k": fused_aug_out["effective_k"],
        "effective_k_lraes": fused_aug_out.get("effective_k_lraes", 0),
        "effective_k_zpia": fused_aug_out.get("effective_k_zpia", 0),
        "direction_bank_meta": fused_aug_out.get("direction_bank_meta", {}),
        "audit_rows": fused_aug_out.get("audit_rows", []),
        **_summarize_osf_audit_rows(fused_aug_out.get("audit_rows", [])),
        **(_summarize_spectral_audit_rows(fused_aug_out.get("audit_rows", [])) if fusion_mode == "spectral_osf" else {}),
        "viz_payload": {
            "Z_orig": X_train_z,
            "Z_aug": fused_aug_out["z_aug"],
            "y_aug": fused_aug_out["y_aug"],
            "X_aug_raw": np.stack([trial["x"] for trial in fused_aug_out["aug_trials"][:20]])
            if fused_aug_out["aug_trials"]
            else None,
        },
    }


def _run_act_rc4_fused_pipeline(
    *,
    args,
    seed: int,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    X_train_z: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
) -> Dict[str, object]:
    return _run_act_fused_core_pipeline(
        args=args,
        seed=seed,
        X_train_raw=X_train_raw,
        y_train=y_train,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        X_train_z=X_train_z,
        train_recs=train_recs,
        mean_log=mean_log,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        fusion_mode="rank1_osf",
        algo_label="rc4_fused",
    )


def _run_act_spectral_osf_pipeline(
    *,
    args,
    seed: int,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    X_train_z: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
) -> Dict[str, object]:
    return _run_act_fused_core_pipeline(
        args=args,
        seed=seed,
        X_train_raw=X_train_raw,
        y_train=y_train,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        X_train_z=X_train_z,
        train_recs=train_recs,
        mean_log=mean_log,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        fusion_mode="spectral_osf",
        algo_label="spectral_osf",
    )


def _run_act_rc4_multiz_fused_pipeline(
    *,
    args,
    seed: int,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    X_train_z: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
) -> Dict[str, object]:
    print("Fitting baseline...")
    res_base = _fit_host_model(
        args=args,
        X_tr=X_train_raw,
        y_tr=y_train,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        return_model_obj=args.theory_diagnostics,
        loader_seed=seed,
    )

    aug_out = _build_rc4_multiz_fused_aug_out(
        args=args,
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        train_recs=train_recs,
        mean_log=mean_log,
        algo_label="rc4_multiz_fused",
    )

    if aug_out["aug_trials"]:
        X_mix = np.concatenate([X_train_raw, aug_out["X_aug_raw"]], axis=0)
        y_mix = np.concatenate([y_train, aug_out["y_aug_np"]], axis=0)
    else:
        X_mix = X_train_raw
        y_mix = y_train

    alignment_metrics = _run_analysis_probe(
        args=args,
        model_obj=res_base.get("model_obj"),
        tid_aug=aug_out["tid_aug"],
        aug_trials=aug_out["aug_trials"],
        tid_to_rec=aug_out["tid_to_rec"],
    )

    print(f"Fitting rc4_multiz_fused core model ({len(X_mix)} samples)...")
    res_act = _fit_host_model(
        args=args,
        X_tr=X_mix,
        y_tr=y_mix,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        return_model_obj=False,
        loader_seed=seed,
    )

    return {
        "res_base": res_base,
        "res_act": res_act,
        "avg_bridge": aug_out["avg_bridge"],
        "safe_radius_ratio_mean": aug_out["safe_radius_ratio_mean"],
        "manifold_margin_mean": aug_out["manifold_margin_mean"],
        "host_geom_cosine_mean": alignment_metrics["host_geom_cosine_mean"],
        "host_conflict_rate": alignment_metrics["host_conflict_rate"],
        "candidate_total_count": aug_out["candidate_total_count"],
        "aug_total_count": aug_out["aug_total_count"],
        "effective_k": aug_out["effective_k"],
        "effective_k_lraes": aug_out.get("effective_k_lraes", 0),
        "effective_k_zpia": aug_out.get("effective_k_zpia", 0),
        "direction_bank_meta": aug_out.get("direction_bank_meta", {}),
        "audit_rows": aug_out.get("audit_rows", []),
        **_summarize_osf_audit_rows(aug_out.get("audit_rows", [])),
        **_summarize_multitemplate_audit_rows(aug_out.get("audit_rows", [])),
        "multi_template_pairs": int(aug_out.get("multi_template_pairs", 0)),
        "viz_payload": {
            "Z_orig": X_train_z,
            "Z_aug": aug_out["z_aug"],
            "y_aug": aug_out["y_aug"],
            "X_aug_raw": np.stack([trial["x"] for trial in aug_out["aug_trials"][:20]])
            if aug_out["aug_trials"]
            else None,
        },
    }


def _run_mba_feedback_adaptive_pipeline(
    *,
    args,
    seed: int,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    X_train_z: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
) -> Dict[str, object]:
    print("Fitting baseline...")
    res_base = _fit_host_model(
        args=args,
        X_tr=X_train_raw,
        y_tr=y_train,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        return_model_obj=True,
        loader_seed=seed,
    )

    if getattr(args, "direction_bank_source", "lraes") == "orthogonal_fusion":
        print("Executing Orthogonal Subspace Fusion (Sprint 5)...")
        aug_out_lraes = _build_rc4_fused_aug_out(
            args=args,
            seed=seed,
            X_train_z=X_train_z,
            y_train=y_train,
            train_recs=train_recs,
            mean_log=mean_log,
            model_obj=res_base.get("model_obj"),
            batch_size=batch_size,
            include_feedback_scores=True,
            algo_label="adaptive",
        )
        aug_out_zpia = {
            "X_aug_raw": None,
            "y_aug_np": None,
            "aug_dataset": None,
            "aug_total_count": 0,
            "candidate_total_count": 0,
            "avg_bridge": {},
            "audit_rows": [],
            "direction_bank_meta": {},
            "safe_radius_ratio_mean": 0.0,
            "manifold_margin_mean": 0.0,
            "eta_safe": aug_out_lraes.get("eta_safe", 0.5),
            "effective_k": 0,
            "z_aug": np.empty((0, aug_out_lraes["z_aug"].shape[-1]), dtype=np.float32),
            "y_aug": np.empty((0,), dtype=np.int64),
        }
        print("RC-4 fusion complete. Continuing with single fused stream.")
    else:
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
        for engine_name, aug_out in [("lraes", aug_out_lraes), ("zpia", aug_out_zpia)]:
            _attach_feedback_scores_to_aug_out(
                aug_out=aug_out,
                model_obj=res_base.get("model_obj"),
                device=args.device,
                batch_size=batch_size,
                feedback_margin_temperature=args.feedback_margin_temperature,
                engine_id=engine_name,
            )

    print(
        f"Fitting adaptive MBA feedback model "
        f"({len(y_train)} orig + {int(aug_out_lraes['aug_total_count'])} lraes + {int(aug_out_zpia['aug_total_count'])} zpia aug)..."
    )


    # Build TauScheduler if Sprint 2 is activated
    _tau_sched_adaptive: Optional[TauScheduler] = None
    if getattr(args, "aug_weight_mode", "sigmoid") != "sigmoid" or getattr(args, "tau_max", None) is not None:
        _tau_sched_adaptive = TauScheduler(
            total_epochs=epochs,
            tau_max=getattr(args, "tau_max", 2.0),
            tau_min=getattr(args, "tau_min", 0.1),
            warmup_ratio=getattr(args, "tau_warmup_ratio", 0.3),
        )

    res_act = _fit_host_model_adaptive_aug_ce(
        args=args,
        X_tr=X_train_raw,
        y_tr=y_train,
        X_aug_lraes=aug_out_lraes["X_aug_raw"] if aug_out_lraes["aug_dataset"] is None else None,
        y_aug_lraes=aug_out_lraes["y_aug_np"] if aug_out_lraes["aug_dataset"] is None else None,
        X_aug_zpia=aug_out_zpia["X_aug_raw"] if aug_out_zpia["aug_dataset"] is None else None,
        y_aug_zpia=aug_out_zpia["y_aug_np"] if aug_out_zpia["aug_dataset"] is None else None,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        loader_seed=seed,
        aug_dataset_lraes=aug_out_lraes["aug_dataset"],
        aug_dataset_zpia=aug_out_zpia["aug_dataset"],
        tau_scheduler=_tau_sched_adaptive,
    )

    def _weighted_bridge_mean(key: str) -> float:
        count_l = float(aug_out_lraes.get("aug_total_count", 0))
        count_z = float(aug_out_zpia.get("aug_total_count", 0))
        total = count_l + count_z
        if total <= 0.0:
            return 0.0
        return (
            count_l * float(aug_out_lraes.get("avg_bridge", {}).get(key, 0.0))
            + count_z * float(aug_out_zpia.get("avg_bridge", {}).get(key, 0.0))
        ) / total

    avg_bridge = {
        "transport_error_fro": _weighted_bridge_mean("transport_error_fro"),
        "transport_error_logeuc": _weighted_bridge_mean("transport_error_logeuc"),
        "bridge_cond_A": _weighted_bridge_mean("bridge_cond_A"),
        "metric_preservation_error": _weighted_bridge_mean("metric_preservation_error"),
    }
    audit_rows = list(aug_out_lraes.get("audit_rows", [])) + list(aug_out_zpia.get("audit_rows", []))
    osf_summary = _summarize_osf_audit_rows(audit_rows)

    def _weighted_engine_mean(key: str) -> float:
        count_l = float(aug_out_lraes.get("aug_total_count", 0))
        count_z = float(aug_out_zpia.get("aug_total_count", 0))
        total = count_l + count_z
        if total <= 0.0:
            return 0.0
        return (
            count_l * float(aug_out_lraes.get(key, 0.0))
            + count_z * float(aug_out_zpia.get(key, 0.0))
        ) / total

    return {
        "res_base": res_base,
        "res_act": res_act,
        "avg_bridge": avg_bridge,
        "audit_rows": audit_rows,
        "direction_bank_meta": {
            "bank_source": "orthogonal_fusion"
            if getattr(args, "direction_bank_source", "lraes") == "orthogonal_fusion"
            else "adaptive_dual",
            "engine_sources": [
                aug_out_lraes.get("direction_bank_meta", {}).get("bank_source", "lraes"),
                aug_out_zpia.get("direction_bank_meta", {}).get("bank_source", "zpia_telm2"),
            ],
            "lraes_meta": aug_out_lraes.get("direction_bank_meta", {}),
            "zpia_meta": aug_out_zpia.get("direction_bank_meta", {}),
        },
        "safe_radius_ratio_mean": float(_weighted_engine_mean("safe_radius_ratio_mean")),
        "manifold_margin_mean": float(_weighted_engine_mean("manifold_margin_mean")),
        "host_geom_cosine_mean": 0.0,
        "host_conflict_rate": 0.0,
        "candidate_total_count": int(aug_out_lraes.get("candidate_total_count", 0)) + int(
            aug_out_zpia.get("candidate_total_count", 0)
        ),
        "aug_total_count": int(aug_out_lraes.get("aug_total_count", 0)) + int(aug_out_zpia.get("aug_total_count", 0)),
        "effective_k": int(max(aug_out_lraes.get("effective_k", 0), aug_out_zpia.get("effective_k", 0))),
        "effective_k_lraes": int(aug_out_lraes.get("effective_k", 0)),
        "effective_k_zpia": int(aug_out_zpia.get("effective_k", 0)),
        **osf_summary,
        "router_trace": list(res_act.get("router_trace", [])),
        "viz_payload": {
            "Z_orig": X_train_z,
            "Z_aug": np.concatenate([aug_out_lraes["z_aug"], aug_out_zpia["z_aug"]], axis=0),
            "y_aug": np.concatenate([aug_out_lraes["y_aug"], aug_out_zpia["y_aug"]], axis=0),
            "X_aug_raw": np.concatenate(
                [
                    aug_out_lraes["X_aug_raw"][:10] if aug_out_lraes["X_aug_raw"] is not None else np.empty((0,) + X_train_raw.shape[1:], dtype=np.float32),
                    aug_out_zpia["X_aug_raw"][:10] if aug_out_zpia["X_aug_raw"] is not None else np.empty((0,) + X_train_raw.shape[1:], dtype=np.float32),
                ],
                axis=0,
            ),
        },
    }


def run_experiment(dataset_name, args):
    print(f"\n>>>> Dataset: {dataset_name} | Model: {args.model} <<<<")
    try:
        all_trials = load_trials_for_dataset(dataset_name)
    except Exception as exc:
        print(f"Failed to load {dataset_name}: {exc}")
        return [
            {
                "dataset": dataset_name,
                "seed": -1,
                "status": "failed",
                "fail_reason": str(exc),
                "requested_k_dir": args.k_dir,
                "effective_k_dir": 0,
                "algo": args.algo,
                "model": args.model,
                "pipeline": "act" if args.pipeline == "mba" else args.pipeline,
            }
        ]

    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    patience = args.patience

    if args.host_config != "none":
        if args.host_config == "resnet1d_default":
            epochs, lr, batch_size, patience = 30, 1e-3, 64, 10
        elif args.host_config == "patchtst_default":
            epochs, lr, batch_size, patience = 100, 5e-4, 64, 15
        elif args.host_config == "timesnet_default":
            epochs, lr, batch_size, patience = 100, 5e-4, 32, 15

    results = []
    seeds = [int(seed) for seed in args.seeds.split(",")]
    for seed in seeds:
        print(f"Seed {seed}...")
        try:
            train_trials, test_trials, val_trials = make_trial_split(all_trials, seed=seed, val_ratio=args.val_ratio)
            train_recs, mean_log = _build_trial_records(train_trials)
            test_recs, _ = _build_trial_records(test_trials)
            val_recs, _ = _build_trial_records(val_trials)

            X_train_raw = np.stack([record.x_raw for record in train_recs])
            y_train = np.asarray([record.y for record in train_recs], dtype=np.int64)
            X_test_raw = np.stack([record.x_raw for record in test_recs])
            y_test = np.asarray([record.y for record in test_recs], dtype=np.int64)

            X_val_raw, y_val = None, None
            if val_recs:
                X_val_raw = np.stack([record.x_raw for record in val_recs])
                y_val = np.asarray([record.y for record in val_recs], dtype=np.int64)

            X_train_z = np.stack([record.z for record in train_recs])
            if args.algo in {
                "progressive_zpia_only",
                "progressive_zpia_osf",
                "random_progressive_osf",
                "progressive_zpia_only_pure",
                "progressive_zpia_pure_cumulative",
                "progressive_zpia_exposure_matched",
                "progressive_zpia_osf_highdose",
            }:
                pipeline_out = _run_act_progressive_pipeline(
                    args=args,
                    seed=seed,
                    X_train_raw=X_train_raw,
                    y_train=y_train,
                    X_val_raw=X_val_raw,
                    y_val=y_val,
                    X_test_raw=X_test_raw,
                    y_test=y_test,
                    X_train_z=X_train_z,
                    train_recs=train_recs,
                    mean_log=mean_log,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    patience=patience,
                )
            elif args.algo == "zpia_top1_pool":
                pipeline_out = _run_act_zpia_template_pool_pipeline(
                    args=args,
                    seed=seed,
                    X_train_raw=X_train_raw,
                    y_train=y_train,
                    X_val_raw=X_val_raw,
                    y_val=y_val,
                    X_test_raw=X_test_raw,
                    y_test=y_test,
                    X_train_z=X_train_z,
                    train_recs=train_recs,
                    mean_log=mean_log,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    patience=patience,
                    algo_label="zpia_top1_pool",
                    top1_only=True,
                )
            elif args.algo == "zpia_multidir_pool":
                pipeline_out = _run_act_zpia_template_pool_pipeline(
                    args=args,
                    seed=seed,
                    X_train_raw=X_train_raw,
                    y_train=y_train,
                    X_val_raw=X_val_raw,
                    y_val=y_val,
                    X_test_raw=X_test_raw,
                    y_test=y_test,
                    X_train_z=X_train_z,
                    train_recs=train_recs,
                    mean_log=mean_log,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    patience=patience,
                    algo_label="zpia_multidir_pool",
                    top1_only=False,
                )
            elif args.algo == "rc4_multiz_fused":
                pipeline_out = _run_act_rc4_multiz_fused_pipeline(
                    args=args,
                    seed=seed,
                    X_train_raw=X_train_raw,
                    y_train=y_train,
                    X_val_raw=X_val_raw,
                    y_val=y_val,
                    X_test_raw=X_test_raw,
                    y_test=y_test,
                    X_train_z=X_train_z,
                    train_recs=train_recs,
                    mean_log=mean_log,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    patience=patience,
                )
            elif args.algo == "rc4_fused":
                pipeline_out = _run_act_rc4_fused_pipeline(
                    args=args,
                    seed=seed,
                    X_train_raw=X_train_raw,
                    y_train=y_train,
                    X_val_raw=X_val_raw,
                    y_val=y_val,
                    X_test_raw=X_test_raw,
                    y_test=y_test,
                    X_train_z=X_train_z,
                    train_recs=train_recs,
                    mean_log=mean_log,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    patience=patience,
                )
            elif args.algo == "spectral_osf":
                pipeline_out = _run_act_spectral_osf_pipeline(
                    args=args,
                    seed=seed,
                    X_train_raw=X_train_raw,
                    y_train=y_train,
                    X_val_raw=X_val_raw,
                    y_val=y_val,
                    X_test_raw=X_test_raw,
                    y_test=y_test,
                    X_train_z=X_train_z,
                    train_recs=train_recs,
                    mean_log=mean_log,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    patience=patience,
                )
            elif args.pipeline == "mba_feedback" and args.algo == "adaptive":
                pipeline_out = _run_mba_feedback_adaptive_pipeline(
                    args=args,
                    seed=seed,
                    X_train_raw=X_train_raw,
                    y_train=y_train,
                    X_val_raw=X_val_raw,
                    y_val=y_val,
                    X_test_raw=X_test_raw,
                    y_test=y_test,
                    X_train_z=X_train_z,
                    train_recs=train_recs,
                    mean_log=mean_log,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    patience=patience,
                )
            elif args.pipeline == "mba_feedback":
                pipeline_out = _run_mba_feedback_pipeline(
                    args=args,
                    seed=seed,
                    X_train_raw=X_train_raw,
                    y_train=y_train,
                    X_val_raw=X_val_raw,
                    y_val=y_val,
                    X_test_raw=X_test_raw,
                    y_test=y_test,
                    X_train_z=X_train_z,
                    train_recs=train_recs,
                    mean_log=mean_log,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    patience=patience,
                )
            else:
                pipeline_out = _run_act_pipeline(
                    args=args,
                    seed=seed,
                    X_train_raw=X_train_raw,
                    y_train=y_train,
                    X_val_raw=X_val_raw,
                    y_val=y_val,
                    X_test_raw=X_test_raw,
                    y_test=y_test,
                    X_train_z=X_train_z,
                    train_recs=train_recs,
                    mean_log=mean_log,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    patience=patience,
                )

            res_base = pipeline_out["res_base"]
            res_act = pipeline_out["res_act"]
            avg_bridge = pipeline_out.get("avg_bridge", {})
            gain = float(res_act["macro_f1"] - res_base["macro_f1"])
            summary = {
                "dataset": dataset_name,
                "seed": seed,
                "status": "success",
                "algo": args.algo,
                "model": args.model,
                "pipeline": "act" if args.pipeline == "mba" else args.pipeline,
                "base_f1": float(res_base["macro_f1"]),
                "act_f1": float(res_act["macro_f1"]),
                "gain": gain,
                "f1_gain_pct": gain / (float(res_base["macro_f1"]) + 1e-7) * 100.0,
                "base_stop_epoch": int(res_base.get("stop_epoch", 0)),
                "act_stop_epoch": int(res_act.get("stop_epoch", 0)),
                "base_best_val_f1": float(res_base.get("best_val_f1", 0.0)),
                "act_best_val_f1": float(res_act.get("best_val_f1", 0.0)),
                "transport_error_fro_mean": float(avg_bridge.get("transport_error_fro", 0.0)),
                "transport_error_logeuc_mean": float(avg_bridge.get("transport_error_logeuc", 0.0)),
                "bridge_cond_A_mean": float(avg_bridge.get("bridge_cond_A", 0.0)),
                "metric_preservation_error_mean": float(avg_bridge.get("metric_preservation_error", 0.0)),
                "safe_radius_ratio_mean": float(pipeline_out.get("safe_radius_ratio_mean", 1.0)),
                "manifold_margin_mean": float(pipeline_out.get("manifold_margin_mean", 0.0)),
                "host_geom_cosine_mean": float(pipeline_out.get("host_geom_cosine_mean", 0.0)),
                "host_conflict_rate": float(pipeline_out.get("host_conflict_rate", 0.0)),
                "candidate_total_count": int(pipeline_out.get("candidate_total_count", 0)),
                "aug_total_count": int(pipeline_out.get("aug_total_count", 0)),
                "requested_k_dir": int(args.k_dir),
                "effective_k_dir": int(pipeline_out.get("effective_k", 0)),
            }
            direction_meta = dict(pipeline_out.get("direction_bank_meta", {}))
            summary["direction_bank_source"] = str(direction_meta.get("bank_source", args.algo))
            if direction_meta.get("bank_source") == "zpia_telm2" or args.algo == "zpia":
                summary.update(
                    {
                        "zpia_z_dim": int(direction_meta.get("z_dim", 0)),
                        "zpia_n_train": int(direction_meta.get("n_train", 0)),
                        "zpia_n_train_lt_z_dim": bool(direction_meta.get("n_train_lt_z_dim", False)),
                        "zpia_row_norm_min": float(direction_meta.get("row_norm_min", 0.0)),
                        "zpia_row_norm_max": float(direction_meta.get("row_norm_max", 0.0)),
                        "zpia_row_norm_mean": float(direction_meta.get("row_norm_mean", 0.0)),
                        "zpia_fallback_row_count": int(direction_meta.get("fallback_row_count", 0)),
                        "telm2_recon_last": float(direction_meta.get("telm2_recon_last", 0.0)),
                        "telm2_recon_mean": float(direction_meta.get("telm2_recon_mean", 0.0)),
                        "telm2_recon_std": float(direction_meta.get("telm2_recon_std", 0.0)),
                        "telm2_n_iters": int(direction_meta.get("telm2_n_iters", args.telm2_n_iters)),
                        "telm2_c_repr": float(direction_meta.get("telm2_c_repr", args.telm2_c_repr)),
                        "telm2_activation": str(direction_meta.get("telm2_activation", args.telm2_activation)),
                        "telm2_bias_update_mode": str(
                            direction_meta.get("telm2_bias_update_mode", args.telm2_bias_update_mode)
                        ),
                    }
                )
            if args.algo in {"rc4_fused", "spectral_osf"}:
                summary.update(
                    {
                        "utilization_mode": "core_concat",
                        "core_training_mode": "concat_all",
                        "aug_train_ratio": float(pipeline_out.get("aug_total_count", 0)) / max(float(len(y_train)), 1.0),
                        "osf_alpha": float(args.osf_alpha),
                        "osf_beta": float(args.osf_beta),
                        "osf_kappa": float(args.osf_kappa),
                        "effective_k_dir_lraes": int(pipeline_out.get("effective_k_lraes", 0)),
                        "effective_k_dir_zpia": int(pipeline_out.get("effective_k_zpia", 0)),
                        "adaptive_engine_sources": ",".join(
                            [str(x) for x in direction_meta.get("engine_sources", [])]
                        ),
                        "osf_structure_overflow_rate": float(
                            pipeline_out.get("osf_structure_overflow_rate", 0.0)
                        ),
                        "osf_alpha_eff_mean": float(pipeline_out.get("osf_alpha_eff_mean", 0.0)),
                        "osf_risk_scale_mean": float(pipeline_out.get("osf_risk_scale_mean", 0.0)),
                        "osf_risk_zero_perp_rate": float(pipeline_out.get("osf_risk_zero_perp_rate", 0.0)),
                        "osf_risk_clipped_rate": float(pipeline_out.get("osf_risk_clipped_rate", 0.0)),
                    }
                )
            if args.algo == "spectral_osf":
                summary.update(
                    {
                        "direction_bank_source": "spectral_osf",
                        "spectral_osf_rho": float(args.spectral_osf_rho),
                        "spectral_k_eff": int(direction_meta.get("spectral_k_eff", 0)),
                        "spectral_rank_raw": int(direction_meta.get("spectral_rank_raw", 0)),
                        "spectral_energy_ratio_eff": float(direction_meta.get("spectral_energy_ratio_eff", 0.0)),
                        "spectral_basis_orth_error": float(direction_meta.get("spectral_basis_orth_error", 0.0)),
                        "spectral_rank_deficient": bool(direction_meta.get("spectral_rank_deficient", False)),
                        "spectral_proj_norm_ratio_mean": float(
                            pipeline_out.get("spectral_proj_norm_ratio_mean", 0.0)
                        ),
                        "spectral_perp_norm_ratio_mean": float(
                            pipeline_out.get("spectral_perp_norm_ratio_mean", 0.0)
                        ),
                        "spectral_zero_perp_rate": float(pipeline_out.get("spectral_zero_perp_rate", 0.0)),
                        "spectral_risk_clipped_rate": float(
                            pipeline_out.get("spectral_risk_clipped_rate", 0.0)
                        ),
                        "spectral_risk_scale_mean": float(pipeline_out.get("spectral_risk_scale_mean", 0.0)),
                        "spectral_structure_overflow_rate": float(
                            pipeline_out.get("spectral_structure_overflow_rate", 0.0)
                        ),
                        "spectral_alpha_eff_mean": float(pipeline_out.get("spectral_alpha_eff_mean", 0.0)),
                    }
                )
            if args.algo in {"zpia_top1_pool", "zpia_multidir_pool", "rc4_multiz_fused"}:
                summary.update(
                    {
                        "utilization_mode": "core_concat",
                        "core_training_mode": "concat_all",
                        "aug_train_ratio": float(pipeline_out.get("aug_total_count", 0)) / max(float(len(y_train)), 1.0),
                        "multi_template_pairs": int(pipeline_out.get("multi_template_pairs", 0)),
                        "template_selection": str(direction_meta.get("template_selection", "anchor_top_abs_response")),
                        "template_usage_entropy": float(pipeline_out.get("template_usage_entropy", 0.0)),
                        "top_template_concentration": float(pipeline_out.get("top_template_concentration", 0.0)),
                        "effective_k_dir_zpia": int(pipeline_out.get("effective_k_zpia", 0)),
                        "u_perp_norm_ratio_mean": float(pipeline_out.get("u_perp_norm_ratio_mean", 0.0)),
                        "u_perp_zero_rate": float(pipeline_out.get("u_perp_zero_rate", 0.0)),
                    }
                )
                zpia_meta = dict(direction_meta.get("zpia_meta", {}))
                if zpia_meta:
                    summary.update(
                        {
                            "zpia_z_dim": int(zpia_meta.get("z_dim", 0)),
                            "zpia_n_train": int(zpia_meta.get("n_train", 0)),
                            "zpia_n_train_lt_z_dim": bool(zpia_meta.get("n_train_lt_z_dim", False)),
                            "zpia_row_norm_min": float(zpia_meta.get("row_norm_min", 0.0)),
                            "zpia_row_norm_max": float(zpia_meta.get("row_norm_max", 0.0)),
                            "zpia_row_norm_mean": float(zpia_meta.get("row_norm_mean", 0.0)),
                            "zpia_fallback_row_count": int(zpia_meta.get("fallback_row_count", 0)),
                            "telm2_recon_last": float(zpia_meta.get("telm2_recon_last", 0.0)),
                            "telm2_recon_mean": float(zpia_meta.get("telm2_recon_mean", 0.0)),
                            "telm2_recon_std": float(zpia_meta.get("telm2_recon_std", 0.0)),
                        }
                    )
                if args.algo == "rc4_multiz_fused":
                    summary.update(
                        {
                            "osf_alpha": float(args.osf_alpha),
                            "osf_beta": float(args.osf_beta),
                            "osf_kappa": float(args.osf_kappa),
                            "effective_k_dir_lraes": int(pipeline_out.get("effective_k_lraes", 0)),
                            "adaptive_engine_sources": ",".join(
                                [str(x) for x in direction_meta.get("engine_sources", [])]
                            ),
                            "osf_structure_overflow_rate": float(
                                pipeline_out.get("osf_structure_overflow_rate", 0.0)
                            ),
                            "osf_alpha_eff_mean": float(pipeline_out.get("osf_alpha_eff_mean", 0.0)),
                            "osf_risk_scale_mean": float(pipeline_out.get("osf_risk_scale_mean", 0.0)),
                            "osf_risk_zero_perp_rate": float(pipeline_out.get("osf_risk_zero_perp_rate", 0.0)),
                            "osf_risk_clipped_rate": float(pipeline_out.get("osf_risk_clipped_rate", 0.0)),
                        }
                    )
            if args.algo in {
                "progressive_zpia_only",
                "progressive_zpia_osf",
                "random_progressive_osf",
                "progressive_zpia_only_pure",
                "progressive_zpia_pure_cumulative",
                "progressive_zpia_exposure_matched",
                "progressive_zpia_osf_highdose",
            }:
                summary.update(
                    {
                        "utilization_mode": "progressive_rolling_core_concat",
                        "core_training_mode": "continuous_rounds",
                        "progressive_rounds": int(pipeline_out.get("progressive_rounds", args.progressive_rounds)),
                        "per_round_multiplier": int(
                            pipeline_out.get("per_round_multiplier", args.per_round_multiplier)
                        ),
                        "progressive_cumulative_aug_count": int(
                            pipeline_out.get("progressive_cumulative_aug_count", 0)
                        ),
                        "progressive_active_aug_count_mean": float(
                            pipeline_out.get("progressive_active_aug_count_mean", 0.0)
                        ),
                        "progressive_aug_exposure_count": int(
                            pipeline_out.get("progressive_aug_exposure_count", 0)
                        ),
                        "progressive_optimizer_steps": int(
                            pipeline_out.get("progressive_optimizer_steps", 0)
                        ),
                        "progressive_osf_p_final": float(pipeline_out.get("progressive_osf_p_final", 0.0)),
                        "progressive_beta_final": float(pipeline_out.get("progressive_beta_final", 0.0)),
                        "progressive_gamma_scale_final": float(
                            pipeline_out.get("progressive_gamma_scale_final", 1.0)
                        ),
                        "progressive_useful_zpia_mean": float(
                            pipeline_out.get("progressive_useful_zpia_mean", 0.0)
                        ),
                        "progressive_useful_osf_mean": float(
                            pipeline_out.get("progressive_useful_osf_mean", 0.0)
                        ),
                        "progressive_mode_zpia_rate": float(
                            pipeline_out.get("progressive_mode_zpia_rate", 0.0)
                        ),
                        "progressive_mode_osf_rate": float(
                            pipeline_out.get("progressive_mode_osf_rate", 0.0)
                        ),
                        "progressive_mode_conservative_rate": float(
                            pipeline_out.get("progressive_mode_conservative_rate", 0.0)
                        ),
                        "progressive_pool_keep_rounds": int(args.progressive_pool_keep_rounds),
                        "progressive_pool_mode": "cumulative"
                        if int(args.progressive_pool_keep_rounds) < 0
                        else ("none" if int(args.progressive_pool_keep_rounds) == 0 else "rolling"),
                        "progressive_disable_conservative": bool(args.progressive_disable_conservative),
                        "progressive_warmup_epochs": int(args.progressive_warmup_epochs),
                        "progressive_round_epochs": int(args.progressive_round_epochs),
                    }
                )
                zpia_meta = dict(direction_meta.get("zpia_meta", {}))
                if zpia_meta:
                    summary.update(
                        {
                            "zpia_z_dim": int(zpia_meta.get("z_dim", 0)),
                            "zpia_n_train": int(zpia_meta.get("n_train", 0)),
                            "zpia_n_train_lt_z_dim": bool(zpia_meta.get("n_train_lt_z_dim", False)),
                            "zpia_row_norm_min": float(zpia_meta.get("row_norm_min", 0.0)),
                            "zpia_row_norm_max": float(zpia_meta.get("row_norm_max", 0.0)),
                            "zpia_row_norm_mean": float(zpia_meta.get("row_norm_mean", 0.0)),
                            "zpia_fallback_row_count": int(zpia_meta.get("fallback_row_count", 0)),
                            "telm2_recon_last": float(zpia_meta.get("telm2_recon_last", 0.0)),
                            "telm2_recon_mean": float(zpia_meta.get("telm2_recon_mean", 0.0)),
                            "telm2_recon_std": float(zpia_meta.get("telm2_recon_std", 0.0)),
                        }
                    )
            if args.pipeline == "mba_feedback":
                summary.update(
                    {
                        "feedback_margin_temperature": float(args.feedback_margin_temperature),
                        "aug_loss_weight": float(args.aug_loss_weight),
                        "feedback_weight_mean": float(res_act.get("feedback_weight_mean", 0.0)),
                        "feedback_weight_std": float(res_act.get("feedback_weight_std", 0.0)),
                        "last_orig_ce_loss": float(res_act.get("last_orig_ce_loss", 0.0)),
                        "last_weighted_aug_ce_loss": float(res_act.get("last_weighted_aug_ce_loss", 0.0)),
                        "last_aug_margin_mean": float(res_act.get("last_aug_margin_mean", 0.0)),
                    }
                )
                if args.algo == "adaptive":
                    summary.update(
                        {
                            "router_temperature": float(args.router_temperature),
                            "router_min_prob": float(args.router_min_prob),
                            "router_smoothing": float(args.router_smoothing),
                            "router_reward": str(args.router_reward),
                            "router_p_lraes_final": float(res_act.get("router_p_lraes_final", 0.0)),
                            "router_p_zpia_final": float(res_act.get("router_p_zpia_final", 0.0)),
                            "router_reward_lraes_last": float(res_act.get("router_reward_lraes_last", 0.0)),
                            "router_reward_zpia_last": float(res_act.get("router_reward_zpia_last", 0.0)),
                            "adaptive_best_engine_final": str(res_act.get("adaptive_best_engine_final", "")),
                            "effective_k_dir_lraes": int(pipeline_out.get("effective_k_lraes", 0)),
                            "effective_k_dir_zpia": int(pipeline_out.get("effective_k_zpia", 0)),
                            "feedback_weight_mean_lraes": float(res_act.get("feedback_weight_mean_lraes", 0.0)),
                            "feedback_weight_mean_zpia": float(res_act.get("feedback_weight_mean_zpia", 0.0)),
                            "last_weighted_aug_ce_loss_lraes": float(
                                res_act.get("last_weighted_aug_ce_loss_lraes", 0.0)
                            ),
                            "last_weighted_aug_ce_loss_zpia": float(
                                res_act.get("last_weighted_aug_ce_loss_zpia", 0.0)
                            ),
                            "last_aug_margin_mean_lraes": float(res_act.get("last_aug_margin_mean_lraes", 0.0)),
                            "last_aug_margin_mean_zpia": float(res_act.get("last_aug_margin_mean_zpia", 0.0)),
                            "adaptive_engine_sources": ",".join(
                                [str(x) for x in direction_meta.get("engine_sources", [])]
                            ),
                        }
                    )
                    if getattr(args, "direction_bank_source", "lraes") == "orthogonal_fusion":
                        summary.update(
                            {
                                "osf_alpha": float(args.osf_alpha),
                                "osf_beta": float(args.osf_beta),
                                "osf_kappa": float(args.osf_kappa),
                                "osf_structure_overflow_rate": float(
                                    pipeline_out.get("osf_structure_overflow_rate", 0.0)
                                ),
                                "osf_alpha_eff_mean": float(pipeline_out.get("osf_alpha_eff_mean", 0.0)),
                                "osf_risk_scale_mean": float(pipeline_out.get("osf_risk_scale_mean", 0.0)),
                                "osf_risk_zero_perp_rate": float(
                                    pipeline_out.get("osf_risk_zero_perp_rate", 0.0)
                                ),
                                "osf_risk_clipped_rate": float(
                                    pipeline_out.get("osf_risk_clipped_rate", 0.0)
                                ),
                            }
                        )
            audit_rows = pipeline_out.get("audit_rows", [])
            if audit_rows:
                audit_dir = os.path.join(args.out_root, "audit")
                os.makedirs(audit_dir, exist_ok=True)
                pd.DataFrame(audit_rows).to_csv(
                    os.path.join(audit_dir, f"{dataset_name}_s{seed}_{args.algo}_candidates.csv"),
                    index=False,
                )
            if args.pipeline == "mba_feedback" and args.algo == "adaptive":
                router_trace = pipeline_out.get("router_trace", [])
                if router_trace:
                    trace_dir = os.path.join(args.out_root, "router")
                    os.makedirs(trace_dir, exist_ok=True)
                    pd.DataFrame(router_trace).to_csv(
                        os.path.join(trace_dir, f"{dataset_name}_s{seed}_router_trace.csv"),
                        index=False,
                    )
            if args.algo in {
                "progressive_zpia_only",
                "progressive_zpia_osf",
                "random_progressive_osf",
                "progressive_zpia_only_pure",
                "progressive_zpia_pure_cumulative",
                "progressive_zpia_exposure_matched",
                "progressive_zpia_osf_highdose",
            }:
                progressive_trace = pipeline_out.get("progressive_trace", [])
                if progressive_trace:
                    trace_dir = os.path.join(args.out_root, "progressive")
                    os.makedirs(trace_dir, exist_ok=True)
                    pd.DataFrame(progressive_trace).to_csv(
                        os.path.join(trace_dir, f"{dataset_name}_s{seed}_progressive_trace.csv"),
                        index=False,
                    )
            print(
                f"Base: {summary['base_f1']:.4f} | "
                f"ACT: {summary['act_f1']:.4f} | "
                f"Gain: {summary['gain']:.4f} ({summary['f1_gain_pct']:.1f}%)"
            )
            results.append(summary)

            if args.save_viz_samples:
                viz_dir = os.path.join(args.out_root, "viz_data")
                os.makedirs(viz_dir, exist_ok=True)
                save_path = os.path.join(viz_dir, f"{dataset_name}_s{seed}_viz.npz")
                np.savez(
                    save_path,
                    Z_orig=pipeline_out["viz_payload"].get("Z_orig", X_train_z),
                    y_orig=y_train,
                    Z_aug=pipeline_out["viz_payload"]["Z_aug"],
                    y_aug=pipeline_out["viz_payload"]["y_aug"],
                    X_orig_raw=X_train_raw[:20],
                    X_aug_raw=pipeline_out["viz_payload"]["X_aug_raw"],
                    mean_log=mean_log,
                )
                print(f"Visualization samples saved to {save_path}")

        except Exception as exc:
            import traceback

            traceback.print_exc()
            print(f"Error in {dataset_name} Seed {seed}: {exc}")
            results.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "status": "failed",
                    "fail_reason": str(exc),
                    "requested_k_dir": args.k_dir,
                    "effective_k_dir": 0,
                    "algo": args.algo,
                    "model": args.model,
                    "pipeline": "act" if args.pipeline == "mba" else args.pipeline,
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(description="ACT_ManifoldBridge original ACT runner")
    parser.add_argument("--dataset", type=str, default="natops")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--pipeline", type=str, choices=["act", "mba", "mba_feedback"], default="act")
    parser.add_argument(
        "--algo",
        type=str,
        choices=[
            "pia",
            "lraes",
            "zpia",
            "adaptive",
            "rc4_fused",
            "spectral_osf",
            "zpia_top1_pool",
            "zpia_multidir_pool",
            "rc4_multiz_fused",
            "progressive_zpia_only",
            "progressive_zpia_osf",
            "random_progressive_osf",
            "progressive_zpia_only_pure",
            "progressive_zpia_pure_cumulative",
            "progressive_zpia_exposure_matched",
            "progressive_zpia_osf_highdose",
        ],
        default="lraes",
    )
    parser.add_argument("--model", type=str, choices=["minirocket", "resnet1d", "patchtst", "timesnet"], default="resnet1d")
    parser.add_argument("--host-config", type=str, choices=["none", "resnet1d_default", "patchtst_default", "timesnet_default"], default="none")
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--k-dir", type=int, default=10)
    parser.add_argument("--pia-gamma", type=float, default=0.1)
    parser.add_argument("--multiplier", type=int, default=1)
    parser.add_argument("--n-kernels", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--theory-diagnostics", action="store_true", help="Enable host-alignment diagnostics for sampled augmented trials")
    parser.add_argument("--disable-safe-step", action="store_true", help="Disable Safe-Step constraint")
    parser.add_argument("--save-viz-samples", action="store_true", help="Save latent and raw samples for visualization")
    parser.add_argument("--feedback-margin-temperature", type=float, default=1.0)
    parser.add_argument("--aug-loss-weight", type=float, default=1.0)
    parser.add_argument("--telm2-n-iters", type=int, default=3)
    parser.add_argument("--telm2-c-repr", type=float, default=1.0)
    parser.add_argument("--telm2-activation", type=str, choices=["sine", "sigmoid"], default="sine")
    parser.add_argument("--telm2-bias-update-mode", type=str, choices=["off", "act_mean", "residual"], default="residual")
    parser.add_argument("--router-temperature", type=float, default=0.05)
    parser.add_argument("--router-min-prob", type=float, default=0.10)
    parser.add_argument("--router-smoothing", type=float, default=0.5)
    parser.add_argument("--router-reward", type=str, choices=["feedback_weight"], default="feedback_weight")
    # --- V2 Sprint 1+2 parameters ---
    parser.add_argument("--onthefly-aug", action="store_true",
                        help="Use on-the-fly ManifoldAugDataset instead of pre-materialised X_aug_raw (Sprint 1)")
    parser.add_argument("--steps-per-epoch", type=int, default=0,
                        help="Fix step count per epoch when using on-the-fly aug (0 = len(aug_loader))")
    parser.add_argument("--aug-weight-mode", type=str, choices=["sigmoid", "focal"], default="sigmoid",
                        help="Augmented sample weighting: sigmoid (V1) or focal U-shape (Sprint 2)")
    parser.add_argument("--tau-max", type=float, default=2.0,
                        help="TauScheduler: initial (high) temperature during exploration phase")
    parser.add_argument("--tau-min", type=float, default=0.1,
                        help="TauScheduler: final (low) temperature after annealing")
    parser.add_argument("--tau-warmup-ratio", type=float, default=0.3,
                        help="TauScheduler: fraction of epochs to keep tau at tau_max before annealing")
    # --- V2 Sprint 3 parameters ---
    parser.add_argument("--consistency-regularization", action="store_true",
                        help="Enable cross-engine consistency regularization (Task 3.1, adaptive only)")
    parser.add_argument("--lambda-consistency", type=float, default=0.1,
                        help="Weight for cross-engine consistency loss (default 0.1)")
    parser.add_argument("--consistency-mode", type=str, choices=["mse", "kl"], default="mse",
                        help="Consistency loss type: mse (stop-grad on zpia feats) or kl divergence")
    parser.add_argument("--direction-bank-source", type=str, choices=["lraes", "zpia_telm2", "orthogonal_fusion"], default="lraes")
    parser.add_argument("--osf-alpha", type=float, default=1.0)
    parser.add_argument("--osf-beta", type=float, default=1.0)
    parser.add_argument("--osf-kappa", type=float, default=1.0)
    parser.add_argument("--spectral-osf-rho", type=float, default=0.90)
    parser.add_argument("--multi-template-pairs", type=int, default=0)
    parser.add_argument("--progressive-rounds", type=int, default=5)
    parser.add_argument("--per-round-multiplier", type=int, default=2)
    parser.add_argument("--progressive-warmup-epochs", type=int, default=5)
    parser.add_argument("--progressive-round-epochs", type=int, default=5)
    parser.add_argument("--progressive-pool-keep-rounds", type=int, default=2)
    parser.add_argument(
        "--progressive-disable-conservative",
        action="store_true",
        help="Disable anchor-level conservative zPIA fallback for pure/exposure V1.1 probes.",
    )
    parser.add_argument("--progressive-osf-p-init", type=float, default=0.15)
    parser.add_argument("--progressive-osf-p-max", type=float, default=0.4)
    parser.add_argument("--progressive-osf-beta-init", type=float, default=0.25)
    parser.add_argument("--progressive-osf-beta-min", type=float, default=0.1)
    parser.add_argument("--progressive-osf-beta-max", type=float, default=0.5)
    parser.add_argument("--progressive-margin-upper", type=float, default=5.0)
    parser.add_argument("--progressive-trigger-quantile", type=float, default=0.25)
    parser.add_argument("--out-root", type=str, default="standalone_projects/ACT_ManifoldBridge/results/act_core")
    args = parser.parse_args()

    if args.pipeline == "mba":
        print("Using legacy pipeline alias 'mba' -> 'act'.")
    if args.feedback_margin_temperature <= 0.0:
        raise ValueError("--feedback-margin-temperature must be positive.")
    if args.aug_loss_weight < 0.0:
        raise ValueError("--aug-loss-weight must be non-negative.")
    if args.algo == "zpia":
        if args.k_dir <= 0:
            raise ValueError("--algo zpia requires --k-dir > 0.")
        if args.telm2_c_repr <= 0.0:
            raise ValueError("--telm2-c-repr must be positive.")
        if args.telm2_n_iters < 0:
            raise ValueError("--telm2-n-iters must be non-negative.")
    if args.algo == "adaptive":
        if args.pipeline != "mba_feedback":
            raise ValueError("--algo adaptive currently supports --pipeline mba_feedback only.")
        if args.model != "resnet1d":
            raise ValueError("--algo adaptive v1 supports --model resnet1d only.")
        if args.router_temperature <= 0.0:
            raise ValueError("--router-temperature must be positive.")
        if not (0.0 <= args.router_min_prob < 0.5):
            raise ValueError("--router-min-prob must satisfy 0 <= value < 0.5.")
        if not (0.0 <= args.router_smoothing <= 1.0):
            raise ValueError("--router-smoothing must satisfy 0 <= value <= 1.")
        if args.direction_bank_source == "orthogonal_fusion":
            if args.disable_safe_step:
                raise ValueError("--direction-bank-source orthogonal_fusion requires safe-step; do not pass --disable-safe-step.")
            if args.osf_kappa < 0.0:
                raise ValueError("--osf-kappa must satisfy value >= 0.")
    if args.algo == "rc4_fused":
        if args.pipeline not in {"act", "mba"}:
            raise ValueError("--algo rc4_fused currently supports --pipeline act only.")
        if args.model != "resnet1d":
            raise ValueError("--algo rc4_fused currently supports --model resnet1d only.")
        if args.disable_safe_step:
            raise ValueError("--algo rc4_fused requires safe-step; do not pass --disable-safe-step.")
        if args.osf_kappa < 0.0:
            raise ValueError("--osf-kappa must satisfy value >= 0.")
    if args.algo == "spectral_osf":
        if args.pipeline not in {"act", "mba"}:
            raise ValueError("--algo spectral_osf currently supports --pipeline act only.")
        if args.model != "resnet1d":
            raise ValueError("--algo spectral_osf currently supports --model resnet1d only.")
        if args.disable_safe_step:
            raise ValueError("--algo spectral_osf requires safe-step; do not pass --disable-safe-step.")
        if args.osf_kappa < 0.0:
            raise ValueError("--osf-kappa must satisfy value >= 0.")
        if not (0.0 < float(args.spectral_osf_rho) <= 1.0):
            raise ValueError("--spectral-osf-rho must satisfy 0 < value <= 1.")
    if args.algo in {"zpia_top1_pool", "zpia_multidir_pool", "rc4_multiz_fused"}:
        if args.pipeline not in {"act", "mba"}:
            raise ValueError(f"--algo {args.algo} currently supports --pipeline act only.")
        if args.model != "resnet1d":
            raise ValueError(f"--algo {args.algo} currently supports --model resnet1d only.")
        if args.disable_safe_step:
            raise ValueError(f"--algo {args.algo} requires safe-step; do not pass --disable-safe-step.")
        if args.k_dir <= 0:
            raise ValueError(f"--algo {args.algo} requires --k-dir > 0.")
        if args.multiplier <= 0 or args.multiplier % 2 != 0:
            raise ValueError(f"--algo {args.algo} requires a positive even --multiplier.")
        if args.multi_template_pairs < 0:
            raise ValueError("--multi-template-pairs must be >= 0.")
        if args.algo in {"zpia_multidir_pool", "rc4_multiz_fused"}:
            pairs = args.multi_template_pairs if args.multi_template_pairs > 0 else args.multiplier // 2
            if 2 * pairs != args.multiplier:
                raise ValueError("--multi-template-pairs must satisfy 2 * pairs == multiplier.")
            if pairs > args.k_dir:
                raise ValueError("--multi-template-pairs must be <= --k-dir.")
        if args.algo == "rc4_multiz_fused" and args.osf_kappa < 0.0:
            raise ValueError("--osf-kappa must satisfy value >= 0.")
    if args.algo in {
        "progressive_zpia_only",
        "progressive_zpia_osf",
        "random_progressive_osf",
        "progressive_zpia_only_pure",
        "progressive_zpia_pure_cumulative",
        "progressive_zpia_exposure_matched",
        "progressive_zpia_osf_highdose",
    }:
        if args.pipeline not in {"act", "mba"}:
            raise ValueError(f"--algo {args.algo} currently supports --pipeline act only.")
        if args.model != "resnet1d":
            raise ValueError(f"--algo {args.algo} currently supports --model resnet1d only.")
        if args.disable_safe_step:
            raise ValueError(f"--algo {args.algo} requires safe-step; do not pass --disable-safe-step.")
        if args.k_dir <= 0:
            raise ValueError(f"--algo {args.algo} requires --k-dir > 0.")
        if args.progressive_rounds <= 0:
            raise ValueError("--progressive-rounds must be positive.")
        if args.per_round_multiplier <= 0 or args.per_round_multiplier % 2 != 0:
            raise ValueError("--per-round-multiplier must be a positive even integer.")
        if args.progressive_warmup_epochs < 0:
            raise ValueError("--progressive-warmup-epochs must be >= 0.")
        if args.progressive_round_epochs <= 0:
            raise ValueError("--progressive-round-epochs must be positive.")
        if args.progressive_pool_keep_rounds < -1:
            raise ValueError("--progressive-pool-keep-rounds must be >= -1 (-1 means cumulative pool).")
        if not (0.0 <= args.progressive_osf_p_init <= args.progressive_osf_p_max <= 1.0):
            raise ValueError("--progressive-osf-p-init/max must satisfy 0 <= init <= max <= 1.")
        if not (0.0 < args.progressive_osf_beta_min <= args.progressive_osf_beta_init <= args.progressive_osf_beta_max):
            raise ValueError("--progressive OSF beta values must satisfy 0 < min <= init <= max.")
        if args.progressive_margin_upper <= 0.0:
            raise ValueError("--progressive-margin-upper must be positive.")
        if not (0.0 <= args.progressive_trigger_quantile <= 1.0):
            raise ValueError("--progressive-trigger-quantile must satisfy 0 <= value <= 1.")
        if args.osf_kappa < 0.0:
            raise ValueError("--osf-kappa must satisfy value >= 0.")
    if args.pipeline == "mba_feedback" and args.model == "minirocket":
        raise ValueError("--pipeline mba_feedback supports resnet1d, patchtst, and timesnet only.")

    os.makedirs(args.out_root, exist_ok=True)
    datasets = [args.dataset]
    if args.all_datasets:
        datasets = sorted(list(AEON_FIXED_SPLIT_SPECS.keys()))

    all_results = []
    for dataset_name in datasets:
        try:
            result_rows = run_experiment(dataset_name, args)
            all_results.extend(result_rows)
            pd.DataFrame(all_results).to_csv(os.path.join(args.out_root, "sweep_results.csv"), index=False)
        except Exception as exc:
            print(f"Failed {dataset_name}: {exc}")

    final_df = pd.DataFrame(all_results)
    out_csv = os.path.join(args.out_root, f"{datasets[0]}_results.csv")
    final_df.to_csv(out_csv, index=False)
    print(f"\nSweep complete. Results saved to {out_csv}")


if __name__ == "__main__":
    main()
