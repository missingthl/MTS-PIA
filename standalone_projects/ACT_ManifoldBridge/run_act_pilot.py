import os

if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from core.bridge import bridge_single, logvec_to_spd
from core.curriculum import active_direction_probs, build_curriculum_aug_candidates
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
            if args.algo == "rc4_fused":
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
        choices=["pia", "lraes", "zpia", "adaptive", "rc4_fused", "spectral_osf"],
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
