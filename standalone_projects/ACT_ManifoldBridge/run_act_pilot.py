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
from core.curriculum import (
    active_direction_probs,
    build_curriculum_aug_candidates,
    estimate_local_manifold_margins,
    find_same_class_knn_neighbors,
)
from core.pia import (
    FisherPIAConfig,
    LRAESConfig,
    _build_spectral_structure_basis_from_zpia_bank,
    build_lraes_direction_bank,
    build_pia_direction_bank,
    build_zpia_direction_bank,
)
from core.pia_audit import write_candidate_audit
from host_alignment_probe import compute_gradient_alignment
from utils.datasets import AEON_FIXED_SPLIT_SPECS, load_trials_for_dataset, make_trial_split
from utils.evaluators import (
    ManifoldAugDataset,
    TauScheduler,
    build_model,
    fit_eval_minirocket,
    fit_eval_patchtst,
    fit_eval_patchtst_weighted_aug_ce,
    fit_eval_resnet1d,
    fit_eval_resnet1d_weighted_aug_ce,
    fit_eval_timesnet,
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
        if getattr(args, "disable_safe_step", False):
            # If disabled, we just cap it to shared radius but don't error out
            scale = r_shared / (norm_final + eps)
            delta_z = delta_z * scale
            norm_final = torch.norm(delta_z, p=2, dim=-1, keepdim=True)
        else:
            raise RuntimeError(
                f"RC-4 OSF final norm exceeded shared radius by {max_overshoot:.6e}; "
                "this indicates a bug in the structure-first risk-budget logic."
            )

    zero_perp_mask = (~has_perp).reshape(-1)
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






def _run_analysis_probe(
    *,
    args,
    model_obj=None,
    tid_aug: List[str],
    X_aug: Optional[np.ndarray] = None,
    tid_to_rec: Dict[str, TrialRecord],
) -> Dict[str, float]:
    if not args.theory_diagnostics or model_obj is None or len(tid_aug) == 0 or X_aug is None:
        return {"host_geom_cosine_mean": 0.0, "host_conflict_rate": 0.0}

    try:
        model_obj.eval()
        grads = {}
        delta_zs = []
        
        # We compute alignment in the model's internal feature space
        for i, tid in enumerate(tid_aug):
            rec = tid_to_rec[tid]
            x_orig_torch = torch.from_numpy(rec.x_raw).float().unsqueeze(0).to(args.device)
            x_aug_torch = torch.from_numpy(X_aug[i]).float().unsqueeze(0).to(args.device)
            y_torch = torch.tensor([rec.y], dtype=torch.long).to(args.device)
            
            # 1. Get internal latent representations
            with torch.no_grad():
                if hasattr(model_obj, "encode"):
                    z_orig = model_obj.encode(x_orig_torch)
                    z_aug_curr = model_obj.encode(x_aug_torch)
                elif hasattr(model_obj, "backbone"):
                    z_orig = model_obj.backbone(x_orig_torch)
                    z_aug_curr = model_obj.backbone(x_aug_torch)
                else:
                    # Fallback if no encoder found
                    continue
                
                dz_internal = (z_aug_curr - z_orig).view(-1).cpu().numpy()
            
            # 2. Get gradient wrt internal latent state
            z_base = z_orig.detach().clone().requires_grad_(True)
            if hasattr(model_obj, "classify"):
                out = model_obj.classify(z_base)
            elif hasattr(model_obj, "projection"):
                out = model_obj.projection(z_base)
            else:
                continue
                
            loss = torch.nn.functional.cross_entropy(out, y_torch)
            loss.backward()
            
            if z_base.grad is not None:
                g_internal = z_base.grad.view(-1).detach().cpu().numpy()
                
                # Compute cosine with -grad (descent direction)
                target_desc = -g_internal
                norm_g = np.linalg.norm(target_desc)
                norm_dz = np.linalg.norm(dz_internal)
                
                if norm_g < 1e-12 or norm_dz < 1e-12:
                    cos = 0.0
                else:
                    cos = np.dot(target_desc, dz_internal) / (norm_g * norm_dz)
                
                delta_zs.append(cos)
        
        cosines = delta_zs
        conflicts = sum(1 for c in cosines if c < 0)
        
        return {
            "host_geom_cosine_mean": float(np.mean(cosines)) if cosines else 0.0,
            "host_conflict_rate": float(conflicts / len(cosines)) if cosines else 0.0,
        }
    except Exception as e:
        print(f"Analysis probe failed: {e}")
        return {"host_geom_cosine_mean": 0.0, "host_conflict_rate": 0.0}


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
    elif algo_name == "pca":
        from core.pia import build_pca_direction_bank
        direction_bank, direction_meta = build_pca_direction_bank(
            X_train_z,
            k_dir=args.k_dir,
            seed=seed,
        )
    elif algo_name == "random_orth":
        from core.pia import build_random_orthogonal_direction_bank
        direction_bank, direction_meta = build_random_orthogonal_direction_bank(
            X_train_z,
            k_dir=args.k_dir,
            seed=seed,
        )
    else:
        direction_bank, direction_meta = build_pia_direction_bank(X_train_z, k_dir=args.k_dir, seed=seed)
    return {"bank": direction_bank, "meta": direction_meta}


def _resolve_multi_template_pairs(*, args, effective_k: int, top1_only: bool) -> int:
    if int(args.multiplier) <= 0:
        raise ValueError("multi-template pools require --multiplier > 0.")
    if top1_only:
        pairs = 1
        if pairs > int(effective_k):
            raise ValueError(
                f"--multi-template-pairs={pairs} exceeds effective zPIA bank size {effective_k}."
            )
        return pairs
    if int(args.multiplier) % 2 != 0:
        raise ValueError("multi-template pools require an even --multiplier for +/- template slots.")
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


def _normalize_unit_interval(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return vals
    finite = np.isfinite(vals)
    if not finite.any():
        return np.zeros_like(vals, dtype=np.float64)
    lo = float(np.min(vals[finite]))
    hi = float(np.max(vals[finite]))
    if hi <= lo + 1e-12:
        out = np.zeros_like(vals, dtype=np.float64)
        out[finite] = 0.5
        return out
    out = (vals - lo) / (hi - lo)
    out[~finite] = 0.0
    return out


def _build_top_response_template_slots(
    *,
    args,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    train_recs: List[TrialRecord],
    zpia_bank: np.ndarray,
    seed: int,
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

    mode = getattr(args, "template_selection", "top_response")
    neighbor_indices = None
    group_size = int(getattr(args, "group_size", 5))
    fv_selector_modes = {"fv_filter_top5", "fv_score_top5", "random_feasible_selector"}
    is_fv_selector = mode in fv_selector_modes
    fv_top_k = 5
    fv_rho = 0.75

    if mode.startswith("group_") or mode == "sameclass_zmix":
        if mode == "group_top_random_sameclass":
            # For each anchor, pick k random neighbors from the SAME class
            neighbor_indices = np.zeros((len(y_arr), group_size), dtype=np.int64)
            for c in np.unique(y_arr):
                idx_c = np.where(y_arr == c)[0]
                for i_local, idx_global in enumerate(idx_c):
                    rng = np.random.default_rng(int(idx_global) + int(seed))
                    neighbor_indices[idx_global] = rng.choice(idx_c, size=group_size, replace=True)
        else:
            # Default kNN group
            neighbor_indices = find_same_class_knn_neighbors(X_train_z, y_arr, k=group_size)

    def _top_ids_for_idx(idx: int) -> np.ndarray:
        k = zpia_bank.shape[0]
        if mode == "random":
            rng = np.random.default_rng(int(idx) + int(seed))
            return rng.choice(np.arange(k), size=(pairs,), replace=False)
        elif mode == "fixed":
            return np.arange(pairs) % k
        elif mode == "group_random":
            group_ids = neighbor_indices[idx]
            group_seed = int(np.sum(group_ids)) + int(seed)
            rng = np.random.default_rng(group_seed)
            return rng.choice(np.arange(k), size=(pairs,), replace=False)
        elif mode == "group_top" or mode == "group_top_random_sameclass":
            # Group top: use the center of the group to select templates
            group_ids = neighbor_indices[idx]
            z_G = np.mean(X_train_z[group_ids], axis=0)
            responses = np.abs(np.asarray(z_G, dtype=np.float64) @ zpia_bank.T)
            order = np.lexsort((np.arange(k), -responses))
            return order[:pairs]
        elif mode == "group_avg_response":
            # Group avg response: average the absolute responses across the group
            group_ids = neighbor_indices[idx]
            group_pts = X_train_z[group_ids] # (k_group, z_dim)
            group_responses = np.abs(np.asarray(group_pts, dtype=np.float64) @ zpia_bank.T) # (k_group, n_templates)
            mean_responses = np.mean(group_responses, axis=0)
            order = np.lexsort((np.arange(k), -mean_responses))
            return order[:pairs]
        elif mode.startswith("topk_softmax_tau_"):
            tau = float(mode.split("_")[-1])
            top_k_num = 5
            responses = np.abs(np.asarray(X_train_z[idx], dtype=np.float64) @ zpia_bank.T)
            top_indices = np.lexsort((np.arange(k), -responses))[:top_k_num]
            top_responses = responses[top_indices]
            logits = top_responses / max(float(tau), 1e-12)
            logits = logits - float(np.max(logits))
            exp_r = np.exp(logits)
            probs = exp_r / np.sum(exp_r)
            rng = np.random.default_rng(int(idx) + int(seed))
            chosen = rng.choice(top_indices, size=(pairs,), replace=True, p=probs)
            return chosen
        elif mode.startswith("topk_uniform_top"):
            top_k_num = int(mode.split("top")[-1])
            responses = np.abs(np.asarray(X_train_z[idx], dtype=np.float64) @ zpia_bank.T)
            top_indices = np.lexsort((np.arange(k), -responses))[:top_k_num]
            rng = np.random.default_rng(int(idx) + int(seed))
            chosen = rng.choice(top_indices, size=(pairs,), replace=True)
            return chosen
        else:
            # Default: top_response
            responses = np.abs(np.asarray(X_train_z[idx], dtype=np.float64) @ zpia_bank.T)
            # Sort by response descending, then template id ascending for deterministic ties.
            order = np.lexsort((np.arange(k), -responses))
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
    selector_total_proposed = 0
    selector_total_feasible = 0
    selector_total_selected = 0
    pre_filter_reject_count = 0
    reject_reason_zero_gamma = 0
    reject_reason_safe_radius = 0
    reject_reason_zero_direction = 0
    reject_reason_zero_margin = 0
    relevance_scores: List[float] = []
    safe_balance_scores: List[float] = []
    variety_scores: List[float] = []
    fv_scores: List[float] = []
    eps = 1e-12
    pair_cycle = 2 * pairs

    def _candidate_components(idx: int, template_id: int, template_sign: float) -> Dict[str, object]:
        direction = np.asarray(zpia_bank[int(template_id)], dtype=np.float64)
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
            safe_radius_ratio = (
                float(abs(gamma_used) * direction_norm / (safe_radius + eps))
                if safe_radius > 0
                else 0.0
            )
        W_i = (float(template_sign) * gamma_used * direction).astype(np.float32)
        response_abs = float(abs(np.dot(np.asarray(X_train_z[idx], dtype=np.float64), direction)))
        z_delta_norm = float(np.linalg.norm(W_i))
        zero_gamma = bool(abs(gamma_used) <= eps)
        zero_direction = bool(direction_norm <= eps)
        zero_margin = bool(d_min <= eps)
        safe_radius_bad = bool(safe_radius_ratio > 1.0 + 1e-9)
        feasible = not (zero_gamma or zero_direction or zero_margin or safe_radius_bad)
        return {
            "template_id": int(template_id),
            "template_sign": float(template_sign),
            "direction": direction,
            "direction_norm": direction_norm,
            "manifold_margin": d_min,
            "gamma_requested": gamma_requested,
            "gamma_used": float(gamma_used),
            "safe_upper_bound": float(safe_upper_bound),
            "safe_radius_ratio": float(safe_radius_ratio),
            "W_i": W_i,
            "response_abs": response_abs,
            "z_delta_norm": z_delta_norm,
            "is_clipped": float(gamma_requested > (safe_upper_bound + 1e-9)),
            "feasible": bool(feasible),
            "reject_zero_gamma": int(zero_gamma),
            "reject_safe_radius": int(safe_radius_bad),
            "reject_zero_direction": int(zero_direction),
            "reject_zero_margin": int(zero_margin),
        }

    def _select_fv_candidate(idx: int, candidate_order: int) -> Dict[str, object]:
        responses = np.abs(np.asarray(X_train_z[idx], dtype=np.float64) @ zpia_bank.T)
        top_indices = np.lexsort((np.arange(zpia_bank.shape[0]), -responses))[: min(fv_top_k, zpia_bank.shape[0])]
        pool: List[Dict[str, object]] = []
        for rank, template_id in enumerate(top_indices):
            for sign_val in (1.0, -1.0):
                cand = _candidate_components(idx, int(template_id), sign_val)
                cand["template_rank"] = int(rank)
                pool.append(cand)

        rel = _normalize_unit_interval(np.asarray([float(c["response_abs"]) for c in pool], dtype=np.float64))
        safe_raw = np.asarray([-abs(float(c["safe_radius_ratio"]) - fv_rho) for c in pool], dtype=np.float64)
        safe_bal = _normalize_unit_interval(safe_raw)
        disp = _normalize_unit_interval(np.asarray([float(c["z_delta_norm"]) for c in pool], dtype=np.float64))
        counts_so_far = {int(t): template_ids_used.count(int(t)) for t in top_indices}
        diversity_bonus = np.asarray(
            [1.0 / (1.0 + float(counts_so_far.get(int(c["template_id"]), 0))) for c in pool],
            dtype=np.float64,
        )
        for j, cand in enumerate(pool):
            variety = float(disp[j] + diversity_bonus[j])
            cand["relevance_score"] = float(rel[j])
            cand["safe_balance_score"] = float(safe_bal[j])
            cand["variety_score"] = variety
            cand["fv_score"] = float(rel[j] + safe_bal[j] + 0.5 * variety)
            cand["template_diversity_bonus"] = float(diversity_bonus[j])

        feasible_pool = [c for c in pool if bool(c["feasible"])]
        rng = np.random.default_rng(int(idx) * 1009 + int(seed) * 9176 + int(candidate_order))
        if feasible_pool:
            if mode == "fv_score_top5":
                ordered = sorted(feasible_pool, key=lambda c: (-float(c["fv_score"]), int(c["template_id"]), -float(c["template_sign"])))
                shortlist = ordered[: min(fv_top_k, len(ordered))]
                selected = dict(shortlist[int(rng.integers(0, len(shortlist)))])
            else:
                selected = dict(feasible_pool[int(rng.integers(0, len(feasible_pool)))])
        else:
            # Keep the run shape stable, but make the fallback explicit in audit.
            selected = dict(pool[int(rng.integers(0, len(pool)))])
            selected["selector_fallback_no_feasible"] = 1

        selected["selector_candidate_pool_size"] = int(len(pool))
        selected["selector_feasible_count"] = int(len(feasible_pool))
        selected["pre_filter_reject_count"] = int(len(pool) - len(feasible_pool))
        selected["reject_reason_zero_gamma"] = int(sum(int(c["reject_zero_gamma"]) for c in pool))
        selected["reject_reason_safe_radius"] = int(sum(int(c["reject_safe_radius"]) for c in pool))
        selected["reject_reason_zero_direction"] = int(sum(int(c["reject_zero_direction"]) for c in pool))
        selected["reject_reason_zero_margin"] = int(sum(int(c["reject_zero_margin"]) for c in pool))
        return selected

    for slot_idx, spec in enumerate(row_specs):
        idx = int(spec["anchor_index"])
        tid = spec["tid"]
        candidate_order = int(spec["candidate_order"])
        if is_fv_selector:
            selected = _select_fv_candidate(idx, candidate_order)
            pair_pos = int(selected.get("template_rank", 0))
            template_sign = float(selected["template_sign"])
            template_id = int(selected["template_id"])
            direction_norm = float(selected["direction_norm"])
            d_min = float(selected["manifold_margin"])
            gamma_requested = float(selected["gamma_requested"])
            gamma_used = float(selected["gamma_used"])
            safe_upper_bound = float(selected["safe_upper_bound"])
            safe_radius_ratio = float(selected["safe_radius_ratio"])
            W_i = np.asarray(selected["W_i"], dtype=np.float32)
            response_abs = float(selected["response_abs"])
            is_clipped = float(selected["is_clipped"])
            selector_total_proposed += int(selected.get("selector_candidate_pool_size", 0))
            selector_total_feasible += int(selected.get("selector_feasible_count", 0))
            selector_total_selected += 1
            pre_filter_reject_count += int(selected.get("pre_filter_reject_count", 0))
            reject_reason_zero_gamma += int(selected.get("reject_reason_zero_gamma", 0))
            reject_reason_safe_radius += int(selected.get("reject_reason_safe_radius", 0))
            reject_reason_zero_direction += int(selected.get("reject_reason_zero_direction", 0))
            reject_reason_zero_margin += int(selected.get("reject_reason_zero_margin", 0))
            relevance_scores.append(float(selected.get("relevance_score", np.nan)))
            safe_balance_scores.append(float(selected.get("safe_balance_score", np.nan)))
            variety_scores.append(float(selected.get("variety_score", np.nan)))
            fv_scores.append(float(selected.get("fv_score", np.nan)))
        else:
            top_ids = _top_ids_for_idx(idx)
            pair_pos = (candidate_order % pair_cycle) // 2
            template_sign = 1.0 if (candidate_order % 2 == 0) else -1.0
            template_id = int(top_ids[pair_pos])
            comp = _candidate_components(idx, template_id, template_sign)
            direction_norm = float(comp["direction_norm"])
            d_min = float(comp["manifold_margin"])
            gamma_requested = float(comp["gamma_requested"])
            gamma_used = float(comp["gamma_used"])
            safe_upper_bound = float(comp["safe_upper_bound"])
            safe_radius_ratio = float(comp["safe_radius_ratio"])
            W_i = np.asarray(comp["W_i"], dtype=np.float32)
            response_abs = float(comp["response_abs"])
            is_clipped = float(comp["is_clipped"])
        if mode == "sameclass_zmix":
            group_ids = neighbor_indices[idx]
            rng = np.random.default_rng(int(idx) + int(seed) + int(candidate_order))
            others = group_ids[group_ids != idx]
            neighbor_idx = rng.choice(others) if others.size > 0 else idx
            mix_lambda = rng.uniform(0.1, 0.9)
            z_aug_val = (1.0 - mix_lambda) * X_train_z[idx] + mix_lambda * X_train_z[neighbor_idx]
            W_i = (z_aug_val - X_train_z[idx]).astype(np.float32)
            template_id = -1
            response_abs = 0.0
            is_clipped = 0.0

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
                "is_clipped": is_clipped,
                "selection_stage": "pre_bridge_fv" if is_fv_selector else "response_only",
                "selector_name": mode,
                "feasible_flag": float(selected.get("feasible", 1.0)) if is_fv_selector else 1.0,
                "selector_accept_flag": 1.0,
                "selector_candidate_pool_size": int(selected.get("selector_candidate_pool_size", 1)) if is_fv_selector else 1,
                "selector_feasible_count": int(selected.get("selector_feasible_count", 1)) if is_fv_selector else 1,
                "pre_filter_reject_count": int(selected.get("pre_filter_reject_count", 0)) if is_fv_selector else 0,
                "reject_reason_zero_gamma": int(selected.get("reject_reason_zero_gamma", 0)) if is_fv_selector else 0,
                "reject_reason_safe_radius": int(selected.get("reject_reason_safe_radius", 0)) if is_fv_selector else 0,
                "reject_reason_zero_direction": int(selected.get("reject_reason_zero_direction", 0)) if is_fv_selector else 0,
                "reject_reason_zero_margin": int(selected.get("reject_reason_zero_margin", 0)) if is_fv_selector else 0,
                "relevance_score": float(selected.get("relevance_score", np.nan)) if is_fv_selector else np.nan,
                "safe_balance_score": float(selected.get("safe_balance_score", np.nan)) if is_fv_selector else np.nan,
                "variety_score": float(selected.get("variety_score", np.nan)) if is_fv_selector else np.nan,
                "fv_score": float(selected.get("fv_score", np.nan)) if is_fv_selector else np.nan,
                "template_diversity_bonus": float(selected.get("template_diversity_bonus", np.nan)) if is_fv_selector else np.nan,
            }
        )

    usage_stats = _template_usage_stats(template_ids_used)
    feasible_rate = float(selector_total_feasible / max(float(selector_total_proposed), 1.0)) if is_fv_selector else 1.0
    selector_accept_rate = float(selector_total_selected / max(float(selector_total_feasible), 1.0)) if is_fv_selector else 1.0
    return {
        "z_aug": np.stack(z_aug).astype(np.float32) if z_aug else np.empty((0, X_train_z.shape[1]), dtype=np.float32),
        "y_aug": np.asarray(y_aug, dtype=np.int64),
        "tid_aug": np.asarray(tid_aug, dtype=object),
        "W_slots": np.stack(w_slots).astype(np.float32) if w_slots else np.empty((0, X_train_z.shape[1]), dtype=np.float32),
        "candidate_rows": slot_rows,
        "multi_template_pairs": int(pairs),
        "template_usage_entropy": usage_stats["template_usage_entropy"],
        "top_template_concentration": usage_stats["top_template_concentration"],
        "selection_stage": "pre_bridge_fv" if is_fv_selector else "response_only",
        "selector_name": mode,
        "feasible_rate": feasible_rate,
        "selector_accept_rate": selector_accept_rate,
        "pre_filter_reject_count": int(pre_filter_reject_count),
        "reject_reason_zero_gamma": int(reject_reason_zero_gamma),
        "reject_reason_safe_radius": int(reject_reason_safe_radius),
        "reject_reason_zero_direction": int(reject_reason_zero_direction),
        "reject_reason_zero_margin": int(reject_reason_zero_margin),
        "relevance_score_mean": float(np.nanmean(relevance_scores)) if relevance_scores else np.nan,
        "safe_balance_score_mean": float(np.nanmean(safe_balance_scores)) if safe_balance_scores else np.nan,
        "fidelity_score_mean": float(np.nanmean(np.asarray(relevance_scores) + np.asarray(safe_balance_scores))) if relevance_scores else np.nan,
        "variety_score_mean": float(np.nanmean(variety_scores)) if variety_scores else np.nan,
        "fv_score_mean": float(np.nanmean(fv_scores)) if fv_scores else np.nan,
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
        transport_error = float(bridge_meta.get("transport_error_logeuc", np.nan))
        bridge_success = bool(np.isfinite(transport_error))
        post_bridge_reject_reason = "" if bridge_success else "rejected_bridge_fail"
        row.update(
            {
                "algo": algo_name,
                "engine_id": engine_id,
                "direction_bank_source": direction_bank_meta.get("bank_source", algo_name),
                "transport_error_fro": float(bridge_meta.get("transport_error_fro", 0.0)),
                "transport_error_logeuc": transport_error,
                "bridge_cond_A": float(bridge_meta.get("bridge_cond_A", 0.0)),
                "metric_preservation_error": float(bridge_meta.get("metric_preservation_error", 0.0)),
                "bridge_success": bridge_success,
                "post_bridge_reject_flag": float(0.0 if bridge_success else 1.0),
                "post_bridge_reject_reason": post_bridge_reject_reason,
                "candidate_status": "accepted" if bridge_success else post_bridge_reject_reason,
                "reject_reason": post_bridge_reject_reason,
            }
        )
        out_rows.append(row)

    X_aug_raw = np.stack([trial["x"] for trial in aug_trials]) if aug_trials else None
    y_aug_np = np.asarray([trial["y"] for trial in aug_trials], dtype=np.int64) if aug_trials else None
    avg_bridge = pd.DataFrame(bridge_metrics).mean().to_dict() if bridge_metrics else {}
    safe_ratios = [float(row.get("safe_radius_ratio", 0.0)) for row in out_rows]
    clip_flags = [float(row.get("is_clipped", 0.0)) for row in out_rows]
    margins = [float(row.get("manifold_margin", 0.0)) for row in out_rows]
    gamma_used = [float(row.get("gamma_used", 0.0)) for row in out_rows]
    gamma_req = [float(row.get("gamma_requested", 0.0)) for row in out_rows]
    z_delta_norms = [float(row.get("zpia_delta_norm", row.get("z_displacement_norm", 0.0))) for row in out_rows]
    post_bridge_flags = [float(row.get("post_bridge_reject_flag", 0.0)) for row in out_rows]
    bridge_fail_count = int(sum(1 for row in out_rows if row.get("post_bridge_reject_reason", "") == "rejected_bridge_fail"))

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
        "safe_clip_rate": float(np.mean(clip_flags)) if clip_flags else 0.0,
        "manifold_margin_mean": float(np.mean(margins)) if margins else 0.0,
        "gamma_requested_mean": float(np.mean(gamma_req)) if gamma_req else 0.0,
        "gamma_used_mean": float(np.mean(gamma_used)) if gamma_used else 0.0,
        "z_displacement_norm_mean": float(np.mean(z_delta_norms)) if z_delta_norms else 0.0,
        "gamma_zero_rate": float(np.mean([1.0 if g < 1e-12 else 0.0 for g in gamma_used])) if gamma_used else 0.0,
        "aug_valid_rate": float(1.0 - (float(np.mean([1.0 if g < 1e-12 else 0.0 for g in gamma_used])) if gamma_used else 0.0)),
        "post_bridge_reject_count": int(sum(post_bridge_flags)),
        "reject_reason_bridge_fail": int(bridge_fail_count),
        "reject_reason_transport_error": 0,
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
    eta_safe = None if args.disable_safe_step else args.eta_safe
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

    # V2.1: Diagnostic metrics for safe-step audit
    gamma_used_vals = [float(row.get("gamma_used", 0.0)) for row in audit_rows]
    gamma_req_vals = [float(row.get("gamma_requested", 0.0)) for row in audit_rows]
    clip_flags = [float(row.get("is_clipped", 0.0)) for row in audit_rows]
    safe_ratios = [float(row.get("safe_radius_ratio", 0.0)) for row in audit_rows]
    margins = [float(row.get("manifold_margin", 0.0)) for row in audit_rows]

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
        "safe_radius_ratio_mean": float(np.mean(safe_ratios)) if safe_ratios else 1.0,
        "manifold_margin_mean": float(np.mean(margins)) if margins else 0.0,
        "gamma_requested_mean": float(np.mean(gamma_req_vals)) if gamma_req_vals else 0.0,
        "gamma_used_mean": float(np.mean(gamma_used_vals)) if gamma_used_vals else 0.0,
        "gamma_zero_rate": float(np.mean([1.0 if g < 1e-12 else 0.0 for g in gamma_used_vals])) if gamma_used_vals else 0.0,
        "safe_clip_rate": float(np.mean(clip_flags)) if clip_flags else 0.0,
        "eta_safe": eta_safe,
        "candidate_total_count": int(len(z_aug)),
        "aug_total_count": int(len(z_aug)),
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
        args=args,
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        algo_override=args.template_source,
    )
    zpia_bank = np.asarray(bank_out["bank"], dtype=np.float64)
    zpia_meta = dict(bank_out["meta"])
    eta_safe = None if args.disable_safe_step else args.eta_safe
    slots = _build_top_response_template_slots(
        args=args,
        X_train_z=X_train_z,
        y_train=y_train,
        train_recs=train_recs,
        zpia_bank=zpia_bank,
        seed=seed,
        candidate_rows=None,
        top1_only=top1_only,
        eta_safe=eta_safe,
    )
    direction_meta = {
        "bank_source": algo_label,
        "zpia_meta": zpia_meta,
        "template_selection": str(getattr(args, "template_selection", "top_response")),
        "template_slot_policy": "dual_sign_fixed_budget",
        "multi_template_pairs": int(slots["multi_template_pairs"]),
    }
    return _materialize_z_aug_out(
        z_aug=slots["z_aug"],
        y_aug=slots["y_aug"],
        tid_aug=slots["tid_aug"],
        audit_rows=slots["candidate_rows"],
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
            "selection_stage": str(slots.get("selection_stage", "response_only")),
            "selector_name": str(slots.get("selector_name", getattr(args, "template_selection", ""))),
            "feasible_rate": float(slots.get("feasible_rate", 1.0)),
            "selector_accept_rate": float(slots.get("selector_accept_rate", 1.0)),
            "pre_filter_reject_count": int(slots.get("pre_filter_reject_count", 0)),
            "reject_reason_zero_gamma": int(slots.get("reject_reason_zero_gamma", 0)),
            "reject_reason_safe_radius": int(slots.get("reject_reason_safe_radius", 0)),
            "reject_reason_zero_direction": int(slots.get("reject_reason_zero_direction", 0)),
            "reject_reason_zero_margin": int(slots.get("reject_reason_zero_margin", 0)),
            "relevance_score_mean": float(slots.get("relevance_score_mean", np.nan)),
            "safe_balance_score_mean": float(slots.get("safe_balance_score_mean", np.nan)),
            "fidelity_score_mean": float(slots.get("fidelity_score_mean", np.nan)),
            "variety_score_mean": float(slots.get("variety_score_mean", np.nan)),
            "fv_score_mean": float(slots.get("fv_score_mean", np.nan)),
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

    # V2.1: Diagnostic metrics for safe-step audit
    gamma_req_vals = [float(row.get("gamma_requested", 0.0)) for row in fused_audit_rows]
    gamma_used_vals = [float(row.get("gamma_used", 0.0)) for row in fused_audit_rows]
    if not gamma_req_vals:
        # Fallback for fused rows that might not have base gamma requested
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


# --- Core Production Pipelines ---
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
        X_aug=aug_out["X_aug_raw"],
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
        "gamma_requested_mean": aug_out.get("gamma_requested_mean", 0.0),
        "gamma_used_mean": aug_out.get("gamma_used_mean", 0.0),
        "gamma_zero_rate": aug_out.get("gamma_zero_rate", 0.0),
        "safe_clip_rate": aug_out.get("safe_clip_rate", 0.0),
        "selection_stage": aug_out.get("selection_stage", "response_only"),
        "selector_name": aug_out.get("selector_name", str(getattr(args, "template_selection", ""))),
        "feasible_rate": aug_out.get("feasible_rate", 1.0),
        "selector_accept_rate": aug_out.get("selector_accept_rate", 1.0),
        "pre_filter_reject_count": aug_out.get("pre_filter_reject_count", 0),
        "post_bridge_reject_count": aug_out.get("post_bridge_reject_count", 0),
        "reject_reason_zero_gamma": aug_out.get("reject_reason_zero_gamma", 0),
        "reject_reason_safe_radius": aug_out.get("reject_reason_safe_radius", 0),
        "reject_reason_bridge_fail": aug_out.get("reject_reason_bridge_fail", 0),
        "reject_reason_transport_error": aug_out.get("reject_reason_transport_error", 0),
        "relevance_score_mean": aug_out.get("relevance_score_mean", np.nan),
        "safe_balance_score_mean": aug_out.get("safe_balance_score_mean", np.nan),
        "fidelity_score_mean": aug_out.get("fidelity_score_mean", np.nan),
        "variety_score_mean": aug_out.get("variety_score_mean", np.nan),
        "fv_score_mean": aug_out.get("fv_score_mean", np.nan),
        "z_displacement_norm_mean": aug_out.get("z_displacement_norm_mean", 0.0),
        "host_geom_cosine_mean": alignment_metrics["host_geom_cosine_mean"],
        "host_conflict_rate": alignment_metrics["host_conflict_rate"],
        "candidate_total_count": aug_out["candidate_total_count"],
        "aug_total_count": aug_out["aug_total_count"],
        "effective_k": aug_out["effective_k"],
        "direction_bank_meta": aug_out.get("direction_bank_meta", {}),
        "audit_rows": aug_out.get("audit_rows", []),
        "eta_safe": aug_out.get("eta_safe", None),
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
        X_aug=aug_out["X_aug_raw"],
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
        "gamma_requested_mean": aug_out.get("gamma_requested_mean", 0.0),
        "gamma_used_mean": aug_out.get("gamma_used_mean", 0.0),
        "gamma_zero_rate": aug_out.get("gamma_zero_rate", 0.0),
        "safe_clip_rate": aug_out.get("safe_clip_rate", 0.0),
        "selection_stage": aug_out.get("selection_stage", "response_only"),
        "selector_name": aug_out.get("selector_name", str(getattr(args, "template_selection", ""))),
        "feasible_rate": aug_out.get("feasible_rate", 1.0),
        "selector_accept_rate": aug_out.get("selector_accept_rate", 1.0),
        "pre_filter_reject_count": aug_out.get("pre_filter_reject_count", 0),
        "post_bridge_reject_count": aug_out.get("post_bridge_reject_count", 0),
        "reject_reason_zero_gamma": aug_out.get("reject_reason_zero_gamma", 0),
        "reject_reason_safe_radius": aug_out.get("reject_reason_safe_radius", 0),
        "reject_reason_bridge_fail": aug_out.get("reject_reason_bridge_fail", 0),
        "reject_reason_transport_error": aug_out.get("reject_reason_transport_error", 0),
        "relevance_score_mean": aug_out.get("relevance_score_mean", np.nan),
        "safe_balance_score_mean": aug_out.get("safe_balance_score_mean", np.nan),
        "fidelity_score_mean": aug_out.get("fidelity_score_mean", np.nan),
        "variety_score_mean": aug_out.get("variety_score_mean", np.nan),
        "fv_score_mean": aug_out.get("fv_score_mean", np.nan),
        "z_displacement_norm_mean": aug_out.get("z_displacement_norm_mean", 0.0),
        "host_geom_cosine_mean": alignment_metrics["host_geom_cosine_mean"],
        "host_conflict_rate": alignment_metrics["host_conflict_rate"],
        "candidate_total_count": aug_out["candidate_total_count"],
        "aug_total_count": aug_out["aug_total_count"],
        "effective_k": aug_out["effective_k"],
        "effective_k_zpia": aug_out.get("effective_k_zpia", 0),
        "direction_bank_meta": aug_out.get("direction_bank_meta", {}),
        "audit_rows": aug_out.get("audit_rows", []),
        "eta_safe": aug_out.get("eta_safe", None),
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
        model_obj=res_base.get("model_obj"),
        batch_size=batch_size,
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
        X_aug=aug_out["X_aug_raw"],
        tid_to_rec=aug_out["tid_to_rec"],
    )

    print(f"Fitting RC4-MultiZ ACT model ({len(X_mix)} samples)...")
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
        "gamma_requested_mean": aug_out.get("gamma_requested_mean", 0.0),
        "gamma_used_mean": aug_out.get("gamma_used_mean", 0.0),
        "gamma_zero_rate": aug_out.get("gamma_zero_rate", 0.0),
        "safe_clip_rate": aug_out.get("safe_clip_rate", 0.0),
        "host_geom_cosine_mean": alignment_metrics["host_geom_cosine_mean"],
        "host_conflict_rate": alignment_metrics["host_conflict_rate"],
        "candidate_total_count": aug_out["candidate_total_count"],
        "aug_total_count": aug_out["aug_total_count"],
        "effective_k": aug_out["effective_k"],
        "effective_k_lraes": aug_out.get("effective_k_lraes", 0),
        "effective_k_zpia": aug_out.get("effective_k_zpia", 0),
        "direction_bank_meta": aug_out.get("direction_bank_meta", {}),
        "audit_rows": aug_out.get("audit_rows", []),
        "eta_safe": aug_out.get("eta_safe", None),
        **_summarize_multitemplate_audit_rows(aug_out.get("audit_rows", [])),
        "viz_payload": {
            "Z_orig": X_train_z,
            "Z_aug": aug_out["z_aug"],
            "y_aug": aug_out["y_aug"],
            "X_aug_raw": np.stack([trial["x"] for trial in aug_out["aug_trials"][:20]])
            if aug_out["aug_trials"]
            else None,
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
        "eta_safe": aug_out.get("eta_safe", None),
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
        X_aug=fused_aug_out["X_aug_raw"],
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
        "eta_safe": fused_aug_out.get("eta_safe", None),
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
            if args.algo == "zpia_top1_pool":
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
            elif args.algo == "pia":
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
            else:
                # Default to base ACT pipeline for lraes/zpia
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
            avg_bridge = pipeline_out["avg_bridge"]
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
                "gamma_requested_mean": float(pipeline_out.get("gamma_requested_mean", 0.0)),
                "gamma_used_mean": float(pipeline_out.get("gamma_used_mean", 0.0)),
                "gamma_zero_rate": float(pipeline_out.get("gamma_zero_rate", 0.0)),
                "host_geom_cosine_mean": float(pipeline_out.get("host_geom_cosine_mean", 0.0)),
                "host_conflict_rate": float(pipeline_out.get("host_conflict_rate", 0.0)),
                "candidate_total_count": int(pipeline_out.get("candidate_total_count", 0)),
                "aug_total_count": int(pipeline_out.get("aug_total_count", 0)),
                "requested_k_dir": int(args.k_dir),
                "effective_k_dir": int(pipeline_out.get("effective_k", 0)),
                "safe_clip_rate": float(pipeline_out.get("safe_clip_rate", 0.0)),
                "template_usage_entropy": float(pipeline_out.get("template_usage_entropy", 0.0)),
                "top_template_concentration": float(pipeline_out.get("top_template_concentration", 0.0)),
                "selection_stage": str(pipeline_out.get("selection_stage", "response_only")),
                "selector_name": str(pipeline_out.get("selector_name", getattr(args, "template_selection", ""))),
                "feasible_rate": float(pipeline_out.get("feasible_rate", 1.0)),
                "selector_accept_rate": float(pipeline_out.get("selector_accept_rate", 1.0)),
                "pre_filter_reject_count": int(pipeline_out.get("pre_filter_reject_count", 0)),
                "post_bridge_reject_count": int(pipeline_out.get("post_bridge_reject_count", 0)),
                "reject_reason_zero_gamma": int(pipeline_out.get("reject_reason_zero_gamma", 0)),
                "reject_reason_safe_radius": int(pipeline_out.get("reject_reason_safe_radius", 0)),
                "reject_reason_zero_direction": int(pipeline_out.get("reject_reason_zero_direction", 0)),
                "reject_reason_zero_margin": int(pipeline_out.get("reject_reason_zero_margin", 0)),
                "reject_reason_bridge_fail": int(pipeline_out.get("reject_reason_bridge_fail", 0)),
                "reject_reason_transport_error": int(pipeline_out.get("reject_reason_transport_error", 0)),
                "relevance_score_mean": float(pipeline_out.get("relevance_score_mean", np.nan)),
                "safe_balance_score_mean": float(pipeline_out.get("safe_balance_score_mean", np.nan)),
                "fidelity_score_mean": float(pipeline_out.get("fidelity_score_mean", np.nan)),
                "variety_score_mean": float(pipeline_out.get("variety_score_mean", np.nan)),
                "fv_score_mean": float(pipeline_out.get("fv_score_mean", np.nan)),
                "z_displacement_norm_mean": float(pipeline_out.get("z_displacement_norm_mean", 0.0)),
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
                            "safe_clip_rate": float(pipeline_out.get("safe_clip_rate", 0.0)),
                            "selected_template_histogram": str(pipeline_out.get("template_counts", {})),
                        }
                    )
                # Summary for production algos
            print(
                f"Base: {summary['base_f1']:.4f} | "
                f"ACT: {summary['act_f1']:.4f} | "
                f"Gain: {summary['gain']:.4f} ({summary['f1_gain_pct']:.1f}%)"
            )
            audit_rows = list(pipeline_out.get("audit_rows", []))
            if audit_rows:
                audit_summary = write_candidate_audit(
                    audit_rows,
                    out_dir=os.path.join(args.out_root, "candidate_audit"),
                    dataset=dataset_name,
                    seed=int(seed),
                    method=str(getattr(args, "audit_method_label", "") or args.algo),
                    activation_policy=str(getattr(args, "template_selection", args.algo)),
                    eta_safe=pipeline_out.get("eta_safe", None),
                )
                summary.update(audit_summary)
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
            "rc4_fused",
            "zpia_top1_pool",
            "zpia_multidir_pool",
            "rc4_multiz_fused",
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
    parser.add_argument("--out-root", type=str, default="standalone_projects/ACT_ManifoldBridge/results/act_core")
    parser.add_argument("--theory-diagnostics", action="store_true")
    parser.add_argument("--save-viz-samples", action="store_true")
    parser.add_argument("--disable-safe-step", action="store_true")
    parser.add_argument("--osf-alpha", type=float, default=0.5)
    parser.add_argument("--osf-beta", type=float, default=0.5)
    parser.add_argument("--osf-kappa", type=float, default=1.0)
    parser.add_argument("--multi-template-pairs", type=int, default=0)
    parser.add_argument("--telm2-c-repr", type=float, default=10.0)
    parser.add_argument("--telm2-n-iters", type=int, default=50)
    parser.add_argument("--telm2-activation", type=str, choices=["sine", "sigmoid", "none"], default="sine")
    parser.add_argument("--telm2-bias-update-mode", type=str, choices=["none", "mean", "ema"], default="none")
    parser.add_argument("--feedback-margin-temperature", type=float, default=1.0)
    parser.add_argument("--aug-loss-weight", type=float, default=0.5)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--aug-weight-mode", type=str, default="sigmoid")
    parser.add_argument(
        "--template-selection",
        type=str,
        choices=[
            "top_response",
            "random",
            "fixed",
            "group_random",
            "group_top",
            "group_avg_response",
            "group_top_random_sameclass",
            "sameclass_zmix",
            "topk_softmax_tau_0.05",
            "topk_softmax_tau_0.10",
            "topk_softmax_tau_0.20",
            "topk_uniform_top5",
            "fv_filter_top5",
            "fv_score_top5",
            "random_feasible_selector",
        ],
        default="top_response",
    )
    parser.add_argument("--template-source", type=str, choices=["zpia", "pca", "random_orth"], default="zpia")
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--eta-safe", type=float, default=0.5)
    parser.add_argument("--audit-method-label", type=str, default="")
    args = parser.parse_args()

    if args.pipeline == "mba":
        print("Using legacy pipeline alias 'mba' -> 'act'.")
    if args.feedback_margin_temperature <= 0.0:
        raise ValueError("--feedback-margin-temperature must be positive.")
    if args.aug_loss_weight < 0.0:
        raise ValueError("--aug-loss-weight must be non-negative.")
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
