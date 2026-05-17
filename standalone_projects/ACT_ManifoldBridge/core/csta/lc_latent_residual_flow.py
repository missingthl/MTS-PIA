from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from core.bridge import bridge_single, logvec_to_spd
from core.curriculum import estimate_local_manifold_margins

from .latent_residual_flow import (
    LatentResidualConfig,
    _build_rbf_sampler,
    _cosine,
    _direction_diversity,
    _effective_rank,
    _pairwise_cosine_mean,
    _predict_latent_velocity,
    _unit,
    fit_latent_residual_operator,
)
from .materialize import materialize_z_aug_out
from .state import TrialRecord
from .task_guided_latent_residual_flow import (
    _score_logits,
    _softmax_margin_and_ce,
    _train_warmup_resnet1d,
)


LC_LATENT_METHODS = {"lc_residual_direct", "lc_latent_residual_flow"}


@dataclass(frozen=True)
class LCResidualConfig:
    beta: float = 1.0
    margin_floor: float = 0.0
    gamma_eps: float = 1e-12
    warmup_epochs: int = 10
    max_candidates: int = 0
    eps: float = 1e-12


def _entropy(p: np.ndarray, eps: float) -> float:
    p = np.asarray(p, dtype=np.float64)
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p + eps)))


def build_lc_residual_sampler(
    *,
    Z: np.ndarray,
    y: np.ndarray,
    X_train_raw: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    warmup: Dict[str, object],
    latent_cfg: LatentResidualConfig,
    lc_cfg: LCResidualConfig,
    gamma_requested: float,
    eta_safe: float | None,
    batch_size: int,
) -> Dict[str, object]:
    """Build label-consistent boundary-band residual sampling distributions."""
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).ravel()
    geo = _build_rbf_sampler(Z, y, cfg=latent_cfg)
    margins = estimate_local_manifold_margins(Z, y)

    tid_to_rec = {record.tid: record for record in train_recs}
    tid_arr = np.asarray([record.tid for record in train_recs], dtype=object)
    per_anchor_records: List[List[Dict[str, object]]] = []
    x_to_score: List[np.ndarray] = []
    score_refs: List[tuple[int, int]] = []

    for i in range(len(Z)):
        cand = np.asarray(geo["candidates"][i], dtype=np.int64)
        p_geo = np.asarray(geo["probs"][i], dtype=np.float64)
        if int(lc_cfg.max_candidates) > 0 and cand.size > int(lc_cfg.max_candidates):
            order = np.argsort(-p_geo)[: int(lc_cfg.max_candidates)]
            cand = cand[order]
            p_geo = p_geo[order]
            p_geo = p_geo / max(float(np.sum(p_geo)), float(lc_cfg.eps))
        records: List[Dict[str, object]] = []
        for pos, j in enumerate(cand):
            raw = Z[int(j)] - Z[i]
            direction, direction_norm = _unit(raw, eps=float(lc_cfg.eps))
            d_min = float(margins[i])
            if direction_norm <= float(lc_cfg.eps):
                gamma_used = 0.0
                safe_upper_bound = 0.0
                safe_radius_ratio = 0.0
            elif eta_safe is None:
                gamma_used = float(gamma_requested)
                safe_upper_bound = float("inf")
                safe_radius_ratio = 1.0
            else:
                safe_upper_bound = float(eta_safe) * d_min / (direction_norm + float(lc_cfg.eps))
                gamma_used = min(float(gamma_requested), safe_upper_bound)
                safe_radius = float(eta_safe) * d_min
                safe_radius_ratio = (
                    float(abs(gamma_used) * direction_norm / (safe_radius + float(lc_cfg.eps))) if safe_radius > 0 else 0.0
                )
            transport_error = np.nan
            bridge_success = False
            x_aug_np = None
            if abs(gamma_used) > float(lc_cfg.gamma_eps) and direction_norm > float(lc_cfg.eps):
                try:
                    src = tid_to_rec[tid_arr[i]]
                    z_target = (Z[i] + gamma_used * direction).astype(np.float32)
                    sigma_aug = logvec_to_spd(z_target, mean_log)
                    x_aug, bridge_meta = bridge_single(
                        torch.from_numpy(src.x_raw),
                        torch.from_numpy(src.sigma_orig),
                        torch.from_numpy(sigma_aug),
                    )
                    transport_error = float(bridge_meta.get("transport_error_logeuc", np.nan))
                    bridge_success = bool(np.isfinite(transport_error))
                    if bridge_success:
                        x_aug_np = x_aug.numpy()
                except Exception:
                    bridge_success = False
            valid_prelogit = bool(
                bridge_success
                and x_aug_np is not None
                and np.isfinite(transport_error)
                and abs(gamma_used) > float(lc_cfg.gamma_eps)
            )
            rec = {
                "target_index": int(j),
                "target_class": int(y[int(j)]),
                "target_dist": float(np.linalg.norm(raw)),
                "target_sampling_prob_geo": float(p_geo[pos]) if p_geo.size else 0.0,
                "residual": direction,
                "residual_norm": float(direction_norm),
                "gamma_used": float(gamma_used),
                "safe_upper_bound": float(safe_upper_bound),
                "safe_radius_ratio": float(safe_radius_ratio),
                "manifold_margin": d_min,
                "transport_error_logeuc": float(transport_error),
                "valid_prelogit": valid_prelogit,
                "ce": np.nan,
                "margin": np.nan,
                "pred": -1,
                "utility": np.nan,
                "valid": False,
            }
            if valid_prelogit:
                score_refs.append((i, len(records)))
                x_to_score.append(x_aug_np)
            records.append(rec)
        per_anchor_records.append(records)

    logits_all = _score_logits(
        warmup["model"],
        warmup["device"],
        np.stack(x_to_score).astype(np.float32) if x_to_score else np.empty((0,) + tuple(X_train_raw.shape[1:]), dtype=np.float32),
        batch_size=batch_size,
    )
    for row_idx, (anchor_i, rec_i) in enumerate(score_refs):
        rec = per_anchor_records[anchor_i][rec_i]
        ce, margin, pred = _softmax_margin_and_ce(logits_all[row_idx], int(y[anchor_i]))
        valid = bool(
            np.isfinite(ce)
            and np.isfinite(margin)
            and int(pred) == int(y[anchor_i])
            and float(margin) >= float(lc_cfg.margin_floor)
            and abs(float(rec["gamma_used"])) > float(lc_cfg.gamma_eps)
            and np.isfinite(float(rec["transport_error_logeuc"]))
        )
        rec.update({"ce": float(ce), "margin": float(margin), "pred": int(pred), "valid": valid})

    lc_candidates: List[np.ndarray] = []
    lc_probs: List[np.ndarray] = []
    candidate_meta: List[List[Dict[str, object]]] = []
    fallback_flags: List[bool] = []
    fallback_reasons: List[str] = []
    entropies: List[float] = []
    supports: List[float] = []
    kls: List[float] = []
    margins_all: List[float] = []
    target_margins: List[float] = []
    valid_rates: List[float] = []
    bad_margin_mass: List[float] = []
    wrong_pred_mass: List[float] = []
    top1_masses: List[float] = []

    for i, records in enumerate(per_anchor_records):
        cand = np.asarray([int(r["target_index"]) for r in records], dtype=np.int64)
        p_geo = np.asarray([float(r["target_sampling_prob_geo"]) for r in records], dtype=np.float64)
        if p_geo.size:
            p_geo = p_geo / max(float(np.sum(p_geo)), float(lc_cfg.eps))
        valid = np.asarray([bool(r["valid"]) for r in records], dtype=bool)
        fallback = False
        reason = ""
        if cand.size == 0:
            p_lc = np.asarray([], dtype=np.float64)
            fallback = True
            reason = "singleton_class"
        elif not np.any(valid):
            p_lc = p_geo.copy()
            fallback = True
            reason = "no_valid_label_consistent_candidates"
        else:
            margins = np.asarray([float(r["margin"]) for r in records], dtype=np.float64)
            valid_m = margins[valid]
            m_star = float(np.quantile(valid_m, 0.25))
            util = np.zeros_like(margins, dtype=np.float64)
            util[valid] = -np.square(margins[valid] - m_star)
            u_valid = util[valid]
            u_std = float(np.std(u_valid))
            if not np.isfinite(u_std) or u_std <= float(lc_cfg.eps):
                u_std = 1.0
            u_clip = np.zeros_like(util, dtype=np.float64)
            u_clip[valid] = np.clip((u_valid - float(np.mean(u_valid))) / u_std, -3.0, 3.0)
            weights = np.zeros_like(p_geo, dtype=np.float64)
            weights[valid] = p_geo[valid] * np.exp(float(lc_cfg.beta) * u_clip[valid])
            w_sum = float(np.sum(weights))
            if not np.isfinite(w_sum) or w_sum <= float(lc_cfg.eps):
                p_lc = p_geo.copy()
                fallback = True
                reason = "degenerate_lc_weights"
            else:
                p_lc = weights / w_sum
            for idx, rec in enumerate(records):
                rec["utility"] = float(util[idx]) if np.isfinite(util[idx]) else np.nan
                rec["lc_margin_target"] = m_star
            target_margins.append(m_star)
        lc_candidates.append(cand)
        lc_probs.append(p_lc.astype(np.float64))
        candidate_meta.append(records)
        fallback_flags.append(fallback)
        fallback_reasons.append(reason)
        valid_rates.append(float(np.mean(valid.astype(float))) if valid.size else 0.0)
        if p_lc.size:
            ent = _entropy(p_lc, float(lc_cfg.eps))
            entropies.append(ent)
            supports.append(float(np.exp(ent)))
            if p_geo.size == p_lc.size and np.sum(p_geo) > 0:
                kls.append(float(np.sum(p_lc * (np.log(p_lc + float(lc_cfg.eps)) - np.log(p_geo + float(lc_cfg.eps))))))
            margins = np.asarray([float(r["margin"]) for r in records], dtype=np.float64)
            preds = np.asarray([int(r["pred"]) for r in records], dtype=np.int64)
            finite_margin = np.isfinite(margins)
            bad_margin_mass.append(float(np.sum(p_lc[(margins < float(lc_cfg.margin_floor)) & finite_margin])) if finite_margin.any() else 0.0)
            wrong_pred_mass.append(float(np.sum(p_lc[preds != int(y[i])])) if preds.size else 0.0)
            top1_masses.append(float(np.max(p_lc)))
        margins_all.extend(float(r["margin"]) for r in records if np.isfinite(float(r["margin"])))

    return {
        "candidates": lc_candidates,
        "probs": lc_probs,
        "geo_sampler": geo,
        "candidate_meta": candidate_meta,
        "fallback_flags": np.asarray(fallback_flags, dtype=bool),
        "fallback_reasons": fallback_reasons,
        "summary": {
            "lc_valid_candidate_rate": float(np.mean(valid_rates)) if valid_rates else 0.0,
            "lc_no_valid_fallback_rate": float(np.mean(fallback_flags)) if fallback_flags else 0.0,
            "lc_bad_margin_mass": float(np.mean(bad_margin_mass)) if bad_margin_mass else np.nan,
            "lc_wrong_pred_mass": float(np.mean(wrong_pred_mass)) if wrong_pred_mass else np.nan,
            "lc_sampling_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "lc_sampling_effective_support": float(np.mean(supports)) if supports else 0.0,
            "lc_kl_lc_vs_geo": float(np.mean(kls)) if kls else 0.0,
            "lc_margin_mean": float(np.mean(margins_all)) if margins_all else np.nan,
            "lc_margin_std": float(np.std(margins_all)) if margins_all else np.nan,
            "lc_margin_target_mean": float(np.mean(target_margins)) if target_margins else np.nan,
            "lc_weight_top1_mass": float(np.mean(top1_masses)) if top1_masses else np.nan,
            "lc_beta": float(lc_cfg.beta),
            "lc_margin_floor": float(lc_cfg.margin_floor),
            "lc_gamma_eps": float(lc_cfg.gamma_eps),
            "lc_warmup_epochs": int(lc_cfg.warmup_epochs),
            "lc_max_candidates": int(lc_cfg.max_candidates),
            "lc_warmup_train_loss_mean": float(warmup.get("task_warmup_train_loss_mean", np.nan)),
        },
    }


def _sample_lc_residual(
    *,
    sampler: Dict[str, object],
    anchor_index: int,
    rng: np.random.Generator,
    z_dim: int,
    eps: float,
) -> Dict[str, object]:
    cand = sampler["candidates"][int(anchor_index)]
    probs = sampler["probs"][int(anchor_index)]
    meta = sampler["candidate_meta"][int(anchor_index)]
    lc_fallback = bool(sampler["fallback_flags"][int(anchor_index)])
    lc_fallback_reason = str(sampler["fallback_reasons"][int(anchor_index)])
    if len(cand) == 0:
        residual, residual_norm = _unit(rng.normal(size=(z_dim,)), eps=eps)
        return {
            "target_index": -1,
            "target_class": -1,
            "target_dist": float(residual_norm),
            "target_sampling_prob": 0.0,
            "target_sampling_prob_geo": 0.0,
            "residual": residual,
            "residual_norm": float(residual_norm),
            "fallback_flag": True,
            "fallback_reason": "singleton_class_seeded_random",
            "lc_utility": np.nan,
            "lc_margin": np.nan,
            "lc_margin_target": np.nan,
            "lc_fallback_flag": True,
            "lc_fallback_reason": "singleton_class",
        }
    pos = int(rng.choice(np.arange(len(cand)), p=probs))
    rec = dict(meta[pos])
    return {
        "target_index": int(rec["target_index"]),
        "target_class": int(rec["target_class"]),
        "target_dist": float(rec["target_dist"]),
        "target_sampling_prob": float(probs[pos]),
        "target_sampling_prob_geo": float(rec.get("target_sampling_prob_geo", np.nan)),
        "residual": np.asarray(rec["residual"], dtype=np.float64),
        "residual_norm": float(rec["residual_norm"]),
        "fallback_flag": False,
        "fallback_reason": "",
        "lc_utility": float(rec.get("utility", np.nan)),
        "lc_margin": float(rec.get("margin", np.nan)),
        "lc_margin_target": float(rec.get("lc_margin_target", np.nan)),
        "lc_fallback_flag": lc_fallback,
        "lc_fallback_reason": lc_fallback_reason,
    }


def _build_lc_targets(
    *,
    Z: np.ndarray,
    y: np.ndarray,
    sampler: Dict[str, object],
    seed: int,
    latent_cfg: LatentResidualConfig,
) -> Dict[str, object]:
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).ravel()
    n, z_dim = Z.shape
    rng = np.random.default_rng(int(seed) + 13011)
    residuals = np.zeros((n, z_dim), dtype=np.float64)
    epsilons = np.zeros((n, z_dim), dtype=np.float64)
    r_s = np.zeros((n, z_dim), dtype=np.float64)
    s_train = rng.uniform(0.0, 1.0, size=(n,)).astype(np.float64)
    u_target = np.zeros((n, z_dim), dtype=np.float64)
    target_indices = np.full((n,), -1, dtype=np.int64)
    target_classes = np.asarray(y, dtype=np.int64).copy()
    target_dists: List[float] = []
    target_probs: List[float] = []
    residual_norms: List[float] = []
    target_velocity_norms: List[float] = []
    fallback_flags: List[bool] = []
    fallback_reasons: List[str] = []
    for i in range(n):
        sample = _sample_lc_residual(sampler=sampler, anchor_index=i, rng=rng, z_dim=z_dim, eps=float(latent_cfg.eps))
        eps_vec, _ = _unit(rng.normal(size=(z_dim,)), eps=float(latent_cfg.eps))
        residual = np.asarray(sample["residual"], dtype=np.float64)
        u = residual - eps_vec
        residuals[i] = residual
        epsilons[i] = eps_vec
        r_s[i] = (1.0 - s_train[i]) * eps_vec + s_train[i] * residual
        u_target[i] = u
        target_indices[i] = int(sample["target_index"])
        target_classes[i] = int(sample["target_class"])
        target_dists.append(float(sample["target_dist"]))
        target_probs.append(float(sample["target_sampling_prob"]))
        residual_norms.append(float(sample["residual_norm"]))
        target_velocity_norms.append(float(np.linalg.norm(u)))
        fallback_flags.append(bool(sample["fallback_flag"]))
        fallback_reasons.append(str(sample["fallback_reason"]))
    summary = {
        "latent_target_dist_mean": float(np.mean(target_dists)) if target_dists else np.nan,
        "latent_target_dist_std": float(np.std(target_dists)) if target_dists else np.nan,
        "latent_target_sampling_entropy": float(sampler["summary"].get("lc_sampling_entropy", np.nan)),
        "latent_target_sampling_entropy_by_class_mean": np.nan,
        "latent_target_sampling_entropy_by_class_min": np.nan,
        "latent_fallback_rate": float(np.mean(fallback_flags)) if fallback_flags else 0.0,
        "latent_residual_effective_rank": _effective_rank(residuals),
        "latent_residual_pairwise_cosine_mean": _pairwise_cosine_mean(residuals),
        "latent_target_velocity_norm_mean": float(np.mean(target_velocity_norms)) if target_velocity_norms else np.nan,
        "latent_target_velocity_norm_std": float(np.std(target_velocity_norms)) if target_velocity_norms else np.nan,
    }
    return {
        "residuals": residuals,
        "epsilons": epsilons,
        "r_s": r_s,
        "s_train": s_train,
        "u_target": u_target,
        "target_indices": target_indices,
        "target_classes": target_classes,
        "target_dists": np.asarray(target_dists, dtype=np.float64),
        "target_sampling_probs": np.asarray(target_probs, dtype=np.float64),
        "residual_norms": np.asarray(residual_norms, dtype=np.float64),
        "target_velocity_norms": np.asarray(target_velocity_norms, dtype=np.float64),
        "fallback_flags": np.asarray(fallback_flags, dtype=bool),
        "fallback_reasons": fallback_reasons,
        "summary": summary,
    }


def build_lc_latent_aug_out(
    *,
    args,
    seed: int,
    X_train_raw: np.ndarray,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    method: str,
) -> Dict[str, object]:
    if method not in LC_LATENT_METHODS:
        raise ValueError(f"Unknown LC latent residual method: {method}")
    if str(getattr(args, "template_selection", "top_response")) != "top_response":
        raise ValueError("LC Latent Residual Flow must not use template-selection modes.")
    latent_cfg = LatentResidualConfig(
        flow_epochs=int(getattr(args, "latent_flow_epochs", 50)),
        flow_batch_size=int(getattr(args, "latent_flow_batch_size", 128)),
        flow_lr=float(getattr(args, "latent_flow_lr", 1e-3)),
        flow_weight_decay=float(getattr(args, "latent_flow_weight_decay", 1e-4)),
        hidden_layers=int(getattr(args, "latent_hidden_layers", 2)),
        hidden_width=int(getattr(args, "latent_hidden_width", 0)),
        class_embedding_dim=int(getattr(args, "latent_class_embedding_dim", 0)),
        lambda_cos=float(getattr(args, "latent_lambda_cos", 0.5)),
        rbf_tau_floor=float(getattr(args, "latent_rbf_tau_floor", 1e-12)),
    )
    lc_cfg = LCResidualConfig(
        beta=float(getattr(args, "lc_beta", 1.0)),
        margin_floor=float(getattr(args, "lc_margin_floor", 0.0)),
        gamma_eps=float(getattr(args, "lc_gamma_eps", 1e-12)),
        warmup_epochs=int(getattr(args, "lc_warmup_epochs", 10)),
        max_candidates=int(getattr(args, "lc_max_candidates", 0)),
    )
    Z = np.asarray(X_train_z, dtype=np.float64)
    y_arr = np.asarray(y_train, dtype=np.int64).ravel()
    eta_safe = None if bool(getattr(args, "disable_safe_step", False)) else float(getattr(args, "eta_safe", 0.75))
    gamma_requested = float(getattr(args, "pia_gamma", 0.1))
    warmup = _train_warmup_resnet1d(
        X_train_raw=X_train_raw,
        y_train=y_arr,
        seed=seed,
        device=str(getattr(args, "device", "cpu")),
        epochs=int(lc_cfg.warmup_epochs),
        lr=float(getattr(args, "lr", 1e-3)),
        batch_size=int(getattr(args, "batch_size", 64)),
    )
    sampler = build_lc_residual_sampler(
        Z=Z,
        y=y_arr,
        X_train_raw=X_train_raw,
        train_recs=train_recs,
        mean_log=mean_log,
        warmup=warmup,
        latent_cfg=latent_cfg,
        lc_cfg=lc_cfg,
        gamma_requested=gamma_requested,
        eta_safe=eta_safe,
        batch_size=int(getattr(args, "batch_size", 64)),
    )
    target_out = _build_lc_targets(Z=Z, y=y_arr, sampler=sampler, seed=seed, latent_cfg=latent_cfg)
    if method == "lc_residual_direct":
        op = None
        op_summary = {
            "latent_train_mse_mean": 0.0,
            "latent_train_cosine_loss_mean": 0.0,
            "latent_train_pred_target_cosine_mean": 1.0,
            "latent_pred_velocity_norm_mean": float(np.mean(target_out["target_velocity_norms"])),
            "latent_pred_velocity_norm_std": float(np.std(target_out["target_velocity_norms"])),
            "latent_hidden_width": 0,
            "latent_class_embedding_dim": 0,
            "latent_hidden_layers": 0,
            "latent_lambda_cos": float(latent_cfg.lambda_cos),
        }
    else:
        op = fit_latent_residual_operator(
            Z,
            y_arr,
            target_out,
            seed=seed,
            device=str(getattr(args, "device", "cpu")),
            cfg=latent_cfg,
        )
        op_summary = dict(op["summary"])

    tid_arr = np.asarray([record.tid for record in train_recs], dtype=object)
    margins = estimate_local_manifold_margins(Z, y_arr)
    multiplier = int(getattr(args, "multiplier", 10))
    rng = np.random.default_rng(int(seed) + 13107)
    z_aug: List[np.ndarray] = []
    y_aug: List[int] = []
    tid_aug: List[object] = []
    rows: List[Dict[str, object]] = []
    pred_cosines: List[float] = []
    for i in range(len(Z)):
        for c in range(multiplier):
            sample = _sample_lc_residual(
                sampler=sampler,
                anchor_index=i,
                rng=rng,
                z_dim=Z.shape[1],
                eps=float(latent_cfg.eps),
            )
            residual = np.asarray(sample["residual"], dtype=np.float64)
            eps_vec, eps_norm = _unit(rng.normal(size=(Z.shape[1],)), eps=float(latent_cfg.eps))
            if method == "lc_residual_direct":
                target_u = residual - eps_vec
                pred_u = target_u
                direction = residual
                pred_target_cos = 1.0
                s_value = 1.0
            else:
                target_u = residual - eps_vec
                pred_u = _predict_latent_velocity(op, Z[i], int(y_arr[i]), eps_vec, 0.0)
                r_hat = eps_vec + pred_u
                direction, _ = _unit(r_hat, eps=float(latent_cfg.eps))
                pred_target_cos = _cosine(pred_u, target_u, eps=float(latent_cfg.eps))
                s_value = 0.0
            if not np.isfinite(pred_target_cos):
                pred_target_cos = 0.0
            direction, direction_norm = _unit(direction, eps=float(latent_cfg.eps))
            if direction_norm <= float(latent_cfg.eps):
                direction, direction_norm = _unit(rng.normal(size=(Z.shape[1],)), eps=float(latent_cfg.eps))
            d_min = float(margins[i])
            if eta_safe is None:
                gamma_used = gamma_requested
                safe_upper_bound = float("inf")
                safe_radius_ratio = 1.0
            else:
                safe_upper_bound = float(eta_safe) * d_min / (direction_norm + latent_cfg.eps)
                gamma_used = min(gamma_requested, safe_upper_bound)
                safe_radius = float(eta_safe) * d_min
                safe_radius_ratio = float(abs(gamma_used) * direction_norm / (safe_radius + latent_cfg.eps)) if safe_radius > 0 else 0.0
            W_i = (gamma_used * direction).astype(np.float32)
            z_aug.append((Z[i] + W_i).astype(np.float32))
            y_aug.append(int(y_arr[i]))
            tid_aug.append(tid_arr[i])
            pred_cosines.append(float(pred_target_cos))
            rows.append(
                {
                    "anchor_index": int(i),
                    "tid": tid_arr[i],
                    "class_id": int(y_arr[i]),
                    "candidate_order": int(c),
                    "slot_index": int(len(rows)),
                    "direction_source": "label_consistent_latent_residual_operator",
                    "template_id": -1,
                    "template_rank": -1,
                    "template_sign": np.nan,
                    "template_response_abs": np.nan,
                    "selected_template_rank": -1,
                    "selected_template_response_abs": np.nan,
                    "direction_id": -1,
                    "latent_s": float(s_value),
                    "latent_eps_norm": float(eps_norm),
                    "latent_target_index": int(sample["target_index"]),
                    "latent_target_class": int(sample["target_class"]),
                    "latent_target_dist": float(sample["target_dist"]),
                    "latent_target_sampling_prob": float(sample["target_sampling_prob"]),
                    "latent_target_sampling_prob_geo": float(sample["target_sampling_prob_geo"]),
                    "latent_residual_norm": float(sample["residual_norm"]),
                    "latent_pred_velocity_norm": float(np.linalg.norm(pred_u)),
                    "latent_pred_target_cosine": float(pred_target_cos),
                    "latent_fallback_flag": bool(sample["fallback_flag"]),
                    "latent_fallback_reason": str(sample["fallback_reason"]),
                    "lc_utility": float(sample["lc_utility"]),
                    "lc_margin": float(sample["lc_margin"]),
                    "lc_margin_target": float(sample["lc_margin_target"]),
                    "lc_fallback_flag": bool(sample["lc_fallback_flag"]),
                    "lc_fallback_reason": str(sample["lc_fallback_reason"]),
                    "gamma_requested": float(gamma_requested),
                    "gamma_used": float(gamma_used),
                    "gamma_used_ratio": float(gamma_used / gamma_requested) if abs(gamma_requested) > latent_cfg.eps else np.nan,
                    "direction_norm": float(direction_norm),
                    "pre_safe_displacement_norm": float(abs(gamma_requested) * direction_norm),
                    "post_safe_displacement_norm": float(np.linalg.norm(W_i)),
                    "z_displacement_norm": float(np.linalg.norm(W_i)),
                    "safe_upper_bound": float(safe_upper_bound),
                    "safe_radius_ratio": float(safe_radius_ratio),
                    "manifold_margin": d_min,
                    "is_clipped": float(gamma_requested > safe_upper_bound + 1e-9),
                    "selection_stage": "label_consistent_latent_residual_operator",
                    "selector_name": method,
                    "feasible_flag": 1.0,
                    "selector_accept_flag": 1.0,
                    "_direction_vec": direction,
                }
            )

    diversity = _direction_diversity(rows, eps=float(latent_cfg.eps))
    for row in rows:
        row.pop("_direction_vec", None)
    lc_summary = dict(sampler["summary"])
    lc_summary["lc_fallback_reason"] = ";".join(sorted({r for r in sampler["fallback_reasons"] if r}))
    latent_summary = {
        **dict(target_out["summary"]),
        **op_summary,
        **diversity,
        **lc_summary,
        "latent_pred_target_cosine_mean": float(np.mean(pred_cosines)) if pred_cosines else np.nan,
        "latent_flow_epochs": int(latent_cfg.flow_epochs),
        "latent_flow_batch_size": int(latent_cfg.flow_batch_size),
        "latent_flow_lr": float(latent_cfg.flow_lr),
        "latent_flow_weight_decay": float(latent_cfg.flow_weight_decay),
        "latent_rbf_tau_floor": float(latent_cfg.rbf_tau_floor),
    }
    direction_meta = {
        "bank_source": "label_consistent_latent_residual_operator",
        "direction_source": "label_consistent_latent_residual_operator",
        "operator_source": method,
        **latent_summary,
    }
    return materialize_z_aug_out(
        z_aug=np.stack(z_aug).astype(np.float32) if z_aug else np.empty((0, Z.shape[1]), dtype=np.float32),
        y_aug=np.asarray(y_aug, dtype=np.int64),
        tid_aug=np.asarray(tid_aug, dtype=object),
        audit_rows=rows,
        train_recs=train_recs,
        mean_log=mean_log,
        direction_bank_meta=direction_meta,
        effective_k=1,
        eta_safe=eta_safe,
        algo_name=method,
        engine_id=method,
        extra_meta={
            **latent_summary,
            "selection_stage": "label_consistent_latent_residual_operator",
            "selector_name": method,
            "multi_template_pairs": 0,
            "template_usage_entropy": 0.0,
            "top_template_concentration": 0.0,
        },
    )
