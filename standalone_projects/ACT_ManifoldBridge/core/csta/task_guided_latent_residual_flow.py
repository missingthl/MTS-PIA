from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from core.bridge import bridge_single, logvec_to_spd
from core.curriculum import estimate_local_manifold_margins
from core.resnet1d import ResNet1DClassifier

from .latent_residual_flow import (
    LatentResidualConfig,
    _cosine,
    _direction_diversity,
    _effective_rank,
    _pairwise_cosine_mean,
    _unit,
    _build_rbf_sampler,
    _predict_latent_velocity,
    fit_latent_residual_operator,
)
from .materialize import materialize_z_aug_out
from .state import TrialRecord


TASK_GUIDED_LATENT_METHODS = {"task_guided_residual_direct", "task_guided_latent_residual_flow"}


@dataclass(frozen=True)
class TaskGuidanceConfig:
    beta: float = 1.0
    margin_min: float = 0.0
    lambda_margin: float = 1.0
    warmup_epochs: int = 10
    max_candidates: int = 0
    eps: float = 1e-12


def _train_warmup_resnet1d(
    *,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    device: str,
    epochs: int,
    lr: float,
    batch_size: int,
) -> Dict[str, object]:
    """Train a train-only warm-up model used only for residual utility scoring."""
    y = np.asarray(y_train, dtype=np.int64).ravel()
    n_classes = int(np.max(y)) + 1
    torch.manual_seed(int(seed) + 11821)
    np.random.seed(int(seed) + 11821)
    dev = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = ResNet1DClassifier(in_channels=int(X_train_raw.shape[1]), num_classes=n_classes).to(dev)
    ds = TensorDataset(torch.from_numpy(np.asarray(X_train_raw, dtype=np.float32)), torch.from_numpy(y))
    generator = torch.Generator()
    generator.manual_seed(int(seed) + 11821)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=True, generator=generator)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    model.train()
    losses: List[float] = []
    for _ in range(int(epochs)):
        for bx, by in loader:
            bx = bx.to(dev, non_blocking=True)
            by = by.to(dev, non_blocking=True)
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
    return {
        "model": model.eval(),
        "device": dev,
        "task_warmup_train_loss_mean": float(np.mean(losses)) if losses else np.nan,
        "task_warmup_train_epochs": int(epochs),
    }


def _score_logits(model: torch.nn.Module, device: torch.device, X: np.ndarray, batch_size: int) -> np.ndarray:
    if len(X) == 0:
        return np.empty((0, 0), dtype=np.float64)
    ds = TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32)))
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)
    outs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (bx,) in loader:
            logits = model(bx.to(device, non_blocking=True))
            outs.append(logits.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outs, axis=0) if outs else np.empty((0, 0), dtype=np.float64)


def _softmax_margin_and_ce(logits: np.ndarray, y_true: int) -> tuple[float, float, int]:
    logits = np.asarray(logits, dtype=np.float64).ravel()
    if logits.size == 0 or not np.all(np.isfinite(logits)):
        return np.nan, np.nan, -1
    y_idx = int(y_true)
    shifted = logits - float(np.max(logits))
    logsumexp = float(np.log(np.sum(np.exp(shifted))) + np.max(logits))
    ce = float(logsumexp - logits[y_idx])
    pred = int(np.argmax(logits))
    if logits.size <= 1:
        margin = float(logits[y_idx])
    else:
        masked = logits.copy()
        masked[y_idx] = -np.inf
        margin = float(logits[y_idx] - np.max(masked))
    return ce, margin, pred


def build_task_guided_sampler(
    *,
    Z: np.ndarray,
    y: np.ndarray,
    X_train_raw: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    warmup: Dict[str, object],
    seed: int,
    latent_cfg: LatentResidualConfig,
    task_cfg: TaskGuidanceConfig,
    gamma_requested: float,
    eta_safe: float | None,
    batch_size: int,
) -> Dict[str, object]:
    """Build p_task(j|i) from train-only same-class residual candidates."""
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).ravel()
    geo = _build_rbf_sampler(Z, y, cfg=latent_cfg)
    margins = estimate_local_manifold_margins(Z, y)
    rng = np.random.default_rng(int(seed) + 11903)

    task_probs: List[np.ndarray] = []
    task_candidates: List[np.ndarray] = []
    candidate_meta: List[List[Dict[str, object]]] = []
    fallback_flags: List[bool] = []
    fallback_reasons: List[str] = []
    all_utilities: List[float] = []
    all_margins: List[float] = []
    all_weights: List[float] = []
    all_bad_margin_mass: List[float] = []
    all_wrong_pred_mass: List[float] = []
    entropies: List[float] = []
    effective_supports: List[float] = []
    kls: List[float] = []
    invalid_rates: List[float] = []
    x_to_score: List[np.ndarray] = []
    score_refs: List[tuple[int, int]] = []

    tid_to_rec = {record.tid: record for record in train_recs}
    tid_arr = np.asarray([record.tid for record in train_recs], dtype=object)
    per_anchor_records: List[List[Dict[str, object]]] = []

    for i in range(len(Z)):
        cand = np.asarray(geo["candidates"][i], dtype=np.int64)
        p_geo = np.asarray(geo["probs"][i], dtype=np.float64)
        if int(task_cfg.max_candidates) > 0 and cand.size > int(task_cfg.max_candidates):
            order = np.argsort(-p_geo)[: int(task_cfg.max_candidates)]
            cand = cand[order]
            p_geo = p_geo[order]
            p_geo = p_geo / max(float(np.sum(p_geo)), float(task_cfg.eps))
        records: List[Dict[str, object]] = []
        for pos, j in enumerate(cand):
            raw = Z[int(j)] - Z[i]
            direction, direction_norm = _unit(raw, eps=float(task_cfg.eps))
            d_min = float(margins[i])
            if direction_norm <= float(task_cfg.eps):
                gamma_used = 0.0
                safe_upper_bound = 0.0
                safe_radius_ratio = 0.0
            elif eta_safe is None:
                gamma_used = float(gamma_requested)
                safe_upper_bound = float("inf")
                safe_radius_ratio = 1.0
            else:
                safe_upper_bound = float(eta_safe) * d_min / (direction_norm + float(task_cfg.eps))
                gamma_used = min(float(gamma_requested), safe_upper_bound)
                safe_radius = float(eta_safe) * d_min
                safe_radius_ratio = (
                    float(abs(gamma_used) * direction_norm / (safe_radius + float(task_cfg.eps))) if safe_radius > 0 else 0.0
                )
            transport_error = np.nan
            bridge_success = False
            x_aug_np = None
            if abs(gamma_used) > float(task_cfg.eps) and direction_norm > float(task_cfg.eps):
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
            valid_prelogit = bool(bridge_success and x_aug_np is not None and np.isfinite(transport_error) and abs(gamma_used) > float(task_cfg.eps))
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
        logits = logits_all[row_idx]
        ce, margin, pred = _softmax_margin_and_ce(logits, int(y[anchor_i]))
        valid = bool(np.isfinite(ce) and np.isfinite(margin))
        penalty = float(task_cfg.lambda_margin) * max(0.0, float(task_cfg.margin_min) - float(margin if np.isfinite(margin) else 0.0)) ** 2
        rec.update(
            {
                "ce": float(ce),
                "margin": float(margin),
                "pred": int(pred),
                "utility": float(ce - penalty) if valid else np.nan,
                "valid": valid,
            }
        )

    for i, records in enumerate(per_anchor_records):
        cand = np.asarray([int(r["target_index"]) for r in records], dtype=np.int64)
        p_geo = np.asarray([float(r["target_sampling_prob_geo"]) for r in records], dtype=np.float64)
        if p_geo.size:
            p_geo = p_geo / max(float(np.sum(p_geo)), float(task_cfg.eps))
        valid = np.asarray([bool(r["valid"]) for r in records], dtype=bool)
        utilities = np.asarray([float(r["utility"]) if np.isfinite(float(r["utility"])) else np.nan for r in records], dtype=np.float64)
        fallback = False
        reason = ""
        if cand.size == 0:
            p_task = np.asarray([], dtype=np.float64)
            fallback = True
            reason = "singleton_class"
        elif not np.any(valid):
            p_task = p_geo.copy()
            fallback = True
            reason = "all_invalid_candidates"
        else:
            valid_u = utilities[valid]
            u_mean = float(np.mean(valid_u))
            u_std = float(np.std(valid_u))
            if not np.isfinite(u_std) or u_std <= float(task_cfg.eps):
                u_std = 1.0
            u_std_all = np.zeros_like(utilities, dtype=np.float64)
            u_std_all[valid] = np.clip((utilities[valid] - u_mean) / u_std, -3.0, 3.0)
            weights = np.zeros_like(p_geo, dtype=np.float64)
            weights[valid] = p_geo[valid] * np.exp(float(task_cfg.beta) * u_std_all[valid])
            w_sum = float(np.sum(weights))
            if not np.isfinite(w_sum) or w_sum <= float(task_cfg.eps):
                p_task = p_geo.copy()
                fallback = True
                reason = "degenerate_task_weights"
            else:
                p_task = weights / w_sum
        task_candidates.append(cand)
        task_probs.append(p_task.astype(np.float64))
        candidate_meta.append(records)
        fallback_flags.append(bool(fallback))
        fallback_reasons.append(reason)
        if p_task.size:
            entropies.append(float(-np.sum(p_task * np.log(p_task + float(task_cfg.eps)))))
            effective_supports.append(float(np.exp(entropies[-1])))
            if p_geo.size == p_task.size and np.sum(p_geo) > 0:
                kls.append(float(np.sum(p_task * (np.log(p_task + float(task_cfg.eps)) - np.log(p_geo + float(task_cfg.eps))))))
            margins_i = np.asarray([float(r["margin"]) for r in records], dtype=np.float64)
            preds_i = np.asarray([int(r["pred"]) for r in records], dtype=np.int64)
            valid_margin = np.isfinite(margins_i)
            bad_mass = float(np.sum(p_task[(margins_i < float(task_cfg.margin_min)) & valid_margin])) if valid_margin.any() else 0.0
            wrong_mass = float(np.sum(p_task[preds_i != int(y[i])])) if preds_i.size else 0.0
            all_bad_margin_mass.append(bad_mass)
            all_wrong_pred_mass.append(wrong_mass)
            all_weights.extend(float(w) for w in p_task)
        invalid_rates.append(float(1.0 - np.mean(valid.astype(float))) if valid.size else 1.0)
        all_utilities.extend(float(v) for v in utilities[np.isfinite(utilities)])
        all_margins.extend(float(r["margin"]) for r in records if np.isfinite(float(r["margin"])))

    return {
        "candidates": task_candidates,
        "probs": task_probs,
        "geo_sampler": geo,
        "candidate_meta": candidate_meta,
        "fallback_flags": np.asarray(fallback_flags, dtype=bool),
        "fallback_reasons": fallback_reasons,
        "summary": {
            "task_utility_mean": float(np.mean(all_utilities)) if all_utilities else np.nan,
            "task_utility_std": float(np.std(all_utilities)) if all_utilities else np.nan,
            "task_margin_mean": float(np.mean(all_margins)) if all_margins else np.nan,
            "task_margin_std": float(np.std(all_margins)) if all_margins else np.nan,
            "task_bad_margin_mass": float(np.mean(all_bad_margin_mass)) if all_bad_margin_mass else np.nan,
            "task_wrong_pred_mass": float(np.mean(all_wrong_pred_mass)) if all_wrong_pred_mass else np.nan,
            "task_sampling_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "task_sampling_effective_support": float(np.mean(effective_supports)) if effective_supports else 0.0,
            "task_kl_task_vs_geo": float(np.mean(kls)) if kls else 0.0,
            "task_guidance_fallback_rate": float(np.mean(fallback_flags)) if fallback_flags else 0.0,
            "task_invalid_candidate_rate": float(np.mean(invalid_rates)) if invalid_rates else 1.0,
            "task_warmup_train_epochs": int(task_cfg.warmup_epochs),
            "task_warmup_train_loss_mean": float(warmup.get("task_warmup_train_loss_mean", np.nan)),
            "task_guidance_beta": float(task_cfg.beta),
            "task_guidance_margin_min": float(task_cfg.margin_min),
            "task_guidance_lambda_margin": float(task_cfg.lambda_margin),
            "task_guidance_max_candidates": int(task_cfg.max_candidates),
        },
    }


def _sample_task_residual(
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
    task_fallback = bool(sampler["fallback_flags"][int(anchor_index)])
    task_fallback_reason = str(sampler["fallback_reasons"][int(anchor_index)])
    if len(cand) == 0:
        residual, residual_norm = _unit(rng.normal(size=(z_dim,)), eps=eps)
        return {
            "target_index": -1,
            "target_class": -1,
            "target_dist": float(residual_norm),
            "target_sampling_prob": 0.0,
            "residual": residual,
            "residual_norm": float(residual_norm),
            "fallback_flag": True,
            "fallback_reason": "singleton_class_seeded_random",
            "task_utility": np.nan,
            "task_margin": np.nan,
            "task_guidance_fallback_flag": True,
            "task_guidance_fallback_reason": "singleton_class",
            "target_sampling_prob_geo": 0.0,
        }
    pos = int(rng.choice(np.arange(len(cand)), p=probs))
    rec = dict(meta[pos])
    return {
        "target_index": int(rec["target_index"]),
        "target_class": int(rec["target_class"]),
        "target_dist": float(rec["target_dist"]),
        "target_sampling_prob": float(probs[pos]),
        "residual": np.asarray(rec["residual"], dtype=np.float64),
        "residual_norm": float(rec["residual_norm"]),
        "fallback_flag": False,
        "fallback_reason": "",
        "task_utility": float(rec.get("utility", np.nan)),
        "task_margin": float(rec.get("margin", np.nan)),
        "task_guidance_fallback_flag": task_fallback,
        "task_guidance_fallback_reason": task_fallback_reason,
        "target_sampling_prob_geo": float(rec.get("target_sampling_prob_geo", np.nan)),
    }


def build_task_guided_targets(
    *,
    Z: np.ndarray,
    y: np.ndarray,
    task_sampler: Dict[str, object],
    seed: int,
    latent_cfg: LatentResidualConfig,
) -> Dict[str, object]:
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).ravel()
    n, z_dim = Z.shape
    rng = np.random.default_rng(int(seed) + 12011)
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
        sample = _sample_task_residual(sampler=task_sampler, anchor_index=i, rng=rng, z_dim=z_dim, eps=float(latent_cfg.eps))
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
        "latent_target_sampling_entropy": float(task_sampler["summary"].get("task_sampling_entropy", np.nan)),
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


def build_task_guided_latent_aug_out(
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
    if method not in TASK_GUIDED_LATENT_METHODS:
        raise ValueError(f"Unknown task-guided latent method: {method}")
    if str(getattr(args, "template_selection", "top_response")) != "top_response":
        raise ValueError("Task-Guided Latent Residual Flow must not use template-selection modes.")
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
    task_cfg = TaskGuidanceConfig(
        beta=float(getattr(args, "task_guidance_beta", 1.0)),
        margin_min=float(getattr(args, "task_guidance_margin_min", 0.0)),
        lambda_margin=float(getattr(args, "task_guidance_lambda_margin", 1.0)),
        warmup_epochs=int(getattr(args, "task_guidance_warmup_epochs", 10)),
        max_candidates=int(getattr(args, "task_guidance_max_candidates", 0)),
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
        epochs=int(task_cfg.warmup_epochs),
        lr=float(getattr(args, "lr", 1e-3)),
        batch_size=int(getattr(args, "batch_size", 64)),
    )
    task_sampler = build_task_guided_sampler(
        Z=Z,
        y=y_arr,
        X_train_raw=X_train_raw,
        train_recs=train_recs,
        mean_log=mean_log,
        warmup=warmup,
        seed=seed,
        latent_cfg=latent_cfg,
        task_cfg=task_cfg,
        gamma_requested=gamma_requested,
        eta_safe=eta_safe,
        batch_size=int(getattr(args, "batch_size", 64)),
    )
    target_out = build_task_guided_targets(Z=Z, y=y_arr, task_sampler=task_sampler, seed=seed, latent_cfg=latent_cfg)
    if method == "task_guided_residual_direct":
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
    rng = np.random.default_rng(int(seed) + 12097)
    z_aug: List[np.ndarray] = []
    y_aug: List[int] = []
    tid_aug: List[object] = []
    rows: List[Dict[str, object]] = []
    pred_cosines: List[float] = []
    for i in range(len(Z)):
        for c in range(multiplier):
            sample = _sample_task_residual(
                sampler=task_sampler,
                anchor_index=i,
                rng=rng,
                z_dim=Z.shape[1],
                eps=float(latent_cfg.eps),
            )
            residual = np.asarray(sample["residual"], dtype=np.float64)
            eps_vec, eps_norm = _unit(rng.normal(size=(Z.shape[1],)), eps=float(latent_cfg.eps))
            if method == "task_guided_residual_direct":
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
                    "direction_source": "task_guided_latent_residual_operator",
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
                    "task_utility": float(sample["task_utility"]),
                    "task_margin": float(sample["task_margin"]),
                    "task_guidance_fallback_flag": bool(sample["task_guidance_fallback_flag"]),
                    "task_guidance_fallback_reason": str(sample["task_guidance_fallback_reason"]),
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
                    "selection_stage": "task_guided_latent_residual_operator",
                    "selector_name": method,
                    "feasible_flag": 1.0,
                    "selector_accept_flag": 1.0,
                    "_direction_vec": direction,
                }
            )

    diversity = _direction_diversity(rows, eps=float(latent_cfg.eps))
    for row in rows:
        row.pop("_direction_vec", None)
    task_summary = dict(task_sampler["summary"])
    task_summary["task_guidance_fallback_reason"] = ";".join(
        sorted({r for r in task_sampler["fallback_reasons"] if r})
    )
    latent_summary = {
        **dict(target_out["summary"]),
        **op_summary,
        **diversity,
        **task_summary,
        "latent_pred_target_cosine_mean": float(np.mean(pred_cosines)) if pred_cosines else np.nan,
        "latent_flow_epochs": int(latent_cfg.flow_epochs),
        "latent_flow_batch_size": int(latent_cfg.flow_batch_size),
        "latent_flow_lr": float(latent_cfg.flow_lr),
        "latent_flow_weight_decay": float(latent_cfg.flow_weight_decay),
        "latent_rbf_tau_floor": float(latent_cfg.rbf_tau_floor),
    }
    direction_meta = {
        "bank_source": "task_guided_latent_residual_operator",
        "direction_source": "task_guided_latent_residual_operator",
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
            "selection_stage": "task_guided_latent_residual_operator",
            "selector_name": method,
            "multi_template_pairs": 0,
            "template_usage_entropy": 0.0,
            "top_template_concentration": 0.0,
        },
    )
