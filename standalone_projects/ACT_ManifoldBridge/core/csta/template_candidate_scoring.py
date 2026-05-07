from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .diagnostics import normalize_unit_interval
from .template_policies import (
    FV_SAFE_RATIO_TARGET,
    FV_TOP_K,
    response_vector_for_anchor,
    template_responses_for_anchor,
)


def build_template_candidate_components(
    *,
    args,
    X_train_z: np.ndarray,
    zpia_bank: np.ndarray,
    margins: np.ndarray,
    idx: int,
    template_id: int,
    template_sign: float,
    eta_safe,
    y_arr: Optional[np.ndarray] = None,
    direction_meta: Optional[Dict[str, object]] = None,
    response_class_means: Optional[Dict[int, np.ndarray]] = None,
    eps: float = 1e-12,
) -> Dict[str, object]:
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
        safe_radius_ratio = float(abs(gamma_used) * direction_norm / (safe_radius + eps)) if safe_radius > 0 else 0.0
    W_i = (float(template_sign) * gamma_used * direction).astype(np.float32)
    z_response = response_vector_for_anchor(
        idx=idx,
        X_train_z=X_train_z,
        y_arr=y_arr,
        direction_meta=direction_meta,
        response_class_means=response_class_means,
    )
    response_abs = float(abs(np.dot(z_response, direction)))
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
        "pre_safe_displacement_norm": float(abs(gamma_requested) * direction_norm),
        "post_safe_displacement_norm": z_delta_norm,
        "gamma_used_ratio": float(gamma_used / gamma_requested) if abs(gamma_requested) > eps else np.nan,
        "z_delta_norm": z_delta_norm,
        "is_clipped": float(gamma_requested > (safe_upper_bound + 1e-9)),
        "feasible": bool(feasible),
        "reject_zero_gamma": int(zero_gamma),
        "reject_safe_radius": int(safe_radius_bad),
        "reject_zero_direction": int(zero_direction),
        "reject_zero_margin": int(zero_margin),
    }


def select_fv_candidate(
    *,
    mode: str,
    args,
    X_train_z: np.ndarray,
    zpia_bank: np.ndarray,
    margins: np.ndarray,
    idx: int,
    candidate_order: int,
    seed: int,
    template_ids_used: List[int],
    eta_safe,
    y_arr: Optional[np.ndarray] = None,
    direction_meta: Optional[Dict[str, object]] = None,
    response_class_means: Optional[Dict[int, np.ndarray]] = None,
    fv_top_k: int = FV_TOP_K,
    fv_rho: float = FV_SAFE_RATIO_TARGET,
    eps: float = 1e-12,
) -> Dict[str, object]:
    responses = template_responses_for_anchor(
        idx=idx,
        X_train_z=X_train_z,
        zpia_bank=zpia_bank,
        y_arr=y_arr,
        direction_meta=direction_meta,
        response_class_means=response_class_means,
    )
    top_indices = np.lexsort((np.arange(zpia_bank.shape[0]), -responses))[: min(fv_top_k, zpia_bank.shape[0])]
    pool: List[Dict[str, object]] = []
    for rank, template_id in enumerate(top_indices):
        for sign_val in (1.0, -1.0):
            cand = build_template_candidate_components(
                args=args,
                X_train_z=X_train_z,
                zpia_bank=zpia_bank,
                margins=margins,
                idx=idx,
                template_id=int(template_id),
                template_sign=sign_val,
                eta_safe=eta_safe,
                y_arr=y_arr,
                direction_meta=direction_meta,
                response_class_means=response_class_means,
                eps=eps,
            )
            cand["template_rank"] = int(rank)
            pool.append(cand)

    rel = normalize_unit_interval(np.asarray([float(c["response_abs"]) for c in pool], dtype=np.float64))
    safe_raw = np.asarray([-abs(float(c["safe_radius_ratio"]) - fv_rho) for c in pool], dtype=np.float64)
    safe_bal = normalize_unit_interval(safe_raw)
    disp = normalize_unit_interval(np.asarray([float(c["z_delta_norm"]) for c in pool], dtype=np.float64))
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
