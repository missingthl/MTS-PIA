from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from core.curriculum import estimate_local_manifold_margins
from .diagnostics import template_response_profile, template_usage_stats
from .state import TrialRecord
from .template_candidate_scoring import build_template_candidate_components, select_fv_candidate
from .template_policies import (
    FV_SELECTOR_MODES,
    FV_TOP_K,
    build_response_class_means,
    prepare_template_neighbor_indices,
    response_vector_for_anchor,
    select_template_ids_for_policy,
)


def resolve_multi_template_pairs(*, args, effective_k: int, top1_only: bool) -> int:
    if int(args.multiplier) <= 0:
        raise ValueError("multi-template pools require --multiplier > 0.")
    if top1_only:
        pairs = 1
        if pairs > effective_k:
            raise ValueError(f"--multi-template-pairs={pairs} exceeds effective zPIA bank size {effective_k}.")
        return pairs
    if int(args.multiplier) % 2 != 0:
        raise ValueError("multi-template pools require an even --multiplier for +/- template slots.")
    configured = int(getattr(args, "multi_template_pairs", 0))
    pairs = configured if configured > 0 else int(args.multiplier) // 2
    if 2 * pairs != int(args.multiplier):
        raise ValueError(
            "--multi-template-pairs must satisfy 2 * pairs == multiplier for zpia_multidir_pool "
            "so every template receives +/- slots."
        )
    if pairs <= 0:
        raise ValueError("--multi-template-pairs must be positive.")
    if pairs > effective_k:
        raise ValueError(f"--multi-template-pairs={pairs} exceeds effective zPIA bank size {effective_k}.")
    return pairs


def build_top_response_template_slots(
    *,
    args,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    train_recs: List[TrialRecord],
    zpia_bank: np.ndarray,
    seed: int,
    candidate_rows: Optional[List[Dict[str, object]]] = None,
    top1_only: bool = False,
    eta_safe: Optional[float] = 0.75,
    direction_meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build PIA candidate slots without realizing them in raw space."""

    zpia_bank = np.asarray(zpia_bank, dtype=np.float64)
    if zpia_bank.ndim != 2 or zpia_bank.shape[0] <= 0:
        raise ValueError("multi-template zPIA requires a non-empty 2D direction bank.")
    pairs = resolve_multi_template_pairs(args=args, effective_k=zpia_bank.shape[0], top1_only=top1_only)

    tid_arr = np.asarray([record.tid for record in train_recs], dtype=object)
    y_arr = np.asarray(y_train, dtype=np.int64).ravel()
    response_class_means = build_response_class_means(X_train_z, y_arr, direction_meta)
    tid_to_idx = {tid: i for i, tid in enumerate(tid_arr)}
    margins = estimate_local_manifold_margins(X_train_z, y_arr)

    mode = getattr(args, "template_selection", "top_response")
    group_size = int(getattr(args, "group_size", 5))
    is_fv_selector = mode in FV_SELECTOR_MODES
    fv_top_k = FV_TOP_K
    neighbor_indices = prepare_template_neighbor_indices(
        mode=mode,
        X_train_z=X_train_z,
        y_arr=y_arr,
        group_size=group_size,
        seed=seed,
    )

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
    response_top1_vals: List[float] = []
    response_top5_mean_vals: List[float] = []
    response_gap_vals: List[float] = []
    response_entropy_vals: List[float] = []
    pre_safe_norms: List[float] = []
    post_safe_norms: List[float] = []
    gamma_used_ratios: List[float] = []
    eps = 1e-12
    pair_cycle = 2 * pairs

    for slot_idx, spec in enumerate(row_specs):
        idx = int(spec["anchor_index"])
        tid = spec["tid"]
        candidate_order = int(spec["candidate_order"])
        z_response = response_vector_for_anchor(
            idx=idx,
            X_train_z=X_train_z,
            y_arr=y_arr,
            direction_meta=direction_meta,
            response_class_means=response_class_means,
        )
        response_profile = template_response_profile(z_response, zpia_bank)
        if is_fv_selector:
            selected = select_fv_candidate(
                mode=mode,
                args=args,
                X_train_z=X_train_z,
                zpia_bank=zpia_bank,
                margins=margins,
                idx=idx,
                candidate_order=candidate_order,
                seed=seed,
                template_ids_used=template_ids_used,
                eta_safe=eta_safe,
                y_arr=y_arr,
                direction_meta=direction_meta,
                response_class_means=response_class_means,
                fv_top_k=fv_top_k,
                eps=eps,
            )
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
            pre_safe_displacement_norm = float(selected.get("pre_safe_displacement_norm", abs(gamma_requested) * direction_norm))
            post_safe_displacement_norm = float(selected.get("post_safe_displacement_norm", np.linalg.norm(W_i)))
            gamma_used_ratio = float(selected.get("gamma_used_ratio", gamma_used / gamma_requested if abs(gamma_requested) > eps else np.nan))
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
            top_ids = select_template_ids_for_policy(
                mode=mode,
                idx=idx,
                pairs=pairs,
                X_train_z=X_train_z,
                zpia_bank=zpia_bank,
                seed=seed,
                y_arr=y_arr,
                direction_meta=direction_meta,
                response_class_means=response_class_means,
                neighbor_indices=neighbor_indices,
            )
            pair_pos = (candidate_order % pair_cycle) // 2
            template_sign = 1.0 if (candidate_order % 2 == 0) else -1.0
            template_id = int(top_ids[pair_pos])
            comp = build_template_candidate_components(
                args=args,
                X_train_z=X_train_z,
                zpia_bank=zpia_bank,
                margins=margins,
                idx=idx,
                template_id=template_id,
                template_sign=template_sign,
                eta_safe=eta_safe,
                y_arr=y_arr,
                direction_meta=direction_meta,
                response_class_means=response_class_means,
                eps=eps,
            )
            direction_norm = float(comp["direction_norm"])
            d_min = float(comp["manifold_margin"])
            gamma_requested = float(comp["gamma_requested"])
            gamma_used = float(comp["gamma_used"])
            safe_upper_bound = float(comp["safe_upper_bound"])
            safe_radius_ratio = float(comp["safe_radius_ratio"])
            W_i = np.asarray(comp["W_i"], dtype=np.float32)
            response_abs = float(comp["response_abs"])
            is_clipped = float(comp["is_clipped"])
            pre_safe_displacement_norm = float(comp.get("pre_safe_displacement_norm", abs(gamma_requested) * direction_norm))
            post_safe_displacement_norm = float(comp.get("post_safe_displacement_norm", np.linalg.norm(W_i)))
            gamma_used_ratio = float(comp.get("gamma_used_ratio", gamma_used / gamma_requested if abs(gamma_requested) > eps else np.nan))
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
            pre_safe_displacement_norm = float(np.linalg.norm(W_i))
            post_safe_displacement_norm = float(np.linalg.norm(W_i))
            gamma_used_ratio = np.nan

        z_aug.append((np.asarray(X_train_z[idx], dtype=np.float64) + W_i).astype(np.float32))
        y_aug.append(int(y_arr[idx]))
        tid_aug.append(tid)
        w_slots.append(W_i.astype(np.float32))
        template_ids_used.append(template_id)
        response_top1_vals.append(float(response_profile["template_response_top1"]))
        response_top5_mean_vals.append(float(response_profile["template_response_top5_mean"]))
        response_gap_vals.append(float(response_profile["template_response_gap_top1_top5"]))
        response_entropy_vals.append(float(response_profile["template_response_entropy"]))
        pre_safe_norms.append(float(pre_safe_displacement_norm))
        post_safe_norms.append(float(post_safe_displacement_norm))
        gamma_used_ratios.append(float(gamma_used_ratio))
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
                "template_response_top1": float(response_profile["template_response_top1"]),
                "template_response_top5_mean": float(response_profile["template_response_top5_mean"]),
                "template_response_gap_top1_top5": float(response_profile["template_response_gap_top1_top5"]),
                "template_response_entropy": float(response_profile["template_response_entropy"]),
                "selected_template_rank": int(pair_pos),
                "selected_template_response_abs": response_abs,
                "direction_id": template_id,
                "sign": float(template_sign),
                "gamma_requested": gamma_requested,
                "gamma_used": float(gamma_used),
                "gamma_used_ratio": float(gamma_used_ratio),
                "direction_norm": direction_norm,
                "pre_safe_displacement_norm": float(pre_safe_displacement_norm),
                "post_safe_displacement_norm": float(post_safe_displacement_norm),
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

    usage_stats = template_usage_stats(template_ids_used)
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
        "template_response_top1_mean": float(np.nanmean(response_top1_vals)) if response_top1_vals else np.nan,
        "template_response_top5_mean": float(np.nanmean(response_top5_mean_vals)) if response_top5_mean_vals else np.nan,
        "template_response_gap_top1_top5_mean": float(np.nanmean(response_gap_vals)) if response_gap_vals else np.nan,
        "template_response_entropy_mean": float(np.nanmean(response_entropy_vals)) if response_entropy_vals else np.nan,
        "pre_safe_displacement_norm_mean": float(np.nanmean(pre_safe_norms)) if pre_safe_norms else np.nan,
        "post_safe_displacement_norm_mean": float(np.nanmean(post_safe_norms)) if post_safe_norms else np.nan,
        "gamma_used_ratio_mean": float(np.nanmean(gamma_used_ratios)) if gamma_used_ratios else np.nan,
    }
