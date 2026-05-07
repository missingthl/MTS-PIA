from __future__ import annotations

from typing import Dict, List

import numpy as np

from .direction_banks import build_direction_bank_for_args as _build_direction_bank_for_args
from .materialize import materialize_z_aug_out as _materialize_z_aug_out
from .state import TrialRecord
from .template_slots import build_top_response_template_slots as _build_top_response_template_slots


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
        direction_meta=zpia_meta,
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
            "template_response_top1_mean": float(slots.get("template_response_top1_mean", np.nan)),
            "template_response_top5_mean": float(slots.get("template_response_top5_mean", np.nan)),
            "template_response_gap_top1_top5_mean": float(slots.get("template_response_gap_top1_top5_mean", np.nan)),
            "template_response_entropy_mean": float(slots.get("template_response_entropy_mean", np.nan)),
            "pre_safe_displacement_norm_mean": float(slots.get("pre_safe_displacement_norm_mean", np.nan)),
            "post_safe_displacement_norm_mean": float(slots.get("post_safe_displacement_norm_mean", np.nan)),
            "gamma_used_ratio_mean": float(slots.get("gamma_used_ratio_mean", np.nan)),
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
