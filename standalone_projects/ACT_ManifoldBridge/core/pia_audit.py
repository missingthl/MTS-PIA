from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from .pia_operator import pia_operator_metadata


P0_CANDIDATE_AUDIT_COLUMNS = [
    "dataset",
    "seed",
    "method",
    "operator_name",
    "dictionary_estimator",
    "activation_policy",
    "activation_scope",
    "activation_topk",
    "activation_tau",
    "safe_generator",
    "bridge_realizer",
    "candidate_uid",
    "anchor_index",
    "tid",
    "class_id",
    "candidate_order",
    "slot_index",
    "template_id",
    "template_rank",
    "template_sign",
    "template_response_abs",
    "gamma_requested",
    "gamma_used",
    "eta_safe",
    "direction_norm",
    "z_displacement_norm",
    "safe_upper_bound",
    "safe_radius_ratio",
    "safe_clip_flag",
    "gamma_zero_flag",
    "manifold_margin",
    "bridge_success",
    "transport_error_logeuc",
    "x_aug_nan_flag",
    "x_aug_inf_flag",
    "selection_stage",
    "selector_name",
    "feasible_flag",
    "selector_accept_flag",
    "selector_candidate_pool_size",
    "selector_feasible_count",
    "feasible_rate",
    "selector_accept_rate",
    "pre_filter_reject_count",
    "post_bridge_reject_count",
    "reject_reason_zero_gamma",
    "reject_reason_safe_radius",
    "reject_reason_bridge_fail",
    "reject_reason_transport_error",
    "relevance_score",
    "safe_balance_score",
    "fidelity_score",
    "variety_score",
    "fv_score",
    "template_diversity_bonus",
    "post_bridge_reject_reason",
    "candidate_status",
    "reject_reason",
]


def make_candidate_uid(
    *,
    dataset: str,
    seed: int,
    method: str,
    tid: object,
    candidate_order: object,
    slot_index: object,
) -> str:
    order = "na" if pd.isna(candidate_order) else str(candidate_order)
    slot = "na" if pd.isna(slot_index) else str(slot_index)
    return f"{dataset}/s{int(seed)}/{method}/{tid}/c{order}/slot{slot}"


def normalize_candidate_audit_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    dataset: str,
    seed: int,
    method: str,
    activation_policy: str,
    eta_safe: Optional[float],
) -> pd.DataFrame:
    operator_meta = pia_operator_metadata(activation_policy)
    out: List[Dict[str, object]] = []
    for fallback_slot, row_in in enumerate(rows):
        row = dict(row_in)
        tid = row.get("tid", "")
        candidate_order = row.get("candidate_order", fallback_slot)
        slot_index = row.get("slot_index", fallback_slot)
        template_id = row.get("zpia_template_id", row.get("template_id", row.get("direction_id", np.nan)))
        template_rank = row.get("zpia_template_rank", row.get("template_rank", row.get("direction_id", np.nan)))
        template_sign = row.get("zpia_template_sign", row.get("template_sign", row.get("sign", np.nan)))
        response_abs = row.get("zpia_template_response_abs", row.get("template_response_abs", np.nan))
        gamma_used = _to_float(row.get("gamma_used", np.nan))
        direction_norm = _to_float(row.get("direction_norm", np.nan))
        transport_error = _to_float(row.get("transport_error_logeuc", np.nan))
        bridge_success = bool(row.get("bridge_success", np.isfinite(transport_error)))
        x_aug_nan_flag = False
        x_aug_inf_flag = False
        reject_reason = str(row.get("reject_reason", "") or "")
        if not bridge_success:
            reject_reason = "rejected_bridge_fail"
        if bool(x_aug_nan_flag):
            reject_reason = "rejected_nan"
        if bool(x_aug_inf_flag):
            reject_reason = "rejected_inf"
        candidate_status = str(row.get("candidate_status", "") or ("accepted" if reject_reason == "" else reject_reason))
        relevance_score = _to_float(row.get("relevance_score", np.nan))
        safe_balance_score = _to_float(row.get("safe_balance_score", np.nan))
        fidelity_score = _to_float(row.get("fidelity_score", np.nan))
        if not np.isfinite(fidelity_score) and np.isfinite(relevance_score) and np.isfinite(safe_balance_score):
            fidelity_score = float(relevance_score + safe_balance_score)

        out.append(
            {
                "dataset": str(dataset),
                "seed": int(seed),
                "method": str(method),
                **operator_meta,
                "candidate_uid": make_candidate_uid(
                    dataset=str(dataset),
                    seed=int(seed),
                    method=str(method),
                    tid=tid,
                    candidate_order=candidate_order,
                    slot_index=slot_index,
                ),
                "anchor_index": _to_int(row.get("anchor_index", -1)),
                "tid": tid,
                "class_id": _to_int(row.get("class_id", row.get("label", -1))),
                "candidate_order": _to_int(candidate_order),
                "slot_index": _to_int(slot_index),
                "template_id": _to_int(template_id),
                "template_rank": _to_int(template_rank),
                "template_sign": _to_float(template_sign),
                "template_response_abs": _to_float(response_abs),
                "gamma_requested": _to_float(row.get("gamma_requested", np.nan)),
                "gamma_used": gamma_used,
                "eta_safe": np.nan if eta_safe is None else float(eta_safe),
                "direction_norm": direction_norm,
                "z_displacement_norm": _safe_product_abs(gamma_used, direction_norm),
                "safe_upper_bound": _to_float(row.get("safe_upper_bound", np.nan)),
                "safe_radius_ratio": _to_float(row.get("safe_radius_ratio", np.nan)),
                "safe_clip_flag": _to_float(row.get("is_clipped", row.get("safe_clip_flag", 0.0))),
                "gamma_zero_flag": float(abs(gamma_used) <= 1e-12) if np.isfinite(gamma_used) else np.nan,
                "manifold_margin": _to_float(row.get("manifold_margin", np.nan)),
                "bridge_success": bridge_success,
                "transport_error_logeuc": transport_error,
                "x_aug_nan_flag": x_aug_nan_flag,
                "x_aug_inf_flag": x_aug_inf_flag,
                "selection_stage": str(row.get("selection_stage", "")),
                "selector_name": str(row.get("selector_name", "")),
                "feasible_flag": _to_float(row.get("feasible_flag", np.nan)),
                "selector_accept_flag": _to_float(row.get("selector_accept_flag", np.nan)),
                "selector_candidate_pool_size": _to_float(row.get("selector_candidate_pool_size", np.nan)),
                "selector_feasible_count": _to_float(row.get("selector_feasible_count", np.nan)),
                "feasible_rate": _to_float(row.get("feasible_rate", np.nan)),
                "selector_accept_rate": _to_float(row.get("selector_accept_rate", np.nan)),
                "pre_filter_reject_count": _to_float(row.get("pre_filter_reject_count", np.nan)),
                "post_bridge_reject_count": _to_float(row.get("post_bridge_reject_count", np.nan)),
                "reject_reason_zero_gamma": _to_float(row.get("reject_reason_zero_gamma", np.nan)),
                "reject_reason_safe_radius": _to_float(row.get("reject_reason_safe_radius", np.nan)),
                "reject_reason_bridge_fail": _to_float(row.get("reject_reason_bridge_fail", np.nan)),
                "reject_reason_transport_error": _to_float(row.get("reject_reason_transport_error", np.nan)),
                "relevance_score": relevance_score,
                "safe_balance_score": safe_balance_score,
                "fidelity_score": fidelity_score,
                "variety_score": _to_float(row.get("variety_score", np.nan)),
                "fv_score": _to_float(row.get("fv_score", np.nan)),
                "template_diversity_bonus": _to_float(row.get("template_diversity_bonus", np.nan)),
                "post_bridge_reject_reason": str(row.get("post_bridge_reject_reason", "")),
                "candidate_status": candidate_status,
                "reject_reason": reject_reason,
            }
        )
    df = pd.DataFrame(out)
    for col in P0_CANDIDATE_AUDIT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[P0_CANDIDATE_AUDIT_COLUMNS]


def summarize_candidate_audit(df: pd.DataFrame) -> Dict[str, object]:
    if df is None or df.empty:
        return {
            "candidate_audit_rows": 0,
            "candidate_audit_available": False,
            "aug_valid_rate": 0.0,
        }
    out: Dict[str, object] = {
        "candidate_audit_rows": int(len(df)),
        "candidate_audit_available": True,
        "aug_valid_rate": float(df["bridge_success"].astype(float).mean()) if "bridge_success" in df else 0.0,
        "candidate_accept_rate": float((df["candidate_status"] == "accepted").mean())
        if "candidate_status" in df
        else 0.0,
    }
    numeric_aggs = {
        "template_response_abs": "template_response_abs_mean",
        "gamma_requested": "gamma_requested_mean_audit",
        "gamma_used": "gamma_used_mean_audit",
        "z_displacement_norm": "z_displacement_norm_mean",
        "safe_radius_ratio": "safe_radius_ratio_mean_audit",
        "safe_clip_flag": "safe_clip_rate_audit",
        "gamma_zero_flag": "gamma_zero_rate_audit",
        "manifold_margin": "manifold_margin_mean_audit",
        "transport_error_logeuc": "transport_error_logeuc_mean_audit",
        "feasible_flag": "feasible_rate_audit",
        "selector_accept_flag": "selector_accept_rate_audit",
        "relevance_score": "relevance_score_mean_audit",
        "safe_balance_score": "safe_balance_score_mean_audit",
        "fidelity_score": "fidelity_score_mean_audit",
        "variety_score": "variety_score_mean_audit",
        "fv_score": "fv_score_mean_audit",
    }
    for col, key in numeric_aggs.items():
        if col in df:
            vals = pd.to_numeric(df[col], errors="coerce")
            out[key] = float(vals.mean()) if vals.notna().any() else np.nan
    if "template_id" in df:
        valid_ids = pd.to_numeric(df["template_id"], errors="coerce").dropna()
        valid_ids = valid_ids[valid_ids >= 0]
        if not valid_ids.empty:
            _, counts = np.unique(valid_ids.astype(int).to_numpy(), return_counts=True)
            probs = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
            out["template_usage_entropy_audit"] = float(-np.sum(probs * np.log(probs + 1e-12)))
            out["top_template_concentration_audit"] = float(probs.max())
    physics = validate_candidate_audit_physics(df)
    out.update(physics)
    for col in [
        "pre_filter_reject_count",
        "post_bridge_reject_count",
        "reject_reason_zero_gamma",
        "reject_reason_safe_radius",
        "reject_reason_bridge_fail",
        "reject_reason_transport_error",
    ]:
        if col in df:
            vals = pd.to_numeric(df[col], errors="coerce")
            out[col] = int(vals.fillna(0).sum())
    return out


def validate_candidate_audit_physics(df: pd.DataFrame) -> Dict[str, object]:
    if df is None or df.empty:
        return {"candidate_physics_ok": False, "gamma_used_gt_requested_count": 0}
    gamma_req = pd.to_numeric(df.get("gamma_requested", pd.Series(dtype=float)), errors="coerce")
    gamma_used = pd.to_numeric(df.get("gamma_used", pd.Series(dtype=float)), errors="coerce")
    safe_ratio = pd.to_numeric(df.get("safe_radius_ratio", pd.Series(dtype=float)), errors="coerce")
    finite_cols = ["gamma_requested", "gamma_used", "safe_radius_ratio", "transport_error_logeuc"]
    has_bad_finite = False
    for col in finite_cols:
        if col not in df:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        finite_vals = vals.dropna().to_numpy(dtype=float)
        has_bad_finite = bool(has_bad_finite or np.isinf(finite_vals).any())
    gt_count = int((gamma_used > gamma_req + 1e-9).sum()) if len(gamma_req) and len(gamma_used) else 0
    ratio_bad = int(((safe_ratio < -1e-9) | (safe_ratio > 1.0 + 1e-6)).sum()) if len(safe_ratio) else 0
    return {
        "candidate_physics_ok": bool(gt_count == 0 and ratio_bad == 0 and not has_bad_finite),
        "gamma_used_gt_requested_count": gt_count,
        "safe_radius_ratio_out_of_bounds_count": ratio_bad,
    }


def write_candidate_audit(
    rows: Iterable[Mapping[str, object]],
    *,
    out_dir: Path,
    dataset: str,
    seed: int,
    method: str,
    activation_policy: str,
    eta_safe: Optional[float],
) -> Dict[str, object]:
    df = normalize_candidate_audit_rows(
        rows,
        dataset=dataset,
        seed=seed,
        method=method,
        activation_policy=activation_policy,
        eta_safe=eta_safe,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset}_s{int(seed)}_{method}_candidate_audit.csv.gz"
    df.to_csv(path, index=False, compression="gzip")
    summary = summarize_candidate_audit(df)
    summary["candidate_audit_path"] = str(path)
    return summary


def _to_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out


def _to_int(value: object) -> int:
    try:
        if pd.isna(value):
            return -1
        return int(value)
    except Exception:
        return -1


def _safe_product_abs(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    return float(abs(a) * abs(b))
