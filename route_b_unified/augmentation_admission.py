from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

import numpy as np


def _clip(v: float, lo: float, hi: float) -> float:
    return float(min(max(float(v), float(lo)), float(hi)))


def _minmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def _safe_metric(meta: Mapping[str, object], key: str) -> float:
    value = meta.get(key, 0.0)
    try:
        out = float(value)
    except Exception:
        out = 0.0
    if not np.isfinite(out):
        return 0.0
    return out


@dataclass(frozen=True)
class HybridAdmissionConfig:
    risk_score_version: str = "hybrid_v1"
    bridge_cond_weight: float = 0.5
    bridge_covdist_weight: float = 0.5
    risk_weight_bridge_sample: float = 1.0
    risk_weight_flip_group: float = 1.0
    risk_weight_distortion_group: float = 1.0
    lambda_flip: float = 0.20
    lambda_distortion: float = 0.20
    mild_base_keep_ratio: float = 0.85
    strict_base_keep_ratio: float = 0.60
    min_keep_ratio: float = 0.20
    max_keep_ratio: float = 0.95


@dataclass(frozen=True)
class HybridAdmissionResult:
    gate_mode: str
    ratio_mode: str
    ratio_value: float
    base_keep_ratio: float
    effective_keep_ratio: float
    threshold_setting: str
    accepted_indices: List[int] = field(default_factory=list)
    rejected_indices: List[int] = field(default_factory=list)
    accepted_aug_trials: List[Dict[str, object]] = field(default_factory=list)
    rejected_aug_trials: List[Dict[str, object]] = field(default_factory=list)
    accepted_bridge_meta: List[Dict[str, object]] = field(default_factory=list)
    rejected_bridge_meta: List[Dict[str, object]] = field(default_factory=list)
    risk_rows: List[Dict[str, object]] = field(default_factory=list)
    group_risk_summary: Dict[str, float] = field(default_factory=dict)
    summary: Dict[str, object] = field(default_factory=dict)


def _group_flip_norm(flip_rate: float) -> float:
    return _clip(float(flip_rate), 0.0, 1.0)


def _group_distortion_norm(distortion_mean: float) -> float:
    dist = max(0.0, float(distortion_mean))
    return float(dist / (1.0 + dist))


def _base_keep_ratio(cfg: HybridAdmissionConfig, gate_mode: str) -> float:
    gate = str(gate_mode).strip().lower()
    if gate == "mild":
        return float(cfg.mild_base_keep_ratio)
    if gate == "strict":
        return float(cfg.strict_base_keep_ratio)
    raise ValueError(f"unknown gate_mode: {gate_mode}")


def _sample_bridge_risk(
    bridge_meta: Sequence[Mapping[str, object]],
    cfg: HybridAdmissionConfig,
) -> tuple[np.ndarray, Dict[str, object]]:
    cond_raw = np.asarray(
        [np.log1p(max(0.0, _safe_metric(m, "bridge_cond_A"))) for m in bridge_meta],
        dtype=np.float64,
    )
    covdist_raw = np.asarray(
        [np.log1p(max(0.0, _safe_metric(m, "bridge_cov_to_orig_distance_logeuc"))) for m in bridge_meta],
        dtype=np.float64,
    )
    cond_norm = _minmax(cond_raw)
    covdist_norm = _minmax(covdist_raw)
    bridge_risk = (
        float(cfg.bridge_cond_weight) * cond_norm
        + float(cfg.bridge_covdist_weight) * covdist_norm
    )
    bridge_weight_sum = float(cfg.bridge_cond_weight + cfg.bridge_covdist_weight)
    if bridge_weight_sum > 1e-12:
        bridge_risk = bridge_risk / bridge_weight_sum
    return np.asarray(bridge_risk, dtype=np.float64), {
        "cond_log1p": cond_raw.tolist(),
        "cond_norm": cond_norm.tolist(),
        "covdist_log1p": covdist_raw.tolist(),
        "covdist_norm": covdist_norm.tolist(),
        "bridge_risk_sample": bridge_risk.tolist(),
    }


def apply_hybrid_admission(
    *,
    aug_trials: Sequence[Dict[str, object]],
    per_aug_bridge_meta: Sequence[Mapping[str, object]],
    flip_rate_group: float,
    distortion_risk_group: float,
    gate_mode: str,
    cfg: HybridAdmissionConfig,
    ratio_mode: str = "orig_plus_100pct_filtered_aug",
    ratio_value: float = 1.0,
) -> HybridAdmissionResult:
    if len(aug_trials) != len(per_aug_bridge_meta):
        raise ValueError("aug_trials and per_aug_bridge_meta must have the same length")

    n_aug = int(len(aug_trials))
    if n_aug == 0:
        base_keep = _base_keep_ratio(cfg, gate_mode)
        return HybridAdmissionResult(
            gate_mode=str(gate_mode),
            ratio_mode=str(ratio_mode),
            ratio_value=float(ratio_value),
            base_keep_ratio=base_keep,
            effective_keep_ratio=0.0,
            threshold_setting="no_aug_trials",
            group_risk_summary={
                "flip_risk_group": _group_flip_norm(flip_rate_group),
                "distortion_risk_group": _group_distortion_norm(distortion_risk_group),
            },
            summary={"accepted_aug_count": 0, "rejected_aug_count": 0, "accept_ratio": 0.0},
        )

    bridge_sample_risk, bridge_risk_meta = _sample_bridge_risk(per_aug_bridge_meta, cfg)
    flip_norm = _group_flip_norm(flip_rate_group)
    distortion_norm = _group_distortion_norm(distortion_risk_group)
    group_bias = (
        float(cfg.risk_weight_flip_group) * flip_norm
        + float(cfg.risk_weight_distortion_group) * distortion_norm
    )
    hybrid_risk = float(cfg.risk_weight_bridge_sample) * bridge_sample_risk + group_bias

    base_keep = _base_keep_ratio(cfg, gate_mode)
    effective_keep = base_keep - float(cfg.lambda_flip) * flip_norm - float(cfg.lambda_distortion) * distortion_norm
    effective_keep = _clip(effective_keep, float(cfg.min_keep_ratio), float(cfg.max_keep_ratio))

    order = np.argsort(hybrid_risk, kind="mergesort")
    keep_count = int(np.floor(effective_keep * float(n_aug)))
    keep_count = max(1, keep_count)
    keep_count = min(keep_count, n_aug)

    if str(ratio_mode) == "orig_plus_50pct_filtered_aug":
        keep_count = max(1, int(np.floor(float(ratio_value) * float(keep_count))))
    elif str(ratio_mode) != "orig_plus_100pct_filtered_aug":
        raise ValueError(f"unknown ratio_mode: {ratio_mode}")

    accepted = np.sort(order[:keep_count]).astype(np.int64)
    rejected = np.sort(order[keep_count:]).astype(np.int64)

    accepted_set = set(int(v) for v in accepted.tolist())
    risk_rows: List[Dict[str, object]] = []
    for idx in range(n_aug):
        meta = dict(per_aug_bridge_meta[idx])
        risk_rows.append(
            {
                "aug_index": int(idx),
                "aug_tid": str(meta.get("aug_tid", f"aug_{idx:06d}")),
                "source_tid": str(meta.get("source_tid", "")),
                "label": int(meta.get("label", -1)),
                "bridge_risk_sample": float(bridge_sample_risk[idx]),
                "flip_risk_group": float(flip_norm),
                "distortion_risk_group": float(distortion_norm),
                "hybrid_risk": float(hybrid_risk[idx]),
                "accepted": bool(idx in accepted_set),
                "bridge_cond_A": _safe_metric(meta, "bridge_cond_A"),
                "bridge_cov_to_orig_distance_logeuc": _safe_metric(meta, "bridge_cov_to_orig_distance_logeuc"),
            }
        )

    accepted_list = [int(v) for v in accepted.tolist()]
    rejected_list = [int(v) for v in rejected.tolist()]
    accepted_trials = [dict(aug_trials[i]) for i in accepted_list]
    rejected_trials = [dict(aug_trials[i]) for i in rejected_list]
    accepted_meta = [dict(per_aug_bridge_meta[i]) for i in accepted_list]
    rejected_meta = [dict(per_aug_bridge_meta[i]) for i in rejected_list]

    threshold = float(hybrid_risk[order[keep_count - 1]]) if keep_count > 0 else 0.0
    return HybridAdmissionResult(
        gate_mode=str(gate_mode),
        ratio_mode=str(ratio_mode),
        ratio_value=float(ratio_value),
        base_keep_ratio=float(base_keep),
        effective_keep_ratio=float(keep_count / max(1, n_aug)),
        threshold_setting=f"keep_lowest_hybrid_risk_until_threshold={threshold:.6f}",
        accepted_indices=accepted_list,
        rejected_indices=rejected_list,
        accepted_aug_trials=accepted_trials,
        rejected_aug_trials=rejected_trials,
        accepted_bridge_meta=accepted_meta,
        rejected_bridge_meta=rejected_meta,
        risk_rows=risk_rows,
        group_risk_summary={
            "flip_risk_group": float(flip_norm),
            "distortion_risk_group": float(distortion_norm),
            "group_bias_term": float(group_bias),
            "effective_keep_ratio_formula_flip_coeff": float(cfg.lambda_flip),
            "effective_keep_ratio_formula_distortion_coeff": float(cfg.lambda_distortion),
        },
        summary={
            "accepted_aug_count": int(len(accepted_list)),
            "rejected_aug_count": int(len(rejected_list)),
            "accept_ratio": float(len(accepted_list) / max(1, n_aug)),
            "bridge_cond_risk_mean": float(np.mean([r["bridge_cond_A"] for r in risk_rows])) if risk_rows else 0.0,
            "bridge_covdist_risk_mean": float(
                np.mean([r["bridge_cov_to_orig_distance_logeuc"] for r in risk_rows])
            )
            if risk_rows
            else 0.0,
        },
    )
