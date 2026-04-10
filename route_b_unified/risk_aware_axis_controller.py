from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from scripts.fisher_pia_utils import FisherPIAConfig, compute_fisher_pia_terms


@dataclass(frozen=True)
class FragilityProbeConfig:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    eps: float = 1e-8
    probe_level: str = "class"
    probe_type: str = "fisher_risk_expand_boundary_purity_v1"


@dataclass(frozen=True)
class DiscreteAxisScaleControllerConfig:
    available_scales: Tuple[float, float, float] = (0.80, 0.75, 0.70)
    controller_mode: str = "risk_aware_discrete_axis2_v1"
    low_threshold: float = 1.0 / 3.0
    high_threshold: float = 2.0 / 3.0


def _summary_stats(arr: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(list(arr), dtype=np.float64).ravel()
    if x.size == 0:
        return {"min": 0.0, "mean": 0.0, "std": 0.0, "max": 0.0}
    return {
        "min": float(np.min(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "max": float(np.max(x)),
    }


def _normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    keys = sorted(scores.keys())
    vals = np.asarray([float(scores[k]) for k in keys], dtype=np.float64)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) <= 1e-12:
        return {int(k): 0.5 for k in keys}
    norm = (vals - vmin) / (vmax - vmin)
    return {int(keys[i]): float(norm[i]) for i in range(len(keys))}


def compute_class_fragility_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    fisher_cfg: FisherPIAConfig,
    probe_cfg: FragilityProbeConfig,
) -> Tuple[Dict[int, float], List[Dict[str, float]], Dict[str, object]]:
    class_terms, terms_meta = compute_fisher_pia_terms(X_train, y_train, cfg=fisher_cfg)

    raw_scores: Dict[int, float] = {}
    rows: List[Dict[str, float]] = []
    for cls in sorted(class_terms.keys()):
        term = class_terms[int(cls)]
        S_risk = np.asarray(term["S_risk"], dtype=np.float64)
        S_expand = np.asarray(term["S_expand"], dtype=np.float64)
        tr_risk = float(np.trace(S_risk))
        tr_expand = float(np.trace(S_expand))
        class_count = int(term["class_count"])
        boundary_count = int(term["boundary_count"])
        boundary_fraction = float(boundary_count / max(1, class_count))
        boundary_purity_mean = float(term["boundary_purity_summary"].get("mean", 0.0))
        risk_expand_ratio = tr_risk / (tr_expand + float(probe_cfg.eps))
        fragility = (
            float(probe_cfg.alpha) * risk_expand_ratio
            + float(probe_cfg.beta) * boundary_fraction
            + float(probe_cfg.gamma) * (1.0 - boundary_purity_mean)
        )
        raw_scores[int(cls)] = float(fragility)
        rows.append(
            {
                "class_id": int(cls),
                "fragility_score_raw": float(fragility),
                "risk_expand_ratio": float(risk_expand_ratio),
                "boundary_fraction": float(boundary_fraction),
                "boundary_purity_mean": float(boundary_purity_mean),
                "trace_S_risk": float(tr_risk),
                "trace_S_expand": float(tr_expand),
                "class_count": float(class_count),
                "boundary_count": float(boundary_count),
            }
        )

    norm_scores = _normalize_scores(raw_scores)
    for row in rows:
        cls = int(row["class_id"])
        row["fragility_score_norm"] = float(norm_scores[cls])

    meta = {
        "probe_type": str(probe_cfg.probe_type),
        "probe_level": str(probe_cfg.probe_level),
        "terms_meta": terms_meta,
        "raw_score_summary": _summary_stats(raw_scores.values()),
        "norm_score_summary": _summary_stats(norm_scores.values()),
    }
    return raw_scores, rows, meta


def select_discrete_scales_from_fragility(
    class_fragility_scores: Dict[int, float],
    *,
    controller_cfg: DiscreteAxisScaleControllerConfig,
) -> Tuple[Dict[int, float], List[Dict[str, object]], Dict[str, object]]:
    if len(controller_cfg.available_scales) != 3:
        raise ValueError("available_scales must contain exactly 3 values: low/mid/high risk scales")

    norm_scores = _normalize_scores(class_fragility_scores)
    low_scale, mid_scale, high_scale = [float(v) for v in controller_cfg.available_scales]

    scale_map: Dict[int, float] = {}
    rows: List[Dict[str, object]] = []
    for cls in sorted(norm_scores.keys()):
        score = float(norm_scores[int(cls)])
        if score <= float(controller_cfg.low_threshold):
            level = "safe"
            scale = low_scale
        elif score <= float(controller_cfg.high_threshold):
            level = "mild"
            scale = mid_scale
        else:
            level = "fragile"
            scale = high_scale
        scale_map[int(cls)] = float(scale)
        rows.append(
            {
                "class_id": int(cls),
                "fragility_score_norm": float(score),
                "fragility_level": str(level),
                "selected_second_axis_scale": float(scale),
            }
        )

    meta = {
        "controller_mode": str(controller_cfg.controller_mode),
        "available_scales": [float(v) for v in controller_cfg.available_scales],
        "thresholds": {
            "low_threshold": float(controller_cfg.low_threshold),
            "high_threshold": float(controller_cfg.high_threshold),
        },
        "selected_scale_summary": _summary_stats(scale_map.values()),
        "level_counts": {
            level: int(sum(1 for row in rows if row["fragility_level"] == level))
            for level in ["safe", "mild", "fragile"]
        },
    }
    return scale_map, rows, meta
