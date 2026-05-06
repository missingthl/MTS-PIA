from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from host_alignment_probe import compute_gradient_alignment


def run_analysis_probe(
    *,
    args,
    model_obj,
    tid_aug,
    X_aug,
    tid_to_rec: Dict[object, object],
) -> Dict[str, float]:
    if not getattr(args, "theory_diagnostics", False):
        return {"host_geom_cosine_mean": 0.0, "host_conflict_rate": 0.0}
    if model_obj is None or X_aug is None or len(tid_aug) == 0:
        return {"host_geom_cosine_mean": 0.0, "host_conflict_rate": 0.0}

    sample_n = min(int(getattr(args, "theory_sample_n", 64)), len(tid_aug))
    rows = []
    for i in range(sample_n):
        tid = tid_aug[i]
        rec = tid_to_rec.get(tid)
        if rec is None:
            continue
        align = compute_gradient_alignment(
            model_obj,
            rec.x_raw,
            X_aug[i],
            int(rec.y),
            device=str(getattr(args, "device", "cpu")),
        )
        if align is not None:
            rows.append(align)
    if not rows:
        return {"host_geom_cosine_mean": 0.0, "host_conflict_rate": 0.0}
    df = pd.DataFrame(rows)
    return {
        "host_geom_cosine_mean": float(df["cosine"].mean()),
        "host_conflict_rate": float((df["cosine"] < 0).mean()),
    }


def template_usage_stats(template_ids: List[int]) -> Dict[str, float]:
    if not template_ids:
        return {"template_usage_entropy": 0.0, "top_template_concentration": 0.0}
    _, counts = np.unique(np.asarray(template_ids, dtype=np.int64), return_counts=True)
    probs = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return {
        "template_usage_entropy": entropy,
        "top_template_concentration": float(probs.max()) if probs.size else 0.0,
    }


def normalize_unit_interval(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out
    lo = float(np.min(arr[finite]))
    hi = float(np.max(arr[finite]))
    if hi <= lo + 1e-12:
        out[finite] = 0.5
        return out
    out[finite] = (arr[finite] - lo) / (hi - lo)
    out[~finite] = 0.0
    return out


def summarize_multitemplate_audit_rows(audit_rows: List[Dict[str, object]]) -> Dict[str, float]:
    out = {
        "template_usage_entropy": 0.0,
        "top_template_concentration": 0.0,
        "template_response_abs_mean": 0.0,
    }
    if not audit_rows:
        return out
    df = pd.DataFrame(audit_rows)
    if "zpia_template_id" in df.columns:
        template_ids = pd.to_numeric(df["zpia_template_id"], errors="coerce").dropna().astype(int).tolist()
        out.update(template_usage_stats(template_ids))
    if "zpia_template_response_abs" in df.columns:
        vals = pd.to_numeric(df["zpia_template_response_abs"], errors="coerce")
        out["template_response_abs_mean"] = float(vals.mean()) if vals.notna().any() else 0.0
    out.update(
        {
            "template_usage_entropy": float(out.get("template_usage_entropy", 0.0)),
            "top_template_concentration": float(out.get("top_template_concentration", 0.0)),
        }
    )
    return out


def template_response_profile(z: np.ndarray, bank: np.ndarray) -> Dict[str, float]:
    D = np.asarray(bank, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] <= 0:
        return {
            "template_response_top1": np.nan,
            "template_response_top5_mean": np.nan,
            "template_response_gap_top1_top5": np.nan,
            "template_response_entropy": np.nan,
        }
    responses = np.abs(np.asarray(z, dtype=np.float64).ravel() @ D.T)
    order = np.lexsort((np.arange(D.shape[0]), -responses))
    top_vals = responses[order]
    top1 = float(top_vals[0]) if top_vals.size else np.nan
    top5_vals = top_vals[: min(5, top_vals.size)]
    probs = top_vals / (float(np.sum(top_vals)) + 1e-12)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum()) if probs.size else np.nan
    return {
        "template_response_top1": top1,
        "template_response_top5_mean": float(np.mean(top5_vals)) if top5_vals.size else np.nan,
        "template_response_gap_top1_top5": float(top1 - np.mean(top5_vals)) if top5_vals.size else np.nan,
        "template_response_entropy": entropy,
    }

