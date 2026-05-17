from __future__ import annotations

from typing import Dict, List, Optional

import time
import numpy as np
import pandas as pd
import torch

from core.bridge import bridge_single, logvec_to_spd
from .state import TrialRecord


def materialize_z_aug_out(
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
    """Realize z-space candidates in raw time-series space via the bridge."""

    tid_to_rec = {record.tid: record for record in train_recs}
    aug_trials: List[Dict[str, object]] = []
    bridge_metrics: List[Dict[str, object]] = []
    out_rows: List[Dict[str, object]] = []
    
    t0 = time.perf_counter()
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
    t1 = time.perf_counter()
    bridge_realization_sec = t1 - t0

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
    bridge_success_rate = float(np.mean([1.0 if row.get("bridge_success", False) else 0.0 for row in out_rows])) if out_rows else 0.0

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
        "bridge_realization_sec": bridge_realization_sec,
        "safe_radius_ratio_mean": float(np.mean(safe_ratios)) if safe_ratios else 0.0,
        "safe_clip_rate": float(np.mean(clip_flags)) if clip_flags else 0.0,
        "manifold_margin_mean": float(np.mean(margins)) if margins else 0.0,
        "gamma_requested_mean": float(np.mean(gamma_req)) if gamma_req else 0.0,
        "gamma_used_mean": float(np.mean(gamma_used)) if gamma_used else 0.0,
        "z_displacement_norm_mean": float(np.mean(z_delta_norms)) if z_delta_norms else 0.0,
        "gamma_zero_rate": float(np.mean([1.0 if g < 1e-12 else 0.0 for g in gamma_used])) if gamma_used else 0.0,
        "aug_valid_rate": float(
            1.0 - (float(np.mean([1.0 if g < 1e-12 else 0.0 for g in gamma_used])) if gamma_used else 0.0)
        ),
        "post_bridge_reject_count": int(sum(post_bridge_flags)),
        "reject_reason_bridge_fail": int(bridge_fail_count),
        "reject_reason_transport_error": 0,
        "bridge_success_rate": bridge_success_rate,
        "eta_safe": eta_safe,
        "candidate_total_count": int(len(z_aug)),
        "aug_total_count": int(len(z_aug)),
        "aug_dataset": None,
    }
    if extra_meta:
        out.update(extra_meta)
    return out
