from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from core.bridge import bridge_single, logvec_to_spd
from core.curriculum import active_direction_probs, build_curriculum_aug_candidates
from utils.evaluators import ManifoldAugDataset

from .direction_banks import build_direction_bank_for_args as _build_direction_bank_for_args
from .state import TrialRecord


def _score_aug_margins(
    *,
    model_obj,
    X_aug: Optional[np.ndarray],
    y_aug: Optional[np.ndarray],
    device: str,
    batch_size: int,
) -> np.ndarray:
    if model_obj is None or X_aug is None or y_aug is None or len(y_aug) == 0:
        return np.empty((0,), dtype=np.float64)
    use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    dev = torch.device(device if use_cuda else "cpu")
    model_obj.to(dev)
    model_obj.eval()
    margins: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(y_aug), int(batch_size)):
            bx = torch.from_numpy(X_aug[start : start + int(batch_size)]).float().to(dev)
            by = torch.from_numpy(y_aug[start : start + int(batch_size)]).long().to(dev)
            logits = model_obj(bx)
            true_logits = logits.gather(1, by.view(-1, 1)).squeeze(1)
            if logits.shape[1] <= 1:
                margin = true_logits
            else:
                masked = logits.clone()
                masked.scatter_(1, by.view(-1, 1), -torch.inf)
                other_logits = torch.max(masked, dim=1).values
                margin = true_logits - other_logits
            margins.append(margin.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(margins) if margins else np.empty((0,), dtype=np.float64)


def _attach_feedback_scores_to_aug_out(
    *,
    aug_out: Dict[str, object],
    model_obj,
    device: str,
    batch_size: int,
    feedback_margin_temperature: float,
    engine_id: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    margins = _score_aug_margins(
        model_obj=model_obj,
        X_aug=aug_out.get("X_aug_raw"),
        y_aug=aug_out.get("y_aug_np"),
        device=device,
        batch_size=batch_size,
    )
    scaled_margins = np.clip(margins / max(float(feedback_margin_temperature), 1e-6), -60.0, 60.0)
    weights = 1.0 / (1.0 + np.exp(-scaled_margins))
    for idx, row in enumerate(aug_out.get("audit_rows", [])):
        if engine_id is not None:
            row["engine_id"] = str(engine_id)
        row["margin_aug"] = float(margins[idx]) if idx < len(margins) else 0.0
        row["feedback_weight"] = float(weights[idx]) if idx < len(weights) else 0.0
    return {"margins": margins, "weights": weights}


def _build_act_realized_augmentations(
    *,
    args,
    seed: int,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    algo_override: Optional[str] = None,
    engine_id: Optional[str] = None,
) -> Dict[str, object]:
    algo_name = str(algo_override or args.algo)
    bank_out = _build_direction_bank_for_args(
        args=args,
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        algo_override=algo_name,
    )
    direction_bank = bank_out["bank"]
    direction_meta = bank_out["meta"]

    effective_k = int(direction_bank.shape[0])
    print(
        f"Requested K: {args.k_dir} | Effective K: {effective_k} | "
        f"Source: {direction_meta.get('bank_source', algo_name)} | Classes: {len(np.unique(y_train))}"
    )

    gamma_budget = np.full((effective_k,), float(args.pia_gamma), dtype=np.float64)
    direction_probs = active_direction_probs(gamma_budget, freeze_eps=0.01)
    eta_safe = None if args.disable_safe_step else args.eta_safe
    tid_train = np.asarray([record.tid for record in train_recs], dtype=object)

    z_aug, y_aug, tid_aug, _, _, aug_meta = build_curriculum_aug_candidates(
        X_train_z,
        y_train,
        tid_train,
        direction_bank=direction_bank,
        direction_probs=direction_probs,
        gamma_by_dir=gamma_budget,
        multiplier=args.multiplier,
        seed=seed + 42,
        eta_safe=eta_safe,
    )

    tid_to_rec = {record.tid: record for record in train_recs}
    aug_trials: List[Dict[str, object]] = []
    bridge_metrics: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    candidate_rows = list(aug_meta.get("candidate_rows", []))
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
        audit = candidate_rows[i].copy() if i < len(candidate_rows) else {
            "anchor_index": -1,
            "tid": tid_aug[i],
            "class_id": int(y_aug[i]),
            "candidate_order": int(i),
            "direction_id": -1,
            "sign": 0.0,
            "gamma_used": 0.0,
            "safe_radius_ratio": 0.0,
        }
        audit.update(
            {
                "algo": algo_name,
                "engine_id": str(engine_id or algo_name),
                "direction_bank_source": direction_meta.get("bank_source", algo_name),
                "transport_error_fro": float(bridge_meta.get("transport_error_fro", 0.0)),
                "transport_error_logeuc": float(bridge_meta.get("transport_error_logeuc", 0.0)),
                "bridge_cond_A": float(bridge_meta.get("bridge_cond_A", 0.0)),
                "metric_preservation_error": float(bridge_meta.get("metric_preservation_error", 0.0)),
            }
        )
        audit_rows.append(audit)

    X_aug_raw = np.stack([trial["x"] for trial in aug_trials]) if aug_trials else None
    y_aug_np = np.asarray([trial["y"] for trial in aug_trials], dtype=np.int64) if aug_trials else None
    avg_bridge = pd.DataFrame(bridge_metrics).mean().to_dict() if bridge_metrics else {}

    gamma_used_vals = [float(row.get("gamma_used", 0.0)) for row in audit_rows]
    gamma_req_vals = [float(row.get("gamma_requested", 0.0)) for row in audit_rows]
    clip_flags = [float(row.get("is_clipped", 0.0)) for row in audit_rows]
    safe_ratios = [float(row.get("safe_radius_ratio", 0.0)) for row in audit_rows]
    margins = [float(row.get("manifold_margin", 0.0)) for row in audit_rows]

    aug_dataset_out: Optional[ManifoldAugDataset] = None
    if getattr(args, "onthefly_aug", False) and len(z_aug) > 0:
        anchor_x_raws = [tid_to_rec[tid_aug[i]].x_raw for i in range(len(z_aug))]
        anchor_sigma_origs = [tid_to_rec[tid_aug[i]].sigma_orig for i in range(len(z_aug))]
        aug_dataset_out = ManifoldAugDataset(
            anchor_x_raws=anchor_x_raws,
            anchor_sigma_origs=anchor_sigma_origs,
            z_cands=z_aug,
            y_cands=y_aug,
            mean_log=mean_log,
        )

    return {
        "effective_k": effective_k,
        "direction_bank": direction_bank,
        "z_aug": z_aug,
        "y_aug": y_aug,
        "tid_aug": tid_aug,
        "aug_trials": aug_trials,
        "X_aug_raw": X_aug_raw,
        "y_aug_np": y_aug_np,
        "tid_to_rec": tid_to_rec,
        "avg_bridge": avg_bridge,
        "audit_rows": audit_rows,
        "direction_bank_meta": direction_meta,
        "safe_radius_ratio_mean": float(np.mean(safe_ratios)) if safe_ratios else 1.0,
        "manifold_margin_mean": float(np.mean(margins)) if margins else 0.0,
        "gamma_requested_mean": float(np.mean(gamma_req_vals)) if gamma_req_vals else 0.0,
        "gamma_used_mean": float(np.mean(gamma_used_vals)) if gamma_used_vals else 0.0,
        "gamma_zero_rate": float(np.mean([1.0 if g < 1e-12 else 0.0 for g in gamma_used_vals])) if gamma_used_vals else 0.0,
        "safe_clip_rate": float(np.mean(clip_flags)) if clip_flags else 0.0,
        "eta_safe": eta_safe,
        "candidate_total_count": int(len(z_aug)),
        "aug_total_count": int(len(z_aug)),
        "aug_dataset": aug_dataset_out,
    }
