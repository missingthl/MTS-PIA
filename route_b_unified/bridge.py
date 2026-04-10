from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch

from route_b_unified.types import BridgeResult, RepresentationState, TargetRoundState
from scripts.run_raw_bridge_probe import TrialRecord
from transforms.whiten_color_bridge import bridge_single, covariance_from_signal, logvec_to_spd


@dataclass(frozen=True)
class BridgeConfig:
    eps: float = 1e-4


def _record_to_trial_dict(rec: TrialRecord) -> Dict[str, object]:
    return {
        "trial_id_str": str(rec.tid),
        "label": int(rec.y),
        "x_trial": np.asarray(rec.x_raw, dtype=np.float32),
    }


def _covariance_from_trial_np(x: np.ndarray, eps: float) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64)
    xx = xx - xx.mean(axis=1, keepdims=True)
    denom = max(1, int(xx.shape[1]) - 1)
    cov = (xx @ xx.T) / float(denom)
    cov = 0.5 * (cov + cov.T)
    cov = cov + float(eps) * np.eye(cov.shape[0], dtype=np.float64)
    return cov.astype(np.float32)


def _json_text(obj: Dict[str, object]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _mean_cov_by_class(records: Sequence[TrialRecord]) -> Dict[int, np.ndarray]:
    buckets: Dict[int, List[np.ndarray]] = {}
    for r in records:
        buckets.setdefault(int(r.y), []).append(np.asarray(r.sigma_orig, dtype=np.float64))
    return {int(k): np.mean(np.stack(v, axis=0), axis=0) for k, v in buckets.items()}


def _pairwise_margin_stats(class_covs: Dict[int, np.ndarray]) -> Dict[str, float]:
    keys = sorted(int(k) for k in class_covs)
    if len(keys) <= 1:
        return {"pair_count": 0.0, "min": 0.0, "mean": 0.0, "max": 0.0}
    dists: List[float] = []
    for i, ki in enumerate(keys):
        for kj in keys[i + 1 :]:
            dists.append(float(np.linalg.norm(class_covs[ki] - class_covs[kj], ord="fro")))
    arr = np.asarray(dists, dtype=np.float64)
    return {
        "pair_count": float(arr.size),
        "min": float(np.min(arr)) if arr.size else 0.0,
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
    }


def apply_bridge(rep_state: RepresentationState, target_state: TargetRoundState, bridge_cfg: BridgeConfig, *, variant: str) -> BridgeResult:
    tid_to_rec = {str(r.tid): r for r in rep_state.train_records}
    aug_trials: List[TrialRecord] = []
    per_aug_bridge_meta: List[Dict[str, object]] = []

    cov_match = []
    cov_match_fro = []
    cov_match_logeuc = []
    cov_to_orig_fro = []
    cov_to_orig_logeuc = []
    energy_ratio = []
    cond_A = []
    raw_mean_shift = []

    classwise_shift: Dict[int, List[float]] = {}
    classwise_bridge_covs: Dict[int, List[np.ndarray]] = {}

    orig_class_covs = _mean_cov_by_class(rep_state.train_records)
    for i, (z_vec, y_val, tid) in enumerate(zip(target_state.z_aug, target_state.y_aug, target_state.tid_aug)):
        src = tid_to_rec[str(tid)]
        sigma_aug = logvec_to_spd(np.asarray(z_vec, dtype=np.float32), rep_state.mean_log_train)
        x_aug, bmeta = bridge_single(
            torch.from_numpy(np.asarray(src.x_raw, dtype=np.float32)),
            torch.from_numpy(np.asarray(src.sigma_orig, dtype=np.float32)),
            torch.from_numpy(np.asarray(sigma_aug, dtype=np.float32)),
            eps=float(bridge_cfg.eps),
        )
        x_aug_np = x_aug.cpu().numpy().astype(np.float32)
        sigma_emp = _covariance_from_trial_np(x_aug_np, float(bridge_cfg.eps))
        new_tid = f"{src.tid}__{variant}_r{int(target_state.round_index)}_aug_{i:06d}"
        aug_trials.append(
            TrialRecord(
                tid=new_tid,
                y=int(y_val),
                x_raw=x_aug_np,
                sigma_orig=np.asarray(sigma_emp, dtype=np.float32),
                log_cov=np.asarray(src.log_cov, dtype=np.float32),
                z=np.asarray(z_vec, dtype=np.float32),
            )
        )
        per_aug_bridge_meta.append(
            {
                "aug_index": int(i),
                "aug_tid": str(new_tid),
                "source_tid": str(src.tid),
                "label": int(y_val),
                "bridge_cov_match_error": float(bmeta["bridge_cov_match_error"]),
                "bridge_cov_match_error_fro": float(bmeta["bridge_cov_match_error_fro"]),
                "bridge_cov_match_error_logeuc": float(bmeta["bridge_cov_match_error_logeuc"]),
                "bridge_cov_to_orig_distance_fro": float(bmeta["bridge_cov_to_orig_distance_fro"]),
                "bridge_cov_to_orig_distance_logeuc": float(bmeta["bridge_cov_to_orig_distance_logeuc"]),
                "bridge_gain_norm": float(bmeta["bridge_gain_norm"]),
                "bridge_energy_ratio": float(bmeta["bridge_energy_ratio"]),
                "bridge_cond_A": float(bmeta["bridge_cond_A"]),
                "sigma_orig_min_eig": float(bmeta["sigma_orig_min_eig"]),
                "sigma_orig_max_eig": float(bmeta["sigma_orig_max_eig"]),
                "raw_mean_shift_abs": float(bmeta["raw_mean_shift_abs"]),
            }
        )
        cov_match.append(float(bmeta["bridge_cov_match_error"]))
        cov_match_fro.append(float(bmeta["bridge_cov_match_error_fro"]))
        cov_match_logeuc.append(float(bmeta["bridge_cov_match_error_logeuc"]))
        cov_to_orig_fro.append(float(bmeta["bridge_cov_to_orig_distance_fro"]))
        cov_to_orig_logeuc.append(float(bmeta["bridge_cov_to_orig_distance_logeuc"]))
        energy_ratio.append(float(bmeta["bridge_energy_ratio"]))
        cond_A.append(float(bmeta["bridge_cond_A"]))
        raw_mean_shift.append(float(bmeta["raw_mean_shift_abs"]))
        classwise_shift.setdefault(int(y_val), []).append(float(bmeta["raw_mean_shift_abs"]))
        classwise_bridge_covs.setdefault(int(y_val), []).append(np.asarray(sigma_emp, dtype=np.float64))

    bridge_class_covs = {
        int(k): np.mean(np.stack(v, axis=0), axis=0) for k, v in classwise_bridge_covs.items() if len(v) > 0
    }
    classwise_cov_dist = {
        str(int(k)): float(np.linalg.norm(bridge_class_covs[k] - orig_class_covs[k], ord="fro"))
        for k in sorted(set(orig_class_covs.keys()) & set(bridge_class_covs.keys()))
    }
    classwise_shift_summary = {
        str(int(k)): float(np.mean(v)) if v else 0.0 for k, v in sorted(classwise_shift.items(), key=lambda kv: kv[0])
    }
    orig_margin = _pairwise_margin_stats(orig_class_covs)
    bridge_margin = _pairwise_margin_stats(bridge_class_covs)
    margin_proxy = {
        "orig_min": float(orig_margin["min"]),
        "orig_mean": float(orig_margin["mean"]),
        "bridge_min": float(bridge_margin["min"]),
        "bridge_mean": float(bridge_margin["mean"]),
        "delta_min": float(bridge_margin["min"] - orig_margin["min"]),
        "delta_mean": float(bridge_margin["mean"] - orig_margin["mean"]),
        "ratio_mean": float(bridge_margin["mean"] / (orig_margin["mean"] + 1e-12)) if orig_margin["mean"] > 0 else 0.0,
    }
    cov_dist_mean = float(np.mean(list(classwise_cov_dist.values()))) if classwise_cov_dist else 0.0
    if cov_dist_mean <= 0.15 and float(margin_proxy["delta_mean"]) >= -0.05:
        task_risk = "classwise_stable_margin_preserved"
    elif float(margin_proxy["delta_mean"]) < -0.05:
        task_risk = "bridge_margin_shrink_risk"
    else:
        task_risk = "classwise_drift_watch"

    global_fidelity = {
        "bridge_aug_count": int(len(aug_trials)),
        "train_selected_aug_ratio": float(len(aug_trials) / max(1, len(rep_state.train_records))),
        "bridge_cov_match_error_mean": float(np.mean(cov_match)) if cov_match else 0.0,
        "bridge_cov_match_error_fro_mean": float(np.mean(cov_match_fro)) if cov_match_fro else 0.0,
        "bridge_cov_match_error_logeuc_mean": float(np.mean(cov_match_logeuc)) if cov_match_logeuc else 0.0,
        "bridge_cov_to_orig_distance_fro_mean": float(np.mean(cov_to_orig_fro)) if cov_to_orig_fro else 0.0,
        "bridge_cov_to_orig_distance_logeuc_mean": float(np.mean(cov_to_orig_logeuc)) if cov_to_orig_logeuc else 0.0,
        "energy_ratio_mean": float(np.mean(energy_ratio)) if energy_ratio else 0.0,
        "cond_A_mean": float(np.mean(cond_A)) if cond_A else 0.0,
        "raw_mean_shift_abs_mean": float(np.mean(raw_mean_shift)) if raw_mean_shift else 0.0,
    }
    classwise_fidelity = {
        "classwise_mean_shift_summary": classwise_shift_summary,
        "classwise_covariance_distortion_summary": classwise_cov_dist,
        "classwise_covariance_distortion_mean": cov_dist_mean,
    }
    orig_train_trial_dicts = [_record_to_trial_dict(r) for r in rep_state.train_records]
    aug_train_trial_dicts = [_record_to_trial_dict(r) for r in aug_trials]
    return BridgeResult(
        dataset=str(rep_state.dataset),
        seed=int(rep_state.seed),
        variant=str(variant),
        round_index=int(target_state.round_index),
        train_trials=list(orig_train_trial_dicts) + list(aug_train_trial_dicts),
        val_trials=list(rep_state.val_trial_dicts),
        test_trials=list(rep_state.test_trial_dicts),
        global_fidelity=global_fidelity,
        classwise_fidelity=classwise_fidelity,
        margin_proxy=margin_proxy,
        task_risk_comment=task_risk,
        orig_train_trials=list(orig_train_trial_dicts),
        aug_train_trials=list(aug_train_trial_dicts),
        per_aug_bridge_meta=list(per_aug_bridge_meta),
        meta={
            "classwise_mean_shift_summary_json": _json_text(classwise_shift_summary),
            "classwise_covariance_distortion_summary_json": _json_text(classwise_cov_dist),
        },
    )
