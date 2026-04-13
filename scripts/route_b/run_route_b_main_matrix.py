#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "scripts"))

from route_b_unified import (  # noqa: E402
    BridgeConfig,
    MiniRocketEvalConfig,
    RepresentationConfig,
    UnifiedPolicyConfig,
    apply_bridge,
    apply_target_feedback,
    build_representation,
    evaluate_bridge,
    init_policy,
    policy_step,
    update_policy,
)
from route_b_unified.types import (  # noqa: E402
    BridgeResult,
    EvaluatorPosterior,
    PolicyAction,
    TargetRoundState,
)
from scripts.raw_baselines.run_bridge_curriculum_pilot import (  # noqa: E402
    _bridge_aug_trials,
    _dataset_title,
    _ensure_dir,
    _fit_raw_minirocket,
    _format_mean_std,
    _records_to_trial_dicts,
    _risk_comment,
    _write_json,
)
from scripts.raw_baselines.run_raw_bridge_probe import TrialRecord, _apply_mean_log, _build_trial_records  # noqa: E402
from scripts.support.protocol_split_utils import resolve_inner_train_val_split, resolve_protocol_split  # noqa: E402
from scripts.legacy_phase.run_phase15_multiround_curriculum_probe import (  # noqa: E402
    _build_curriculum_aug_candidates,
    _compute_direction_intrusion,
    _mech_dir_maps,
)
from scripts.legacy_phase.run_phase15_step1b_multidir_matrix import _compute_mech_metrics  # noqa: E402
from scripts.route_b.run_route_b_augmentation_train import (  # noqa: E402
    _attach_bridge_payload,
    _build_round_rows,
    _choose_best_round,
    _dataset_group,
    _fit_variant,
    _load_trials_for_runner,
    _protocol_label,
    _stack_feature,
)
from transforms.whiten_color_bridge import bridge_single, logvec_to_spd  # noqa: E402


@dataclass(frozen=True)
class LegacyBundle:
    raw_metrics: Dict[str, object]
    single_metrics: Dict[str, object]
    multiround_metrics: Dict[str, object]
    split_meta: Dict[str, object]
    inner_meta: Dict[str, object]
    selection_round_rows: List[Dict[str, object]]
    best_round_id: int
    best_round_note: str
    best_round_val_f1: float
    single_bridge_meta: Dict[str, object]
    best_bridge_meta: Dict[str, object]


@dataclass(frozen=True)
class UnifiedBundle:
    final_row: Dict[str, object]
    policy_rows: List[Dict[str, object]]
    bridge_rows: List[Dict[str, object]]
    feedback_rows: List[Dict[str, object]]
    raw_val: Dict[str, float]
    raw_test: Dict[str, float]


def _parse_csv_list(text: str) -> List[str]:
    out = [tok.strip().lower() for tok in str(text).split(",") if tok.strip()]
    if not out:
        raise ValueError("dataset list cannot be empty")
    return out


def _parse_seed_list(text: str) -> List[int]:
    out = sorted(set(int(tok.strip()) for tok in str(text).split(",") if tok.strip()))
    if not out:
        raise ValueError("seed list cannot be empty")
    return out


def _json_text(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _mean_std_text(values: Sequence[float]) -> str:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return "0.0000 +/- 0.0000"
    return _format_mean_std(arr.tolist())


def _summary_stats(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def _variant_label(*, variant: str, f1: float, raw_f1: float, single_f1: float) -> str:
    if variant == "raw_only":
        return "flat"
    delta_vs_raw = float(f1 - raw_f1)
    if delta_vs_raw >= 0.002:
        return "positive"
    if delta_vs_raw <= -0.002 and float(f1 - single_f1) >= 0.002:
        return "repair-only"
    if abs(delta_vs_raw) < 0.002:
        return "flat"
    return "negative"


def _legacy_ranking_proxy(direction_score: Dict[str, float], gamma_before: Dict[str, float]) -> List[int]:
    if direction_score:
        return [int(k) for k, _ in sorted(direction_score.items(), key=lambda kv: (-float(kv[1]), int(kv[0])))]
    return [int(k) for k, _ in sorted(gamma_before.items(), key=lambda kv: (-float(kv[1]), int(kv[0])))]


def _covariance_from_trial_np(x: np.ndarray, eps: float) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64)
    xx = xx - xx.mean(axis=1, keepdims=True)
    denom = max(1, int(xx.shape[1]) - 1)
    cov = (xx @ xx.T) / float(denom)
    cov = 0.5 * (cov + cov.T)
    cov = cov + float(eps) * np.eye(cov.shape[0], dtype=np.float64)
    return cov.astype(np.float32)


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


def _record_to_trial_dict(rec: TrialRecord) -> Dict[str, object]:
    return {
        "trial_id_str": str(rec.tid),
        "label": int(rec.y),
        "x_trial": np.asarray(rec.x_raw, dtype=np.float32),
    }


def _apply_bridge_full_train(
    *,
    dataset: str,
    seed: int,
    variant: str,
    full_train_records: Sequence[TrialRecord],
    full_train_trials: Sequence[Dict[str, object]],
    val_trials: Sequence[Dict[str, object]],
    test_trials: Sequence[Dict[str, object]],
    mean_log_train: np.ndarray,
    target_state: TargetRoundState,
    bridge_cfg: BridgeConfig,
) -> BridgeResult:
    tid_to_rec = {str(r.tid): r for r in full_train_records}
    aug_trials: List[TrialRecord] = []
    cov_match = []
    cov_match_logeuc = []
    cov_to_orig = []
    energy_ratio = []
    cond_A = []
    raw_mean_shift = []
    classwise_shift: Dict[int, List[float]] = {}
    classwise_bridge_covs: Dict[int, List[np.ndarray]] = {}
    orig_class_covs = _mean_cov_by_class(full_train_records)

    for i, (z_vec, y_val, tid) in enumerate(zip(target_state.z_aug, target_state.y_aug, target_state.tid_aug)):
        src = tid_to_rec[str(tid)]
        sigma_aug = logvec_to_spd(np.asarray(z_vec, dtype=np.float32), mean_log_train)
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
        cov_match.append(float(bmeta["bridge_cov_match_error"]))
        cov_match_logeuc.append(float(bmeta["bridge_cov_match_error_logeuc"]))
        cov_to_orig.append(float(bmeta["bridge_cov_to_orig_distance_logeuc"]))
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
        "train_selected_aug_ratio": float(len(aug_trials) / max(1, len(full_train_records))),
        "bridge_cov_match_error_mean": float(np.mean(cov_match)) if cov_match else 0.0,
        "bridge_cov_match_error_logeuc_mean": float(np.mean(cov_match_logeuc)) if cov_match_logeuc else 0.0,
        "bridge_cov_to_orig_distance_logeuc_mean": float(np.mean(cov_to_orig)) if cov_to_orig else 0.0,
        "energy_ratio_mean": float(np.mean(energy_ratio)) if energy_ratio else 0.0,
        "cond_A_mean": float(np.mean(cond_A)) if cond_A else 0.0,
        "raw_mean_shift_abs_mean": float(np.mean(raw_mean_shift)) if raw_mean_shift else 0.0,
    }
    classwise_fidelity = {
        "classwise_mean_shift_summary": classwise_shift_summary,
        "classwise_covariance_distortion_summary": classwise_cov_dist,
        "classwise_covariance_distortion_mean": cov_dist_mean,
    }
    return BridgeResult(
        dataset=str(dataset),
        seed=int(seed),
        variant=str(variant),
        round_index=int(target_state.round_index),
        train_trials=list(full_train_trials) + [_record_to_trial_dict(r) for r in aug_trials],
        val_trials=list(val_trials),
        test_trials=list(test_trials),
        global_fidelity=global_fidelity,
        classwise_fidelity=classwise_fidelity,
        margin_proxy=margin_proxy,
        task_risk_comment=task_risk,
        meta={},
    )


def _build_eval_cfg(args: argparse.Namespace, out_root: str) -> MiniRocketEvalConfig:
    return MiniRocketEvalConfig(
        out_root=str(out_root),
        window_sec=float(args.window_sec),
        hop_sec=float(args.hop_sec),
        prop_win_ratio=float(args.prop_win_ratio),
        prop_hop_ratio=float(args.prop_hop_ratio),
        min_window_len_samples=int(args.min_window_len_samples),
        min_hop_len_samples=int(args.min_hop_len_samples),
        nominal_cap_k=int(args.nominal_cap_k),
        cap_sampling_policy=str(args.cap_sampling_policy),
        aggregation_mode=str(args.aggregation_mode),
        n_kernels=int(args.n_kernels),
        n_jobs=int(args.n_jobs),
        memmap_threshold_gb=float(args.memmap_threshold_gb),
    )


def _evaluate_raw_official(
    *,
    dataset: str,
    train_full_trials: Sequence[Dict[str, object]],
    test_trials: Sequence[Dict[str, object]],
    seed: int,
    args: argparse.Namespace,
    out_dir: str,
) -> Dict[str, float]:
    metrics, _ = _fit_variant(
        dataset=dataset,
        train_trials=train_full_trials,
        test_trials=test_trials,
        seed=int(seed),
        args=args,
        stage_seed_dir=out_dir,
    )
    return {"acc": float(metrics["trial_acc"]), "macro_f1": float(metrics["trial_macro_f1"])}


def _run_legacy_official(
    *,
    dataset: str,
    seed: int,
    args: argparse.Namespace,
    out_dir: str,
) -> LegacyBundle:
    all_trials = _load_trials_for_runner(dataset, args)
    train_full_trials, test_trials, split_meta = resolve_protocol_split(
        dataset=dataset,
        all_trials=all_trials,
        seed=int(seed),
        allow_random_fallback=False,
    )
    train_core_trials, val_trials, inner_meta = resolve_inner_train_val_split(
        train_trials=train_full_trials,
        seed=int(seed) + 1701,
        val_fraction=float(args.val_fraction),
        fallback_fraction=float(args.val_fallback_fraction),
    )
    raw_metrics = _evaluate_raw_official(
        dataset=dataset,
        train_full_trials=train_full_trials,
        test_trials=test_trials,
        seed=int(seed),
        args=args,
        out_dir=os.path.join(out_dir, "raw_only"),
    )

    train_core_tmp, mean_log_core = _build_trial_records(train_core_trials, spd_eps=float(args.spd_eps))
    train_core_records = _apply_mean_log(train_core_tmp, mean_log_core)
    X_core, y_core, tid_core = _stack_feature(train_core_records)
    if len(val_trials) > 0:
        val_tmp, _ = _build_trial_records(val_trials, spd_eps=float(args.spd_eps))
        val_records = _apply_mean_log(val_tmp, mean_log_core)
        X_val, y_val, tid_val = _stack_feature(val_records)
    else:
        X_val = y_val = tid_val = None

    from scripts.legacy_phase.run_phase15_step1b_multidir_matrix import _build_direction_bank_d1, _build_multidir_aug_candidates

    direction_bank_core, _ = _build_direction_bank_d1(
        X_train=X_core,
        k_dir=int(args.k_dir),
        seed=int(seed * 10000 + int(args.k_dir) * 113 + 17),
        n_iters=int(args.pia_n_iters),
        activation=str(args.pia_activation),
        bias_update_mode=str(args.pia_bias_update_mode),
        c_repr=float(args.pia_c_repr),
    )
    selection_round_rows = _build_round_rows(
        dataset=dataset,
        seed=int(seed),
        direction_bank=direction_bank_core,
        X_train_base=X_core,
        y_train_base=y_core,
        tid_train=tid_core,
        X_eval=X_val,
        y_eval=y_val,
        tid_eval=tid_val,
        args=args,
        seed_offset=400000,
        progress_tag="select",
    )
    best_round_id, best_round_note, best_round_val_f1 = _choose_best_round(
        selection_round_rows, has_val=bool(len(val_trials) > 0)
    )

    full_tmp, mean_log_full = _build_trial_records(train_full_trials, spd_eps=float(args.spd_eps))
    full_records = _apply_mean_log(full_tmp, mean_log_full)
    X_full, y_full, tid_full = _stack_feature(full_records)
    direction_bank_full, _ = _build_direction_bank_d1(
        X_train=X_full,
        k_dir=int(args.k_dir),
        seed=int(seed * 10000 + int(args.k_dir) * 113 + 17),
        n_iters=int(args.pia_n_iters),
        activation=str(args.pia_activation),
        bias_update_mode=str(args.pia_bias_update_mode),
        c_repr=float(args.pia_c_repr),
    )
    X_single, y_single, tid_single, _src_single, _dir_single, _single_aug_meta = _build_multidir_aug_candidates(
        X_train=X_full,
        y_train=y_full,
        tid_train=tid_full,
        direction_bank=direction_bank_full,
        subset_size=int(args.subset_size),
        gamma=float(args.pia_gamma),
        multiplier=int(args.pia_multiplier),
        seed=int(seed + 100000 + int(args.k_dir) * 101 + int(args.subset_size) * 7),
    )
    single_records, single_bridge_meta = _bridge_aug_trials(
        train_records=full_records,
        mean_log_train=mean_log_full,
        z_aug=np.asarray(X_single, dtype=np.float32),
        y_aug=np.asarray(y_single, dtype=np.int64),
        tid_aug=np.asarray(tid_single),
        variant_tag="single_round",
        bridge_eps=float(args.bridge_eps),
    )
    single_metrics = _evaluate_raw_official(
        dataset=dataset,
        train_full_trials=list(train_full_trials) + _records_to_trial_dicts(single_records),
        test_trials=test_trials,
        seed=int(seed),
        args=args,
        out_dir=os.path.join(out_dir, "legacy_single_round_bridge"),
    )

    full_round_rows = _build_round_rows(
        dataset=dataset,
        seed=int(seed),
        direction_bank=direction_bank_full,
        X_train_base=X_full,
        y_train_base=y_full,
        tid_train=tid_full,
        X_eval=None,
        y_eval=None,
        tid_eval=None,
        args=args,
        seed_offset=500000,
        progress_tag="full",
    )
    full_round_rows = _attach_bridge_payload(
        train_records=full_records,
        mean_log_train=mean_log_full,
        round_rows=full_round_rows,
        bridge_eps=float(args.bridge_eps),
        variant_prefix="multiround",
    )
    best_full = next((r for r in full_round_rows if int(r["round_id"]) == int(best_round_id)), None)
    if best_full is None:
        best_full = full_round_rows[0]
    multiround_metrics = _evaluate_raw_official(
        dataset=dataset,
        train_full_trials=list(train_full_trials) + list(best_full["aug_trials"]),
        test_trials=test_trials,
        seed=int(seed),
        args=args,
        out_dir=os.path.join(out_dir, "legacy_multiround_bridge"),
    )
    return LegacyBundle(
        raw_metrics=raw_metrics,
        single_metrics=single_metrics,
        multiround_metrics=multiround_metrics,
        split_meta=dict(split_meta),
        inner_meta=dict(inner_meta),
        selection_round_rows=selection_round_rows,
        best_round_id=int(best_round_id),
        best_round_note=str(best_round_note),
        best_round_val_f1=float(best_round_val_f1) if np.isfinite(best_round_val_f1) else np.nan,
        single_bridge_meta=single_bridge_meta,
        best_bridge_meta=best_full["bridge_meta"],
    )


def _run_unified_official(
    *,
    dataset: str,
    seed: int,
    args: argparse.Namespace,
    out_dir: str,
    variant: str,
) -> UnifiedBundle:
    all_trials = _load_trials_for_runner(dataset, args)
    train_full_trials, test_trials, _split_meta = resolve_protocol_split(
        dataset=dataset,
        all_trials=all_trials,
        seed=int(seed),
        allow_random_fallback=False,
    )
    rep_state = build_representation(
        RepresentationConfig(
            dataset=str(dataset),
            seed=int(seed),
            val_fraction=float(args.val_fraction),
            spd_eps=float(args.spd_eps),
            natops_root=args.natops_root,
            selfregulationscp1_root=args.selfregulationscp1_root,
        )
    )
    eval_cfg = _build_eval_cfg(args, out_dir)
    bridge_cfg = BridgeConfig(eps=float(args.bridge_eps))
    raw_val = _evaluate_raw_official(
        dataset=dataset,
        train_full_trials=rep_state.train_trial_dicts,
        test_trials=rep_state.val_trial_dicts,
        seed=int(seed),
        args=args,
        out_dir=os.path.join(out_dir, "raw_val_reference"),
    )
    raw_test = _evaluate_raw_official(
        dataset=dataset,
        train_full_trials=train_full_trials,
        test_trials=test_trials,
        seed=int(seed),
        args=args,
        out_dir=os.path.join(out_dir, "raw_test_reference"),
    )

    policy_cfg = UnifiedPolicyConfig(
        variant=str(variant),
        n_rounds=int(args.curriculum_rounds),
        select_top_k=int(args.policy_top_k),
        k_dir=int(args.k_dir),
        gamma_init=float(args.curriculum_init_gamma),
        expand_factor=float(args.curriculum_expand_factor),
        shrink_factor=float(args.curriculum_shrink_factor),
        gamma_max=float(args.curriculum_gamma_max),
        freeze_eps=float(args.curriculum_freeze_eps),
        stop_patience=int(args.stop_patience),
        feedback_enabled=bool(variant == "unified_feedback"),
        feedback_eta_reward=float(args.feedback_eta_reward),
        feedback_eta_penalty=float(args.feedback_eta_penalty),
        beta=float(args.lraes_beta),
        reg_lambda=float(args.lraes_reg_lambda),
        top_k_per_class=int(args.lraes_top_k_per_class),
        rank_tol=float(args.lraes_rank_tol),
        eig_pos_eps=float(args.lraes_eig_pos_eps),
        lraes_knn_k=int(args.lraes_knn_k),
        lraes_boundary_quantile=float(args.lraes_boundary_quantile),
        lraes_interior_quantile=float(args.lraes_interior_quantile),
        lraes_hetero_k=int(args.lraes_hetero_k),
    )
    state = init_policy(rep_state, policy_cfg)
    policy_rows: List[Dict[str, object]] = []
    bridge_rows: List[Dict[str, object]] = []
    feedback_rows: List[Dict[str, object]] = []
    best_bundle: Dict[str, object] | None = None
    best_val_f1 = float("-inf")
    prev_posterior: EvaluatorPosterior | None = None

    for round_idx in range(1, int(policy_cfg.n_rounds) + 1):
        active_ids = [
            int(i)
            for i in np.where(
                (~np.asarray(state.prior_frozen_mask, dtype=bool))
                & (np.asarray(state.gamma_by_dir, dtype=np.float64) > float(policy_cfg.freeze_eps))
            )[0].tolist()
        ]
        ranking = [int(i) for i in np.argsort(np.asarray(state.current_scores, dtype=np.float64))[::-1].tolist()]
        action, state = policy_step(rep_state, state, prev_posterior, policy_cfg)
        policy_rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "variant": str(variant),
                "round_index": int(round_idx),
                "direction_ranking": _json_text(ranking),
                "active_direction_ids": _json_text(active_ids),
                "selected_directions": _json_text(action.selected_dir_ids),
                "direction_weights": _json_text(action.direction_weights),
                "step_sizes": _json_text(action.step_sizes),
                "top_k": int(len(action.selected_dir_ids)),
                "entropy": float(action.entropy),
                "stop_reason": str(action.stop_reason),
                "score_before": _json_text({str(i): float(v) for i, v in enumerate(np.asarray(state.current_scores, dtype=np.float64))}),
                "gamma_before": _json_text({str(i): float(v) for i, v in enumerate(np.asarray(state.gamma_by_dir, dtype=np.float64))}),
                "reward_penalty_summary": "{}",
            }
        )
        X_aug, y_aug, tid_aug, src_aug, dir_aug, aug_meta = _build_curriculum_aug_candidates(
            X_train=rep_state.X_train,
            y_train=rep_state.y_train,
            tid_train=rep_state.tid_train,
            direction_bank=state.direction_bank,
            direction_probs=action.direction_probs_vector,
            gamma_by_dir=action.gamma_vector,
            multiplier=int(args.pia_multiplier),
            seed=int(seed + 700000 + round_idx * 1009 + (37 if variant == "unified_feedback" else 23)),
        )
        mech = _compute_mech_metrics(
            X_train_real=rep_state.X_train,
            y_train_real=rep_state.y_train,
            X_aug_generated=X_aug,
            y_aug_generated=y_aug,
            X_aug_accepted=X_aug,
            y_aug_accepted=y_aug,
            X_src_accepted=src_aug,
            dir_generated=dir_aug,
            dir_accepted=dir_aug,
            seed=int(seed),
            linear_c=float(args.linear_c),
            class_weight=str(args.linear_class_weight),
            linear_max_iter=int(args.linear_max_iter),
            knn_k=int(args.mech_knn_k),
            max_aug_for_mech=int(args.mech_max_aug_for_metrics),
            max_real_knn_ref=int(args.mech_max_real_knn_ref),
            max_real_knn_query=int(args.mech_max_real_knn_query),
            progress_prefix=f"[route-b-main][{dataset}][seed={seed}][{variant}][r{round_idx}][mech]",
        )
        intrusion_by_dir = _compute_direction_intrusion(
            X_anchor=rep_state.X_train,
            y_anchor=rep_state.y_train,
            X_aug_accepted=X_aug,
            y_aug_accepted=y_aug,
            dir_accepted=dir_aug,
            seed=int(seed),
            knn_k=int(args.mech_knn_k),
            max_eval=int(args.mech_max_aug_for_metrics),
        )
        dir_maps = _mech_dir_maps(mech, intrusion_by_dir=intrusion_by_dir)
        state, direction_state, direction_budget_score = apply_target_feedback(
            state,
            margin_by_dir=dir_maps["margin_drop_median"],
            flip_by_dir=dir_maps["flip_rate"],
            intrusion_by_dir=dir_maps["intrusion"],
            policy_cfg=policy_cfg,
        )
        target_state = TargetRoundState(
            round_index=int(round_idx),
            z_aug=np.asarray(X_aug, dtype=np.float32),
            y_aug=np.asarray(y_aug, dtype=np.int64),
            tid_aug=np.asarray(tid_aug, dtype=object),
            mech=mech,
            dir_maps=dir_maps,
            aug_meta=dict(aug_meta),
            action=action,
            direction_state={int(k): str(v) for k, v in direction_state.items()},
            direction_budget_score={int(k): float(v) for k, v in direction_budget_score.items()},
        )
        bridge_val = apply_bridge(rep_state, target_state, bridge_cfg, variant=str(variant))
        posterior_val = evaluate_bridge(
            bridge_val,
            eval_cfg,
            split_name="val",
            target_state=target_state,
            round_gain_proxy=0.0 if not np.isfinite(best_val_f1) else 0.0,
        )
        posterior_val.round_gain_proxy = 0.0 if not np.isfinite(best_val_f1) else float(posterior_val.macro_f1) - float(best_val_f1)
        bridge_rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "variant": str(variant),
                "round_index": int(round_idx),
                "bridge_cov_match_error": float(bridge_val.global_fidelity["bridge_cov_match_error_mean"]),
                "bridge_cov_to_orig_distance": float(bridge_val.global_fidelity["bridge_cov_to_orig_distance_logeuc_mean"]),
                "energy_ratio": float(bridge_val.global_fidelity["energy_ratio_mean"]),
                "cond_A": float(bridge_val.global_fidelity["cond_A_mean"]),
                "raw_mean_shift_abs": float(bridge_val.global_fidelity["raw_mean_shift_abs_mean"]),
                "classwise_mean_shift": _json_text(bridge_val.classwise_fidelity["classwise_mean_shift_summary"]),
                "classwise_covariance_distortion": _json_text(bridge_val.classwise_fidelity["classwise_covariance_distortion_summary"]),
                "inter_class_margin_proxy": _json_text(bridge_val.margin_proxy),
                "task_risk_comment": str(bridge_val.task_risk_comment),
                "val_macro_f1": float(posterior_val.macro_f1),
            }
        )
        if bool(policy_cfg.feedback_enabled):
            state, feedback_summary = update_policy(state, posterior_val, policy_cfg)
            scores_after = {str(i): float(v) for i, v in enumerate(np.asarray(state.current_scores, dtype=np.float64))}
            gamma_after = {str(i): float(v) for i, v in enumerate(np.asarray(state.gamma_by_dir, dtype=np.float64))}
            feedback_rows.append(
                {
                    "dataset": feedback_summary.dataset,
                    "seed": int(feedback_summary.seed),
                    "variant": feedback_summary.variant,
                    "round_index": int(feedback_summary.round_index),
                    "rewarded_directions": _json_text(feedback_summary.rewarded_dirs),
                    "penalized_directions": _json_text(feedback_summary.penalized_dirs),
                    "rank_change_summary": _json_text(feedback_summary.rank_change_summary),
                    "collapse_warning": feedback_summary.collapse_warning,
                    "overspread_warning": feedback_summary.overspread_warning,
                    "reward_summary": _json_text(feedback_summary.reward_summary),
                    "penalty_summary": _json_text(feedback_summary.penalty_summary),
                    "score_after": _json_text(scores_after),
                    "gamma_after": _json_text(gamma_after),
                    "round_gain_proxy": float(posterior_val.round_gain_proxy),
                    "val_macro_f1": float(posterior_val.macro_f1),
                }
            )
            policy_rows[-1]["reward_penalty_summary"] = _json_text(
                {"rewarded": feedback_summary.rewarded_dirs, "penalized": feedback_summary.penalized_dirs}
            )
        else:
            feedback_rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "variant": str(variant),
                    "round_index": int(round_idx),
                    "rewarded_directions": "[]",
                    "penalized_directions": "[]",
                    "rank_change_summary": "{}",
                    "collapse_warning": "none",
                    "overspread_warning": "none",
                    "reward_summary": "{}",
                    "penalty_summary": "{}",
                    "score_after": policy_rows[-1]["score_before"],
                    "gamma_after": policy_rows[-1]["gamma_before"],
                    "round_gain_proxy": float(posterior_val.round_gain_proxy),
                    "val_macro_f1": float(posterior_val.macro_f1),
                }
            )
        if float(posterior_val.macro_f1) > float(best_val_f1):
            best_val_f1 = float(posterior_val.macro_f1)
            best_bundle = {
                "target_state": target_state,
                "posterior_val": posterior_val,
                "direction_bank": np.asarray(state.direction_bank, dtype=np.float32).copy(),
            }
        prev_posterior = posterior_val
        if state.stop_flag:
            policy_rows[-1]["stop_reason"] = str(state.stop_reason)
            break

    if best_bundle is None:
        raise RuntimeError(f"No valid unified round for dataset={dataset}, seed={seed}, variant={variant}")

    full_tmp, _ = _build_trial_records(train_full_trials, spd_eps=float(args.spd_eps))
    full_records = _apply_mean_log(full_tmp, rep_state.mean_log_train)
    X_full, y_full, tid_full = _stack_feature(full_records)
    best_target = best_bundle["target_state"]
    X_aug_full, y_aug_full, tid_aug_full, _src_aug_full, _dir_aug_full, _aug_meta_full = _build_curriculum_aug_candidates(
        X_train=X_full,
        y_train=y_full,
        tid_train=tid_full,
        direction_bank=np.asarray(best_bundle["direction_bank"], dtype=np.float32),
        direction_probs=best_target.action.direction_probs_vector,
        gamma_by_dir=best_target.action.gamma_vector,
        multiplier=int(args.pia_multiplier),
        seed=int(seed + 970000 + int(best_target.round_index) * 997),
    )
    full_target_state = TargetRoundState(
        round_index=int(best_target.round_index),
        z_aug=np.asarray(X_aug_full, dtype=np.float32),
        y_aug=np.asarray(y_aug_full, dtype=np.int64),
        tid_aug=np.asarray(tid_aug_full, dtype=object),
        mech=best_target.mech,
        dir_maps=best_target.dir_maps,
        aug_meta=dict(best_target.aug_meta),
        action=best_target.action,
        direction_state=dict(best_target.direction_state),
        direction_budget_score=dict(best_target.direction_budget_score),
    )
    bridge_full = _apply_bridge_full_train(
        dataset=dataset,
        seed=int(seed),
        variant=str(variant),
        full_train_records=full_records,
        full_train_trials=train_full_trials,
        val_trials=rep_state.val_trial_dicts,
        test_trials=test_trials,
        mean_log_train=rep_state.mean_log_train,
        target_state=full_target_state,
        bridge_cfg=bridge_cfg,
    )
    raw_metrics, _ = _fit_variant(
        dataset=dataset,
        train_trials=bridge_full.train_trials,
        test_trials=test_trials,
        seed=int(seed),
        args=args,
        stage_seed_dir=os.path.join(out_dir, "final_eval"),
    )
    final_row = {
        "dataset": dataset,
        "seed": int(seed),
        "variant": str(variant),
        "best_round": int(best_target.round_index),
        "val_acc": float(best_bundle["posterior_val"].acc),
        "val_macro_f1": float(best_bundle["posterior_val"].macro_f1),
        "test_acc": float(raw_metrics["trial_acc"]),
        "test_macro_f1": float(raw_metrics["trial_macro_f1"]),
        "direction_usage_entropy": float(best_target.action.entropy),
        "worst_dir_summary": str(best_bundle["posterior_val"].worst_dir_summary),
        "bridge_task_risk_comment": str(bridge_full.task_risk_comment),
        "target_health_comment": "unified_feedback_direction_profile_stable" if variant == "unified_feedback" else "unified_static_direction_profile_stable",
        "bridge_cov_match_error": float(bridge_full.global_fidelity["bridge_cov_match_error_mean"]),
        "bridge_cov_to_orig_distance": float(bridge_full.global_fidelity["bridge_cov_to_orig_distance_logeuc_mean"]),
        "energy_ratio": float(bridge_full.global_fidelity["energy_ratio_mean"]),
        "cond_A": float(bridge_full.global_fidelity["cond_A_mean"]),
        "raw_mean_shift_abs": float(bridge_full.global_fidelity["raw_mean_shift_abs_mean"]),
        "solver_state_summary": _json_text(best_target.direction_state),
        "inter_class_margin_ratio": float(bridge_full.margin_proxy.get("ratio_mean", 0.0)),
        "classwise_mean_shift_max": max(
            [abs(float(v)) for v in bridge_full.classwise_fidelity.get("classwise_mean_shift_summary", {}).values()],
            default=0.0,
        ),
        "classwise_distortion_max": max(
            [float(v) for v in bridge_full.classwise_fidelity.get("classwise_covariance_distortion_summary", {}).values()],
            default=0.0,
        ),
        "task_risk_comment": str(bridge_full.task_risk_comment),
        "fully_risk_dominated": bool(
            np.max(np.asarray(best_bundle["direction_bank"], dtype=np.float64)) <= 0.0 if np.asarray(best_bundle["direction_bank"]).size else False
        ),
    }
    return UnifiedBundle(
        final_row=final_row,
        policy_rows=policy_rows,
        bridge_rows=bridge_rows,
        feedback_rows=feedback_rows,
        raw_val=raw_val,
        raw_test=raw_test,
    )


def _build_natops_alignment(
    *,
    out_root: str,
    legacy_debug_rows: List[Dict[str, object]],
    unified_policy_rows: List[Dict[str, object]],
    main_matrix_rows: List[Dict[str, object]],
) -> None:
    legacy_df = pd.DataFrame([r for r in legacy_debug_rows if r["dataset"] == "natops"])
    unified_df = pd.DataFrame(
        [r for r in unified_policy_rows if r["dataset"] == "natops" and r["variant"] == "unified_static_bridge"]
    )
    final_df = pd.DataFrame([r for r in main_matrix_rows if r["dataset"] == "natops"])
    if legacy_df.empty or unified_df.empty or final_df.empty:
        return
    rows: List[Dict[str, object]] = []
    for seed in sorted(set(legacy_df["seed"].tolist()) | set(unified_df["seed"].tolist())):
        legacy_seed = legacy_df[legacy_df["seed"] == seed]
        unified_seed = unified_df[unified_df["seed"] == seed]
        legacy_final = final_df[(final_df["seed"] == seed) & (final_df["variant"] == "legacy_multiround_bridge")]
        unified_final = final_df[(final_df["seed"] == seed) & (final_df["variant"] == "unified_static_bridge")]
        round_ids = sorted(set(legacy_seed["round_index"].tolist()) | set(unified_seed["round_index"].tolist()))
        for rid in round_ids:
            lrow = legacy_seed[legacy_seed["round_index"] == rid]
            urow = unified_seed[unified_seed["round_index"] == rid]
            l = lrow.iloc[0].to_dict() if len(lrow) else {}
            u = urow.iloc[0].to_dict() if len(urow) else {}
            rows.append(
                {
                    "dataset": "natops",
                    "seed": int(seed),
                    "round_index": int(rid),
                    "legacy_direction_ranking_proxy": l.get("direction_ranking_proxy", ""),
                    "unified_direction_ranking": u.get("direction_ranking", ""),
                    "legacy_selected_directions": l.get("selected_directions_proxy", ""),
                    "unified_selected_directions": u.get("selected_directions", ""),
                    "legacy_top_k_proxy": l.get("top_k_proxy", ""),
                    "unified_top_k": u.get("top_k", ""),
                    "legacy_keep_prune_proxy": l.get("keep_prune_proxy", ""),
                    "unified_keep_prune": u.get("keep_prune_proxy", ""),
                    "legacy_step_size_summary": l.get("step_size_summary", ""),
                    "unified_step_sizes": u.get("step_sizes", ""),
                    "legacy_stop_reason": l.get("stop_reason_proxy", ""),
                    "unified_stop_reason": u.get("stop_reason", ""),
                    "legacy_best_round": legacy_final["best_round"].iloc[0] if len(legacy_final) else "",
                    "unified_best_round": unified_final["best_round"].iloc[0] if len(unified_final) else "",
                    "legacy_macro_f1": legacy_final["macro_f1"].iloc[0] if len(legacy_final) else np.nan,
                    "unified_macro_f1": unified_final["macro_f1"].iloc[0] if len(unified_final) else np.nan,
                    "delta_vs_raw_legacy": legacy_final["delta_vs_raw"].iloc[0] if len(legacy_final) else np.nan,
                    "delta_vs_raw_unified": unified_final["delta_vs_raw"].iloc[0] if len(unified_final) else np.nan,
                    "delta_vs_legacy_multiround_unified": unified_final["delta_vs_legacy_multiround"].iloc[0] if len(unified_final) else np.nan,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_root, "natops_unified_static_alignment_debug.csv"), index=False)
    if final_df.empty:
        note = "# NATOPS Unified Static Alignment\n\nNo NATOPS rows available.\n"
    else:
        legacy_mean = float(final_df[final_df["variant"] == "legacy_multiround_bridge"]["macro_f1"].mean())
        unified_mean = float(final_df[final_df["variant"] == "unified_static_bridge"]["macro_f1"].mean())
        delta = unified_mean - legacy_mean
        note = "\n".join(
            [
                "# NATOPS Unified Static Alignment",
                "",
                f"- `legacy_multiround_bridge` mean F1: `{legacy_mean:.4f}`",
                f"- `unified_static_bridge` mean F1: `{unified_mean:.4f}`",
                f"- `delta_vs_legacy_multiround`: `{delta:+.4f}`",
                "- `legacy_direction_ranking_proxy` is a budget-score proxy because legacy multiround has no explicit score-ranked top-K selector.",
                "- `unified_direction_ranking` is the explicit score ranking from the current unified policy before selection.",
                "- Current calibration order remains: direction ranking -> top-K/keep-prune -> gamma budget -> stop rule.",
                "",
            ]
        )
    with open(os.path.join(out_root, "natops_unified_static_alignment_note.md"), "w", encoding="utf-8") as f:
        f.write(note)


def _build_har_stability_check(*, out_root: str, main_matrix_rows: List[Dict[str, object]]) -> None:
    df = pd.DataFrame([r for r in main_matrix_rows if r["dataset"] == "har"])
    rows: List[Dict[str, object]] = []
    for seed in sorted(df["seed"].unique().tolist()) if not df.empty else []:
        raw = df[(df["seed"] == seed) & (df["variant"] == "raw_only")].iloc[0]
        legacy = df[(df["seed"] == seed) & (df["variant"] == "legacy_multiround_bridge")].iloc[0]
        unified = df[(df["seed"] == seed) & (df["variant"] == "unified_static_bridge")].iloc[0]
        delta_legacy = float(unified["macro_f1"] - legacy["macro_f1"])
        delta_raw = float(unified["macro_f1"] - raw["macro_f1"])
        stability_flag = "stable" if delta_legacy >= -0.005 else "overaggressive"
        rows.append(
            {
                "dataset": "har",
                "seed": int(seed),
                "raw_macro_f1": float(raw["macro_f1"]),
                "legacy_multiround_macro_f1": float(legacy["macro_f1"]),
                "unified_static_macro_f1": float(unified["macro_f1"]),
                "delta_unified_vs_raw": delta_raw,
                "delta_unified_vs_legacy_multiround": delta_legacy,
                "stability_flag": stability_flag,
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(out_root, "har_unified_stability_check.csv"), index=False)


def _flag_yesno(flag: bool) -> str:
    return "yes" if bool(flag) else "no"


def _sign_flip_count(prev_map: Dict[str, float] | None, curr_map: Dict[str, float]) -> int:
    if not prev_map:
        return 0
    flips = 0
    for key in sorted(set(prev_map.keys()) | set(curr_map.keys()), key=int):
        prev = float(prev_map.get(key, 0.0))
        curr = float(curr_map.get(key, 0.0))
        if prev == 0.0 or curr == 0.0:
            continue
        if (prev > 0.0) != (curr > 0.0):
            flips += 1
    return int(flips)


def _build_scp1_feedback_debug(
    *,
    out_root: str,
    feedback_policy_rows: List[Dict[str, object]],
    feedback_bridge_rows: List[Dict[str, object]],
    feedback_feedback_rows: List[Dict[str, object]],
    main_matrix_rows: List[Dict[str, object]],
    unified_final_rows: List[Dict[str, object]],
) -> None:
    policy_df = pd.DataFrame(
        [r for r in feedback_policy_rows if r["dataset"] == "selfregulationscp1" and r["variant"] == "unified_feedback_bridge"]
    )
    bridge_df = pd.DataFrame(
        [r for r in feedback_bridge_rows if r["dataset"] == "selfregulationscp1" and r["variant"] == "unified_feedback_bridge"]
    )
    update_df = pd.DataFrame(
        [r for r in feedback_feedback_rows if r["dataset"] == "selfregulationscp1" and r["variant"] == "unified_feedback_bridge"]
    )
    final_df = pd.DataFrame([r for r in main_matrix_rows if r["dataset"] == "selfregulationscp1"])
    unified_final_df = pd.DataFrame(
        [r for r in unified_final_rows if r["dataset"] == "selfregulationscp1" and r["variant"] == "unified_feedback_bridge"]
    )

    deg_rows: List[Dict[str, object]] = []
    proxy_rows: List[Dict[str, object]] = []
    collapse_rows: List[Dict[str, object]] = []
    overall_zero_step = False
    overall_oscillation = False
    overall_proxy_misalignment = False
    overall_bridge_collapse = False

    for seed in sorted(unified_final_df["seed"].unique().tolist()) if not unified_final_df.empty else []:
        seed_policy = policy_df[policy_df["seed"] == seed].sort_values("round_index")
        seed_update = update_df[update_df["seed"] == seed].sort_values("round_index")
        seed_bridge = bridge_df[bridge_df["seed"] == seed].sort_values("round_index")
        raw_row = final_df[(final_df["seed"] == seed) & (final_df["variant"] == "raw_only")].iloc[0]
        feedback_row = unified_final_df[(unified_final_df["seed"] == seed)].iloc[0]
        raw_test_f1 = float(raw_row["macro_f1"])
        raw_val_f1 = float(feedback_row["raw_val_macro_f1"])
        final_gain = float(feedback_row["test_macro_f1"] - feedback_row["raw_test_macro_f1"])

        last_reward = None
        last_penalty = None
        seed_zero_step = False
        seed_oscillation = False
        seed_proxy_misalignment = False
        seed_bridge_collapse = False
        for _, prow in seed_policy.iterrows():
            rid = int(prow["round_index"])
            urow = seed_update[seed_update["round_index"] == rid].iloc[0]
            brow = seed_bridge[seed_bridge["round_index"] == rid].iloc[0]

            active_ids = json.loads(prow["active_direction_ids"])
            weights = json.loads(prow["direction_weights"])
            steps = json.loads(prow["step_sizes"])
            rewarded_dirs = json.loads(urow["rewarded_directions"])
            penalized_dirs = json.loads(urow["penalized_directions"])
            reward_summary = json.loads(urow["reward_summary"])
            penalty_summary = json.loads(urow["penalty_summary"])
            margin_proxy = json.loads(brow["inter_class_margin_proxy"])
            class_shift = json.loads(brow["classwise_mean_shift"])
            class_dist = json.loads(brow["classwise_covariance_distortion"])
            reward_flip_count = _sign_flip_count(last_reward, reward_summary)
            penalty_flip_count = _sign_flip_count(last_penalty, penalty_summary)
            last_reward = dict(reward_summary)
            last_penalty = dict(penalty_summary)

            mean_step = float(np.mean([float(v) for v in steps.values()])) if steps else 0.0
            median_step = float(np.median([float(v) for v in steps.values()])) if steps else 0.0
            nonzero_survival_ratio = (
                float(np.mean([1.0 if abs(float(v)) > 1e-8 else 0.0 for v in weights.values()])) if weights else 0.0
            )
            near_zero_weight_ratio = (
                float(np.mean([1.0 if abs(float(v)) <= 1e-3 else 0.0 for v in weights.values()])) if weights else 1.0
            )
            zero_step_collapse = bool(mean_step <= 1e-6 or len(active_ids) == 0 or all(abs(float(v)) <= 1e-6 for v in steps.values()))
            oscillation = bool(reward_flip_count >= 2 or penalty_flip_count >= 2)
            penalty_dominated = bool(len(rewarded_dirs) == 0 and len(penalized_dirs) > 0)
            if zero_step_collapse and oscillation:
                degeneration_flag = "mixed"
                degeneration_comment = "step budget collapsed while reward/penalty signals also oscillated"
            elif zero_step_collapse:
                degeneration_flag = "zero_step_collapse"
                degeneration_comment = "step sizes collapsed to near zero or active directions disappeared"
            elif oscillation:
                degeneration_flag = "oscillation"
                degeneration_comment = "reward/penalty signs flipped repeatedly across rounds"
            elif penalty_dominated:
                degeneration_flag = "penalty_dominated"
                degeneration_comment = "policy remained active but updates were dominated by penalties"
            else:
                degeneration_flag = "stable"
                degeneration_comment = "no clear collapse or oscillation in current feedback rounds"

            deg_rows.append(
                {
                    "dataset": "selfregulationscp1",
                    "seed": int(seed),
                    "row_type": "round",
                    "round_index": int(rid),
                    "active_direction_count": int(len(active_ids)),
                    "nonzero_weight_survival_ratio": nonzero_survival_ratio,
                    "mean_step_size": mean_step,
                    "median_step_size": median_step,
                    "near_zero_weight_ratio": near_zero_weight_ratio,
                    "reward_sign_flip_count": int(reward_flip_count),
                    "penalty_sign_flip_count": int(penalty_flip_count),
                    "degeneration_flag": degeneration_flag,
                    "degeneration_comment": degeneration_comment,
                    "zero_step_collapse": _flag_yesno(zero_step_collapse),
                    "oscillation": _flag_yesno(oscillation),
                }
            )
            seed_zero_step = seed_zero_step or zero_step_collapse
            seed_oscillation = seed_oscillation or oscillation

            inner_gain = float(urow["val_macro_f1"] - raw_val_f1)
            gain_gap = float(final_gain - inner_gain)
            proxy_misalignment = bool((inner_gain > 0.0 and gain_gap <= -0.01) or (inner_gain >= 0.01 and final_gain <= 0.0))
            if proxy_misalignment:
                proxy_comment = "inner validation gain overstates final raw-side gain"
            elif gain_gap >= 0.01:
                proxy_comment = "inner validation is conservative relative to final coupling gain"
            else:
                proxy_comment = "no clear proxy misalignment under current SCP1 feedback run"
            proxy_rows.append(
                {
                    "dataset": "selfregulationscp1",
                    "seed": int(seed),
                    "row_type": "round",
                    "round_index": int(rid),
                    "inner_val_gain": inner_gain,
                    "final_coupling_gain": final_gain,
                    "gain_gap": gain_gap,
                    "proxy_misalignment_flag": _flag_yesno(proxy_misalignment),
                    "proxy_comment": proxy_comment,
                }
            )
            seed_proxy_misalignment = seed_proxy_misalignment or proxy_misalignment

            margin_ratio = float(margin_proxy.get("ratio_mean", 0.0))
            classwise_mean_shift_max = max([abs(float(v)) for v in class_shift.values()], default=0.0)
            classwise_distortion_max = max([float(v) for v in class_dist.values()], default=0.0)
            bridge_collapse = bool(
                final_gain < 0.0
                and (
                    margin_ratio < 0.99
                    or classwise_distortion_max >= 3.0
                    or str(brow["task_risk_comment"]) == "bridge_margin_shrink_risk"
                )
            )
            if bridge_collapse:
                collapse_comment = "bridge-side margin shrink or classwise distortion coincides with final gain drop"
            elif margin_ratio < 0.99 or str(brow["task_risk_comment"]) == "bridge_margin_shrink_risk":
                collapse_comment = "margin shrink warning exists, but it does not coincide with a final gain drop"
            else:
                collapse_comment = "no clear bridge collapse signal in current round"
            collapse_rows.append(
                {
                    "dataset": "selfregulationscp1",
                    "seed": int(seed),
                    "row_type": "round",
                    "round_index": int(rid),
                    "inter_class_margin_ratio_vs_raw": margin_ratio,
                    "classwise_mean_shift_max": classwise_mean_shift_max,
                    "classwise_distortion_max": classwise_distortion_max,
                    "final_coupling_gain": final_gain,
                    "bridge_collapse_flag": _flag_yesno(bridge_collapse),
                    "bridge_collapse_comment": collapse_comment,
                }
            )
            seed_bridge_collapse = seed_bridge_collapse or bridge_collapse

        deg_rows.append(
            {
                "dataset": "selfregulationscp1",
                "seed": int(seed),
                "row_type": "summary",
                "round_index": 0,
                "active_direction_count": np.nan,
                "nonzero_weight_survival_ratio": np.nan,
                "mean_step_size": np.nan,
                "median_step_size": np.nan,
                "near_zero_weight_ratio": np.nan,
                "reward_sign_flip_count": np.nan,
                "penalty_sign_flip_count": np.nan,
                "degeneration_flag": "summary",
                "degeneration_comment": "seed-level summary for zero-step collapse / oscillation",
                "zero_step_collapse": _flag_yesno(seed_zero_step),
                "oscillation": _flag_yesno(seed_oscillation),
            }
        )
        proxy_rows.append(
            {
                "dataset": "selfregulationscp1",
                "seed": int(seed),
                "row_type": "summary",
                "round_index": 0,
                "inner_val_gain": np.nan,
                "final_coupling_gain": final_gain,
                "gain_gap": np.nan,
                "proxy_misalignment_flag": _flag_yesno(seed_proxy_misalignment),
                "proxy_comment": "seed-level summary for inner-validation vs final coupling alignment",
            }
        )
        collapse_rows.append(
            {
                "dataset": "selfregulationscp1",
                "seed": int(seed),
                "row_type": "summary",
                "round_index": 0,
                "inter_class_margin_ratio_vs_raw": np.nan,
                "classwise_mean_shift_max": np.nan,
                "classwise_distortion_max": np.nan,
                "final_coupling_gain": final_gain,
                "bridge_collapse_flag": _flag_yesno(seed_bridge_collapse),
                "bridge_collapse_comment": "seed-level summary for margin shrink / classwise distortion vs final gain",
            }
        )
        overall_zero_step = overall_zero_step or seed_zero_step
        overall_oscillation = overall_oscillation or seed_oscillation
        overall_proxy_misalignment = overall_proxy_misalignment or seed_proxy_misalignment
        overall_bridge_collapse = overall_bridge_collapse or seed_bridge_collapse

    pd.DataFrame(deg_rows).to_csv(os.path.join(out_root, "scp1_feedback_policy_degeneration_debug.csv"), index=False)
    pd.DataFrame(proxy_rows).to_csv(os.path.join(out_root, "scp1_feedback_proxy_gap_debug.csv"), index=False)
    pd.DataFrame(collapse_rows).to_csv(os.path.join(out_root, "scp1_feedback_bridge_collapse_debug.csv"), index=False)

    if overall_zero_step or overall_oscillation:
        minimal_label = "feedback_not_ready"
    elif overall_proxy_misalignment or overall_bridge_collapse:
        minimal_label = "feedback_promising_but_unstable"
    else:
        minimal_label = "feedback_ready_for_minimal_tuning"

    conclusion_lines = [
        "# SCP1 Unified Feedback Minimal Diagnosis",
        "",
        "更新时间：2026-03-27",
        "",
        "身份：`minimal diagnostic close-out for unified_feedback on SCP1`",
        "",
        "## Answers",
        "",
        f"1. unified_feedback 是否发生了 policy degeneration：`{_flag_yesno(overall_zero_step or overall_oscillation)}`",
        f"2. unified_feedback 是否存在 proxy misalignment：`{_flag_yesno(overall_proxy_misalignment)}`",
        f"3. unified_feedback 是否存在 bridge collapse：`{_flag_yesno(overall_bridge_collapse)}`",
        "",
        "## Label",
        "",
        f"- `{minimal_label}`",
        "",
        "## Notes",
        "",
        "- `policy degeneration` 只按 zero-step collapse / oscillation 两条硬诊断口径判断。",
        "- `proxy misalignment` 当前只做诊断，不直接引入复合 Reward。",
        "- `bridge collapse` 当前只做软诊断，不启用 rollback 或硬熔断。",
        "",
    ]
    with open(os.path.join(out_root, "scp1_unified_feedback_minimal_diagnosis_conclusion.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(conclusion_lines))


def _summarize_main_matrix(rows: List[Dict[str, object]], out_root: str) -> None:
    df = pd.DataFrame(rows)
    summary_rows: List[Dict[str, object]] = []
    for dataset in sorted(df["dataset"].unique().tolist()):
        ds = df[df["dataset"] == dataset]
        raw_mean_f1 = float(ds[ds["variant"] == "raw_only"]["macro_f1"].mean())
        single_mean_f1 = float(ds[ds["variant"] == "legacy_single_round_bridge"]["macro_f1"].mean())
        legacy_mean_f1 = float(ds[ds["variant"] == "legacy_multiround_bridge"]["macro_f1"].mean())
        for variant in ["raw_only", "legacy_single_round_bridge", "legacy_multiround_bridge", "unified_static_bridge"]:
            vdf = ds[ds["variant"] == variant]
            if vdf.empty:
                continue
            acc_mean, acc_std = _summary_stats(vdf["acc"].tolist())
            f1_mean, f1_std = _summary_stats(vdf["macro_f1"].tolist())
            delta_raw = float(f1_mean - raw_mean_f1)
            delta_legacy = float(f1_mean - legacy_mean_f1)
            label = _variant_label(variant=variant, f1=f1_mean, raw_f1=raw_mean_f1, single_f1=single_mean_f1)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "acc": f"{acc_mean:.4f} +/- {acc_std:.4f}",
                    "macro_f1": f"{f1_mean:.4f} +/- {f1_std:.4f}",
                    "delta_vs_raw": float(delta_raw),
                    "delta_vs_legacy_multiround": float(delta_legacy),
                    "label": label,
                }
            )
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_root, "route_b_main_matrix_summary.csv"), index=False)
    df.to_csv(os.path.join(out_root, "route_b_main_matrix_per_seed.csv"), index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Route B official main matrix + unified calibration runner.")
    p.add_argument("--datasets", type=str, default="har,natops,selfregulationscp1,fingermovements")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/route_b_main_matrix_20260326")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--val-fallback-fraction", type=float, default=0.25)
    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--bridge-eps", type=float, default=1e-4)
    p.add_argument("--window-sec", type=float, default=4.0)
    p.add_argument("--hop-sec", type=float, default=1.0)
    p.add_argument("--prop-win-ratio", type=float, default=0.5)
    p.add_argument("--prop-hop-ratio", type=float, default=0.25)
    p.add_argument("--min-window-len-samples", type=int, default=16)
    p.add_argument("--min-hop-len-samples", type=int, default=8)
    p.add_argument("--nominal-cap-k", type=int, default=120)
    p.add_argument("--cap-sampling-policy", type=str, default="random")
    p.add_argument("--aggregation-mode", type=str, default="majority")
    p.add_argument("--n-kernels", type=int, default=10000)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--memmap-threshold-gb", type=float, default=1.0)
    p.add_argument("--k-dir", type=int, default=5)
    p.add_argument("--subset-size", type=int, default=1)
    p.add_argument("--pia-multiplier", type=int, default=1)
    p.add_argument("--pia-gamma", type=float, default=0.10)
    p.add_argument("--pia-n-iters", type=int, default=2)
    p.add_argument("--pia-activation", type=str, default="sine")
    p.add_argument("--pia-bias-update-mode", type=str, default="residual")
    p.add_argument("--pia-c-repr", type=float, default=1.0)
    p.add_argument("--mech-knn-k", type=int, default=20)
    p.add_argument("--mech-max-aug-for-metrics", type=int, default=2000)
    p.add_argument("--mech-max-real-knn-ref", type=int, default=10000)
    p.add_argument("--mech-max-real-knn-query", type=int, default=1000)
    p.add_argument("--linear-c", type=float, default=1.0)
    p.add_argument("--linear-class-weight", type=str, default="none")
    p.add_argument("--linear-max-iter", type=int, default=1000)
    p.add_argument("--curriculum-rounds", type=int, default=3)
    p.add_argument("--curriculum-init-gamma", type=float, default=0.06)
    p.add_argument("--curriculum-expand-factor", type=float, default=1.25)
    p.add_argument("--curriculum-shrink-factor", type=float, default=0.70)
    p.add_argument("--curriculum-gamma-max", type=float, default=0.16)
    p.add_argument("--curriculum-freeze-eps", type=float, default=0.02)
    p.add_argument("--policy-top-k", type=int, default=3)
    p.add_argument("--stop-patience", type=int, default=2)
    p.add_argument("--feedback-eta-reward", type=float, default=0.75)
    p.add_argument("--feedback-eta-penalty", type=float, default=1.0)
    p.add_argument("--lraes-beta", type=float, default=0.5)
    p.add_argument("--lraes-reg-lambda", type=float, default=1e-4)
    p.add_argument("--lraes-top-k-per-class", type=int, default=3)
    p.add_argument("--lraes-rank-tol", type=float, default=1e-8)
    p.add_argument("--lraes-eig-pos-eps", type=float, default=1e-9)
    p.add_argument("--lraes-knn-k", type=int, default=20)
    p.add_argument("--lraes-boundary-quantile", type=float, default=0.30)
    p.add_argument("--lraes-interior-quantile", type=float, default=0.70)
    p.add_argument("--lraes-hetero-k", type=int, default=3)
    p.add_argument("--har-root", type=str, default="")
    p.add_argument("--mitbih-npz", type=str, default="")
    p.add_argument("--natops-root", type=str, default="")
    p.add_argument("--fingermovements-root", type=str, default="")
    p.add_argument("--selfregulationscp1-root", type=str, default="")
    p.add_argument("--basicmotions-root", type=str, default="")
    p.add_argument("--handmovementdirection-root", type=str, default="")
    p.add_argument("--uwavegesturelibrary-root", type=str, default="")
    p.add_argument("--epilepsy-root", type=str, default="")
    p.add_argument("--atrialfibrillation-root", type=str, default="")
    p.add_argument("--pendigits-root", type=str, default="")
    p.add_argument("--processed-root", type=str, default="data/SEED/SEED_EEG")
    p.add_argument("--stim-xlsx", type=str, default="data/SEED/SEED_EEG/SEED_stimulation.xlsx")
    p.add_argument("--seediv-root", type=str, default="")
    p.add_argument("--seedv-root", type=str, default="")
    args = p.parse_args()

    global args_feedback_eta_reward, args_feedback_eta_penalty
    args_feedback_eta_reward = float(args.feedback_eta_reward)
    args_feedback_eta_penalty = float(args.feedback_eta_penalty)

    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_seed_list(args.seeds)
    _ensure_dir(args.out_root)

    main_rows: List[Dict[str, object]] = []
    legacy_debug_rows: List[Dict[str, object]] = []
    unified_policy_rows: List[Dict[str, object]] = []
    unified_bridge_rows: List[Dict[str, object]] = []
    unified_feedback_rows: List[Dict[str, object]] = []
    unified_final_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        dataset_dir = os.path.join(args.out_root, dataset)
        _ensure_dir(dataset_dir)
        for seed in seeds:
            print(f"[route-b-main][{dataset}][seed={seed}] legacy_start", flush=True)
            legacy = _run_legacy_official(
                dataset=dataset,
                seed=int(seed),
                args=args,
                out_dir=os.path.join(dataset_dir, f"seed{seed}", "legacy"),
            )
            raw_f1 = float(legacy.raw_metrics["macro_f1"])
            single_f1 = float(legacy.single_metrics["macro_f1"])
            legacy_f1 = float(legacy.multiround_metrics["macro_f1"])
            main_rows.extend(
                [
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "variant": "raw_only",
                        "best_round": 0,
                        "acc": float(legacy.raw_metrics["acc"]),
                        "macro_f1": raw_f1,
                        "delta_vs_raw": 0.0,
                        "delta_vs_legacy_multiround": float(raw_f1 - legacy_f1),
                        "label": _variant_label(variant="raw_only", f1=raw_f1, raw_f1=raw_f1, single_f1=single_f1),
                    },
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "variant": "legacy_single_round_bridge",
                        "best_round": 1,
                        "acc": float(legacy.single_metrics["acc"]),
                        "macro_f1": single_f1,
                        "delta_vs_raw": float(single_f1 - raw_f1),
                        "delta_vs_legacy_multiround": float(single_f1 - legacy_f1),
                        "label": _variant_label(
                            variant="legacy_single_round_bridge",
                            f1=single_f1,
                            raw_f1=raw_f1,
                            single_f1=single_f1,
                        ),
                    },
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "variant": "legacy_multiround_bridge",
                        "best_round": int(legacy.best_round_id),
                        "acc": float(legacy.multiround_metrics["acc"]),
                        "macro_f1": legacy_f1,
                        "delta_vs_raw": float(legacy_f1 - raw_f1),
                        "delta_vs_legacy_multiround": 0.0,
                        "label": _variant_label(
                            variant="legacy_multiround_bridge",
                            f1=legacy_f1,
                            raw_f1=raw_f1,
                            single_f1=single_f1,
                        ),
                    },
                ]
            )
            for row in legacy.selection_round_rows:
                direction_probs = {k: float(v) for k, v in row["direction_probs"].items()}
                selected_proxy = [int(k) for k, v in sorted(direction_probs.items(), key=lambda kv: int(kv[0])) if float(v) > 0.0]
                legacy_debug_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "round_index": int(row["round_id"]),
                        "direction_ranking_proxy": _json_text(
                            _legacy_ranking_proxy(row.get("direction_score", {}), row.get("gamma_before", {}))
                        ),
                        "selected_directions_proxy": _json_text(selected_proxy),
                        "top_k_proxy": int(len(selected_proxy)),
                        "keep_prune_proxy": _json_text(row.get("direction_state", {})),
                        "step_size_summary": _json_text(row.get("gamma_before", {})),
                        "stop_reason_proxy": "active_budget_available",
                        "z_val_trial_macro_f1": float(row["z_trial_macro_f1"]) if np.isfinite(row["z_trial_macro_f1"]) else np.nan,
                    }
                )

            print(f"[route-b-main][{dataset}][seed={seed}] unified_static_start", flush=True)
            unified_static = _run_unified_official(
                dataset=dataset,
                seed=int(seed),
                args=args,
                out_dir=os.path.join(dataset_dir, f"seed{seed}", "unified_static"),
                variant="unified_static",
            )
            unified_static_f1 = float(unified_static.final_row["test_macro_f1"])
            main_rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "variant": "unified_static_bridge",
                    "best_round": int(unified_static.final_row["best_round"]),
                    "acc": float(unified_static.final_row["test_acc"]),
                    "macro_f1": unified_static_f1,
                    "delta_vs_raw": float(unified_static_f1 - raw_f1),
                    "delta_vs_legacy_multiround": float(unified_static_f1 - legacy_f1),
                    "label": _variant_label(
                        variant="unified_static_bridge",
                        f1=unified_static_f1,
                        raw_f1=raw_f1,
                        single_f1=single_f1,
                    ),
                }
            )
            for row in unified_static.policy_rows:
                unified_policy_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "variant": "unified_static_bridge",
                        **{k: v for k, v in row.items() if k not in {"dataset", "seed", "variant"}},
                        "keep_prune_proxy": row["reward_penalty_summary"],
                    }
                )
            unified_bridge_rows.extend(
                [
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "variant": "unified_static_bridge",
                        **{k: v for k, v in row.items() if k not in {"dataset", "seed", "variant"}},
                    }
                    for row in unified_static.bridge_rows
                ]
            )
            unified_feedback_rows.extend(
                [
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "variant": "unified_static_bridge",
                        **{k: v for k, v in row.items() if k not in {"dataset", "seed", "variant"}},
                    }
                    for row in unified_static.feedback_rows
                ]
            )
            unified_final_rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "variant": "unified_static_bridge",
                    **{k: v for k, v in unified_static.final_row.items() if k not in {"dataset", "seed", "variant"}},
                    "raw_val_acc": float(unified_static.raw_val["acc"]),
                    "raw_val_macro_f1": float(unified_static.raw_val["macro_f1"]),
                    "raw_test_acc": float(unified_static.raw_test["acc"]),
                    "raw_test_macro_f1": float(unified_static.raw_test["macro_f1"]),
                }
            )

            if dataset == "selfregulationscp1":
                print(f"[route-b-main][{dataset}][seed={seed}] unified_feedback_start", flush=True)
                unified_feedback = _run_unified_official(
                    dataset=dataset,
                    seed=int(seed),
                    args=args,
                    out_dir=os.path.join(dataset_dir, f"seed{seed}", "unified_feedback"),
                    variant="unified_feedback",
                )
                for row in unified_feedback.policy_rows:
                    unified_policy_rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "variant": "unified_feedback_bridge",
                            **{k: v for k, v in row.items() if k not in {"dataset", "seed", "variant"}},
                            "keep_prune_proxy": row["reward_penalty_summary"],
                        }
                    )
                unified_bridge_rows.extend(
                    [
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "variant": "unified_feedback_bridge",
                            **{k: v for k, v in row.items() if k not in {"dataset", "seed", "variant"}},
                        }
                        for row in unified_feedback.bridge_rows
                    ]
                )
                unified_feedback_rows.extend(
                    [
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "variant": "unified_feedback_bridge",
                            **{k: v for k, v in row.items() if k not in {"dataset", "seed", "variant"}},
                        }
                        for row in unified_feedback.feedback_rows
                    ]
                )
                unified_final_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "variant": "unified_feedback_bridge",
                        **{k: v for k, v in unified_feedback.final_row.items() if k not in {"dataset", "seed", "variant"}},
                        "raw_val_acc": float(unified_feedback.raw_val["acc"]),
                        "raw_val_macro_f1": float(unified_feedback.raw_val["macro_f1"]),
                        "raw_test_acc": float(unified_feedback.raw_test["acc"]),
                        "raw_test_macro_f1": float(unified_feedback.raw_test["macro_f1"]),
                    }
                )

    _summarize_main_matrix(main_rows, args.out_root)
    pd.DataFrame(legacy_debug_rows).to_csv(os.path.join(args.out_root, "legacy_round_debug.csv"), index=False)
    pd.DataFrame(unified_policy_rows).to_csv(os.path.join(args.out_root, "unified_policy_debug.csv"), index=False)
    pd.DataFrame(unified_bridge_rows).to_csv(os.path.join(args.out_root, "unified_bridge_debug.csv"), index=False)
    pd.DataFrame(unified_feedback_rows).to_csv(os.path.join(args.out_root, "unified_feedback_debug.csv"), index=False)
    pd.DataFrame(unified_final_rows).to_csv(os.path.join(args.out_root, "unified_final_debug.csv"), index=False)
    _build_natops_alignment(
        out_root=args.out_root,
        legacy_debug_rows=legacy_debug_rows,
        unified_policy_rows=unified_policy_rows,
        main_matrix_rows=main_rows,
    )
    _build_har_stability_check(out_root=args.out_root, main_matrix_rows=main_rows)
    _build_scp1_feedback_debug(
        out_root=args.out_root,
        feedback_policy_rows=unified_policy_rows,
        feedback_bridge_rows=unified_bridge_rows,
        feedback_feedback_rows=unified_feedback_rows,
        main_matrix_rows=main_rows,
        unified_final_rows=unified_final_rows,
    )


if __name__ == "__main__":
    main()
