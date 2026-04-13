#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "scripts"))

from route_b_unified import (  # noqa: E402
    BridgeConfig,
    BridgeResult,
    MiniRocketEvalConfig,
    PolicyAction,
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
from route_b_unified.types import EvaluatorPosterior, TargetRoundState  # noqa: E402
from scripts.legacy_phase.run_phase15_multiround_curriculum_probe import (  # noqa: E402
    _build_curriculum_aug_candidates,
    _compute_direction_intrusion,
    _mech_dir_maps,
)
from scripts.legacy_phase.run_phase15_step1b_multidir_matrix import _compute_mech_metrics  # noqa: E402


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


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _json_text(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _format_mean_std(values: Iterable[float]) -> str:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return "0.0000 +/- 0.0000"
    return f"{float(np.mean(arr)):.4f} +/- {float(np.std(arr)):.4f}"


def _result_label(delta_vs_legacy: float, delta_vs_static: float) -> str:
    if delta_vs_legacy > 1e-6 and delta_vs_static > 1e-6:
        return "positive"
    if delta_vs_legacy >= -1e-6 and delta_vs_static >= -1e-6:
        return "flat"
    return "negative"


def _comment_from_sources(*, delta_vs_legacy: float, bridge_task_risk: str, policy_variant: str) -> str:
    if delta_vs_legacy > 1e-6 and "margin" not in str(bridge_task_risk):
        return f"{policy_variant}_gain_more_likely_from_policy_plus_bridge_stability"
    if "margin" in str(bridge_task_risk):
        return f"{policy_variant}_limited_by_bridge_task_margin"
    return f"{policy_variant}_gain_more_likely_from_target_side_only"


def _target_health_comment(worst_dir_summary: str, entropy: float, variant: str) -> str:
    if variant == "legacy_multiround":
        return "legacy_uniform_direction_policy"
    if entropy < 0.3:
        return f"{variant}_high_direction_concentration"
    if "margin=-" in str(worst_dir_summary):
        return f"{variant}_mixed_dir_quality"
    return f"{variant}_direction_profile_stable"


def _run_policy_variant(
    *,
    dataset: str,
    seed: int,
    rep_state,
    out_root: str,
    eval_cfg: MiniRocketEvalConfig,
    policy_cfg: UnifiedPolicyConfig,
    bridge_cfg: BridgeConfig,
    linear_c: float,
    linear_class_weight: str,
    linear_max_iter: int,
    mech_knn_k: int,
    mech_max_aug: int,
    mech_max_real_ref: int,
    mech_max_real_query: int,
    pia_multiplier: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    state = init_policy(rep_state, policy_cfg)
    round_policy_rows: List[Dict[str, object]] = []
    round_bridge_rows: List[Dict[str, object]] = []
    round_feedback_rows: List[Dict[str, object]] = []
    best_val_f1 = float("-inf")
    best_bundle: Dict[str, object] | None = None
    prev_posterior: EvaluatorPosterior | None = None
    variant_seed_offset = {
        "legacy_multiround": 11,
        "unified_static": 23,
        "unified_feedback": 37,
    }.get(str(policy_cfg.variant), 53)

    for round_idx in range(1, int(policy_cfg.n_rounds) + 1):
        action, state = policy_step(rep_state, state, prev_posterior, policy_cfg)
        round_policy_rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "variant": str(policy_cfg.variant),
                "round_index": int(round_idx),
                "selected_directions": _json_text(action.selected_dir_ids),
                "direction_weights": _json_text(action.direction_weights),
                "step_sizes": _json_text(action.step_sizes),
                "entropy": float(action.entropy),
                "stop_reason": str(action.stop_reason),
                "reward_penalty_summary": _json_text({}),
            }
        )
        if action.stop_flag:
            if action.stop_flag:
                break
        X_aug, y_aug, tid_aug, src_aug, dir_aug, aug_meta = _build_curriculum_aug_candidates(
            X_train=rep_state.X_train,
            y_train=rep_state.y_train,
            tid_train=rep_state.tid_train,
            direction_bank=state.direction_bank,
            direction_probs=action.direction_probs_vector,
            gamma_by_dir=action.gamma_vector,
            multiplier=int(pia_multiplier),
            seed=int(seed + 700000 + round_idx * 1009 + variant_seed_offset),
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
            linear_c=float(linear_c),
            class_weight=str(linear_class_weight),
            linear_max_iter=int(linear_max_iter),
            knn_k=int(mech_knn_k),
            max_aug_for_mech=int(mech_max_aug),
            max_real_knn_ref=int(mech_max_real_ref),
            max_real_knn_query=int(mech_max_real_query),
            progress_prefix=f"[route-b-unified][{dataset}][seed={seed}][{policy_cfg.variant}][round={round_idx}][mech]",
        )
        intrusion_by_dir = _compute_direction_intrusion(
            X_anchor=rep_state.X_train,
            y_anchor=rep_state.y_train,
            X_aug_accepted=X_aug,
            y_aug_accepted=y_aug,
            dir_accepted=dir_aug,
            seed=int(seed),
            knn_k=int(mech_knn_k),
            max_eval=int(mech_max_aug),
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
        bridge_result = apply_bridge(rep_state, target_state, bridge_cfg, variant=str(policy_cfg.variant))
        round_gain = 0.0 if not np.isfinite(best_val_f1) else 0.0
        posterior_val = evaluate_bridge(
            bridge_result,
            eval_cfg,
            split_name="val",
            target_state=target_state,
            round_gain_proxy=round_gain,
        )
        posterior_val.round_gain_proxy = (
            0.0 if not np.isfinite(best_val_f1) else float(posterior_val.macro_f1) - float(best_val_f1)
        )
        round_bridge_rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "variant": str(policy_cfg.variant),
                "round_index": int(round_idx),
                "bridge_cov_match_error": float(bridge_result.global_fidelity["bridge_cov_match_error_mean"]),
                "bridge_cov_to_orig_distance": float(bridge_result.global_fidelity["bridge_cov_to_orig_distance_logeuc_mean"]),
                "energy_ratio": float(bridge_result.global_fidelity["energy_ratio_mean"]),
                "cond_A": float(bridge_result.global_fidelity["cond_A_mean"]),
                "raw_mean_shift_abs": float(bridge_result.global_fidelity["raw_mean_shift_abs_mean"]),
                "classwise_mean_shift": _json_text(bridge_result.classwise_fidelity["classwise_mean_shift_summary"]),
                "classwise_covariance_distortion": _json_text(
                    bridge_result.classwise_fidelity["classwise_covariance_distortion_summary"]
                ),
                "inter_class_margin_proxy": _json_text(bridge_result.margin_proxy),
                "task_risk_comment": str(bridge_result.task_risk_comment),
            }
        )
        if bool(policy_cfg.feedback_enabled):
            state, feedback_summary = update_policy(state, posterior_val, policy_cfg)
            round_feedback_rows.append(
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
                }
            )
            round_policy_rows[-1]["reward_penalty_summary"] = _json_text(
                {
                    "rewarded": feedback_summary.rewarded_dirs,
                    "penalized": feedback_summary.penalized_dirs,
                }
            )
        else:
            round_feedback_rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "variant": str(policy_cfg.variant),
                    "round_index": int(round_idx),
                    "rewarded_directions": "[]",
                    "penalized_directions": "[]",
                    "rank_change_summary": "{}",
                    "collapse_warning": "none",
                    "overspread_warning": "none",
                    "reward_summary": "{}",
                    "penalty_summary": "{}",
                }
            )

        if float(posterior_val.macro_f1) > float(best_val_f1):
            best_val_f1 = float(posterior_val.macro_f1)
            best_bundle = {
                "target_state": target_state,
                "bridge_result": bridge_result,
                "posterior_val": posterior_val,
            }
        prev_posterior = posterior_val
        if state.stop_flag:
            round_policy_rows[-1]["stop_reason"] = str(state.stop_reason)
            break
    if best_bundle is None:
        raise RuntimeError(f"No valid round produced for dataset={dataset}, seed={seed}, variant={policy_cfg.variant}")
    posterior_test = evaluate_bridge(
        best_bundle["bridge_result"],
        eval_cfg,
        split_name="test",
        target_state=best_bundle["target_state"],
        round_gain_proxy=0.0,
    )
    best_target = best_bundle["target_state"]
    final_row = {
        "dataset": dataset,
        "seed": int(seed),
        "variant": str(policy_cfg.variant),
        "best_round": int(best_target.round_index),
        "val_acc": float(best_bundle["posterior_val"].acc),
        "val_macro_f1": float(best_bundle["posterior_val"].macro_f1),
        "test_acc": float(posterior_test.acc),
        "test_macro_f1": float(posterior_test.macro_f1),
        "direction_usage_entropy": float(best_target.action.entropy),
        "worst_dir_summary": str(best_bundle["posterior_val"].worst_dir_summary),
        "bridge_task_risk_comment": str(best_bundle["posterior_val"].task_risk_comment),
        "target_health_comment": _target_health_comment(
            str(best_bundle["posterior_val"].worst_dir_summary),
            float(best_target.action.entropy),
            str(policy_cfg.variant),
        ),
        "bridge_cov_match_error": float(best_bundle["bridge_result"].global_fidelity["bridge_cov_match_error_mean"]),
        "bridge_cov_to_orig_distance": float(
            best_bundle["bridge_result"].global_fidelity["bridge_cov_to_orig_distance_logeuc_mean"]
        ),
        "energy_ratio": float(best_bundle["bridge_result"].global_fidelity["energy_ratio_mean"]),
        "cond_A": float(best_bundle["bridge_result"].global_fidelity["cond_A_mean"]),
        "raw_mean_shift_abs": float(best_bundle["bridge_result"].global_fidelity["raw_mean_shift_abs_mean"]),
        "solver_state_summary": _json_text(best_bundle["target_state"].direction_state),
    }
    return final_row, round_policy_rows, round_bridge_rows, round_feedback_rows


def main() -> None:
    p = argparse.ArgumentParser(description="Route B unified policy-driven probe.")
    p.add_argument("--datasets", type=str, default="selfregulationscp1,natops")
    p.add_argument("--seeds", type=str, default="1")
    p.add_argument("--out-root", type=str, default="out/route_b_unified_probe_20260322")
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--bridge-eps", type=float, default=1e-4)
    p.add_argument("--k-dir", type=int, default=5)
    p.add_argument("--policy-top-k", type=int, default=3)
    p.add_argument("--curriculum-rounds", type=int, default=3)
    p.add_argument("--curriculum-init-gamma", type=float, default=0.06)
    p.add_argument("--curriculum-expand-factor", type=float, default=1.25)
    p.add_argument("--curriculum-shrink-factor", type=float, default=0.70)
    p.add_argument("--curriculum-gamma-max", type=float, default=0.16)
    p.add_argument("--curriculum-freeze-eps", type=float, default=0.02)
    p.add_argument("--feedback-eta-reward", type=float, default=0.75)
    p.add_argument("--feedback-eta-penalty", type=float, default=1.0)
    p.add_argument("--stop-patience", type=int, default=2)
    p.add_argument("--lraes-beta", type=float, default=0.5)
    p.add_argument("--lraes-reg-lambda", type=float, default=1e-4)
    p.add_argument("--lraes-top-k-per-class", type=int, default=3)
    p.add_argument("--lraes-rank-tol", type=float, default=1e-8)
    p.add_argument("--lraes-eig-pos-eps", type=float, default=1e-9)
    p.add_argument("--lraes-knn-k", type=int, default=20)
    p.add_argument("--lraes-boundary-quantile", type=float, default=0.30)
    p.add_argument("--lraes-interior-quantile", type=float, default=0.70)
    p.add_argument("--lraes-hetero-k", type=int, default=3)
    p.add_argument("--nominal-cap-k", type=int, default=120)
    p.add_argument("--cap-sampling-policy", type=str, default="random")
    p.add_argument("--aggregation-mode", type=str, default="majority")
    p.add_argument("--n-kernels", type=int, default=10000)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--memmap-threshold-gb", type=float, default=1.0)
    p.add_argument("--window-sec", type=float, default=4.0)
    p.add_argument("--hop-sec", type=float, default=1.0)
    p.add_argument("--prop-win-ratio", type=float, default=0.5)
    p.add_argument("--prop-hop-ratio", type=float, default=0.25)
    p.add_argument("--min-window-len-samples", type=int, default=16)
    p.add_argument("--min-hop-len-samples", type=int, default=8)
    p.add_argument("--linear-c", type=float, default=1.0)
    p.add_argument("--linear-class-weight", type=str, default="none")
    p.add_argument("--linear-max-iter", type=int, default=1000)
    p.add_argument("--mech-knn-k", type=int, default=20)
    p.add_argument("--mech-max-aug-for-metrics", type=int, default=2000)
    p.add_argument("--mech-max-real-knn-ref", type=int, default=10000)
    p.add_argument("--mech-max-real-knn-query", type=int, default=1000)
    p.add_argument("--pia-multiplier", type=int, default=1)
    args = p.parse_args()

    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_seed_list(args.seeds)
    _ensure_dir(args.out_root)

    eval_cfg = MiniRocketEvalConfig(
        out_root=str(args.out_root),
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
    bridge_cfg = BridgeConfig(eps=float(args.bridge_eps))

    all_policy_rows: List[Dict[str, object]] = []
    all_bridge_rows: List[Dict[str, object]] = []
    all_feedback_rows: List[Dict[str, object]] = []
    final_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        for seed in seeds:
            print(f"[route-b-unified][{dataset}][seed={seed}] representation_start", flush=True)
            rep_state = build_representation(
                RepresentationConfig(
                    dataset=str(dataset),
                    seed=int(seed),
                    val_fraction=float(args.val_fraction),
                    spd_eps=float(args.spd_eps),
                )
            )
            raw_only_val = evaluate_bridge(
                BridgeResult(
                    dataset=str(dataset),
                    seed=int(seed),
                    variant="raw_only",
                    round_index=0,
                    train_trials=list(rep_state.train_trial_dicts),
                    val_trials=list(rep_state.val_trial_dicts),
                    test_trials=list(rep_state.test_trial_dicts),
                    global_fidelity={
                        "bridge_cov_match_error_mean": 0.0,
                        "bridge_cov_to_orig_distance_logeuc_mean": 0.0,
                        "energy_ratio_mean": 1.0,
                        "cond_A_mean": 1.0,
                        "raw_mean_shift_abs_mean": 0.0,
                    },
                    classwise_fidelity={
                        "classwise_mean_shift_summary": {},
                        "classwise_covariance_distortion_summary": {},
                        "classwise_covariance_distortion_mean": 0.0,
                    },
                    margin_proxy={"delta_mean": 0.0},
                    task_risk_comment="raw_reference",
                ),
                eval_cfg,
                split_name="val",
                target_state=TargetRoundState(
                    round_index=0,
                    z_aug=np.zeros((0, rep_state.X_train.shape[1]), dtype=np.float32),
                    y_aug=np.zeros((0,), dtype=np.int64),
                    tid_aug=np.asarray([], dtype=object),
                    mech={"dir_profile": {}},
                    dir_maps={"margin_drop_median": {}, "flip_rate": {}, "intrusion": {}},
                    aug_meta={},
                    action=PolicyAction(
                        round_index=0,
                        selected_dir_ids=[],
                        direction_weights={},
                        step_sizes={},
                        direction_probs_vector=np.zeros((0,), dtype=np.float64),
                        gamma_vector=np.zeros((0,), dtype=np.float64),
                        entropy=0.0,
                        stop_flag=False,
                        stop_reason="raw_reference",
                    ),
                    direction_state={},
                    direction_budget_score={},
                ),
                round_gain_proxy=0.0,
            )
            raw_only_test = evaluate_bridge(
                BridgeResult(
                    dataset=str(dataset),
                    seed=int(seed),
                    variant="raw_only",
                    round_index=0,
                    train_trials=list(rep_state.train_trial_dicts),
                    val_trials=list(rep_state.val_trial_dicts),
                    test_trials=list(rep_state.test_trial_dicts),
                    global_fidelity={
                        "bridge_cov_match_error_mean": 0.0,
                        "bridge_cov_to_orig_distance_logeuc_mean": 0.0,
                        "energy_ratio_mean": 1.0,
                        "cond_A_mean": 1.0,
                        "raw_mean_shift_abs_mean": 0.0,
                    },
                    classwise_fidelity={
                        "classwise_mean_shift_summary": {},
                        "classwise_covariance_distortion_summary": {},
                        "classwise_covariance_distortion_mean": 0.0,
                    },
                    margin_proxy={"delta_mean": 0.0},
                    task_risk_comment="raw_reference",
                ),
                eval_cfg,
                split_name="test",
                target_state=TargetRoundState(
                    round_index=0,
                    z_aug=np.zeros((0, rep_state.X_train.shape[1]), dtype=np.float32),
                    y_aug=np.zeros((0,), dtype=np.int64),
                    tid_aug=np.asarray([], dtype=object),
                    mech={"dir_profile": {}},
                    dir_maps={"margin_drop_median": {}, "flip_rate": {}, "intrusion": {}},
                    aug_meta={},
                    action=PolicyAction(
                        round_index=0,
                        selected_dir_ids=[],
                        direction_weights={},
                        step_sizes={},
                        direction_probs_vector=np.zeros((0,), dtype=np.float64),
                        gamma_vector=np.zeros((0,), dtype=np.float64),
                        entropy=0.0,
                        stop_flag=False,
                        stop_reason="raw_reference",
                    ),
                    direction_state={},
                    direction_budget_score={},
                ),
                round_gain_proxy=0.0,
            )
            final_rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "variant": "raw_only",
                    "best_round": 0,
                    "val_acc": float(raw_only_val.acc),
                    "val_macro_f1": float(raw_only_val.macro_f1),
                    "test_acc": float(raw_only_test.acc),
                    "test_macro_f1": float(raw_only_test.macro_f1),
                    "delta_vs_raw_only": 0.0,
                    "delta_vs_legacy_multiround": 0.0,
                    "delta_vs_static_unified": 0.0,
                    "bridge_task_risk_comment": "raw_reference",
                    "improvement_source_comment": "raw_reference",
                }
            )

            variant_cfgs = [
                UnifiedPolicyConfig(
                    variant="legacy_multiround",
                    n_rounds=int(args.curriculum_rounds),
                    select_top_k=int(args.policy_top_k),
                    k_dir=int(args.k_dir),
                    gamma_init=float(args.curriculum_init_gamma),
                    expand_factor=float(args.curriculum_expand_factor),
                    shrink_factor=float(args.curriculum_shrink_factor),
                    gamma_max=float(args.curriculum_gamma_max),
                    freeze_eps=float(args.curriculum_freeze_eps),
                    stop_patience=int(args.stop_patience),
                    feedback_enabled=False,
                    beta=float(args.lraes_beta),
                    reg_lambda=float(args.lraes_reg_lambda),
                    top_k_per_class=int(args.lraes_top_k_per_class),
                    rank_tol=float(args.lraes_rank_tol),
                    eig_pos_eps=float(args.lraes_eig_pos_eps),
                    lraes_knn_k=int(args.lraes_knn_k),
                    lraes_boundary_quantile=float(args.lraes_boundary_quantile),
                    lraes_interior_quantile=float(args.lraes_interior_quantile),
                    lraes_hetero_k=int(args.lraes_hetero_k),
                ),
                UnifiedPolicyConfig(
                    variant="unified_static",
                    n_rounds=int(args.curriculum_rounds),
                    select_top_k=int(args.policy_top_k),
                    k_dir=int(args.k_dir),
                    gamma_init=float(args.curriculum_init_gamma),
                    expand_factor=float(args.curriculum_expand_factor),
                    shrink_factor=float(args.curriculum_shrink_factor),
                    gamma_max=float(args.curriculum_gamma_max),
                    freeze_eps=float(args.curriculum_freeze_eps),
                    stop_patience=int(args.stop_patience),
                    feedback_enabled=False,
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
                ),
                UnifiedPolicyConfig(
                    variant="unified_feedback",
                    n_rounds=int(args.curriculum_rounds),
                    select_top_k=int(args.policy_top_k),
                    k_dir=int(args.k_dir),
                    gamma_init=float(args.curriculum_init_gamma),
                    expand_factor=float(args.curriculum_expand_factor),
                    shrink_factor=float(args.curriculum_shrink_factor),
                    gamma_max=float(args.curriculum_gamma_max),
                    freeze_eps=float(args.curriculum_freeze_eps),
                    stop_patience=int(args.stop_patience),
                    feedback_enabled=True,
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
                ),
            ]

            seed_variant_rows = []
            for policy_cfg in variant_cfgs:
                print(f"[route-b-unified][{dataset}][seed={seed}][{policy_cfg.variant}] start", flush=True)
                final_row, policy_rows, bridge_rows, feedback_rows = _run_policy_variant(
                    dataset=dataset,
                    seed=int(seed),
                    rep_state=rep_state,
                    out_root=str(args.out_root),
                    eval_cfg=eval_cfg,
                    policy_cfg=policy_cfg,
                    bridge_cfg=bridge_cfg,
                    linear_c=float(args.linear_c),
                    linear_class_weight=str(args.linear_class_weight),
                    linear_max_iter=int(args.linear_max_iter),
                    mech_knn_k=int(args.mech_knn_k),
                    mech_max_aug=int(args.mech_max_aug_for_metrics),
                    mech_max_real_ref=int(args.mech_max_real_knn_ref),
                    mech_max_real_query=int(args.mech_max_real_knn_query),
                    pia_multiplier=int(args.pia_multiplier),
                )
                all_policy_rows.extend(policy_rows)
                all_bridge_rows.extend(bridge_rows)
                all_feedback_rows.extend(feedback_rows)
                seed_variant_rows.append(final_row)

            legacy_row = next(r for r in seed_variant_rows if r["variant"] == "legacy_multiround")
            static_row = next(r for r in seed_variant_rows if r["variant"] == "unified_static")
            for row in seed_variant_rows:
                row["delta_vs_raw_only"] = float(row["test_macro_f1"]) - float(raw_only_test.macro_f1)
                row["delta_vs_legacy_multiround"] = float(row["test_macro_f1"]) - float(legacy_row["test_macro_f1"])
                row["delta_vs_static_unified"] = float(row["test_macro_f1"]) - float(static_row["test_macro_f1"])
                row["result_label"] = _result_label(
                    float(row["delta_vs_legacy_multiround"]),
                    float(row["delta_vs_static_unified"]),
                )
                row["improvement_source_comment"] = _comment_from_sources(
                    delta_vs_legacy=float(row["delta_vs_legacy_multiround"]),
                    bridge_task_risk=str(row["bridge_task_risk_comment"]),
                    policy_variant=str(row["variant"]),
                )
            final_rows.extend(seed_variant_rows)

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(os.path.join(args.out_root, "final_coupling_per_seed.csv"), index=False)
    pd.DataFrame(all_policy_rows).to_csv(os.path.join(args.out_root, "policy_summary.csv"), index=False)
    pd.DataFrame(all_bridge_rows).to_csv(os.path.join(args.out_root, "classwise_bridge_summary.csv"), index=False)
    pd.DataFrame(all_feedback_rows).to_csv(os.path.join(args.out_root, "feedback_update_summary.csv"), index=False)

    summary_rows: List[Dict[str, object]] = []
    for (dataset, variant), g in final_df.groupby(["dataset", "variant"], sort=True):
        summary_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "val_macro_f1": _format_mean_std(g["val_macro_f1"].tolist()),
                "test_macro_f1": _format_mean_std(g["test_macro_f1"].tolist()),
                "delta_vs_raw_only": _format_mean_std(g["delta_vs_raw_only"].tolist()),
                "delta_vs_legacy_multiround": _format_mean_std(g["delta_vs_legacy_multiround"].tolist()),
                "delta_vs_static_unified": _format_mean_std(g["delta_vs_static_unified"].tolist()),
                "result_label": str(g["result_label"].iloc[0]) if len(g) else "n/a",
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.out_root, "final_coupling_summary.csv"), index=False)

    conclusion_lines = [
        "# Route B Unified Probe",
        "",
        "身份：`new namespace unified policy-driven main chain prototype`",
        "",
        "- `not for historical freeze table`",
        "- `feedback source = inner validation only`",
        "- `legacy multiround kept as reference baseline`",
        "",
        "## Datasets",
        "",
    ]
    for ds in datasets:
        conclusion_lines.append(f"- `{ds}`")
    conclusion_lines.extend(["", "## Final Summary", ""])
    for row in summary_rows:
        conclusion_lines.append(
            f"- `{row['dataset']} / {row['variant']}`: "
            f"test_f1={row['test_macro_f1']}, delta_vs_legacy={row['delta_vs_legacy_multiround']}"
        )
    conclusion_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- 当前 unified runner 同时输出 `policy_summary / classwise_bridge_summary / feedback_update_summary / final_coupling_summary`。",
            "- `unified_static` 只合并了 LRAES + curriculum，不接 evaluator posterior 更新。",
            "- `unified_feedback` 才会使用 inner-validation posterior 更新 direction ranking / weights / budgets。",
        ]
    )
    with open(os.path.join(args.out_root, "route_b_unified_conclusion.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(conclusion_lines) + "\n")


if __name__ == "__main__":
    main()
