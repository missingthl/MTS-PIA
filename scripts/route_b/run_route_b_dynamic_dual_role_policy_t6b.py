#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from route_b_unified.trajectory_classifier import TrajectoryModelConfig  # noqa: E402
from route_b_unified.trajectory_dual_role_policy import (  # noqa: E402
    TrajectoryDualRolePolicyResult,
    build_dual_role_augmented_trajectories,
)
from route_b_unified.trajectory_feedback_pool import (  # noqa: E402
    TrajectoryFeedbackPoolConfig,
    build_trajectory_feedback_pool,
)
from route_b_unified.trajectory_feedback_pool_windows import (  # noqa: E402
    TrajectoryWindowFeedbackPoolConfig,
    build_window_feedback_pool,
    build_window_feedback_reference_stats,
)
from route_b_unified.trajectory_feedback_rebasis import fit_trajectory_feedback_rebasis  # noqa: E402
from route_b_unified.trajectory_pia_evaluator import (  # noqa: E402
    TrajectoryPIAEvalConfig,
    TrajectoryPIAEvalResult,
    compute_trajectory_diagnostics,
    evaluate_trajectory_pia_t2a,
    evaluate_trajectory_train_final,
)
from route_b_unified.trajectory_pia_operator import TrajectoryPIAOperator, TrajectoryPIAOperatorConfig  # noqa: E402
from route_b_unified.trajectory_representation import (  # noqa: E402
    TrajectoryRepresentationConfig,
    build_trajectory_representation,
)
from route_b_unified.trajectory_unified_window_policy import (  # noqa: E402
    TrajectoryUnifiedWindowPolicyResult,
    build_unified_window_augmented_trajectories,
)


def _parse_csv_list(text: str) -> List[str]:
    return [tok.strip().lower() for tok in str(text).split(",") if tok.strip()]


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


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def _format_mean_std(values: Sequence[float]) -> str:
    mean, std = _mean_std(values)
    if not np.isfinite(mean):
        return "n/a"
    return f"{mean:.4f} +/- {std:.4f}"


def _save_result_json(path: str, result: TrajectoryPIAEvalResult) -> None:
    _write_json(
        path,
        {
            "dataset": result.dataset,
            "seed": result.seed,
            "operator_mode": result.operator_mode,
            "train_metrics": result.train_metrics,
            "val_metrics": result.val_metrics,
            "test_metrics": result.test_metrics,
            "best_epoch": result.best_epoch,
            "diagnostics": result.diagnostics,
            "operator_meta": result.operator_meta,
            "meta": result.meta,
            "history_rows": result.history_rows,
        },
    )


def _baseline_train_bundle(state) -> tuple[List[str], List[int], List[np.ndarray], Dict[str, object], Dict[str, float]]:
    train_tids = [str(v) for v in state.train.tids.tolist()]
    train_labels = [int(v) for v in state.train.y.tolist()]
    train_seqs = [np.asarray(v, dtype=np.float32) for v in state.train.z_seq_list]
    diagnostics = compute_trajectory_diagnostics(train_seqs, train_labels, continuity_ratio=1.0)
    return train_tids, train_labels, train_seqs, {"mode": "baseline"}, diagnostics


def _augment_with_shared_operator(
    state,
    *,
    operator: TrajectoryPIAOperator,
    gamma_main: float,
    smooth_lambda: float,
    mode: str,
    extra_meta: Dict[str, object] | None = None,
) -> tuple[List[str], List[int], List[np.ndarray], Dict[str, object], Dict[str, float]]:
    train_tids = [str(v) for v in state.train.tids.tolist()]
    train_labels = [int(v) for v in state.train.y.tolist()]
    train_seqs = [np.asarray(v, dtype=np.float32) for v in state.train.z_seq_list]
    aug_seqs, _delta_list, op_meta = operator.transform_many(
        train_seqs,
        gamma_main=float(gamma_main),
        smooth_lambda=float(smooth_lambda),
    )
    aug_tids = [f"{tid}__{mode}_aug" for tid in train_tids]
    final_tids = list(train_tids) + list(aug_tids)
    final_labels = list(train_labels) + list(train_labels)
    final_seqs = list(train_seqs) + list(aug_seqs)
    diagnostics = compute_trajectory_diagnostics(
        final_seqs,
        final_labels,
        continuity_ratio=float(op_meta["mean_continuity_distortion_ratio"]),
    )
    meta = dict(op_meta)
    if extra_meta:
        meta.update(dict(extra_meta))
    return final_tids, final_labels, final_seqs, meta, diagnostics


def _augment_with_dual_role_policy(
    state,
    *,
    dual_role_result: TrajectoryDualRolePolicyResult,
    rebasis_summary: Dict[str, object],
    constructive_pool_summary: Dict[str, object],
    discriminative_pool_summary: Dict[str, object],
    constructive_class_coverage_rows: Sequence[Dict[str, object]],
    discriminative_class_coverage_rows: Sequence[Dict[str, object]],
    policy_mode: str,
    scope_note: str,
) -> tuple[List[str], List[int], List[np.ndarray], Dict[str, object], Dict[str, float]]:
    train_tids = [str(v) for v in state.train.tids.tolist()]
    train_labels = [int(v) for v in state.train.y.tolist()]
    train_seqs = [np.asarray(v, dtype=np.float32) for v in state.train.z_seq_list]
    final_tids = list(train_tids) + list(dual_role_result.aug_tids)
    final_labels = list(train_labels) + list(dual_role_result.aug_labels)
    final_seqs = list(train_seqs) + list(dual_role_result.aug_z_seq_list)
    diagnostics = compute_trajectory_diagnostics(
        final_seqs,
        final_labels,
        continuity_ratio=float(dual_role_result.stitching_summary["stitched_continuity_distortion_ratio"]),
    )
    meta = {
        "generator_mode": "t2a_default_frozen",
        "policy_mode": str(policy_mode),
        "constructive_summary": dict(constructive_pool_summary),
        "discriminative_summary": dict(discriminative_pool_summary),
        "constructive_class_coverage_rows": [dict(v) for v in constructive_class_coverage_rows],
        "discriminative_class_coverage_rows": [dict(v) for v in discriminative_class_coverage_rows],
        "role_overlap_summary": dict(dual_role_result.role_overlap_summary),
        "stitching_summary": dict(dual_role_result.stitching_summary),
        "rebasis_summary": dict(rebasis_summary),
        "scope_note": str(scope_note),
    }
    return final_tids, final_labels, final_seqs, meta, diagnostics


def _rebuild_selected_rows_with_rebased_windows(
    *,
    train_tids: Sequence[str],
    train_z_seq_list: Sequence[np.ndarray],
    operator_new: TrajectoryPIAOperator,
    selected_rows: Sequence[Dict[str, object]],
    gamma_main: float,
    smooth_lambda: float,
) -> List[Dict[str, object]]:
    seqs = [np.asarray(v, dtype=np.float32) for v in train_z_seq_list]
    rebased_aug_seqs, _delta_list, _op_meta = operator_new.transform_many(
        seqs,
        gamma_main=float(gamma_main),
        smooth_lambda=float(smooth_lambda),
    )
    tid_to_index = {str(tid): idx for idx, tid in enumerate(train_tids)}
    out: List[Dict[str, object]] = []
    for row in selected_rows:
        tid = str(row["trial_id"])
        trial_index = int(tid_to_index[tid])
        win = int(row["window_index"])
        rebased_window = np.asarray(rebased_aug_seqs[trial_index][win], dtype=np.float32)
        row_new = dict(row)
        row_new["z_window_aug"] = rebased_window
        out.append(row_new)
    return out


def _augment_with_unified_policy(
    state,
    *,
    unified_result: TrajectoryUnifiedWindowPolicyResult,
    pool_mode: str,
    rebasis_summary: Dict[str, object],
) -> tuple[List[str], List[int], List[np.ndarray], Dict[str, object], Dict[str, float]]:
    train_tids = [str(v) for v in state.train.tids.tolist()]
    train_labels = [int(v) for v in state.train.y.tolist()]
    train_seqs = [np.asarray(v, dtype=np.float32) for v in state.train.z_seq_list]
    final_tids = list(train_tids) + list(unified_result.aug_tids)
    final_labels = list(train_labels) + list(unified_result.aug_labels)
    final_seqs = list(train_seqs) + list(unified_result.aug_z_seq_list)
    diagnostics = compute_trajectory_diagnostics(
        final_seqs,
        final_labels,
        continuity_ratio=float(unified_result.stitching_summary["stitched_continuity_distortion_ratio"]),
    )
    meta = {
        "generator_mode": "t2a_default_frozen",
        "window_policy_mode": str(pool_mode),
        "pool_summary": dict(unified_result.pool_summary),
        "class_coverage_rows": [dict(v) for v in unified_result.class_coverage_rows],
        "stitching_summary": dict(unified_result.stitching_summary),
        "rebasis_summary": dict(rebasis_summary),
        "unified_role_note": (
            "T6a-1 measures a unified-role local kNN margin policy. The same admitted window set drives both "
            "shared rebasis and final z_seq-level writeback."
        ),
    }
    return final_tids, final_labels, final_seqs, meta, diagnostics


def main() -> None:
    p = argparse.ArgumentParser(description="T6b safety-constructive + local-kNN-discriminative dual-role probe.")
    p.add_argument("--main-datasets", type=str, default="selfregulationscp1")
    p.add_argument("--anchor-datasets", type=str, default="natops")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/route_b_dynamic_dual_role_policy_t6b_20260330_formal")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--prop-win-ratio", type=float, default=0.20)
    p.add_argument("--prop-hop-ratio", type=float, default=0.10)
    p.add_argument("--min-window-extra-channels", type=int, default=4)
    p.add_argument("--min-hop-len", type=int, default=4)
    p.add_argument("--gru-hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--gamma-main", type=float, default=0.05)
    p.add_argument("--smooth-lambda", type=float, default=0.50)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--max-purity-drop", type=float, default=0.10)
    p.add_argument("--continuity-quantile", type=float, default=75.0)
    args = p.parse_args()

    main_datasets = _parse_csv_list(args.main_datasets)
    anchor_datasets = _parse_csv_list(args.anchor_datasets)
    datasets = list(dict.fromkeys(main_datasets + anchor_datasets))
    dataset_role = {ds: ("main" if ds in main_datasets else "anchor") for ds in datasets}
    seeds = _parse_seed_list(args.seeds)
    out_root = str(args.out_root)
    _ensure_dir(out_root)

    config_rows: List[Dict[str, object]] = []
    per_seed_rows: List[Dict[str, object]] = []
    constructive_rows_csv: List[Dict[str, object]] = []
    discriminative_rows_csv: List[Dict[str, object]] = []
    class_coverage_rows_csv: List[Dict[str, object]] = []
    overlap_rows_csv: List[Dict[str, object]] = []
    basis_rows: List[Dict[str, object]] = []
    stitching_rows: List[Dict[str, object]] = []
    diagnostics_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        role = dataset_role[str(dataset)]
        for seed in seeds:
            state = build_trajectory_representation(
                TrajectoryRepresentationConfig(
                    dataset=str(dataset),
                    seed=int(seed),
                    val_fraction=float(args.val_fraction),
                    spd_eps=float(args.spd_eps),
                    prop_win_ratio=float(args.prop_win_ratio),
                    prop_hop_ratio=float(args.prop_hop_ratio),
                    min_window_extra_channels=int(args.min_window_extra_channels),
                    min_hop_len=int(args.min_hop_len),
                )
            )
            seed_dir = os.path.join(out_root, f"{dataset}_seed{seed}")
            _ensure_dir(seed_dir)

            model_cfg = TrajectoryModelConfig(
                z_dim=int(state.z_dim),
                num_classes=int(state.num_classes),
                gru_hidden_dim=int(args.gru_hidden_dim),
                dropout=float(args.dropout),
            )
            eval_common = {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "patience": int(args.patience),
                "device": str(args.device),
            }
            operator_cfg = TrajectoryPIAOperatorConfig(seed=int(seed))
            frozen_operator = TrajectoryPIAOperator(operator_cfg).fit(state.train.z_seq_list)
            train_tids = [str(v) for v in state.train.tids.tolist()]
            train_labels = [int(v) for v in state.train.y.tolist()]
            reference_stats = build_window_feedback_reference_stats(
                train_labels=train_labels,
                train_z_seq_list=state.train.z_seq_list,
            )

            _write_json(
                os.path.join(seed_dir, "dynamic_dual_role_policy_t6b_split_meta.json"),
                {
                    "dataset": str(dataset),
                    "dataset_role": str(role),
                    "seed": int(seed),
                    "split_meta": dict(state.split_meta),
                    "meta": dict(state.meta),
                    "window_len": int(state.window_len),
                    "hop_len": int(state.hop_len),
                    "frozen_generator": {
                        "operator": "t2a_default",
                        "gamma_main": float(args.gamma_main),
                        "smooth_lambda": float(args.smooth_lambda),
                    },
                    "reference_scope": {
                        "knn": "orig_train_only_windows",
                        "self_exclusion": "same_class_excludes_current_window",
                        "class_centers": "orig_train_only_window_class_centers_frozen_per_dataset_seed",
                    },
                    "constructive_role": "window_safety_only",
                    "discriminative_role": "local_kNN_margin",
                    "coverage_guard": "max(8, ceil(0.05 * safe_window_count_class))",
                    "class_coverage_interpretation": "class_level_first_then_dataset_summary",
                },
            )

            mode_results: Dict[str, TrajectoryPIAEvalResult] = {}

            if role == "main":
                baseline_tids, baseline_labels, baseline_seqs, baseline_meta, baseline_diag = _baseline_train_bundle(state)
                baseline_result = evaluate_trajectory_train_final(
                    state,
                    seed=int(seed),
                    model_cfg=model_cfg,
                    eval_cfg=TrajectoryPIAEvalConfig(
                        operator_mode="baseline",
                        gamma_main=float(args.gamma_main),
                        smooth_lambda=float(args.smooth_lambda),
                        **eval_common,
                    ),
                    train_tids=baseline_tids,
                    train_labels=baseline_labels,
                    train_z_seq_list=baseline_seqs,
                    diagnostics=baseline_diag,
                    operator_meta=baseline_meta,
                )
                mode_results["baseline"] = baseline_result
                _save_result_json(os.path.join(seed_dir, "baseline_result.json"), baseline_result)

                t2a_result = evaluate_trajectory_pia_t2a(
                    state,
                    seed=int(seed),
                    model_cfg=model_cfg,
                    eval_cfg=TrajectoryPIAEvalConfig(
                        operator_mode="t2a_default",
                        gamma_main=float(args.gamma_main),
                        smooth_lambda=float(args.smooth_lambda),
                        **eval_common,
                    ),
                    operator_cfg=operator_cfg,
                    prefit_operator=frozen_operator,
                )
                mode_results["t2a_default"] = t2a_result
                _save_result_json(os.path.join(seed_dir, "t2a_default_result.json"), t2a_result)

            # T3 comparator
            t3_pool = build_trajectory_feedback_pool(
                train_tids=train_tids,
                train_labels=train_labels,
                train_z_seq_list=state.train.z_seq_list,
                operator=frozen_operator,
                cfg=TrajectoryFeedbackPoolConfig(
                    gamma_main=float(args.gamma_main),
                    smooth_lambda=float(args.smooth_lambda),
                    knn_k=int(args.knn_k),
                    max_purity_drop=float(args.max_purity_drop),
                    continuity_quantile=float(args.continuity_quantile),
                ),
            )
            t3_rebasis = fit_trajectory_feedback_rebasis(
                orig_train_z_seq_list=state.train.z_seq_list,
                feedback_z_seq_list=t3_pool.accepted_z_seq_list,
                old_operator=frozen_operator,
                operator_cfg=operator_cfg,
            )
            t3_tids, t3_labels, t3_seqs, t3_meta, t3_diag = _augment_with_shared_operator(
                state,
                operator=t3_rebasis.operator_new,
                gamma_main=float(args.gamma_main),
                smooth_lambda=float(args.smooth_lambda),
                mode="t3_rebasis",
                extra_meta={
                    "feedback_summary": dict(t3_pool.summary),
                    "rebasis_summary": dict(t3_rebasis.summary),
                    "generator_mode": "frozen_t2a_default",
                },
            )
            t3_result = evaluate_trajectory_train_final(
                state,
                seed=int(seed),
                model_cfg=model_cfg,
                eval_cfg=TrajectoryPIAEvalConfig(
                    operator_mode="t3_shared_rebasis",
                    gamma_main=float(args.gamma_main),
                    smooth_lambda=float(args.smooth_lambda),
                    **eval_common,
                ),
                train_tids=t3_tids,
                train_labels=t3_labels,
                train_z_seq_list=t3_seqs,
                diagnostics=t3_diag,
                operator_meta=t3_meta,
            )
            mode_results["t3_shared_rebasis"] = t3_result
            _save_result_json(os.path.join(seed_dir, "t3_shared_rebasis_result.json"), t3_result)
            basis_rows.append(
                {
                    "dataset": str(dataset),
                    "dataset_role": str(role),
                    "seed": int(seed),
                    "mode": "t3_shared_rebasis",
                    **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in t3_rebasis.summary.items()},
                }
            )

            # T4b radial and margin comparators
            for mode_name, gate_name in [
                ("t4b_window_radial_gate", "radial"),
                ("t4b_window_margin_gate", "margin"),
            ]:
                pool_result = build_window_feedback_pool(
                    train_tids=train_tids,
                    train_labels=train_labels,
                    train_z_seq_list=state.train.z_seq_list,
                    operator=frozen_operator,
                    reference_stats=reference_stats,
                    cfg=TrajectoryWindowFeedbackPoolConfig(
                        gamma_main=float(args.gamma_main),
                        smooth_lambda=float(args.smooth_lambda),
                        knn_k=int(args.knn_k),
                        max_purity_drop=float(args.max_purity_drop),
                        continuity_quantile=float(args.continuity_quantile),
                        informative_gate=str(gate_name),
                    ),
                )
                rebasis_result = fit_trajectory_feedback_rebasis(
                    orig_train_z_seq_list=state.train.z_seq_list,
                    feedback_z_seq_list=pool_result.accepted_window_seq_list,
                    old_operator=frozen_operator,
                    operator_cfg=operator_cfg,
                )
                aug_tids, aug_labels, aug_seqs, op_meta, diag = _augment_with_shared_operator(
                    state,
                    operator=rebasis_result.operator_new,
                    gamma_main=float(args.gamma_main),
                    smooth_lambda=float(args.smooth_lambda),
                    mode=mode_name,
                    extra_meta={
                        "feedback_summary": dict(pool_result.summary),
                        "rebasis_summary": dict(rebasis_result.summary),
                        "generator_mode": "frozen_t2a_default",
                    },
                )
                result = evaluate_trajectory_train_final(
                    state,
                    seed=int(seed),
                    model_cfg=model_cfg,
                    eval_cfg=TrajectoryPIAEvalConfig(
                        operator_mode=str(mode_name),
                        gamma_main=float(args.gamma_main),
                        smooth_lambda=float(args.smooth_lambda),
                        **eval_common,
                    ),
                    train_tids=aug_tids,
                    train_labels=aug_labels,
                    train_z_seq_list=aug_seqs,
                    diagnostics=diag,
                    operator_meta=op_meta,
                )
                mode_results[str(mode_name)] = result
                _save_result_json(os.path.join(seed_dir, f"{mode_name}_result.json"), result)

            # T5 comparator on main dataset only
            if role == "main":
                t5_constructive_pool = build_window_feedback_pool(
                    train_tids=train_tids,
                    train_labels=train_labels,
                    train_z_seq_list=state.train.z_seq_list,
                    operator=frozen_operator,
                    reference_stats=reference_stats,
                    cfg=TrajectoryWindowFeedbackPoolConfig(
                        gamma_main=float(args.gamma_main),
                        smooth_lambda=float(args.smooth_lambda),
                        knn_k=int(args.knn_k),
                        max_purity_drop=float(args.max_purity_drop),
                        continuity_quantile=float(args.continuity_quantile),
                        informative_gate="radial",
                    ),
                )
                t5_constructive_rebasis = fit_trajectory_feedback_rebasis(
                    orig_train_z_seq_list=state.train.z_seq_list,
                    feedback_z_seq_list=t5_constructive_pool.accepted_window_seq_list,
                    old_operator=frozen_operator,
                    operator_cfg=operator_cfg,
                )
                t5_discriminative_pool = build_window_feedback_pool(
                    train_tids=train_tids,
                    train_labels=train_labels,
                    train_z_seq_list=state.train.z_seq_list,
                    operator=t5_constructive_rebasis.operator_new,
                    reference_stats=reference_stats,
                    cfg=TrajectoryWindowFeedbackPoolConfig(
                        gamma_main=float(args.gamma_main),
                        smooth_lambda=float(args.smooth_lambda),
                        knn_k=int(args.knn_k),
                        max_purity_drop=float(args.max_purity_drop),
                        continuity_quantile=float(args.continuity_quantile),
                        informative_gate="margin",
                    ),
                )
                t5_dual_role = build_dual_role_augmented_trajectories(
                    train_tids=train_tids,
                    train_labels=train_labels,
                    train_z_seq_list=state.train.z_seq_list,
                    constructive_pool_summary=t5_constructive_pool.summary,
                    discriminative_pool_summary=t5_discriminative_pool.summary,
                    constructive_rows=t5_constructive_pool.accepted_window_rows,
                    discriminative_rows=t5_discriminative_pool.accepted_window_rows,
                )
                t5_tids, t5_labels, t5_seqs, t5_meta, t5_diag = _augment_with_dual_role_policy(
                    state,
                    dual_role_result=t5_dual_role,
                    rebasis_summary=t5_constructive_rebasis.summary,
                    constructive_pool_summary=t5_constructive_pool.summary,
                    discriminative_pool_summary=t5_discriminative_pool.summary,
                    constructive_class_coverage_rows=t5_constructive_pool.class_coverage_rows,
                    discriminative_class_coverage_rows=t5_discriminative_pool.class_coverage_rows,
                    policy_mode="t5_dual_role_policy",
                    scope_note=(
                        "T5 first version fixes constructive=radial and discriminative=margin as an overall dual-role policy. "
                        "This does not isolate their unique causal contributions."
                    ),
                )
                t5_result = evaluate_trajectory_train_final(
                    state,
                    seed=int(seed),
                    model_cfg=model_cfg,
                    eval_cfg=TrajectoryPIAEvalConfig(
                        operator_mode="t5_dual_role_policy",
                        gamma_main=float(args.gamma_main),
                        smooth_lambda=float(args.smooth_lambda),
                        **eval_common,
                    ),
                    train_tids=t5_tids,
                    train_labels=t5_labels,
                    train_z_seq_list=t5_seqs,
                    diagnostics=t5_diag,
                    operator_meta=t5_meta,
                )
                mode_results["t5_dual_role_policy"] = t5_result
                _save_result_json(os.path.join(seed_dir, "t5_dual_role_policy_result.json"), t5_result)

            # T6a-1 comparator
            t6a_pool = build_window_feedback_pool(
                train_tids=train_tids,
                train_labels=train_labels,
                train_z_seq_list=state.train.z_seq_list,
                operator=frozen_operator,
                reference_stats=reference_stats,
                cfg=TrajectoryWindowFeedbackPoolConfig(
                    gamma_main=float(args.gamma_main),
                    smooth_lambda=float(args.smooth_lambda),
                    knn_k=int(args.knn_k),
                    max_purity_drop=float(args.max_purity_drop),
                    continuity_quantile=float(args.continuity_quantile),
                    informative_gate="local_knn_margin",
                ),
            )
            t6a_rebasis = fit_trajectory_feedback_rebasis(
                orig_train_z_seq_list=state.train.z_seq_list,
                feedback_z_seq_list=t6a_pool.accepted_window_seq_list,
                old_operator=frozen_operator,
                operator_cfg=operator_cfg,
            )
            t6a_rebased_rows = _rebuild_selected_rows_with_rebased_windows(
                train_tids=train_tids,
                train_z_seq_list=state.train.z_seq_list,
                operator_new=t6a_rebasis.operator_new,
                selected_rows=t6a_pool.accepted_window_rows,
                gamma_main=float(args.gamma_main),
                smooth_lambda=float(args.smooth_lambda),
            )
            t6a_unified = build_unified_window_augmented_trajectories(
                train_tids=train_tids,
                train_labels=train_labels,
                train_z_seq_list=state.train.z_seq_list,
                selected_rows=t6a_rebased_rows,
                pool_summary=t6a_pool.summary,
                class_coverage_rows=t6a_pool.class_coverage_rows,
            )
            t6a_tids, t6a_labels, t6a_seqs, t6a_meta, t6a_diag = _augment_with_unified_policy(
                state,
                unified_result=t6a_unified,
                pool_mode="t6a1_local_knn_margin_unified",
                rebasis_summary=t6a_rebasis.summary,
            )
            t6a_result = evaluate_trajectory_train_final(
                state,
                seed=int(seed),
                model_cfg=model_cfg,
                eval_cfg=TrajectoryPIAEvalConfig(
                    operator_mode="t6a1_local_knn_margin_unified",
                    gamma_main=float(args.gamma_main),
                    smooth_lambda=float(args.smooth_lambda),
                    **eval_common,
                ),
                train_tids=t6a_tids,
                train_labels=t6a_labels,
                train_z_seq_list=t6a_seqs,
                diagnostics=t6a_diag,
                operator_meta=t6a_meta,
            )
            mode_results["t6a1_local_knn_margin_unified"] = t6a_result
            _save_result_json(os.path.join(seed_dir, "t6a1_local_knn_margin_unified_result.json"), t6a_result)

            # T6b target policy
            constructive_pool = build_window_feedback_pool(
                train_tids=train_tids,
                train_labels=train_labels,
                train_z_seq_list=state.train.z_seq_list,
                operator=frozen_operator,
                reference_stats=reference_stats,
                cfg=TrajectoryWindowFeedbackPoolConfig(
                    gamma_main=float(args.gamma_main),
                    smooth_lambda=float(args.smooth_lambda),
                    knn_k=int(args.knn_k),
                    max_purity_drop=float(args.max_purity_drop),
                    continuity_quantile=float(args.continuity_quantile),
                    informative_gate="safety_only",
                ),
            )
            constructive_rebasis = fit_trajectory_feedback_rebasis(
                orig_train_z_seq_list=state.train.z_seq_list,
                feedback_z_seq_list=constructive_pool.accepted_window_seq_list,
                old_operator=frozen_operator,
                operator_cfg=operator_cfg,
            )
            discriminative_pool = build_window_feedback_pool(
                train_tids=train_tids,
                train_labels=train_labels,
                train_z_seq_list=state.train.z_seq_list,
                operator=constructive_rebasis.operator_new,
                reference_stats=reference_stats,
                cfg=TrajectoryWindowFeedbackPoolConfig(
                    gamma_main=float(args.gamma_main),
                    smooth_lambda=float(args.smooth_lambda),
                    knn_k=int(args.knn_k),
                    max_purity_drop=float(args.max_purity_drop),
                    continuity_quantile=float(args.continuity_quantile),
                    informative_gate="local_knn_margin",
                ),
            )
            dual_role_result = build_dual_role_augmented_trajectories(
                train_tids=train_tids,
                train_labels=train_labels,
                train_z_seq_list=state.train.z_seq_list,
                constructive_pool_summary=constructive_pool.summary,
                discriminative_pool_summary=discriminative_pool.summary,
                constructive_rows=constructive_pool.accepted_window_rows,
                discriminative_rows=discriminative_pool.accepted_window_rows,
            )
            t6b_tids, t6b_labels, t6b_seqs, t6b_meta, t6b_diag = _augment_with_dual_role_policy(
                state,
                dual_role_result=dual_role_result,
                rebasis_summary=constructive_rebasis.summary,
                constructive_pool_summary=constructive_pool.summary,
                discriminative_pool_summary=discriminative_pool.summary,
                constructive_class_coverage_rows=constructive_pool.class_coverage_rows,
                discriminative_class_coverage_rows=discriminative_pool.class_coverage_rows,
                policy_mode="t6b_safety_constructive_local_knn_discriminative",
                scope_note=(
                    "T6b measures the overall effect of a dual-role policy with constructive=window_safety_only "
                    "and discriminative=local_kNN_margin. It does not prove these are uniquely optimal final roles."
                ),
            )
            t6b_result = evaluate_trajectory_train_final(
                state,
                seed=int(seed),
                model_cfg=model_cfg,
                eval_cfg=TrajectoryPIAEvalConfig(
                    operator_mode="t6b_dual_role_policy",
                    gamma_main=float(args.gamma_main),
                    smooth_lambda=float(args.smooth_lambda),
                    **eval_common,
                ),
                train_tids=t6b_tids,
                train_labels=t6b_labels,
                train_z_seq_list=t6b_seqs,
                diagnostics=t6b_diag,
                operator_meta=t6b_meta,
            )
            mode_results["t6b_dual_role_policy"] = t6b_result
            _save_result_json(os.path.join(seed_dir, "t6b_dual_role_policy_result.json"), t6b_result)

            constructive_rows_csv.append(
                {
                    "dataset": str(dataset),
                    "dataset_role": str(role),
                    "seed": int(seed),
                    "mode": "t6b_dual_role_policy",
                    **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in constructive_pool.summary.items()},
                }
            )
            discriminative_rows_csv.append(
                {
                    "dataset": str(dataset),
                    "dataset_role": str(role),
                    "seed": int(seed),
                    "mode": "t6b_dual_role_policy",
                    **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in discriminative_pool.summary.items()},
                }
            )
            for row in constructive_pool.class_coverage_rows:
                class_coverage_rows_csv.append(
                    {
                        "dataset": str(dataset),
                        "dataset_role": str(role),
                        "seed": int(seed),
                        "mode": "t6b_dual_role_policy",
                        "pool_role": "constructive",
                        **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in row.items()},
                    }
                )
            for row in discriminative_pool.class_coverage_rows:
                class_coverage_rows_csv.append(
                    {
                        "dataset": str(dataset),
                        "dataset_role": str(role),
                        "seed": int(seed),
                        "mode": "t6b_dual_role_policy",
                        "pool_role": "discriminative",
                        **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in row.items()},
                    }
                )
            overlap_rows_csv.append(
                {
                    "dataset": str(dataset),
                    "dataset_role": str(role),
                    "seed": int(seed),
                    **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in dual_role_result.role_overlap_summary.items()},
                }
            )
            basis_rows.append(
                {
                    "dataset": str(dataset),
                    "dataset_role": str(role),
                    "seed": int(seed),
                    "mode": "t6b_dual_role_policy",
                    **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in constructive_rebasis.summary.items()},
                }
            )
            stitching_rows.append(
                {
                    "dataset": str(dataset),
                    "dataset_role": str(role),
                    "seed": int(seed),
                    **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in dual_role_result.stitching_summary.items()},
                }
            )

            mode_order = (
                [
                    "baseline",
                    "t2a_default",
                    "t3_shared_rebasis",
                    "t4b_window_radial_gate",
                    "t4b_window_margin_gate",
                    "t5_dual_role_policy",
                    "t6a1_local_knn_margin_unified",
                    "t6b_dual_role_policy",
                ]
                if role == "main"
                else [
                    "t3_shared_rebasis",
                    "t4b_window_radial_gate",
                    "t4b_window_margin_gate",
                    "t6a1_local_knn_margin_unified",
                    "t6b_dual_role_policy",
                ]
            )
            baseline_f1 = float(mode_results["baseline"].test_metrics["macro_f1"]) if "baseline" in mode_results else float("nan")
            t2a_f1 = float(mode_results["t2a_default"].test_metrics["macro_f1"]) if "t2a_default" in mode_results else float("nan")
            t3_f1 = float(mode_results["t3_shared_rebasis"].test_metrics["macro_f1"])
            t4b_radial_f1 = float(mode_results["t4b_window_radial_gate"].test_metrics["macro_f1"])
            t4b_margin_f1 = float(mode_results["t4b_window_margin_gate"].test_metrics["macro_f1"])
            t6a_f1 = float(mode_results["t6a1_local_knn_margin_unified"].test_metrics["macro_f1"])
            t6b_f1 = float(mode_results["t6b_dual_role_policy"].test_metrics["macro_f1"])

            for mode in mode_order:
                result = mode_results[str(mode)]
                config_rows.append(
                    {
                        "dataset": str(dataset),
                        "dataset_role": str(role),
                        "seed": int(seed),
                        "mode": str(mode),
                        "gamma_main": float(args.gamma_main),
                        "smooth_lambda": float(args.smooth_lambda),
                        "window_len": int(state.window_len),
                        "hop_len": int(state.hop_len),
                        "epochs": int(args.epochs),
                        "batch_size": int(args.batch_size),
                    }
                )
                per_seed_rows.append(
                    {
                        "dataset": str(dataset),
                        "dataset_role": str(role),
                        "seed": int(seed),
                        "mode": str(mode),
                        "test_macro_f1": float(result.test_metrics["macro_f1"]),
                        "delta_vs_baseline": float(result.test_metrics["macro_f1"]) - baseline_f1 if np.isfinite(baseline_f1) else float("nan"),
                        "delta_vs_t2a_default": float(result.test_metrics["macro_f1"]) - t2a_f1 if np.isfinite(t2a_f1) else float("nan"),
                        "delta_vs_t3_shared_rebasis": float(result.test_metrics["macro_f1"]) - t3_f1,
                        "delta_vs_t4b_window_radial_gate": float(result.test_metrics["macro_f1"]) - t4b_radial_f1,
                        "delta_vs_t4b_window_margin_gate": float(result.test_metrics["macro_f1"]) - t4b_margin_f1,
                        "delta_vs_t6a1_local_knn_margin_unified": float(result.test_metrics["macro_f1"]) - t6a_f1,
                        "delta_vs_t6b_dual_role_policy": float(result.test_metrics["macro_f1"]) - t6b_f1,
                    }
                )
                diagnostics_rows.append(
                    {
                        "dataset": str(dataset),
                        "dataset_role": str(role),
                        "seed": int(seed),
                        "mode": str(mode),
                        **{k: float(v) for k, v in result.diagnostics.items()},
                    }
                )

    config_df = pd.DataFrame(config_rows)
    per_seed_df = pd.DataFrame(per_seed_rows)
    constructive_df = pd.DataFrame(constructive_rows_csv)
    discriminative_df = pd.DataFrame(discriminative_rows_csv)
    class_coverage_df = pd.DataFrame(class_coverage_rows_csv)
    overlap_df = pd.DataFrame(overlap_rows_csv)
    basis_df = pd.DataFrame(basis_rows)
    stitching_df = pd.DataFrame(stitching_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows)

    summary_rows: List[Dict[str, object]] = []
    for dataset in datasets:
        ds = per_seed_df[per_seed_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        row = {"dataset": str(dataset), "dataset_role": str(dataset_role[str(dataset)])}
        for mode in [
            "baseline",
            "t2a_default",
            "t3_shared_rebasis",
            "t4b_window_radial_gate",
            "t4b_window_margin_gate",
            "t5_dual_role_policy",
            "t6a1_local_knn_margin_unified",
            "t6b_dual_role_policy",
        ]:
            vals = ds[ds["mode"] == mode]["test_macro_f1"].astype(float).tolist()
            mean, std = _mean_std(vals)
            row[f"{mode}_macro_f1_mean"] = mean
            row[f"{mode}_macro_f1_std"] = std
        candidates = {
            mode: row.get(f"{mode}_macro_f1_mean", float("nan"))
            for mode in [
                "t3_shared_rebasis",
                "t4b_window_radial_gate",
                "t4b_window_margin_gate",
                "t6a1_local_knn_margin_unified",
                "t6b_dual_role_policy",
            ]
            if np.isfinite(row.get(f"{mode}_macro_f1_mean", float("nan")))
        }
        row["best_mode"] = max(candidates.items(), key=lambda kv: kv[1])[0] if candidates else "n/a"
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    config_csv = os.path.join(out_root, "dynamic_dual_role_policy_t6b_config_table.csv")
    per_seed_csv = os.path.join(out_root, "dynamic_dual_role_policy_t6b_per_seed.csv")
    summary_csv = os.path.join(out_root, "dynamic_dual_role_policy_t6b_dataset_summary.csv")
    constructive_csv = os.path.join(out_root, "dynamic_dual_role_policy_t6b_constructive_pool_summary.csv")
    discriminative_csv = os.path.join(out_root, "dynamic_dual_role_policy_t6b_discriminative_pool_summary.csv")
    class_coverage_csv = os.path.join(out_root, "dynamic_dual_role_policy_t6b_class_coverage_summary.csv")
    overlap_csv = os.path.join(out_root, "dynamic_dual_role_policy_t6b_role_overlap_summary.csv")
    basis_csv = os.path.join(out_root, "dynamic_dual_role_policy_t6b_basis_shift_summary.csv")
    stitching_csv = os.path.join(out_root, "dynamic_dual_role_policy_t6b_stitching_summary.csv")
    diagnostics_csv = os.path.join(out_root, "dynamic_dual_role_policy_t6b_diagnostics_summary.csv")
    config_df.to_csv(config_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    constructive_df.to_csv(constructive_csv, index=False)
    discriminative_df.to_csv(discriminative_csv, index=False)
    class_coverage_df.to_csv(class_coverage_csv, index=False)
    overlap_df.to_csv(overlap_csv, index=False)
    basis_df.to_csv(basis_csv, index=False)
    stitching_df.to_csv(stitching_csv, index=False)
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    lines: List[str] = [
        "# T6b Dual-role Sample Policy Conclusion",
        "",
        "This first-pass T6b probe freezes generator, shared rebasis, and classifier, and separates constructive windows from discriminative windows.",
        "",
        "Interpretation guardrails:",
        "",
        "- `window_safety_only` is used as the first constructive-side scaffold because it is safer and less biased than the current stronger geometry-moving gates; this does not prove it is the uniquely optimal constructive gate.",
        "- `local_kNN_margin` is used as the first discriminative-side local proxy because T6a-1 suggests it is not suitable as a unified gate; this does not prove it is the uniquely optimal discriminative gate.",
        "- all kNN queries use `orig-train-only windows`, and same-class local neighbors exclude the current window itself.",
        "- final stitched trajectories are constructed only at the `z_seq` representation level; no raw-level stitching is used.",
        "- low-coverage interpretation is class-level first, then dataset summary; average coverage alone is not sufficient to claim the gate has been effectively triggered.",
        "",
    ]
    for dataset in datasets:
        ds_summary = summary_df[summary_df["dataset"] == dataset]
        if ds_summary.empty:
            continue
        role = str(ds_summary["dataset_role"].iloc[0])
        lines.append(f"## {dataset} ({role})")
        lines.append("")
        if role == "main":
            lines.append(
                f"- baseline: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 'baseline')]['test_macro_f1'].tolist())}"
            )
            lines.append(
                f"- t2a_default: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't2a_default')]['test_macro_f1'].tolist())}"
            )
            lines.append(
                f"- t5_dual_role_policy: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't5_dual_role_policy')]['test_macro_f1'].tolist())}"
            )
        lines.append(
            f"- t3_shared_rebasis: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't3_shared_rebasis')]['test_macro_f1'].tolist())}"
        )
        lines.append(
            f"- t4b_window_radial_gate: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't4b_window_radial_gate')]['test_macro_f1'].tolist())}"
        )
        lines.append(
            f"- t4b_window_margin_gate: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't4b_window_margin_gate')]['test_macro_f1'].tolist())}"
        )
        lines.append(
            f"- t6a1_local_knn_margin_unified: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't6a1_local_knn_margin_unified')]['test_macro_f1'].tolist())}"
        )
        lines.append(
            f"- t6b_dual_role_policy: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't6b_dual_role_policy')]['test_macro_f1'].tolist())}"
        )
        lines.append("")

    lines.extend(
        [
            "## Success Layers",
            "",
            "- Weak success: `t6b_dual_role_policy > t3_shared_rebasis` and no obvious stitching pathology.",
            "- Medium success: `t6b_dual_role_policy > max(t4b_window_margin_gate, t6a1_local_knn_margin_unified)` and role overlap is not near-total.",
            "- Strong success: SCP1 improves in both basis movement and end performance while class-level low-coverage does not dominate key classes.",
        ]
    )
    conclusion_md = os.path.join(out_root, "dynamic_dual_role_policy_t6b_conclusion.md")
    with open(conclusion_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[T6b] wrote config to: {config_csv}")
    print(f"[T6b] wrote per-seed to: {per_seed_csv}")
    print(f"[T6b] wrote dataset summary to: {summary_csv}")
    print(f"[T6b] wrote constructive-pool summary to: {constructive_csv}")
    print(f"[T6b] wrote discriminative-pool summary to: {discriminative_csv}")
    print(f"[T6b] wrote class-coverage summary to: {class_coverage_csv}")
    print(f"[T6b] wrote role-overlap summary to: {overlap_csv}")
    print(f"[T6b] wrote basis-shift summary to: {basis_csv}")
    print(f"[T6b] wrote stitching summary to: {stitching_csv}")
    print(f"[T6b] wrote diagnostics to: {diagnostics_csv}")
    print(f"[T6b] wrote conclusion to: {conclusion_md}")


if __name__ == "__main__":
    main()
