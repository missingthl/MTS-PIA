#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
        "generator_mode": "t5_dual_role_policy",
        "constructive_summary": dict(dual_role_result.constructive_summary),
        "discriminative_summary": dict(dual_role_result.discriminative_summary),
        "role_overlap_summary": dict(dual_role_result.role_overlap_summary),
        "stitching_summary": dict(dual_role_result.stitching_summary),
        "causal_scope_note": (
            "T5 measures the overall effect of a dual-role sample policy; "
            "it does not yet isolate the unique causal contribution of radial-only constructive gating "
            "versus margin-only discriminative gating."
        ),
    }
    return final_tids, final_labels, final_seqs, meta, diagnostics


def main() -> None:
    p = argparse.ArgumentParser(description="T5 dual-role sample policy probe.")
    p.add_argument("--main-datasets", type=str, default="selfregulationscp1")
    p.add_argument("--anchor-datasets", type=str, default="natops")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/route_b_dynamic_dual_role_policy_t5_20260329_formal")
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
                os.path.join(seed_dir, "dynamic_dual_role_policy_t5_split_meta.json"),
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
                    "window_reference_scope": "orig_train_only_windows",
                    "margin_center_scope": "orig_train_only_window_class_centers_frozen_per_dataset_seed",
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

            # T3 baseline comparator
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
            t4b_results: Dict[str, tuple[TrajectoryPIAEvalResult, Dict[str, object], Dict[str, object]]] = {}
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
                t4b_results[str(mode_name)] = (result, dict(pool_result.summary), dict(rebasis_result.summary))
                _save_result_json(os.path.join(seed_dir, f"{mode_name}_result.json"), result)
                basis_rows.append(
                    {
                        "dataset": str(dataset),
                        "dataset_role": str(role),
                        "seed": int(seed),
                        "mode": str(mode_name),
                        **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in rebasis_result.summary.items()},
                    }
                )

            # T5 constructive -> rebasis, discriminative -> final train writeback on z_seq only
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
                    informative_gate="radial",
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
                    informative_gate="margin",
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
            t5_tids, t5_labels, t5_seqs, t5_meta, t5_diag = _augment_with_dual_role_policy(
                state,
                dual_role_result=dual_role_result,
            )
            t5_meta["rebasis_summary"] = dict(constructive_rebasis.summary)
            t5_meta["constructive_scope_note"] = (
                "T5 first version fixes radial_gain as the constructive-side proxy because T4b showed stronger basis-moving behavior under radial gating. "
                "This does not prove radial is the uniquely optimal constructive gate."
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
            basis_rows.append(
                {
                    "dataset": str(dataset),
                    "dataset_role": str(role),
                    "seed": int(seed),
                    "mode": "t5_dual_role_policy",
                    **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in constructive_rebasis.summary.items()},
                }
            )
            constructive_rows_csv.append(
                {
                    "dataset": str(dataset),
                    "dataset_role": str(role),
                    "seed": int(seed),
                    "mode": "t5_dual_role_policy",
                    **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in constructive_pool.summary.items()},
                }
            )
            discriminative_rows_csv.append(
                {
                    "dataset": str(dataset),
                    "dataset_role": str(role),
                    "seed": int(seed),
                    "mode": "t5_dual_role_policy",
                    **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in discriminative_pool.summary.items()},
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
            stitching_rows.append(
                {
                    "dataset": str(dataset),
                    "dataset_role": str(role),
                    "seed": int(seed),
                    **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in dual_role_result.stitching_summary.items()},
                }
            )

            mode_order = (
                ["baseline", "t2a_default", "t3_shared_rebasis", "t4b_window_radial_gate", "t4b_window_margin_gate", "t5_dual_role_policy"]
                if role == "main"
                else ["t3_shared_rebasis", "t4b_window_radial_gate", "t4b_window_margin_gate", "t5_dual_role_policy"]
            )
            baseline_f1 = float(mode_results["baseline"].test_metrics["macro_f1"]) if "baseline" in mode_results else float("nan")
            t2a_f1 = float(mode_results["t2a_default"].test_metrics["macro_f1"]) if "t2a_default" in mode_results else float("nan")
            t3_f1 = float(mode_results["t3_shared_rebasis"].test_metrics["macro_f1"])
            t4b_radial_f1 = float(mode_results["t4b_window_radial_gate"].test_metrics["macro_f1"])
            t4b_margin_f1 = float(mode_results["t4b_window_margin_gate"].test_metrics["macro_f1"])
            t5_f1 = float(mode_results["t5_dual_role_policy"].test_metrics["macro_f1"])

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
                        "delta_vs_t5_dual_role_policy": float(result.test_metrics["macro_f1"]) - t5_f1,
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
        ]:
            vals = ds[ds["mode"] == mode]["test_macro_f1"].astype(float).tolist()
            mean, std = _mean_std(vals)
            row[f"{mode}_macro_f1_mean"] = mean
            row[f"{mode}_macro_f1_std"] = std
        candidates = {
            mode: row.get(f"{mode}_macro_f1_mean", float("nan"))
            for mode in ["t3_shared_rebasis", "t4b_window_radial_gate", "t4b_window_margin_gate", "t5_dual_role_policy"]
            if np.isfinite(row.get(f"{mode}_macro_f1_mean", float("nan")))
        }
        row["best_mode"] = max(candidates.items(), key=lambda kv: kv[1])[0] if candidates else "n/a"
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    config_csv = os.path.join(out_root, "dynamic_dual_role_policy_t5_config_table.csv")
    per_seed_csv = os.path.join(out_root, "dynamic_dual_role_policy_t5_per_seed.csv")
    summary_csv = os.path.join(out_root, "dynamic_dual_role_policy_t5_dataset_summary.csv")
    constructive_csv = os.path.join(out_root, "dynamic_dual_role_policy_t5_constructive_pool_summary.csv")
    discriminative_csv = os.path.join(out_root, "dynamic_dual_role_policy_t5_discriminative_pool_summary.csv")
    overlap_csv = os.path.join(out_root, "dynamic_dual_role_policy_t5_role_overlap_summary.csv")
    basis_csv = os.path.join(out_root, "dynamic_dual_role_policy_t5_basis_shift_summary.csv")
    stitching_csv = os.path.join(out_root, "dynamic_dual_role_policy_t5_stitching_summary.csv")
    diagnostics_csv = os.path.join(out_root, "dynamic_dual_role_policy_t5_diagnostics_summary.csv")
    config_df.to_csv(config_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    constructive_df.to_csv(constructive_csv, index=False)
    discriminative_df.to_csv(discriminative_csv, index=False)
    overlap_df.to_csv(overlap_csv, index=False)
    basis_df.to_csv(basis_csv, index=False)
    stitching_df.to_csv(stitching_csv, index=False)
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    lines: List[str] = [
        "# T5 Dual-role Sample Policy Conclusion",
        "",
        "This first-pass T5 probe freezes generator, window policy, shared rebasis, and classifier, and separates constructive samples from discriminative samples.",
        "",
        "Interpretation guardrails:",
        "",
        "- `radial_gain` is fixed as the constructive-side proxy because T4b showed stronger basis-moving behavior under radial gating; this does not prove radial is the uniquely optimal constructive gate.",
        "- `margin_gain` is fixed as the discriminative-side proxy because it is more directly aligned with class separation; this does not prove margin is the uniquely optimal discriminative gate.",
        "- final stitched trajectories are constructed only at the `z_seq` representation level; no raw-level stitching is used.",
        "- T5 measures the overall effect of a dual-role sample policy, not a strict causal decomposition of radial versus margin.",
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
            f"- t3_shared_rebasis: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't3_shared_rebasis')]['test_macro_f1'].tolist())}"
        )
        lines.append(
            f"- t4b_window_radial_gate: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't4b_window_radial_gate')]['test_macro_f1'].tolist())}"
        )
        lines.append(
            f"- t4b_window_margin_gate: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't4b_window_margin_gate')]['test_macro_f1'].tolist())}"
        )
        lines.append(
            f"- t5_dual_role_policy: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't5_dual_role_policy')]['test_macro_f1'].tolist())}"
        )
        lines.append("")

    lines.extend(
        [
            "## Success Layers",
            "",
            "- Weak success: `t5_dual_role_policy > t3_shared_rebasis` and no obvious stitching pathology.",
            "- Medium success: `t5_dual_role_policy > max(t4b_window_radial_gate, t4b_window_margin_gate)` and role overlap is not near-total.",
            "- Strong success: SCP1 improves in both basis movement and end performance while stitching continuity remains healthy.",
        ]
    )
    conclusion_md = os.path.join(out_root, "dynamic_dual_role_policy_t5_conclusion.md")
    with open(conclusion_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[T5] wrote config to: {config_csv}")
    print(f"[T5] wrote per-seed to: {per_seed_csv}")
    print(f"[T5] wrote dataset summary to: {summary_csv}")
    print(f"[T5] wrote constructive pool summary to: {constructive_csv}")
    print(f"[T5] wrote discriminative pool summary to: {discriminative_csv}")
    print(f"[T5] wrote role overlap summary to: {overlap_csv}")
    print(f"[T5] wrote basis-shift summary to: {basis_csv}")
    print(f"[T5] wrote stitching summary to: {stitching_csv}")
    print(f"[T5] wrote diagnostics to: {diagnostics_csv}")
    print(f"[T5] wrote conclusion to: {conclusion_md}")


if __name__ == "__main__":
    main()
