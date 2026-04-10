#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from typing import Dict, List, Sequence

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from route_b_unified.pia_operator_value_probe import (  # noqa: E402
    CenterBasedOperatorConfig,
    ContinuousGeometricCouplingConfig,
    FixedReferenceGeometryConfig,
    OperatorApplyResult,
    SingleTemplatePIADiscriminativeConfig,
    SingleTemplatePIAStageARepairConfig,
    SingleTemplatePIAValueConfig,
    apply_center_based_operator,
    apply_continuous_geometric_coupling,
    apply_single_template_pia_stage_a_variant,
    build_fixed_reference_geometry,
    fit_single_template_pia_operator,
    fit_single_template_pia_operator_discriminative,
)
from route_b_unified.scp_prototype_memory import (  # noqa: E402
    SCPPrototypeMemoryConfig,
    build_scp_prototype_memory,
)
from route_b_unified.trajectory_feedback_pool_windows import (  # noqa: E402
    build_window_feedback_reference_stats,
)
from route_b_unified.trajectory_representation import (  # noqa: E402
    TrajectoryRepresentationConfig,
    TrajectoryRepresentationState,
    build_trajectory_representation,
)


def _parse_csv_list(text: str) -> List[str]:
    out = [tok.strip().lower() for tok in str(text).split(",") if tok.strip()]
    if not out:
        raise ValueError("csv list cannot be empty")
    return out


def _parse_int_csv(text: str) -> List[int]:
    out = sorted(set(int(tok.strip()) for tok in str(text).split(",") if tok.strip()))
    if not out:
        raise ValueError("integer csv list cannot be empty")
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
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def _format_mean_std(values: Sequence[float]) -> str:
    mean, std = _mean_std(values)
    return f"{mean:.4f} +/- {std:.4f}"


def _build_dense_state(args: argparse.Namespace, dataset: str, seed: int) -> TrajectoryRepresentationState:
    return build_trajectory_representation(
        TrajectoryRepresentationConfig(
            dataset=str(dataset),
            seed=int(seed),
            val_fraction=float(args.val_fraction),
            spd_eps=float(args.spd_eps),
            prop_win_ratio=float(args.prop_win_ratio),
            prop_hop_ratio=float(args.prop_hop_ratio),
            min_window_extra_channels=int(args.min_window_extra_channels),
            min_hop_len=int(args.min_hop_len),
            force_hop_len=int(args.force_hop_len),
        )
    )


def _clone_state_with_zseqs(
    state: TrajectoryRepresentationState,
    *,
    train_z_seq_list: Sequence[np.ndarray],
    val_z_seq_list: Sequence[np.ndarray],
    test_z_seq_list: Sequence[np.ndarray],
) -> TrajectoryRepresentationState:
    train_split = replace(state.train, z_seq_list=[np.asarray(v, dtype=np.float32) for v in train_z_seq_list])
    val_split = replace(state.val, z_seq_list=[np.asarray(v, dtype=np.float32) for v in val_z_seq_list])
    test_split = replace(state.test, z_seq_list=[np.asarray(v, dtype=np.float32) for v in test_z_seq_list])
    return replace(state, train=train_split, val=val_split, test=test_split)


def _evaluate_terminal(
    state: TrajectoryRepresentationState,
    *,
    seed: int,
    args: argparse.Namespace,
) -> tuple[str, Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, object]]:
    terminal = str(args.terminal).strip().lower()
    if terminal == "dynamic_minirocket":
        from route_b_unified.trajectory_minirocket_evaluator import (  # noqa: E402
            TrajectoryMiniRocketEvalConfig,
            evaluate_dynamic_minirocket_classifier,
        )

        eval_cfg = TrajectoryMiniRocketEvalConfig(
            n_kernels=int(args.minirocket_n_kernels),
            n_jobs=int(args.minirocket_n_jobs),
            padding_mode="edge",
            target_len_mode="train_max_len",
        )
        result = evaluate_dynamic_minirocket_classifier(
            state,
            seed=int(seed),
            eval_cfg=eval_cfg,
        )
        return terminal, dict(result.train_metrics), dict(result.val_metrics), dict(result.test_metrics), dict(result.meta)
    if terminal == "dynamic_gru":
        from route_b_unified.trajectory_classifier import TrajectoryModelConfig  # noqa: E402
        from route_b_unified.trajectory_evaluator import (  # noqa: E402
            TrajectoryEvalConfig,
            TrajectoryEvalResult,
            evaluate_trajectory_classifier,
        )

        model_cfg = TrajectoryModelConfig(
            z_dim=int(state.z_dim),
            num_classes=int(state.num_classes),
            gru_hidden_dim=int(args.gru_hidden_dim),
            dropout=float(args.dropout),
        )
        eval_cfg = TrajectoryEvalConfig(
            variant="dynamic_gru",
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            patience=int(args.patience),
            device=str(args.device),
        )
        result: TrajectoryEvalResult = evaluate_trajectory_classifier(
            state,
            seed=int(seed),
            model_cfg=model_cfg,
            eval_cfg=eval_cfg,
        )
        return terminal, dict(result.train_metrics), dict(result.val_metrics), dict(result.test_metrics), {**dict(result.meta), "best_epoch": int(result.best_epoch)}
    raise ValueError("terminal must be one of: dynamic_minirocket, dynamic_gru")


def _build_structure_diag(
    *,
    tids: Sequence[str],
    labels: Sequence[int],
    z_seq_list: Sequence[np.ndarray],
    prototype_count: int,
    seed: int,
) -> Dict[str, float]:
    memory = build_scp_prototype_memory(
        train_tids=[str(v) for v in tids],
        train_labels=[int(v) for v in labels],
        train_z_seq_list=[np.asarray(v, dtype=np.float32) for v in z_seq_list],
        reference_stats=build_window_feedback_reference_stats(
            train_labels=[int(v) for v in labels],
            train_z_seq_list=[np.asarray(v, dtype=np.float32) for v in z_seq_list],
        ),
        cfg=SCPPrototypeMemoryConfig(
            prototype_count=int(prototype_count),
            cluster_mode="kmeans_centroid",
            seed=int(seed),
        ),
    )
    return next(r for r in memory.structure_rows if str(r["mode"]) == "prototype_memory")


def _weighted_mean(summary_rows: Sequence[Dict[str, object]], key: str, weight_key: str = "applied_window_count") -> float:
    vals: List[float] = []
    wts: List[float] = []
    for row in summary_rows:
        if key not in row:
            continue
        vals.append(float(row[key]))
        wts.append(float(row.get(weight_key, 1.0)))
    if not vals:
        return 0.0
    arr = np.asarray(vals, dtype=np.float64)
    wt = np.asarray(wts, dtype=np.float64)
    if float(np.sum(wt)) <= 0.0:
        return float(np.mean(arr))
    return float(np.sum(arr * wt) / np.sum(wt))


def _merge_apply_summaries(
    arm_name: str,
    *,
    train_result: OperatorApplyResult,
    val_result: OperatorApplyResult,
    test_result: OperatorApplyResult,
) -> Dict[str, object]:
    rows = [
        dict(train_result.summary, split="train"),
        dict(val_result.summary, split="val"),
        dict(test_result.summary, split="test"),
    ]
    return {
        "arm": str(arm_name),
        "fit_window_count": int(train_result.summary.get("fit_window_count", 0)),
        "fit_trial_count": int(train_result.summary.get("fit_trial_count", 0)),
        "operator_norm_mean": _weighted_mean(rows, "operator_norm_mean"),
        "operator_norm_p95": _weighted_mean(rows, "operator_norm_p95"),
        "operator_to_step_ratio_mean": _weighted_mean(rows, "operator_to_step_ratio_mean"),
        "local_step_distortion_ratio_mean": _weighted_mean(rows, "local_step_distortion_ratio_mean"),
        "local_step_distortion_ratio_p95": _weighted_mean(rows, "local_step_distortion_ratio_p95"),
        "operator_direction_stability": _weighted_mean(rows, "operator_direction_stability"),
        "response_vs_margin_correlation": _weighted_mean(rows, "response_vs_margin_correlation"),
        "activation_coverage_ratio": _weighted_mean(rows, "activation_coverage_ratio"),
        "preactivation_clip_rate": _weighted_mean(rows, "preactivation_clip_rate"),
        "response_centering_std_after_fix": _weighted_mean(rows, "response_centering_std_after_fix"),
        "gate_saturation_ratio": _weighted_mean(rows, "gate_saturation_ratio"),
        "geometry_coupling_mean": _weighted_mean(rows, "geometry_coupling_mean"),
        "geometry_coupling_abs_mean": _weighted_mean(rows, "geometry_coupling_abs_mean"),
        "budget_scale_factor": _weighted_mean(rows, "budget_scale_factor"),
        "applied_window_count_total": int(sum(int(r.get("applied_window_count", 0)) for r in rows)),
        "fit_mode": str(train_result.summary.get("fit_mode", "")),
        "fit_target_mode": str(train_result.summary.get("fit_target_mode", "auto_associative")),
        "target_mode": str(train_result.summary.get("target_mode", "")),
        "response_stats_mode": str(train_result.summary.get("response_stats_mode", "fit_pool")),
        "template_count": int(train_result.summary.get("template_count", 1)),
        "template_readout_mode": str(train_result.summary.get("template_readout_mode", "first_row")),
        "weight_kernel_name": str(train_result.summary.get("weight_kernel_name", "")),
        "template_mean_direction_cosine": float(train_result.summary.get("template_mean_direction_cosine", 0.0)),
        "effective_sample_size": float(train_result.summary.get("effective_sample_size", 0.0)),
        "effective_sample_ratio": float(train_result.summary.get("effective_sample_ratio", 0.0)),
        "same_opp_count_ratio": float(train_result.summary.get("same_opp_count_ratio", 0.0)),
        "same_opp_weight_mass_ratio": float(train_result.summary.get("same_opp_weight_mass_ratio", 0.0)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="SCP-Branch P0a.1 B3 continuous geometric coupling smoke")
    p.add_argument("--datasets", type=str, default="natops")
    p.add_argument("--seeds", type=str, default="1")
    p.add_argument("--out-root", type=str, default="out/_active/verify_route_b_pia_operator_p0a1_b3_20260401_smoke")
    p.add_argument("--r-dimensions", type=str, default="1,4")
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--prop-win-ratio", type=float, default=0.20)
    p.add_argument("--prop-hop-ratio", type=float, default=0.10)
    p.add_argument("--min-window-extra-channels", type=int, default=4)
    p.add_argument("--min-hop-len", type=int, default=4)
    p.add_argument("--force-hop-len", type=int, default=1)
    p.add_argument("--terminal", type=str, default="dynamic_gru")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--gru-hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--minirocket-n-kernels", type=int, default=10000)
    p.add_argument("--minirocket-n-jobs", type=int, default=1)
    p.add_argument("--prototype-count", type=int, default=4)
    p.add_argument("--anchors-per-prototype", type=int, default=8)
    p.add_argument("--same-dist-quantile", type=float, default=50.0)
    p.add_argument("--anchor-selection-mode", type=str, default="tight_margin")
    p.add_argument("--center-epsilon-scale", type=float, default=0.10)
    p.add_argument("--pia-epsilon-scale", type=float, default=0.10)
    p.add_argument("--operator-smooth-lambda", type=float, default=0.50)
    p.add_argument("--pia-activation", type=str, default="sigmoid")
    p.add_argument("--pia-n-iters", type=int, default=3)
    p.add_argument("--pia-c-repr", type=float, default=1.0)
    p.add_argument("--pia-bias-lr", type=float, default=0.25)
    p.add_argument("--pia-bias-update-mode", type=str, default="residual")
    p.add_argument("--c3-opp-pair-rule", type=str, default="nearest_opposite_prototype")
    args = p.parse_args()

    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_int_csv(args.seeds)
    r_dimensions = _parse_int_csv(args.r_dimensions)
    _ensure_dir(args.out_root)

    config_rows: List[Dict[str, object]] = []
    per_seed_rows: List[Dict[str, object]] = []
    score_rows_all: List[Dict[str, object]] = []
    response_rows_all: List[Dict[str, object]] = []

    for dataset in datasets:
        for seed in seeds:
            state = _build_dense_state(args, dataset, seed)
            seed_dir = os.path.join(args.out_root, f"{dataset}_seed{seed}")
            _ensure_dir(seed_dir)

            geometry = build_fixed_reference_geometry(
                train_tids=state.train.tids.tolist(),
                train_labels=state.train.y.tolist(),
                train_z_seq_list=state.train.z_seq_list,
                cfg=FixedReferenceGeometryConfig(
                    prototype_count=int(args.prototype_count),
                    anchors_per_prototype=int(args.anchors_per_prototype),
                    same_dist_quantile=float(args.same_dist_quantile),
                    anchor_selection_mode=str(args.anchor_selection_mode),
                    seed=int(seed),
                ),
            )
            before_diag = _build_structure_diag(
                tids=state.train.tids.tolist(),
                labels=state.train.y.tolist(),
                z_seq_list=state.train.z_seq_list,
                prototype_count=int(args.prototype_count),
                seed=int(seed),
            )

            arm_a_train = apply_center_based_operator(
                z_seq_list=state.train.z_seq_list,
                geometry=geometry,
                cfg=CenterBasedOperatorConfig(
                    epsilon_scale=float(args.center_epsilon_scale),
                    smooth_lambda=float(args.operator_smooth_lambda),
                ),
            )
            arm_a_val = apply_center_based_operator(
                z_seq_list=state.val.z_seq_list,
                geometry=geometry,
                cfg=CenterBasedOperatorConfig(
                    epsilon_scale=float(args.center_epsilon_scale),
                    smooth_lambda=float(args.operator_smooth_lambda),
                ),
            )
            arm_a_test = apply_center_based_operator(
                z_seq_list=state.test.z_seq_list,
                geometry=geometry,
                cfg=CenterBasedOperatorConfig(
                    epsilon_scale=float(args.center_epsilon_scale),
                    smooth_lambda=float(args.operator_smooth_lambda),
                ),
            )
            arm_a_state = _clone_state_with_zseqs(
                state,
                train_z_seq_list=arm_a_train.z_seq_list,
                val_z_seq_list=arm_a_val.z_seq_list,
                test_z_seq_list=arm_a_test.z_seq_list,
            )

            pia_cfg_base = SingleTemplatePIAValueConfig(
                r_dimension=1,
                n_iters=int(args.pia_n_iters),
                C_repr=float(args.pia_c_repr),
                activation=str(args.pia_activation),
                bias_lr=float(args.pia_bias_lr),
                bias_update_mode=str(args.pia_bias_update_mode),
                epsilon_scale=float(args.pia_epsilon_scale),
                smooth_lambda=float(args.operator_smooth_lambda),
                fit_mode="unweighted",
                template_readout_mode="mean_committee",
                seed=int(seed),
            )
            op_c0 = fit_single_template_pia_operator(geometry=geometry, cfg=replace(pia_cfg_base, fit_mode="unweighted", r_dimension=1))

            b0_train = apply_single_template_pia_stage_a_variant(
                z_seq_list=state.train.z_seq_list,
                operator=op_c0,
                cfg=SingleTemplatePIAStageARepairConfig(
                    variant="current_sigmoid_minimal",
                    epsilon_scale=float(args.pia_epsilon_scale),
                    smooth_lambda=float(args.operator_smooth_lambda),
                ),
            )
            budget_target = float(b0_train.summary["operator_to_step_ratio_mean"])

            def _run_global_a2r(operator, arm_name: str):
                train_res = apply_single_template_pia_stage_a_variant(
                    z_seq_list=state.train.z_seq_list,
                    operator=operator,
                    cfg=SingleTemplatePIAStageARepairConfig(
                        variant="sigmoid_clip_tanh_local_median_scaled_iqr",
                        epsilon_scale=float(args.pia_epsilon_scale),
                        smooth_lambda=float(args.operator_smooth_lambda),
                        budget_target_operator_to_step_ratio=float(budget_target),
                    ),
                )
                budget_scale = float(train_res.summary.get("budget_scale_factor", 1.0))
                val_res = apply_single_template_pia_stage_a_variant(
                    z_seq_list=state.val.z_seq_list,
                    operator=operator,
                    cfg=SingleTemplatePIAStageARepairConfig(
                        variant="sigmoid_clip_tanh_local_median_scaled_iqr",
                        epsilon_scale=float(args.pia_epsilon_scale),
                        smooth_lambda=float(args.operator_smooth_lambda),
                        budget_scale_factor=float(budget_scale),
                    ),
                )
                test_res = apply_single_template_pia_stage_a_variant(
                    z_seq_list=state.test.z_seq_list,
                    operator=operator,
                    cfg=SingleTemplatePIAStageARepairConfig(
                        variant="sigmoid_clip_tanh_local_median_scaled_iqr",
                        epsilon_scale=float(args.pia_epsilon_scale),
                        smooth_lambda=float(args.operator_smooth_lambda),
                        budget_scale_factor=float(budget_scale),
                    ),
                )
                summary = _merge_apply_summaries(arm_name, train_result=train_res, val_result=val_res, test_result=test_res)
                shaped_state = _clone_state_with_zseqs(
                    state,
                    train_z_seq_list=train_res.z_seq_list,
                    val_z_seq_list=val_res.z_seq_list,
                    test_z_seq_list=test_res.z_seq_list,
                )
                after_diag = _build_structure_diag(
                    tids=state.train.tids.tolist(),
                    labels=state.train.y.tolist(),
                    z_seq_list=shaped_state.train.z_seq_list,
                    prototype_count=int(args.prototype_count),
                    seed=int(seed),
                )
                return train_res, val_res, test_res, summary, shaped_state, after_diag

            def _run_b3(operator, arm_name: str):
                train_res = apply_continuous_geometric_coupling(
                    z_seq_list=state.train.z_seq_list,
                    operator=operator,
                    cfg=ContinuousGeometricCouplingConfig(
                        epsilon_scale=float(args.pia_epsilon_scale),
                        smooth_lambda=float(args.operator_smooth_lambda),
                        response_variant="sigmoid_clip_tanh_local_median_scaled_iqr",
                        budget_target_operator_to_step_ratio=float(budget_target),
                    ),
                )
                budget_scale = float(train_res.summary.get("budget_scale_factor", 1.0))
                val_res = apply_continuous_geometric_coupling(
                    z_seq_list=state.val.z_seq_list,
                    operator=operator,
                    cfg=ContinuousGeometricCouplingConfig(
                        epsilon_scale=float(args.pia_epsilon_scale),
                        smooth_lambda=float(args.operator_smooth_lambda),
                        response_variant="sigmoid_clip_tanh_local_median_scaled_iqr",
                        budget_scale_factor=float(budget_scale),
                    ),
                )
                test_res = apply_continuous_geometric_coupling(
                    z_seq_list=state.test.z_seq_list,
                    operator=operator,
                    cfg=ContinuousGeometricCouplingConfig(
                        epsilon_scale=float(args.pia_epsilon_scale),
                        smooth_lambda=float(args.operator_smooth_lambda),
                        response_variant="sigmoid_clip_tanh_local_median_scaled_iqr",
                        budget_scale_factor=float(budget_scale),
                    ),
                )
                summary = _merge_apply_summaries(arm_name, train_result=train_res, val_result=val_res, test_result=test_res)
                shaped_state = _clone_state_with_zseqs(
                    state,
                    train_z_seq_list=train_res.z_seq_list,
                    val_z_seq_list=val_res.z_seq_list,
                    test_z_seq_list=test_res.z_seq_list,
                )
                after_diag = _build_structure_diag(
                    tids=state.train.tids.tolist(),
                    labels=state.train.y.tolist(),
                    z_seq_list=shaped_state.train.z_seq_list,
                    prototype_count=int(args.prototype_count),
                    seed=int(seed),
                )
                return train_res, val_res, test_res, summary, shaped_state, after_diag

            base_train, base_val, base_test, base_meta = _evaluate_terminal(state, seed=int(seed), args=args)[1:]
            arm_a_train_metrics, arm_a_val_metrics, arm_a_test_metrics, arm_a_meta = _evaluate_terminal(arm_a_state, seed=int(seed), args=args)[1:]

            arm_results: Dict[str, Dict[str, object]] = {}
            train_deploy_windows = np.concatenate([np.asarray(v, dtype=np.float64) for v in state.train.z_seq_list], axis=0).astype(np.float64)
            for r_dim in r_dimensions:
                operator = fit_single_template_pia_operator_discriminative(
                    geometry=geometry,
                    cfg=SingleTemplatePIADiscriminativeConfig(
                        r_dimension=int(r_dim),
                        n_iters=int(args.pia_n_iters),
                        C_repr=float(args.pia_c_repr),
                        activation=str(args.pia_activation),
                        bias_lr=float(args.pia_bias_lrr) if hasattr(args, "pia_bias_lrr") else float(args.pia_bias_lr),
                        bias_update_mode=str(args.pia_bias_update_mode),
                        target_mode="linear_pm1",
                        template_readout_mode="mean_committee",
                        opp_pair_rule=str(args.c3_opp_pair_rule),
                        seed=int(seed),
                    ),
                    response_stats_windows=train_deploy_windows,
                    response_stats_mode="deployment_train",
                )
                g_train, g_val, g_test, g_summary, g_state, after_g_diag = _run_global_a2r(operator, f"c3lr_r{int(r_dim)}_global_a2r")
                b_train, b_val, b_test, b_summary, b_state, after_b_diag = _run_b3(operator, f"b3_r{int(r_dim)}_continuous_geom")
                g_train_metrics, g_val_metrics, g_test_metrics, g_meta = _evaluate_terminal(g_state, seed=int(seed), args=args)[1:]
                b_train_metrics, b_val_metrics, b_test_metrics, b_meta = _evaluate_terminal(b_state, seed=int(seed), args=args)[1:]
                arm_results[f"c3lr_r{int(r_dim)}_global_a2r"] = {
                    "train_res": g_train,
                    "val_res": g_val,
                    "test_res": g_test,
                    "summary": g_summary,
                    "train_metrics": g_train_metrics,
                    "val_metrics": g_val_metrics,
                    "test_metrics": g_test_metrics,
                    "meta": g_meta,
                    "after_diag": after_g_diag,
                    "template_count": int(r_dim),
                }
                arm_results[f"b3_r{int(r_dim)}_continuous_geom"] = {
                    "train_res": b_train,
                    "val_res": b_val,
                    "test_res": b_test,
                    "summary": b_summary,
                    "train_metrics": b_train_metrics,
                    "val_metrics": b_val_metrics,
                    "test_metrics": b_test_metrics,
                    "meta": b_meta,
                    "after_diag": after_b_diag,
                    "template_count": int(r_dim),
                }

            c0_train, c0_val, c0_test, c0_summary, c0_state, after_c0_diag = _run_global_a2r(op_c0, "c0_unweighted_a2r")
            c0_train_metrics, c0_val_metrics, c0_test_metrics, c0_meta = _evaluate_terminal(c0_state, seed=int(seed), args=args)[1:]

            result_json = {
                "baseline_0": {"test_macro_f1": float(base_test["macro_f1"]), "meta": dict(base_meta)},
                "mean_centered": {"test_macro_f1": float(arm_a_test_metrics["macro_f1"]), "meta": dict(arm_a_meta)},
                "c0_unweighted_a2r": {"test_macro_f1": float(c0_test_metrics["macro_f1"]), "summary": dict(c0_summary), "meta": dict(c0_meta)},
            }
            for arm_name, payload in arm_results.items():
                result_json[str(arm_name)] = {
                    "test_macro_f1": float(payload["test_metrics"]["macro_f1"]),
                    "summary": dict(payload["summary"]),
                    "meta": dict(payload["meta"]),
                }
            _write_json(os.path.join(seed_dir, "pia_operator_p0a1_b3_result.json"), result_json)

            row = {
                "dataset": str(dataset),
                "seed": int(seed),
                "terminal": str(args.terminal),
                "same_backbone_no_shaping_test_macro_f1": float(base_test["macro_f1"]),
                "mean_centered_test_macro_f1": float(arm_a_test_metrics["macro_f1"]),
                "c0_unweighted_a2r_test_macro_f1": float(c0_test_metrics["macro_f1"]),
                "c0_response_vs_margin_correlation": float(c0_summary["response_vs_margin_correlation"]),
                "c0_template_mean_direction_cosine": float(c0_summary["template_mean_direction_cosine"]),
                "c0_geometry_coupling_abs_mean": float(c0_summary.get("geometry_coupling_abs_mean", 0.0)),
            }
            for arm_name, payload in arm_results.items():
                summary = payload["summary"]
                test_metrics = payload["test_metrics"]
                after_diag = payload["after_diag"]
                row[f"{arm_name}_test_macro_f1"] = float(test_metrics["macro_f1"])
                row[f"{arm_name}_delta_vs_c0"] = float(test_metrics["macro_f1"] - c0_test_metrics["macro_f1"])
                row[f"{arm_name}_response_vs_margin_correlation"] = float(summary["response_vs_margin_correlation"])
                row[f"{arm_name}_template_mean_direction_cosine"] = float(summary["template_mean_direction_cosine"])
                row[f"{arm_name}_geometry_coupling_abs_mean"] = float(summary.get("geometry_coupling_abs_mean", 0.0))
                row[f"{arm_name}_template_count"] = int(summary.get("template_count", payload["template_count"]))
                row[f"{arm_name}_delta_margin"] = float(after_diag["nearest_prototype_margin"] - before_diag["nearest_prototype_margin"])
                row[f"{arm_name}_margin_gain_per_unit_distortion"] = float(
                    (after_diag["nearest_prototype_margin"] - before_diag["nearest_prototype_margin"]) / max(1e-6, float(summary["operator_to_step_ratio_mean"]))
                )
            per_seed_rows.append(row)

            score_rows_all.extend(
                [
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "arm": "baseline_0",
                        "terminal": str(args.terminal),
                        "train_macro_f1": float(base_train["macro_f1"]),
                        "val_macro_f1": float(base_val["macro_f1"]),
                        "test_macro_f1": float(base_test["macro_f1"]),
                    },
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "arm": "mean_centered",
                        "terminal": str(args.terminal),
                        "train_macro_f1": float(arm_a_train_metrics["macro_f1"]),
                        "val_macro_f1": float(arm_a_val_metrics["macro_f1"]),
                        "test_macro_f1": float(arm_a_test_metrics["macro_f1"]),
                    },
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "arm": "c0_unweighted_a2r",
                        "terminal": str(args.terminal),
                        "train_macro_f1": float(c0_train_metrics["macro_f1"]),
                        "val_macro_f1": float(c0_val_metrics["macro_f1"]),
                        "test_macro_f1": float(c0_test_metrics["macro_f1"]),
                    },
                ]
            )
            response_rows_all.append({"dataset": str(dataset), "seed": int(seed), **dict(c0_summary)})
            for arm_name, payload in arm_results.items():
                score_rows_all.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "arm": str(arm_name),
                        "terminal": str(args.terminal),
                        "train_macro_f1": float(payload["train_metrics"]["macro_f1"]),
                        "val_macro_f1": float(payload["val_metrics"]["macro_f1"]),
                        "test_macro_f1": float(payload["test_metrics"]["macro_f1"]),
                    }
                )
                response_rows_all.append({"dataset": str(dataset), "seed": int(seed), **dict(payload["summary"])})

            config_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "terminal": str(args.terminal),
                    "window_len": int(state.window_len),
                    "hop_len": int(state.hop_len),
                    "z_dim": int(state.z_dim),
                    "prototype_count": int(args.prototype_count),
                    "anchors_per_prototype": int(args.anchors_per_prototype),
                    "r_dimensions": ",".join(str(v) for v in r_dimensions),
                    "fit_window_count": int(geometry.fit_window_count),
                    "fit_trial_count": int(geometry.fit_trial_count),
                    "stage_b_response_variant": "sigmoid_clip_tanh_local_median_scaled_iqr",
                    "c3_target_mode": "linear_pm1",
                    "c3_response_stats_mode": "deployment_train",
                    "c3_opp_pair_rule": str(args.c3_opp_pair_rule),
                }
            )

    config_df = pd.DataFrame(config_rows)
    per_seed_df = pd.DataFrame(per_seed_rows)
    score_df = pd.DataFrame(score_rows_all)
    response_df = pd.DataFrame(response_rows_all)

    config_csv = os.path.join(args.out_root, "pia_operator_p0a1_b3_config_table.csv")
    per_seed_csv = os.path.join(args.out_root, "pia_operator_p0a1_b3_per_seed.csv")
    score_csv = os.path.join(args.out_root, "pia_operator_p0a1_b3_score_diagnostics.csv")
    response_csv = os.path.join(args.out_root, "pia_operator_p0a1_b3_response_diagnostics.csv")
    conclusion_md = os.path.join(args.out_root, "pia_operator_p0a1_b3_conclusion.md")

    config_df.to_csv(config_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    score_df.to_csv(score_csv, index=False)
    response_df.to_csv(response_csv, index=False)

    lines: List[str] = [
        "# P0a.1 B3 Conclusion",
        "",
        "更新时间：2026-04-01",
        "",
        "当前比较固定 `A2r` 前端与 `C3LR` 判别目标链，只推进两件事：",
        "- `B3 continuous geometric coupling`",
        "- `r_dimension` 从 `1` 解放到更高模板数，并通过 `mean_committee` 读出接入同一条闭式解链",
        "",
    ]
    for dataset in datasets:
        ds = per_seed_df[per_seed_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(f"- `same_backbone_no_shaping`: {_format_mean_std(ds['same_backbone_no_shaping_test_macro_f1'].tolist())}")
        lines.append(f"- `mean_centered`: {_format_mean_std(ds['mean_centered_test_macro_f1'].tolist())}")
        lines.append(f"- `c0_unweighted_a2r`: {_format_mean_std(ds['c0_unweighted_a2r_test_macro_f1'].tolist())}")
        for r_dim in r_dimensions:
            lines.append(f"- `c3lr_r{int(r_dim)}_global_a2r`: {_format_mean_std(ds[f'c3lr_r{int(r_dim)}_global_a2r_test_macro_f1'].tolist())}")
            lines.append(f"- `b3_r{int(r_dim)}_continuous_geom`: {_format_mean_std(ds[f'b3_r{int(r_dim)}_continuous_geom_test_macro_f1'].tolist())}")
            lines.append(f"- `b3_r{int(r_dim)}_continuous_geom_template_mean_direction_cosine`: {_format_mean_std(ds[f'b3_r{int(r_dim)}_continuous_geom_template_mean_direction_cosine'].tolist())}")
            lines.append(f"- `b3_r{int(r_dim)}_continuous_geom_response_vs_margin_correlation`: {_format_mean_std(ds[f'b3_r{int(r_dim)}_continuous_geom_response_vs_margin_correlation'].tolist())}")
            lines.append(f"- `b3_r{int(r_dim)}_continuous_geom_geometry_coupling_abs_mean`: {_format_mean_std(ds[f'b3_r{int(r_dim)}_continuous_geom_geometry_coupling_abs_mean'].tolist())}")
        lines.append("")

    with open(conclusion_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(f"[pia-operator-b3] wrote {config_csv}")
    print(f"[pia-operator-b3] wrote {per_seed_csv}")
    print(f"[pia-operator-b3] wrote {score_csv}")
    print(f"[pia-operator-b3] wrote {response_csv}")
    print(f"[pia-operator-b3] wrote {conclusion_md}")


if __name__ == "__main__":
    main()
