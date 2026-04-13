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

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from route_b_unified.pia_operator_value_probe import (  # noqa: E402
    CenterBasedOperatorConfig,
    FixedReferenceGeometryConfig,
    OperatorApplyResult,
    SingleTemplatePIAStageARepairConfig,
    SingleTemplatePIAValueConfig,
    apply_center_based_operator,
    apply_single_template_pia_stage_a_variant,
    build_fixed_reference_geometry,
    fit_single_template_pia_operator,
)
from route_b_unified.scp_prototype_memory import (  # noqa: E402
    SCPPrototypeMemoryConfig,
    build_scp_prototype_memory,
)
from route_b_unified.trajectory_feedback_pool_windows import (  # noqa: E402
    build_window_feedback_reference_stats,
)
from route_b_unified.trajectory_evaluator import (  # noqa: E402
    TrajectoryEvalConfig,
    TrajectoryEvalResult,
    evaluate_trajectory_classifier,
)
from route_b_unified.trajectory_classifier import (  # noqa: E402
    TrajectoryModelConfig,
)
from route_b_unified.trajectory_minirocket_evaluator import (  # noqa: E402
    TrajectoryMiniRocketEvalConfig,
    evaluate_dynamic_minirocket_classifier,
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
        return (
            terminal,
            dict(result.train_metrics),
            dict(result.val_metrics),
            dict(result.test_metrics),
            dict(result.meta),
        )
    if terminal == "dynamic_gru":
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
        return (
            terminal,
            dict(result.train_metrics),
            dict(result.val_metrics),
            dict(result.test_metrics),
            {**dict(result.meta), "best_epoch": int(result.best_epoch)},
        )
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
        "budget_scale_factor": _weighted_mean(rows, "budget_scale_factor"),
        "applied_window_count_total": int(sum(int(r.get("applied_window_count", 0)) for r in rows)),
        "fit_mode": str(train_result.summary.get("fit_mode", "")),
        "weight_kernel_name": str(train_result.summary.get("weight_kernel_name", "")),
        "template_mean_direction_cosine": float(train_result.summary.get("template_mean_direction_cosine", 0.0)),
        "effective_sample_size": float(train_result.summary.get("effective_sample_size", 0.0)),
        "effective_sample_ratio": float(train_result.summary.get("effective_sample_ratio", 0.0)),
        "min_proto_effective_sample_size": float(train_result.summary.get("min_proto_effective_sample_size", 0.0)),
        "median_proto_effective_sample_size": float(train_result.summary.get("median_proto_effective_sample_size", 0.0)),
        "fit_anchor_margin_mean": float(train_result.summary.get("fit_anchor_margin_mean", 0.0)),
        "fit_anchor_same_dist_mean": float(train_result.summary.get("fit_anchor_same_dist_mean", 0.0)),
        "proto_weight_scale_mean": float(train_result.summary.get("proto_weight_scale_mean", 0.0)),
        "proto_weight_scale_min": float(train_result.summary.get("proto_weight_scale_min", 0.0)),
    }


def _anchor_diag_rows(dataset: str, seed: int, geometry) -> List[Dict[str, object]]:
    rows = list(geometry.anchor_rows)
    if not rows:
        return []
    by_proto: Dict[tuple[int, int], List[Dict[str, object]]] = {}
    for row in rows:
        key = (int(row["class_id"]), int(row["prototype_id"]))
        by_proto.setdefault(key, []).append(row)
    out: List[Dict[str, object]] = []
    for (class_id, prototype_id), proto_rows in by_proto.items():
        margins = np.asarray([float(r["admitted_margin_before"]) for r in proto_rows], dtype=np.float64)
        same_dists = np.asarray([float(r["admitted_same_dist_before"]) for r in proto_rows], dtype=np.float64)
        out.append(
            {
                "dataset": str(dataset),
                "seed": int(seed),
                "arm": "",
                "class_id": int(class_id),
                "prototype_id": int(prototype_id),
                "fit_anchor_count": int(len(proto_rows)),
                "fit_anchor_margin_mean": float(np.mean(margins)) if margins.size else 0.0,
                "fit_anchor_same_dist_mean": float(np.mean(same_dists)) if same_dists.size else 0.0,
                "anchor_coverage_ratio_mean": float(np.mean([float(r["anchor_coverage_ratio"]) for r in proto_rows])),
                "weight_mean": np.nan,
                "weight_min": np.nan,
                "weight_max": np.nan,
                "effective_sample_size": np.nan,
                "effective_sample_ratio": np.nan,
                "weight_scale": np.nan,
                "weight_kernel_name": "",
                "fallback_unweighted": np.nan,
            }
        )
    return out


def _weighted_anchor_diag_rows(dataset: str, seed: int, arm_name: str, operator) -> List[Dict[str, object]]:
    rows = list(operator.meta.get("prototype_weight_rows", []))
    if not rows:
        return []
    out: List[Dict[str, object]] = []
    for row in rows:
        out.append(
            {
                "dataset": str(dataset),
                "seed": int(seed),
                "arm": str(arm_name),
                "class_id": int(row["class_id"]),
                "prototype_id": int(row["prototype_id"]),
                "fit_anchor_count": int(row["fit_anchor_count"]),
                "fit_anchor_margin_mean": float(row["fit_anchor_margin_mean"]),
                "fit_anchor_same_dist_mean": float(row["fit_anchor_same_dist_mean"]),
                "anchor_coverage_ratio_mean": np.nan,
                "weight_mean": float(row["weight_mean"]),
                "weight_min": float(row["weight_min"]),
                "weight_max": float(row["weight_max"]),
                "effective_sample_size": float(row["effective_sample_size"]),
                "effective_sample_ratio": float(row.get("effective_sample_ratio", 0.0)),
                "weight_scale": float(row.get("weight_scale", 0.0)),
                "weight_kernel_name": str(row.get("weight_kernel_name", "")),
                "fallback_unweighted": bool(row.get("fallback_unweighted", False)),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="SCP-Branch P0a.1 Stage C_next lambda geometry repair probe")
    p.add_argument("--datasets", type=str, default="natops")
    p.add_argument("--seeds", type=str, default="1")
    p.add_argument("--out-root", type=str, default="out/_active/verify_route_b_pia_operator_p0a1_cnext_20260401_smoke")
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
    args = p.parse_args()

    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_seed_list(args.seeds)
    _ensure_dir(args.out_root)

    config_rows: List[Dict[str, object]] = []
    per_seed_rows: List[Dict[str, object]] = []
    structure_rows_all: List[Dict[str, object]] = []
    score_rows_all: List[Dict[str, object]] = []
    response_rows_all: List[Dict[str, object]] = []
    anchor_rows_all: List[Dict[str, object]] = []

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
            anchor_rows_all.extend(_anchor_diag_rows(dataset, seed, geometry))

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
            after_a_diag = _build_structure_diag(
                tids=state.train.tids.tolist(),
                labels=state.train.y.tolist(),
                z_seq_list=arm_a_state.train.z_seq_list,
                prototype_count=int(args.prototype_count),
                seed=int(seed),
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
                seed=int(seed),
            )

            op_c0 = fit_single_template_pia_operator(geometry=geometry, cfg=replace(pia_cfg_base, fit_mode="unweighted"))
            op_c1 = fit_single_template_pia_operator(geometry=geometry, cfg=replace(pia_cfg_base, fit_mode="mean_dist_weighted"))
            op_c2 = fit_single_template_pia_operator(geometry=geometry, cfg=replace(pia_cfg_base, fit_mode="median_min_weighted"))
            anchor_rows_all.extend(_weighted_anchor_diag_rows(dataset, seed, "c0_unweighted_a2r", op_c0))
            anchor_rows_all.extend(_weighted_anchor_diag_rows(dataset, seed, "c1_mean_dist_weighted_a2r", op_c1))
            anchor_rows_all.extend(_weighted_anchor_diag_rows(dataset, seed, "c2_median_min_weighted_a2r", op_c2))

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

            def _run_a2r_arm(operator, arm_name: str):
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

            c0_train, c0_val, c0_test, c0_summary, c0_state, after_c0_diag = _run_a2r_arm(op_c0, "c0_unweighted_a2r")
            c1_train, c1_val, c1_test, c1_summary, c1_state, after_c1_diag = _run_a2r_arm(op_c1, "c1_mean_dist_weighted_a2r")
            c2_train, c2_val, c2_test, c2_summary, c2_state, after_c2_diag = _run_a2r_arm(op_c2, "c2_median_min_weighted_a2r")

            terminal_name, base_train, base_val, base_test, base_meta = _evaluate_terminal(state, seed=int(seed), args=args)
            _ta, arm_a_train_metrics, arm_a_val_metrics, arm_a_test_metrics, arm_a_meta = _evaluate_terminal(arm_a_state, seed=int(seed), args=args)
            _tc0, c0_train_metrics, c0_val_metrics, c0_test_metrics, c0_meta = _evaluate_terminal(c0_state, seed=int(seed), args=args)
            _tc1, c1_train_metrics, c1_val_metrics, c1_test_metrics, c1_meta = _evaluate_terminal(c1_state, seed=int(seed), args=args)
            _tc2, c2_train_metrics, c2_val_metrics, c2_test_metrics, c2_meta = _evaluate_terminal(c2_state, seed=int(seed), args=args)

            _write_json(
                os.path.join(seed_dir, "pia_operator_p0a1_cnext_result.json"),
                {
                    "baseline_0": {"terminal": str(terminal_name), "test_macro_f1": float(base_test["macro_f1"]), "meta": dict(base_meta)},
                    "arm_a_mean_centered": {"test_macro_f1": float(arm_a_test_metrics["macro_f1"]), "summary": dict(_merge_apply_summaries("mean_centered", train_result=arm_a_train, val_result=arm_a_val, test_result=arm_a_test)), "meta": dict(arm_a_meta)},
                    "c0_unweighted_a2r": {"test_macro_f1": float(c0_test_metrics["macro_f1"]), "summary": dict(c0_summary), "meta": dict(c0_meta)},
                    "c1_mean_dist_weighted_a2r": {"test_macro_f1": float(c1_test_metrics["macro_f1"]), "summary": dict(c1_summary), "meta": dict(c1_meta)},
                    "c2_median_min_weighted_a2r": {"test_macro_f1": float(c2_test_metrics["macro_f1"]), "summary": dict(c2_summary), "meta": dict(c2_meta)},
                    "structure_before": dict(before_diag),
                    "structure_after_arm_a": dict(after_a_diag),
                    "structure_after_c0": dict(after_c0_diag),
                    "structure_after_c1": dict(after_c1_diag),
                    "structure_after_c2": dict(after_c2_diag),
                },
            )

            per_seed_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "terminal": str(terminal_name),
                    "same_backbone_no_shaping_test_macro_f1": float(base_test["macro_f1"]),
                    "mean_centered_test_macro_f1": float(arm_a_test_metrics["macro_f1"]),
                    "c0_unweighted_a2r_test_macro_f1": float(c0_test_metrics["macro_f1"]),
                    "c1_mean_dist_weighted_a2r_test_macro_f1": float(c1_test_metrics["macro_f1"]),
                    "c2_median_min_weighted_a2r_test_macro_f1": float(c2_test_metrics["macro_f1"]),
                    "delta_c0_vs_no_shaping": float(c0_test_metrics["macro_f1"] - base_test["macro_f1"]),
                    "delta_c1_vs_no_shaping": float(c1_test_metrics["macro_f1"] - base_test["macro_f1"]),
                    "delta_c2_vs_no_shaping": float(c2_test_metrics["macro_f1"] - base_test["macro_f1"]),
                    "delta_c1_vs_c0": float(c1_test_metrics["macro_f1"] - c0_test_metrics["macro_f1"]),
                    "delta_c2_vs_c0": float(c2_test_metrics["macro_f1"] - c0_test_metrics["macro_f1"]),
                    "c0_response_vs_margin_correlation": float(c0_summary["response_vs_margin_correlation"]),
                    "c1_response_vs_margin_correlation": float(c1_summary["response_vs_margin_correlation"]),
                    "c2_response_vs_margin_correlation": float(c2_summary["response_vs_margin_correlation"]),
                    "c0_activation_coverage_ratio": float(c0_summary["activation_coverage_ratio"]),
                    "c1_activation_coverage_ratio": float(c1_summary["activation_coverage_ratio"]),
                    "c2_activation_coverage_ratio": float(c2_summary["activation_coverage_ratio"]),
                    "c0_template_mean_direction_cosine": float(c0_summary["template_mean_direction_cosine"]),
                    "c1_template_mean_direction_cosine": float(c1_summary["template_mean_direction_cosine"]),
                    "c2_template_mean_direction_cosine": float(c2_summary["template_mean_direction_cosine"]),
                    "c0_effective_sample_size": float(c0_summary["effective_sample_size"]),
                    "c1_effective_sample_size": float(c1_summary["effective_sample_size"]),
                    "c2_effective_sample_size": float(c2_summary["effective_sample_size"]),
                    "c0_effective_sample_ratio": float(c0_summary["effective_sample_ratio"]),
                    "c1_effective_sample_ratio": float(c1_summary["effective_sample_ratio"]),
                    "c2_effective_sample_ratio": float(c2_summary["effective_sample_ratio"]),
                    "c0_min_proto_effective_sample_size": float(c0_summary["min_proto_effective_sample_size"]),
                    "c1_min_proto_effective_sample_size": float(c1_summary["min_proto_effective_sample_size"]),
                    "c2_min_proto_effective_sample_size": float(c2_summary["min_proto_effective_sample_size"]),
                    "c0_median_proto_effective_sample_size": float(c0_summary["median_proto_effective_sample_size"]),
                    "c1_median_proto_effective_sample_size": float(c1_summary["median_proto_effective_sample_size"]),
                    "c2_median_proto_effective_sample_size": float(c2_summary["median_proto_effective_sample_size"]),
                    "c0_proto_weight_scale_mean": float(c0_summary["proto_weight_scale_mean"]),
                    "c1_proto_weight_scale_mean": float(c1_summary["proto_weight_scale_mean"]),
                    "c2_proto_weight_scale_mean": float(c2_summary["proto_weight_scale_mean"]),
                    "c0_proto_weight_scale_min": float(c0_summary["proto_weight_scale_min"]),
                    "c1_proto_weight_scale_min": float(c1_summary["proto_weight_scale_min"]),
                    "c2_proto_weight_scale_min": float(c2_summary["proto_weight_scale_min"]),
                    "c0_weight_kernel_name": str(c0_summary["weight_kernel_name"]),
                    "c1_weight_kernel_name": str(c1_summary["weight_kernel_name"]),
                    "c2_weight_kernel_name": str(c2_summary["weight_kernel_name"]),
                    "delta_c0_margin": float(after_c0_diag["nearest_prototype_margin"] - before_diag["nearest_prototype_margin"]),
                    "delta_c1_margin": float(after_c1_diag["nearest_prototype_margin"] - before_diag["nearest_prototype_margin"]),
                    "delta_c2_margin": float(after_c2_diag["nearest_prototype_margin"] - before_diag["nearest_prototype_margin"]),
                    "c0_margin_gain_per_unit_distortion": float(
                        (after_c0_diag["nearest_prototype_margin"] - before_diag["nearest_prototype_margin"]) / max(1e-6, float(c0_summary["operator_to_step_ratio_mean"]))
                    ),
                    "c1_margin_gain_per_unit_distortion": float(
                        (after_c1_diag["nearest_prototype_margin"] - before_diag["nearest_prototype_margin"]) / max(1e-6, float(c1_summary["operator_to_step_ratio_mean"]))
                    ),
                    "c2_margin_gain_per_unit_distortion": float(
                        (after_c2_diag["nearest_prototype_margin"] - before_diag["nearest_prototype_margin"]) / max(1e-6, float(c2_summary["operator_to_step_ratio_mean"]))
                    ),
                }
            )

            for stage_name, diag in [
                ("before", before_diag),
                ("after_mean_centered", after_a_diag),
                ("after_c0_unweighted_a2r", after_c0_diag),
                ("after_c1_mean_dist_weighted_a2r", after_c1_diag),
                ("after_c2_median_min_weighted_a2r", after_c2_diag),
            ]:
                structure_rows_all.append({"dataset": str(dataset), "seed": int(seed), "stage": str(stage_name), **dict(diag)})

            score_specs = [
                ("baseline_0", base_train, base_val, base_test),
                ("mean_centered", arm_a_train_metrics, arm_a_val_metrics, arm_a_test_metrics),
                ("c0_unweighted_a2r", c0_train_metrics, c0_val_metrics, c0_test_metrics),
                ("c1_mean_dist_weighted_a2r", c1_train_metrics, c1_val_metrics, c1_test_metrics),
                ("c2_median_min_weighted_a2r", c2_train_metrics, c2_val_metrics, c2_test_metrics),
            ]
            for arm_name, train_metrics, val_metrics, test_metrics in score_specs:
                score_rows_all.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "arm": str(arm_name),
                        "terminal": str(terminal_name),
                        "train_macro_f1": float(train_metrics["macro_f1"]),
                        "val_macro_f1": float(val_metrics["macro_f1"]),
                        "test_macro_f1": float(test_metrics["macro_f1"]),
                        "delta_vs_no_shaping": float(test_metrics["macro_f1"] - base_test["macro_f1"]),
                        "delta_vs_c0": float(test_metrics["macro_f1"] - c0_test_metrics["macro_f1"]) if arm_name != "c0_unweighted_a2r" else 0.0,
                    }
                )

            for summary in [c0_summary, c1_summary, c2_summary]:
                response_rows_all.append({"dataset": str(dataset), "seed": int(seed), **dict(summary)})

            config_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "terminal": str(terminal_name),
                    "window_len": int(state.window_len),
                    "hop_len": int(state.hop_len),
                    "z_dim": int(state.z_dim),
                    "prototype_count": int(args.prototype_count),
                    "anchors_per_prototype": int(args.anchors_per_prototype),
                    "anchor_selection_mode": str(args.anchor_selection_mode),
                    "same_dist_quantile": float(args.same_dist_quantile),
                    "center_epsilon_scale": float(args.center_epsilon_scale),
                    "pia_epsilon_scale": float(args.pia_epsilon_scale),
                    "operator_smooth_lambda": float(args.operator_smooth_lambda),
                    "pia_activation": str(args.pia_activation),
                    "pia_n_iters": int(args.pia_n_iters),
                    "pia_c_repr": float(args.pia_c_repr),
                    "pia_bias_lr": float(args.pia_bias_lr),
                    "pia_bias_update_mode": str(args.pia_bias_update_mode),
                    "fixed_stage_a_variant": "sigmoid_clip_tanh_local_median_scaled_iqr",
                    "fit_window_count": int(geometry.fit_window_count),
                    "fit_trial_count": int(geometry.fit_trial_count),
                    "preactivation_clip_lower": float(op_c0.preactivation_clip_lower),
                    "preactivation_clip_upper": float(op_c0.preactivation_clip_upper),
                    "response_scale": float(op_c0.response_scale),
                    "response_scale_iqr": float(op_c0.response_scale_iqr),
                }
            )

    per_seed_df = pd.DataFrame(per_seed_rows)
    structure_df = pd.DataFrame(structure_rows_all)
    score_df = pd.DataFrame(score_rows_all)
    response_df = pd.DataFrame(response_rows_all)
    anchor_df = pd.DataFrame(anchor_rows_all)
    config_df = pd.DataFrame(config_rows)

    config_csv = os.path.join(args.out_root, "pia_operator_p0a1_cnext_config_table.csv")
    per_seed_csv = os.path.join(args.out_root, "pia_operator_p0a1_cnext_per_seed.csv")
    structure_csv = os.path.join(args.out_root, "pia_operator_p0a1_cnext_structure_diagnostics.csv")
    score_csv = os.path.join(args.out_root, "pia_operator_p0a1_cnext_score_diagnostics.csv")
    response_csv = os.path.join(args.out_root, "pia_operator_p0a1_cnext_response_diagnostics.csv")
    anchor_csv = os.path.join(args.out_root, "pia_operator_p0a1_cnext_anchor_diagnostics.csv")
    conclusion_md = os.path.join(args.out_root, "pia_operator_p0a1_cnext_conclusion.md")

    config_df.to_csv(config_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    structure_df.to_csv(structure_csv, index=False)
    score_df.to_csv(score_csv, index=False)
    response_df.to_csv(response_csv, index=False)
    anchor_df.to_csv(anchor_csv, index=False)

    lines: List[str] = [
        "# P0a.1 C_next Conclusion",
        "",
        "更新时间：2026-04-01",
        "",
        "当前实现固定 `A2r` 前端，只比较阶段 C 的三条线：`C0 unweighted`、`C1 mean_dist_weighted`、`C2 median_min_weighted`。",
        "本轮仍保留 TELM2 的广义逆/闭式解主线；变化仅发生在样本度量矩阵 `Lambda`。",
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
        lines.append(f"- `c1_mean_dist_weighted_a2r`: {_format_mean_std(ds['c1_mean_dist_weighted_a2r_test_macro_f1'].tolist())}")
        lines.append(f"- `c2_median_min_weighted_a2r`: {_format_mean_std(ds['c2_median_min_weighted_a2r_test_macro_f1'].tolist())}")
        lines.append(f"- `c0_template_mean_direction_cosine`: {_format_mean_std(ds['c0_template_mean_direction_cosine'].tolist())}")
        lines.append(f"- `c1_template_mean_direction_cosine`: {_format_mean_std(ds['c1_template_mean_direction_cosine'].tolist())}")
        lines.append(f"- `c2_template_mean_direction_cosine`: {_format_mean_std(ds['c2_template_mean_direction_cosine'].tolist())}")
        lines.append(f"- `c0_effective_sample_ratio`: {_format_mean_std(ds['c0_effective_sample_ratio'].tolist())}")
        lines.append(f"- `c1_effective_sample_ratio`: {_format_mean_std(ds['c1_effective_sample_ratio'].tolist())}")
        lines.append(f"- `c2_effective_sample_ratio`: {_format_mean_std(ds['c2_effective_sample_ratio'].tolist())}")
        lines.append(f"- `c0_min_proto_effective_sample_size`: {_format_mean_std(ds['c0_min_proto_effective_sample_size'].tolist())}")
        lines.append(f"- `c1_min_proto_effective_sample_size`: {_format_mean_std(ds['c1_min_proto_effective_sample_size'].tolist())}")
        lines.append(f"- `c2_min_proto_effective_sample_size`: {_format_mean_std(ds['c2_min_proto_effective_sample_size'].tolist())}")
        lines.append("")

    with open(conclusion_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(f"[pia-operator-cnext] wrote {config_csv}")
    print(f"[pia-operator-cnext] wrote {per_seed_csv}")
    print(f"[pia-operator-cnext] wrote {structure_csv}")
    print(f"[pia-operator-cnext] wrote {score_csv}")
    print(f"[pia-operator-cnext] wrote {response_csv}")
    print(f"[pia-operator-cnext] wrote {anchor_csv}")
    print(f"[pia-operator-cnext] wrote {conclusion_md}")


if __name__ == "__main__":
    main()
