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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from route_b_unified.pia_operator_value_probe import (  # noqa: E402
    FixedReferenceGeometryConfig,
    OperatorApplyResult,
    SingleTemplatePIADiscriminativeConfig,
    SingleTemplatePIAStageARepairConfig,
    SingleTemplatePIAValueConfig,
    apply_single_template_pia_stage_a_variant,
    build_fixed_reference_geometry,
    fit_single_template_pia_operator,
    fit_single_template_pia_operator_discriminative,
    rebuild_fixed_reference_geometry_with_frozen_identities,
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
        "applied_window_count_total": int(sum(int(r.get("applied_window_count", 0)) for r in rows)),
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
        "same_pool_count": int(train_result.summary.get("same_pool_count", 0)),
        "opp_pool_count": int(train_result.summary.get("opp_pool_count", 0)),
        "same_weight_mass": float(train_result.summary.get("same_weight_mass", 0.0)),
        "opp_weight_mass": float(train_result.summary.get("opp_weight_mass", 0.0)),
        "discriminative_target_gap": float(train_result.summary.get("discriminative_target_gap", 0.0)),
        "opp_pair_rule": str(train_result.summary.get("opp_pair_rule", "")),
        "pool_mode": str(train_result.summary.get("pool_mode", "")),
    }


def _apply_a2r_to_state(
    *,
    base_state: TrajectoryRepresentationState,
    operator,
    budget_target: float,
    epsilon_scale: float,
    smooth_lambda: float,
    arm_name: str,
    prototype_count: int,
    seed: int,
) -> tuple[OperatorApplyResult, OperatorApplyResult, OperatorApplyResult, Dict[str, object], TrajectoryRepresentationState, Dict[str, float]]:
    train_res = apply_single_template_pia_stage_a_variant(
        z_seq_list=base_state.train.z_seq_list,
        operator=operator,
        cfg=SingleTemplatePIAStageARepairConfig(
            variant="sigmoid_clip_tanh_local_median_scaled_iqr",
            epsilon_scale=float(epsilon_scale),
            smooth_lambda=float(smooth_lambda),
            budget_target_operator_to_step_ratio=float(budget_target),
        ),
    )
    budget_scale = float(train_res.summary.get("budget_scale_factor", 1.0))
    val_res = apply_single_template_pia_stage_a_variant(
        z_seq_list=base_state.val.z_seq_list,
        operator=operator,
        cfg=SingleTemplatePIAStageARepairConfig(
            variant="sigmoid_clip_tanh_local_median_scaled_iqr",
            epsilon_scale=float(epsilon_scale),
            smooth_lambda=float(smooth_lambda),
            budget_scale_factor=float(budget_scale),
        ),
    )
    test_res = apply_single_template_pia_stage_a_variant(
        z_seq_list=base_state.test.z_seq_list,
        operator=operator,
        cfg=SingleTemplatePIAStageARepairConfig(
            variant="sigmoid_clip_tanh_local_median_scaled_iqr",
            epsilon_scale=float(epsilon_scale),
            smooth_lambda=float(smooth_lambda),
            budget_scale_factor=float(budget_scale),
        ),
    )
    summary = _merge_apply_summaries(arm_name, train_result=train_res, val_result=val_res, test_result=test_res)
    shaped_state = _clone_state_with_zseqs(
        base_state,
        train_z_seq_list=train_res.z_seq_list,
        val_z_seq_list=val_res.z_seq_list,
        test_z_seq_list=test_res.z_seq_list,
    )
    after_diag = _build_structure_diag(
        tids=shaped_state.train.tids.tolist(),
        labels=shaped_state.train.y.tolist(),
        z_seq_list=shaped_state.train.z_seq_list,
        prototype_count=int(prototype_count),
        seed=int(seed),
    )
    return train_res, val_res, test_res, summary, shaped_state, after_diag


def main() -> None:
    p = argparse.ArgumentParser(description="SCP-Branch P0b-lite one-step delayed refresh probe")
    p.add_argument("--datasets", type=str, default="natops")
    p.add_argument("--seeds", type=str, default="1")
    p.add_argument("--out-root", type=str, default="out/_active/verify_route_b_pia_operator_p0b_lite_20260401_smoke")
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
    p.add_argument("--pia-epsilon-scale", type=float, default=0.10)
    p.add_argument("--operator-smooth-lambda", type=float, default=0.50)
    p.add_argument("--pia-activation", type=str, default="sigmoid")
    p.add_argument("--pia-n-iters", type=int, default=3)
    p.add_argument("--pia-c-repr", type=float, default=1.0)
    p.add_argument("--pia-bias-lr", type=float, default=0.25)
    p.add_argument("--pia-bias-update-mode", type=str, default="residual")
    p.add_argument("--fast-r-dimension", type=int, default=4)
    p.add_argument("--c3-opp-pair-rule", type=str, default="nearest_opposite_prototype")
    args = p.parse_args()

    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_int_csv(args.seeds)
    _ensure_dir(args.out_root)

    config_rows: List[Dict[str, object]] = []
    per_seed_rows: List[Dict[str, object]] = []
    geometry_rows_all: List[Dict[str, object]] = []
    operator_rows_all: List[Dict[str, object]] = []
    score_rows_all: List[Dict[str, object]] = []
    response_rows_all: List[Dict[str, object]] = []

    for dataset in datasets:
        for seed in seeds:
            state = _build_dense_state(args, dataset, seed)
            seed_dir = os.path.join(args.out_root, f"{dataset}_seed{seed}")
            _ensure_dir(seed_dir)

            geometry_cfg = FixedReferenceGeometryConfig(
                prototype_count=int(args.prototype_count),
                anchors_per_prototype=int(args.anchors_per_prototype),
                same_dist_quantile=float(args.same_dist_quantile),
                anchor_selection_mode=str(args.anchor_selection_mode),
                seed=int(seed),
            )
            geometry0 = build_fixed_reference_geometry(
                train_tids=state.train.tids.tolist(),
                train_labels=state.train.y.tolist(),
                train_z_seq_list=state.train.z_seq_list,
                cfg=geometry_cfg,
            )
            before_diag = _build_structure_diag(
                tids=state.train.tids.tolist(),
                labels=state.train.y.tolist(),
                z_seq_list=state.train.z_seq_list,
                prototype_count=int(args.prototype_count),
                seed=int(seed),
            )

            op_c0 = fit_single_template_pia_operator(
                geometry=geometry0,
                cfg=SingleTemplatePIAValueConfig(
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
                ),
            )
            budget_probe = apply_single_template_pia_stage_a_variant(
                z_seq_list=state.train.z_seq_list,
                operator=op_c0,
                cfg=SingleTemplatePIAStageARepairConfig(
                    variant="current_sigmoid_minimal",
                    epsilon_scale=float(args.pia_epsilon_scale),
                    smooth_lambda=float(args.operator_smooth_lambda),
                ),
            )
            budget_target = float(budget_probe.summary["operator_to_step_ratio_mean"])

            _, base_train, base_val, base_test, base_meta = _evaluate_terminal(state, seed=int(seed), args=args)

            train_deploy_windows0 = np.concatenate([np.asarray(v, dtype=np.float64) for v in state.train.z_seq_list], axis=0).astype(np.float64)
            operator_f1 = fit_single_template_pia_operator_discriminative(
                geometry=geometry0,
                cfg=SingleTemplatePIADiscriminativeConfig(
                    r_dimension=int(args.fast_r_dimension),
                    n_iters=int(args.pia_n_iters),
                    C_repr=float(args.pia_c_repr),
                    activation=str(args.pia_activation),
                    bias_lr=float(args.pia_bias_lr),
                    bias_update_mode=str(args.pia_bias_update_mode),
                    target_mode="linear_pm1",
                    template_readout_mode="mean_committee",
                    opp_pair_rule=str(args.c3_opp_pair_rule),
                    seed=int(seed),
                ),
                response_stats_windows=train_deploy_windows0,
                response_stats_mode="deployment_train",
            )

            f1_train, f1_val, f1_test, f1_summary, state_f1, post_fast_diag = _apply_a2r_to_state(
                base_state=state,
                operator=operator_f1,
                budget_target=float(budget_target),
                epsilon_scale=float(args.pia_epsilon_scale),
                smooth_lambda=float(args.operator_smooth_lambda),
                arm_name="f1_fast_mainline",
                prototype_count=int(args.prototype_count),
                seed=int(seed),
            )
            _, f1_train_metrics, f1_val_metrics, f1_test_metrics, f1_meta = _evaluate_terminal(state_f1, seed=int(seed), args=args)

            geometry_r0 = rebuild_fixed_reference_geometry_with_frozen_identities(
                geometry=geometry0,
                train_tids=state_f1.train.tids.tolist(),
                train_z_seq_list=state_f1.train.z_seq_list,
            )
            train_deploy_windows1 = np.concatenate([np.asarray(v, dtype=np.float64) for v in state_f1.train.z_seq_list], axis=0).astype(np.float64)
            operator_r0 = fit_single_template_pia_operator_discriminative(
                geometry=geometry_r0,
                cfg=SingleTemplatePIADiscriminativeConfig(
                    r_dimension=int(args.fast_r_dimension),
                    n_iters=int(args.pia_n_iters),
                    C_repr=float(args.pia_c_repr),
                    activation=str(args.pia_activation),
                    bias_lr=float(args.pia_bias_lr),
                    bias_update_mode=str(args.pia_bias_update_mode),
                    target_mode="linear_pm1",
                    template_readout_mode="mean_committee",
                    opp_pair_rule=str(args.c3_opp_pair_rule),
                    seed=int(seed),
                ),
                response_stats_windows=train_deploy_windows1,
                response_stats_mode="deployment_train",
            )
            r0_train, r0_val, r0_test, r0_summary, state_r0, refit_only_diag = _apply_a2r_to_state(
                base_state=state_f1,
                operator=operator_r0,
                budget_target=float(budget_target),
                epsilon_scale=float(args.pia_epsilon_scale),
                smooth_lambda=float(args.operator_smooth_lambda),
                arm_name="r0_post_fast_refit_no_rebuild",
                prototype_count=int(args.prototype_count),
                seed=int(seed),
            )
            _, r0_train_metrics, r0_val_metrics, r0_test_metrics, r0_meta = _evaluate_terminal(state_r0, seed=int(seed), args=args)

            geometry_p0b = build_fixed_reference_geometry(
                train_tids=state_f1.train.tids.tolist(),
                train_labels=state_f1.train.y.tolist(),
                train_z_seq_list=state_f1.train.z_seq_list,
                cfg=geometry_cfg,
            )
            operator_p0b = fit_single_template_pia_operator_discriminative(
                geometry=geometry_p0b,
                cfg=SingleTemplatePIADiscriminativeConfig(
                    r_dimension=int(args.fast_r_dimension),
                    n_iters=int(args.pia_n_iters),
                    C_repr=float(args.pia_c_repr),
                    activation=str(args.pia_activation),
                    bias_lr=float(args.pia_bias_lr),
                    bias_update_mode=str(args.pia_bias_update_mode),
                    target_mode="linear_pm1",
                    template_readout_mode="mean_committee",
                    opp_pair_rule=str(args.c3_opp_pair_rule),
                    seed=int(seed),
                ),
                response_stats_windows=train_deploy_windows1,
                response_stats_mode="deployment_train",
            )
            p0b_train, p0b_val, p0b_test, p0b_summary, state_p0b, delayed_refresh_diag = _apply_a2r_to_state(
                base_state=state_f1,
                operator=operator_p0b,
                budget_target=float(budget_target),
                epsilon_scale=float(args.pia_epsilon_scale),
                smooth_lambda=float(args.operator_smooth_lambda),
                arm_name="p0b_lite_delayed_refresh",
                prototype_count=int(args.prototype_count),
                seed=int(seed),
            )
            _, p0b_train_metrics, p0b_val_metrics, p0b_test_metrics, p0b_meta = _evaluate_terminal(state_p0b, seed=int(seed), args=args)

            result_json = {
                "baseline_0": {"test_macro_f1": float(base_test["macro_f1"]), "meta": dict(base_meta)},
                "f1_fast_mainline": {"test_macro_f1": float(f1_test_metrics["macro_f1"]), "summary": dict(f1_summary), "meta": dict(f1_meta)},
                "r0_post_fast_refit_no_rebuild": {"test_macro_f1": float(r0_test_metrics["macro_f1"]), "summary": dict(r0_summary), "meta": dict(r0_meta)},
                "p0b_lite_delayed_refresh": {"test_macro_f1": float(p0b_test_metrics["macro_f1"]), "summary": dict(p0b_summary), "meta": dict(p0b_meta)},
            }
            _write_json(os.path.join(seed_dir, "pia_operator_p0b_lite_result.json"), result_json)

            geometry_rows_all.extend(
                [
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "stage": "before_fast",
                        **dict(before_diag),
                    },
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "stage": "post_fast_geometry",
                        **dict(post_fast_diag),
                    },
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "stage": "refit_only_geometry",
                        **dict(refit_only_diag),
                    },
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "stage": "delayed_refresh_geometry",
                        **dict(delayed_refresh_diag),
                    },
                ]
            )

            for arm_name, summary, train_metrics, val_metrics, test_metrics in [
                ("f1_fast_mainline", f1_summary, f1_train_metrics, f1_val_metrics, f1_test_metrics),
                ("r0_post_fast_refit_no_rebuild", r0_summary, r0_train_metrics, r0_val_metrics, r0_test_metrics),
                ("p0b_lite_delayed_refresh", p0b_summary, p0b_train_metrics, p0b_val_metrics, p0b_test_metrics),
            ]:
                operator_rows_all.append({"dataset": str(dataset), "seed": int(seed), **dict(summary)})
                response_rows_all.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "arm": str(arm_name),
                        "response_vs_margin_correlation": float(summary["response_vs_margin_correlation"]),
                        "activation_coverage_ratio": float(summary["activation_coverage_ratio"]),
                        "margin_gain_per_unit_distortion": float(
                            (
                                (post_fast_diag["nearest_prototype_margin"] - before_diag["nearest_prototype_margin"])
                                if arm_name == "f1_fast_mainline"
                                else (refit_only_diag["nearest_prototype_margin"] - before_diag["nearest_prototype_margin"])
                                if arm_name == "r0_post_fast_refit_no_rebuild"
                                else (delayed_refresh_diag["nearest_prototype_margin"] - before_diag["nearest_prototype_margin"])
                            )
                            / max(1e-6, float(summary["operator_to_step_ratio_mean"]))
                        ),
                        "template_mean_direction_cosine": float(summary["template_mean_direction_cosine"]),
                        "operator_to_step_ratio_mean": float(summary["operator_to_step_ratio_mean"]),
                        "preactivation_clip_rate": float(summary["preactivation_clip_rate"]),
                        "gate_saturation_ratio": float(summary["gate_saturation_ratio"]),
                        "budget_scale_factor": float(summary["budget_scale_factor"]),
                    }
                )
                score_rows_all.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "arm": str(arm_name),
                        "terminal": str(args.terminal),
                        "train_macro_f1": float(train_metrics["macro_f1"]),
                        "val_macro_f1": float(val_metrics["macro_f1"]),
                        "test_macro_f1": float(test_metrics["macro_f1"]),
                    }
                )
            score_rows_all.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "arm": "baseline_0",
                    "terminal": str(args.terminal),
                    "train_macro_f1": float(base_train["macro_f1"]),
                    "val_macro_f1": float(base_val["macro_f1"]),
                    "test_macro_f1": float(base_test["macro_f1"]),
                }
            )

            per_seed_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "terminal": str(args.terminal),
                    "baseline_0_test_macro_f1": float(base_test["macro_f1"]),
                    "f1_fast_mainline_test_macro_f1": float(f1_test_metrics["macro_f1"]),
                    "r0_post_fast_refit_no_rebuild_test_macro_f1": float(r0_test_metrics["macro_f1"]),
                    "p0b_lite_delayed_refresh_test_macro_f1": float(p0b_test_metrics["macro_f1"]),
                    "refit_only_delta_vs_f1": float(r0_test_metrics["macro_f1"] - f1_test_metrics["macro_f1"]),
                    "delayed_refresh_delta_vs_f1": float(p0b_test_metrics["macro_f1"] - f1_test_metrics["macro_f1"]),
                    "delayed_refresh_delta_vs_r0": float(p0b_test_metrics["macro_f1"] - r0_test_metrics["macro_f1"]),
                    "f1_response_vs_margin_correlation": float(f1_summary["response_vs_margin_correlation"]),
                    "r0_response_vs_margin_correlation": float(r0_summary["response_vs_margin_correlation"]),
                    "p0b_response_vs_margin_correlation": float(p0b_summary["response_vs_margin_correlation"]),
                    "f1_activation_coverage_ratio": float(f1_summary["activation_coverage_ratio"]),
                    "r0_activation_coverage_ratio": float(r0_summary["activation_coverage_ratio"]),
                    "p0b_activation_coverage_ratio": float(p0b_summary["activation_coverage_ratio"]),
                    "f1_template_mean_direction_cosine": float(f1_summary["template_mean_direction_cosine"]),
                    "r0_template_mean_direction_cosine": float(r0_summary["template_mean_direction_cosine"]),
                    "p0b_template_mean_direction_cosine": float(p0b_summary["template_mean_direction_cosine"]),
                    "second_pass_template_mean_direction_cosine": float(p0b_summary["template_mean_direction_cosine"]),
                    "post_fast_geometry_within_compactness": float(post_fast_diag["within_prototype_compactness"]),
                    "post_fast_geometry_between_separation": float(post_fast_diag["between_prototype_separation"]),
                    "post_fast_geometry_margin": float(post_fast_diag["nearest_prototype_margin"]),
                    "post_fast_geometry_temporal_stability": float(post_fast_diag["temporal_assignment_stability"]),
                    "delayed_refresh_within_compactness": float(delayed_refresh_diag["within_prototype_compactness"]),
                    "delayed_refresh_between_separation": float(delayed_refresh_diag["between_prototype_separation"]),
                    "delayed_refresh_margin": float(delayed_refresh_diag["nearest_prototype_margin"]),
                    "delayed_refresh_temporal_stability": float(delayed_refresh_diag["temporal_assignment_stability"]),
                    "geometry_rebuild_window_count": int(geometry_p0b.fit_window_count),
                    "geometry_rebuild_prototype_count": int(len(geometry_p0b.all_prototypes)),
                }
            )

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
                    "fast_mainline": "A2r+C3LR+r4+global",
                    "fast_r_dimension": int(args.fast_r_dimension),
                    "fit_window_count": int(geometry0.fit_window_count),
                    "fit_trial_count": int(geometry0.fit_trial_count),
                    "opp_pair_rule": str(args.c3_opp_pair_rule),
                    "response_stats_mode": "deployment_train",
                    "r0_geometry_rebuild_mode": "frozen_identity_post_fast_refit",
                }
            )

    config_df = pd.DataFrame(config_rows)
    per_seed_df = pd.DataFrame(per_seed_rows)
    geometry_df = pd.DataFrame(geometry_rows_all)
    operator_df = pd.DataFrame(operator_rows_all)
    score_df = pd.DataFrame(score_rows_all)
    response_df = pd.DataFrame(response_rows_all)

    config_csv = os.path.join(args.out_root, "pia_operator_p0b_lite_config_table.csv")
    per_seed_csv = os.path.join(args.out_root, "pia_operator_p0b_lite_per_seed.csv")
    geometry_csv = os.path.join(args.out_root, "pia_operator_p0b_lite_geometry_diagnostics.csv")
    operator_csv = os.path.join(args.out_root, "pia_operator_p0b_lite_operator_diagnostics.csv")
    score_csv = os.path.join(args.out_root, "pia_operator_p0b_lite_score_diagnostics.csv")
    response_csv = os.path.join(args.out_root, "pia_operator_p0b_lite_response_diagnostics.csv")
    conclusion_md = os.path.join(args.out_root, "pia_operator_p0b_lite_conclusion.md")

    config_df.to_csv(config_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    geometry_df.to_csv(geometry_csv, index=False)
    operator_df.to_csv(operator_csv, index=False)
    score_df.to_csv(score_csv, index=False)
    response_df.to_csv(response_csv, index=False)

    lines: List[str] = [
        "# P0b-lite Conclusion",
        "",
        "更新时间：2026-04-01",
        "",
        "当前固定快层主线：`A2r + C3LR + r4 + global readout`。",
        "本轮通过 `B0 / F1 / R0 / P0b-lite` 四臂对照，把“二次 refit 收益”和“真实 delayed rebuild 收益”分开。",
        "",
    ]
    for dataset in datasets:
        ds = per_seed_df[per_seed_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(f"- `baseline_0`: {_format_mean_std(ds['baseline_0_test_macro_f1'].tolist())}")
        lines.append(f"- `f1_fast_mainline`: {_format_mean_std(ds['f1_fast_mainline_test_macro_f1'].tolist())}")
        lines.append(f"- `r0_post_fast_refit_no_rebuild`: {_format_mean_std(ds['r0_post_fast_refit_no_rebuild_test_macro_f1'].tolist())}")
        lines.append(f"- `p0b_lite_delayed_refresh`: {_format_mean_std(ds['p0b_lite_delayed_refresh_test_macro_f1'].tolist())}")
        lines.append(f"- `refit_only_delta_vs_f1`: {_format_mean_std(ds['refit_only_delta_vs_f1'].tolist())}")
        lines.append(f"- `delayed_refresh_delta_vs_f1`: {_format_mean_std(ds['delayed_refresh_delta_vs_f1'].tolist())}")
        lines.append(f"- `delayed_refresh_delta_vs_r0`: {_format_mean_std(ds['delayed_refresh_delta_vs_r0'].tolist())}")
        lines.append(f"- `p0b_template_mean_direction_cosine`: {_format_mean_std(ds['p0b_template_mean_direction_cosine'].tolist())}")
        lines.append(f"- `p0b_response_vs_margin_correlation`: {_format_mean_std(ds['p0b_response_vs_margin_correlation'].tolist())}")
        lines.append(f"- `delayed_refresh_margin`: {_format_mean_std(ds['delayed_refresh_margin'].tolist())}")
        lines.append("")

    with open(conclusion_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(f"[pia-operator-p0b-lite] wrote {config_csv}")
    print(f"[pia-operator-p0b-lite] wrote {per_seed_csv}")
    print(f"[pia-operator-p0b-lite] wrote {geometry_csv}")
    print(f"[pia-operator-p0b-lite] wrote {operator_csv}")
    print(f"[pia-operator-p0b-lite] wrote {score_csv}")
    print(f"[pia-operator-p0b-lite] wrote {response_csv}")
    print(f"[pia-operator-p0b-lite] wrote {conclusion_md}")


if __name__ == "__main__":
    main()
