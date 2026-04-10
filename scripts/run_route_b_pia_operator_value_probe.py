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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from route_b_unified.pia_operator_value_probe import (  # noqa: E402
    CenterBasedOperatorConfig,
    FixedReferenceGeometryConfig,
    OperatorApplyResult,
    SingleTemplatePIAValueConfig,
    apply_center_based_operator,
    apply_single_template_pia_operator,
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


def _load_backbone_reference_map(path: str) -> Dict[tuple[str, int], Dict[str, float]]:
    if not os.path.isfile(path):
        return {}
    df = pd.read_csv(path)
    required = {"dataset", "seed"}
    if not required.issubset(set(df.columns)):
        return {}
    out: Dict[tuple[str, int], Dict[str, float]] = {}
    for _, row in df.iterrows():
        out[(str(row["dataset"]).strip().lower(), int(row["seed"]))] = {
            "static_linear_test_macro_f1": float(row["static_linear_test_macro_f1"])
            if "static_linear_test_macro_f1" in row and not pd.isna(row["static_linear_test_macro_f1"])
            else np.nan,
            "dense_dynamic_gru_test_macro_f1": float(row["dense_dynamic_gru_test_macro_f1"])
            if "dense_dynamic_gru_test_macro_f1" in row and not pd.isna(row["dense_dynamic_gru_test_macro_f1"])
            else np.nan,
            "dense_dynamic_minirocket_test_macro_f1": float(row["dense_dynamic_minirocket_test_macro_f1"])
            if "dense_dynamic_minirocket_test_macro_f1" in row and not pd.isna(row["dense_dynamic_minirocket_test_macro_f1"])
            else np.nan,
            "raw_minirocket_test_macro_f1": float(row["raw_minirocket_test_macro_f1"])
            if "raw_minirocket_test_macro_f1" in row and not pd.isna(row["raw_minirocket_test_macro_f1"])
            else np.nan,
        }
    return out


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


def _safe_cosine(a: Sequence[float], b: Sequence[float]) -> float:
    xa = np.asarray(list(a), dtype=np.float64).ravel()
    xb = np.asarray(list(b), dtype=np.float64).ravel()
    if xa.size <= 0 or xb.size <= 0 or xa.shape != xb.shape:
        return 0.0
    na = float(np.linalg.norm(xa))
    nb = float(np.linalg.norm(xb))
    if na <= 1e-12 or nb <= 1e-12:
        return 1.0
    return float(np.dot(xa, xb) / (na * nb + 1e-12))


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
    out = {
        "arm": str(arm_name),
        "fit_window_count": int(train_result.summary.get("fit_window_count", 0)),
        "fit_trial_count": int(train_result.summary.get("fit_trial_count", 0)),
        "accepted_window_ratio": _weighted_mean(rows, "accepted_window_ratio"),
        "shaped_window_ratio": _weighted_mean(rows, "shaped_window_ratio"),
        "operator_norm_mean": _weighted_mean(rows, "operator_norm_mean"),
        "operator_norm_p95": _weighted_mean(rows, "operator_norm_p95"),
        "operator_to_step_ratio_mean": _weighted_mean(rows, "operator_to_step_ratio_mean"),
        "local_step_distortion_ratio_mean": _weighted_mean(rows, "local_step_distortion_ratio_mean"),
        "local_step_distortion_ratio_p95": _weighted_mean(rows, "local_step_distortion_ratio_p95"),
        "operator_direction_stability": _weighted_mean(rows, "operator_direction_stability"),
        "response_vs_margin_correlation": _weighted_mean(rows, "response_vs_margin_correlation"),
        "activation_coverage_ratio": _weighted_mean(rows, "activation_coverage_ratio"),
        "applied_window_count_total": int(sum(int(r.get("applied_window_count", 0)) for r in rows)),
    }
    for split_name, result in [("train", train_result), ("val", val_result), ("test", test_result)]:
        out[f"{split_name}_applied_window_count"] = int(result.summary.get("applied_window_count", 0))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="SCP-Branch P0: PIA Operator Value Probe")
    p.add_argument("--datasets", type=str, default="selfregulationscp1")
    p.add_argument("--seeds", type=str, default="1")
    p.add_argument("--out-root", type=str, default="out/_active/verify_route_b_pia_operator_value_probe_20260401")
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--prop-win-ratio", type=float, default=0.20)
    p.add_argument("--prop-hop-ratio", type=float, default=0.10)
    p.add_argument("--min-window-extra-channels", type=int, default=4)
    p.add_argument("--min-hop-len", type=int, default=4)
    p.add_argument("--force-hop-len", type=int, default=1)
    p.add_argument("--terminal", type=str, default="dynamic_minirocket")
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
    p.add_argument(
        "--backbone-reference-csv",
        type=str,
        default="/home/THL/project/MTS-PIA/out/route_b_dense_trajectory_probe_20260330_formal/dense_trajectory_probe_per_seed.csv",
    )
    args = p.parse_args()

    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_seed_list(args.seeds)
    _ensure_dir(args.out_root)
    backbone_ref_map = _load_backbone_reference_map(args.backbone_reference_csv)

    config_rows: List[Dict[str, object]] = []
    per_seed_rows: List[Dict[str, object]] = []
    dataset_summary_rows: List[Dict[str, object]] = []
    structure_rows_all: List[Dict[str, object]] = []
    score_rows_all: List[Dict[str, object]] = []
    acceptance_rows_all: List[Dict[str, object]] = []

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
            after_a_diag = _build_structure_diag(
                tids=state.train.tids.tolist(),
                labels=state.train.y.tolist(),
                z_seq_list=arm_a_state.train.z_seq_list,
                prototype_count=int(args.prototype_count),
                seed=int(seed),
            )

            pia_operator = fit_single_template_pia_operator(
                geometry=geometry,
                cfg=SingleTemplatePIAValueConfig(
                    r_dimension=1,
                    n_iters=int(args.pia_n_iters),
                    C_repr=float(args.pia_c_repr),
                    activation=str(args.pia_activation),
                    bias_lr=float(args.pia_bias_lr),
                    bias_update_mode=str(args.pia_bias_update_mode),
                    epsilon_scale=float(args.pia_epsilon_scale),
                    smooth_lambda=float(args.operator_smooth_lambda),
                    seed=int(seed),
                ),
            )
            arm_b_train = apply_single_template_pia_operator(
                z_seq_list=state.train.z_seq_list,
                operator=pia_operator,
                cfg=SingleTemplatePIAValueConfig(
                    activation=str(args.pia_activation),
                    epsilon_scale=float(args.pia_epsilon_scale),
                    smooth_lambda=float(args.operator_smooth_lambda),
                ),
            )
            arm_b_val = apply_single_template_pia_operator(
                z_seq_list=state.val.z_seq_list,
                operator=pia_operator,
                cfg=SingleTemplatePIAValueConfig(
                    activation=str(args.pia_activation),
                    epsilon_scale=float(args.pia_epsilon_scale),
                    smooth_lambda=float(args.operator_smooth_lambda),
                ),
            )
            arm_b_test = apply_single_template_pia_operator(
                z_seq_list=state.test.z_seq_list,
                operator=pia_operator,
                cfg=SingleTemplatePIAValueConfig(
                    activation=str(args.pia_activation),
                    epsilon_scale=float(args.pia_epsilon_scale),
                    smooth_lambda=float(args.operator_smooth_lambda),
                ),
            )
            arm_b_state = _clone_state_with_zseqs(
                state,
                train_z_seq_list=arm_b_train.z_seq_list,
                val_z_seq_list=arm_b_val.z_seq_list,
                test_z_seq_list=arm_b_test.z_seq_list,
            )
            after_b_diag = _build_structure_diag(
                tids=state.train.tids.tolist(),
                labels=state.train.y.tolist(),
                z_seq_list=arm_b_state.train.z_seq_list,
                prototype_count=int(args.prototype_count),
                seed=int(seed),
            )

            terminal_name, base_train, base_val, base_test, base_meta = _evaluate_terminal(
                state,
                seed=int(seed),
                args=args,
            )
            _terminal_name_a, arm_a_train_metrics, arm_a_val_metrics, arm_a_test_metrics, arm_a_meta = _evaluate_terminal(
                arm_a_state,
                seed=int(seed),
                args=args,
            )
            _terminal_name_b, arm_b_train_metrics, arm_b_val_metrics, arm_b_test_metrics, arm_b_meta = _evaluate_terminal(
                arm_b_state,
                seed=int(seed),
                args=args,
            )

            arm_a_summary = _merge_apply_summaries("mean_centered", train_result=arm_a_train, val_result=arm_a_val, test_result=arm_a_test)
            arm_b_summary = _merge_apply_summaries("single_template_pia", train_result=arm_b_train, val_result=arm_b_val, test_result=arm_b_test)
            template_mean_direction_cosine = _safe_cosine(
                arm_a_train.meta.get("mean_direction_vector", []),
                pia_operator.direction.tolist(),
            )

            ref = backbone_ref_map.get((str(dataset), int(seed)), {})

            arm_a_margin_delta = float(after_a_diag["nearest_prototype_margin"] - before_diag["nearest_prototype_margin"])
            arm_b_margin_delta = float(after_b_diag["nearest_prototype_margin"] - before_diag["nearest_prototype_margin"])
            arm_a_distortion = float(arm_a_summary["operator_to_step_ratio_mean"])
            arm_b_distortion = float(arm_b_summary["operator_to_step_ratio_mean"])

            _write_json(
                os.path.join(seed_dir, "pia_operator_value_result.json"),
                {
                    "baseline_0": {
                        "terminal": str(terminal_name),
                        "test_macro_f1": float(base_test["macro_f1"]),
                        "val_macro_f1": float(base_val["macro_f1"]),
                        "meta": dict(base_meta),
                    },
                    "arm_a_mean_centered": {
                        "terminal": str(terminal_name),
                        "test_macro_f1": float(arm_a_test_metrics["macro_f1"]),
                        "val_macro_f1": float(arm_a_val_metrics["macro_f1"]),
                        "summary": dict(arm_a_summary),
                        "meta": dict(arm_a_meta),
                    },
                    "arm_b_single_template_pia": {
                        "terminal": str(terminal_name),
                        "test_macro_f1": float(arm_b_test_metrics["macro_f1"]),
                        "val_macro_f1": float(arm_b_val_metrics["macro_f1"]),
                        "summary": dict(arm_b_summary),
                        "pia_operator_meta": dict(pia_operator.meta),
                        "meta": dict(arm_b_meta),
                    },
                    "structure_before": dict(before_diag),
                    "structure_after_arm_a": dict(after_a_diag),
                    "structure_after_arm_b": dict(after_b_diag),
                    "reference_geometry": {
                        "fit_window_count": int(geometry.fit_window_count),
                        "fit_trial_count": int(geometry.fit_trial_count),
                        "meta": dict(geometry.meta),
                    },
                },
            )

            per_seed_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "terminal": str(terminal_name),
                    "same_backbone_no_shaping_test_macro_f1": float(base_test["macro_f1"]),
                    "mean_centered_test_macro_f1": float(arm_a_test_metrics["macro_f1"]),
                    "single_template_pia_test_macro_f1": float(arm_b_test_metrics["macro_f1"]),
                    "delta_mean_centered_vs_no_shaping": float(arm_a_test_metrics["macro_f1"] - base_test["macro_f1"]),
                    "delta_single_template_pia_vs_no_shaping": float(arm_b_test_metrics["macro_f1"] - base_test["macro_f1"]),
                    "delta_single_template_pia_vs_mean_centered": float(arm_b_test_metrics["macro_f1"] - arm_a_test_metrics["macro_f1"]),
                    "delta_mean_centered_between": float(after_a_diag["between_prototype_separation"] - before_diag["between_prototype_separation"]),
                    "delta_mean_centered_margin": float(arm_a_margin_delta),
                    "delta_mean_centered_within": float(after_a_diag["within_prototype_compactness"] - before_diag["within_prototype_compactness"]),
                    "delta_mean_centered_stability": float(after_a_diag["temporal_assignment_stability"] - before_diag["temporal_assignment_stability"]),
                    "delta_single_template_pia_between": float(after_b_diag["between_prototype_separation"] - before_diag["between_prototype_separation"]),
                    "delta_single_template_pia_margin": float(arm_b_margin_delta),
                    "delta_single_template_pia_within": float(after_b_diag["within_prototype_compactness"] - before_diag["within_prototype_compactness"]),
                    "delta_single_template_pia_stability": float(after_b_diag["temporal_assignment_stability"] - before_diag["temporal_assignment_stability"]),
                    "mean_centered_local_step_distortion_ratio_mean": float(arm_a_summary["local_step_distortion_ratio_mean"]),
                    "single_template_pia_local_step_distortion_ratio_mean": float(arm_b_summary["local_step_distortion_ratio_mean"]),
                    "mean_centered_operator_norm_mean": float(arm_a_summary["operator_norm_mean"]),
                    "single_template_pia_operator_norm_mean": float(arm_b_summary["operator_norm_mean"]),
                    "mean_centered_operator_direction_stability": float(arm_a_summary["operator_direction_stability"]),
                    "single_template_pia_operator_direction_stability": float(arm_b_summary["operator_direction_stability"]),
                    "template_mean_direction_cosine": float(template_mean_direction_cosine),
                    "single_template_pia_response_vs_margin_correlation": float(arm_b_summary["response_vs_margin_correlation"]),
                    "single_template_pia_activation_coverage_ratio": float(arm_b_summary["activation_coverage_ratio"]),
                    "mean_centered_margin_to_score_conversion": float(
                        (arm_a_test_metrics["macro_f1"] - base_test["macro_f1"])
                        / (arm_a_margin_delta + 1e-8)
                    ),
                    "single_template_pia_margin_to_score_conversion": float(
                        (arm_b_test_metrics["macro_f1"] - base_test["macro_f1"])
                        / (arm_b_margin_delta + 1e-8)
                    ),
                    "mean_centered_margin_gain_per_unit_distortion": float(arm_a_margin_delta / max(1e-6, arm_a_distortion)),
                    "single_template_pia_margin_gain_per_unit_distortion": float(arm_b_margin_delta / max(1e-6, arm_b_distortion)),
                    "static_linear_test_macro_f1": float(ref.get("static_linear_test_macro_f1", np.nan)),
                    "dense_dynamic_gru_test_macro_f1": float(ref.get("dense_dynamic_gru_test_macro_f1", np.nan)),
                    "raw_minirocket_test_macro_f1": float(ref.get("raw_minirocket_test_macro_f1", np.nan)),
                }
            )

            for stage_name, diag in [
                ("before", before_diag),
                ("after_mean_centered", after_a_diag),
                ("after_single_template_pia", after_b_diag),
            ]:
                structure_rows_all.append({"dataset": str(dataset), "seed": int(seed), "stage": str(stage_name), **dict(diag)})

            score_specs = [
                ("baseline_0", base_train, base_val, base_test, 0.0),
                ("mean_centered", arm_a_train_metrics, arm_a_val_metrics, arm_a_test_metrics, arm_a_margin_delta),
                ("single_template_pia", arm_b_train_metrics, arm_b_val_metrics, arm_b_test_metrics, arm_b_margin_delta),
            ]
            for arm_name, train_metrics, val_metrics, test_metrics, margin_delta in score_specs:
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
                        "delta_vs_mean_centered": float(test_metrics["macro_f1"] - arm_a_test_metrics["macro_f1"])
                        if arm_name != "baseline_0"
                        else 0.0,
                        "margin_to_score_conversion": float(
                            (test_metrics["macro_f1"] - base_test["macro_f1"]) / (margin_delta + 1e-8)
                        )
                        if arm_name != "baseline_0"
                        else 0.0,
                    }
                )

            acceptance_rows_all.append({"dataset": str(dataset), "seed": int(seed), **dict(arm_a_summary)})
            acceptance_rows_all.append({"dataset": str(dataset), "seed": int(seed), **dict(arm_b_summary)})

            config_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
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
                    "terminal": str(terminal_name),
                    "pia_activation": str(args.pia_activation),
                    "pia_n_iters": int(args.pia_n_iters),
                    "pia_c_repr": float(args.pia_c_repr),
                    "pia_bias_lr": float(args.pia_bias_lr),
                    "pia_bias_update_mode": str(args.pia_bias_update_mode),
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "patience": int(args.patience),
                    "gru_hidden_dim": int(args.gru_hidden_dim),
                    "dropout": float(args.dropout),
                    "fit_window_count": int(geometry.fit_window_count),
                    "fit_trial_count": int(geometry.fit_trial_count),
                    "template_mean_direction_cosine": float(template_mean_direction_cosine),
                }
            )

    per_seed_df = pd.DataFrame(per_seed_rows)
    structure_df = pd.DataFrame(structure_rows_all)
    score_df = pd.DataFrame(score_rows_all)
    acceptance_df = pd.DataFrame(acceptance_rows_all)
    config_df = pd.DataFrame(config_rows)

    for dataset in datasets:
        ds = per_seed_df[per_seed_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        dataset_summary_rows.append(
            {
                "dataset": str(dataset),
                "same_backbone_no_shaping_mean": _mean_std(ds["same_backbone_no_shaping_test_macro_f1"].tolist())[0],
                "same_backbone_no_shaping_std": _mean_std(ds["same_backbone_no_shaping_test_macro_f1"].tolist())[1],
                "mean_centered_mean": _mean_std(ds["mean_centered_test_macro_f1"].tolist())[0],
                "mean_centered_std": _mean_std(ds["mean_centered_test_macro_f1"].tolist())[1],
                "single_template_pia_mean": _mean_std(ds["single_template_pia_test_macro_f1"].tolist())[0],
                "single_template_pia_std": _mean_std(ds["single_template_pia_test_macro_f1"].tolist())[1],
                "delta_single_template_pia_vs_mean_centered_mean": _mean_std(ds["delta_single_template_pia_vs_mean_centered"].tolist())[0],
                "delta_mean_centered_margin_mean": _mean_std(ds["delta_mean_centered_margin"].tolist())[0],
                "delta_single_template_pia_margin_mean": _mean_std(ds["delta_single_template_pia_margin"].tolist())[0],
                "mean_centered_margin_gain_per_unit_distortion_mean": _mean_std(ds["mean_centered_margin_gain_per_unit_distortion"].tolist())[0],
                "single_template_pia_margin_gain_per_unit_distortion_mean": _mean_std(ds["single_template_pia_margin_gain_per_unit_distortion"].tolist())[0],
                "template_mean_direction_cosine_mean": _mean_std(ds["template_mean_direction_cosine"].tolist())[0],
                "single_template_pia_response_vs_margin_correlation_mean": _mean_std(ds["single_template_pia_response_vs_margin_correlation"].tolist())[0],
                "single_template_pia_activation_coverage_ratio_mean": _mean_std(ds["single_template_pia_activation_coverage_ratio"].tolist())[0],
            }
        )
    dataset_summary_df = pd.DataFrame(dataset_summary_rows)

    config_csv = os.path.join(args.out_root, "pia_operator_value_config_table.csv")
    per_seed_csv = os.path.join(args.out_root, "pia_operator_value_per_seed.csv")
    dataset_summary_csv = os.path.join(args.out_root, "pia_operator_value_dataset_summary.csv")
    structure_csv = os.path.join(args.out_root, "pia_operator_value_structure_diagnostics.csv")
    score_csv = os.path.join(args.out_root, "pia_operator_value_score_diagnostics.csv")
    acceptance_csv = os.path.join(args.out_root, "pia_operator_value_acceptance_summary.csv")
    conclusion_md = os.path.join(args.out_root, "pia_operator_value_conclusion.md")

    config_df.to_csv(config_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    dataset_summary_df.to_csv(dataset_summary_csv, index=False)
    structure_df.to_csv(structure_csv, index=False)
    score_df.to_csv(score_csv, index=False)
    acceptance_df.to_csv(acceptance_csv, index=False)

    lines: List[str] = [
        "# PIA Operator Value Probe Conclusion",
        "",
        "更新时间：2026-04-01",
        "",
        "主实验只比较 Arm A（mean/prototype-centered update）与 Arm B（single-template PIA update）。",
        "参考几何固定；不做 slow-layer refresh、不做 replay、不做 test-time adaptation。",
        "operator 统一在 train-only 上拟合，拟合后冻结，并以同一参数作用于 train/val/test。",
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
        lines.append(f"- `single_template_pia`: {_format_mean_std(ds['single_template_pia_test_macro_f1'].tolist())}")
        lines.append(f"- `delta_single_template_pia_vs_mean_centered`: {_format_mean_std(ds['delta_single_template_pia_vs_mean_centered'].tolist())}")
        lines.append(f"- `delta_mean_centered_margin`: {_format_mean_std(ds['delta_mean_centered_margin'].tolist())}")
        lines.append(f"- `delta_single_template_pia_margin`: {_format_mean_std(ds['delta_single_template_pia_margin'].tolist())}")
        lines.append(f"- `mean_centered_margin_gain_per_unit_distortion`: {_format_mean_std(ds['mean_centered_margin_gain_per_unit_distortion'].tolist())}")
        lines.append(f"- `single_template_pia_margin_gain_per_unit_distortion`: {_format_mean_std(ds['single_template_pia_margin_gain_per_unit_distortion'].tolist())}")
        lines.append(f"- `template_mean_direction_cosine`: {_format_mean_std(ds['template_mean_direction_cosine'].tolist())}")
        lines.append(f"- `single_template_pia_response_vs_margin_correlation`: {_format_mean_std(ds['single_template_pia_response_vs_margin_correlation'].tolist())}")
        lines.append(f"- `single_template_pia_activation_coverage_ratio`: {_format_mean_std(ds['single_template_pia_activation_coverage_ratio'].tolist())}")
        lines.append("")

    with open(conclusion_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(f"[pia-operator-value] wrote {config_csv}")
    print(f"[pia-operator-value] wrote {per_seed_csv}")
    print(f"[pia-operator-value] wrote {dataset_summary_csv}")
    print(f"[pia-operator-value] wrote {structure_csv}")
    print(f"[pia-operator-value] wrote {score_csv}")
    print(f"[pia-operator-value] wrote {acceptance_csv}")
    print(f"[pia-operator-value] wrote {conclusion_md}")


if __name__ == "__main__":
    main()
