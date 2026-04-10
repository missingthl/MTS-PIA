#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from PIA.telm2 import TELM2Config, TELM2Transformer  # noqa: E402
from route_b_unified.pia_operator_value_probe import (  # noqa: E402
    FixedReferenceGeometry,
    FixedReferenceGeometryConfig,
    OperatorApplyResult,
    SingleTemplatePIADiscriminativeConfig,
    SingleTemplatePIAStageARepairConfig,
    SingleTemplatePIAValueConfig,
    _assign_same_opp_from_fixed_geometry,
    _build_single_template_operator_from_fit,
    _median_min_side_weights,
    _nearest_opposite_prototype_for_same_prototype,
    _normalize_direction,
    _safe_cosine,
    _smooth_delta,
    _summarize_operator_application,
    _window_local_step_mean,
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
from route_b_unified.trajectory_minirocket_evaluator import (  # noqa: E402
    TrajectoryMiniRocketEvalConfig,
    evaluate_dynamic_minirocket_classifier,
)
from route_b_unified.trajectory_representation import (  # noqa: E402
    TrajectoryRepresentationConfig,
    TrajectoryRepresentationState,
    build_trajectory_representation,
)


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


def _load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_raw_minirocket_reference(dataset: str, seed: int) -> Dict[str, object]:
    dataset = str(dataset).lower()
    candidates = [
        (
            "official_fixedsplit_20260318",
            os.path.join(ROOT, "out", "raw_minirocket_official_fixedsplit_20260318", dataset, "summary_per_seed.csv"),
            "test_macro_f1",
            int(seed),
        ),
        (
            "official_fixedsplit_aeon6_20260321",
            os.path.join(ROOT, "out", "raw_minirocket_official_fixedsplit_aeon6_20260321", dataset, "summary_per_seed.csv"),
            "test_macro_f1",
            int(seed),
        ),
        (
            "internal_baseline_seed",
            os.path.join(ROOT, "out", "raw_minirocket_baseline", dataset, "summary_per_seed.csv"),
            "trial_macro_f1",
            int(seed),
        ),
    ]
    for source_name, path, metric_key, target_seed in candidates:
        if not os.path.exists(path):
            continue
        rows = _load_csv_rows(path)
        if not rows:
            continue
        chosen = None
        for row in rows:
            if int(row.get("seed", -1)) == int(target_seed):
                chosen = row
                break
        if chosen is None:
            chosen = rows[0]
        value = float(chosen[metric_key])
        return {
            "available": True,
            "source": str(source_name),
            "metric_key": str(metric_key),
            "path": str(path),
            "test_macro_f1": float(value),
            "seed_requested": int(seed),
            "seed_used": int(chosen.get("seed", seed)),
        }

    agg_path = os.path.join(ROOT, "out", "raw_minirocket_baseline", "_runs", "success_metrics_agg.csv")
    if os.path.exists(agg_path):
        rows = _load_csv_rows(agg_path)
        for row in rows:
            if str(row.get("dataset", "")).lower() == dataset and str(row.get("metric", "")) == "trial_macro_f1":
                return {
                    "available": True,
                    "source": "internal_baseline_agg",
                    "metric_key": "trial_macro_f1",
                    "path": str(agg_path),
                    "test_macro_f1": float(row["mean"]),
                    "seed_requested": int(seed),
                    "seed_used": None,
                }

    return {
        "available": False,
        "source": "",
        "metric_key": "",
        "path": "",
        "test_macro_f1": np.nan,
        "seed_requested": int(seed),
        "seed_used": None,
    }


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
) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, object]]:
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
    return dict(result.train_metrics), dict(result.val_metrics), dict(result.test_metrics), dict(result.meta)


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
        "template_mean_direction_cosine": _weighted_mean(rows, "template_mean_direction_cosine"),
        "local_pair_axis_router_cosine_mean": _weighted_mean(rows, "local_pair_axis_router_cosine_mean"),
        "local_template_direction_cosine_mean": _weighted_mean(rows, "local_template_direction_cosine_mean"),
        "local_pool_same_count_mean": _weighted_mean(rows, "local_pool_same_count_mean"),
        "local_pool_opp_count_mean": _weighted_mean(rows, "local_pool_opp_count_mean"),
        "local_same_weight_mass_mean": _weighted_mean(rows, "local_same_weight_mass_mean"),
        "local_opp_weight_mass_mean": _weighted_mean(rows, "local_opp_weight_mass_mean"),
        "local_operator_fit_count": int(sum(int(r.get("local_operator_fit_count", 0)) for r in rows)),
        "local_operator_runtime_seconds": float(sum(float(r.get("local_operator_runtime_seconds", 0.0)) for r in rows)),
        "router_mode": str(train_result.summary.get("router_mode", "")),
        "local_topk": int(train_result.summary.get("local_topk", 0)),
        "applied_window_count_total": int(sum(int(r.get("applied_window_count", 0)) for r in rows)),
    }


def _build_proto_row_index(geometry: FixedReferenceGeometry) -> Dict[Tuple[int, int], List[int]]:
    out: Dict[Tuple[int, int], List[int]] = {}
    for idx, row in enumerate(geometry.fit_rows):
        key = (int(row["class_id"]), int(row["prototype_id"]))
        out.setdefault(key, []).append(int(idx))
    return out


def _fit_local_operator_for_query(
    *,
    query_z: np.ndarray,
    geometry: FixedReferenceGeometry,
    proto_row_index: Dict[Tuple[int, int], List[int]],
    cfg: SingleTemplatePIADiscriminativeConfig,
    local_topk: int,
):
    query = np.asarray(query_z, dtype=np.float64)
    router_same_cls, router_same_pid, p_same_router, _p_opp_query, same_dist_router, opp_dist_router = _assign_same_opp_from_fixed_geometry(
        query,
        geometry=geometry,
    )
    router_opp_cls, router_opp_pid, p_opp_router, _ = _nearest_opposite_prototype_for_same_prototype(
        geometry,
        class_id=int(router_same_cls),
        prototype_id=int(router_same_pid),
    )

    same_row_ids = list(proto_row_index.get((int(router_same_cls), int(router_same_pid)), []))
    opp_row_ids = list(proto_row_index.get((int(router_opp_cls), int(router_opp_pid)), []))
    if not same_row_ids or not opp_row_ids:
        raise RuntimeError("local bipolar pool requires non-empty same and opp candidate sets")

    same_arr_all = np.asarray(geometry.fit_windows[same_row_ids], dtype=np.float64)
    opp_arr_all = np.asarray(geometry.fit_windows[opp_row_ids], dtype=np.float64)
    same_query_dists = np.linalg.norm(same_arr_all - query[None, :], axis=1)
    opp_query_dists = np.linalg.norm(opp_arr_all - query[None, :], axis=1)

    k_same = int(max(1, min(int(local_topk), int(same_arr_all.shape[0]))))
    k_opp = int(max(1, min(int(local_topk), int(opp_arr_all.shape[0]))))
    same_keep = np.argsort(same_query_dists)[:k_same]
    opp_keep = np.argsort(opp_query_dists)[:k_opp]

    same_arr = np.asarray(same_arr_all[same_keep], dtype=np.float64)
    opp_arr = np.asarray(opp_arr_all[opp_keep], dtype=np.float64)
    same_dists = np.asarray(same_query_dists[same_keep], dtype=np.float64)
    opp_dists = np.asarray(opp_query_dists[opp_keep], dtype=np.float64)

    same_w_raw, same_scale, same_fallback = _median_min_side_weights(same_dists)
    opp_w_raw, opp_scale, opp_fallback = _median_min_side_weights(opp_dists)
    same_w = np.asarray(same_w_raw / (float(np.sum(same_w_raw)) + 1e-8), dtype=np.float64)
    opp_w = np.asarray(opp_w_raw / (float(np.sum(opp_w_raw)) + 1e-8), dtype=np.float64)

    same_center = np.sum(same_arr * same_w[:, None], axis=0)
    opp_center = np.sum(opp_arr * opp_w[:, None], axis=0)
    pair_axis = _normalize_direction(np.asarray(same_center - opp_center, dtype=np.float64))
    router_axis = _normalize_direction(np.asarray(p_same_router - p_opp_router, dtype=np.float64))
    pair_axis_router_cosine = float(_safe_cosine(pair_axis, router_axis))

    fit_arr = np.concatenate([same_arr, opp_arr], axis=0).astype(np.float64)
    target_arr = np.concatenate(
        [
            np.repeat(pair_axis[None, :], int(same_arr.shape[0]), axis=0),
            np.repeat((-pair_axis)[None, :], int(opp_arr.shape[0]), axis=0),
        ],
        axis=0,
    ).astype(np.float64)
    sample_weights = np.concatenate([same_w, opp_w], axis=0).astype(np.float64)

    telm = TELM2Transformer(
        TELM2Config(
            r_dimension=int(cfg.r_dimension),
            n_iters=int(cfg.n_iters),
            C_repr=float(cfg.C_repr),
            activation=str(cfg.activation),
            bias_lr=float(cfg.bias_lr),
            orthogonalize=bool(cfg.orthogonalize),
            enable_repr_learning=bool(cfg.enable_repr_learning),
            bias_update_mode=str(cfg.bias_update_mode),
            seed=None if cfg.seed is None else int(cfg.seed),
        )
    ).fit(
        fit_arr,
        sample_weights=sample_weights,
        target_override=target_arr,
    )
    arts = telm.get_artifacts()

    fit_rows: List[Dict[str, object]] = []
    fit_trials: set[str] = set()
    for local_idx, base_idx in enumerate(same_keep.tolist()):
        row = dict(geometry.fit_rows[int(same_row_ids[int(base_idx)])])
        row["sample_side"] = "same"
        row["u_geom_vector"] = np.asarray(pair_axis, dtype=np.float64)
        fit_rows.append(row)
        fit_trials.add(str(row["trial_id"]))
    for local_idx, base_idx in enumerate(opp_keep.tolist()):
        row = dict(geometry.fit_rows[int(opp_row_ids[int(base_idx)])])
        row["sample_side"] = "opp"
        row["u_geom_vector"] = np.asarray(pair_axis, dtype=np.float64)
        fit_rows.append(row)
        fit_trials.add(str(row["trial_id"]))

    operator = _build_single_template_operator_from_fit(
        fit_windows=fit_arr,
        fit_rows=fit_rows,
        fit_trial_count=int(len(fit_trials)),
        geometry=geometry,
        activation=str(cfg.activation),
        telm_artifacts=arts,
        meta={
            "fit_target_mode": "hetero_associative_discriminative_local_linear_pm1",
            "target_mode": "linear_pm1",
            "pool_mode": "local_bipolar_knn",
            "router_mode": "nearest_prototype_frozen_geometry",
            "opp_pair_rule": "nearest_opposite_prototype_from_router_same",
            "template_readout_mode": str(cfg.template_readout_mode),
            "template_count": int(cfg.r_dimension),
            "same_pool_count": int(same_arr.shape[0]),
            "opp_pool_count": int(opp_arr.shape[0]),
            "same_weight_mass": float(np.sum(same_w)),
            "opp_weight_mass": float(np.sum(opp_w)),
            "same_opp_count_ratio": float(same_arr.shape[0] / max(1, int(opp_arr.shape[0]))),
            "same_opp_weight_mass_ratio": float(np.sum(same_w) / max(1e-8, float(np.sum(opp_w)))),
            "same_proto_effective_sample_size": float((np.sum(same_w) ** 2) / max(1e-8, np.sum(np.square(same_w)))),
            "opp_proto_effective_sample_size": float((np.sum(opp_w) ** 2) / max(1e-8, np.sum(np.square(opp_w)))),
            "discriminative_target_gap": 2.0,
            "local_pair_axis_router_cosine": float(pair_axis_router_cosine),
            "local_same_weight_scale": float(same_scale),
            "local_opp_weight_scale": float(opp_scale),
            "local_same_fallback_unweighted": bool(same_fallback),
            "local_opp_fallback_unweighted": bool(opp_fallback),
            "router_same_class_id": int(router_same_cls),
            "router_same_prototype_id": int(router_same_pid),
            "router_opp_class_id": int(router_opp_cls),
            "router_opp_prototype_id": int(router_opp_pid),
            "router_same_dist": float(same_dist_router),
            "router_opp_dist": float(opp_dist_router),
        },
        response_stats_windows=fit_arr,
        response_stats_mode="local_fit_pool",
    )
    return operator


def _apply_local_wls_probe(
    *,
    z_seq_list: Sequence[np.ndarray],
    geometry: FixedReferenceGeometry,
    cfg: SingleTemplatePIADiscriminativeConfig,
    epsilon_scale: float,
    smooth_lambda: float,
    local_topk: int,
    budget_target_operator_to_step_ratio: float | None = None,
    budget_scale_factor: float | None = None,
) -> OperatorApplyResult:
    proto_row_index = _build_proto_row_index(geometry)
    z_aug_list: List[np.ndarray] = []
    delta_list: List[np.ndarray] = []
    direction_list: List[np.ndarray] = []
    diagnostic_rows: List[Dict[str, object]] = []
    raw_delta_list: List[np.ndarray] = []
    local_step_vals: List[float] = []
    local_template_cosines: List[float] = []
    local_router_axis_cosines: List[float] = []
    same_pool_counts: List[float] = []
    opp_pool_counts: List[float] = []
    same_weight_masses: List[float] = []
    opp_weight_masses: List[float] = []
    local_fit_count = 0
    fit_runtime_seconds = 0.0

    for seq_idx, seq in enumerate(z_seq_list):
        arr = np.asarray(seq, dtype=np.float64)
        raw_delta = np.zeros_like(arr, dtype=np.float64)
        raw_dir = np.zeros_like(arr, dtype=np.float64)
        clipped_seq = np.zeros((int(arr.shape[0]),), dtype=np.float64)
        scale_seq = np.ones((int(arr.shape[0]),), dtype=np.float64)
        raw_preact_seq = np.zeros((int(arr.shape[0]),), dtype=np.float64)
        local_ops: List[object] = []

        for window_index in range(int(arr.shape[0])):
            fit_start = time.perf_counter()
            operator = _fit_local_operator_for_query(
                query_z=np.asarray(arr[int(window_index)], dtype=np.float64),
                geometry=geometry,
                proto_row_index=proto_row_index,
                cfg=cfg,
                local_topk=int(local_topk),
            )
            fit_runtime_seconds += float(time.perf_counter() - fit_start)
            local_fit_count += 1
            local_ops.append(operator)

            z = np.asarray(arr[int(window_index)], dtype=np.float64)
            raw_preact = float(z @ np.asarray(operator.readout_w, dtype=np.float64) + float(operator.readout_b))
            clipped = float(np.clip(raw_preact, float(operator.preactivation_clip_lower), float(operator.preactivation_clip_upper)))
            raw_preact_seq[int(window_index)] = float(raw_preact)
            clipped_seq[int(window_index)] = float(clipped)
            scale_seq[int(window_index)] = max(1e-6, float(operator.response_scale_iqr))
            raw_dir[int(window_index)] = np.asarray(operator.direction, dtype=np.float64)

            same_cls, same_pid, _p_same, _p_opp, same_dist, opp_dist = _assign_same_opp_from_fixed_geometry(
                z,
                geometry=geometry,
            )
            margin_before = float(opp_dist - same_dist)

            local_template_cosines.append(float(operator.meta.get("template_mean_direction_cosine", 0.0)))
            local_router_axis_cosines.append(float(operator.meta.get("local_pair_axis_router_cosine", 0.0)))
            same_pool_counts.append(float(operator.meta.get("same_pool_count", 0.0)))
            opp_pool_counts.append(float(operator.meta.get("opp_pool_count", 0.0)))
            same_weight_masses.append(float(operator.meta.get("same_weight_mass", 0.0)))
            opp_weight_masses.append(float(operator.meta.get("opp_weight_mass", 0.0)))

            diagnostic_rows.append(
                {
                    "sequence_index": int(seq_idx),
                    "window_index": int(window_index),
                    "assigned_class_id": int(same_cls),
                    "assigned_prototype_id": int(same_pid),
                    "same_dist": float(same_dist),
                    "opp_dist": float(opp_dist),
                    "margin_before": float(margin_before),
                    "raw_preactivation": float(raw_preact),
                    "clipped_preactivation": float(clipped),
                    "local_response_scale_iqr": float(scale_seq[int(window_index)]),
                    "local_template_mean_direction_cosine": float(operator.meta.get("template_mean_direction_cosine", 0.0)),
                    "local_pair_axis_router_cosine": float(operator.meta.get("local_pair_axis_router_cosine", 0.0)),
                    "same_pool_count": int(operator.meta.get("same_pool_count", 0)),
                    "opp_pool_count": int(operator.meta.get("opp_pool_count", 0)),
                    "same_weight_mass": float(operator.meta.get("same_weight_mass", 0.0)),
                    "opp_weight_mass": float(operator.meta.get("opp_weight_mass", 0.0)),
                }
            )

        seq_local_median = float(np.median(clipped_seq)) if clipped_seq.size else 0.0
        for window_index in range(int(arr.shape[0])):
            local_step = float(_window_local_step_mean(arr, int(window_index)))
            activation_driver = float((clipped_seq[int(window_index)] - seq_local_median) / max(1e-6, float(scale_seq[int(window_index)])))
            response_force = float(np.tanh(activation_driver))
            raw_delta[int(window_index)] = float(epsilon_scale) * float(local_step) * float(response_force) * np.asarray(raw_dir[int(window_index)], dtype=np.float64)
            local_step_vals.append(float(local_step))
            diagnostic_rows[-int(arr.shape[0]) + int(window_index)]["activation_driver"] = float(activation_driver)
            diagnostic_rows[-int(arr.shape[0]) + int(window_index)]["response_force"] = float(response_force)
            diagnostic_rows[-int(arr.shape[0]) + int(window_index)]["response_gate"] = float(abs(response_force))
            diagnostic_rows[-int(arr.shape[0]) + int(window_index)]["delta_norm_raw"] = float(np.linalg.norm(raw_delta[int(window_index)]))

        raw_delta_list.append(np.asarray(raw_delta, dtype=np.float64))
        direction_list.append(np.asarray(raw_dir, dtype=np.float32))

    smooth_delta_list = [_smooth_delta(v, float(smooth_lambda)) for v in raw_delta_list]
    if budget_scale_factor is not None:
        budget_scale = float(budget_scale_factor)
    elif float(budget_target_operator_to_step_ratio or 0.0) > 0.0:
        delta_norm_vals = [float(np.linalg.norm(v[i])) for v, seq in zip(smooth_delta_list, z_seq_list) for i in range(int(np.asarray(seq).shape[0]))]
        current_ratio = float(np.mean(np.asarray(delta_norm_vals, dtype=np.float64) / np.maximum(1e-6, np.asarray(local_step_vals, dtype=np.float64)))) if local_step_vals else 0.0
        budget_scale = float(budget_target_operator_to_step_ratio) / max(1e-8, current_ratio) if current_ratio > 0.0 else 1.0
    else:
        budget_scale = 1.0
    smooth_delta_list = [np.asarray(v * float(budget_scale), dtype=np.float64) for v in smooth_delta_list]

    for seq, delta_smooth in zip(z_seq_list, smooth_delta_list):
        arr = np.asarray(seq, dtype=np.float64)
        z_aug = arr + delta_smooth
        z_aug_list.append(np.asarray(z_aug, dtype=np.float32))
        delta_list.append(np.asarray(delta_smooth, dtype=np.float32))

    response_force_arr = np.asarray([float(r.get("response_force", 0.0)) for r in diagnostic_rows], dtype=np.float64)
    activation_driver_arr = np.asarray([float(r.get("activation_driver", 0.0)) for r in diagnostic_rows], dtype=np.float64)
    margin_arr = np.asarray([float(r.get("margin_before", 0.0)) for r in diagnostic_rows], dtype=np.float64)
    preactivation_clip_rate = 0.0
    if response_force_arr.size >= 2 and np.std(response_force_arr) > 1e-12 and np.std(margin_arr) > 1e-12:
        response_vs_margin_correlation = float(np.corrcoef(response_force_arr, margin_arr)[0, 1])
    else:
        response_vs_margin_correlation = 0.0
    activation_coverage_ratio = float(np.mean(np.abs(activation_driver_arr) >= 1.0)) if activation_driver_arr.size else 0.0
    gate_saturation_ratio = float(np.mean(np.abs(response_force_arr) >= 0.95)) if response_force_arr.size else 0.0

    summary = _summarize_operator_application(
        original_z_seq_list=z_seq_list,
        z_seq_list=z_aug_list,
        delta_seq_list=delta_list,
        direction_seq_list=direction_list,
        diagnostics_rows=diagnostic_rows,
        fit_window_count=int(geometry.fit_window_count),
        fit_trial_count=int(geometry.fit_trial_count),
        meta={
            "operator_arm": "p1a_stage1_local_wls",
            "fit_target_mode": "hetero_associative_discriminative_local_linear_pm1",
            "pool_mode": "local_bipolar_knn",
            "router_mode": "nearest_prototype_frozen_geometry",
            "local_topk": int(local_topk),
            "budget_scale_factor": float(budget_scale),
            "response_vs_margin_correlation": float(response_vs_margin_correlation),
            "activation_coverage_ratio": float(activation_coverage_ratio),
            "preactivation_clip_rate": float(preactivation_clip_rate),
            "response_centering_std_after_fix": float(np.std(activation_driver_arr)) if activation_driver_arr.size else 0.0,
            "gate_saturation_ratio": float(gate_saturation_ratio),
            "template_mean_direction_cosine": float(np.mean(np.asarray(local_template_cosines, dtype=np.float64))) if local_template_cosines else 0.0,
            "local_pair_axis_router_cosine_mean": float(np.mean(np.asarray(local_router_axis_cosines, dtype=np.float64))) if local_router_axis_cosines else 0.0,
            "local_template_direction_cosine_mean": float(np.mean(np.asarray(local_template_cosines, dtype=np.float64))) if local_template_cosines else 0.0,
            "local_pool_same_count_mean": float(np.mean(np.asarray(same_pool_counts, dtype=np.float64))) if same_pool_counts else 0.0,
            "local_pool_opp_count_mean": float(np.mean(np.asarray(opp_pool_counts, dtype=np.float64))) if opp_pool_counts else 0.0,
            "local_same_weight_mass_mean": float(np.mean(np.asarray(same_weight_masses, dtype=np.float64))) if same_weight_masses else 0.0,
            "local_opp_weight_mass_mean": float(np.mean(np.asarray(opp_weight_masses, dtype=np.float64))) if opp_weight_masses else 0.0,
            "local_operator_fit_count": int(local_fit_count),
            "local_operator_runtime_seconds": float(fit_runtime_seconds),
        },
    )
    return OperatorApplyResult(
        z_seq_list=z_aug_list,
        summary=summary,
        diagnostics_rows=diagnostic_rows,
        meta={
            "operator_arm": "p1a_stage1_local_wls",
            "mean_direction_vector": [],
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description="PIA P1a stage1 offline local WLS probe")
    p.add_argument("--datasets", type=str, default="fingermovements")
    p.add_argument("--seeds", type=str, default="1")
    p.add_argument("--out-root", type=str, default="out/_active/verify_route_b_pia_operator_p1a_stage1_local_wls_20260403_smoke")
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--prop-win-ratio", type=float, default=0.20)
    p.add_argument("--prop-hop-ratio", type=float, default=0.10)
    p.add_argument("--min-window-extra-channels", type=int, default=4)
    p.add_argument("--min-hop-len", type=int, default=4)
    p.add_argument("--force-hop-len", type=int, default=1)
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
    p.add_argument("--local-topk", type=int, default=0)
    args = p.parse_args()

    datasets = [tok.strip().lower() for tok in str(args.datasets).split(",") if tok.strip()]
    seeds = sorted(set(int(tok.strip()) for tok in str(args.seeds).split(",") if tok.strip()))
    _ensure_dir(args.out_root)

    config_rows: List[Dict[str, object]] = []
    per_seed_rows: List[Dict[str, object]] = []
    structure_rows_all: List[Dict[str, object]] = []
    response_rows_all: List[Dict[str, object]] = []

    for dataset in datasets:
        for seed in seeds:
            state = _build_dense_state(args, dataset, seed)
            seed_dir = os.path.join(args.out_root, f"{dataset}_seed{seed}")
            _ensure_dir(seed_dir)
            raw_ref = _load_raw_minirocket_reference(dataset, seed)

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

            op_c0 = fit_single_template_pia_operator(
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

            base_train_metrics, base_val_metrics, base_test_metrics, base_meta = _evaluate_terminal(state, seed=int(seed), args=args)

            train_deploy_windows = np.concatenate([np.asarray(v, dtype=np.float64) for v in state.train.z_seq_list], axis=0).astype(np.float64)
            operator_global = fit_single_template_pia_operator_discriminative(
                geometry=geometry,
                cfg=SingleTemplatePIADiscriminativeConfig(
                    r_dimension=int(args.fast_r_dimension),
                    n_iters=int(args.pia_n_iters),
                    C_repr=float(args.pia_c_repr),
                    activation=str(args.pia_activation),
                    bias_lr=float(args.pia_bias_lr),
                    bias_update_mode=str(args.pia_bias_update_mode),
                    target_mode="linear_pm1",
                    template_readout_mode="mean_committee",
                    opp_pair_rule="nearest_opposite_prototype",
                    seed=int(seed),
                ),
                response_stats_windows=train_deploy_windows,
                response_stats_mode="deployment_train",
            )
            f1_train_res = apply_single_template_pia_stage_a_variant(
                z_seq_list=state.train.z_seq_list,
                operator=operator_global,
                cfg=SingleTemplatePIAStageARepairConfig(
                    variant="sigmoid_clip_tanh_local_median_scaled_iqr",
                    epsilon_scale=float(args.pia_epsilon_scale),
                    smooth_lambda=float(args.operator_smooth_lambda),
                    budget_target_operator_to_step_ratio=float(budget_target),
                ),
            )
            f1_budget_scale = float(f1_train_res.summary.get("budget_scale_factor", 1.0))
            f1_val_res = apply_single_template_pia_stage_a_variant(
                z_seq_list=state.val.z_seq_list,
                operator=operator_global,
                cfg=SingleTemplatePIAStageARepairConfig(
                    variant="sigmoid_clip_tanh_local_median_scaled_iqr",
                    epsilon_scale=float(args.pia_epsilon_scale),
                    smooth_lambda=float(args.operator_smooth_lambda),
                    budget_scale_factor=float(f1_budget_scale),
                ),
            )
            f1_test_res = apply_single_template_pia_stage_a_variant(
                z_seq_list=state.test.z_seq_list,
                operator=operator_global,
                cfg=SingleTemplatePIAStageARepairConfig(
                    variant="sigmoid_clip_tanh_local_median_scaled_iqr",
                    epsilon_scale=float(args.pia_epsilon_scale),
                    smooth_lambda=float(args.operator_smooth_lambda),
                    budget_scale_factor=float(f1_budget_scale),
                ),
            )
            f1_summary = _merge_apply_summaries("f1_global_mainline", train_result=f1_train_res, val_result=f1_val_res, test_result=f1_test_res)
            f1_state = _clone_state_with_zseqs(
                state,
                train_z_seq_list=f1_train_res.z_seq_list,
                val_z_seq_list=f1_val_res.z_seq_list,
                test_z_seq_list=f1_test_res.z_seq_list,
            )
            f1_train_metrics, f1_val_metrics, f1_test_metrics, f1_meta = _evaluate_terminal(f1_state, seed=int(seed), args=args)
            after_f1_diag = _build_structure_diag(
                tids=state.train.tids.tolist(),
                labels=state.train.y.tolist(),
                z_seq_list=f1_state.train.z_seq_list,
                prototype_count=int(args.prototype_count),
                seed=int(seed),
            )

            local_topk = int(args.local_topk) if int(args.local_topk) > 0 else max(1, int(args.anchors_per_prototype) // 2)
            local_cfg = SingleTemplatePIADiscriminativeConfig(
                r_dimension=int(args.fast_r_dimension),
                n_iters=int(args.pia_n_iters),
                C_repr=float(args.pia_c_repr),
                activation=str(args.pia_activation),
                bias_lr=float(args.pia_bias_lr),
                bias_update_mode=str(args.pia_bias_update_mode),
                target_mode="linear_pm1",
                template_readout_mode="mean_committee",
                opp_pair_rule="nearest_opposite_prototype",
                seed=int(seed),
            )
            p1_train_res = _apply_local_wls_probe(
                z_seq_list=state.train.z_seq_list,
                geometry=geometry,
                cfg=local_cfg,
                epsilon_scale=float(args.pia_epsilon_scale),
                smooth_lambda=float(args.operator_smooth_lambda),
                local_topk=int(local_topk),
                budget_target_operator_to_step_ratio=float(budget_target),
            )
            p1_budget_scale = float(p1_train_res.summary.get("budget_scale_factor", 1.0))
            p1_val_res = _apply_local_wls_probe(
                z_seq_list=state.val.z_seq_list,
                geometry=geometry,
                cfg=local_cfg,
                epsilon_scale=float(args.pia_epsilon_scale),
                smooth_lambda=float(args.operator_smooth_lambda),
                local_topk=int(local_topk),
                budget_scale_factor=float(p1_budget_scale),
            )
            p1_test_res = _apply_local_wls_probe(
                z_seq_list=state.test.z_seq_list,
                geometry=geometry,
                cfg=local_cfg,
                epsilon_scale=float(args.pia_epsilon_scale),
                smooth_lambda=float(args.operator_smooth_lambda),
                local_topk=int(local_topk),
                budget_scale_factor=float(p1_budget_scale),
            )
            p1_summary = _merge_apply_summaries("p1a_s1_offline_local_wls", train_result=p1_train_res, val_result=p1_val_res, test_result=p1_test_res)
            p1_state = _clone_state_with_zseqs(
                state,
                train_z_seq_list=p1_train_res.z_seq_list,
                val_z_seq_list=p1_val_res.z_seq_list,
                test_z_seq_list=p1_test_res.z_seq_list,
            )
            p1_train_metrics, p1_val_metrics, p1_test_metrics, p1_meta = _evaluate_terminal(p1_state, seed=int(seed), args=args)
            after_p1_diag = _build_structure_diag(
                tids=state.train.tids.tolist(),
                labels=state.train.y.tolist(),
                z_seq_list=p1_state.train.z_seq_list,
                prototype_count=int(args.prototype_count),
                seed=int(seed),
            )

            structure_rows_all.extend(
                [
                    {"dataset": str(dataset), "seed": int(seed), "stage": "before_fast", **dict(before_diag)},
                    {"dataset": str(dataset), "seed": int(seed), "stage": "after_f1_global", **dict(after_f1_diag)},
                    {"dataset": str(dataset), "seed": int(seed), "stage": "after_p1a_local", **dict(after_p1_diag)},
                ]
            )
            response_rows_all.extend(
                [{"dataset": str(dataset), "seed": int(seed), "arm": "f1_global_mainline", **dict(f1_summary)}]
                + [{"dataset": str(dataset), "seed": int(seed), "arm": "p1a_s1_offline_local_wls", **dict(p1_summary)}]
            )

            per_seed_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "baseline_0_test_macro_f1": float(base_test_metrics["macro_f1"]),
                    "f1_global_mainline_test_macro_f1": float(f1_test_metrics["macro_f1"]),
                    "p1a_s1_offline_local_wls_test_macro_f1": float(p1_test_metrics["macro_f1"]),
                    "raw_minirocket_reference_available": bool(raw_ref["available"]),
                    "raw_minirocket_reference_source": str(raw_ref["source"]),
                    "raw_minirocket_reference_metric_key": str(raw_ref["metric_key"]),
                    "raw_minirocket_reference_seed_used": raw_ref["seed_used"],
                    "raw_minirocket_reference_test_macro_f1": float(raw_ref["test_macro_f1"]) if bool(raw_ref["available"]) else np.nan,
                    "p1a_delta_vs_baseline": float(p1_test_metrics["macro_f1"] - base_test_metrics["macro_f1"]),
                    "p1a_delta_vs_f1": float(p1_test_metrics["macro_f1"] - f1_test_metrics["macro_f1"]),
                    "baseline_delta_vs_raw_minirocket": float(base_test_metrics["macro_f1"] - raw_ref["test_macro_f1"]) if bool(raw_ref["available"]) else np.nan,
                    "f1_delta_vs_raw_minirocket": float(f1_test_metrics["macro_f1"] - raw_ref["test_macro_f1"]) if bool(raw_ref["available"]) else np.nan,
                    "p1a_delta_vs_raw_minirocket": float(p1_test_metrics["macro_f1"] - raw_ref["test_macro_f1"]) if bool(raw_ref["available"]) else np.nan,
                    "f1_response_vs_margin_correlation": float(f1_summary.get("response_vs_margin_correlation", 0.0)),
                    "p1a_response_vs_margin_correlation": float(p1_summary.get("response_vs_margin_correlation", 0.0)),
                    "f1_template_mean_direction_cosine": float(f1_summary.get("template_mean_direction_cosine", 0.0)),
                    "p1a_template_mean_direction_cosine": float(p1_summary.get("template_mean_direction_cosine", 0.0)),
                    "p1a_local_pair_axis_router_cosine_mean": float(p1_summary.get("local_pair_axis_router_cosine_mean", 0.0)),
                    "p1a_local_operator_runtime_seconds": float(p1_summary.get("local_operator_runtime_seconds", 0.0)),
                    "p1a_local_operator_fit_count": int(p1_summary.get("local_operator_fit_count", 0)),
                    "local_topk": int(local_topk),
                }
            )
            config_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "terminal": "dynamic_minirocket",
                    "prototype_count": int(args.prototype_count),
                    "anchors_per_prototype": int(args.anchors_per_prototype),
                    "local_topk": int(local_topk),
                    "fast_r_dimension": int(args.fast_r_dimension),
                    "pia_activation": str(args.pia_activation),
                    "pia_n_iters": int(args.pia_n_iters),
                    "pia_c_repr": float(args.pia_c_repr),
                    "pia_bias_lr": float(args.pia_bias_lr),
                    "pia_bias_update_mode": str(args.pia_bias_update_mode),
                    "prop_win_ratio": float(args.prop_win_ratio),
                    "prop_hop_ratio": float(args.prop_hop_ratio),
                    "min_window_extra_channels": int(args.min_window_extra_channels),
                    "force_hop_len": int(args.force_hop_len),
                }
            )

            result_json = {
                "baseline_0": {"test_macro_f1": float(base_test_metrics["macro_f1"]), "meta": dict(base_meta)},
                "raw_minirocket_reference": dict(raw_ref),
                "f1_global_mainline": {
                    "test_macro_f1": float(f1_test_metrics["macro_f1"]),
                    "summary": dict(f1_summary),
                    "meta": dict(f1_meta),
                },
                "p1a_s1_offline_local_wls": {
                    "test_macro_f1": float(p1_test_metrics["macro_f1"]),
                    "summary": dict(p1_summary),
                    "meta": dict(p1_meta),
                },
            }
            _write_json(os.path.join(seed_dir, "pia_operator_p1a_stage1_result.json"), result_json)

    config_df = pd.DataFrame(config_rows)
    per_seed_df = pd.DataFrame(per_seed_rows)
    structure_df = pd.DataFrame(structure_rows_all)
    response_df = pd.DataFrame(response_rows_all)

    config_path = os.path.join(args.out_root, "pia_operator_p1a_stage1_config_table.csv")
    per_seed_path = os.path.join(args.out_root, "pia_operator_p1a_stage1_per_seed.csv")
    structure_path = os.path.join(args.out_root, "pia_operator_p1a_stage1_structure_diagnostics.csv")
    response_path = os.path.join(args.out_root, "pia_operator_p1a_stage1_response_diagnostics.csv")
    conclusion_path = os.path.join(args.out_root, "pia_operator_p1a_stage1_conclusion.md")

    config_df.to_csv(config_path, index=False)
    per_seed_df.to_csv(per_seed_path, index=False)
    structure_df.to_csv(structure_path, index=False)
    response_df.to_csv(response_path, index=False)

    lines = [
        "# P1a Stage1 Conclusion",
        "",
        "本轮只做最小 local-WLS probe：`FingerMovements + dynamic_minirocket + frozen geometry + nearest-prototype router + offline local bipolar closed-form`。",
        "",
        "## 结果",
        "",
    ]
    if not per_seed_df.empty:
        for dataset, ds in per_seed_df.groupby("dataset"):
            lines.append(f"### {dataset}")
            lines.append("")
            raw_available = bool(ds["raw_minirocket_reference_available"].fillna(False).astype(bool).any())
            if raw_available:
                raw_vals = ds.loc[ds["raw_minirocket_reference_available"] == True, "raw_minirocket_reference_test_macro_f1"].tolist()
                raw_sources = sorted(set(str(v) for v in ds["raw_minirocket_reference_source"].dropna().tolist() if str(v)))
                lines.append(f"- `raw_minirocket_reference`: {_format_mean_std(raw_vals)}")
                lines.append(f"- `raw_minirocket_reference_source`: `{', '.join(raw_sources)}`")
            lines.append(f"- `baseline_0`: {_format_mean_std(ds['baseline_0_test_macro_f1'].tolist())}")
            lines.append(f"- `f1_global_mainline`: {_format_mean_std(ds['f1_global_mainline_test_macro_f1'].tolist())}")
            lines.append(f"- `p1a_s1_offline_local_wls`: {_format_mean_std(ds['p1a_s1_offline_local_wls_test_macro_f1'].tolist())}")
            lines.append(f"- `p1a_delta_vs_baseline`: {_format_mean_std(ds['p1a_delta_vs_baseline'].tolist())}")
            lines.append(f"- `p1a_delta_vs_f1`: {_format_mean_std(ds['p1a_delta_vs_f1'].tolist())}")
            if raw_available:
                lines.append(f"- `baseline_delta_vs_raw_minirocket`: {_format_mean_std(ds['baseline_delta_vs_raw_minirocket'].dropna().tolist())}")
                lines.append(f"- `f1_delta_vs_raw_minirocket`: {_format_mean_std(ds['f1_delta_vs_raw_minirocket'].dropna().tolist())}")
                lines.append(f"- `p1a_delta_vs_raw_minirocket`: {_format_mean_std(ds['p1a_delta_vs_raw_minirocket'].dropna().tolist())}")
            lines.append(f"- `p1a_local_pair_axis_router_cosine_mean`: {_format_mean_std(ds['p1a_local_pair_axis_router_cosine_mean'].tolist())}")
            lines.append(f"- `p1a_local_operator_runtime_seconds`: {_format_mean_std(ds['p1a_local_operator_runtime_seconds'].tolist())}")
            lines.append("")

    with open(conclusion_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")

    print(config_path)
    print(per_seed_path)
    print(structure_path)
    print(response_path)
    print(conclusion_path)


if __name__ == "__main__":
    main()
