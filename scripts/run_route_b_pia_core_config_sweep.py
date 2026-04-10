#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "scripts"))

from route_b_unified import (  # noqa: E402
    BridgeConfig,
    MiniRocketEvalConfig,
    PIACore,
    PIACoreConfig,
    PolicyAction,
    RepresentationState,
    TargetRoundState,
    apply_bridge,
    evaluate_bridge,
)
from run_phase15_step1b_multidir_matrix import _compute_mech_metrics  # noqa: E402
from scripts.protocol_split_utils import resolve_inner_train_val_split, resolve_protocol_split  # noqa: E402
from scripts.run_raw_bridge_probe import TrialRecord, _apply_mean_log, _build_trial_records  # noqa: E402
from scripts.run_route_b_main_matrix import _build_eval_cfg  # noqa: E402
from datasets.trial_dataset_factory import (  # noqa: E402
    DEFAULT_NATOPS_ROOT,
    DEFAULT_SELFREGULATIONSCP1_ROOT,
    load_trials_for_dataset,
)


REFERENCE_MATRIX_CSV = (
    "/home/THL/project/MTS-PIA/out/route_b_main_matrix_20260326/route_b_main_matrix_per_seed.csv"
)


@dataclass(frozen=True)
class SweepConfig:
    config_id: str
    operator_strength_label: str
    gamma_main: float
    axis_count_label: str
    axis_count: int
    fidelity_constraint_label: str
    pullback_alpha: float
    axis_selection_strategy: str = "top_projection_energy_desc"


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


def _parse_float_list(text: str) -> List[float]:
    out = [float(tok.strip()) for tok in str(text).split(",") if tok.strip()]
    if not out:
        raise ValueError("float list cannot be empty")
    return out


def _parse_int_list(text: str) -> List[int]:
    out = [int(tok.strip()) for tok in str(text).split(",") if tok.strip()]
    if not out:
        raise ValueError("int list cannot be empty")
    return out


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _format_mean_std(values: Iterable[float]) -> str:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return "0.0000 +/- 0.0000"
    return f"{float(np.mean(arr)):.4f} +/- {float(np.std(arr)):.4f}"


def _result_label(delta_vs_raw: float) -> str:
    if delta_vs_raw >= 0.002:
        return "positive"
    if delta_vs_raw <= -0.002:
        return "negative"
    return "flat"


def _sign_text(delta: float) -> str:
    if delta > 1e-9:
        return "+"
    if delta < -1e-9:
        return "-"
    return "0"


def _adaptive_priority(param: str) -> str:
    high = {
        "operator_gamma_main",
        "operator_gamma_pattern",
        "operator_axis_count",
        "operator_axis_selection_strategy",
        "operator_pullback_alpha",
        "bridge_fidelity_safety_threshold",
    }
    medium = {
        "template_r_dimension",
        "template_c_repr",
        "template_n_iters",
        "operator_residual_preservation_strength",
        "bridge_eps",
    }
    if param in high:
        return "high_priority_adaptive_candidate"
    if param in medium:
        return "medium_priority"
    return "not_recommended_for_dynamic_adaptation"


def _record_to_trial_dict(rec: TrialRecord) -> Dict[str, object]:
    return {
        "trial_id_str": str(rec.tid),
        "label": int(rec.y),
        "x_trial": np.asarray(rec.x_raw, dtype=np.float32),
    }


def _records_to_trial_dicts(records: Sequence[TrialRecord]) -> List[Dict[str, object]]:
    return [_record_to_trial_dict(r) for r in records]


def _stack_feature(records: Sequence[TrialRecord]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.asarray([], dtype=object),
        )
    return (
        np.stack([np.asarray(r.z, dtype=np.float32) for r in records], axis=0),
        np.asarray([int(r.y) for r in records], dtype=np.int64),
        np.asarray([r.tid for r in records], dtype=object),
    )


def _build_rep_state_from_trials(
    *,
    dataset: str,
    seed: int,
    train_trials: Sequence[Dict[str, object]],
    val_trials: Sequence[Dict[str, object]],
    test_trials: Sequence[Dict[str, object]],
    spd_eps: float,
    protocol_type: str,
    protocol_note: str,
    mean_log_train: np.ndarray | None = None,
) -> RepresentationState:
    train_tmp, learned_mean_log = _build_trial_records(train_trials, float(spd_eps))
    mean_log = learned_mean_log if mean_log_train is None else np.asarray(mean_log_train, dtype=np.float32)
    val_tmp, _ = _build_trial_records(val_trials, float(spd_eps)) if len(val_trials) > 0 else ([], mean_log)
    test_tmp, _ = _build_trial_records(test_trials, float(spd_eps)) if len(test_trials) > 0 else ([], mean_log)

    train_records = _apply_mean_log(train_tmp, mean_log)
    val_records = _apply_mean_log(val_tmp, mean_log) if val_tmp else []
    test_records = _apply_mean_log(test_tmp, mean_log) if test_tmp else []

    X_train, y_train, tid_train = _stack_feature(train_records)
    X_val, y_val, tid_val = _stack_feature(val_records)
    X_test, y_test, tid_test = _stack_feature(test_records)

    return RepresentationState(
        dataset=str(dataset),
        seed=int(seed),
        split_meta={"protocol_type": str(protocol_type), "protocol_note": str(protocol_note)},
        mean_log_train=np.asarray(mean_log, dtype=np.float32),
        train_records=list(train_records),
        val_records=list(val_records),
        test_records=list(test_records),
        train_trial_dicts=_records_to_trial_dicts(train_records),
        val_trial_dicts=_records_to_trial_dicts(val_records),
        test_trial_dicts=_records_to_trial_dicts(test_records),
        X_train=X_train,
        y_train=y_train,
        tid_train=tid_train,
        X_val=X_val,
        y_val=y_val,
        tid_val=tid_val,
        X_test=X_test,
        y_test=y_test,
        tid_test=tid_test,
        meta={
            "protocol_type": str(protocol_type),
            "protocol_note": str(protocol_note),
            "train_trial_count": int(len(train_trials)),
            "val_trial_count": int(len(val_trials)),
            "test_trial_count": int(len(test_trials)),
        },
    )


def _raw_reference_target_state(z_dim: int) -> TargetRoundState:
    return TargetRoundState(
        round_index=0,
        z_aug=np.zeros((0, z_dim), dtype=np.float32),
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
    )


def _raw_reference_bridge_result(rep_state: RepresentationState):
    from route_b_unified.types import BridgeResult

    return BridgeResult(
        dataset=str(rep_state.dataset),
        seed=int(rep_state.seed),
        variant="raw_only",
        round_index=0,
        train_trials=list(rep_state.train_trial_dicts),
        val_trials=list(rep_state.val_trial_dicts),
        test_trials=list(rep_state.test_trial_dicts),
        global_fidelity={
            "bridge_cov_match_error_mean": 0.0,
            "bridge_cov_match_error_logeuc_mean": 0.0,
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
        margin_proxy={"delta_mean": 0.0, "ratio_mean": 1.0},
        task_risk_comment="raw_reference",
    )


def _load_reference_baselines(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dataset"] = df["dataset"].astype(str).str.lower()
    df["variant"] = df["variant"].astype(str)
    df["seed"] = df["seed"].astype(int)
    return df


def _reference_metric(df: pd.DataFrame, *, dataset: str, seed: int, variant: str) -> Tuple[float, float]:
    row = df.loc[
        (df["dataset"] == str(dataset).lower())
        & (df["seed"] == int(seed))
        & (df["variant"] == str(variant)),
        ["acc", "macro_f1"],
    ]
    if row.empty:
        raise KeyError(f"Missing reference metric for dataset={dataset}, seed={seed}, variant={variant}")
    return float(row.iloc[0]["acc"]), float(row.iloc[0]["macro_f1"])


def _label_strength(v: float) -> str:
    if v <= 0.051:
        return "conservative"
    if v <= 0.101:
        return "medium"
    return "aggressive"


def _label_axis_count(v: int) -> str:
    if int(v) <= 1:
        return "low_dof"
    if int(v) == 2:
        return "current_scale"
    return "higher_dof"


def _label_pullback(v: float) -> str:
    if float(v) >= 0.999:
        return "full_affine"
    if float(v) >= 0.74:
        return "moderate_pullback"
    return "strong_pullback"


def _build_sweep_configs(
    gamma_values: Sequence[float],
    axis_counts: Sequence[int],
    pullback_values: Sequence[float],
) -> List[SweepConfig]:
    out: List[SweepConfig] = []
    for g in gamma_values:
        for k in axis_counts:
            for pb in pullback_values:
                cfg_id = f"g{int(round(g * 1000)):03d}_k{int(k)}_pb{int(round(pb * 100)):03d}"
                out.append(
                    SweepConfig(
                        config_id=cfg_id,
                        operator_strength_label=_label_strength(float(g)),
                        gamma_main=float(g),
                        axis_count_label=_label_axis_count(int(k)),
                        axis_count=int(k),
                        fidelity_constraint_label=_label_pullback(float(pb)),
                        pullback_alpha=float(pb),
                    )
                )
    return out


def _mech_global_metrics(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_aug: np.ndarray,
    seed: int,
    linear_c: float,
    linear_class_weight: str,
    linear_max_iter: int,
    mech_knn_k: int,
    mech_max_aug: int,
    mech_max_real_ref: int,
    mech_max_real_query: int,
) -> Dict[str, object]:
    dir_zero = np.zeros((int(X_aug.shape[0]),), dtype=np.int64)
    mech = _compute_mech_metrics(
        X_train_real=np.asarray(X_train, dtype=np.float32),
        y_train_real=np.asarray(y_train, dtype=np.int64),
        X_aug_generated=np.asarray(X_aug, dtype=np.float32),
        y_aug_generated=np.asarray(y_train, dtype=np.int64),
        X_aug_accepted=np.asarray(X_aug, dtype=np.float32),
        y_aug_accepted=np.asarray(y_train, dtype=np.int64),
        X_src_accepted=np.asarray(X_train, dtype=np.float32),
        dir_generated=dir_zero,
        dir_accepted=dir_zero,
        seed=int(seed),
        linear_c=float(linear_c),
        class_weight=str(linear_class_weight),
        linear_max_iter=int(linear_max_iter),
        knn_k=int(mech_knn_k),
        max_aug_for_mech=int(mech_max_aug),
        max_real_knn_ref=int(mech_max_real_ref),
        max_real_knn_query=int(mech_max_real_query),
        progress_prefix=None,
    )
    return {
        "margin_drop_median": float(mech["margin_drop_median"]),
        "flip_rate": float(mech["flip_rate"]),
        "intrusion_rate": float(mech["knn_intrusion_rate"]),
        "real_intrusion_rate": float(mech["real_knn_intrusion_rate"]),
        "delta_intrusion": float(mech["delta_intrusion"]),
    }


def _evaluate_config(
    *,
    dataset: str,
    seed: int,
    rep_state: RepresentationState,
    pia_core: PIACore,
    selected_axis_ids: Sequence[int],
    cfg: SweepConfig,
    bridge_cfg: BridgeConfig,
    eval_cfg: MiniRocketEvalConfig,
    split_name: str,
    linear_c: float,
    linear_class_weight: str,
    linear_max_iter: int,
    mech_knn_k: int,
    mech_max_aug: int,
    mech_max_real_ref: int,
    mech_max_real_query: int,
) -> Tuple[Dict[str, object], Dict[str, object], object]:
    axis_ids = [int(v) for v in list(selected_axis_ids)[: int(cfg.axis_count)]]
    if not axis_ids:
        axis_ids = [0]
    gamma_vec = [float(cfg.gamma_main)] * len(axis_ids)
    op_result = pia_core.apply_affine(
        rep_state.X_train,
        gamma_vector=gamma_vec,
        axis_ids=axis_ids,
        pullback_alpha=float(cfg.pullback_alpha),
    )
    mech = _mech_global_metrics(
        X_train=rep_state.X_train,
        y_train=rep_state.y_train,
        X_aug=op_result.X_aug,
        seed=int(seed),
        linear_c=float(linear_c),
        linear_class_weight=str(linear_class_weight),
        linear_max_iter=int(linear_max_iter),
        mech_knn_k=int(mech_knn_k),
        mech_max_aug=int(mech_max_aug),
        mech_max_real_ref=int(mech_max_real_ref),
        mech_max_real_query=int(mech_max_real_query),
    )
    target_state = TargetRoundState(
        round_index=1,
        z_aug=np.asarray(op_result.X_aug, dtype=np.float32),
        y_aug=np.asarray(rep_state.y_train, dtype=np.int64),
        tid_aug=np.asarray(rep_state.tid_train, dtype=object),
        mech={
            "dir_profile": {
                "worst_dir_id": int(axis_ids[0]),
                "dir_profile_summary": f"pia_core_affine_cfg={cfg.config_id}",
            }
        },
        dir_maps={
            "margin_drop_median": {int(axis_ids[0]): float(mech["margin_drop_median"])},
            "flip_rate": {int(axis_ids[0]): float(mech["flip_rate"])},
            "intrusion": {int(axis_ids[0]): float(mech["intrusion_rate"])},
        },
        aug_meta=dict(op_result.meta),
        action=PolicyAction(
            round_index=1,
            selected_dir_ids=list(axis_ids),
            direction_weights={int(a): float(1.0 / len(axis_ids)) for a in axis_ids},
            step_sizes={int(a): float(cfg.gamma_main) for a in axis_ids},
            direction_probs_vector=np.asarray(
                [1.0 / len(axis_ids) if i in axis_ids else 0.0 for i in range(pia_core.get_artifacts().directions.shape[0])],
                dtype=np.float64,
            ),
            gamma_vector=np.asarray(
                [float(cfg.gamma_main) if i in axis_ids else 0.0 for i in range(pia_core.get_artifacts().directions.shape[0])],
                dtype=np.float64,
            ),
            entropy=float(np.log(len(axis_ids))) if len(axis_ids) > 1 else 0.0,
            stop_flag=False,
            stop_reason="pia_core_config_sweep",
        ),
        direction_state={int(a): "pia_core_affine_active" for a in axis_ids},
        direction_budget_score={int(a): 1.0 for a in axis_ids},
    )
    bridge_result = apply_bridge(rep_state, target_state, bridge_cfg, variant=f"pia_core_{cfg.config_id}")
    posterior = evaluate_bridge(
        bridge_result,
        eval_cfg,
        split_name=str(split_name),
        target_state=target_state,
        round_gain_proxy=0.0,
    )
    diag = {
        "config_id": cfg.config_id,
        "dataset": dataset,
        "seed": int(seed),
        "split_name": str(split_name),
        "selected_axis_ids": json.dumps(axis_ids, ensure_ascii=False),
        "gamma_vector": json.dumps(gamma_vec, ensure_ascii=False),
        "pullback_alpha": float(cfg.pullback_alpha),
        "operator_semantics": str(op_result.meta["operator_semantics"]),
        "margin_drop_median": float(mech["margin_drop_median"]),
        "flip_rate": float(mech["flip_rate"]),
        "intrusion_rate": float(mech["intrusion_rate"]),
        "bridge_cov_match_error": float(bridge_result.global_fidelity["bridge_cov_match_error_mean"]),
        "bridge_cov_to_orig_distance": float(bridge_result.global_fidelity["bridge_cov_to_orig_distance_logeuc_mean"]),
        "energy_ratio": float(bridge_result.global_fidelity["energy_ratio_mean"]),
        "cond_A": float(bridge_result.global_fidelity["cond_A_mean"]),
        "raw_mean_shift_abs": float(bridge_result.global_fidelity["raw_mean_shift_abs_mean"]),
        "classwise_covariance_distortion_mean": float(
            bridge_result.classwise_fidelity["classwise_covariance_distortion_mean"]
        ),
        "classwise_mean_shift_summary": json.dumps(
            bridge_result.classwise_fidelity["classwise_mean_shift_summary"], ensure_ascii=False
        ),
        "classwise_covariance_distortion_summary": json.dumps(
            bridge_result.classwise_fidelity["classwise_covariance_distortion_summary"], ensure_ascii=False
        ),
        "inter_class_margin_proxy": json.dumps(bridge_result.margin_proxy, ensure_ascii=False),
        "bridge_task_risk_comment": str(bridge_result.task_risk_comment),
        "aligned_energy_ratio_mean": float(op_result.meta["aligned_energy_ratio_mean"]),
        "residual_energy_mean": float(op_result.meta["residual_energy_mean"]),
    }
    metrics = {
        "acc": float(posterior.acc),
        "macro_f1": float(posterior.macro_f1),
        "metrics": dict(posterior.metrics),
    }
    return metrics, diag, bridge_result


def _config_rows(configs: Sequence[SweepConfig], pia_cfg: PIACoreConfig) -> pd.DataFrame:
    rows = []
    for cfg in configs:
        rows.append(
            {
                "config_id": cfg.config_id,
                "operator_strength_label": cfg.operator_strength_label,
                "gamma_main": float(cfg.gamma_main),
                "axis_count_label": cfg.axis_count_label,
                "axis_count": int(cfg.axis_count),
                "fidelity_constraint_label": cfg.fidelity_constraint_label,
                "pullback_alpha": float(cfg.pullback_alpha),
                "axis_selection_strategy": cfg.axis_selection_strategy,
                "template_r_dimension": int(pia_cfg.r_dimension),
                "template_n_iters": int(pia_cfg.n_iters),
                "template_c_repr": float(pia_cfg.C_repr),
                "template_activation": str(pia_cfg.activation),
                "template_bias_update_mode": str(pia_cfg.bias_update_mode),
                "template_orthogonalize": bool(pia_cfg.orthogonalize),
                "input_scaling_mode": "featurewise_minmax_to_activation_range",
                "residual_preservation_strength": 1.0,
                "center_mode": "train_mean",
            }
        )
    return pd.DataFrame(rows)


def _parameter_ledger(args: argparse.Namespace, pia_cfg: PIACoreConfig) -> pd.DataFrame:
    rows = [
        ("template_r_dimension", "template_learner", str(int(pia_cfg.r_dimension)), "fixed"),
        ("template_n_iters", "template_learner", str(int(pia_cfg.n_iters)), "fixed"),
        ("template_c_repr", "template_learner", str(float(pia_cfg.C_repr)), "fixed"),
        ("template_activation", "template_learner", str(pia_cfg.activation), "fixed"),
        ("template_input_scaling_mode", "template_learner", "featurewise_minmax_to_activation_range", "fixed"),
        ("template_orthogonalize", "template_learner", str(bool(pia_cfg.orthogonalize)), "fixed"),
        ("template_bias_update_mode", "template_learner", str(pia_cfg.bias_update_mode), "fixed"),
        ("operator_gamma_main", "pia_affine_operator", str(args.gamma_values), "scanned"),
        ("operator_gamma_pattern", "pia_affine_operator", "uniform_on_selected_axes", "fixed"),
        ("operator_axis_count", "pia_affine_operator", str(args.axis_counts), "scanned"),
        ("operator_axis_selection_strategy", "pia_affine_operator", "top_projection_energy_desc", "fixed"),
        ("operator_pullback_alpha", "pia_affine_operator", str(args.pullback_values), "scanned"),
        ("operator_residual_preservation_strength", "pia_affine_operator", "1.0", "fixed"),
        ("operator_center_mode", "pia_affine_operator", "train_mean", "fixed"),
        ("bridge_eps", "bridge", str(float(args.bridge_eps)), "fixed"),
        ("bridge_fidelity_safety_threshold", "bridge", "not_used_in_this_round_soft_diagnosis_only", "fixed"),
        ("bridge_raw_mean_writeback", "bridge", "original_trial_mean_restore", "fixed"),
        ("evaluator_window_sec", "evaluator", str(float(args.window_sec)), "fixed"),
        ("evaluator_hop_sec", "evaluator", str(float(args.hop_sec)), "fixed"),
        ("evaluator_aggregation_mode", "evaluator", str(args.aggregation_mode), "fixed"),
    ]
    out = []
    for name, layer, value, mode in rows:
        out.append(
            {
                "parameter_name": name,
                "layer": layer,
                "current_value_or_scan_values": value,
                "mode": mode,
                "adaptive_priority": _adaptive_priority(name),
            }
        )
    return pd.DataFrame(out)


def _dataset_result_summary(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, config_id), df_sub in per_seed_df.groupby(["dataset", "config_id"], sort=True):
        signs = ",".join(_sign_text(v) for v in df_sub["delta_vs_raw"].tolist())
        mean_delta = float(df_sub["delta_vs_raw"].mean())
        if mean_delta >= 0.002:
            aug_label = "positive"
        elif mean_delta <= -0.002:
            aug_label = "negative"
        else:
            aug_label = "flat"
        rows.append(
            {
                "dataset": str(dataset),
                "config_id": str(config_id),
                "raw_test_macro_f1": _format_mean_std(df_sub["raw_test_macro_f1"].tolist()),
                "bridge_multiround_test_macro_f1": _format_mean_std(df_sub["bridge_multiround_test_macro_f1"].tolist()),
                "pia_core_test_macro_f1": _format_mean_std(df_sub["pia_core_test_macro_f1"].tolist()),
                "delta_vs_raw_mean": mean_delta,
                "delta_vs_bridge_multiround_mean": float(df_sub["delta_vs_bridge_multiround"].mean()),
                "per_seed_sign": signs,
                "mean_test_acc": float(df_sub["pia_core_test_acc"].mean()),
                "std_test_acc": float(df_sub["pia_core_test_acc"].std(ddof=0)),
                "label": aug_label,
            }
        )
    return pd.DataFrame(rows)


def _mechanism_summary(diag_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, config_id), df_sub in diag_df.groupby(["dataset", "config_id"], sort=True):
        rows.append(
            {
                "dataset": str(dataset),
                "config_id": str(config_id),
                "margin_drop_median_mean": float(df_sub["margin_drop_median"].mean()),
                "flip_rate_mean": float(df_sub["flip_rate"].mean()),
                "intrusion_rate_mean": float(df_sub["intrusion_rate"].mean()),
                "bridge_cov_to_orig_distance_mean": float(df_sub["bridge_cov_to_orig_distance"].mean()),
                "cond_A_mean": float(df_sub["cond_A"].mean()),
                "raw_mean_shift_abs_mean": float(df_sub["raw_mean_shift_abs"].mean()),
                "classwise_covariance_distortion_mean": float(df_sub["classwise_covariance_distortion_mean"].mean()),
                "task_risk_mode": str(df_sub["bridge_task_risk_comment"].mode().iloc[0]) if not df_sub.empty else "n/a",
            }
        )
    return pd.DataFrame(rows)


def _write_conclusion(
    path: str,
    *,
    summary_df: pd.DataFrame,
    per_seed_df: pd.DataFrame,
) -> None:
    natops = summary_df.loc[summary_df["dataset"] == "natops"].copy()
    scp1 = summary_df.loc[summary_df["dataset"] == "selfregulationscp1"].copy()

    candidate_config = None
    if not natops.empty and not scp1.empty:
        merged = natops.merge(
            scp1[["config_id", "delta_vs_raw_mean", "label"]],
            on="config_id",
            suffixes=("_natops", "_scp1"),
        )
        eligible = merged.loc[
            (merged["delta_vs_raw_mean_natops"] >= 0.002)
            & (merged["delta_vs_raw_mean_scp1"] >= 0.005)
        ].copy()
        if not eligible.empty:
            eligible = eligible.sort_values(
                ["delta_vs_raw_mean_natops", "delta_vs_raw_mean_scp1"],
                ascending=[False, False],
            )
            candidate_config = str(eligible.iloc[0]["config_id"])

    if candidate_config is None:
        verdict = "no_formal_head_to_head_candidate_yet"
    else:
        verdict = "formal_head_to_head_candidate_found"

    lines = [
        "# PIA Core Config Sweep Conclusion",
        "",
        "更新时间：2026-03-27",
        "",
        f"- 当前是否已经出现值得进入正式主对比轮的 PIA Core 配置点：`{verdict}`",
    ]
    if candidate_config is not None:
        cand_nat = natops.loc[natops["config_id"] == candidate_config].iloc[0]
        cand_scp1 = scp1.loc[scp1["config_id"] == candidate_config].iloc[0]
        lines.extend(
            [
                f"- 推荐配置点：`{candidate_config}`",
                f"- NATOPS delta_vs_raw_mean：`{float(cand_nat['delta_vs_raw_mean']):+.4f}`",
                f"- NATOPS delta_vs_bridge_multiround_mean：`{float(cand_nat['delta_vs_bridge_multiround_mean']):+.4f}`",
                f"- SCP1 delta_vs_raw_mean：`{float(cand_scp1['delta_vs_raw_mean']):+.4f}`",
                f"- 推荐理由：`NATOPS 已进入正区，且 SCP1 正信号未被明显破坏`",
            ]
        )
    else:
        top_nat = natops.sort_values("delta_vs_raw_mean", ascending=False).head(3)
        lines.append("- 当前未出现同时满足 NATOPS 正区 + SCP1 稳定正信号的配置点。")
        if not top_nat.empty:
            lines.append("- NATOPS 当前最强的候选点：")
            for _, row in top_nat.iterrows():
                lines.append(
                    f"  - `{row['config_id']}`: delta_vs_raw_mean={float(row['delta_vs_raw_mean']):+.4f}, "
                    f"delta_vs_bridge_multiround_mean={float(row['delta_vs_bridge_multiround_mean']):+.4f}"
                )
        nat_bad = natops.sort_values("delta_vs_raw_mean", ascending=False).iloc[0] if not natops.empty else None
        scp_best = scp1.sort_values("delta_vs_raw_mean", ascending=False).iloc[0] if not scp1.empty else None
        if nat_bad is not None and scp_best is not None:
            lines.append(
                f"- 当前主要卡点更像：`NATOPS 对 operator 强度/轴自由度/回拉约束 的平衡仍未打稳；SCP1 已能稳定受益`"
            )

    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="PIA Core config sweep on NATOPS + SCP1 with official protocol alignment.")
    p.add_argument("--datasets", type=str, default="natops,selfregulationscp1")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/route_b_pia_core_config_sweep_20260327")
    p.add_argument("--reference-matrix-csv", type=str, default=REFERENCE_MATRIX_CSV)
    p.add_argument("--gamma-values", type=str, default="0.05,0.10,0.18")
    p.add_argument("--axis-counts", type=str, default="1,2,3")
    p.add_argument("--pullback-values", type=str, default="1.0,0.75")
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--val-fallback-fraction", type=float, default=0.25)
    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--bridge-eps", type=float, default=1e-4)
    p.add_argument("--pia-r-dimension", type=int, default=3)
    p.add_argument("--pia-n-iters", type=int, default=3)
    p.add_argument("--pia-c-repr", type=float, default=1.0)
    p.add_argument("--pia-activation", type=str, default="sine")
    p.add_argument("--pia-bias-update-mode", type=str, default="residual")
    p.add_argument("--pia-orthogonalize", type=int, default=1)
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
    p.add_argument("--linear-c", type=float, default=1.0)
    p.add_argument("--linear-class-weight", type=str, default="none")
    p.add_argument("--linear-max-iter", type=int, default=1000)
    p.add_argument("--mech-knn-k", type=int, default=20)
    p.add_argument("--mech-max-aug-for-metrics", type=int, default=2000)
    p.add_argument("--mech-max-real-knn-ref", type=int, default=10000)
    p.add_argument("--mech-max-real-knn-query", type=int, default=1000)
    p.add_argument("--natops-root", type=str, default=DEFAULT_NATOPS_ROOT)
    p.add_argument("--selfregulationscp1-root", type=str, default=DEFAULT_SELFREGULATIONSCP1_ROOT)
    args = p.parse_args()

    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_seed_list(args.seeds)
    gamma_values = _parse_float_list(args.gamma_values)
    axis_counts = _parse_int_list(args.axis_counts)
    pullback_values = _parse_float_list(args.pullback_values)
    _ensure_dir(args.out_root)

    pia_cfg = PIACoreConfig(
        r_dimension=int(args.pia_r_dimension),
        n_iters=int(args.pia_n_iters),
        C_repr=float(args.pia_c_repr),
        activation=str(args.pia_activation),
        bias_update_mode=str(args.pia_bias_update_mode),
        orthogonalize=bool(int(args.pia_orthogonalize)),
    )
    configs = _build_sweep_configs(gamma_values, axis_counts, pullback_values)
    config_df = _config_rows(configs, pia_cfg)
    config_df.to_csv(os.path.join(args.out_root, "pia_core_config_table.csv"), index=False)
    _parameter_ledger(args, pia_cfg).to_csv(os.path.join(args.out_root, "pia_core_parameter_ledger.csv"), index=False)

    ref_df = _load_reference_baselines(str(args.reference_matrix_csv))
    eval_cfg = _build_eval_cfg(args, str(args.out_root))
    bridge_cfg = BridgeConfig(eps=float(args.bridge_eps))

    per_seed_rows: List[Dict[str, object]] = []
    mechanism_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        all_trials = load_trials_for_dataset(
            dataset=str(dataset),
            natops_root=str(args.natops_root),
            selfregulationscp1_root=str(args.selfregulationscp1_root),
        )
        for seed in seeds:
            print(f"[pia-core-sweep][{dataset}][seed={seed}] split_start", flush=True)
            train_full_trials, test_trials, split_meta = resolve_protocol_split(
                dataset=str(dataset),
                all_trials=list(all_trials),
                seed=int(seed),
                allow_random_fallback=False,
            )
            train_core_trials, val_trials, inner_meta = resolve_inner_train_val_split(
                train_trials=train_full_trials,
                seed=int(seed) + 1701,
                val_fraction=float(args.val_fraction),
                fallback_fraction=float(args.val_fallback_fraction),
            )
            rep_core = _build_rep_state_from_trials(
                dataset=str(dataset),
                seed=int(seed),
                train_trials=train_core_trials,
                val_trials=val_trials,
                test_trials=test_trials,
                spd_eps=float(args.spd_eps),
                protocol_type=str(split_meta.get("protocol_type", "")),
                protocol_note=str(split_meta.get("protocol_note", "")),
            )
            rep_full = _build_rep_state_from_trials(
                dataset=str(dataset),
                seed=int(seed),
                train_trials=train_full_trials,
                val_trials=val_trials,
                test_trials=test_trials,
                spd_eps=float(args.spd_eps),
                protocol_type=str(split_meta.get("protocol_type", "")),
                protocol_note=str(split_meta.get("protocol_note", "")),
            )

            raw_core_val = evaluate_bridge(
                _raw_reference_bridge_result(rep_core),
                eval_cfg,
                split_name="val",
                target_state=_raw_reference_target_state(rep_core.X_train.shape[1]),
                round_gain_proxy=0.0,
            )
            raw_test_acc_ref, raw_test_f1_ref = _reference_metric(
                ref_df, dataset=str(dataset), seed=int(seed), variant="raw_only"
            )
            bridge_test_acc_ref, bridge_test_f1_ref = _reference_metric(
                ref_df, dataset=str(dataset), seed=int(seed), variant="legacy_multiround_bridge"
            )

            core_pia = PIACore(replace(pia_cfg, seed=int(seed))).fit(rep_core.X_train)
            full_pia = PIACore(replace(pia_cfg, seed=int(seed))).fit(rep_full.X_train)
            ranked_core_axes, ranked_core_meta = core_pia.rank_axes_by_energy(rep_core.X_train)
            ranked_full_axes, ranked_full_meta = full_pia.rank_axes_by_energy(rep_full.X_train)

            for cfg in configs:
                print(f"[pia-core-sweep][{dataset}][seed={seed}][{cfg.config_id}] start", flush=True)
                val_metrics, _val_diag, _ = _evaluate_config(
                    dataset=str(dataset),
                    seed=int(seed),
                    rep_state=rep_core,
                    pia_core=core_pia,
                    selected_axis_ids=ranked_core_axes,
                    cfg=cfg,
                    bridge_cfg=bridge_cfg,
                    eval_cfg=eval_cfg,
                    split_name="val",
                    linear_c=float(args.linear_c),
                    linear_class_weight=str(args.linear_class_weight),
                    linear_max_iter=int(args.linear_max_iter),
                    mech_knn_k=int(args.mech_knn_k),
                    mech_max_aug=int(args.mech_max_aug_for_metrics),
                    mech_max_real_ref=int(args.mech_max_real_knn_ref),
                    mech_max_real_query=int(args.mech_max_real_knn_query),
                )
                test_metrics, test_diag, bridge_result = _evaluate_config(
                    dataset=str(dataset),
                    seed=int(seed),
                    rep_state=rep_full,
                    pia_core=full_pia,
                    selected_axis_ids=ranked_full_axes,
                    cfg=cfg,
                    bridge_cfg=bridge_cfg,
                    eval_cfg=eval_cfg,
                    split_name="test",
                    linear_c=float(args.linear_c),
                    linear_class_weight=str(args.linear_class_weight),
                    linear_max_iter=int(args.linear_max_iter),
                    mech_knn_k=int(args.mech_knn_k),
                    mech_max_aug=int(args.mech_max_aug_for_metrics),
                    mech_max_real_ref=int(args.mech_max_real_knn_ref),
                    mech_max_real_query=int(args.mech_max_real_knn_query),
                )
                delta_vs_raw = float(test_metrics["macro_f1"]) - float(raw_test_f1_ref)
                delta_vs_bridge = float(test_metrics["macro_f1"]) - float(bridge_test_f1_ref)
                per_seed_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "config_id": str(cfg.config_id),
                        "operator_strength_label": str(cfg.operator_strength_label),
                        "gamma_main": float(cfg.gamma_main),
                        "axis_count": int(cfg.axis_count),
                        "pullback_alpha": float(cfg.pullback_alpha),
                        "protocol_type": str(split_meta.get("protocol_type", "")),
                        "raw_val_macro_f1": float(raw_core_val.macro_f1),
                        "pia_core_val_macro_f1": float(val_metrics["macro_f1"]),
                        "raw_test_acc": float(raw_test_acc_ref),
                        "raw_test_macro_f1": float(raw_test_f1_ref),
                        "bridge_multiround_test_acc": float(bridge_test_acc_ref),
                        "bridge_multiround_test_macro_f1": float(bridge_test_f1_ref),
                        "pia_core_test_acc": float(test_metrics["acc"]),
                        "pia_core_test_macro_f1": float(test_metrics["macro_f1"]),
                        "delta_vs_raw": delta_vs_raw,
                        "delta_vs_bridge_multiround": delta_vs_bridge,
                        "sign_vs_raw": _sign_text(delta_vs_raw),
                        "label": _result_label(delta_vs_raw),
                        "selected_axis_ids": test_diag["selected_axis_ids"],
                        "gamma_vector": test_diag["gamma_vector"],
                        "axis_ranking_full_train": json.dumps(ranked_full_axes, ensure_ascii=False),
                        "axis_energy_vector_full_train": json.dumps(
                            ranked_full_meta["axis_energy_vector"], ensure_ascii=False
                        ),
                        "bridge_task_risk_comment": str(test_diag["bridge_task_risk_comment"]),
                        "operator_semantics": str(test_diag["operator_semantics"]),
                    }
                )
                mechanism_rows.append(test_diag)

                run_dir = os.path.join(args.out_root, str(dataset), f"seed{seed}", str(cfg.config_id))
                _write_json(
                    os.path.join(run_dir, "config_eval_artifacts.json"),
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "config": cfg.__dict__,
                        "split_meta": dict(split_meta),
                        "inner_meta": dict(inner_meta),
                        "raw_val_macro_f1": float(raw_core_val.macro_f1),
                        "raw_test_macro_f1_reference": float(raw_test_f1_ref),
                        "bridge_multiround_macro_f1_reference": float(bridge_test_f1_ref),
                        "val_metrics": val_metrics,
                        "test_metrics": test_metrics,
                        "test_diag": test_diag,
                        "bridge_global_fidelity": dict(bridge_result.global_fidelity),
                        "bridge_classwise_fidelity": dict(bridge_result.classwise_fidelity),
                        "bridge_margin_proxy": dict(bridge_result.margin_proxy),
                    },
                )

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_path = os.path.join(args.out_root, "pia_core_config_sweep_per_seed.csv")
    per_seed_df.to_csv(per_seed_path, index=False)

    mechanism_df = pd.DataFrame(mechanism_rows)
    mechanism_path = os.path.join(args.out_root, "pia_core_mechanism_diagnostics_per_seed.csv")
    mechanism_df.to_csv(mechanism_path, index=False)

    summary_df = _dataset_result_summary(per_seed_df)
    summary_path = os.path.join(args.out_root, "pia_core_dataset_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    mech_summary_df = _mechanism_summary(mechanism_df)
    mech_summary_path = os.path.join(args.out_root, "pia_core_mechanism_summary.csv")
    mech_summary_df.to_csv(mech_summary_path, index=False)

    _write_conclusion(
        os.path.join(args.out_root, "pia_core_config_sweep_conclusion.md"),
        summary_df=summary_df,
        per_seed_df=per_seed_df,
    )
    print(f"[pia-core-sweep] wrote {per_seed_path}", flush=True)
    print(f"[pia-core-sweep] wrote {summary_path}", flush=True)
    print(f"[pia-core-sweep] wrote {mechanism_path}", flush=True)
    print(f"[pia-core-sweep] wrote {mech_summary_path}", flush=True)


if __name__ == "__main__":
    main()
