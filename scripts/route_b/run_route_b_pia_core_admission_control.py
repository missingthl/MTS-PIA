#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "scripts"))

from datasets.trial_dataset_factory import (  # noqa: E402
    DEFAULT_NATOPS_ROOT,
    DEFAULT_SELFREGULATIONSCP1_ROOT,
    load_trials_for_dataset,
)
from route_b_unified import (  # noqa: E402
    BridgeConfig,
    MiniRocketEvalConfig,
    PIACore,
    PIACoreConfig,
    PolicyAction,
    TargetRoundState,
    apply_bridge,
    evaluate_bridge,
)
from route_b_unified.augmentation_admission import (  # noqa: E402
    HybridAdmissionConfig,
    HybridAdmissionResult,
    apply_hybrid_admission,
)
from route_b_unified.types import BridgeResult, RepresentationState  # noqa: E402
from scripts.support.protocol_split_utils import resolve_protocol_split  # noqa: E402
from scripts.route_b.run_route_b_main_matrix import _build_eval_cfg  # noqa: E402
from scripts.route_b.run_route_b_pia_core_config_sweep import (  # noqa: E402
    REFERENCE_MATRIX_CSV,
    _build_rep_state_from_trials,
    _load_reference_baselines,
    _mech_global_metrics,
    _reference_metric,
)


FORMAL_CONFIG_TABLE_CSV = (
    "/home/THL/project/MTS-PIA/out/route_b_pia_core_config_sweep_20260327_formal/pia_core_config_table.csv"
)


@dataclass(frozen=True)
class BasePIAConfigPoint:
    config_id: str
    gamma_main: float
    axis_count: int
    pullback_alpha: float
    operator_strength_label: str
    axis_count_label: str
    fidelity_constraint_label: str
    axis_selection_strategy: str


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


def _format_mean_std(values: Iterable[float]) -> str:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return "0.0000 +/- 0.0000"
    return f"{float(np.mean(arr)):.4f} +/- {float(np.std(arr)):.4f}"


def _json_text(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _sign_text(delta: float) -> str:
    if delta > 1e-9:
        return "+"
    if delta < -1e-9:
        return "-"
    return "0"


def _summary_stats(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def _load_base_config_points(path: str) -> Dict[str, BasePIAConfigPoint]:
    df = pd.read_csv(path)
    out: Dict[str, BasePIAConfigPoint] = {}
    for _, row in df.iterrows():
        out[str(row["config_id"])] = BasePIAConfigPoint(
            config_id=str(row["config_id"]),
            gamma_main=float(row["gamma_main"]),
            axis_count=int(row["axis_count"]),
            pullback_alpha=float(row["pullback_alpha"]),
            operator_strength_label=str(row["operator_strength_label"]),
            axis_count_label=str(row["axis_count_label"]),
            fidelity_constraint_label=str(row["fidelity_constraint_label"]),
            axis_selection_strategy=str(row.get("axis_selection_strategy", "top_projection_energy_desc")),
        )
    return out


def _base_config_ids_for_dataset(dataset: str, args: argparse.Namespace) -> List[str]:
    if str(dataset).lower() == "natops":
        return _parse_csv_list(args.natops_base_configs)
    if str(dataset).lower() == "selfregulationscp1":
        return _parse_csv_list(args.scp1_base_configs)
    raise ValueError(f"unsupported dataset for admission control: {dataset}")


def _covariance_from_trial_np(x: np.ndarray, eps: float) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64)
    xx = xx - xx.mean(axis=1, keepdims=True)
    denom = max(1, int(xx.shape[1]) - 1)
    cov = (xx @ xx.T) / float(denom)
    cov = 0.5 * (cov + cov.T)
    cov = cov + float(eps) * np.eye(cov.shape[0], dtype=np.float64)
    return cov.astype(np.float32)


def _mean_cov_by_class_from_rep(rep_state: RepresentationState) -> Dict[int, np.ndarray]:
    buckets: Dict[int, List[np.ndarray]] = {}
    for rec in rep_state.train_records:
        buckets.setdefault(int(rec.y), []).append(np.asarray(rec.sigma_orig, dtype=np.float64))
    return {int(k): np.mean(np.stack(v, axis=0), axis=0) for k, v in buckets.items()}


def _mean_cov_by_class_from_trials(trials: Sequence[Dict[str, object]], eps: float) -> Dict[int, np.ndarray]:
    buckets: Dict[int, List[np.ndarray]] = {}
    for trial in trials:
        cov = _covariance_from_trial_np(np.asarray(trial["x_trial"], dtype=np.float32), float(eps))
        buckets.setdefault(int(trial["label"]), []).append(np.asarray(cov, dtype=np.float64))
    return {int(k): np.mean(np.stack(v, axis=0), axis=0) for k, v in buckets.items() if len(v) > 0}


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


def _summarize_filtered_bridge(
    *,
    rep_state: RepresentationState,
    accepted_aug_trials: Sequence[Dict[str, object]],
    accepted_bridge_meta: Sequence[Mapping[str, object]],
    bridge_eps: float,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, float], str]:
    if not accepted_aug_trials:
        return (
            {
                "bridge_aug_count": 0,
                "train_selected_aug_ratio": 0.0,
                "bridge_cov_match_error_mean": 0.0,
                "bridge_cov_match_error_fro_mean": 0.0,
                "bridge_cov_match_error_logeuc_mean": 0.0,
                "bridge_cov_to_orig_distance_fro_mean": 0.0,
                "bridge_cov_to_orig_distance_logeuc_mean": 0.0,
                "energy_ratio_mean": 0.0,
                "cond_A_mean": 0.0,
                "raw_mean_shift_abs_mean": 0.0,
            },
            {
                "classwise_mean_shift_summary": {},
                "classwise_covariance_distortion_summary": {},
                "classwise_covariance_distortion_mean": 0.0,
            },
            {"orig_min": 0.0, "orig_mean": 0.0, "bridge_min": 0.0, "bridge_mean": 0.0, "delta_min": 0.0, "delta_mean": 0.0, "ratio_mean": 0.0},
            "no_aug_after_admission",
        )

    orig_class_covs = _mean_cov_by_class_from_rep(rep_state)
    bridge_class_covs = _mean_cov_by_class_from_trials(accepted_aug_trials, bridge_eps)
    classwise_shift: Dict[str, float] = {}
    shift_buckets: Dict[int, List[float]] = {}
    for meta in accepted_bridge_meta:
        shift_buckets.setdefault(int(meta.get("label", -1)), []).append(float(meta.get("raw_mean_shift_abs", 0.0)))
    for label, vals in sorted(shift_buckets.items(), key=lambda kv: kv[0]):
        classwise_shift[str(int(label))] = float(np.mean(vals)) if vals else 0.0
    classwise_cov_dist = {
        str(int(k)): float(np.linalg.norm(bridge_class_covs[k] - orig_class_covs[k], ord="fro"))
        for k in sorted(set(orig_class_covs.keys()) & set(bridge_class_covs.keys()))
    }
    cov_dist_mean = float(np.mean(list(classwise_cov_dist.values()))) if classwise_cov_dist else 0.0
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
    if cov_dist_mean <= 0.15 and float(margin_proxy["delta_mean"]) >= -0.05:
        task_risk = "classwise_stable_margin_preserved"
    elif float(margin_proxy["delta_mean"]) < -0.05:
        task_risk = "bridge_margin_shrink_risk"
    else:
        task_risk = "classwise_drift_watch"

    def _mean_from_meta(key: str) -> float:
        values = [float(meta.get(key, 0.0)) for meta in accepted_bridge_meta]
        return float(np.mean(values)) if values else 0.0

    global_fidelity = {
        "bridge_aug_count": int(len(accepted_aug_trials)),
        "train_selected_aug_ratio": float(len(accepted_aug_trials) / max(1, len(rep_state.train_records))),
        "bridge_cov_match_error_mean": _mean_from_meta("bridge_cov_match_error"),
        "bridge_cov_match_error_fro_mean": _mean_from_meta("bridge_cov_match_error_fro"),
        "bridge_cov_match_error_logeuc_mean": _mean_from_meta("bridge_cov_match_error_logeuc"),
        "bridge_cov_to_orig_distance_fro_mean": _mean_from_meta("bridge_cov_to_orig_distance_fro"),
        "bridge_cov_to_orig_distance_logeuc_mean": _mean_from_meta("bridge_cov_to_orig_distance_logeuc"),
        "energy_ratio_mean": _mean_from_meta("bridge_energy_ratio"),
        "cond_A_mean": _mean_from_meta("bridge_cond_A"),
        "raw_mean_shift_abs_mean": _mean_from_meta("raw_mean_shift_abs"),
    }
    classwise_fidelity = {
        "classwise_mean_shift_summary": classwise_shift,
        "classwise_covariance_distortion_summary": classwise_cov_dist,
        "classwise_covariance_distortion_mean": cov_dist_mean,
    }
    return global_fidelity, classwise_fidelity, margin_proxy, task_risk


def _build_target_state(
    *,
    rep_state: RepresentationState,
    pia_core: PIACore,
    cfg: BasePIAConfigPoint,
    linear_c: float,
    linear_class_weight: str,
    linear_max_iter: int,
    mech_knn_k: int,
    mech_max_aug: int,
    mech_max_real_ref: int,
    mech_max_real_query: int,
    seed: int,
) -> Tuple[TargetRoundState, Dict[str, object]]:
    ranked_axes, ranked_meta = pia_core.rank_axes_by_energy(rep_state.X_train)
    axis_ids = [int(v) for v in ranked_axes[: int(cfg.axis_count)]]
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
        aug_meta=dict(op_result.meta) | {
            "axis_ranking_full_train": ranked_axes,
            "axis_energy_vector_full_train": ranked_meta["axis_energy_vector"],
        },
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
            stop_reason="pia_core_admission_control",
        ),
        direction_state={int(a): "pia_core_affine_active" for a in axis_ids},
        direction_budget_score={int(a): 1.0 for a in axis_ids},
    )
    diag = {
        "selected_axis_ids": list(axis_ids),
        "gamma_vector": list(gamma_vec),
        "margin_drop_median": float(mech["margin_drop_median"]),
        "flip_rate": float(mech["flip_rate"]),
        "intrusion_rate": float(mech["intrusion_rate"]),
        "operator_semantics": str(op_result.meta["operator_semantics"]),
        "ranked_axes": ranked_axes,
        "ranked_axis_energy": ranked_meta["axis_energy_vector"],
    }
    return target_state, diag


def _evaluate_base_pia(
    *,
    dataset: str,
    seed: int,
    rep_state: RepresentationState,
    pia_core: PIACore,
    cfg: BasePIAConfigPoint,
    bridge_cfg: BridgeConfig,
    eval_cfg: MiniRocketEvalConfig,
    linear_c: float,
    linear_class_weight: str,
    linear_max_iter: int,
    mech_knn_k: int,
    mech_max_aug: int,
    mech_max_real_ref: int,
    mech_max_real_query: int,
) -> Tuple[TargetRoundState, Dict[str, object], BridgeResult, Dict[str, float]]:
    target_state, diag = _build_target_state(
        rep_state=rep_state,
        pia_core=pia_core,
        cfg=cfg,
        linear_c=linear_c,
        linear_class_weight=linear_class_weight,
        linear_max_iter=linear_max_iter,
        mech_knn_k=mech_knn_k,
        mech_max_aug=mech_max_aug,
        mech_max_real_ref=mech_max_real_ref,
        mech_max_real_query=mech_max_real_query,
        seed=seed,
    )
    bridge_result = apply_bridge(rep_state, target_state, bridge_cfg, variant=f"pia_core_{cfg.config_id}")
    posterior = evaluate_bridge(
        bridge_result,
        eval_cfg,
        split_name="test",
        target_state=target_state,
        round_gain_proxy=0.0,
    )
    metrics = {"acc": float(posterior.acc), "macro_f1": float(posterior.macro_f1)}
    return target_state, diag, bridge_result, metrics


def _evaluate_admission_variant(
    *,
    base_bridge_result: BridgeResult,
    rep_state: RepresentationState,
    target_state: TargetRoundState,
    eval_cfg: MiniRocketEvalConfig,
    admission_result: HybridAdmissionResult,
    bridge_eps: float,
    variant_name: str,
) -> Tuple[BridgeResult, Dict[str, float], Dict[str, object]]:
    global_fidelity, classwise_fidelity, margin_proxy, task_risk = _summarize_filtered_bridge(
        rep_state=rep_state,
        accepted_aug_trials=admission_result.accepted_aug_trials,
        accepted_bridge_meta=admission_result.accepted_bridge_meta,
        bridge_eps=float(bridge_eps),
    )
    updated_bridge = replace(
        base_bridge_result,
        variant=str(variant_name),
        train_trials=list(base_bridge_result.orig_train_trials) + list(admission_result.accepted_aug_trials),
        aug_train_trials=list(admission_result.accepted_aug_trials),
        per_aug_bridge_meta=list(admission_result.accepted_bridge_meta),
        global_fidelity=global_fidelity,
        classwise_fidelity=classwise_fidelity,
        margin_proxy=margin_proxy,
        task_risk_comment=str(task_risk),
        meta=dict(base_bridge_result.meta)
        | {
            "admission_gate_mode": str(admission_result.gate_mode),
            "admission_effective_keep_ratio": float(admission_result.effective_keep_ratio),
            "admission_threshold_setting": str(admission_result.threshold_setting),
        },
    )
    posterior = evaluate_bridge(
        updated_bridge,
        eval_cfg,
        split_name="test",
        target_state=target_state,
        round_gain_proxy=0.0,
    )
    return updated_bridge, {"acc": float(posterior.acc), "macro_f1": float(posterior.macro_f1)}, {
        "admission_effective_keep_ratio": float(admission_result.effective_keep_ratio),
        "admission_threshold_setting": str(admission_result.threshold_setting),
    }


def _accepted_mechanism_metrics(
    *,
    rep_state: RepresentationState,
    target_state: TargetRoundState,
    accepted_indices: Sequence[int],
    seed: int,
    linear_c: float,
    linear_class_weight: str,
    linear_max_iter: int,
    mech_knn_k: int,
    mech_max_aug: int,
    mech_max_real_ref: int,
    mech_max_real_query: int,
) -> Dict[str, float]:
    idx = np.asarray(list(accepted_indices), dtype=np.int64)
    if idx.size == 0:
        return {
            "margin_drop_median": 0.0,
            "flip_rate": 0.0,
            "intrusion_rate": 0.0,
            "real_intrusion_rate": 0.0,
            "delta_intrusion": 0.0,
        }
    mech = _mech_global_metrics(
        X_train=np.asarray(rep_state.X_train[idx], dtype=np.float32),
        y_train=np.asarray(rep_state.y_train[idx], dtype=np.int64),
        X_aug=np.asarray(target_state.z_aug[idx], dtype=np.float32),
        seed=int(seed),
        linear_c=float(linear_c),
        linear_class_weight=str(linear_class_weight),
        linear_max_iter=int(linear_max_iter),
        mech_knn_k=int(mech_knn_k),
        mech_max_aug=int(mech_max_aug),
        mech_max_real_ref=int(mech_max_real_ref),
        mech_max_real_query=int(mech_max_real_query),
    )
    return mech


def _dataset_summary(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, base_pia_config, gate_mode, ratio_mode), df_sub in per_seed_df.groupby(
        ["dataset", "base_pia_config", "gate_mode", "ratio_mode"],
        sort=True,
    ):
        rows.append(
            {
                "dataset": str(dataset),
                "base_pia_config": str(base_pia_config),
                "gate_mode": str(gate_mode),
                "ratio_mode": str(ratio_mode),
                "raw_baseline_macro_f1": _format_mean_std(df_sub["raw_test_macro_f1"].tolist()),
                "bridge_multiround_macro_f1": _format_mean_std(df_sub["bridge_multiround_test_macro_f1"].tolist()),
                "base_pia_macro_f1": _format_mean_std(df_sub["base_pia_test_macro_f1"].tolist()),
                "admission_macro_f1": _format_mean_std(df_sub["admission_test_macro_f1"].tolist()),
                "delta_vs_raw_mean": float(df_sub["delta_vs_raw"].mean()),
                "delta_vs_base_pia_mean": float(df_sub["delta_vs_base_pia"].mean()),
                "delta_vs_bridge_multiround_mean": float(df_sub["delta_vs_bridge_multiround"].mean()),
                "accept_ratio_mean": float(df_sub["accept_ratio"].mean()),
                "accepted_aug_count_mean": float(df_sub["accepted_aug_count"].mean()),
                "rejected_aug_count_mean": float(df_sub["rejected_aug_count"].mean()),
                "per_seed_sign": ",".join(_sign_text(v) for v in df_sub["delta_vs_base_pia"].tolist()),
            }
        )
    return pd.DataFrame(rows)


def _mechanism_summary(mech_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, base_pia_config, gate_mode, ratio_mode), df_sub in mech_df.groupby(
        ["dataset", "base_pia_config", "gate_mode", "ratio_mode"],
        sort=True,
    ):
        rows.append(
            {
                "dataset": str(dataset),
                "base_pia_config": str(base_pia_config),
                "gate_mode": str(gate_mode),
                "ratio_mode": str(ratio_mode),
                "base_flip_rate_mean": float(df_sub["base_flip_rate"].mean()),
                "after_admission_flip_rate_mean": float(df_sub["after_admission_flip_rate"].mean()),
                "base_margin_drop_median_mean": float(df_sub["base_margin_drop_median"].mean()),
                "after_admission_margin_drop_median_mean": float(df_sub["after_admission_margin_drop_median"].mean()),
                "base_intrusion_mean": float(df_sub["base_intrusion"].mean()),
                "after_admission_intrusion_mean": float(df_sub["after_admission_intrusion"].mean()),
                "base_classwise_covariance_distortion_mean": float(
                    df_sub["base_classwise_covariance_distortion_mean"].mean()
                ),
                "after_admission_classwise_covariance_distortion_mean": float(
                    df_sub["after_admission_classwise_covariance_distortion_mean"].mean()
                ),
                "base_cond_A_mean": float(df_sub["base_cond_A"].mean()),
                "after_admission_cond_A_mean": float(df_sub["after_admission_cond_A"].mean()),
                "base_bridge_cov_to_orig_distance_mean": float(df_sub["base_bridge_cov_to_orig_distance_mean"].mean()),
                "after_admission_bridge_cov_to_orig_distance_mean": float(
                    df_sub["after_admission_bridge_cov_to_orig_distance_mean"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _config_table(per_seed_df: pd.DataFrame, cfg: HybridAdmissionConfig) -> pd.DataFrame:
    rows = []
    for (dataset, base_pia_config, gate_mode, ratio_mode, ratio_value), df_sub in per_seed_df.groupby(
        ["dataset", "base_pia_config", "gate_mode", "ratio_mode", "ratio_value"],
        sort=True,
    ):
        if str(gate_mode) == "base_pia":
            base_keep_ratio = ""
            eff_keep = "1.0000 +/- 0.0000"
            threshold_setting = "no_admission_gate"
            risk_version = "none"
        else:
            base_keep_ratio = (
                float(cfg.mild_base_keep_ratio) if str(gate_mode) == "mild" else float(cfg.strict_base_keep_ratio)
            )
            eff_keep = _format_mean_std(df_sub["effective_keep_ratio"].tolist())
            threshold_setting = (
                f"keep_lowest_hybrid_risk; threshold_mean={float(df_sub['threshold_value'].mean()):.6f}"
            )
            risk_version = str(cfg.risk_score_version)
        rows.append(
            {
                "dataset": str(dataset),
                "base_pia_config": str(base_pia_config),
                "gate_mode": str(gate_mode),
                "hybrid_risk_score_version": risk_version,
                "base_keep_ratio": base_keep_ratio,
                "effective_keep_ratio_summary": eff_keep,
                "threshold_setting": threshold_setting,
                "ratio_mode": str(ratio_mode),
                "ratio_value": float(ratio_value),
            }
        )
    return pd.DataFrame(rows)


def _write_conclusion(path: str, summary_df: pd.DataFrame, mech_df: pd.DataFrame) -> None:
    lines = [
        "# Admission Control Conclusion",
        "",
        "更新时间：2026-03-27",
        "",
        "- v1 结论口径：`hybrid admission`",
        "- 不是纯 sample-level flip/distortion gate。",
        "- group-level risk 通过 gate 强度（effective_keep_ratio）起作用。",
        "- after_admission 机制指标基于 filtered_aug_trials 重算。",
        "",
    ]

    nat = summary_df.loc[
        (summary_df["dataset"] == "natops") & (summary_df["gate_mode"] != "base_pia")
    ].copy()
    scp = summary_df.loc[
        (summary_df["dataset"] == "selfregulationscp1") & (summary_df["gate_mode"] != "base_pia")
    ].copy()
    mech_scp = mech_df.loc[
        (mech_df["dataset"] == "selfregulationscp1") & (mech_df["gate_mode"] != "base_pia")
    ].copy()

    useful = False
    for _, row in summary_df.iterrows():
        if str(row["gate_mode"]) != "base_pia" and float(row["delta_vs_base_pia_mean"]) >= 0.002:
            useful = True
            break

    scp_improved = False
    nat_preserved = False
    if not scp.empty:
        scp_best = scp.sort_values("delta_vs_base_pia_mean", ascending=False).iloc[0]
        scp_improved = float(scp_best["delta_vs_base_pia_mean"]) >= 0.002
        lines.append(
            f"- SCP1 最优 admission 变体：`{scp_best['base_pia_config']} / {scp_best['gate_mode']} / {scp_best['ratio_mode']}`，"
            f"`delta_vs_base_pia_mean={float(scp_best['delta_vs_base_pia_mean']):+.4f}`，"
            f"`delta_vs_raw_mean={float(scp_best['delta_vs_raw_mean']):+.4f}`。"
        )
    else:
        lines.append("- SCP1 当前没有真实 admission 变体可供比较。")
    if not nat.empty:
        nat_best = nat.sort_values("delta_vs_base_pia_mean", ascending=False).iloc[0]
        nat_preserved = float(nat_best["delta_vs_raw_mean"]) >= 0.002
        lines.append(
            f"- NATOPS 最优 admission 变体：`{nat_best['base_pia_config']} / {nat_best['gate_mode']} / {nat_best['ratio_mode']}`，"
            f"`delta_vs_base_pia_mean={float(nat_best['delta_vs_base_pia_mean']):+.4f}`，"
            f"`delta_vs_raw_mean={float(nat_best['delta_vs_raw_mean']):+.4f}`。"
        )
    if not mech_scp.empty:
        mech_best = mech_scp.sort_values(
            [
                "after_admission_classwise_covariance_distortion_mean",
                "after_admission_flip_rate_mean",
            ],
            ascending=[True, True],
        ).iloc[0]
        lines.append(
            f"- SCP1 机制改善最明显的 admission 变体：`{mech_best['base_pia_config']} / {mech_best['gate_mode']} / {mech_best['ratio_mode']}`，"
            f"`flip_rate {float(mech_best['base_flip_rate_mean']):.4f} -> {float(mech_best['after_admission_flip_rate_mean']):.4f}`，"
            f"`classwise_cov_dist {float(mech_best['base_classwise_covariance_distortion_mean']):.4f} -> "
            f"{float(mech_best['after_admission_classwise_covariance_distortion_mean']):.4f}`。"
        )
    else:
        lines.append("- SCP1 当前还没有 after_admission 机制重算结果。")

    if scp_improved and nat_preserved:
        verdict = "hybrid_admission_useful_now"
    elif useful:
        verdict = "hybrid_admission_partially_useful"
    else:
        verdict = "hybrid_admission_not_yet_useful"
    lines.extend(
        [
            "",
            f"- 判定：`{verdict}`",
        ]
    )

    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Hybrid admission/composition control on top of formal PIA Core candidates.")
    p.add_argument("--datasets", type=str, default="natops,selfregulationscp1")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/route_b_pia_core_admission_control_20260327")
    p.add_argument("--formal-config-table-csv", type=str, default=FORMAL_CONFIG_TABLE_CSV)
    p.add_argument("--reference-matrix-csv", type=str, default=REFERENCE_MATRIX_CSV)
    p.add_argument("--natops-base-configs", type=str, default="g100_k2_pb100,g100_k2_pb075")
    p.add_argument("--scp1-base-configs", type=str, default="g180_k1_pb075")
    p.add_argument("--bridge-eps", type=float, default=1e-4)
    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--val-fallback-fraction", type=float, default=0.25)
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
    p.add_argument("--mild-base-keep-ratio", type=float, default=0.85)
    p.add_argument("--strict-base-keep-ratio", type=float, default=0.60)
    p.add_argument("--min-keep-ratio", type=float, default=0.20)
    p.add_argument("--max-keep-ratio", type=float, default=0.95)
    p.add_argument("--lambda-flip", type=float, default=0.20)
    p.add_argument("--lambda-distortion", type=float, default=0.20)
    p.add_argument("--bridge-cond-weight", type=float, default=0.5)
    p.add_argument("--bridge-covdist-weight", type=float, default=0.5)
    p.add_argument("--risk-weight-bridge-sample", type=float, default=1.0)
    p.add_argument("--risk-weight-flip-group", type=float, default=1.0)
    p.add_argument("--risk-weight-distortion-group", type=float, default=1.0)
    p.add_argument("--include-half-ratio", type=int, default=0)
    p.add_argument("--natops-root", type=str, default=DEFAULT_NATOPS_ROOT)
    p.add_argument("--selfregulationscp1-root", type=str, default=DEFAULT_SELFREGULATIONSCP1_ROOT)
    args = p.parse_args()

    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_seed_list(args.seeds)
    _ensure_dir(args.out_root)

    base_cfg_map = _load_base_config_points(str(args.formal_config_table_csv))
    ref_df = _load_reference_baselines(str(args.reference_matrix_csv))
    eval_cfg = _build_eval_cfg(args, str(args.out_root))
    bridge_cfg = BridgeConfig(eps=float(args.bridge_eps))
    pia_cfg = PIACoreConfig(
        r_dimension=int(args.pia_r_dimension),
        n_iters=int(args.pia_n_iters),
        C_repr=float(args.pia_c_repr),
        activation=str(args.pia_activation),
        bias_update_mode=str(args.pia_bias_update_mode),
        orthogonalize=bool(int(args.pia_orthogonalize)),
    )
    admission_cfg = HybridAdmissionConfig(
        bridge_cond_weight=float(args.bridge_cond_weight),
        bridge_covdist_weight=float(args.bridge_covdist_weight),
        risk_weight_bridge_sample=float(args.risk_weight_bridge_sample),
        risk_weight_flip_group=float(args.risk_weight_flip_group),
        risk_weight_distortion_group=float(args.risk_weight_distortion_group),
        lambda_flip=float(args.lambda_flip),
        lambda_distortion=float(args.lambda_distortion),
        mild_base_keep_ratio=float(args.mild_base_keep_ratio),
        strict_base_keep_ratio=float(args.strict_base_keep_ratio),
        min_keep_ratio=float(args.min_keep_ratio),
        max_keep_ratio=float(args.max_keep_ratio),
    )

    ratio_variants = [("orig_plus_100pct_filtered_aug", 1.0)]
    if int(args.include_half_ratio):
        ratio_variants.append(("orig_plus_50pct_filtered_aug", 0.5))

    per_seed_rows: List[Dict[str, object]] = []
    mechanism_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        config_ids = _base_config_ids_for_dataset(dataset, args)
        all_trials = load_trials_for_dataset(
            dataset=str(dataset),
            natops_root=str(args.natops_root),
            selfregulationscp1_root=str(args.selfregulationscp1_root),
        )
        for seed in seeds:
            print(f"[admission][{dataset}][seed={seed}] split_start", flush=True)
            train_full_trials, test_trials, split_meta = resolve_protocol_split(
                dataset=str(dataset),
                all_trials=list(all_trials),
                seed=int(seed),
                allow_random_fallback=False,
            )
            rep_full = _build_rep_state_from_trials(
                dataset=str(dataset),
                seed=int(seed),
                train_trials=train_full_trials,
                val_trials=[],
                test_trials=test_trials,
                spd_eps=float(args.spd_eps),
                protocol_type=str(split_meta.get("protocol_type", "")),
                protocol_note=str(split_meta.get("protocol_note", "")),
            )
            raw_test_acc_ref, raw_test_f1_ref = _reference_metric(
                ref_df,
                dataset=str(dataset),
                seed=int(seed),
                variant="raw_only",
            )
            bridge_test_acc_ref, bridge_test_f1_ref = _reference_metric(
                ref_df,
                dataset=str(dataset),
                seed=int(seed),
                variant="legacy_multiround_bridge",
            )
            full_pia = PIACore(replace(pia_cfg, seed=int(seed))).fit(rep_full.X_train)

            for base_cfg_id in config_ids:
                if base_cfg_id not in base_cfg_map:
                    raise KeyError(f"missing base config in formal config table: {base_cfg_id}")
                base_cfg = base_cfg_map[base_cfg_id]
                print(f"[admission][{dataset}][seed={seed}][{base_cfg_id}] base_start", flush=True)
                target_state, base_diag, base_bridge_result, base_metrics = _evaluate_base_pia(
                    dataset=str(dataset),
                    seed=int(seed),
                    rep_state=rep_full,
                    pia_core=full_pia,
                    cfg=base_cfg,
                    bridge_cfg=bridge_cfg,
                    eval_cfg=eval_cfg,
                    linear_c=float(args.linear_c),
                    linear_class_weight=str(args.linear_class_weight),
                    linear_max_iter=int(args.linear_max_iter),
                    mech_knn_k=int(args.mech_knn_k),
                    mech_max_aug=int(args.mech_max_aug_for_metrics),
                    mech_max_real_ref=int(args.mech_max_real_knn_ref),
                    mech_max_real_query=int(args.mech_max_real_knn_query),
                )
                base_flip = float(base_diag["flip_rate"])
                base_margin = float(base_diag["margin_drop_median"])
                base_intrusion = float(base_diag["intrusion_rate"])
                base_dist = float(base_bridge_result.classwise_fidelity["classwise_covariance_distortion_mean"])
                base_cond = float(base_bridge_result.global_fidelity["cond_A_mean"])
                base_covdist = float(base_bridge_result.global_fidelity["bridge_cov_to_orig_distance_logeuc_mean"])

                per_seed_rows.append(
                    {
                        "dataset": str(dataset),
                        "base_pia_config": str(base_cfg_id),
                        "config_id": str(base_cfg_id),
                        "gate_mode": "base_pia",
                        "ratio_mode": "orig_plus_100pct_filtered_aug",
                        "ratio_value": 1.0,
                        "seed": int(seed),
                        "raw_test_macro_f1": float(raw_test_f1_ref),
                        "bridge_multiround_test_macro_f1": float(bridge_test_f1_ref),
                        "base_pia_test_macro_f1": float(base_metrics["macro_f1"]),
                        "admission_test_macro_f1": float(base_metrics["macro_f1"]),
                        "delta_vs_raw": float(base_metrics["macro_f1"] - raw_test_f1_ref),
                        "delta_vs_base_pia": 0.0,
                        "delta_vs_bridge_multiround": float(base_metrics["macro_f1"] - bridge_test_f1_ref),
                        "accepted_aug_count": int(len(base_bridge_result.aug_train_trials)),
                        "rejected_aug_count": 0,
                        "accept_ratio": 1.0,
                        "effective_keep_ratio": 1.0,
                        "threshold_value": 0.0,
                    }
                )
                mechanism_rows.append(
                    {
                        "dataset": str(dataset),
                        "base_pia_config": str(base_cfg_id),
                        "gate_mode": "base_pia",
                        "ratio_mode": "orig_plus_100pct_filtered_aug",
                        "seed": int(seed),
                        "base_flip_rate": base_flip,
                        "after_admission_flip_rate": base_flip,
                        "base_margin_drop_median": base_margin,
                        "after_admission_margin_drop_median": base_margin,
                        "base_intrusion": base_intrusion,
                        "after_admission_intrusion": base_intrusion,
                        "base_classwise_covariance_distortion_mean": base_dist,
                        "after_admission_classwise_covariance_distortion_mean": base_dist,
                        "base_cond_A": base_cond,
                        "after_admission_cond_A": base_cond,
                        "base_bridge_cov_to_orig_distance_mean": base_covdist,
                        "after_admission_bridge_cov_to_orig_distance_mean": base_covdist,
                    }
                )

                for gate_mode in ("mild", "strict"):
                    for ratio_mode, ratio_value in ratio_variants:
                        print(
                            f"[admission][{dataset}][seed={seed}][{base_cfg_id}][{gate_mode}][{ratio_mode}] start",
                            flush=True,
                        )
                        admission = apply_hybrid_admission(
                            aug_trials=base_bridge_result.aug_train_trials,
                            per_aug_bridge_meta=base_bridge_result.per_aug_bridge_meta,
                            flip_rate_group=base_flip,
                            distortion_risk_group=base_dist,
                            gate_mode=str(gate_mode),
                            cfg=admission_cfg,
                            ratio_mode=str(ratio_mode),
                            ratio_value=float(ratio_value),
                        )
                        admission_variant = f"pia_core_{base_cfg_id}__{gate_mode}"
                        if str(ratio_mode) != "orig_plus_100pct_filtered_aug":
                            admission_variant += "__half"
                        updated_bridge, adm_metrics, adm_diag = _evaluate_admission_variant(
                            base_bridge_result=base_bridge_result,
                            rep_state=rep_full,
                            target_state=target_state,
                            eval_cfg=eval_cfg,
                            admission_result=admission,
                            bridge_eps=float(args.bridge_eps),
                            variant_name=admission_variant,
                        )
                        after_mech = _accepted_mechanism_metrics(
                            rep_state=rep_full,
                            target_state=target_state,
                            accepted_indices=admission.accepted_indices,
                            seed=int(seed),
                            linear_c=float(args.linear_c),
                            linear_class_weight=str(args.linear_class_weight),
                            linear_max_iter=int(args.linear_max_iter),
                            mech_knn_k=int(args.mech_knn_k),
                            mech_max_aug=int(args.mech_max_aug_for_metrics),
                            mech_max_real_ref=int(args.mech_max_real_knn_ref),
                            mech_max_real_query=int(args.mech_max_real_knn_query),
                        )
                        threshold_value = 0.0
                        if str(admission.threshold_setting).startswith("keep_lowest_hybrid_risk_until_threshold="):
                            try:
                                threshold_value = float(str(admission.threshold_setting).split("=")[-1])
                            except Exception:
                                threshold_value = 0.0
                        per_seed_rows.append(
                            {
                                "dataset": str(dataset),
                                "base_pia_config": str(base_cfg_id),
                                "config_id": f"{base_cfg_id}__{gate_mode}" + ("__half" if str(ratio_mode) != "orig_plus_100pct_filtered_aug" else ""),
                                "gate_mode": str(gate_mode),
                                "ratio_mode": str(ratio_mode),
                                "ratio_value": float(ratio_value),
                                "seed": int(seed),
                                "raw_test_macro_f1": float(raw_test_f1_ref),
                                "bridge_multiround_test_macro_f1": float(bridge_test_f1_ref),
                                "base_pia_test_macro_f1": float(base_metrics["macro_f1"]),
                                "admission_test_macro_f1": float(adm_metrics["macro_f1"]),
                                "delta_vs_raw": float(adm_metrics["macro_f1"] - raw_test_f1_ref),
                                "delta_vs_base_pia": float(adm_metrics["macro_f1"] - base_metrics["macro_f1"]),
                                "delta_vs_bridge_multiround": float(adm_metrics["macro_f1"] - bridge_test_f1_ref),
                                "accepted_aug_count": int(len(admission.accepted_aug_trials)),
                                "rejected_aug_count": int(len(admission.rejected_aug_trials)),
                                "accept_ratio": float(admission.summary["accept_ratio"]),
                                "effective_keep_ratio": float(admission.effective_keep_ratio),
                                "threshold_value": float(threshold_value),
                            }
                        )
                        mechanism_rows.append(
                            {
                                "dataset": str(dataset),
                                "base_pia_config": str(base_cfg_id),
                                "gate_mode": str(gate_mode),
                                "ratio_mode": str(ratio_mode),
                                "seed": int(seed),
                                "base_flip_rate": base_flip,
                                "after_admission_flip_rate": float(after_mech["flip_rate"]),
                                "base_margin_drop_median": base_margin,
                                "after_admission_margin_drop_median": float(after_mech["margin_drop_median"]),
                                "base_intrusion": base_intrusion,
                                "after_admission_intrusion": float(after_mech["intrusion_rate"]),
                                "base_classwise_covariance_distortion_mean": base_dist,
                                "after_admission_classwise_covariance_distortion_mean": float(
                                    updated_bridge.classwise_fidelity["classwise_covariance_distortion_mean"]
                                ),
                                "base_cond_A": base_cond,
                                "after_admission_cond_A": float(updated_bridge.global_fidelity["cond_A_mean"]),
                                "base_bridge_cov_to_orig_distance_mean": base_covdist,
                                "after_admission_bridge_cov_to_orig_distance_mean": float(
                                    updated_bridge.global_fidelity["bridge_cov_to_orig_distance_logeuc_mean"]
                                ),
                            }
                        )

                        run_dir = os.path.join(args.out_root, str(dataset), f"seed{seed}", str(base_cfg_id), str(gate_mode), str(ratio_mode))
                        _ensure_dir(run_dir)
                        pd.DataFrame(admission.risk_rows).to_csv(
                            os.path.join(run_dir, "admission_risk_rows.csv"),
                            index=False,
                        )
                        with open(os.path.join(run_dir, "admission_result.json"), "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "dataset": str(dataset),
                                    "seed": int(seed),
                                    "base_pia_config": str(base_cfg_id),
                                    "gate_mode": str(gate_mode),
                                    "ratio_mode": str(ratio_mode),
                                    "ratio_value": float(ratio_value),
                                    "hybrid_risk_score_version": str(admission_cfg.risk_score_version),
                                    "base_metrics": base_metrics,
                                    "admission_metrics": adm_metrics,
                                    "admission_group_risk_summary": admission.group_risk_summary,
                                    "admission_summary": admission.summary,
                                    "admission_threshold_setting": str(admission.threshold_setting),
                                    "after_admission_bridge_global_fidelity": updated_bridge.global_fidelity,
                                    "after_admission_bridge_classwise_fidelity": updated_bridge.classwise_fidelity,
                                    "after_admission_margin_proxy": updated_bridge.margin_proxy,
                                    "after_admission_task_risk": updated_bridge.task_risk_comment,
                                },
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )

    per_seed_df = pd.DataFrame(per_seed_rows)
    mechanism_df = pd.DataFrame(mechanism_rows)
    summary_df = _dataset_summary(per_seed_df)
    config_df = _config_table(per_seed_df, admission_cfg)
    mechanism_summary_df = _mechanism_summary(mechanism_df)

    per_seed_df.to_csv(os.path.join(args.out_root, "admission_per_seed.csv"), index=False)
    summary_df.to_csv(os.path.join(args.out_root, "admission_dataset_summary.csv"), index=False)
    mechanism_summary_df.to_csv(os.path.join(args.out_root, "admission_mechanism_summary.csv"), index=False)
    config_df.to_csv(os.path.join(args.out_root, "admission_config_table.csv"), index=False)
    _write_conclusion(os.path.join(args.out_root, "admission_conclusion.md"), summary_df, mechanism_summary_df)


if __name__ == "__main__":
    main()
