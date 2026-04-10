#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
from route_b_unified.risk_aware_axis_controller import (  # noqa: E402
    DiscreteAxisScaleControllerConfig,
    FragilityProbeConfig,
    compute_class_fragility_scores,
    select_discrete_scales_from_fragility,
)
from route_b_unified.types import BridgeResult, RepresentationState  # noqa: E402
from scripts.fisher_pia_utils import FisherPIAConfig  # noqa: E402
from scripts.protocol_split_utils import resolve_protocol_split  # noqa: E402
from scripts.run_route_b_main_matrix import _build_eval_cfg  # noqa: E402
from scripts.run_route_b_pia_core_axis_refine import (  # noqa: E402
    FORMAL_CONFIG_TABLE_CSV,
    BasePIAConfigPoint,
    _evaluate_variant,
    _load_base_config_points,
)
from scripts.run_route_b_pia_core_config_sweep import (  # noqa: E402
    REFERENCE_MATRIX_CSV,
    _build_rep_state_from_trials,
    _load_reference_baselines,
    _mech_global_metrics,
    _reference_metric,
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


def _parse_float_triplet(text: str) -> Tuple[float, float, float]:
    vals = [float(tok.strip()) for tok in str(text).split(",") if tok.strip()]
    if len(vals) != 3:
        raise ValueError("expected exactly 3 scales")
    return float(vals[0]), float(vals[1]), float(vals[2])


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _format_mean_std(values: Iterable[float]) -> str:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return "0.0000 +/- 0.0000"
    return f"{float(np.mean(arr)):.4f} +/- {float(np.std(arr)):.4f}"


def _sign_text(delta: float) -> str:
    if delta > 1e-9:
        return "+"
    if delta < -1e-9:
        return "-"
    return "0"


def _dataset_static_ref_scale(dataset: str, *, natops_scale: float, scp1_scale: float) -> float:
    ds = str(dataset).lower()
    if ds == "natops":
        return float(natops_scale)
    if ds == "selfregulationscp1":
        return float(scp1_scale)
    raise ValueError(f"unsupported dataset: {dataset}")


def _config_table(rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    keep_cols = [
        "dataset",
        "base_pia_config",
        "fragility_probe_type",
        "fragility_probe_level",
        "controller_mode",
        "available_scales",
        "fixed_pullback_alpha",
        "static_reference_config",
        "config_id",
    ]
    return df[keep_cols].drop_duplicates().sort_values(["dataset", "config_id"]).reset_index(drop=True)


def _dataset_summary(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, df_sub in per_seed_df.groupby("dataset", sort=True):
        rows.append(
            {
                "dataset": str(dataset),
                "raw_baseline_macro_f1": _format_mean_std(df_sub["raw_test_macro_f1"].tolist()),
                "base_pia_macro_f1": _format_mean_std(df_sub["base_pia_test_macro_f1"].tolist()),
                "static_refined_macro_f1": _format_mean_std(df_sub["static_refined_test_macro_f1"].tolist()),
                "risk_aware_macro_f1": _format_mean_std(df_sub["risk_aware_test_macro_f1"].tolist()),
                "delta_vs_raw_mean": float(df_sub["delta_vs_raw"].mean()),
                "delta_vs_base_pia_mean": float(df_sub["delta_vs_base_pia"].mean()),
                "delta_vs_static_refined_mean": float(df_sub["delta_vs_static_refined"].mean()),
                "per_seed_sign_vs_static_refined": ",".join(
                    _sign_text(v) for v in df_sub["delta_vs_static_refined"].tolist()
                ),
            }
        )
    return pd.DataFrame(rows)


def _mechanism_summary(mech_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, df_sub in mech_df.groupby("dataset", sort=True):
        rows.append(
            {
                "dataset": str(dataset),
                "base_pia_config": str(df_sub["base_pia_config"].iloc[0]),
                "static_reference_config": str(df_sub["static_reference_config"].iloc[0]),
                "base_flip_rate_mean": float(df_sub["base_flip_rate"].mean()),
                "static_refined_flip_rate_mean": float(df_sub["static_refined_flip_rate"].mean()),
                "risk_aware_flip_rate_mean": float(df_sub["risk_aware_flip_rate"].mean()),
                "base_margin_drop_median_mean": float(df_sub["base_margin_drop_median"].mean()),
                "static_refined_margin_drop_median_mean": float(df_sub["static_refined_margin_drop_median"].mean()),
                "risk_aware_margin_drop_median_mean": float(df_sub["risk_aware_margin_drop_median"].mean()),
                "base_classwise_covariance_distortion_mean": float(
                    df_sub["base_classwise_covariance_distortion_mean"].mean()
                ),
                "static_refined_classwise_covariance_distortion_mean": float(
                    df_sub["static_refined_classwise_covariance_distortion_mean"].mean()
                ),
                "risk_aware_classwise_covariance_distortion_mean": float(
                    df_sub["risk_aware_classwise_covariance_distortion_mean"].mean()
                ),
                "base_cond_A_mean": float(df_sub["base_cond_A"].mean()),
                "static_refined_cond_A_mean": float(df_sub["static_refined_cond_A"].mean()),
                "risk_aware_cond_A_mean": float(df_sub["risk_aware_cond_A"].mean()),
                "risk_aware_selected_scale_patterns": json.dumps(
                    sorted(set(str(v) for v in df_sub["risk_aware_selected_scale_map"].tolist())),
                    ensure_ascii=False,
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_classwise_risk_aware_target_state(
    *,
    rep_state: RepresentationState,
    pia_core: PIACore,
    axis_ids: Sequence[int],
    gamma_main: float,
    pullback_alpha: float,
    class_scale_map: Dict[int, float],
    variant_note: str,
    seed: int,
    linear_c: float,
    linear_class_weight: str,
    linear_max_iter: int,
    mech_knn_k: int,
    mech_max_aug: int,
    mech_max_real_ref: int,
    mech_max_real_query: int,
) -> Tuple[TargetRoundState, Dict[str, object]]:
    X_train = np.asarray(rep_state.X_train, dtype=np.float64)
    y_train = np.asarray(rep_state.y_train, dtype=np.int64)
    tid_train = np.asarray(rep_state.tid_train, dtype=object)
    axis_ids_list = [int(v) for v in list(axis_ids)]

    X_aug = np.zeros_like(X_train, dtype=np.float32)
    class_scale_rows: List[Dict[str, object]] = []
    per_class_gamma_vectors: Dict[int, List[float]] = {}
    for cls in sorted(np.unique(y_train).tolist()):
        cls_i = int(cls)
        idx = np.where(y_train == cls_i)[0]
        scale = float(class_scale_map.get(cls_i, 0.75))
        gamma_vec, gamma_meta = pia_core.build_two_axis_gamma_vector(
            axis_ids=axis_ids_list,
            gamma_main=float(gamma_main),
            second_axis_scale=float(scale),
        )
        op_result = pia_core.apply_affine(
            X_train[idx],
            gamma_vector=gamma_vec,
            axis_ids=axis_ids_list,
            pullback_alpha=float(pullback_alpha),
        )
        X_aug[idx] = np.asarray(op_result.X_aug, dtype=np.float32)
        per_class_gamma_vectors[cls_i] = [float(v) for v in gamma_vec]
        class_scale_rows.append(
            {
                "class_id": cls_i,
                "selected_second_axis_scale": float(scale),
                "effective_second_axis_gamma": float(gamma_meta["effective_second_axis_gamma"]),
                "gamma_vector": [float(v) for v in gamma_vec],
            }
        )

    mech = _mech_global_metrics(
        X_train=X_train,
        y_train=y_train,
        X_aug=X_aug,
        seed=int(seed),
        linear_c=float(linear_c),
        linear_class_weight=str(linear_class_weight),
        linear_max_iter=int(linear_max_iter),
        mech_knn_k=int(mech_knn_k),
        mech_max_aug=int(mech_max_aug),
        mech_max_real_ref=int(mech_max_real_ref),
        mech_max_real_query=int(mech_max_real_query),
    )

    avg_second_gamma = float(np.mean([vals[1] if len(vals) > 1 else vals[0] for vals in per_class_gamma_vectors.values()]))
    target_state = TargetRoundState(
        round_index=1,
        z_aug=np.asarray(X_aug, dtype=np.float32),
        y_aug=np.asarray(y_train, dtype=np.int64),
        tid_aug=np.asarray(tid_train, dtype=object),
        mech={
            "dir_profile": {
                "worst_dir_id": int(axis_ids_list[-1]),
                "dir_profile_summary": str(variant_note),
            }
        },
        dir_maps={
            "margin_drop_median": {int(axis_ids_list[-1]): float(mech["margin_drop_median"])},
            "flip_rate": {int(axis_ids_list[-1]): float(mech["flip_rate"])},
            "intrusion": {int(axis_ids_list[-1]): float(mech["intrusion_rate"])},
        },
        aug_meta={
            "operator_type": "pia_explicit_affine",
            "operator_semantics": "classwise_fragility_probe_discrete_axis2_scaling_with_residual_preservation",
            "selected_axis_ids": list(axis_ids_list),
            "gamma_main": float(gamma_main),
            "pullback_alpha": float(pullback_alpha),
            "axis_strength_mode": "risk_aware_discrete_axis2_v1",
            "class_scale_rows": class_scale_rows,
            "class_scale_map": {str(k): float(v) for k, v in class_scale_map.items()},
        },
        action=PolicyAction(
            round_index=1,
            selected_dir_ids=list(axis_ids_list),
            direction_weights={int(a): float(1.0 / len(axis_ids_list)) for a in axis_ids_list},
            step_sizes={int(axis_ids_list[0]): float(gamma_main), int(axis_ids_list[1]): float(avg_second_gamma)},
            direction_probs_vector=np.asarray(
                [1.0 / len(axis_ids_list) if i in axis_ids_list else 0.0 for i in range(pia_core.get_artifacts().directions.shape[0])],
                dtype=np.float64,
            ),
            gamma_vector=np.asarray(
                [
                    float(gamma_main) if i == axis_ids_list[0]
                    else float(avg_second_gamma) if i == axis_ids_list[1]
                    else 0.0
                    for i in range(pia_core.get_artifacts().directions.shape[0])
                ],
                dtype=np.float64,
            ),
            entropy=float(np.log(len(axis_ids_list))) if len(axis_ids_list) > 1 else 0.0,
            stop_flag=False,
            stop_reason="risk_aware_axis2",
        ),
        direction_state={int(a): "risk_aware_axis2_active" for a in axis_ids_list},
        direction_budget_score={int(a): 1.0 for a in axis_ids_list},
    )
    diag = {
        "margin_drop_median": float(mech["margin_drop_median"]),
        "flip_rate": float(mech["flip_rate"]),
        "intrusion_rate": float(mech["intrusion_rate"]),
        "selected_axis_ids": json.dumps(axis_ids_list, ensure_ascii=False),
        "gamma_main": float(gamma_main),
        "pullback_alpha": float(pullback_alpha),
        "operator_semantics": str(target_state.aug_meta["operator_semantics"]),
        "axis_strength_mode": str(target_state.aug_meta["axis_strength_mode"]),
        "class_scale_map": json.dumps(target_state.aug_meta["class_scale_map"], ensure_ascii=False),
    }
    return target_state, diag


def _evaluate_risk_aware_variant(
    *,
    rep_state: RepresentationState,
    pia_core: PIACore,
    axis_ids: Sequence[int],
    gamma_main: float,
    pullback_alpha: float,
    class_scale_map: Dict[int, float],
    variant_name: str,
    seed: int,
    bridge_cfg: BridgeConfig,
    eval_cfg: MiniRocketEvalConfig,
    linear_c: float,
    linear_class_weight: str,
    linear_max_iter: int,
    mech_knn_k: int,
    mech_max_aug: int,
    mech_max_real_ref: int,
    mech_max_real_query: int,
) -> Tuple[Dict[str, float], Dict[str, object], BridgeResult]:
    target_state, diag = _build_classwise_risk_aware_target_state(
        rep_state=rep_state,
        pia_core=pia_core,
        axis_ids=axis_ids,
        gamma_main=gamma_main,
        pullback_alpha=pullback_alpha,
        class_scale_map=class_scale_map,
        variant_note=variant_name,
        seed=seed,
        linear_c=linear_c,
        linear_class_weight=linear_class_weight,
        linear_max_iter=linear_max_iter,
        mech_knn_k=mech_knn_k,
        mech_max_aug=mech_max_aug,
        mech_max_real_ref=mech_max_real_ref,
        mech_max_real_query=mech_max_real_query,
    )
    bridge_result = apply_bridge(rep_state, target_state, bridge_cfg, variant=str(variant_name))
    posterior = evaluate_bridge(
        bridge_result,
        eval_cfg,
        split_name="test",
        target_state=target_state,
        round_gain_proxy=0.0,
    )
    diag.update(
        {
            "classwise_covariance_distortion_mean": float(
                bridge_result.classwise_fidelity["classwise_covariance_distortion_mean"]
            ),
            "cond_A_mean": float(bridge_result.global_fidelity["cond_A_mean"]),
            "bridge_cov_to_orig_distance_mean": float(
                bridge_result.global_fidelity["bridge_cov_to_orig_distance_logeuc_mean"]
            ),
            "bridge_task_risk_comment": str(bridge_result.task_risk_comment),
        }
    )
    return {"acc": float(posterior.acc), "macro_f1": float(posterior.macro_f1)}, diag, bridge_result


def _write_conclusion(path: str, summary_df: pd.DataFrame, mech_df: pd.DataFrame) -> None:
    lines = [
        "# Risk-Aware Axis-2 Conclusion",
        "",
        "更新时间：2026-03-28",
        "",
    ]

    nat = summary_df.loc[summary_df["dataset"] == "natops"].copy()
    scp = summary_df.loc[summary_df["dataset"] == "selfregulationscp1"].copy()
    nat_row = nat.iloc[0] if not nat.empty else None
    scp_row = scp.iloc[0] if not scp.empty else None

    risk_aware_effective = False
    nat_preserved = False
    scp_improved = False
    closer_common = False
    residual = "insufficient_result"

    if nat_row is not None and scp_row is not None:
        nat_preserved = float(nat_row["delta_vs_raw_mean"]) >= 0.002
        scp_improved = float(scp_row["delta_vs_base_pia_mean"]) >= 0.002
        risk_aware_effective = (
            float(nat_row["delta_vs_static_refined_mean"]) >= -0.002
            and float(scp_row["delta_vs_static_refined_mean"]) > 0.0
        )
        closer_common = nat_preserved and scp_improved
        residual = "deeper_structure_or_axis_identity" if not closer_common else "not_applicable"

    lines.append(f"1. Dynamic Risk-Aware Scaling 是否有效：`{'yes' if risk_aware_effective else 'not_yet'}`")
    lines.append(f"2. 是否优于固定 second_axis_scale：`{'yes' if risk_aware_effective else 'not_yet'}`")
    lines.append(f"3. NATOPS 是否仍保正区：`{'yes' if nat_preserved else 'no'}`")
    lines.append(f"4. SCP1 是否进一步改善：`{'yes' if scp_improved else 'not_yet'}`")
    lines.append(f"5. 是否比当前静态第二包更接近共同候选点：`{'yes' if closer_common else 'not_yet'}`")
    lines.append(f"6. 当前剩余矛盾主要是什么：`{residual}`")

    if nat_row is not None:
        lines.append(
            f"   NATOPS：`delta_vs_raw_mean={float(nat_row['delta_vs_raw_mean']):+.4f}`，"
            f"`delta_vs_static_refined_mean={float(nat_row['delta_vs_static_refined_mean']):+.4f}`。"
        )
    if scp_row is not None:
        lines.append(
            f"   SCP1：`delta_vs_base_pia_mean={float(scp_row['delta_vs_base_pia_mean']):+.4f}`，"
            f"`delta_vs_static_refined_mean={float(scp_row['delta_vs_static_refined_mean']):+.4f}`。"
        )
    if not mech_df.empty:
        scp_mech = mech_df.loc[mech_df["dataset"] == "selfregulationscp1"]
        if not scp_mech.empty:
            row = scp_mech.iloc[0]
            lines.append(
                f"   SCP1 机制：`flip {float(row['static_refined_flip_rate_mean']):.4f}->{float(row['risk_aware_flip_rate_mean']):.4f}`，"
                f"`classwise_cov_dist {float(row['static_refined_classwise_covariance_distortion_mean']):.4f}->{float(row['risk_aware_classwise_covariance_distortion_mean']):.4f}`。"
            )

    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Dynamic risk-aware axis-2 scaling with discrete gear selection.")
    p.add_argument("--datasets", type=str, default="natops,selfregulationscp1")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/route_b_pia_core_risk_aware_axis2_20260328_formal")
    p.add_argument("--base-config-id", type=str, default="g100_k2_pb100")
    p.add_argument("--formal-config-table-csv", type=str, default=FORMAL_CONFIG_TABLE_CSV)
    p.add_argument("--reference-matrix-csv", type=str, default=REFERENCE_MATRIX_CSV)
    p.add_argument("--available-scales", type=str, default="0.80,0.75,0.70")
    p.add_argument("--fixed-pullback-alpha", type=float, default=0.90)
    p.add_argument("--natops-static-scale", type=float, default=0.80)
    p.add_argument("--scp1-static-scale", type=float, default=0.70)
    p.add_argument("--fragility-alpha", type=float, default=1.0)
    p.add_argument("--fragility-beta", type=float, default=1.0)
    p.add_argument("--fragility-gamma", type=float, default=1.0)
    p.add_argument("--fragility-eps", type=float, default=1e-8)
    p.add_argument("--fisher-knn-k", type=int, default=20)
    p.add_argument("--fisher-interior-quantile", type=float, default=0.70)
    p.add_argument("--fisher-boundary-quantile", type=float, default=0.30)
    p.add_argument("--fisher-hetero-k", type=int, default=3)
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
    available_scales = _parse_float_triplet(args.available_scales)
    _ensure_dir(args.out_root)

    base_cfg_map = _load_base_config_points(str(args.formal_config_table_csv))
    if str(args.base_config_id) not in base_cfg_map:
        raise KeyError(f"missing base config in formal config table: {args.base_config_id}")
    base_cfg: BasePIAConfigPoint = base_cfg_map[str(args.base_config_id)]

    ref_df = _load_reference_baselines(str(args.reference_matrix_csv))
    eval_cfg: MiniRocketEvalConfig = _build_eval_cfg(args, str(args.out_root))
    bridge_cfg = BridgeConfig(eps=float(args.bridge_eps))
    pia_cfg = PIACoreConfig(
        r_dimension=int(args.pia_r_dimension),
        n_iters=int(args.pia_n_iters),
        C_repr=float(args.pia_c_repr),
        activation=str(args.pia_activation),
        bias_update_mode=str(args.pia_bias_update_mode),
        orthogonalize=bool(int(args.pia_orthogonalize)),
    )
    fisher_cfg = FisherPIAConfig(
        knn_k=int(args.fisher_knn_k),
        interior_quantile=float(args.fisher_interior_quantile),
        boundary_quantile=float(args.fisher_boundary_quantile),
        hetero_k=int(args.fisher_hetero_k),
    )
    probe_cfg = FragilityProbeConfig(
        alpha=float(args.fragility_alpha),
        beta=float(args.fragility_beta),
        gamma=float(args.fragility_gamma),
        eps=float(args.fragility_eps),
    )
    controller_cfg = DiscreteAxisScaleControllerConfig(available_scales=available_scales)

    config_rows: List[Dict[str, object]] = []
    per_seed_rows: List[Dict[str, object]] = []
    mechanism_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        static_ref_scale = _dataset_static_ref_scale(
            dataset,
            natops_scale=float(args.natops_static_scale),
            scp1_scale=float(args.scp1_static_scale),
        )
        static_ref_cfg = f"{base_cfg.config_id}__axis2s{int(round(static_ref_scale * 100)):03d}_pb{int(round(float(args.fixed_pullback_alpha) * 100)):03d}"
        config_rows.append(
            {
                "dataset": str(dataset),
                "base_pia_config": str(base_cfg.config_id),
                "fragility_probe_type": str(probe_cfg.probe_type),
                "fragility_probe_level": str(probe_cfg.probe_level),
                "controller_mode": str(controller_cfg.controller_mode),
                "available_scales": json.dumps([float(v) for v in available_scales], ensure_ascii=False),
                "fixed_pullback_alpha": float(args.fixed_pullback_alpha),
                "static_reference_config": str(static_ref_cfg),
                "config_id": "risk_aware_axis2",
            }
        )

        all_trials = load_trials_for_dataset(
            dataset=str(dataset),
            natops_root=str(args.natops_root),
            selfregulationscp1_root=str(args.selfregulationscp1_root),
        )
        for seed in seeds:
            print(f"[risk-aware-axis2][{dataset}][seed={seed}] split_start", flush=True)
            train_full_trials, test_trials, split_meta = resolve_protocol_split(
                dataset=str(dataset),
                all_trials=list(all_trials),
                seed=int(seed),
                allow_random_fallback=False,
            )
            rep_full: RepresentationState = _build_rep_state_from_trials(
                dataset=str(dataset),
                seed=int(seed),
                train_trials=train_full_trials,
                val_trials=[],
                test_trials=test_trials,
                spd_eps=float(args.spd_eps),
                protocol_type=str(split_meta.get("protocol_type", "")),
                protocol_note=str(split_meta.get("protocol_note", "")),
            )
            _raw_test_acc_ref, raw_test_f1_ref = _reference_metric(
                ref_df,
                dataset=str(dataset),
                seed=int(seed),
                variant="raw_only",
            )

            pia_core = PIACore(replace(pia_cfg, seed=int(seed))).fit(rep_full.X_train)
            ranked_axes, _rank_meta = pia_core.rank_axes_by_energy(rep_full.X_train)
            base_axis_ids = [int(v) for v in ranked_axes[:2]]
            if len(base_axis_ids) < 2:
                base_axis_ids = (base_axis_ids + [0, 1])[:2]

            base_gamma_vector, _base_gamma_meta = pia_core.build_two_axis_gamma_vector(
                axis_ids=base_axis_ids,
                gamma_main=float(base_cfg.gamma_main),
                second_axis_scale=1.0,
            )
            base_metrics, base_diag, _base_bridge = _evaluate_variant(
                rep_state=rep_full,
                pia_core=pia_core,
                axis_ids=base_axis_ids,
                gamma_vector=base_gamma_vector,
                pullback_alpha=float(base_cfg.pullback_alpha),
                variant_name=f"pia_core_{base_cfg.config_id}",
                seed=int(seed),
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

            static_gamma_vector, _static_gamma_meta = pia_core.build_two_axis_gamma_vector(
                axis_ids=base_axis_ids,
                gamma_main=float(base_cfg.gamma_main),
                second_axis_scale=float(static_ref_scale),
            )
            static_metrics, static_diag, _static_bridge = _evaluate_variant(
                rep_state=rep_full,
                pia_core=pia_core,
                axis_ids=base_axis_ids,
                gamma_vector=static_gamma_vector,
                pullback_alpha=float(args.fixed_pullback_alpha),
                variant_name=f"pia_core_{static_ref_cfg}",
                seed=int(seed),
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

            class_scores_raw, fragility_rows, _fragility_meta = compute_class_fragility_scores(
                rep_full.X_train,
                rep_full.y_train,
                fisher_cfg=fisher_cfg,
                probe_cfg=probe_cfg,
            )
            class_scale_map, controller_rows, _controller_meta = select_discrete_scales_from_fragility(
                class_scores_raw,
                controller_cfg=controller_cfg,
            )
            risk_metrics, risk_diag, _risk_bridge = _evaluate_risk_aware_variant(
                rep_state=rep_full,
                pia_core=pia_core,
                axis_ids=base_axis_ids,
                gamma_main=float(base_cfg.gamma_main),
                pullback_alpha=float(args.fixed_pullback_alpha),
                class_scale_map=class_scale_map,
                variant_name="pia_core_risk_aware_axis2",
                seed=int(seed),
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

            per_seed_rows.append(
                {
                    "dataset": str(dataset),
                    "config_id": "risk_aware_axis2",
                    "seed": int(seed),
                    "raw_test_macro_f1": float(raw_test_f1_ref),
                    "base_pia_test_macro_f1": float(base_metrics["macro_f1"]),
                    "static_refined_test_macro_f1": float(static_metrics["macro_f1"]),
                    "risk_aware_test_macro_f1": float(risk_metrics["macro_f1"]),
                    "delta_vs_raw": float(risk_metrics["macro_f1"]) - float(raw_test_f1_ref),
                    "delta_vs_base_pia": float(risk_metrics["macro_f1"]) - float(base_metrics["macro_f1"]),
                    "delta_vs_static_refined": float(risk_metrics["macro_f1"]) - float(static_metrics["macro_f1"]),
                }
            )
            mechanism_rows.append(
                {
                    "dataset": str(dataset),
                    "base_pia_config": str(base_cfg.config_id),
                    "static_reference_config": str(static_ref_cfg),
                    "seed": int(seed),
                    "base_flip_rate": float(base_diag["flip_rate"]),
                    "static_refined_flip_rate": float(static_diag["flip_rate"]),
                    "risk_aware_flip_rate": float(risk_diag["flip_rate"]),
                    "base_margin_drop_median": float(base_diag["margin_drop_median"]),
                    "static_refined_margin_drop_median": float(static_diag["margin_drop_median"]),
                    "risk_aware_margin_drop_median": float(risk_diag["margin_drop_median"]),
                    "base_classwise_covariance_distortion_mean": float(base_diag["classwise_covariance_distortion_mean"]),
                    "static_refined_classwise_covariance_distortion_mean": float(
                        static_diag["classwise_covariance_distortion_mean"]
                    ),
                    "risk_aware_classwise_covariance_distortion_mean": float(
                        risk_diag["classwise_covariance_distortion_mean"]
                    ),
                    "base_cond_A": float(base_diag["cond_A_mean"]),
                    "static_refined_cond_A": float(static_diag["cond_A_mean"]),
                    "risk_aware_cond_A": float(risk_diag["cond_A_mean"]),
                    "risk_aware_selected_scale_map": json.dumps(
                        {str(k): float(v) for k, v in class_scale_map.items()},
                        ensure_ascii=False,
                    ),
                    "fragility_rows": json.dumps(fragility_rows, ensure_ascii=False),
                    "controller_rows": json.dumps(controller_rows, ensure_ascii=False),
                }
            )

    config_df = _config_table(config_rows)
    per_seed_df = pd.DataFrame(per_seed_rows).sort_values(["dataset", "seed"]).reset_index(drop=True)
    summary_df = _dataset_summary(per_seed_df)
    mech_df = _mechanism_summary(pd.DataFrame(mechanism_rows))

    config_df.to_csv(os.path.join(args.out_root, "risk_aware_axis2_config_table.csv"), index=False)
    per_seed_df.to_csv(os.path.join(args.out_root, "risk_aware_axis2_per_seed.csv"), index=False)
    summary_df.to_csv(os.path.join(args.out_root, "risk_aware_axis2_dataset_summary.csv"), index=False)
    mech_df.to_csv(os.path.join(args.out_root, "risk_aware_axis2_mechanism_summary.csv"), index=False)
    _write_conclusion(os.path.join(args.out_root, "risk_aware_axis2_conclusion.md"), summary_df, mech_df)

    print(f"[done] wrote outputs to {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
