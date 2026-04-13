#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Sequence, Tuple

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


@dataclass(frozen=True)
class AxisRefinePoint:
    config_id: str
    base_pia_config: str
    gamma_main: float
    second_axis_scale: float
    pullback_alpha: float
    axis_count: int = 2
    axis_strength_mode: str = "two_axis_refine_v1"
    admission_reference: str = "none"


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


def _parse_float_list(text: str) -> List[float]:
    out = [float(tok.strip()) for tok in str(text).split(",") if tok.strip()]
    if not out:
        raise ValueError("float list cannot be empty")
    return out


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


def _base_configs_for_dataset(dataset: str, args: argparse.Namespace) -> List[str]:
    if str(dataset).lower() == "natops":
        return [tok.strip() for tok in str(args.natops_base_configs).split(",") if tok.strip()]
    if str(dataset).lower() == "selfregulationscp1":
        return [tok.strip() for tok in str(args.scp1_base_configs).split(",") if tok.strip()]
    raise ValueError(f"unsupported dataset: {dataset}")


def _stronger_pullback(base_pullback: float) -> float:
    if float(base_pullback) >= 0.999:
        return 0.85
    return max(0.50, float(base_pullback) - 0.15)


def _build_refine_points(
    *,
    base_cfg: BasePIAConfigPoint,
    second_axis_scales: Sequence[float],
    include_stronger_pullback: bool,
) -> List[AxisRefinePoint]:
    out: List[AxisRefinePoint] = []
    base_pb = float(base_cfg.pullback_alpha)
    stronger_pb = _stronger_pullback(base_pb)

    for scale in second_axis_scales:
        scale_f = float(scale)
        pb_values = [base_pb]
        if include_stronger_pullback and scale_f <= 0.50 + 1e-12 and abs(stronger_pb - base_pb) > 1e-12:
            pb_values.append(stronger_pb)
        for pb in pb_values:
            pb_code = int(round(float(pb) * 100))
            scale_code = int(round(scale_f * 100))
            cfg_id = f"{base_cfg.config_id}__axis2s{scale_code:03d}_pb{pb_code:03d}"
            out.append(
                AxisRefinePoint(
                    config_id=cfg_id,
                    base_pia_config=str(base_cfg.config_id),
                    gamma_main=float(base_cfg.gamma_main),
                    second_axis_scale=scale_f,
                    pullback_alpha=float(pb),
                    axis_count=2,
                )
            )
    return out


def _build_target_state(
    *,
    rep_state: RepresentationState,
    pia_core: PIACore,
    gamma_vector: Sequence[float],
    axis_ids: Sequence[int],
    pullback_alpha: float,
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
    gamma_vec = [float(v) for v in list(gamma_vector)]
    axis_ids_list = [int(v) for v in list(axis_ids)]
    op_result = pia_core.apply_affine(
        rep_state.X_train,
        gamma_vector=gamma_vec,
        axis_ids=axis_ids_list,
        pullback_alpha=float(pullback_alpha),
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
                "worst_dir_id": int(axis_ids_list[-1]),
                "dir_profile_summary": str(variant_note),
            }
        },
        dir_maps={
            "margin_drop_median": {int(axis_ids_list[-1]): float(mech["margin_drop_median"])},
            "flip_rate": {int(axis_ids_list[-1]): float(mech["flip_rate"])},
            "intrusion": {int(axis_ids_list[-1]): float(mech["intrusion_rate"])},
        },
        aug_meta=dict(op_result.meta),
        action=PolicyAction(
            round_index=1,
            selected_dir_ids=list(axis_ids_list),
            direction_weights={int(a): float(1.0 / len(axis_ids_list)) for a in axis_ids_list},
            step_sizes={int(axis_ids_list[i]): float(gamma_vec[i]) for i in range(len(axis_ids_list))},
            direction_probs_vector=np.asarray(
                [1.0 / len(axis_ids_list) if i in axis_ids_list else 0.0 for i in range(pia_core.get_artifacts().directions.shape[0])],
                dtype=np.float64,
            ),
            gamma_vector=np.asarray(
                [float(gamma_vec[axis_ids_list.index(i)]) if i in axis_ids_list else 0.0 for i in range(pia_core.get_artifacts().directions.shape[0])],
                dtype=np.float64,
            ),
            entropy=float(np.log(len(axis_ids_list))) if len(axis_ids_list) > 1 else 0.0,
            stop_flag=False,
            stop_reason="axis_refine",
        ),
        direction_state={int(a): "axis_refine_active" for a in axis_ids_list},
        direction_budget_score={int(a): 1.0 for a in axis_ids_list},
    )
    diag = {
        "margin_drop_median": float(mech["margin_drop_median"]),
        "flip_rate": float(mech["flip_rate"]),
        "intrusion_rate": float(mech["intrusion_rate"]),
        "selected_axis_ids": json.dumps(axis_ids_list, ensure_ascii=False),
        "gamma_vector": json.dumps(gamma_vec, ensure_ascii=False),
        "operator_semantics": str(op_result.meta["operator_semantics"]),
        "axis_strength_mode": str(op_result.meta.get("axis_strength_mode", "manual_gamma_vector")),
        "gamma_main": float(op_result.meta.get("gamma_main", gamma_vec[0] if gamma_vec else 0.0)),
        "second_axis_scale": float(op_result.meta.get("second_axis_scale", 1.0)),
        "effective_second_axis_gamma": float(op_result.meta.get("effective_second_axis_gamma", gamma_vec[1] if len(gamma_vec) > 1 else gamma_vec[0])),
        "pullback_alpha": float(op_result.meta.get("pullback_alpha", pullback_alpha)),
    }
    return target_state, diag


def _evaluate_variant(
    *,
    rep_state: RepresentationState,
    pia_core: PIACore,
    axis_ids: Sequence[int],
    gamma_vector: Sequence[float],
    pullback_alpha: float,
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
    target_state, diag = _build_target_state(
        rep_state=rep_state,
        pia_core=pia_core,
        gamma_vector=gamma_vector,
        axis_ids=axis_ids,
        pullback_alpha=pullback_alpha,
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


def _dataset_summary(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, base_pia_config, config_id), df_sub in per_seed_df.groupby(
        ["dataset", "base_pia_config", "config_id"],
        sort=True,
    ):
        rows.append(
            {
                "dataset": str(dataset),
                "base_pia_config": str(base_pia_config),
                "config_id": str(config_id),
                "raw_baseline_macro_f1": _format_mean_std(df_sub["raw_test_macro_f1"].tolist()),
                "base_pia_macro_f1": _format_mean_std(df_sub["base_pia_test_macro_f1"].tolist()),
                "axis_refined_macro_f1": _format_mean_std(df_sub["axis_refined_test_macro_f1"].tolist()),
                "delta_vs_raw_mean": float(df_sub["delta_vs_raw"].mean()),
                "delta_vs_base_pia_mean": float(df_sub["delta_vs_base_pia"].mean()),
                "per_seed_sign_vs_base_pia": ",".join(_sign_text(v) for v in df_sub["delta_vs_base_pia"].tolist()),
                "per_seed_sign_vs_raw": ",".join(_sign_text(v) for v in df_sub["delta_vs_raw"].tolist()),
            }
        )
    return pd.DataFrame(rows)


def _mechanism_summary(mech_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, base_pia_config, config_id), df_sub in mech_df.groupby(
        ["dataset", "base_pia_config", "config_id"],
        sort=True,
    ):
        rows.append(
            {
                "dataset": str(dataset),
                "base_pia_config": str(base_pia_config),
                "config_id": str(config_id),
                "base_flip_rate_mean": float(df_sub["base_flip_rate"].mean()),
                "after_axis_refine_flip_rate_mean": float(df_sub["after_flip_rate"].mean()),
                "base_margin_drop_median_mean": float(df_sub["base_margin_drop_median"].mean()),
                "after_axis_refine_margin_drop_median_mean": float(df_sub["after_margin_drop_median"].mean()),
                "base_classwise_covariance_distortion_mean": float(
                    df_sub["base_classwise_covariance_distortion_mean"].mean()
                ),
                "after_axis_refine_classwise_covariance_distortion_mean": float(
                    df_sub["after_classwise_covariance_distortion_mean"].mean()
                ),
                "base_cond_A_mean": float(df_sub["base_cond_A"].mean()),
                "after_axis_refine_cond_A_mean": float(df_sub["after_cond_A"].mean()),
                "base_bridge_cov_to_orig_distance_mean": float(df_sub["base_bridge_cov_to_orig_distance_mean"].mean()),
                "after_axis_refine_bridge_cov_to_orig_distance_mean": float(
                    df_sub["after_bridge_cov_to_orig_distance_mean"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _config_table(rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    keep_cols = [
        "dataset",
        "base_pia_config",
        "gamma_main",
        "second_axis_scale",
        "pullback_alpha",
        "axis_count",
        "admission_reference",
        "config_id",
    ]
    return df[keep_cols].drop_duplicates().sort_values(["dataset", "base_pia_config", "config_id"]).reset_index(drop=True)


def _write_conclusion(path: str, summary_df: pd.DataFrame, mech_df: pd.DataFrame) -> None:
    lines = [
        "# Axis Refine Conclusion",
        "",
        "更新时间：2026-03-27",
        "",
    ]

    nat = summary_df.loc[summary_df["dataset"] == "natops"].copy()
    scp = summary_df.loc[summary_df["dataset"] == "selfregulationscp1"].copy()
    mech_scp = mech_df.loc[mech_df["dataset"] == "selfregulationscp1"].copy()

    nat_best = nat.sort_values("delta_vs_base_pia_mean", ascending=False).iloc[0] if not nat.empty else None
    scp_best = scp.sort_values("delta_vs_base_pia_mean", ascending=False).iloc[0] if not scp.empty else None
    scp_mech_best = None
    if not mech_scp.empty:
        scp_mech_best = mech_scp.sort_values(
            [
                "after_axis_refine_classwise_covariance_distortion_mean",
                "after_axis_refine_flip_rate_mean",
            ],
            ascending=[True, True],
        ).iloc[0]

    axis_refine_effective = False
    natops_preserved = False
    scp_side_effect_reduced = False

    if nat_best is not None:
        natops_preserved = float(nat_best["delta_vs_raw_mean"]) >= 0.002
        axis_refine_effective = float(nat_best["delta_vs_base_pia_mean"]) >= 0.002
        lines.append(
            f"1. 第二轴单独收紧是否有效：`{'yes' if axis_refine_effective else 'not_yet'}`"
        )
        lines.append(
            f"   NATOPS 最优 refined 点：`{nat_best['config_id']}`，`delta_vs_base_pia_mean={float(nat_best['delta_vs_base_pia_mean']):+.4f}`。"
        )
    else:
        lines.append("1. 第二轴单独收紧是否有效：`insufficient_result`")

    lines.append(
        f"2. NATOPS 的正收益是否被保住：`{'yes' if natops_preserved else 'no'}`"
    )
    if nat_best is not None:
        lines.append(
            f"   `delta_vs_raw_mean={float(nat_best['delta_vs_raw_mean']):+.4f}`，`per_seed_sign_vs_raw={nat_best['per_seed_sign_vs_raw']}`。"
        )

    if scp_best is not None:
        scp_side_effect_reduced = float(scp_best["delta_vs_base_pia_mean"]) >= 0.002
        lines.append(
            f"3. SCP1 的副作用是否明显减轻：`{'yes' if scp_side_effect_reduced else 'not_yet'}`"
        )
        lines.append(
            f"   SCP1 最优 refined 点：`{scp_best['config_id']}`，`delta_vs_base_pia_mean={float(scp_best['delta_vs_base_pia_mean']):+.4f}`。"
        )
    else:
        lines.append("3. SCP1 的副作用是否明显减轻：`insufficient_result`")

    if scp_mech_best is not None:
        lines.append(
            f"   机制上最平衡的 SCP1 点：`{scp_mech_best['config_id']}`，"
            f"`flip {float(scp_mech_best['base_flip_rate_mean']):.4f}->{float(scp_mech_best['after_axis_refine_flip_rate_mean']):.4f}`，"
            f"`classwise_cov_dist {float(scp_mech_best['base_classwise_covariance_distortion_mean']):.4f}->{float(scp_mech_best['after_axis_refine_classwise_covariance_distortion_mean']):.4f}`。"
        )

    shared_candidate = False
    if nat_best is not None and scp_best is not None:
        shared_candidate = (
            float(nat_best["delta_vs_raw_mean"]) >= 0.002
            and float(scp_best["delta_vs_base_pia_mean"]) >= 0.0
        )
    lines.append(
        f"4. 是否比当前粗粒度 k=2 更接近共同候选点：`{'yes' if shared_candidate else 'not_yet'}`"
    )

    if not shared_candidate:
        lines.append(
            "   当前若仍未形成共同候选点，更像说明：`第二轴强度是风险源之一，但不是全部；SCP1 还存在更深层类条件结构问题。`"
        )

    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Minimal axis-level refinement around current strong 2-axis PIA configs.")
    p.add_argument("--datasets", type=str, default="natops,selfregulationscp1")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/route_b_pia_core_axis_refine_20260327")
    p.add_argument("--formal-config-table-csv", type=str, default=FORMAL_CONFIG_TABLE_CSV)
    p.add_argument("--reference-matrix-csv", type=str, default=REFERENCE_MATRIX_CSV)
    p.add_argument("--natops-base-configs", type=str, default="g100_k2_pb100,g100_k2_pb075")
    p.add_argument("--scp1-base-configs", type=str, default="g100_k2_pb100,g100_k2_pb075")
    p.add_argument("--second-axis-scales", type=str, default="0.75,0.50,0.25")
    p.add_argument("--include-stronger-pullback", type=int, default=1)
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
    second_axis_scales = _parse_float_list(args.second_axis_scales)
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

    config_rows: List[Dict[str, object]] = []
    per_seed_rows: List[Dict[str, object]] = []
    mechanism_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        all_trials = load_trials_for_dataset(
            dataset=str(dataset),
            natops_root=str(args.natops_root),
            selfregulationscp1_root=str(args.selfregulationscp1_root),
        )
        base_cfg_ids = _base_configs_for_dataset(dataset, args)
        for seed in seeds:
            print(f"[axis-refine][{dataset}][seed={seed}] split_start", flush=True)
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
            pia_core = PIACore(replace(pia_cfg, seed=int(seed))).fit(rep_full.X_train)
            ranked_axes, _rank_meta = pia_core.rank_axes_by_energy(rep_full.X_train)
            base_axis_ids = [int(v) for v in ranked_axes[:2]]
            if len(base_axis_ids) < 2:
                base_axis_ids = (base_axis_ids + [0, 1])[:2]

            for base_cfg_id in base_cfg_ids:
                if base_cfg_id not in base_cfg_map:
                    raise KeyError(f"missing base config in formal config table: {base_cfg_id}")
                base_cfg = base_cfg_map[base_cfg_id]
                base_gamma_vector, gamma_meta = pia_core.build_two_axis_gamma_vector(
                    axis_ids=base_axis_ids,
                    gamma_main=float(base_cfg.gamma_main),
                    second_axis_scale=1.0,
                )
                base_metrics, base_diag, base_bridge = _evaluate_variant(
                    rep_state=rep_full,
                    pia_core=pia_core,
                    axis_ids=base_axis_ids,
                    gamma_vector=base_gamma_vector,
                    pullback_alpha=float(base_cfg.pullback_alpha),
                    variant_name=f"pia_core_{base_cfg_id}",
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

                refine_points = _build_refine_points(
                    base_cfg=base_cfg,
                    second_axis_scales=second_axis_scales,
                    include_stronger_pullback=bool(int(args.include_stronger_pullback)),
                )
                for point in refine_points:
                    config_rows.append(
                        {
                            "dataset": str(dataset),
                            "base_pia_config": str(point.base_pia_config),
                            "gamma_main": float(point.gamma_main),
                            "second_axis_scale": float(point.second_axis_scale),
                            "pullback_alpha": float(point.pullback_alpha),
                            "axis_count": int(point.axis_count),
                            "admission_reference": str(point.admission_reference),
                            "config_id": str(point.config_id),
                        }
                    )
                    gamma_vector, gamma_meta = pia_core.build_two_axis_gamma_vector(
                        axis_ids=base_axis_ids,
                        gamma_main=float(point.gamma_main),
                        second_axis_scale=float(point.second_axis_scale),
                    )
                    print(
                        f"[axis-refine][{dataset}][seed={seed}][{point.config_id}] start",
                        flush=True,
                    )
                    refined_metrics, refined_diag, refined_bridge = _evaluate_variant(
                        rep_state=rep_full,
                        pia_core=pia_core,
                        axis_ids=base_axis_ids,
                        gamma_vector=gamma_vector,
                        pullback_alpha=float(point.pullback_alpha),
                        variant_name=f"pia_core_{point.config_id}",
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
                            "base_pia_config": str(point.base_pia_config),
                            "config_id": str(point.config_id),
                            "seed": int(seed),
                            "raw_test_macro_f1": float(raw_test_f1_ref),
                            "base_pia_test_macro_f1": float(base_metrics["macro_f1"]),
                            "axis_refined_test_macro_f1": float(refined_metrics["macro_f1"]),
                            "delta_vs_raw": float(refined_metrics["macro_f1"] - raw_test_f1_ref),
                            "delta_vs_base_pia": float(refined_metrics["macro_f1"] - base_metrics["macro_f1"]),
                            "second_axis_scale": float(point.second_axis_scale),
                            "pullback_alpha": float(point.pullback_alpha),
                            "gamma_main": float(point.gamma_main),
                            "flip_rate": float(refined_diag["flip_rate"]),
                            "margin_drop_median": float(refined_diag["margin_drop_median"]),
                            "intrusion_rate": float(refined_diag["intrusion_rate"]),
                            "classwise_covariance_distortion_mean": float(
                                refined_diag["classwise_covariance_distortion_mean"]
                            ),
                            "cond_A_mean": float(refined_diag["cond_A_mean"]),
                            "bridge_cov_to_orig_distance_mean": float(
                                refined_diag["bridge_cov_to_orig_distance_mean"]
                            ),
                            "selected_axis_ids": str(refined_diag["selected_axis_ids"]),
                            "gamma_vector": str(refined_diag["gamma_vector"]),
                        }
                    )
                    mechanism_rows.append(
                        {
                            "dataset": str(dataset),
                            "base_pia_config": str(point.base_pia_config),
                            "config_id": str(point.config_id),
                            "seed": int(seed),
                            "base_flip_rate": float(base_diag["flip_rate"]),
                            "after_flip_rate": float(refined_diag["flip_rate"]),
                            "base_margin_drop_median": float(base_diag["margin_drop_median"]),
                            "after_margin_drop_median": float(refined_diag["margin_drop_median"]),
                            "base_classwise_covariance_distortion_mean": float(
                                base_diag["classwise_covariance_distortion_mean"]
                            ),
                            "after_classwise_covariance_distortion_mean": float(
                                refined_diag["classwise_covariance_distortion_mean"]
                            ),
                            "base_cond_A": float(base_diag["cond_A_mean"]),
                            "after_cond_A": float(refined_diag["cond_A_mean"]),
                            "base_bridge_cov_to_orig_distance_mean": float(
                                base_diag["bridge_cov_to_orig_distance_mean"]
                            ),
                            "after_bridge_cov_to_orig_distance_mean": float(
                                refined_diag["bridge_cov_to_orig_distance_mean"]
                            ),
                        }
                    )

                    run_dir = os.path.join(args.out_root, str(dataset), f"seed{seed}", str(point.config_id))
                    _ensure_dir(run_dir)
                    with open(os.path.join(run_dir, "axis_refine_eval.json"), "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "dataset": str(dataset),
                                "seed": int(seed),
                                "base_pia_config": str(point.base_pia_config),
                                "config_id": str(point.config_id),
                                "gamma_main": float(point.gamma_main),
                                "second_axis_scale": float(point.second_axis_scale),
                                "pullback_alpha": float(point.pullback_alpha),
                                "base_metrics": base_metrics,
                                "refined_metrics": refined_metrics,
                                "gamma_meta": gamma_meta,
                                "base_diag": base_diag,
                                "refined_diag": refined_diag,
                                "base_bridge_global_fidelity": dict(base_bridge.global_fidelity),
                                "refined_bridge_global_fidelity": dict(refined_bridge.global_fidelity),
                                "base_bridge_classwise_fidelity": dict(base_bridge.classwise_fidelity),
                                "refined_bridge_classwise_fidelity": dict(refined_bridge.classwise_fidelity),
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )

    per_seed_df = pd.DataFrame(per_seed_rows)
    mechanism_df = pd.DataFrame(mechanism_rows)
    config_df = _config_table(config_rows)
    summary_df = _dataset_summary(per_seed_df)
    mech_summary_df = _mechanism_summary(mechanism_df)

    config_df.to_csv(os.path.join(args.out_root, "axis_refine_config_table.csv"), index=False)
    per_seed_df.to_csv(os.path.join(args.out_root, "axis_refine_per_seed.csv"), index=False)
    summary_df.to_csv(os.path.join(args.out_root, "axis_refine_dataset_summary.csv"), index=False)
    mech_summary_df.to_csv(os.path.join(args.out_root, "axis_refine_mechanism_summary.csv"), index=False)
    _write_conclusion(os.path.join(args.out_root, "axis_refine_conclusion.md"), summary_df, mech_summary_df)


if __name__ == "__main__":
    main()
