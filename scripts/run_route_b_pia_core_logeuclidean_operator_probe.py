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
from route_b_unified.types import BridgeResult, RepresentationState  # noqa: E402
from scripts.run_route_b_main_matrix import _build_eval_cfg  # noqa: E402
from scripts.run_route_b_pia_core_config_sweep import (  # noqa: E402
    _build_rep_state_from_trials,
    _mech_global_metrics,
    _raw_reference_bridge_result,
    _raw_reference_target_state,
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


def _build_target_state(
    *,
    rep_state: RepresentationState,
    pia_core: PIACore,
    gamma_vector: Sequence[float],
    axis_ids: Sequence[int],
    pullback_alpha: float,
    operator_mode: str,
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
    if str(operator_mode) == "logeuclidean":
        op_result = pia_core.apply_logeuclidean_affine(
            rep_state.X_train,
            gamma_vector=gamma_vec,
            axis_ids=axis_ids_list,
            pullback_alpha=float(pullback_alpha),
        )
    elif str(operator_mode) == "vector":
        op_result = pia_core.apply_affine(
            rep_state.X_train,
            gamma_vector=gamma_vec,
            axis_ids=axis_ids_list,
            pullback_alpha=float(pullback_alpha),
        )
    else:
        raise ValueError(f"unknown operator_mode: {operator_mode}")

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
                "dir_profile_summary": str(operator_mode),
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
            stop_reason=f"logeuclidean_probe_{operator_mode}",
        ),
        direction_state={int(a): f"{operator_mode}_active" for a in axis_ids_list},
        direction_budget_score={int(a): 1.0 for a in axis_ids_list},
    )
    diag = {
        "margin_drop_median": float(mech["margin_drop_median"]),
        "flip_rate": float(mech["flip_rate"]),
        "intrusion_rate": float(mech["intrusion_rate"]),
        "selected_axis_ids": json.dumps(axis_ids_list, ensure_ascii=False),
        "gamma_vector": json.dumps(gamma_vec, ensure_ascii=False),
        "operator_mode": str(operator_mode),
        "operator_type": str(op_result.meta.get("operator_type", "unknown")),
        "operator_semantics": str(op_result.meta.get("operator_semantics", "unknown")),
        "geometry_mode": str(op_result.meta.get("geometry_mode", "vector_centered_log_covariance")),
        "pullback_alpha": float(op_result.meta.get("pullback_alpha", pullback_alpha)),
        "aligned_energy_ratio_mean": float(op_result.meta.get("aligned_energy_ratio_mean", 0.0)),
        "residual_energy_mean": float(op_result.meta.get("residual_energy_mean", 0.0)),
    }
    return target_state, diag


def _evaluate_operator_variant(
    *,
    rep_state: RepresentationState,
    pia_core: PIACore,
    axis_ids: Sequence[int],
    gamma_vector: Sequence[float],
    pullback_alpha: float,
    operator_mode: str,
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
        operator_mode=operator_mode,
        seed=seed,
        linear_c=linear_c,
        linear_class_weight=linear_class_weight,
        linear_max_iter=linear_max_iter,
        mech_knn_k=mech_knn_k,
        mech_max_aug=mech_max_aug,
        mech_max_real_ref=mech_max_real_ref,
        mech_max_real_query=mech_max_real_query,
    )
    bridge_result = apply_bridge(rep_state, target_state, bridge_cfg, variant=f"scheme_a_{operator_mode}")
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


def _config_table(rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    keep_cols = [
        "dataset",
        "reference_config_label",
        "gamma_main",
        "second_axis_scale",
        "pullback_alpha",
        "axis_count",
        "operator_mode",
        "config_id",
    ]
    return df[keep_cols].drop_duplicates().sort_values(["dataset", "operator_mode"]).reset_index(drop=True)


def _dataset_summary(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, df_sub in per_seed_df.groupby("dataset", sort=True):
        rows.append(
            {
                "dataset": str(dataset),
                "raw_baseline_macro_f1": _format_mean_std(df_sub["raw_test_macro_f1"].tolist()),
                "vector_operator_macro_f1": _format_mean_std(df_sub["vector_test_macro_f1"].tolist()),
                "logeuclidean_operator_macro_f1": _format_mean_std(df_sub["logeuclidean_test_macro_f1"].tolist()),
                "vector_delta_vs_raw_mean": float(df_sub["vector_delta_vs_raw"].mean()),
                "logeuclidean_delta_vs_raw_mean": float(df_sub["logeuclidean_delta_vs_raw"].mean()),
                "logeuclidean_delta_vs_vector_mean": float(df_sub["logeuclidean_delta_vs_vector"].mean()),
                "per_seed_sign_vs_vector": ",".join(_sign_text(v) for v in df_sub["logeuclidean_delta_vs_vector"].tolist()),
            }
        )
    return pd.DataFrame(rows)


def _mechanism_summary(mech_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, df_sub in mech_df.groupby("dataset", sort=True):
        rows.append(
            {
                "dataset": str(dataset),
                "vector_flip_rate_mean": float(df_sub["vector_flip_rate"].mean()),
                "logeuclidean_flip_rate_mean": float(df_sub["logeuclidean_flip_rate"].mean()),
                "vector_margin_drop_median_mean": float(df_sub["vector_margin_drop_median"].mean()),
                "logeuclidean_margin_drop_median_mean": float(df_sub["logeuclidean_margin_drop_median"].mean()),
                "vector_classwise_covariance_distortion_mean": float(
                    df_sub["vector_classwise_covariance_distortion_mean"].mean()
                ),
                "logeuclidean_classwise_covariance_distortion_mean": float(
                    df_sub["logeuclidean_classwise_covariance_distortion_mean"].mean()
                ),
                "vector_cond_A_mean": float(df_sub["vector_cond_A"].mean()),
                "logeuclidean_cond_A_mean": float(df_sub["logeuclidean_cond_A"].mean()),
                "vector_geometry_mode": str(df_sub["vector_geometry_mode"].iloc[0]),
                "logeuclidean_geometry_mode": str(df_sub["logeuclidean_geometry_mode"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _write_conclusion(path: str, summary_df: pd.DataFrame, mech_df: pd.DataFrame) -> None:
    lines = [
        "# Scheme A Log-Euclidean Operator Probe Conclusion",
        "",
        "更新时间：2026-03-29",
        "",
        "当前 probe 保留 TELM2 多轮闭式更新后再拉伸，不做交替迭代。",
        "",
    ]
    for _, row in summary_df.iterrows():
        dataset = str(row["dataset"])
        better = float(row["logeuclidean_delta_vs_vector_mean"]) > 0.0
        lines.append(f"## {dataset}")
        lines.append(
            f"- log-Euclidean 自洽算子是否优于当前向量版：`{'yes' if better else 'not_yet'}`"
        )
        lines.append(
            f"- vector delta vs raw: `{float(row['vector_delta_vs_raw_mean']):+.4f}`"
        )
        lines.append(
            f"- log-Euclidean delta vs raw: `{float(row['logeuclidean_delta_vs_raw_mean']):+.4f}`"
        )
        lines.append(
            f"- log-Euclidean delta vs vector: `{float(row['logeuclidean_delta_vs_vector_mean']):+.4f}`"
        )
        mech_row = mech_df.loc[mech_df["dataset"] == dataset]
        if not mech_row.empty:
            mech = mech_row.iloc[0]
            lines.append(
                f"- 机制对比：`flip {float(mech['vector_flip_rate_mean']):.4f}->{float(mech['logeuclidean_flip_rate_mean']):.4f}`，"
                f"`classwise_cov_dist {float(mech['vector_classwise_covariance_distortion_mean']):.4f}->{float(mech['logeuclidean_classwise_covariance_distortion_mean']):.4f}`，"
                f"`cond_A {float(mech['vector_cond_A_mean']):.4f}->{float(mech['logeuclidean_cond_A_mean']):.4f}`"
            )
        lines.append("")

    lines.append("结论标签：`scheme_a_probe_complete`")
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Probe a log-Euclidean self-consistent PIA operator against the current vector-domain operator."
    )
    p.add_argument("--datasets", type=str, default="natops,selfregulationscp1")
    p.add_argument("--seeds", type=str, default="1")
    p.add_argument("--out-root", type=str, default="out/route_b_pia_core_logeuclidean_operator_probe_20260329")
    p.add_argument("--reference-config-label", type=str, default="g100_k2_pb100__axis2s080_pb090")
    p.add_argument("--gamma-main", type=float, default=0.10)
    p.add_argument("--second-axis-scale", type=float, default=0.80)
    p.add_argument("--pullback-alpha", type=float, default=0.90)
    p.add_argument("--axis-count", type=int, default=2)
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
    _ensure_dir(args.out_root)

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

    config_rows: List[Dict[str, object]] = []
    per_seed_rows: List[Dict[str, object]] = []
    mechanism_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        for seed in seeds:
            all_trials = load_trials_for_dataset(
                dataset=dataset,
                natops_root=str(args.natops_root),
                selfregulationscp1_root=str(args.selfregulationscp1_root),
            )
            train_trials = [t for t in all_trials if str(t.get("split", "")).lower() == "train"]
            test_trials = [t for t in all_trials if str(t.get("split", "")).lower() == "test"]
            if not train_trials or not test_trials:
                raise RuntimeError(f"expected official train/test split trials for dataset={dataset}")
            rep_state = _build_rep_state_from_trials(
                dataset=dataset,
                seed=int(seed),
                train_trials=train_trials,
                val_trials=[],
                test_trials=test_trials,
                spd_eps=float(args.spd_eps),
                protocol_type="fixed_split_official",
                protocol_note="dataset-provided official TRAIN/TEST split",
            )

            pia_core = PIACore(replace(pia_cfg, seed=int(seed))).fit(rep_state.X_train)
            ranked_axis_ids, rank_meta = pia_core.rank_axes_by_energy(rep_state.X_train)
            axis_ids = ranked_axis_ids[: int(args.axis_count)]
            gamma_vec, gamma_meta = pia_core.build_two_axis_gamma_vector(
                axis_ids=axis_ids,
                gamma_main=float(args.gamma_main),
                second_axis_scale=float(args.second_axis_scale),
            )

            raw_bridge = _raw_reference_bridge_result(rep_state)
            raw_target = _raw_reference_target_state(int(rep_state.X_train.shape[1]))
            raw_posterior = evaluate_bridge(
                raw_bridge,
                eval_cfg,
                split_name="test",
                target_state=raw_target,
                round_gain_proxy=0.0,
            )
            raw_test_macro_f1 = float(raw_posterior.macro_f1)

            vector_metrics, vector_diag, _ = _evaluate_operator_variant(
                rep_state=rep_state,
                pia_core=pia_core,
                axis_ids=axis_ids,
                gamma_vector=gamma_vec,
                pullback_alpha=float(args.pullback_alpha),
                operator_mode="vector",
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
            log_metrics, log_diag, _ = _evaluate_operator_variant(
                rep_state=rep_state,
                pia_core=pia_core,
                axis_ids=axis_ids,
                gamma_vector=gamma_vec,
                pullback_alpha=float(args.pullback_alpha),
                operator_mode="logeuclidean",
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

            config_id = f"{args.reference_config_label}__scheme_a_probe"
            config_rows.append(
                {
                    "dataset": dataset,
                    "reference_config_label": str(args.reference_config_label),
                    "gamma_main": float(args.gamma_main),
                    "second_axis_scale": float(args.second_axis_scale),
                    "pullback_alpha": float(args.pullback_alpha),
                    "axis_count": int(args.axis_count),
                    "operator_mode": "vector_vs_logeuclidean",
                    "config_id": config_id,
                }
            )
            per_seed_rows.append(
                {
                    "dataset": dataset,
                    "config_id": config_id,
                    "seed": int(seed),
                    "raw_test_macro_f1": float(raw_test_macro_f1),
                    "vector_test_macro_f1": float(vector_metrics["macro_f1"]),
                    "logeuclidean_test_macro_f1": float(log_metrics["macro_f1"]),
                    "vector_delta_vs_raw": float(vector_metrics["macro_f1"] - raw_test_macro_f1),
                    "logeuclidean_delta_vs_raw": float(log_metrics["macro_f1"] - raw_test_macro_f1),
                    "logeuclidean_delta_vs_vector": float(log_metrics["macro_f1"] - vector_metrics["macro_f1"]),
                    "selected_axis_ids": json.dumps(axis_ids, ensure_ascii=False),
                    "gamma_vector": json.dumps(gamma_vec, ensure_ascii=False),
                    "ranked_axis_ids": json.dumps(ranked_axis_ids, ensure_ascii=False),
                }
            )
            mechanism_rows.append(
                {
                    "dataset": dataset,
                    "config_id": config_id,
                    "seed": int(seed),
                    "vector_flip_rate": float(vector_diag["flip_rate"]),
                    "logeuclidean_flip_rate": float(log_diag["flip_rate"]),
                    "vector_margin_drop_median": float(vector_diag["margin_drop_median"]),
                    "logeuclidean_margin_drop_median": float(log_diag["margin_drop_median"]),
                    "vector_classwise_covariance_distortion_mean": float(
                        vector_diag["classwise_covariance_distortion_mean"]
                    ),
                    "logeuclidean_classwise_covariance_distortion_mean": float(
                        log_diag["classwise_covariance_distortion_mean"]
                    ),
                    "vector_cond_A": float(vector_diag["cond_A_mean"]),
                    "logeuclidean_cond_A": float(log_diag["cond_A_mean"]),
                    "vector_geometry_mode": str(vector_diag["geometry_mode"]),
                    "logeuclidean_geometry_mode": str(log_diag["geometry_mode"]),
                    "rank_meta": json.dumps(rank_meta, ensure_ascii=False),
                    "gamma_meta": json.dumps(gamma_meta, ensure_ascii=False),
                }
            )

    config_df = _config_table(config_rows)
    per_seed_df = pd.DataFrame(per_seed_rows).sort_values(["dataset", "seed"]).reset_index(drop=True)
    mechanism_df = pd.DataFrame(mechanism_rows).sort_values(["dataset", "seed"]).reset_index(drop=True)
    summary_df = _dataset_summary(per_seed_df)
    mech_summary_df = _mechanism_summary(mechanism_df)

    config_df.to_csv(os.path.join(args.out_root, "logeuclidean_operator_probe_config_table.csv"), index=False)
    per_seed_df.to_csv(os.path.join(args.out_root, "logeuclidean_operator_probe_per_seed.csv"), index=False)
    summary_df.to_csv(os.path.join(args.out_root, "logeuclidean_operator_probe_dataset_summary.csv"), index=False)
    mech_summary_df.to_csv(os.path.join(args.out_root, "logeuclidean_operator_probe_mechanism_summary.csv"), index=False)
    _write_conclusion(
        os.path.join(args.out_root, "logeuclidean_operator_probe_conclusion.md"),
        summary_df,
        mech_summary_df,
    )


if __name__ == "__main__":
    main()
