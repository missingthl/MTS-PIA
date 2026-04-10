#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, replace
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
    apply_bridge,
    evaluate_bridge,
)
from route_b_unified.types import RepresentationState  # noqa: E402
from scripts.protocol_split_utils import resolve_protocol_split  # noqa: E402
from scripts.run_route_b_main_matrix import _build_eval_cfg  # noqa: E402
from scripts.run_route_b_pia_core_axis_refine import (  # noqa: E402
    FORMAL_CONFIG_TABLE_CSV,
    BasePIAConfigPoint,
    _build_target_state,
    _evaluate_variant,
    _load_base_config_points,
)
from scripts.run_route_b_pia_core_config_sweep import (  # noqa: E402
    REFERENCE_MATRIX_CSV,
    _build_rep_state_from_trials,
    _load_reference_baselines,
    _reference_metric,
)


@dataclass(frozen=True)
class AxisPullbackRefinePoint:
    config_id: str
    base_pia_config: str
    gamma_main: float
    second_axis_scale: float
    pullback_alpha: float
    axis_count: int = 2


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


def _build_refine_points(
    *,
    base_cfg: BasePIAConfigPoint,
    second_axis_scales: Sequence[float],
    pullback_values: Sequence[float],
) -> List[AxisPullbackRefinePoint]:
    out: List[AxisPullbackRefinePoint] = []
    for scale in second_axis_scales:
        scale_f = float(scale)
        for pb in pullback_values:
            pb_f = float(pb)
            cfg_id = (
                f"{base_cfg.config_id}__axis2s{int(round(scale_f * 100)):03d}"
                f"_pb{int(round(pb_f * 100)):03d}"
            )
            out.append(
                AxisPullbackRefinePoint(
                    config_id=cfg_id,
                    base_pia_config=str(base_cfg.config_id),
                    gamma_main=float(base_cfg.gamma_main),
                    second_axis_scale=scale_f,
                    pullback_alpha=pb_f,
                    axis_count=2,
                )
            )
    return out


def _config_table(rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    keep_cols = [
        "dataset",
        "base_pia_config",
        "gamma_main",
        "second_axis_scale",
        "pullback_alpha",
        "axis_count",
        "config_id",
    ]
    return df[keep_cols].drop_duplicates().sort_values(["dataset", "config_id"]).reset_index(drop=True)


def _dataset_summary(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, config_id), df_sub in per_seed_df.groupby(["dataset", "config_id"], sort=True):
        rows.append(
            {
                "dataset": str(dataset),
                "base_pia_config": str(df_sub["base_pia_config"].iloc[0]),
                "config_id": str(config_id),
                "raw_baseline_macro_f1": _format_mean_std(df_sub["raw_test_macro_f1"].tolist()),
                "base_pia_macro_f1": _format_mean_std(df_sub["base_pia_test_macro_f1"].tolist()),
                "refined_macro_f1": _format_mean_std(df_sub["refined_test_macro_f1"].tolist()),
                "delta_vs_raw_mean": float(df_sub["delta_vs_raw"].mean()),
                "delta_vs_base_pia_mean": float(df_sub["delta_vs_base_pia"].mean()),
                "per_seed_sign_vs_base_pia": ",".join(_sign_text(v) for v in df_sub["delta_vs_base_pia"].tolist()),
                "per_seed_sign_vs_raw": ",".join(_sign_text(v) for v in df_sub["delta_vs_raw"].tolist()),
            }
        )
    return pd.DataFrame(rows)


def _mechanism_summary(mech_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, config_id), df_sub in mech_df.groupby(["dataset", "config_id"], sort=True):
        rows.append(
            {
                "dataset": str(dataset),
                "base_pia_config": str(df_sub["base_pia_config"].iloc[0]),
                "config_id": str(config_id),
                "base_flip_rate_mean": float(df_sub["base_flip_rate"].mean()),
                "after_refine_flip_rate_mean": float(df_sub["after_flip_rate"].mean()),
                "base_margin_drop_median_mean": float(df_sub["base_margin_drop_median"].mean()),
                "after_refine_margin_drop_median_mean": float(df_sub["after_margin_drop_median"].mean()),
                "base_classwise_covariance_distortion_mean": float(
                    df_sub["base_classwise_covariance_distortion_mean"].mean()
                ),
                "after_refine_classwise_covariance_distortion_mean": float(
                    df_sub["after_classwise_covariance_distortion_mean"].mean()
                ),
                "base_cond_A_mean": float(df_sub["base_cond_A"].mean()),
                "after_refine_cond_A_mean": float(df_sub["after_cond_A"].mean()),
                "base_bridge_cov_to_orig_distance_mean": float(df_sub["base_bridge_cov_to_orig_distance_mean"].mean()),
                "after_refine_bridge_cov_to_orig_distance_mean": float(
                    df_sub["after_bridge_cov_to_orig_distance_mean"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_conclusion(path: str, summary_df: pd.DataFrame, mech_df: pd.DataFrame) -> None:
    lines = [
        "# Axis Pullback Refine Conclusion",
        "",
        "更新时间：2026-03-28",
        "",
    ]

    nat = summary_df.loc[summary_df["dataset"] == "natops"].copy()
    scp = summary_df.loc[summary_df["dataset"] == "selfregulationscp1"].copy()
    scp_mech = mech_df.loc[mech_df["dataset"] == "selfregulationscp1"].copy()

    score_cols = ["delta_vs_raw_mean", "delta_vs_base_pia_mean"]
    nat_best = nat.sort_values(score_cols, ascending=[False, False]).iloc[0] if not nat.empty else None
    scp_best = scp.sort_values("delta_vs_base_pia_mean", ascending=False).iloc[0] if not scp.empty else None

    # Reference prior axis-only point from previous round is axis2s075_pb100.
    nat_prev = nat.loc[nat["config_id"].astype(str).str.endswith("__axis2s075_pb100")]
    scp_prev = scp.loc[scp["config_id"].astype(str).str.endswith("__axis2s075_pb100")]
    nat_prev_row = nat_prev.iloc[0] if not nat_prev.empty else None
    scp_prev_row = scp_prev.iloc[0] if not scp_prev.empty else None

    if nat_best is not None and scp_best is not None:
        # Simple heuristic: whichever metric changes more decisively between nearby configs.
        nat_span = float(nat["delta_vs_raw_mean"].max() - nat["delta_vs_raw_mean"].min()) if not nat.empty else 0.0
        scp_span = (
            float(scp["delta_vs_base_pia_mean"].max() - scp["delta_vs_base_pia_mean"].min()) if not scp.empty else 0.0
        )
        predominant = "second_axis_scale" if nat_span >= scp_span * 0.8 else "pullback_alpha"
    else:
        predominant = "insufficient_result"

    nat_preserved = nat_best is not None and float(nat_best["delta_vs_raw_mean"]) >= 0.002
    scp_improved = scp_best is not None and float(scp_best["delta_vs_base_pia_mean"]) >= 0.002

    previous_balance = False
    if nat_prev_row is not None and scp_prev_row is not None and nat_best is not None and scp_best is not None:
        previous_balance = (
            float(nat_best["delta_vs_raw_mean"]) >= float(nat_prev_row["delta_vs_raw_mean"]) - 0.002
            and float(scp_best["delta_vs_base_pia_mean"]) > float(scp_prev_row["delta_vs_base_pia_mean"]) + 1e-9
        )

    lines.append(f"1. second_axis_scale × pullback_alpha 联动是否进一步有效：`{'yes' if previous_balance else 'not_yet'}`")
    if nat_best is not None:
        lines.append(
            f"   NATOPS 最优点：`{nat_best['config_id']}`，`delta_vs_raw_mean={float(nat_best['delta_vs_raw_mean']):+.4f}`，"
            f"`delta_vs_base_pia_mean={float(nat_best['delta_vs_base_pia_mean']):+.4f}`。"
        )
    if scp_best is not None:
        lines.append(
            f"   SCP1 最优点：`{scp_best['config_id']}`，`delta_vs_base_pia_mean={float(scp_best['delta_vs_base_pia_mean']):+.4f}`。"
        )

    lines.append(f"2. NATOPS 正收益是否仍保住：`{'yes' if nat_preserved else 'no'}`")
    lines.append(f"3. SCP1 是否进一步改善：`{'yes' if scp_improved else 'not_yet'}`")
    lines.append(f"4. 是否比上一轮更接近共同候选点：`{'yes' if previous_balance else 'not_yet'}`")
    lines.append(f"5. 当前残余矛盾更像主要来自：`{predominant}`")

    if not previous_balance:
        lines.append(
            "   当前若仍未形成统一强点，更像说明：`低维 axis-2 × pullback 联动已逼近上限，残余问题未必只靠这两个旋钮就能解决。`"
        )

    if not scp_mech.empty:
        mech_best = scp_mech.sort_values(
            [
                "after_refine_classwise_covariance_distortion_mean",
                "after_refine_flip_rate_mean",
            ],
            ascending=[True, True],
        ).iloc[0]
        lines.append(
            f"   机制上最健康的 SCP1 点：`{mech_best['config_id']}`，"
            f"`flip {float(mech_best['base_flip_rate_mean']):.4f}->{float(mech_best['after_refine_flip_rate_mean']):.4f}`，"
            f"`classwise_cov_dist {float(mech_best['base_classwise_covariance_distortion_mean']):.4f}->{float(mech_best['after_refine_classwise_covariance_distortion_mean']):.4f}`。"
        )

    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Narrow closure round around g100_k2_pb100 with local axis-2 × pullback refinement."
    )
    p.add_argument("--datasets", type=str, default="natops,selfregulationscp1")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/route_b_pia_core_axis_pullback_refine_20260328")
    p.add_argument("--base-config-id", type=str, default="g100_k2_pb100")
    p.add_argument("--formal-config-table-csv", type=str, default=FORMAL_CONFIG_TABLE_CSV)
    p.add_argument("--reference-matrix-csv", type=str, default=REFERENCE_MATRIX_CSV)
    p.add_argument("--second-axis-scales", type=str, default="0.80,0.75,0.70")
    p.add_argument("--pullback-alphas", type=str, default="1.00,0.95,0.90")
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
    pullback_alphas = _parse_float_list(args.pullback_alphas)
    _ensure_dir(args.out_root)

    base_cfg_map = _load_base_config_points(str(args.formal_config_table_csv))
    if str(args.base_config_id) not in base_cfg_map:
        raise KeyError(f"missing base config in formal config table: {args.base_config_id}")
    base_cfg = base_cfg_map[str(args.base_config_id)]
    if int(base_cfg.axis_count) != 2:
        raise ValueError("this runner requires a 2-axis base config")

    refine_points = _build_refine_points(
        base_cfg=base_cfg,
        second_axis_scales=second_axis_scales,
        pullback_values=pullback_alphas,
    )
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

    config_rows: List[Dict[str, object]] = []
    per_seed_rows: List[Dict[str, object]] = []
    mechanism_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        all_trials = load_trials_for_dataset(
            dataset=str(dataset),
            natops_root=str(args.natops_root),
            selfregulationscp1_root=str(args.selfregulationscp1_root),
        )
        for seed in seeds:
            print(f"[axis-pullback-refine][{dataset}][seed={seed}] split_start", flush=True)
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

            for point in refine_points:
                config_rows.append(
                    {
                        "dataset": str(dataset),
                        "base_pia_config": str(point.base_pia_config),
                        "gamma_main": float(point.gamma_main),
                        "second_axis_scale": float(point.second_axis_scale),
                        "pullback_alpha": float(point.pullback_alpha),
                        "axis_count": int(point.axis_count),
                        "config_id": str(point.config_id),
                    }
                )
                refine_gamma_vector, gamma_meta = pia_core.build_two_axis_gamma_vector(
                    axis_ids=base_axis_ids,
                    gamma_main=float(point.gamma_main),
                    second_axis_scale=float(point.second_axis_scale),
                )
                refine_metrics, refine_diag, _refine_bridge = _evaluate_variant(
                    rep_state=rep_full,
                    pia_core=pia_core,
                    axis_ids=base_axis_ids,
                    gamma_vector=refine_gamma_vector,
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
                        "refined_test_macro_f1": float(refine_metrics["macro_f1"]),
                        "delta_vs_raw": float(refine_metrics["macro_f1"]) - float(raw_test_f1_ref),
                        "delta_vs_base_pia": float(refine_metrics["macro_f1"]) - float(base_metrics["macro_f1"]),
                    }
                )
                mechanism_rows.append(
                    {
                        "dataset": str(dataset),
                        "base_pia_config": str(point.base_pia_config),
                        "config_id": str(point.config_id),
                        "seed": int(seed),
                        "gamma_main": float(point.gamma_main),
                        "second_axis_scale": float(point.second_axis_scale),
                        "pullback_alpha": float(point.pullback_alpha),
                        "base_flip_rate": float(base_diag["flip_rate"]),
                        "after_flip_rate": float(refine_diag["flip_rate"]),
                        "base_margin_drop_median": float(base_diag["margin_drop_median"]),
                        "after_margin_drop_median": float(refine_diag["margin_drop_median"]),
                        "base_classwise_covariance_distortion_mean": float(
                            base_diag["classwise_covariance_distortion_mean"]
                        ),
                        "after_classwise_covariance_distortion_mean": float(
                            refine_diag["classwise_covariance_distortion_mean"]
                        ),
                        "base_cond_A": float(base_diag["cond_A_mean"]),
                        "after_cond_A": float(refine_diag["cond_A_mean"]),
                        "base_bridge_cov_to_orig_distance_mean": float(base_diag["bridge_cov_to_orig_distance_mean"]),
                        "after_bridge_cov_to_orig_distance_mean": float(
                            refine_diag["bridge_cov_to_orig_distance_mean"]
                        ),
                        "base_selected_axis_ids": str(base_diag["selected_axis_ids"]),
                        "after_selected_axis_ids": str(refine_diag["selected_axis_ids"]),
                        "after_gamma_vector": str(refine_diag["gamma_vector"]),
                        "after_axis_strength_mode": str(gamma_meta["axis_strength_mode"]),
                    }
                )

    config_df = _config_table(config_rows)
    per_seed_df = pd.DataFrame(per_seed_rows).sort_values(["dataset", "config_id", "seed"]).reset_index(drop=True)
    summary_df = _dataset_summary(per_seed_df)
    mech_df = _mechanism_summary(pd.DataFrame(mechanism_rows))

    config_df.to_csv(os.path.join(args.out_root, "axis_pullback_refine_config_table.csv"), index=False)
    per_seed_df.to_csv(os.path.join(args.out_root, "axis_pullback_refine_per_seed.csv"), index=False)
    summary_df.to_csv(os.path.join(args.out_root, "axis_pullback_refine_dataset_summary.csv"), index=False)
    mech_df.to_csv(os.path.join(args.out_root, "axis_pullback_refine_mechanism_summary.csv"), index=False)
    _write_conclusion(os.path.join(args.out_root, "axis_pullback_refine_conclusion.md"), summary_df, mech_df)

    print(f"[done] wrote outputs to {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
