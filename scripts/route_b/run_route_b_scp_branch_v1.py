#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from route_b_unified.scp_local_shaping import (  # noqa: E402
    SCPLocalShapingConfig,
    apply_scp_local_shaping,
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


def _format_mean_std(values: Sequence[float]) -> str:
    arr = np.asarray(list(values), dtype=np.float64)
    return f"{float(np.mean(arr)):.4f} +/- {float(np.std(arr)):.4f}" if arr.size else "0.0000 +/- 0.0000"


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


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


def main() -> None:
    p = argparse.ArgumentParser(description="SCP-Branch v1: prototype-memory-centered local separation shaping probe.")
    p.add_argument("--datasets", type=str, default="selfregulationscp1")
    p.add_argument("--seeds", type=str, default="1")
    p.add_argument("--out-root", type=str, default="out/_active/verify_route_b_scp_branch_v1_20260330")
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
    p.add_argument("--anchors-per-prototype", type=int, default=16)
    p.add_argument("--anchor-selection-mode", type=str, default="nearest")
    p.add_argument("--same-dist-quantile", type=float, default=50.0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--epsilon-scale", type=float, default=0.10)
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
    anchor_rows_all: List[Dict[str, object]] = []
    shaping_rows_all: List[Dict[str, object]] = []
    structure_rows_all: List[Dict[str, object]] = []

    for dataset in datasets:
        for seed in seeds:
            state = _build_dense_state(args, dataset, seed)
            seed_dir = os.path.join(args.out_root, f"{dataset}_seed{seed}")
            _ensure_dir(seed_dir)
            _write_json(
                os.path.join(seed_dir, "dense_backbone_meta.json"),
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "window_len": int(state.window_len),
                    "hop_len": int(state.hop_len),
                    "z_dim": int(state.z_dim),
                    "num_classes": int(state.num_classes),
                },
            )

            # Build structure diagnostics before touching the terminal evaluator. On the current
            # pia env, running MiniROCKET before the sklearn/OpenBLAS-heavy prototype stage can
            # trigger an OpenBLAS memory-region crash on dense SCP1.
            before_memory = build_scp_prototype_memory(
                train_tids=state.train.tids.tolist(),
                train_labels=state.train.y.tolist(),
                train_z_seq_list=state.train.z_seq_list,
                reference_stats=build_window_feedback_reference_stats(
                    train_labels=state.train.y.tolist(),
                    train_z_seq_list=state.train.z_seq_list,
                ),
                cfg=SCPPrototypeMemoryConfig(
                    prototype_count=int(args.prototype_count),
                    cluster_mode="kmeans_centroid",
                    seed=int(seed),
                ),
            )

            shaping_result = apply_scp_local_shaping(
                train_tids=state.train.tids.tolist(),
                train_labels=state.train.y.tolist(),
                train_z_seq_list=state.train.z_seq_list,
                cfg=SCPLocalShapingConfig(
                    prototype_count=int(args.prototype_count),
                    anchors_per_prototype=int(args.anchors_per_prototype),
                    anchor_selection_mode=str(args.anchor_selection_mode),
                    same_dist_quantile=float(args.same_dist_quantile),
                    beta=float(args.beta),
                    epsilon_scale=float(args.epsilon_scale),
                    seed=int(seed),
                ),
            )

            after_memory = build_scp_prototype_memory(
                train_tids=state.train.tids.tolist(),
                train_labels=state.train.y.tolist(),
                train_z_seq_list=shaping_result.shaped_train_z_seq_list,
                reference_stats=build_window_feedback_reference_stats(
                    train_labels=state.train.y.tolist(),
                    train_z_seq_list=shaping_result.shaped_train_z_seq_list,
                ),
                cfg=SCPPrototypeMemoryConfig(
                    prototype_count=int(args.prototype_count),
                    cluster_mode="kmeans_centroid",
                    seed=int(seed),
                ),
            )

            eval_cfg = TrajectoryMiniRocketEvalConfig(
                n_kernels=int(args.minirocket_n_kernels),
                n_jobs=int(args.minirocket_n_jobs),
                padding_mode="edge",
                target_len_mode="train_max_len",
            )

            # Same backbone / same protocol baseline: no shaping write-back.
            baseline_result = evaluate_dynamic_minirocket_classifier(
                state,
                seed=int(seed),
                eval_cfg=eval_cfg,
            )

            shaped_train = replace(
                state.train,
                z_seq_list=[np.asarray(v, dtype=np.float32) for v in shaping_result.shaped_train_z_seq_list],
            )
            shaped_state = replace(state, train=shaped_train)
            shaped_result = evaluate_dynamic_minirocket_classifier(
                shaped_state,
                seed=int(seed),
                eval_cfg=eval_cfg,
            )

            before_diag = next(r for r in before_memory.structure_rows if str(r["mode"]) == "prototype_memory")
            after_diag = next(r for r in after_memory.structure_rows if str(r["mode"]) == "prototype_memory")
            ref = backbone_ref_map.get((str(dataset), int(seed)), {})

            _write_json(
                os.path.join(seed_dir, "scp_branch_v1_result.json"),
                {
                    "same_backbone_no_shaping": {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "test_macro_f1": float(baseline_result.test_metrics["macro_f1"]),
                        "val_macro_f1": float(baseline_result.val_metrics["macro_f1"]),
                        "meta": dict(baseline_result.meta),
                    },
                    "local_shaping": {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "test_macro_f1": float(shaped_result.test_metrics["macro_f1"]),
                        "val_macro_f1": float(shaped_result.val_metrics["macro_f1"]),
                        "meta": dict(shaped_result.meta),
                        "summary": dict(shaping_result.summary),
                    },
                    "structure_before": dict(before_diag),
                    "structure_after": dict(after_diag),
                },
            )

            per_seed_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "same_backbone_no_shaping_test_macro_f1": float(baseline_result.test_metrics["macro_f1"]),
                    "same_backbone_no_shaping_val_macro_f1": float(baseline_result.val_metrics["macro_f1"]),
                    "local_shaping_test_macro_f1": float(shaped_result.test_metrics["macro_f1"]),
                    "local_shaping_val_macro_f1": float(shaped_result.val_metrics["macro_f1"]),
                    "delta_test_macro_f1": float(shaped_result.test_metrics["macro_f1"] - baseline_result.test_metrics["macro_f1"]),
                    "static_linear_test_macro_f1": float(ref.get("static_linear_test_macro_f1", np.nan)),
                    "dense_dynamic_gru_test_macro_f1": float(ref.get("dense_dynamic_gru_test_macro_f1", np.nan)),
                    "raw_minirocket_test_macro_f1": float(ref.get("raw_minirocket_test_macro_f1", np.nan)),
                    "before_within_compactness": float(before_diag["within_prototype_compactness"]),
                    "after_within_compactness": float(after_diag["within_prototype_compactness"]),
                    "before_between_separation": float(before_diag["between_prototype_separation"]),
                    "after_between_separation": float(after_diag["between_prototype_separation"]),
                    "before_nearest_margin": float(before_diag["nearest_prototype_margin"]),
                    "after_nearest_margin": float(after_diag["nearest_prototype_margin"]),
                    "before_temporal_stability": float(before_diag["temporal_assignment_stability"]),
                    "after_temporal_stability": float(after_diag["temporal_assignment_stability"]),
                    "delta_within_compactness": float(after_diag["within_prototype_compactness"] - before_diag["within_prototype_compactness"]),
                    "delta_between_separation": float(after_diag["between_prototype_separation"] - before_diag["between_prototype_separation"]),
                    "delta_nearest_margin": float(after_diag["nearest_prototype_margin"] - before_diag["nearest_prototype_margin"]),
                    "delta_temporal_stability": float(after_diag["temporal_assignment_stability"] - before_diag["temporal_assignment_stability"]),
                    "shaped_window_count": int(shaping_result.summary["shaped_window_count"]),
                    "shaped_window_ratio": float(shaping_result.summary["shaped_window_ratio"]),
                    "epsilon_local_mean": float(shaping_result.summary["epsilon_local_mean"]),
                    "epsilon_local_p95": float(shaping_result.summary["epsilon_local_p95"]),
                    "local_step_distortion_ratio_mean": float(shaping_result.summary["local_step_distortion_ratio_mean"]),
                    "local_step_distortion_ratio_p95": float(shaping_result.summary["local_step_distortion_ratio_p95"]),
                    "margin_gain_mean": float(shaping_result.summary["margin_gain_mean"]),
                    "admitted_margin_mean_before": float(shaping_result.summary["admitted_margin_mean_before"]),
                    "admitted_same_dist_mean_before": float(shaping_result.summary["admitted_same_dist_mean_before"]),
                    "margin_to_score_conversion": float(
                        (shaped_result.test_metrics["macro_f1"] - baseline_result.test_metrics["macro_f1"])
                        / (after_diag["nearest_prototype_margin"] - before_diag["nearest_prototype_margin"] + 1e-8)
                    ),
                }
            )

            for row in shaping_result.anchor_rows:
                anchor_rows_all.append({"dataset": str(dataset), "seed": int(seed), **dict(row)})
            for row in shaping_result.shaping_rows:
                shaping_rows_all.append({"dataset": str(dataset), "seed": int(seed), **dict(row)})
            structure_rows_all.append({"dataset": str(dataset), "seed": int(seed), "stage": "before", **dict(before_diag)})
            structure_rows_all.append({"dataset": str(dataset), "seed": int(seed), "stage": "after", **dict(after_diag)})

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
                    "beta": float(args.beta),
                    "epsilon_scale": float(args.epsilon_scale),
                    "minirocket_n_kernels": int(args.minirocket_n_kernels),
                    "minirocket_n_jobs": int(args.minirocket_n_jobs),
                    "same_backbone_baseline": "dense_dynamic_minirocket_no_shaping",
                }
            )

    per_seed_df = pd.DataFrame(per_seed_rows)
    anchor_df = pd.DataFrame(anchor_rows_all)
    shaping_df = pd.DataFrame(shaping_rows_all)
    structure_df = pd.DataFrame(structure_rows_all)
    config_df = pd.DataFrame(config_rows)

    if not anchor_df.empty:
        anchor_summary_df = (
            anchor_df.groupby(["dataset", "seed", "class_id", "prototype_id"], as_index=False)
            .agg(
                prototype_member_count=("prototype_member_count", "max"),
                admitted_anchor_count=("window_index", "count"),
                anchor_coverage_ratio=("anchor_coverage_ratio", "max"),
            )
            .sort_values(["dataset", "seed", "class_id", "prototype_id"])
        )
    else:
        anchor_summary_df = pd.DataFrame(
            columns=[
                "dataset",
                "seed",
                "class_id",
                "prototype_id",
                "prototype_member_count",
                "admitted_anchor_count",
                "anchor_coverage_ratio",
            ]
        )

    for dataset in datasets:
        ds = per_seed_df[per_seed_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        dataset_summary_rows.append(
            {
                "dataset": str(dataset),
                "same_backbone_no_shaping_mean": _mean_std(ds["same_backbone_no_shaping_test_macro_f1"].tolist())[0],
                "same_backbone_no_shaping_std": _mean_std(ds["same_backbone_no_shaping_test_macro_f1"].tolist())[1],
                "local_shaping_mean": _mean_std(ds["local_shaping_test_macro_f1"].tolist())[0],
                "local_shaping_std": _mean_std(ds["local_shaping_test_macro_f1"].tolist())[1],
                "delta_test_macro_f1_mean": _mean_std(ds["delta_test_macro_f1"].tolist())[0],
                "delta_within_compactness_mean": _mean_std(ds["delta_within_compactness"].tolist())[0],
                "delta_between_separation_mean": _mean_std(ds["delta_between_separation"].tolist())[0],
                "delta_nearest_margin_mean": _mean_std(ds["delta_nearest_margin"].tolist())[0],
                "delta_temporal_stability_mean": _mean_std(ds["delta_temporal_stability"].tolist())[0],
                "local_step_distortion_ratio_mean": _mean_std(ds["local_step_distortion_ratio_mean"].tolist())[0],
                "local_step_distortion_ratio_p95_mean": _mean_std(ds["local_step_distortion_ratio_p95"].tolist())[0],
                "margin_gain_mean": _mean_std(ds["margin_gain_mean"].tolist())[0],
                "admitted_margin_mean_before": _mean_std(ds["admitted_margin_mean_before"].tolist())[0],
                "admitted_same_dist_mean_before": _mean_std(ds["admitted_same_dist_mean_before"].tolist())[0],
                "margin_to_score_conversion_mean": _mean_std(ds["margin_to_score_conversion"].tolist())[0],
                "static_linear_mean": _mean_std(ds["static_linear_test_macro_f1"].dropna().tolist())[0],
                "dense_dynamic_gru_mean": _mean_std(ds["dense_dynamic_gru_test_macro_f1"].dropna().tolist())[0],
                "raw_minirocket_mean": _mean_std(ds["raw_minirocket_test_macro_f1"].dropna().tolist())[0],
            }
        )
    dataset_summary_df = pd.DataFrame(dataset_summary_rows)

    config_csv = os.path.join(args.out_root, "scp_branch_v1_config_table.csv")
    per_seed_csv = os.path.join(args.out_root, "scp_branch_v1_per_seed.csv")
    dataset_summary_csv = os.path.join(args.out_root, "scp_branch_v1_dataset_summary.csv")
    anchor_summary_csv = os.path.join(args.out_root, "scp_branch_v1_anchor_summary.csv")
    shaping_csv = os.path.join(args.out_root, "scp_branch_v1_shaping_diagnostics.csv")
    structure_csv = os.path.join(args.out_root, "scp_branch_v1_structure_diagnostics.csv")
    conclusion_md = os.path.join(args.out_root, "scp_branch_v1_conclusion.md")

    config_df.to_csv(config_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    dataset_summary_df.to_csv(dataset_summary_csv, index=False)
    anchor_summary_df.to_csv(anchor_summary_csv, index=False)
    shaping_df.to_csv(shaping_csv, index=False)
    structure_df.to_csv(structure_csv, index=False)

    lines: List[str] = [
        "# SCP-Branch v1 Conclusion",
        "",
        "更新时间：2026-03-30",
        "",
        "same-backbone 对照口径：baseline 与 v1 复用完全相同的 dense backbone、normalization 与 dynamic_minirocket 训练协议；唯一差别是是否执行 train-only local shaping 写回。",
        "当前不做 replay / curriculum / neighborhood propagation / test-time routing。",
        "",
    ]
    for dataset in datasets:
        ds = per_seed_df[per_seed_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(f"- `same_backbone_no_shaping`: {_format_mean_std(ds['same_backbone_no_shaping_test_macro_f1'].tolist())}")
        lines.append(f"- `local_shaping`: {_format_mean_std(ds['local_shaping_test_macro_f1'].tolist())}")
        lines.append(f"- `delta_test_macro_f1`: {_format_mean_std(ds['delta_test_macro_f1'].tolist())}")
        lines.append(f"- `delta_nearest_margin`: {_format_mean_std(ds['delta_nearest_margin'].tolist())}")
        lines.append(f"- `delta_between_separation`: {_format_mean_std(ds['delta_between_separation'].tolist())}")
        lines.append(f"- `delta_within_compactness`: {_format_mean_std(ds['delta_within_compactness'].tolist())}")
        lines.append(f"- `delta_temporal_stability`: {_format_mean_std(ds['delta_temporal_stability'].tolist())}")
        lines.append(f"- `local_step_distortion_ratio_mean`: {_format_mean_std(ds['local_step_distortion_ratio_mean'].tolist())}")
        lines.append(f"- `admitted_margin_mean_before`: {_format_mean_std(ds['admitted_margin_mean_before'].tolist())}")
        lines.append(f"- `admitted_same_dist_mean_before`: {_format_mean_std(ds['admitted_same_dist_mean_before'].tolist())}")
        lines.append(f"- `margin_to_score_conversion`: {_format_mean_std(ds['margin_to_score_conversion'].tolist())}")
        lines.append("")

    with open(conclusion_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(f"[scp-branch-v1] wrote {config_csv}")
    print(f"[scp-branch-v1] wrote {per_seed_csv}")
    print(f"[scp-branch-v1] wrote {dataset_summary_csv}")
    print(f"[scp-branch-v1] wrote {anchor_summary_csv}")
    print(f"[scp-branch-v1] wrote {shaping_csv}")
    print(f"[scp-branch-v1] wrote {structure_csv}")
    print(f"[scp-branch-v1] wrote {conclusion_md}")


if __name__ == "__main__":
    main()
