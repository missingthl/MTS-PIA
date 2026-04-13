#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from time import perf_counter
from typing import Dict, List, Sequence

# Keep BLAS threading conservative in the `pia` environment; dense SCP probes have
# previously segfaulted in OpenBLAS when KMeans and MiniROCKET were chained in one run.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from route_b_unified.scp_closed_form_update import (  # noqa: E402
    SCPClosedFormUpdateConfig,
    run_scp_closed_form_update,
)
from route_b_unified.scp_local_shaping import SCPLocalShapingConfig, apply_scp_local_shaping  # noqa: E402
from route_b_unified.scp_prototype_memory import SCPPrototypeMemoryConfig, build_scp_prototype_memory  # noqa: E402
from route_b_unified.trajectory_feedback_pool_windows import build_window_feedback_reference_stats  # noqa: E402
from route_b_unified.trajectory_minirocket_evaluator import (  # noqa: E402
    TrajectoryMiniRocketEvalConfig,
    evaluate_dynamic_minirocket_classifier,
)
from route_b_unified.trajectory_representation import TrajectoryRepresentationConfig, build_trajectory_representation  # noqa: E402


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


def main() -> None:
    p = argparse.ArgumentParser(description="SCP-Branch v2: geometry refresh probe.")
    p.add_argument("--datasets", type=str, default="selfregulationscp1")
    p.add_argument("--seeds", type=str, default="1")
    p.add_argument("--out-root", type=str, default="out/_active/verify_route_b_scp_branch_v2_20260330")
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
    p.add_argument("--anchor-selection-mode", type=str, default="tight_margin")
    p.add_argument("--same-dist-quantile", type=float, default=50.0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--epsilon-scale", type=float, default=0.10)
    p.add_argument("--proto-update-alpha", type=float, default=0.2)
    args = p.parse_args()

    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_seed_list(args.seeds)
    _ensure_dir(args.out_root)

    config_rows: List[Dict[str, object]] = []
    per_seed_rows: List[Dict[str, object]] = []
    dataset_summary_rows: List[Dict[str, object]] = []
    prototype_update_rows: List[Dict[str, object]] = []
    direction_rows_all: List[Dict[str, object]] = []
    acceptance_rows_all: List[Dict[str, object]] = []
    retrain_reference_protocol = "same_dense_zseq_dynamic_minirocket_retrain"

    for dataset in datasets:
        for seed in seeds:
            state = build_trajectory_representation(
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

            base_memory = build_scp_prototype_memory(
                train_tids=state.train.tids.tolist(),
                train_labels=state.train.y.tolist(),
                train_z_seq_list=state.train.z_seq_list,
                reference_stats=build_window_feedback_reference_stats(
                    train_labels=state.train.y.tolist(),
                    train_z_seq_list=state.train.z_seq_list,
                ),
                cfg=SCPPrototypeMemoryConfig(prototype_count=int(args.prototype_count), cluster_mode="kmeans_centroid", seed=int(seed)),
            )
            base_diag = next(r for r in base_memory.structure_rows if str(r["mode"]) == "prototype_memory")

            shaping = apply_scp_local_shaping(
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

            shaped_memory = build_scp_prototype_memory(
                train_tids=state.train.tids.tolist(),
                train_labels=state.train.y.tolist(),
                train_z_seq_list=shaping.shaped_train_z_seq_list,
                reference_stats=build_window_feedback_reference_stats(
                    train_labels=state.train.y.tolist(),
                    train_z_seq_list=shaping.shaped_train_z_seq_list,
                ),
                cfg=SCPPrototypeMemoryConfig(prototype_count=int(args.prototype_count), cluster_mode="kmeans_centroid", seed=int(seed)),
            )
            shaped_diag = next(r for r in shaped_memory.structure_rows if str(r["mode"]) == "prototype_memory")

            update_result = run_scp_closed_form_update(
                train_tids=state.train.tids.tolist(),
                train_labels=state.train.y.tolist(),
                train_z_seq_list=shaping.shaped_train_z_seq_list,
                cfg=SCPClosedFormUpdateConfig(
                    prototype_count=int(args.prototype_count),
                    anchors_per_prototype=int(args.anchors_per_prototype),
                    anchor_selection_mode=str(args.anchor_selection_mode),
                    same_dist_quantile=float(args.same_dist_quantile),
                    beta=float(args.beta),
                    epsilon_scale=float(args.epsilon_scale),
                    proto_update_alpha=float(args.proto_update_alpha),
                    seed=int(seed),
                ),
            )

            t0 = perf_counter()
            _ = evaluate_dynamic_minirocket_classifier(
                state,
                seed=int(seed),
                eval_cfg=TrajectoryMiniRocketEvalConfig(
                    n_kernels=int(args.minirocket_n_kernels),
                    n_jobs=int(args.minirocket_n_jobs),
                    padding_mode="edge",
                    target_len_mode="train_max_len",
                ),
            )
            retrain_reference_seconds = float(perf_counter() - t0)

            seed_dir = os.path.join(args.out_root, f"{dataset}_seed{seed}")
            _ensure_dir(seed_dir)
            _write_json(
                os.path.join(seed_dir, "scp_branch_v3_result.json"),
                {
                    "v0_structure": dict(base_diag),
                    "v1b_shaped_structure": dict(shaped_diag),
                    "v2_updated_structure": dict(update_result.updated_structure),
                    "v2_summary": dict(update_result.summary),
                    "retrain_reference_seconds": float(retrain_reference_seconds),
                    "retrain_reference_protocol": str(retrain_reference_protocol),
                },
            )

            per_seed_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "v0_within_compactness": float(base_diag["within_prototype_compactness"]),
                    "v1b_within_compactness": float(shaped_diag["within_prototype_compactness"]),
                    "v3_within_compactness": float(update_result.updated_structure["within_prototype_compactness"]),
                    "v0_between_separation": float(base_diag["between_prototype_separation"]),
                    "v1b_between_separation": float(shaped_diag["between_prototype_separation"]),
                    "v3_between_separation": float(update_result.updated_structure["between_prototype_separation"]),
                    "v0_nearest_margin": float(base_diag["nearest_prototype_margin"]),
                    "v1b_nearest_margin": float(shaped_diag["nearest_prototype_margin"]),
                    "v3_nearest_margin": float(update_result.updated_structure["nearest_prototype_margin"]),
                    "v0_temporal_stability": float(base_diag["temporal_assignment_stability"]),
                    "v1b_temporal_stability": float(shaped_diag["temporal_assignment_stability"]),
                    "v3_temporal_stability": float(update_result.updated_structure["temporal_assignment_stability"]),
                    "delta_v3_vs_v1b_within": float(update_result.updated_structure["within_prototype_compactness"] - shaped_diag["within_prototype_compactness"]),
                    "delta_v3_vs_v1b_between": float(update_result.updated_structure["between_prototype_separation"] - shaped_diag["between_prototype_separation"]),
                    "delta_v3_vs_v1b_margin": float(update_result.updated_structure["nearest_prototype_margin"] - shaped_diag["nearest_prototype_margin"]),
                    "delta_v3_vs_v1b_stability": float(update_result.updated_structure["temporal_assignment_stability"] - shaped_diag["temporal_assignment_stability"]),
                    "update_time_seconds": float(update_result.summary["update_time_seconds"]),
                    "retrain_reference_seconds": float(retrain_reference_seconds),
                    "update_to_retrain_ratio": float(update_result.summary["update_time_seconds"] / max(1e-8, retrain_reference_seconds)),
                    "retrain_reference_protocol": str(retrain_reference_protocol),
                    "accept_between_rate": float(update_result.summary["accept_between_rate"]),
                    "accept_margin_rate": float(update_result.summary["accept_margin_rate"]),
                    "accept_within_rate": float(update_result.summary["accept_within_rate"]),
                    "final_accept_rate": float(update_result.summary["final_accept_rate"]),
                    "prototype_shift_norm_mean": float(update_result.summary["prototype_shift_norm_mean"]),
                    "medoid_refresh_dist_mean": float(update_result.summary["medoid_refresh_dist_mean"]),
                    "direction_norm_mean": float(update_result.summary["direction_norm_mean"]),
                    "direction_cosine_to_old_mean": float(update_result.summary["direction_cosine_to_old_mean"]),
                    "direction_change_angle_proxy_mean": float(update_result.summary["direction_change_angle_proxy_mean"]),
                }
            )

            for row in update_result.prototype_rows:
                prototype_update_rows.append({"dataset": str(dataset), "seed": int(seed), **dict(row)})
            for row in update_result.direction_rows:
                direction_rows_all.append({"dataset": str(dataset), "seed": int(seed), **dict(row)})
            for row in update_result.acceptance_rows:
                acceptance_rows_all.append({"dataset": str(dataset), "seed": int(seed), **dict(row)})

            config_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "shared_backbone": "dense_zseq_dynamic_minirocket",
                    "shared_object": "prototype_memory_v1b_tight_anchor_local_shaping",
                    "semantics": "new_v2_geometry_refresh",
                    "prototype_count": int(args.prototype_count),
                    "anchors_per_prototype": int(args.anchors_per_prototype),
                    "anchor_selection_mode": str(args.anchor_selection_mode),
                    "same_dist_quantile": float(args.same_dist_quantile),
                    "beta": float(args.beta),
                    "epsilon_scale": float(args.epsilon_scale),
                    "proto_update_alpha": float(args.proto_update_alpha),
                    "retrain_reference_protocol": str(retrain_reference_protocol),
                }
            )

    per_seed_df = pd.DataFrame(per_seed_rows)
    prototype_df = pd.DataFrame(prototype_update_rows)
    direction_df = pd.DataFrame(direction_rows_all)
    acceptance_df = pd.DataFrame(acceptance_rows_all)
    config_df = pd.DataFrame(config_rows)
    for dataset in datasets:
        ds = per_seed_df[per_seed_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        dataset_summary_rows.append(
            {
                "dataset": str(dataset),
                "delta_v3_vs_v1b_between_mean": _mean_std(ds["delta_v3_vs_v1b_between"].tolist())[0],
                "delta_v3_vs_v1b_margin_mean": _mean_std(ds["delta_v3_vs_v1b_margin"].tolist())[0],
                "delta_v3_vs_v1b_within_mean": _mean_std(ds["delta_v3_vs_v1b_within"].tolist())[0],
                "delta_v3_vs_v1b_stability_mean": _mean_std(ds["delta_v3_vs_v1b_stability"].tolist())[0],
                "update_time_seconds_mean": _mean_std(ds["update_time_seconds"].tolist())[0],
                "retrain_reference_seconds_mean": _mean_std(ds["retrain_reference_seconds"].tolist())[0],
                "update_to_retrain_ratio_mean": _mean_std(ds["update_to_retrain_ratio"].tolist())[0],
                "accept_between_rate_mean": _mean_std(ds["accept_between_rate"].tolist())[0],
                "accept_margin_rate_mean": _mean_std(ds["accept_margin_rate"].tolist())[0],
                "accept_within_rate_mean": _mean_std(ds["accept_within_rate"].tolist())[0],
                "final_accept_rate_mean": _mean_std(ds["final_accept_rate"].tolist())[0],
                "direction_cosine_to_old_mean": _mean_std(ds["direction_cosine_to_old_mean"].tolist())[0],
                "direction_change_angle_proxy_mean": _mean_std(ds["direction_change_angle_proxy_mean"].tolist())[0],
            }
        )
    dataset_summary_df = pd.DataFrame(dataset_summary_rows)

    config_csv = os.path.join(args.out_root, "scp_branch_v2_config_table.csv")
    per_seed_csv = os.path.join(args.out_root, "scp_branch_v2_per_seed.csv")
    dataset_summary_csv = os.path.join(args.out_root, "scp_branch_v2_dataset_summary.csv")
    prototype_csv = os.path.join(args.out_root, "scp_branch_v2_prototype_update_summary.csv")
    direction_csv = os.path.join(args.out_root, "scp_branch_v2_direction_summary.csv")
    acceptance_csv = os.path.join(args.out_root, "scp_branch_v2_acceptance_summary.csv")
    conclusion_md = os.path.join(args.out_root, "scp_branch_v2_conclusion.md")

    config_df.to_csv(config_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    dataset_summary_df.to_csv(dataset_summary_csv, index=False)
    prototype_df.to_csv(prototype_csv, index=False)
    direction_df.to_csv(direction_csv, index=False)
    acceptance_df.to_csv(acceptance_csv, index=False)

    lines: List[str] = [
        "# SCP-Branch v2 Conclusion",
        "",
        "这是 new v2 geometry refresh，不是旧语义下的 v3。",
        "v2 与 v1b 共用同一套 dense backbone、prototype-memory 与 tight anchors/local shaping 对象定义。",
        f"重训参考固定为：`{retrain_reference_protocol}`。",
        "v2 只看 closed-form local geometry refresh 的结构价值与成本，不追终点 F1。",
        "",
    ]
    for dataset in datasets:
        ds = per_seed_df[per_seed_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(f"- `delta_v2_vs_v1b_between`: {_format_mean_std(ds['delta_v3_vs_v1b_between'].tolist())}")
        lines.append(f"- `delta_v2_vs_v1b_margin`: {_format_mean_std(ds['delta_v3_vs_v1b_margin'].tolist())}")
        lines.append(f"- `delta_v2_vs_v1b_within`: {_format_mean_std(ds['delta_v3_vs_v1b_within'].tolist())}")
        lines.append(f"- `delta_v2_vs_v1b_stability`: {_format_mean_std(ds['delta_v3_vs_v1b_stability'].tolist())}")
        lines.append(f"- `accept_between_rate`: {_format_mean_std(ds['accept_between_rate'].tolist())}")
        lines.append(f"- `accept_margin_rate`: {_format_mean_std(ds['accept_margin_rate'].tolist())}")
        lines.append(f"- `accept_within_rate`: {_format_mean_std(ds['accept_within_rate'].tolist())}")
        lines.append(f"- `final_accept_rate`: {_format_mean_std(ds['final_accept_rate'].tolist())}")
        lines.append(f"- `direction_cosine_to_old_mean`: {_format_mean_std(ds['direction_cosine_to_old_mean'].tolist())}")
        lines.append(f"- `direction_change_angle_proxy_mean`: {_format_mean_std(ds['direction_change_angle_proxy_mean'].tolist())}")
        lines.append(f"- `update_to_retrain_ratio`: {_format_mean_std(ds['update_to_retrain_ratio'].tolist())}")
        lines.append("")

    with open(conclusion_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(f"[scp-branch-v2] wrote {config_csv}")
    print(f"[scp-branch-v2] wrote {per_seed_csv}")
    print(f"[scp-branch-v2] wrote {dataset_summary_csv}")
    print(f"[scp-branch-v2] wrote {prototype_csv}")
    print(f"[scp-branch-v2] wrote {direction_csv}")
    print(f"[scp-branch-v2] wrote {acceptance_csv}")
    print(f"[scp-branch-v2] wrote {conclusion_md}")


if __name__ == "__main__":
    main()
