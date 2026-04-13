#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from route_b_unified.scp_local_shaping import SCPLocalShapingConfig, apply_scp_local_shaping  # noqa: E402
from route_b_unified.scp_replay import SCPSingleReplayConfig, build_single_replay_state  # noqa: E402
from route_b_unified.trajectory_minirocket_evaluator import (  # noqa: E402
    TrajectoryMiniRocketEvalConfig,
    evaluate_dynamic_minirocket_classifier,
)
from route_b_unified.trajectory_representation import (  # noqa: E402
    TrajectoryRepresentationConfig,
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


def main() -> None:
    p = argparse.ArgumentParser(description="SCP-Branch v2: Single replay round under shared dense backbone/object definition.")
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
    args = p.parse_args()

    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_seed_list(args.seeds)
    _ensure_dir(args.out_root)

    config_rows: List[Dict[str, object]] = []
    per_seed_rows: List[Dict[str, object]] = []
    dataset_summary_rows: List[Dict[str, object]] = []
    replay_rows_all: List[Dict[str, object]] = []

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
            replay = build_single_replay_state(
                state=state,
                shaped_train_z_seq_list=shaping.shaped_train_z_seq_list,
                shaping_rows=shaping.shaping_rows,
                cfg=SCPSingleReplayConfig(),
            )

            eval_cfg = TrajectoryMiniRocketEvalConfig(
                n_kernels=int(args.minirocket_n_kernels),
                n_jobs=int(args.minirocket_n_jobs),
                padding_mode="edge",
                target_len_mode="train_max_len",
            )
            baseline = evaluate_dynamic_minirocket_classifier(state, seed=int(seed), eval_cfg=eval_cfg)

            from dataclasses import replace

            shaped_train = replace(state.train, z_seq_list=[np.asarray(v, dtype=np.float32) for v in shaping.shaped_train_z_seq_list])
            shaped_state = replace(state, train=shaped_train)
            v1b = evaluate_dynamic_minirocket_classifier(shaped_state, seed=int(seed), eval_cfg=eval_cfg)
            v2 = evaluate_dynamic_minirocket_classifier(replay.replay_state, seed=int(seed), eval_cfg=eval_cfg)

            seed_dir = os.path.join(args.out_root, f"{dataset}_seed{seed}")
            _ensure_dir(seed_dir)
            _write_json(
                os.path.join(seed_dir, "scp_branch_v2_result.json"),
                {
                    "same_backbone_no_shaping": dict(test_macro_f1=float(baseline.test_metrics["macro_f1"]), val_macro_f1=float(baseline.val_metrics["macro_f1"])),
                    "v1b_local_shaping": dict(test_macro_f1=float(v1b.test_metrics["macro_f1"]), val_macro_f1=float(v1b.val_metrics["macro_f1"])),
                    "v2_single_replay": dict(test_macro_f1=float(v2.test_metrics["macro_f1"]), val_macro_f1=float(v2.val_metrics["macro_f1"])),
                    "replay_summary": dict(replay.summary),
                    "shaping_summary": dict(shaping.summary),
                },
            )

            per_seed_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "same_backbone_no_shaping_test_macro_f1": float(baseline.test_metrics["macro_f1"]),
                    "v1b_local_shaping_test_macro_f1": float(v1b.test_metrics["macro_f1"]),
                    "v2_single_replay_test_macro_f1": float(v2.test_metrics["macro_f1"]),
                    "delta_v1b_vs_no_shaping": float(v1b.test_metrics["macro_f1"] - baseline.test_metrics["macro_f1"]),
                    "delta_v2_vs_no_shaping": float(v2.test_metrics["macro_f1"] - baseline.test_metrics["macro_f1"]),
                    "delta_v2_vs_v1b": float(v2.test_metrics["macro_f1"] - v1b.test_metrics["macro_f1"]),
                    "replay_window_ratio": float(replay.summary["replay_window_ratio"]),
                    "stitch_boundary_count": int(replay.summary["stitch_boundary_count"]),
                    "stitch_boundary_jump_ratio_mean": float(replay.summary["stitch_boundary_jump_ratio_mean"]),
                    "stitch_boundary_jump_ratio_p95": float(replay.summary["stitch_boundary_jump_ratio_p95"]),
                    "replay_continuity_distortion_ratio": float(replay.summary["replay_continuity_distortion_ratio"]),
                    "local_step_distortion_ratio_mean": float(shaping.summary["local_step_distortion_ratio_mean"]),
                }
            )
            for row in replay.replay_rows:
                replay_rows_all.append({"dataset": str(dataset), "seed": int(seed), **dict(row)})

            config_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "shared_backbone": "dense_zseq_dynamic_minirocket",
                    "shared_object": "prototype_memory_v1b_tight_anchor_local_shaping",
                    "prototype_count": int(args.prototype_count),
                    "anchors_per_prototype": int(args.anchors_per_prototype),
                    "anchor_selection_mode": str(args.anchor_selection_mode),
                    "same_dist_quantile": float(args.same_dist_quantile),
                    "beta": float(args.beta),
                    "epsilon_scale": float(args.epsilon_scale),
                }
            )

    per_seed_df = pd.DataFrame(per_seed_rows)
    replay_df = pd.DataFrame(replay_rows_all)
    config_df = pd.DataFrame(config_rows)
    for dataset in datasets:
        ds = per_seed_df[per_seed_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        dataset_summary_rows.append(
            {
                "dataset": str(dataset),
                "same_backbone_no_shaping_mean": _mean_std(ds["same_backbone_no_shaping_test_macro_f1"].tolist())[0],
                "v1b_local_shaping_mean": _mean_std(ds["v1b_local_shaping_test_macro_f1"].tolist())[0],
                "v2_single_replay_mean": _mean_std(ds["v2_single_replay_test_macro_f1"].tolist())[0],
                "delta_v2_vs_no_shaping_mean": _mean_std(ds["delta_v2_vs_no_shaping"].tolist())[0],
                "delta_v2_vs_v1b_mean": _mean_std(ds["delta_v2_vs_v1b"].tolist())[0],
                "stitch_boundary_jump_ratio_mean": _mean_std(ds["stitch_boundary_jump_ratio_mean"].tolist())[0],
                "replay_continuity_distortion_ratio_mean": _mean_std(ds["replay_continuity_distortion_ratio"].tolist())[0],
                "replay_window_ratio_mean": _mean_std(ds["replay_window_ratio"].tolist())[0],
            }
        )
    dataset_summary_df = pd.DataFrame(dataset_summary_rows)

    config_csv = os.path.join(args.out_root, "scp_branch_v2_config_table.csv")
    per_seed_csv = os.path.join(args.out_root, "scp_branch_v2_per_seed.csv")
    dataset_summary_csv = os.path.join(args.out_root, "scp_branch_v2_dataset_summary.csv")
    replay_csv = os.path.join(args.out_root, "scp_branch_v2_replay_diagnostics.csv")
    conclusion_md = os.path.join(args.out_root, "scp_branch_v2_conclusion.md")

    config_df.to_csv(config_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    dataset_summary_df.to_csv(dataset_summary_csv, index=False)
    replay_df.to_csv(replay_csv, index=False)

    lines: List[str] = [
        "# SCP-Branch v2 Conclusion",
        "",
        "两条线共用同一套 dense backbone、prototype-memory 与 v1b tight anchors/local shaping 口径。",
        "v2 只测 single replay round 的训练闭环吸收，不做 online / multi-round / test-time update。",
        "",
    ]
    for dataset in datasets:
        ds = per_seed_df[per_seed_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(f"- `same_backbone_no_shaping`: {_format_mean_std(ds['same_backbone_no_shaping_test_macro_f1'].tolist())}")
        lines.append(f"- `v1b_local_shaping`: {_format_mean_std(ds['v1b_local_shaping_test_macro_f1'].tolist())}")
        lines.append(f"- `v2_single_replay`: {_format_mean_std(ds['v2_single_replay_test_macro_f1'].tolist())}")
        lines.append(f"- `delta_v2_vs_v1b`: {_format_mean_std(ds['delta_v2_vs_v1b'].tolist())}")
        lines.append(f"- `stitch_boundary_jump_ratio_mean`: {_format_mean_std(ds['stitch_boundary_jump_ratio_mean'].tolist())}")
        lines.append(f"- `replay_continuity_distortion_ratio`: {_format_mean_std(ds['replay_continuity_distortion_ratio'].tolist())}")
        lines.append("")

    with open(conclusion_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(f"[scp-branch-v2] wrote {config_csv}")
    print(f"[scp-branch-v2] wrote {per_seed_csv}")
    print(f"[scp-branch-v2] wrote {dataset_summary_csv}")
    print(f"[scp-branch-v2] wrote {replay_csv}")
    print(f"[scp-branch-v2] wrote {conclusion_md}")


if __name__ == "__main__":
    main()
