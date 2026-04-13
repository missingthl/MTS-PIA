#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from route_b_unified.manifold_diagnostics import build_embedding_maps, ensure_dir, resolve_embedding_methods  # noqa: E402
from route_b_unified.trajectory_feedback_pool import (  # noqa: E402
    TrajectoryFeedbackPoolConfig,
    build_trajectory_feedback_pool,
)
from route_b_unified.trajectory_feedback_rebasis import fit_trajectory_feedback_rebasis  # noqa: E402
from route_b_unified.trajectory_pia_operator import TrajectoryPIAOperator, TrajectoryPIAOperatorConfig  # noqa: E402
from route_b_unified.trajectory_pia_operator_t2b import (  # noqa: E402
    TrajectoryPIAT2B0Config,
    TrajectoryPIAT2B0Operator,
)
from route_b_unified.trajectory_representation import (  # noqa: E402
    TrajectoryRepresentationConfig,
    build_trajectory_representation,
)


STAGE_ORDER = ["baseline", "t2a_default", "t2b_saliency", "t3_rebasis"]


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


def _trajectory_means(z_seq_list: Sequence[np.ndarray]) -> np.ndarray:
    rows = [np.mean(np.asarray(seq, dtype=np.float64), axis=0) for seq in z_seq_list]
    return np.stack(rows, axis=0).astype(np.float64)


def _plot_stage_panels(
    path: str,
    *,
    coords_by_stage: Mapping[str, np.ndarray],
    y: np.ndarray,
    dataset: str,
    seed: int,
    method: str,
) -> List[Dict[str, object]]:
    labels = np.asarray(y, dtype=np.int64)
    uniq = sorted(set(int(v) for v in labels.tolist()))
    cmap = plt.cm.get_cmap("tab10", len(uniq))
    fig, axes = plt.subplots(1, 5, figsize=(23, 4.8), constrained_layout=True)

    all_coords = np.concatenate([np.asarray(coords_by_stage[name], dtype=np.float64) for name in STAGE_ORDER], axis=0)
    xmin, ymin = np.min(all_coords, axis=0)
    xmax, ymax = np.max(all_coords, axis=0)
    padx = max(1e-6, 0.08 * float(xmax - xmin))
    pady = max(1e-6, 0.08 * float(ymax - ymin))

    centroid_rows: List[Dict[str, object]] = []
    centroid_map: Dict[str, Dict[int, np.ndarray]] = {stage: {} for stage in STAGE_ORDER}
    stage_titles = {
        "baseline": "baseline",
        "t2a_default": "T2a",
        "t2b_saliency": "T2b-0",
        "t3_rebasis": "T3",
    }

    for ax, stage in zip(axes[:4], STAGE_ORDER):
        coords = np.asarray(coords_by_stage[stage], dtype=np.float64)
        for i, cls in enumerate(uniq):
            mask = labels == int(cls)
            pts = coords[mask]
            ax.scatter(pts[:, 0], pts[:, 1], s=16, alpha=0.35, color=cmap(i), label=f"class {cls}")
            centroid = np.mean(pts, axis=0)
            centroid_map[stage][int(cls)] = np.asarray(centroid, dtype=np.float64)
            ax.scatter(centroid[0], centroid[1], s=70, marker="X", color=cmap(i), edgecolors="black", linewidths=0.5)
            centroid_rows.append(
                {
                    "stage": str(stage),
                    "class_id": int(cls),
                    "centroid_x": float(centroid[0]),
                    "centroid_y": float(centroid[1]),
                }
            )
        ax.set_title(stage_titles[stage])
        ax.set_xlabel("dim-1")
        ax.set_ylabel("dim-2")
        ax.set_xlim(float(xmin - padx), float(xmax + padx))
        ax.set_ylim(float(ymin - pady), float(ymax + pady))

    ax = axes[4]
    for i, cls in enumerate(uniq):
        pts = [centroid_map[stage][int(cls)] for stage in STAGE_ORDER]
        arr = np.stack(pts, axis=0)
        ax.plot(arr[:, 0], arr[:, 1], color=cmap(i), lw=2.0, alpha=0.9)
        ax.scatter(arr[:, 0], arr[:, 1], color=cmap(i), s=60)
        for sidx in range(arr.shape[0] - 1):
            p0 = arr[sidx]
            p1 = arr[sidx + 1]
            ax.annotate(
                "",
                xy=(p1[0], p1[1]),
                xytext=(p0[0], p0[1]),
                arrowprops=dict(arrowstyle="->", color=cmap(i), lw=1.4, alpha=0.9),
            )
        ax.text(arr[-1, 0], arr[-1, 1], f"class {cls}", color=cmap(i), fontsize=8)
    ax.set_title("Centroid Evolution")
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.set_xlim(float(xmin - padx), float(xmax + padx))
    ax.set_ylim(float(ymin - pady), float(ymax + pady))

    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=min(8, len(handles)))
    fig.suptitle(f"{dataset} dynamic manifold evolution ({method}, seed={seed})")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return centroid_rows


def main() -> None:
    p = argparse.ArgumentParser(description="Dynamic manifold evolution plot: baseline -> T2a -> T2b-0 -> T3 on a shared embedding.")
    p.add_argument("--datasets", type=str, default="natops,selfregulationscp1")
    p.add_argument("--seeds", type=str, default="1")
    p.add_argument("--out-root", type=str, default="out/_active/route_b_dynamic_manifold_evolution_20260329")
    p.add_argument("--embedding-methods", type=str, default="auto")
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--prop-win-ratio", type=float, default=0.20)
    p.add_argument("--prop-hop-ratio", type=float, default=0.10)
    p.add_argument("--min-window-extra-channels", type=int, default=4)
    p.add_argument("--min-hop-len", type=int, default=4)
    p.add_argument("--gamma-main", type=float, default=0.05)
    p.add_argument("--smooth-lambda", type=float, default=0.50)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--max-purity-drop", type=float, default=0.10)
    p.add_argument("--continuity-quantile", type=float, default=75.0)
    args = p.parse_args()

    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_seed_list(args.seeds)
    methods = resolve_embedding_methods(str(args.embedding_methods))
    out_root = os.path.abspath(str(args.out_root))
    ensure_dir(out_root)
    vis_root = os.path.join(out_root, "VIS")
    ensure_dir(vis_root)

    summary_rows: List[Dict[str, object]] = []
    point_rows: List[Dict[str, object]] = []
    centroid_rows_all: List[Dict[str, object]] = []

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
                )
            )
            ds_dir = os.path.join(out_root, f"{dataset}_seed{seed}")
            vis_dir = os.path.join(vis_root, f"{dataset}_seed{seed}")
            ensure_dir(ds_dir)
            ensure_dir(vis_dir)

            train_tids = [str(v) for v in state.train.tids.tolist()]
            labels = np.asarray(state.train.y, dtype=np.int64)
            train_seqs = [np.asarray(v, dtype=np.float32) for v in state.train.z_seq_list]

            base_operator = TrajectoryPIAOperator(TrajectoryPIAOperatorConfig(seed=int(seed))).fit(train_seqs)
            t2a_aug, _, _ = base_operator.transform_many(
                train_seqs,
                gamma_main=float(args.gamma_main),
                smooth_lambda=float(args.smooth_lambda),
            )

            t2b_operator = TrajectoryPIAT2B0Operator(
                base_cfg=TrajectoryPIAOperatorConfig(seed=int(seed)),
                t2b_cfg=TrajectoryPIAT2B0Config(
                    gamma_base=float(args.gamma_main),
                    smooth_lambda=float(args.smooth_lambda),
                    seed=int(seed),
                ),
            ).fit(train_seqs, prefit_base_operator=base_operator)
            t2b_aug, _, _, _ = t2b_operator.transform_many(
                train_seqs,
                mode="saliency",
                trial_ids=train_tids,
            )

            pool_result = build_trajectory_feedback_pool(
                train_tids=train_tids,
                train_labels=[int(v) for v in labels.tolist()],
                train_z_seq_list=train_seqs,
                operator=base_operator,
                cfg=TrajectoryFeedbackPoolConfig(
                    gamma_main=float(args.gamma_main),
                    smooth_lambda=float(args.smooth_lambda),
                    knn_k=int(args.knn_k),
                    max_purity_drop=float(args.max_purity_drop),
                    continuity_quantile=float(args.continuity_quantile),
                ),
            )
            rebasis_result = fit_trajectory_feedback_rebasis(
                orig_train_z_seq_list=train_seqs,
                feedback_z_seq_list=pool_result.accepted_z_seq_list,
                old_operator=base_operator,
                operator_cfg=TrajectoryPIAOperatorConfig(seed=int(seed)),
            )
            t3_aug, _, _ = rebasis_result.operator_new.transform_many(
                train_seqs,
                gamma_main=float(args.gamma_main),
                smooth_lambda=float(args.smooth_lambda),
            )

            X_by_stage = {
                "baseline": _trajectory_means(train_seqs),
                "t2a_default": _trajectory_means(t2a_aug),
                "t2b_saliency": _trajectory_means(t2b_aug),
                "t3_rebasis": _trajectory_means(t3_aug),
            }
            for method in methods:
                coords_by_stage = build_embedding_maps(X_by_stage, method=str(method), seed=int(seed))
                plot_path = os.path.join(vis_dir, f"{dataset}_seed{seed}_evolution_{method}.png")
                centroid_rows = _plot_stage_panels(
                    plot_path,
                    coords_by_stage=coords_by_stage,
                    y=labels,
                    dataset=str(dataset),
                    seed=int(seed),
                    method=str(method),
                )
                for stage, coords in coords_by_stage.items():
                    for tid, cls, xy in zip(train_tids, labels.tolist(), np.asarray(coords, dtype=np.float64)):
                        point_rows.append(
                            {
                                "dataset": str(dataset),
                                "seed": int(seed),
                                "method": str(method),
                                "stage": str(stage),
                                "trial_id": str(tid),
                                "class_id": int(cls),
                                "x": float(xy[0]),
                                "y": float(xy[1]),
                            }
                        )
                for row in centroid_rows:
                    centroid_rows_all.append(
                        {
                            "dataset": str(dataset),
                            "seed": int(seed),
                            "method": str(method),
                            **row,
                        }
                    )
                summary_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "method": str(method),
                        "plot_png": os.path.abspath(plot_path),
                        "accepted_feedback_count": int(pool_result.summary["accepted_count"]),
                        "accept_rate": float(pool_result.summary["accept_rate"]),
                        "center_shift_norm": float(rebasis_result.summary["center_shift_norm"]),
                        "basis_cosine_to_old": float(rebasis_result.summary["basis_cosine_to_old"]),
                    }
                )

            _write_path = os.path.join(ds_dir, "evolution_meta.json")
            with open(_write_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "gamma_main": float(args.gamma_main),
                        "smooth_lambda": float(args.smooth_lambda),
                        "feedback_summary": dict(pool_result.summary),
                        "rebasis_summary": dict(rebasis_result.summary),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    pd.DataFrame(summary_rows).to_csv(os.path.join(out_root, "dynamic_manifold_evolution_summary.csv"), index=False)
    pd.DataFrame(point_rows).to_csv(os.path.join(out_root, "dynamic_manifold_evolution_points.csv"), index=False)
    pd.DataFrame(centroid_rows_all).to_csv(os.path.join(out_root, "dynamic_manifold_evolution_centroids.csv"), index=False)

    conclusion_path = os.path.join(out_root, "dynamic_manifold_evolution_conclusion.md")
    with open(conclusion_path, "w", encoding="utf-8") as f:
        f.write("# Dynamic Manifold Evolution Plot\n\n")
        f.write("This figure tracks the same original train trajectories across four dynamic-stage representations:\n\n")
        f.write("- baseline\n")
        f.write("- T2a default\n")
        f.write("- T2b-0 saliency-aware\n")
        f.write("- T3 rebasis\n\n")
        f.write("The plot uses a shared embedding across stages, plus a centroid-evolution panel.\n")
        f.write("This is closer to the current dynamic branch than the older static manifold scatter and can be read as a manifold-evolution view.\n")

    print(f"[evolution] wrote summary to: {os.path.join(out_root, 'dynamic_manifold_evolution_summary.csv')}")
    print(f"[evolution] wrote points to: {os.path.join(out_root, 'dynamic_manifold_evolution_points.csv')}")
    print(f"[evolution] wrote centroids to: {os.path.join(out_root, 'dynamic_manifold_evolution_centroids.csv')}")
    print(f"[evolution] wrote conclusion to: {conclusion_path}")


if __name__ == "__main__":
    main()
