#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from route_b_unified.scp_prototype_memory import (  # noqa: E402
    SCPPrototypeMemoryConfig,
    build_scp_prototype_memory,
)
from route_b_unified.trajectory_classifier import TrajectoryModelConfig  # noqa: E402
from route_b_unified.trajectory_evaluator import TrajectoryEvalConfig, evaluate_trajectory_classifier  # noqa: E402
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


def _load_raw_reference_map(path: str) -> Dict[tuple[str, int], float]:
    if not os.path.isfile(path):
        return {}
    df = pd.read_csv(path)
    score_col = None
    for cand in ["raw_test_macro_f1", "test_macro_f1", "trial_macro_f1"]:
        if cand in df.columns:
            score_col = cand
            break
    if score_col is None or "dataset" not in df.columns or "seed" not in df.columns:
        return {}
    out: Dict[tuple[str, int], float] = {}
    for _, row in df.iterrows():
        out[(str(row["dataset"]).strip().lower(), int(row["seed"]))] = float(row[score_col])
    return out


def _load_dense_backbone_reference_map(path: str) -> Dict[tuple[str, int], Dict[str, float]]:
    if not os.path.isfile(path):
        return {}
    df = pd.read_csv(path)
    required = {
        "dataset",
        "seed",
        "static_linear_test_macro_f1",
        "dense_dynamic_minirocket_test_macro_f1",
    }
    if not required.issubset(set(df.columns)):
        return {}
    out: Dict[tuple[str, int], Dict[str, float]] = {}
    for _, row in df.iterrows():
        out[(str(row["dataset"]).strip().lower(), int(row["seed"]))] = {
            "static_linear_test_macro_f1": float(row["static_linear_test_macro_f1"]),
            "dense_dynamic_gru_test_macro_f1": float(row["dense_dynamic_gru_test_macro_f1"])
            if "dense_dynamic_gru_test_macro_f1" in row and not pd.isna(row["dense_dynamic_gru_test_macro_f1"])
            else np.nan,
            "dense_dynamic_minirocket_test_macro_f1": float(row["dense_dynamic_minirocket_test_macro_f1"]),
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
    p = argparse.ArgumentParser(description="SCP-Branch v0: dense backbone + prototype-memory probe.")
    p.add_argument("--datasets", type=str, default="selfregulationscp1")
    p.add_argument("--seeds", type=str, default="1")
    p.add_argument("--out-root", type=str, default="out/_active/verify_route_b_scp_branch_v0_20260330")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--prop-win-ratio", type=float, default=0.20)
    p.add_argument("--prop-hop-ratio", type=float, default=0.10)
    p.add_argument("--min-window-extra-channels", type=int, default=4)
    p.add_argument("--min-hop-len", type=int, default=4)
    p.add_argument("--force-hop-len", type=int, default=1)
    p.add_argument("--gru-hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--skip-dense-gru", action="store_true")
    p.add_argument("--minirocket-n-kernels", type=int, default=10000)
    p.add_argument("--minirocket-n-jobs", type=int, default=1)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--purity-quantile", type=float, default=50.0)
    p.add_argument("--continuity-quantile", type=float, default=75.0)
    p.add_argument("--prototype-count", type=int, default=4)
    p.add_argument(
        "--backbone-reference-csv",
        type=str,
        default="/home/THL/project/MTS-PIA/out/route_b_dense_trajectory_probe_20260330_formal/dense_trajectory_probe_per_seed.csv",
    )
    p.add_argument(
        "--raw-reference-csv",
        type=str,
        default="/home/THL/project/MTS-PIA/out/route_b_pia_core_minimal_chain_20260327_formal/pia_core_minimal_chain_per_seed.csv",
    )
    args = p.parse_args()

    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_seed_list(args.seeds)
    _ensure_dir(args.out_root)
    raw_ref_map = _load_raw_reference_map(args.raw_reference_csv)
    dense_ref_map = _load_dense_backbone_reference_map(args.backbone_reference_csv)

    config_rows: List[Dict[str, object]] = []
    per_seed_rows: List[Dict[str, object]] = []
    memory_rows: List[Dict[str, object]] = []
    prototype_rows: List[Dict[str, object]] = []
    random_rows: List[Dict[str, object]] = []
    structure_rows: List[Dict[str, object]] = []

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

            ref_key = (str(dataset), int(seed))
            dense_ref = dense_ref_map.get(ref_key)
            if dense_ref is None:
                model_cfg = TrajectoryModelConfig(
                    z_dim=int(state.z_dim),
                    num_classes=int(state.num_classes),
                    gru_hidden_dim=int(args.gru_hidden_dim),
                    dropout=float(args.dropout),
                )
                static_result = evaluate_trajectory_classifier(
                    state,
                    seed=int(seed),
                    model_cfg=model_cfg,
                    eval_cfg=TrajectoryEvalConfig(
                        variant="static_linear",
                        epochs=int(args.epochs),
                        batch_size=int(args.batch_size),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                        patience=int(args.patience),
                        device=str(args.device),
                    ),
                )
                dense_gru = None
                if not bool(args.skip_dense_gru):
                    dense_gru = evaluate_trajectory_classifier(
                        state,
                        seed=int(seed),
                        model_cfg=model_cfg,
                        eval_cfg=TrajectoryEvalConfig(
                            variant="dynamic_gru",
                            epochs=int(args.epochs),
                            batch_size=int(args.batch_size),
                            lr=float(args.lr),
                            weight_decay=float(args.weight_decay),
                            patience=int(args.patience),
                            device=str(args.device),
                        ),
                    )
                dense_mini = evaluate_dynamic_minirocket_classifier(
                    state,
                    seed=int(seed),
                    eval_cfg=TrajectoryMiniRocketEvalConfig(
                        n_kernels=int(args.minirocket_n_kernels),
                        n_jobs=int(args.minirocket_n_jobs),
                        padding_mode="edge",
                        target_len_mode="train_max_len",
                    ),
                )
                static_macro_f1 = float(static_result.test_metrics["macro_f1"])
                dense_gru_macro_f1 = float(dense_gru.test_metrics["macro_f1"]) if dense_gru is not None else np.nan
                dense_mini_macro_f1 = float(dense_mini.test_metrics["macro_f1"])
                backbone_source = "live_eval"
            else:
                static_result = None
                dense_gru = None
                dense_mini = None
                static_macro_f1 = float(dense_ref["static_linear_test_macro_f1"])
                dense_gru_macro_f1 = float(dense_ref["dense_dynamic_gru_test_macro_f1"])
                if bool(args.skip_dense_gru):
                    dense_gru_macro_f1 = np.nan
                dense_mini_macro_f1 = float(dense_ref["dense_dynamic_minirocket_test_macro_f1"])
                backbone_source = "reference_csv"

            ref_stats = build_window_feedback_reference_stats(
                train_labels=state.train.y.tolist(),
                train_z_seq_list=state.train.z_seq_list,
            )
            memory_result = build_scp_prototype_memory(
                train_tids=state.train.tids.tolist(),
                train_labels=state.train.y.tolist(),
                train_z_seq_list=state.train.z_seq_list,
                reference_stats=ref_stats,
                cfg=SCPPrototypeMemoryConfig(
                    knn_k=int(args.knn_k),
                    purity_quantile=float(args.purity_quantile),
                    continuity_quantile=float(args.continuity_quantile),
                    prototype_count=int(args.prototype_count),
                    cluster_mode="kmeans_centroid",
                    seed=int(seed),
                ),
            )

            _write_json(
                os.path.join(seed_dir, "scp_branch_v0_result.json"),
                {
                    "dense_backbone_scores": {
                        "static_linear": dict(static_result.test_metrics) if static_result is not None else None,
                        "dense_dynamic_gru": dict(dense_gru.test_metrics) if dense_gru is not None else None,
                        "dense_dynamic_minirocket": dict(dense_mini.test_metrics) if dense_mini is not None else None,
                        "static_linear_test_macro_f1": static_macro_f1,
                        "dense_dynamic_gru_test_macro_f1": None if np.isnan(dense_gru_macro_f1) else dense_gru_macro_f1,
                        "dense_dynamic_minirocket_test_macro_f1": dense_mini_macro_f1,
                        "backbone_source": backbone_source,
                        "raw_minirocket_test_macro_f1": raw_ref_map.get((str(dataset), int(seed))),
                    },
                    "memory_summary": dict(memory_result.summary),
                    "structure_rows": [dict(v) for v in memory_result.structure_rows],
                },
            )

            prototype_diag = next(r for r in memory_result.structure_rows if str(r["mode"]) == "prototype_memory")
            random_diag = next(r for r in memory_result.structure_rows if str(r["mode"]) == "random_memory_control")
            raw_ref = float(raw_ref_map.get((str(dataset), int(seed)), np.nan))
            per_seed_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "static_linear_test_macro_f1": float(static_macro_f1),
                    "dense_dynamic_gru_test_macro_f1": float(dense_gru_macro_f1),
                    "dense_dynamic_minirocket_test_macro_f1": float(dense_mini_macro_f1),
                    "backbone_source": str(backbone_source),
                    "raw_minirocket_test_macro_f1": raw_ref,
                    "candidate_window_count": int(memory_result.summary["candidate_window_count"]),
                    "safe_window_count": int(memory_result.summary["safe_window_count"]),
                    "prototype_class_count": int(memory_result.summary["prototype_class_count"]),
                    "low_coverage_class_count": int(memory_result.summary["low_coverage_class_count"]),
                    "prototype_within_compactness": float(prototype_diag["within_prototype_compactness"]),
                    "prototype_between_separation": float(prototype_diag["between_prototype_separation"]),
                    "prototype_margin": float(prototype_diag["nearest_prototype_margin"]),
                    "prototype_temporal_stability": float(prototype_diag["temporal_assignment_stability"]),
                    "random_within_compactness": float(random_diag["within_prototype_compactness"]),
                    "random_between_separation": float(random_diag["between_prototype_separation"]),
                    "random_margin": float(random_diag["nearest_prototype_margin"]),
                    "random_temporal_stability": float(random_diag["temporal_assignment_stability"]),
                }
            )

            for row in memory_result.class_summary_rows:
                memory_rows.append({"dataset": str(dataset), "seed": int(seed), **dict(row)})
            for row in memory_result.prototype_rows:
                prototype_rows.append({"dataset": str(dataset), "seed": int(seed), **dict(row)})
            for row in memory_result.random_control_rows:
                random_rows.append({"dataset": str(dataset), "seed": int(seed), **dict(row)})
            for row in memory_result.structure_rows:
                structure_rows.append({"dataset": str(dataset), "seed": int(seed), **dict(row)})

            config_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "window_len": int(state.window_len),
                    "hop_len": int(state.hop_len),
                    "z_dim": int(state.z_dim),
                    "num_classes": int(state.num_classes),
                    "gru_hidden_dim": int(args.gru_hidden_dim),
                    "dropout": float(args.dropout),
                    "skip_dense_gru": int(bool(args.skip_dense_gru)),
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "patience": int(args.patience),
                    "minirocket_n_kernels": int(args.minirocket_n_kernels),
                    "minirocket_n_jobs": int(args.minirocket_n_jobs),
                    "knn_k": int(args.knn_k),
                    "purity_quantile": float(args.purity_quantile),
                    "continuity_quantile": float(args.continuity_quantile),
                    "prototype_count": int(args.prototype_count),
                    "cluster_mode": "kmeans_centroid",
                    "backbone_reference_csv": str(args.backbone_reference_csv),
                }
            )

    config_df = pd.DataFrame(config_rows)
    per_seed_df = pd.DataFrame(per_seed_rows)
    memory_df = pd.DataFrame(memory_rows)
    prototype_df = pd.DataFrame(prototype_rows)
    random_df = pd.DataFrame(random_rows)
    structure_df = pd.DataFrame(structure_rows)

    dataset_summary_rows: List[Dict[str, object]] = []
    for dataset in datasets:
        ds = per_seed_df[per_seed_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        dataset_summary_rows.append(
            {
                "dataset": str(dataset),
                "static_linear_mean": _mean_std(ds["static_linear_test_macro_f1"].tolist())[0],
                "static_linear_std": _mean_std(ds["static_linear_test_macro_f1"].tolist())[1],
                "dense_dynamic_gru_mean": _mean_std(ds["dense_dynamic_gru_test_macro_f1"].dropna().tolist())[0],
                "dense_dynamic_gru_std": _mean_std(ds["dense_dynamic_gru_test_macro_f1"].dropna().tolist())[1],
                "dense_dynamic_minirocket_mean": _mean_std(ds["dense_dynamic_minirocket_test_macro_f1"].tolist())[0],
                "dense_dynamic_minirocket_std": _mean_std(ds["dense_dynamic_minirocket_test_macro_f1"].tolist())[1],
                "raw_minirocket_mean": _mean_std(ds["raw_minirocket_test_macro_f1"].dropna().tolist())[0],
                "raw_minirocket_std": _mean_std(ds["raw_minirocket_test_macro_f1"].dropna().tolist())[1],
                "prototype_within_compactness_mean": _mean_std(ds["prototype_within_compactness"].tolist())[0],
                "prototype_between_separation_mean": _mean_std(ds["prototype_between_separation"].tolist())[0],
                "prototype_margin_mean": _mean_std(ds["prototype_margin"].tolist())[0],
                "prototype_temporal_stability_mean": _mean_std(ds["prototype_temporal_stability"].tolist())[0],
                "random_within_compactness_mean": _mean_std(ds["random_within_compactness"].tolist())[0],
                "random_between_separation_mean": _mean_std(ds["random_between_separation"].tolist())[0],
                "random_margin_mean": _mean_std(ds["random_margin"].tolist())[0],
                "random_temporal_stability_mean": _mean_std(ds["random_temporal_stability"].tolist())[0],
                "within_compactness_delta_mean": _mean_std(
                    (ds["random_within_compactness"] - ds["prototype_within_compactness"]).tolist()
                )[0],
                "between_separation_delta_mean": _mean_std(
                    (ds["prototype_between_separation"] - ds["random_between_separation"]).tolist()
                )[0],
                "margin_delta_mean": _mean_std((ds["prototype_margin"] - ds["random_margin"]).tolist())[0],
                "temporal_stability_delta_mean": _mean_std(
                    (ds["prototype_temporal_stability"] - ds["random_temporal_stability"]).tolist()
                )[0],
                "low_coverage_class_count_mean": _mean_std(ds["low_coverage_class_count"].tolist())[0],
            }
        )
    dataset_summary_df = pd.DataFrame(dataset_summary_rows)

    config_csv = os.path.join(args.out_root, "scp_branch_v0_config_table.csv")
    per_seed_csv = os.path.join(args.out_root, "scp_branch_v0_per_seed.csv")
    dataset_summary_csv = os.path.join(args.out_root, "scp_branch_v0_dataset_summary.csv")
    memory_csv = os.path.join(args.out_root, "scp_branch_v0_memory_summary.csv")
    prototype_csv = os.path.join(args.out_root, "scp_branch_v0_prototype_summary.csv")
    random_csv = os.path.join(args.out_root, "scp_branch_v0_random_control_summary.csv")
    structure_csv = os.path.join(args.out_root, "scp_branch_v0_structure_diagnostics.csv")
    summary_md = os.path.join(args.out_root, "scp_branch_v0_conclusion.md")

    config_df.to_csv(config_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    dataset_summary_df.to_csv(dataset_summary_csv, index=False)
    memory_df.to_csv(memory_csv, index=False)
    prototype_df.to_csv(prototype_csv, index=False)
    random_df.to_csv(random_csv, index=False)
    structure_df.to_csv(structure_csv, index=False)

    lines: List[str] = [
        "# SCP-Branch v0 Conclusion",
        "",
        "更新时间：2026-03-30",
        "",
        "本轮只验证 prototype-memory 对象本身是否能形成比随机窗口集合更有结构的局部代表态。",
        "硬约束：冻结 dense backbone，不做 replay / curriculum / PIA-guided local geometry。",
        "判读口径：prototype 当前不解释为真实稳态中心，而解释为当前训练分布内可复现的 local representative states。",
        "",
    ]
    for dataset in datasets:
        ds = per_seed_df[per_seed_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(f"- `static_linear`: {_format_mean_std(ds['static_linear_test_macro_f1'].tolist())}")
        gru_vals = [float(v) for v in ds["dense_dynamic_gru_test_macro_f1"].dropna().tolist()]
        if gru_vals:
            lines.append(f"- `dense_dynamic_gru`: {_format_mean_std(gru_vals)}")
        else:
            lines.append("- `dense_dynamic_gru`: `skipped`")
        lines.append(f"- `dense_dynamic_minirocket`: {_format_mean_std(ds['dense_dynamic_minirocket_test_macro_f1'].tolist())}")
        raw_vals = [float(v) for v in ds["raw_minirocket_test_macro_f1"].dropna().tolist()]
        if raw_vals:
            lines.append(f"- `raw_minirocket`: {_format_mean_std(raw_vals)}")
        proto_rows = structure_df[(structure_df["dataset"] == dataset) & (structure_df["mode"] == "prototype_memory")]
        rand_rows = structure_df[(structure_df["dataset"] == dataset) & (structure_df["mode"] == "random_memory_control")]
        if not proto_rows.empty and not rand_rows.empty:
            p = proto_rows.iloc[0]
            r = rand_rows.iloc[0]
            lines.append("")
            lines.append(f"- `prototype within_compactness`: `{float(p['within_prototype_compactness']):.4f}`")
            lines.append(f"- `random within_compactness`: `{float(r['within_prototype_compactness']):.4f}`")
            lines.append(f"- `prototype between_separation`: `{float(p['between_prototype_separation']):.4f}`")
            lines.append(f"- `random between_separation`: `{float(r['between_prototype_separation']):.4f}`")
            lines.append(f"- `prototype nearest_margin`: `{float(p['nearest_prototype_margin']):.4f}`")
            lines.append(f"- `random nearest_margin`: `{float(r['nearest_prototype_margin']):.4f}`")
            lines.append(f"- `prototype temporal_stability`: `{float(p['temporal_assignment_stability']):.4f}`")
            lines.append(f"- `random temporal_stability`: `{float(r['temporal_assignment_stability']):.4f}`")
        mem_rows = memory_df[memory_df["dataset"] == dataset]
        if not mem_rows.empty:
            lines.append("")
            lines.append(
                f"- `low_coverage_class_count`: `{int(mem_rows['low_coverage_flag'].sum())}` / `{int(mem_rows.shape[0])}`"
            )
            lines.append(f"- `safe_coverage_mean`: `{float(mem_rows['coverage_ratio_class'].mean()):.4f}`")
        lines.append("")

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(f"[scp-branch-v0] wrote {config_csv}")
    print(f"[scp-branch-v0] wrote {per_seed_csv}")
    print(f"[scp-branch-v0] wrote {dataset_summary_csv}")
    print(f"[scp-branch-v0] wrote {memory_csv}")
    print(f"[scp-branch-v0] wrote {prototype_csv}")
    print(f"[scp-branch-v0] wrote {random_csv}")
    print(f"[scp-branch-v0] wrote {structure_csv}")
    print(f"[scp-branch-v0] wrote {summary_md}")


if __name__ == "__main__":
    main()
