#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from route_b_unified.trajectory_classifier import TrajectoryModelConfig  # noqa: E402
from route_b_unified.trajectory_evaluator import TrajectoryEvalConfig, evaluate_trajectory_classifier  # noqa: E402
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


def _format_mean_std(values: List[float]) -> str:
    arr = np.asarray(list(values), dtype=np.float64)
    return f"{float(np.mean(arr)):.4f} +/- {float(np.std(arr)):.4f}" if arr.size else "0.0000 +/- 0.0000"


def _mean_std(values: List[float]) -> tuple[float, float]:
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


def _compute_dynamic_diag(state: TrajectoryRepresentationState) -> Dict[str, float]:
    seqs = [np.asarray(v, dtype=np.float32) for v in state.train.z_seq_list]
    labels = np.asarray(state.train.y, dtype=np.int64)

    def _pairwise_mean_distance(x: np.ndarray) -> float:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] <= 1:
            return 0.0
        diffs = arr[:, None, :] - arr[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        tri = dists[np.triu_indices(arr.shape[0], k=1)]
        return float(np.mean(tri)) if tri.size else 0.0

    traj_lens = [int(v.shape[0]) for v in seqs]
    step_mags: List[float] = []
    curvature_vals: List[float] = []
    seq_means: List[np.ndarray] = []
    delta_by_class: Dict[int, List[np.ndarray]] = {}

    for seq, y in zip(seqs, labels.tolist()):
        seq_means.append(np.mean(seq, axis=0).astype(np.float32))
        if int(seq.shape[0]) >= 2:
            delta = np.diff(seq, axis=0)
            step_mags.append(float(np.mean(np.linalg.norm(delta, axis=1))))
            delta_by_class.setdefault(int(y), []).append(np.mean(delta, axis=0).astype(np.float32))
        if int(seq.shape[0]) >= 3:
            curv = seq[2:] - 2.0 * seq[1:-1] + seq[:-2]
            curvature_vals.append(float(np.mean(np.linalg.norm(curv, axis=1))))

    seq_means_arr = np.stack(seq_means, axis=0).astype(np.float32) if seq_means else np.zeros((0, state.z_dim), dtype=np.float32)
    class_dispersion_rows: List[float] = []
    for cls in sorted(set(labels.tolist())):
        mask = labels == int(cls)
        if int(np.sum(mask)) <= 1:
            continue
        class_dispersion_rows.append(_pairwise_mean_distance(seq_means_arr[mask]))

    mean_deltas: List[np.ndarray] = []
    for cls in sorted(delta_by_class):
        cls_arr = np.stack(delta_by_class[int(cls)], axis=0).astype(np.float32)
        mean_deltas.append(np.mean(cls_arr, axis=0).astype(np.float32))
    transition_sep = _pairwise_mean_distance(np.stack(mean_deltas, axis=0).astype(np.float32)) if len(mean_deltas) >= 2 else 0.0

    return {
        "trajectory_len_mean": float(np.mean(traj_lens)) if traj_lens else 0.0,
        "trajectory_len_min": float(np.min(traj_lens)) if traj_lens else 0.0,
        "trajectory_len_max": float(np.max(traj_lens)) if traj_lens else 0.0,
        "step_change_mean": float(np.mean(step_mags)) if step_mags else 0.0,
        "local_curvature_proxy": float(np.mean(curvature_vals)) if curvature_vals else 0.0,
        "classwise_dispersion": float(np.mean(class_dispersion_rows)) if class_dispersion_rows else 0.0,
        "transition_separation_proxy": float(transition_sep),
    }


def _build_state(args: argparse.Namespace, dataset: str, seed: int, *, force_hop_len: int | None) -> TrajectoryRepresentationState:
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
            force_hop_len=None if force_hop_len is None else int(force_hop_len),
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Dense z_seq terminal probe: static_linear / dense_dynamic_gru / dense_dynamic_minirocket / raw_minirocket.")
    p.add_argument("--datasets", type=str, default="natops,selfregulationscp1")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/route_b_dense_trajectory_probe_20260330_formal")
    p.add_argument("--epochs", type=int, default=30)
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
    p.add_argument("--minirocket-n-kernels", type=int, default=10000)
    p.add_argument("--minirocket-n-jobs", type=int, default=1)
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

    config_rows: List[Dict[str, object]] = []
    per_seed_rows: List[Dict[str, object]] = []
    padding_rows: List[Dict[str, object]] = []
    stride_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        for seed in seeds:
            sparse_state = _build_state(args, dataset, seed, force_hop_len=None)
            dense_state = _build_state(args, dataset, seed, force_hop_len=int(args.force_hop_len))
            seed_dir = os.path.join(args.out_root, f"{dataset}_seed{seed}")
            _ensure_dir(seed_dir)
            _write_json(
                os.path.join(seed_dir, "representation_compare_meta.json"),
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "sparse_window_len": int(sparse_state.window_len),
                    "sparse_hop_len": int(sparse_state.hop_len),
                    "dense_window_len": int(dense_state.window_len),
                    "dense_hop_len": int(dense_state.hop_len),
                    "force_hop_len": int(args.force_hop_len),
                },
            )

            model_cfg = TrajectoryModelConfig(
                z_dim=int(sparse_state.z_dim),
                num_classes=int(sparse_state.num_classes),
                gru_hidden_dim=int(args.gru_hidden_dim),
                dropout=float(args.dropout),
            )
            sparse_eval_cfg = TrajectoryEvalConfig(
                variant="dynamic_gru",
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                patience=int(args.patience),
                device=str(args.device),
            )
            static_eval_cfg = TrajectoryEvalConfig(
                variant="static_linear",
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                patience=int(args.patience),
                device=str(args.device),
            )
            sparse_gru = evaluate_trajectory_classifier(sparse_state, seed=int(seed), model_cfg=model_cfg, eval_cfg=sparse_eval_cfg)
            static_result = evaluate_trajectory_classifier(sparse_state, seed=int(seed), model_cfg=model_cfg, eval_cfg=static_eval_cfg)
            dense_gru = evaluate_trajectory_classifier(dense_state, seed=int(seed), model_cfg=model_cfg, eval_cfg=sparse_eval_cfg)
            mini_cfg = TrajectoryMiniRocketEvalConfig(
                n_kernels=int(args.minirocket_n_kernels),
                n_jobs=int(args.minirocket_n_jobs),
                padding_mode="edge",
                target_len_mode="train_max_len",
            )
            sparse_mini = evaluate_dynamic_minirocket_classifier(sparse_state, seed=int(seed), eval_cfg=mini_cfg)
            dense_mini = evaluate_dynamic_minirocket_classifier(dense_state, seed=int(seed), eval_cfg=mini_cfg)

            _write_json(
                os.path.join(seed_dir, "dense_probe_results.json"),
                {
                    "static_linear": {
                        "test_metrics": static_result.test_metrics,
                        "best_epoch": int(static_result.best_epoch),
                    },
                    "sparse_dynamic_gru": {
                        "test_metrics": sparse_gru.test_metrics,
                        "best_epoch": int(sparse_gru.best_epoch),
                    },
                    "dense_dynamic_gru": {
                        "test_metrics": dense_gru.test_metrics,
                        "best_epoch": int(dense_gru.best_epoch),
                    },
                    "sparse_dynamic_minirocket": {
                        "test_metrics": sparse_mini.test_metrics,
                        "meta": sparse_mini.meta,
                    },
                    "dense_dynamic_minirocket": {
                        "test_metrics": dense_mini.test_metrics,
                        "meta": dense_mini.meta,
                    },
                },
            )

            raw_ref = float(raw_ref_map.get((str(dataset), int(seed)), np.nan))
            per_seed_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "static_linear_test_macro_f1": float(static_result.test_metrics["macro_f1"]),
                    "sparse_dynamic_gru_test_macro_f1": float(sparse_gru.test_metrics["macro_f1"]),
                    "dense_dynamic_gru_test_macro_f1": float(dense_gru.test_metrics["macro_f1"]),
                    "sparse_dynamic_minirocket_test_macro_f1": float(sparse_mini.test_metrics["macro_f1"]),
                    "dense_dynamic_minirocket_test_macro_f1": float(dense_mini.test_metrics["macro_f1"]),
                    "raw_minirocket_test_macro_f1": raw_ref,
                }
            )

            sparse_diag = _compute_dynamic_diag(sparse_state)
            dense_diag = _compute_dynamic_diag(dense_state)
            stride_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "sparse_hop_len": int(sparse_state.hop_len),
                    "dense_hop_len": int(dense_state.hop_len),
                    "sparse_len_mean": float(sparse_diag["trajectory_len_mean"]),
                    "dense_len_mean": float(dense_diag["trajectory_len_mean"]),
                    "sparse_len_min": float(sparse_diag["trajectory_len_min"]),
                    "dense_len_min": float(dense_diag["trajectory_len_min"]),
                    "sparse_len_max": float(sparse_diag["trajectory_len_max"]),
                    "dense_len_max": float(dense_diag["trajectory_len_max"]),
                    "dense_gru_minus_sparse_gru": float(dense_gru.test_metrics["macro_f1"] - sparse_gru.test_metrics["macro_f1"]),
                    "dense_minirocket_minus_sparse_minirocket": float(dense_mini.test_metrics["macro_f1"] - sparse_mini.test_metrics["macro_f1"]),
                    "dense_minirocket_minus_raw_minirocket": float(dense_mini.test_metrics["macro_f1"] - raw_ref) if not np.isnan(raw_ref) else np.nan,
                }
            )
            for split_name in ["train_padding", "val_padding", "test_padding"]:
                meta = dict(dense_mini.meta.get(split_name, {}))
                padding_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "split_name": str(split_name).replace("_padding", ""),
                        "train_max_len": int(dense_mini.meta["train_max_len"]),
                        "model_input_len": int(dense_mini.meta["model_input_len"]),
                        "trajectory_len_mean": float(meta.get("orig_len_mean", 0.0)),
                        "trajectory_len_min": float(meta.get("orig_len_min", 0.0)),
                        "trajectory_len_max": float(meta.get("orig_len_max", 0.0)),
                        "padding_ratio_mean": float(meta.get("padding_ratio_mean", 0.0)),
                        "truncate_ratio_mean": float(meta.get("truncate_ratio_mean", 0.0)),
                        "n_padded": int(meta.get("n_padded", 0)),
                        "n_truncated": int(meta.get("n_truncated", 0)),
                    }
                )
            config_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "window_len": int(dense_state.window_len),
                    "sparse_hop_len": int(sparse_state.hop_len),
                    "dense_hop_len": int(dense_state.hop_len),
                    "force_hop_len": int(args.force_hop_len),
                    "z_dim": int(dense_state.z_dim),
                    "num_classes": int(dense_state.num_classes),
                    "gru_hidden_dim": int(args.gru_hidden_dim),
                    "dropout": float(args.dropout),
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "patience": int(args.patience),
                    "minirocket_n_kernels": int(args.minirocket_n_kernels),
                    "minirocket_n_jobs": int(args.minirocket_n_jobs),
                }
            )

    per_seed_df = pd.DataFrame(per_seed_rows)
    stride_df = pd.DataFrame(stride_rows)
    padding_df = pd.DataFrame(padding_rows)
    config_df = pd.DataFrame(config_rows)

    summary_rows: List[Dict[str, object]] = []
    for dataset in datasets:
        ds = per_seed_df[per_seed_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        row: Dict[str, object] = {"dataset": str(dataset)}
        for col in [
            "static_linear_test_macro_f1",
            "dense_dynamic_gru_test_macro_f1",
            "dense_dynamic_minirocket_test_macro_f1",
            "raw_minirocket_test_macro_f1",
        ]:
            mean, std = _mean_std(ds[col].dropna().tolist())
            prefix = col.replace("_test_macro_f1", "")
            row[f"{prefix}_mean"] = mean
            row[f"{prefix}_std"] = std
            row[f"{prefix}"] = _format_mean_std(ds[col].dropna().tolist())
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    config_csv = os.path.join(args.out_root, "dense_trajectory_probe_config_table.csv")
    per_seed_csv = os.path.join(args.out_root, "dense_trajectory_probe_per_seed.csv")
    summary_csv = os.path.join(args.out_root, "dense_trajectory_probe_summary.csv")
    padding_csv = os.path.join(args.out_root, "dense_trajectory_padding_summary.csv")
    stride_csv = os.path.join(args.out_root, "dense_trajectory_stride_impact_diagnostics.csv")
    summary_md = os.path.join(args.out_root, "dense_trajectory_conclusion.md")

    config_df.to_csv(config_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    padding_df.to_csv(padding_csv, index=False)
    stride_df.to_csv(stride_csv, index=False)

    lines: List[str] = [
        "# Dense Trajectory Probe Conclusion",
        "",
        "更新时间：2026-03-30",
        "",
        "本轮主比较对象：`static_linear / dense_dynamic_gru / dense_dynamic_minirocket`。",
        "参考绝对强基线：`raw + MiniROCKET`。",
        "硬约束：冻结表示与增强主链，只将 `hop_len` 强制压到 `1`，padding 固定为 `max(train_max_len, 9)` 的 edge pad。",
        "",
    ]
    for dataset in datasets:
        row_df = summary_df[summary_df["dataset"] == dataset]
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(f"- `static_linear`: {row['static_linear']}")
        lines.append(f"- `dense_dynamic_gru`: {row['dense_dynamic_gru']}")
        lines.append(f"- `dense_dynamic_minirocket`: {row['dense_dynamic_minirocket']}")
        lines.append(f"- `raw_minirocket`: {row['raw_minirocket']}")
        stride_row = stride_df[stride_df["dataset"] == dataset]
        if not stride_row.empty:
            srow = stride_row.iloc[0]
            lines.append("")
            lines.append(f"- `sparse_len_mean -> dense_len_mean`: `{float(srow['sparse_len_mean']):.1f} -> {float(srow['dense_len_mean']):.1f}`")
            lines.append(f"- `dense_gru_minus_sparse_gru`: `{float(srow['dense_gru_minus_sparse_gru']):+.4f}`")
            lines.append(f"- `dense_minirocket_minus_sparse_minirocket`: `{float(srow['dense_minirocket_minus_sparse_minirocket']):+.4f}`")
            if not np.isnan(float(srow['dense_minirocket_minus_raw_minirocket'])):
                lines.append(f"- `dense_minirocket_minus_raw_minirocket`: `{float(srow['dense_minirocket_minus_raw_minirocket']):+.4f}`")
        lines.append("")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(f"[dense-trajectory-probe] wrote {config_csv}")
    print(f"[dense-trajectory-probe] wrote {per_seed_csv}")
    print(f"[dense-trajectory-probe] wrote {summary_csv}")
    print(f"[dense-trajectory-probe] wrote {padding_csv}")
    print(f"[dense-trajectory-probe] wrote {stride_csv}")
    print(f"[dense-trajectory-probe] wrote {summary_md}")


if __name__ == "__main__":
    main()

