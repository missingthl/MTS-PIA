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
    if score_col is None or "dataset" not in df.columns:
        return {}
    if "seed" not in df.columns:
        return {}
    out: Dict[tuple[str, int], float] = {}
    for _, row in df.iterrows():
        out[(str(row["dataset"]).strip().lower(), int(row["seed"]))] = float(row[score_col])
    return out


def _compute_diagnostics(state: TrajectoryRepresentationState) -> Dict[str, Dict[str, float]]:
    train_split = state.train
    seqs = [np.asarray(v, dtype=np.float32) for v in train_split.z_seq_list]
    labels = np.asarray(train_split.y, dtype=np.int64)
    static_x = np.asarray(train_split.X_static, dtype=np.float32)

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
        cls_mask = labels == int(cls)
        if int(np.sum(cls_mask)) <= 1:
            continue
        class_dispersion_rows.append(_pairwise_mean_distance(seq_means_arr[cls_mask]))
    classwise_dispersion = float(np.mean(class_dispersion_rows)) if class_dispersion_rows else 0.0

    mean_deltas: List[np.ndarray] = []
    for cls in sorted(delta_by_class):
        cls_arr = np.stack(delta_by_class[int(cls)], axis=0).astype(np.float32)
        mean_deltas.append(np.mean(cls_arr, axis=0).astype(np.float32))
    transition_sep = _pairwise_mean_distance(np.stack(mean_deltas, axis=0).astype(np.float32)) if len(mean_deltas) >= 2 else 0.0

    dynamic_diag = {
        "trajectory_len_mean": float(np.mean(traj_lens)) if traj_lens else 0.0,
        "step_change_mean": float(np.mean(step_mags)) if step_mags else 0.0,
        "local_curvature_proxy": float(np.mean(curvature_vals)) if curvature_vals else 0.0,
        "classwise_dispersion": float(classwise_dispersion),
        "transition_separation_proxy": float(transition_sep),
    }
    static_dispersion_rows: List[float] = []
    for cls in sorted(set(labels.tolist())):
        cls_mask = labels == int(cls)
        if int(np.sum(cls_mask)) <= 1:
            continue
        static_dispersion_rows.append(_pairwise_mean_distance(static_x[cls_mask]))
    static_diag = {
        "trajectory_len_mean": 1.0 if static_x.shape[0] > 0 else 0.0,
        "step_change_mean": 0.0,
        "local_curvature_proxy": 0.0,
        "classwise_dispersion": float(np.mean(static_dispersion_rows)) if static_dispersion_rows else 0.0,
        "transition_separation_proxy": 0.0,
    }
    return {
        "static_linear": static_diag,
        "dynamic_gru": dynamic_diag,
        "dynamic_minirocket": dynamic_diag,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Dynamic terminal unification probe: static_linear / dynamic_gru / dynamic_minirocket / raw_minirocket.")
    p.add_argument("--datasets", type=str, default="natops,selfregulationscp1")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/route_b_dynamic_terminal_probe_20260330_formal")
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
    diagnostics_rows: List[Dict[str, object]] = []

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
            seed_dir = os.path.join(args.out_root, f"{dataset}_seed{seed}")
            _ensure_dir(seed_dir)
            _write_json(
                os.path.join(seed_dir, "trajectory_split_meta.json"),
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "split_meta": dict(state.split_meta),
                    "meta": dict(state.meta),
                    "window_len": int(state.window_len),
                    "hop_len": int(state.hop_len),
                },
            )

            diag_map = _compute_diagnostics(state)
            for model_type, diag in diag_map.items():
                diagnostics_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "model_type": str(model_type),
                        "trajectory_len_mean": float(diag["trajectory_len_mean"]),
                        "step_change_mean": float(diag["step_change_mean"]),
                        "local_curvature_proxy": float(diag["local_curvature_proxy"]),
                        "classwise_dispersion": float(diag["classwise_dispersion"]),
                        "transition_separation_proxy": float(diag["transition_separation_proxy"]),
                    }
                )

            model_cfg = TrajectoryModelConfig(
                z_dim=int(state.z_dim),
                num_classes=int(state.num_classes),
                gru_hidden_dim=int(args.gru_hidden_dim),
                dropout=float(args.dropout),
            )

            for variant in ["static_linear", "dynamic_gru"]:
                eval_cfg = TrajectoryEvalConfig(
                    variant=str(variant),
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    patience=int(args.patience),
                    device=str(args.device),
                )
                result = evaluate_trajectory_classifier(state, seed=int(seed), model_cfg=model_cfg, eval_cfg=eval_cfg)
                _write_json(os.path.join(seed_dir, f"{variant}_result.json"), {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "variant": str(variant),
                    "train_metrics": result.train_metrics,
                    "val_metrics": result.val_metrics,
                    "test_metrics": result.test_metrics,
                    "best_epoch": int(result.best_epoch),
                    "history_rows": result.history_rows,
                    "meta": result.meta,
                })
                per_seed_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "model_type": str(variant),
                        "window_len": int(state.window_len),
                        "hop_len": int(state.hop_len),
                        "test_acc": float(result.test_metrics["acc"]),
                        "test_macro_f1": float(result.test_metrics["macro_f1"]),
                        "val_macro_f1": float(result.val_metrics["macro_f1"]),
                        "best_epoch": int(result.best_epoch),
                        "raw_minirocket_test_macro_f1": float(raw_ref_map.get((str(dataset), int(seed)), np.nan)),
                    }
                )
                config_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "variant": str(variant),
                        "window_len": int(state.window_len),
                        "hop_len": int(state.hop_len),
                        "z_dim": int(state.z_dim),
                        "num_classes": int(state.num_classes),
                        "gru_hidden_dim": int(args.gru_hidden_dim),
                        "dropout": float(args.dropout),
                        "epochs": int(args.epochs),
                        "batch_size": int(args.batch_size),
                        "lr": float(args.lr),
                        "weight_decay": float(args.weight_decay),
                        "patience": int(args.patience),
                    }
                )

            mini_cfg = TrajectoryMiniRocketEvalConfig(
                n_kernels=int(args.minirocket_n_kernels),
                n_jobs=int(args.minirocket_n_jobs),
                padding_mode="edge",
                target_len_mode="train_max_len",
            )
            mini_result = evaluate_dynamic_minirocket_classifier(state, seed=int(seed), eval_cfg=mini_cfg)
            _write_json(os.path.join(seed_dir, "dynamic_minirocket_result.json"), {
                "dataset": str(dataset),
                "seed": int(seed),
                "variant": "dynamic_minirocket",
                "train_metrics": mini_result.train_metrics,
                "val_metrics": mini_result.val_metrics,
                "test_metrics": mini_result.test_metrics,
                "meta": mini_result.meta,
            })
            per_seed_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "model_type": "dynamic_minirocket",
                    "window_len": int(state.window_len),
                    "hop_len": int(state.hop_len),
                    "test_acc": float(mini_result.test_metrics["acc"]),
                    "test_macro_f1": float(mini_result.test_metrics["macro_f1"]),
                    "val_macro_f1": float(mini_result.val_metrics["macro_f1"]),
                    "best_epoch": 0,
                    "raw_minirocket_test_macro_f1": float(raw_ref_map.get((str(dataset), int(seed)), np.nan)),
                }
            )
            config_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "variant": "dynamic_minirocket",
                    "window_len": int(state.window_len),
                    "hop_len": int(state.hop_len),
                    "z_dim": int(state.z_dim),
                    "num_classes": int(state.num_classes),
                    "padding_mode": "edge",
                    "target_len_mode": "train_max_len",
                    "minirocket_n_kernels": int(args.minirocket_n_kernels),
                    "minirocket_n_jobs": int(args.minirocket_n_jobs),
                }
            )
            for split_name in ["train_padding", "val_padding", "test_padding"]:
                meta = dict(mini_result.meta.get(split_name, {}))
                padding_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "split_name": str(split_name).replace("_padding", ""),
                        "train_max_len": int(mini_result.meta["train_max_len"]),
                        "model_input_len": int(mini_result.meta["model_input_len"]),
                        "trajectory_len_mean": float(meta.get("orig_len_mean", 0.0)),
                        "trajectory_len_min": float(meta.get("orig_len_min", 0.0)),
                        "trajectory_len_max": float(meta.get("orig_len_max", 0.0)),
                        "padding_ratio_mean": float(meta.get("padding_ratio_mean", 0.0)),
                        "truncate_ratio_mean": float(meta.get("truncate_ratio_mean", 0.0)),
                        "n_padded": int(meta.get("n_padded", 0)),
                        "n_truncated": int(meta.get("n_truncated", 0)),
                    }
                )

    config_df = pd.DataFrame(config_rows)
    per_seed_df = pd.DataFrame(per_seed_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows)
    padding_df = pd.DataFrame(padding_rows)

    summary_rows: List[Dict[str, object]] = []
    for dataset in datasets:
        ds = per_seed_df[per_seed_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        row: Dict[str, object] = {"dataset": str(dataset)}
        internal_means: Dict[str, float] = {}
        for variant, col_prefix in [
            ("static_linear", "static_manifold"),
            ("dynamic_gru", "dynamic_gru"),
            ("dynamic_minirocket", "dynamic_minirocket"),
        ]:
            sub = ds[ds["model_type"] == variant]
            mean, std = _mean_std(sub["test_macro_f1"].tolist())
            row[f"{col_prefix}_macro_f1_mean"] = mean
            row[f"{col_prefix}_macro_f1_std"] = std
            row[f"{col_prefix}_macro_f1"] = _format_mean_std(sub["test_macro_f1"].tolist())
            internal_means[variant] = mean
        raw_sub = ds.dropna(subset=["raw_minirocket_test_macro_f1"])
        raw_mean, raw_std = _mean_std(raw_sub["raw_minirocket_test_macro_f1"].tolist())
        row["raw_minirocket_macro_f1_mean"] = raw_mean
        row["raw_minirocket_macro_f1_std"] = raw_std
        row["raw_minirocket_macro_f1"] = _format_mean_std(raw_sub["raw_minirocket_test_macro_f1"].tolist())
        row["best_internal_model"] = max(internal_means.items(), key=lambda kv: kv[1])[0]
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    config_csv = os.path.join(args.out_root, "dynamic_terminal_probe_config_table.csv")
    per_seed_csv = os.path.join(args.out_root, "dynamic_terminal_probe_per_seed.csv")
    summary_csv = os.path.join(args.out_root, "dynamic_terminal_probe_dataset_summary.csv")
    padding_csv = os.path.join(args.out_root, "dynamic_terminal_probe_padding_summary.csv")
    diagnostics_csv = os.path.join(args.out_root, "dynamic_terminal_probe_diagnostics_summary.csv")
    config_df.to_csv(config_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    padding_df.to_csv(padding_csv, index=False)
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    lines: List[str] = [
        "# Dynamic Terminal Probe Conclusion",
        "",
        "更新时间：2026-03-30",
        "",
        "本轮主比较对象：`static_linear / dynamic_gru / dynamic_minirocket`。",
        "参考外部强基线：`raw + MiniROCKET`。",
        "硬约束：`dynamic_minirocket` 只吃 `z_seq`，`static` 不做伪序列 MiniROCKET，padding 固定为 `max(train-max-len, 9)` 的 edge pad。",
        "",
    ]
    for dataset in datasets:
        ds = summary_df[summary_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        row = ds.iloc[0]
        static_mean = float(row["static_manifold_macro_f1_mean"])
        gru_mean = float(row["dynamic_gru_macro_f1_mean"])
        mini_mean = float(row["dynamic_minirocket_macro_f1_mean"])
        raw_mean = float(row["raw_minirocket_macro_f1_mean"])
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(f"- `static_linear`: {row['static_manifold_macro_f1']}")
        lines.append(f"- `dynamic_gru`: {row['dynamic_gru_macro_f1']}")
        lines.append(f"- `dynamic_minirocket`: {row['dynamic_minirocket_macro_f1']}")
        lines.append(f"- `raw + MiniROCKET` (reference): {row['raw_minirocket_macro_f1']}")
        lines.append("")
        lines.append(f"- 当前最佳内部终端：`{row['best_internal_model']}`")
        lines.append(f"- `dynamic_minirocket > dynamic_gru`：`{'yes' if mini_mean > gru_mean + 1e-9 else 'not_yet'}`")
        lines.append(f"- `dynamic_minirocket > static_linear`：`{'yes' if mini_mean > static_mean + 1e-9 else 'not_yet'}`")
        if not np.isnan(raw_mean):
            lines.append(f"- `dynamic_minirocket vs raw + MiniROCKET`：`{float(mini_mean - raw_mean):+.4f}`")
            lines.append(f"- `dynamic_gru vs raw + MiniROCKET`：`{float(gru_mean - raw_mean):+.4f}`")
        lines.append("")
    conclusion_path = os.path.join(args.out_root, "dynamic_terminal_probe_conclusion.md")
    with open(conclusion_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(f"[dynamic-terminal-probe] wrote {config_csv}")
    print(f"[dynamic-terminal-probe] wrote {per_seed_csv}")
    print(f"[dynamic-terminal-probe] wrote {summary_csv}")
    print(f"[dynamic-terminal-probe] wrote {padding_csv}")
    print(f"[dynamic-terminal-probe] wrote {diagnostics_csv}")
    print(f"[dynamic-terminal-probe] wrote {conclusion_path}")


if __name__ == "__main__":
    main()
