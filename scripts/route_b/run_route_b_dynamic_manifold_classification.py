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
from route_b_unified.trajectory_evaluator import (  # noqa: E402
    TrajectoryEvalConfig,
    evaluate_trajectory_classifier,
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


def _pairwise_mean_distance(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= 1:
        return 0.0
    diffs = arr[:, None, :] - arr[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    tri = dists[np.triu_indices(arr.shape[0], k=1)]
    return float(np.mean(tri)) if tri.size else 0.0


def _compute_diagnostics(state: TrajectoryRepresentationState) -> Dict[str, Dict[str, float]]:
    train_split = state.train
    seqs = [np.asarray(v, dtype=np.float32) for v in train_split.z_seq_list]
    labels = np.asarray(train_split.y, dtype=np.int64)
    static_x = np.asarray(train_split.X_static, dtype=np.float32)

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
    if len(mean_deltas) >= 2:
        delta_stack = np.stack(mean_deltas, axis=0).astype(np.float32)
        transition_sep = _pairwise_mean_distance(delta_stack)
    else:
        transition_sep = 0.0

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
        "dynamic_meanpool": dynamic_diag,
        "dynamic_gru": dynamic_diag,
    }


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


def main() -> None:
    p = argparse.ArgumentParser(description="Dynamic manifold T0 classification probe: static point vs trajectory sequence.")
    p.add_argument("--datasets", type=str, default="natops,selfregulationscp1")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/route_b_dynamic_manifold_classification_20260329_formal")
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
                        "notes": "dynamic proxies use minimal reproducible definitions",
                    }
                )

            model_cfg = TrajectoryModelConfig(
                z_dim=int(state.z_dim),
                num_classes=int(state.num_classes),
                gru_hidden_dim=int(args.gru_hidden_dim),
                dropout=float(args.dropout),
            )

            for variant in ["static_linear", "dynamic_meanpool", "dynamic_gru"]:
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
                _write_json(
                    os.path.join(seed_dir, f"{variant}_result.json"),
                    {
                        "dataset": result.dataset,
                        "seed": result.seed,
                        "variant": result.variant,
                        "train_metrics": result.train_metrics,
                        "val_metrics": result.val_metrics,
                        "test_metrics": result.test_metrics,
                        "best_epoch": result.best_epoch,
                        "meta": result.meta,
                        "history_rows": result.history_rows,
                    },
                )
                config_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "model_type": str(variant),
                        "window_len": int(state.window_len),
                        "hop_len": int(state.hop_len),
                        "channels": int(state.channels),
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

    config_df = pd.DataFrame(config_rows)
    per_seed_df = pd.DataFrame(per_seed_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows)

    summary_rows: List[Dict[str, object]] = []
    for dataset in datasets:
        ds = per_seed_df[per_seed_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        row: Dict[str, object] = {"dataset": str(dataset)}
        best_model = None
        best_score = float("-inf")
        for key, short in [
            ("static_linear", "static_manifold"),
            ("dynamic_meanpool", "dynamic_meanpool"),
            ("dynamic_gru", "dynamic_gru"),
        ]:
            sub = ds[ds["model_type"] == key]
            mean, std = _mean_std(sub["test_macro_f1"].tolist())
            row[f"{short}_macro_f1_mean"] = mean
            row[f"{short}_macro_f1_std"] = std
            row[f"{short}_macro_f1"] = _format_mean_std(sub["test_macro_f1"].tolist())
            if mean > best_score:
                best_score = mean
                best_model = key
        raw_sub = ds.dropna(subset=["raw_minirocket_test_macro_f1"])
        row["raw_minirocket_macro_f1_mean"], row["raw_minirocket_macro_f1_std"] = _mean_std(
            raw_sub["raw_minirocket_test_macro_f1"].tolist()
        )
        row["raw_minirocket_macro_f1"] = _format_mean_std(raw_sub["raw_minirocket_test_macro_f1"].tolist())
        row["best_model"] = str(best_model) if best_model is not None else ""
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    config_csv = os.path.join(args.out_root, "dynamic_manifold_config_table.csv")
    per_seed_csv = os.path.join(args.out_root, "dynamic_manifold_per_seed.csv")
    summary_csv = os.path.join(args.out_root, "dynamic_manifold_dataset_summary.csv")
    diagnostics_csv = os.path.join(args.out_root, "dynamic_manifold_diagnostics_summary.csv")
    config_df.to_csv(config_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    lines: List[str] = [
        "# Dynamic Manifold Classification Conclusion",
        "",
        "更新时间：2026-03-29",
        "",
        "本轮主比较对象：`static_linear / dynamic_meanpool / dynamic_gru`。",
        "参考外部强基线：`raw + MiniROCKET`。",
        "",
    ]
    for dataset in datasets:
        ds = summary_df[summary_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        row = ds.iloc[0]
        static_mean = float(row["static_manifold_macro_f1_mean"])
        meanpool_mean = float(row["dynamic_meanpool_macro_f1_mean"])
        gru_mean = float(row["dynamic_gru_macro_f1_mean"])
        raw_mean = float(row["raw_minirocket_macro_f1_mean"])

        dynamic_vs_static = "yes" if max(meanpool_mean, gru_mean) > static_mean + 1e-9 else "not_yet"
        gru_vs_meanpool = "yes" if gru_mean > meanpool_mean + 1e-9 else "not_yet"
        natops_guard = "yes" if str(dataset) != "natops" or max(meanpool_mean, gru_mean) >= static_mean - 0.01 else "not_yet"

        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(f"- `static_linear`: {row['static_manifold_macro_f1']}")
        lines.append(f"- `dynamic_meanpool`: {row['dynamic_meanpool_macro_f1']}")
        lines.append(f"- `dynamic_gru`: {row['dynamic_gru_macro_f1']}")
        lines.append(f"- `raw + MiniROCKET` (reference): {row['raw_minirocket_macro_f1']}")
        lines.append("")
        lines.append(f"- 当前最佳动态/静态模型：`{row['best_model']}`。")
        lines.append(f"- `dynamic > static`：`{dynamic_vs_static}`")
        lines.append(f"- `GRU > mean-pool`：`{gru_vs_meanpool}`")
        if str(dataset) == "selfregulationscp1":
            lines.append(
                f"- `SCP1 trajectory benefit`：`{'yes' if max(meanpool_mean, gru_mean) > static_mean + 1e-9 else 'not_yet'}`"
            )
        if str(dataset) == "natops":
            lines.append(f"- `NATOPS no obvious degradation vs static`：`{natops_guard}`")
        if not np.isnan(raw_mean):
            lines.append(
                f"- `best dynamic vs raw+MiniROCKET`：`{float(max(meanpool_mean, gru_mean) - raw_mean):+.4f}`"
            )
        lines.append("")
    conclusion_md = os.path.join(args.out_root, "dynamic_manifold_conclusion.md")
    with open(conclusion_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[dynamic-manifold] wrote {config_csv}")
    print(f"[dynamic-manifold] wrote {per_seed_csv}")
    print(f"[dynamic-manifold] wrote {summary_csv}")
    print(f"[dynamic-manifold] wrote {diagnostics_csv}")
    print(f"[dynamic-manifold] wrote {conclusion_md}")


if __name__ == "__main__":
    main()
