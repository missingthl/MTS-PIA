#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from route_b_unified import RepresentationConfig, build_representation  # noqa: E402
from route_b_unified.dual_stream_classifier import DualStreamModelConfig  # noqa: E402
from route_b_unified.dual_stream_dataset import build_dual_stream_state  # noqa: E402
from route_b_unified.dual_stream_evaluator import (  # noqa: E402
    DualStreamEvalConfig,
    evaluate_dual_stream,
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


def main() -> None:
    p = argparse.ArgumentParser(description="No-bridge dual-stream classification: raw DCNet stream + z-space manifold stream.")
    p.add_argument("--datasets", type=str, default="natops,selfregulationscp1")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/route_b_dual_stream_no_bridge_20260329_formal")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--spatial-proj-channels", type=int, default=16)
    p.add_argument("--spatial-proj-bins", type=int, default=8)
    p.add_argument("--manifold-hidden-dim", type=int, default=128)
    p.add_argument("--manifold-feature-dim", type=int, default=64)
    p.add_argument("--fusion-hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--dual-aux-weight", type=float, default=0.30)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_seed_list(args.seeds)
    _ensure_dir(args.out_root)

    config_rows: List[Dict[str, object]] = []
    per_seed_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        for seed in seeds:
            rep_state = build_representation(
                RepresentationConfig(
                    dataset=str(dataset),
                    seed=int(seed),
                    val_fraction=float(args.val_fraction),
                    spd_eps=float(args.spd_eps),
                )
            )
            ds_state = build_dual_stream_state(rep_state)
            model_cfg = DualStreamModelConfig(
                channels=int(ds_state.channels),
                seq_len=int(ds_state.max_length),
                z_dim=int(ds_state.z_dim),
                num_classes=int(ds_state.num_classes),
                spatial_proj_channels=int(args.spatial_proj_channels),
                spatial_proj_bins=int(args.spatial_proj_bins),
                manifold_hidden_dim=int(args.manifold_hidden_dim),
                manifold_feature_dim=int(args.manifold_feature_dim),
                fusion_hidden_dim=int(args.fusion_hidden_dim),
                dropout=float(args.dropout),
                dual_aux_weight=float(args.dual_aux_weight),
            )

            seed_dir = os.path.join(args.out_root, f"{dataset}_seed{seed}")
            _ensure_dir(seed_dir)
            _write_json(
                os.path.join(seed_dir, "split_meta.json"),
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "split_meta": dict(rep_state.split_meta),
                    "rep_meta": dict(rep_state.meta),
                    "dual_stream_meta": dict(ds_state.meta),
                },
            )

            for variant in ["spatial_only", "manifold_only", "dual_stream"]:
                eval_cfg = DualStreamEvalConfig(
                    variant=str(variant),
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    patience=int(args.patience),
                    device=str(args.device),
                )
                result = evaluate_dual_stream(ds_state, seed=int(seed), model_cfg=model_cfg, eval_cfg=eval_cfg)
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
                        "variant": str(variant),
                        "channels": int(ds_state.channels),
                        "max_length": int(ds_state.max_length),
                        "z_dim": int(ds_state.z_dim),
                        "num_classes": int(ds_state.num_classes),
                        "spatial_proj_channels": int(args.spatial_proj_channels),
                        "spatial_proj_bins": int(args.spatial_proj_bins),
                        "manifold_hidden_dim": int(args.manifold_hidden_dim),
                        "manifold_feature_dim": int(args.manifold_feature_dim),
                        "fusion_hidden_dim": int(args.fusion_hidden_dim),
                        "dropout": float(args.dropout),
                        "dual_aux_weight": float(args.dual_aux_weight),
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
                        "variant": str(variant),
                        "train_acc": float(result.train_metrics["acc"]),
                        "train_macro_f1": float(result.train_metrics["macro_f1"]),
                        "val_acc": float(result.val_metrics["acc"]),
                        "val_macro_f1": float(result.val_metrics["macro_f1"]),
                        "test_acc": float(result.test_metrics["acc"]),
                        "test_macro_f1": float(result.test_metrics["macro_f1"]),
                        "best_epoch": int(result.best_epoch),
                    }
                )

    config_df = pd.DataFrame(config_rows)
    per_seed_df = pd.DataFrame(per_seed_rows)
    summary_rows: List[Dict[str, object]] = []
    for (dataset, variant), sub in per_seed_df.groupby(["dataset", "variant"], sort=True):
        val_macro_f1_mean, val_macro_f1_std = _mean_std(sub["val_macro_f1"].tolist())
        test_macro_f1_mean, test_macro_f1_std = _mean_std(sub["test_macro_f1"].tolist())
        val_acc_mean, val_acc_std = _mean_std(sub["val_acc"].tolist())
        test_acc_mean, test_acc_std = _mean_std(sub["test_acc"].tolist())
        summary_rows.append(
            {
                "dataset": str(dataset),
                "variant": str(variant),
                "val_macro_f1_mean": val_macro_f1_mean,
                "val_macro_f1_std": val_macro_f1_std,
                "test_macro_f1_mean": test_macro_f1_mean,
                "test_macro_f1_std": test_macro_f1_std,
                "val_acc_mean": val_acc_mean,
                "val_acc_std": val_acc_std,
                "test_acc_mean": test_acc_mean,
                "test_acc_std": test_acc_std,
                "val_macro_f1": _format_mean_std(sub["val_macro_f1"].tolist()),
                "test_macro_f1": _format_mean_std(sub["test_macro_f1"].tolist()),
                "val_acc": _format_mean_std(sub["val_acc"].tolist()),
                "test_acc": _format_mean_std(sub["test_acc"].tolist()),
                "best_epoch_mean": float(sub["best_epoch"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    config_csv = os.path.join(args.out_root, "dual_stream_no_bridge_config_table.csv")
    per_seed_csv = os.path.join(args.out_root, "dual_stream_no_bridge_per_seed.csv")
    summary_csv = os.path.join(args.out_root, "dual_stream_no_bridge_dataset_summary.csv")
    config_df.to_csv(config_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    lines: List[str] = [
        "# No-Bridge Dual-Stream Conclusion",
        "",
        "更新时间：2026-03-29",
        "",
        "本轮比较对象：`spatial_only / manifold_only / dual_stream`。",
        "",
    ]
    for dataset in datasets:
        ds = summary_df[summary_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        lines.append(f"## {dataset}")
        lines.append("")
        best = ds.sort_values("test_macro_f1_mean", ascending=False).iloc[0]
        for _, row in ds.iterrows():
            lines.append(
                f"- `{row['variant']}`: val_macro_f1={row['val_macro_f1']}, test_macro_f1={row['test_macro_f1']}"
            )
        lines.append("")
        lines.append(f"- 当前 test_macro_f1 最强的是 `{best['variant']}`。")
        lines.append("")
    conclusion_md = os.path.join(args.out_root, "dual_stream_no_bridge_conclusion.md")
    with open(conclusion_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[dual-stream-no-bridge] wrote {config_csv}")
    print(f"[dual-stream-no-bridge] wrote {per_seed_csv}")
    print(f"[dual-stream-no-bridge] wrote {summary_csv}")
    print(f"[dual-stream-no-bridge] wrote {conclusion_md}")


if __name__ == "__main__":
    main()
