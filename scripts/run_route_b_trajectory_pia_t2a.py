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

from route_b_unified.trajectory_classifier import TrajectoryModelConfig  # noqa: E402
from route_b_unified.trajectory_pia_evaluator import (  # noqa: E402
    TrajectoryPIAEvalConfig,
    evaluate_trajectory_pia_t2a,
)
from route_b_unified.trajectory_pia_operator import TrajectoryPIAOperatorConfig  # noqa: E402
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


def _mean_std(values: List[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def _format_mean_std(values: List[float]) -> str:
    mean, std = _mean_std(values)
    return f"{mean:.4f} +/- {std:.4f}"


def _load_t0_reference_map(path: str) -> Dict[tuple[str, int], float]:
    if not os.path.isfile(path):
        return {}
    df = pd.read_csv(path)
    df = df[df["model_type"].astype(str) == "dynamic_gru"].copy()
    out: Dict[tuple[str, int], float] = {}
    for _, row in df.iterrows():
        out[(str(row["dataset"]).strip().lower(), int(row["seed"]))] = float(row["test_macro_f1"])
    return out


def _load_raw_reference_map(path: str) -> Dict[tuple[str, int], float]:
    if not os.path.isfile(path):
        return {}
    df = pd.read_csv(path)
    out: Dict[tuple[str, int], float] = {}
    for _, row in df.iterrows():
        out[(str(row["dataset"]).strip().lower(), int(row["seed"]))] = float(row["raw_minirocket_test_macro_f1"])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="T2a trajectory-aware PIA operator: baseline vs unsmoothed vs smoothed.")
    p.add_argument("--datasets", type=str, default="natops,selfregulationscp1")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/route_b_trajectory_pia_t2a_20260329_formal")
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
    p.add_argument("--gamma-main", type=float, default=0.10)
    p.add_argument("--smooth-lambdas", type=str, default="0.00,0.50")
    p.add_argument(
        "--t0-reference-csv",
        type=str,
        default="/home/THL/project/MTS-PIA/out/route_b_dynamic_manifold_classification_20260329_formal/dynamic_manifold_per_seed.csv",
    )
    args = p.parse_args()

    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_seed_list(args.seeds)
    smooth_lambdas = [float(v.strip()) for v in str(args.smooth_lambdas).split(",") if v.strip()]
    _ensure_dir(args.out_root)

    t0_map = _load_t0_reference_map(args.t0_reference_csv)
    raw_map = _load_raw_reference_map(args.t0_reference_csv)

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
                os.path.join(seed_dir, "trajectory_pia_t2a_split_meta.json"),
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "split_meta": dict(state.split_meta),
                    "meta": dict(state.meta),
                    "window_len": int(state.window_len),
                    "hop_len": int(state.hop_len),
                },
            )

            model_cfg = TrajectoryModelConfig(
                z_dim=int(state.z_dim),
                num_classes=int(state.num_classes),
                gru_hidden_dim=int(args.gru_hidden_dim),
                dropout=float(args.dropout),
            )
            operator_cfg = TrajectoryPIAOperatorConfig(seed=int(seed))

            operator_settings = [("baseline", 0.0)]
            for lam in smooth_lambdas:
                mode = "operator_unsmoothed" if float(lam) <= 1e-12 else "operator_smoothed"
                operator_settings.append((mode, float(lam)))

            for operator_mode, smooth_lambda in operator_settings:
                eval_cfg = TrajectoryPIAEvalConfig(
                    operator_mode=str(operator_mode),
                    gamma_main=float(args.gamma_main),
                    smooth_lambda=float(smooth_lambda),
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    patience=int(args.patience),
                    device=str(args.device),
                )
                result = evaluate_trajectory_pia_t2a(
                    state,
                    seed=int(seed),
                    model_cfg=model_cfg,
                    eval_cfg=eval_cfg,
                    operator_cfg=operator_cfg,
                )
                _write_json(
                    os.path.join(seed_dir, f"{operator_mode}_result.json"),
                    {
                        "dataset": result.dataset,
                        "seed": result.seed,
                        "operator_mode": result.operator_mode,
                        "train_metrics": result.train_metrics,
                        "val_metrics": result.val_metrics,
                        "test_metrics": result.test_metrics,
                        "best_epoch": result.best_epoch,
                        "diagnostics": result.diagnostics,
                        "operator_meta": result.operator_meta,
                        "meta": result.meta,
                        "history_rows": result.history_rows,
                    },
                )

                config_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "operator_mode": str(operator_mode),
                        "axis_count": 1,
                        "r_dimension": 1,
                        "gamma_main": float(args.gamma_main),
                        "smooth_lambda": float(smooth_lambda),
                        "window_len": int(state.window_len),
                        "hop_len": int(state.hop_len),
                        "gru_hidden_dim": int(args.gru_hidden_dim),
                        "dropout": float(args.dropout),
                        "epochs": int(args.epochs),
                        "batch_size": int(args.batch_size),
                    }
                )
                per_seed_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "operator_mode": str(operator_mode),
                        "axis_count": 1,
                        "gamma_main": float(args.gamma_main),
                        "smooth_lambda": float(smooth_lambda),
                        "test_macro_f1": float(result.test_metrics["macro_f1"]),
                        "delta_vs_dynamic_baseline": np.nan,
                        "t0_dynamic_gru_reference": float(t0_map.get((str(dataset), int(seed)), np.nan)),
                        "raw_minirocket_reference": float(raw_map.get((str(dataset), int(seed)), np.nan)),
                    }
                )
                diagnostics_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "operator_mode": str(operator_mode),
                        **{k: float(v) for k, v in result.diagnostics.items()},
                    }
                )

    per_seed_df = pd.DataFrame(per_seed_rows)
    for dataset in datasets:
        ds = per_seed_df[per_seed_df["dataset"] == dataset].copy()
        baseline_rows = ds[ds["operator_mode"] == "baseline"]
        baseline_map = {int(r["seed"]): float(r["test_macro_f1"]) for _, r in baseline_rows.iterrows()}
        mask = per_seed_df["dataset"] == dataset
        per_seed_df.loc[mask, "delta_vs_dynamic_baseline"] = [
            float(per_seed_df.loc[idx, "test_macro_f1"]) - float(baseline_map.get(int(per_seed_df.loc[idx, "seed"]), np.nan))
            for idx in per_seed_df[mask].index
        ]

    config_df = pd.DataFrame(config_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows)

    summary_rows: List[Dict[str, object]] = []
    for dataset in datasets:
        ds = per_seed_df[per_seed_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        baseline = ds[ds["operator_mode"] == "baseline"]["test_macro_f1"].tolist()
        unsmoothed = ds[ds["operator_mode"] == "operator_unsmoothed"]["test_macro_f1"].tolist()
        smoothed = ds[ds["operator_mode"] == "operator_smoothed"]["test_macro_f1"].tolist()
        best_mode = "baseline"
        best_mean = _mean_std(baseline)[0]
        for mode_name, vals in [("operator_unsmoothed", unsmoothed), ("operator_smoothed", smoothed)]:
            mean_val = _mean_std(vals)[0]
            if mean_val > best_mean:
                best_mean = mean_val
                best_mode = mode_name
        t0_ref = ds["t0_dynamic_gru_reference"].dropna().tolist()
        raw_ref = ds["raw_minirocket_reference"].dropna().tolist()
        row: Dict[str, object] = {
            "dataset": str(dataset),
            "baseline_macro_f1_mean": _mean_std(baseline)[0],
            "baseline_macro_f1_std": _mean_std(baseline)[1],
            "baseline_macro_f1": _format_mean_std(baseline),
            "operator_unsmoothed_macro_f1_mean": _mean_std(unsmoothed)[0],
            "operator_unsmoothed_macro_f1_std": _mean_std(unsmoothed)[1],
            "operator_unsmoothed_macro_f1": _format_mean_std(unsmoothed),
            "operator_smoothed_macro_f1_mean": _mean_std(smoothed)[0],
            "operator_smoothed_macro_f1_std": _mean_std(smoothed)[1],
            "operator_smoothed_macro_f1": _format_mean_std(smoothed),
            "t0_dynamic_gru_macro_f1_mean": _mean_std(t0_ref)[0],
            "t0_dynamic_gru_macro_f1_std": _mean_std(t0_ref)[1],
            "t0_dynamic_gru_macro_f1": _format_mean_std(t0_ref),
            "raw_minirocket_macro_f1_mean": _mean_std(raw_ref)[0],
            "raw_minirocket_macro_f1_std": _mean_std(raw_ref)[1],
            "raw_minirocket_macro_f1": _format_mean_std(raw_ref),
            "best_mode": str(best_mode),
        }
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    config_csv = os.path.join(args.out_root, "trajectory_pia_t2a_config_table.csv")
    per_seed_csv = os.path.join(args.out_root, "trajectory_pia_t2a_per_seed.csv")
    summary_csv = os.path.join(args.out_root, "trajectory_pia_t2a_dataset_summary.csv")
    diagnostics_csv = os.path.join(args.out_root, "trajectory_pia_t2a_diagnostics_summary.csv")
    config_df.to_csv(config_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    lines: List[str] = [
        "# Trajectory PIA T2a Conclusion",
        "",
        "更新时间：2026-03-29",
        "",
        "本轮主比较对象：`baseline / operator_unsmoothed / operator_smoothed`。",
        "背景参考：`T0 dynamic_gru formal / raw + MiniROCKET formal`。",
        "",
    ]
    for dataset in datasets:
        ds = summary_df[summary_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        row = ds.iloc[0]
        baseline_mean = float(row["baseline_macro_f1_mean"])
        unsmoothed_mean = float(row["operator_unsmoothed_macro_f1_mean"])
        smoothed_mean = float(row["operator_smoothed_macro_f1_mean"])
        t0_mean = float(row["t0_dynamic_gru_macro_f1_mean"])
        raw_mean = float(row["raw_minirocket_macro_f1_mean"])
        operator_established = "yes" if max(unsmoothed_mean, smoothed_mean) > baseline_mean + 1e-9 else "not_yet"
        smoothing_needed = "yes" if smoothed_mean > unsmoothed_mean + 1e-9 else "not_yet"
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(f"- `baseline`: {row['baseline_macro_f1']}")
        lines.append(f"- `operator_unsmoothed`: {row['operator_unsmoothed_macro_f1']}")
        lines.append(f"- `operator_smoothed`: {row['operator_smoothed_macro_f1']}")
        lines.append(f"- `T0 dynamic_gru` (reference): {row['t0_dynamic_gru_macro_f1']}")
        lines.append(f"- `raw + MiniROCKET` (reference): {row['raw_minirocket_macro_f1']}")
        lines.append("")
        lines.append(f"- `trajectory operator established`：`{operator_established}`")
        lines.append(f"- `smoothing needed`：`{smoothing_needed}`")
        if not np.isnan(t0_mean):
            lines.append(f"- `best operator vs T0 dynamic_gru`：`{float(max(unsmoothed_mean, smoothed_mean) - t0_mean):+.4f}`")
        if not np.isnan(raw_mean):
            lines.append(f"- `best operator vs raw + MiniROCKET`：`{float(max(unsmoothed_mean, smoothed_mean) - raw_mean):+.4f}`")
        lines.append("")
    conclusion_md = os.path.join(args.out_root, "trajectory_pia_t2a_conclusion.md")
    with open(conclusion_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[trajectory-pia-t2a] wrote {config_csv}")
    print(f"[trajectory-pia-t2a] wrote {per_seed_csv}")
    print(f"[trajectory-pia-t2a] wrote {summary_csv}")
    print(f"[trajectory-pia-t2a] wrote {diagnostics_csv}")
    print(f"[trajectory-pia-t2a] wrote {conclusion_md}")


if __name__ == "__main__":
    main()

