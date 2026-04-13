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

from route_b_unified.trajectory_classifier import TrajectoryModelConfig  # noqa: E402
from route_b_unified.trajectory_pia_evaluator import (  # noqa: E402
    TrajectoryPIAEvalConfig,
    TrajectoryPIAEvalResult,
    compute_trajectory_diagnostics,
    evaluate_trajectory_pia_t2a,
    evaluate_trajectory_train_final,
)
from route_b_unified.trajectory_pia_operator import TrajectoryPIAOperator, TrajectoryPIAOperatorConfig  # noqa: E402
from route_b_unified.trajectory_pia_operator_t2b import (  # noqa: E402
    TrajectoryPIAT2B0Config,
    TrajectoryPIAT2B0Operator,
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


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def _format_mean_std(values: Sequence[float]) -> str:
    mean, std = _mean_std(values)
    return f"{mean:.4f} +/- {std:.4f}"


def _save_result_json(path: str, result: TrajectoryPIAEvalResult) -> None:
    _write_json(
        path,
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


def _baseline_train_bundle(state) -> tuple[List[str], List[int], List[np.ndarray], Dict[str, object], Dict[str, float]]:
    train_tids = [str(v) for v in state.train.tids.tolist()]
    train_labels = [int(v) for v in state.train.y.tolist()]
    train_seqs = [np.asarray(v, dtype=np.float32) for v in state.train.z_seq_list]
    diagnostics = compute_trajectory_diagnostics(train_seqs, train_labels, continuity_ratio=1.0)
    diagnostics.update(
        {
            "saliency_low_ratio": np.nan,
            "saliency_mid_ratio": np.nan,
            "saliency_high_ratio": np.nan,
            "gamma_effective_mean": 0.0,
        }
    )
    return train_tids, train_labels, train_seqs, {"mode": "baseline"}, diagnostics


def _t2b_train_bundle(
    state,
    *,
    operator: TrajectoryPIAT2B0Operator,
    mode: str,
) -> tuple[List[str], List[int], List[np.ndarray], Dict[str, object], Dict[str, float]]:
    train_tids = [str(v) for v in state.train.tids.tolist()]
    train_labels = [int(v) for v in state.train.y.tolist()]
    train_seqs = [np.asarray(v, dtype=np.float32) for v in state.train.z_seq_list]
    aug_seqs, _delta_list, _gamma_list, op_meta = operator.transform_many(
        train_seqs,
        mode=str(mode),
        trial_ids=train_tids,
    )
    aug_tids = [f"{tid}__{mode}_aug" for tid in train_tids]
    final_tids = list(train_tids) + list(aug_tids)
    final_labels = list(train_labels) + list(train_labels)
    final_seqs = list(train_seqs) + list(aug_seqs)
    diagnostics = compute_trajectory_diagnostics(
        final_seqs,
        final_labels,
        continuity_ratio=float(op_meta["mean_continuity_distortion_ratio"]),
    )
    diagnostics.update(
        {
            "saliency_low_ratio": float(op_meta["saliency_low_ratio"]),
            "saliency_mid_ratio": float(op_meta["saliency_mid_ratio"]),
            "saliency_high_ratio": float(op_meta["saliency_high_ratio"]),
            "gamma_effective_mean": float(op_meta["gamma_effective_mean"]),
        }
    )
    return final_tids, final_labels, final_seqs, op_meta, diagnostics


def main() -> None:
    p = argparse.ArgumentParser(
        description="T2b-0 fixed-rule local saliency probe: baseline vs T2a default vs T2b saliency vs randomized control."
    )
    p.add_argument("--datasets", type=str, default="natops,selfregulationscp1")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/route_b_trajectory_pia_t2b0_20260329_formal")
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
    p.add_argument("--gamma-base", type=float, default=0.05)
    p.add_argument("--smooth-lambda", type=float, default=0.50)
    args = p.parse_args()

    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_seed_list(args.seeds)
    out_root = str(args.out_root)
    _ensure_dir(out_root)

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
            seed_dir = os.path.join(out_root, f"{dataset}_seed{seed}")
            _ensure_dir(seed_dir)

            model_cfg = TrajectoryModelConfig(
                z_dim=int(state.z_dim),
                num_classes=int(state.num_classes),
                gru_hidden_dim=int(args.gru_hidden_dim),
                dropout=float(args.dropout),
            )
            eval_common = {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "patience": int(args.patience),
                "device": str(args.device),
            }
            base_operator = TrajectoryPIAOperator(TrajectoryPIAOperatorConfig(seed=int(seed))).fit(state.train.z_seq_list)
            t2b_operator = TrajectoryPIAT2B0Operator(
                base_cfg=TrajectoryPIAOperatorConfig(seed=int(seed)),
                t2b_cfg=TrajectoryPIAT2B0Config(
                    gamma_base=float(args.gamma_base),
                    smooth_lambda=float(args.smooth_lambda),
                    seed=int(seed),
                ),
            ).fit(state.train.z_seq_list, prefit_base_operator=base_operator)

            _write_json(
                os.path.join(seed_dir, "trajectory_pia_t2b0_split_meta.json"),
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "split_meta": dict(state.split_meta),
                    "meta": dict(state.meta),
                    "window_len": int(state.window_len),
                    "hop_len": int(state.hop_len),
                    "t2a_default": {
                        "gamma_main": float(args.gamma_base),
                        "smooth_lambda": float(args.smooth_lambda),
                    },
                },
            )

            # baseline
            baseline_tids, baseline_labels, baseline_seqs, baseline_meta, baseline_diag = _baseline_train_bundle(state)
            baseline_result = evaluate_trajectory_train_final(
                state,
                seed=int(seed),
                model_cfg=model_cfg,
                eval_cfg=TrajectoryPIAEvalConfig(
                    operator_mode="baseline",
                    gamma_main=float(args.gamma_base),
                    smooth_lambda=float(args.smooth_lambda),
                    **eval_common,
                ),
                train_tids=baseline_tids,
                train_labels=baseline_labels,
                train_z_seq_list=baseline_seqs,
                diagnostics=baseline_diag,
                operator_meta=baseline_meta,
            )
            _save_result_json(os.path.join(seed_dir, "baseline_result.json"), baseline_result)

            # T2a default
            t2a_result = evaluate_trajectory_pia_t2a(
                state,
                seed=int(seed),
                model_cfg=model_cfg,
                eval_cfg=TrajectoryPIAEvalConfig(
                    operator_mode="t2a_default",
                    gamma_main=float(args.gamma_base),
                    smooth_lambda=float(args.smooth_lambda),
                    **eval_common,
                ),
                operator_cfg=TrajectoryPIAOperatorConfig(seed=int(seed)),
                prefit_operator=base_operator,
            )
            t2a_diag = dict(t2a_result.diagnostics)
            t2a_diag.setdefault("saliency_low_ratio", np.nan)
            t2a_diag.setdefault("saliency_mid_ratio", np.nan)
            t2a_diag.setdefault("saliency_high_ratio", np.nan)
            t2a_diag.setdefault("gamma_effective_mean", float(args.gamma_base))
            t2a_result.diagnostics = t2a_diag
            _save_result_json(os.path.join(seed_dir, "t2a_default_result.json"), t2a_result)

            # T2b saliency-aware
            sal_tids, sal_labels, sal_seqs, sal_meta, sal_diag = _t2b_train_bundle(state, operator=t2b_operator, mode="saliency")
            sal_result = evaluate_trajectory_train_final(
                state,
                seed=int(seed),
                model_cfg=model_cfg,
                eval_cfg=TrajectoryPIAEvalConfig(
                    operator_mode="t2b_saliency",
                    gamma_main=float(args.gamma_base),
                    smooth_lambda=float(args.smooth_lambda),
                    **eval_common,
                ),
                train_tids=sal_tids,
                train_labels=sal_labels,
                train_z_seq_list=sal_seqs,
                diagnostics=sal_diag,
                operator_meta=sal_meta,
            )
            _save_result_json(os.path.join(seed_dir, "t2b_saliency_result.json"), sal_result)

            # T2b randomized local-varying control
            rnd_tids, rnd_labels, rnd_seqs, rnd_meta, rnd_diag = _t2b_train_bundle(state, operator=t2b_operator, mode="randomized")
            rnd_result = evaluate_trajectory_train_final(
                state,
                seed=int(seed),
                model_cfg=model_cfg,
                eval_cfg=TrajectoryPIAEvalConfig(
                    operator_mode="t2b_randomized",
                    gamma_main=float(args.gamma_base),
                    smooth_lambda=float(args.smooth_lambda),
                    **eval_common,
                ),
                train_tids=rnd_tids,
                train_labels=rnd_labels,
                train_z_seq_list=rnd_seqs,
                diagnostics=rnd_diag,
                operator_meta=rnd_meta,
            )
            _save_result_json(os.path.join(seed_dir, "t2b_randomized_result.json"), rnd_result)

            run_results = [baseline_result, t2a_result, sal_result, rnd_result]
            for result in run_results:
                config_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "operator_mode": str(result.operator_mode),
                        "axis_count": 1,
                        "gamma_base": float(args.gamma_base),
                        "smooth_lambda": float(args.smooth_lambda),
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
                        "operator_mode": str(result.operator_mode),
                        "gamma_base": float(args.gamma_base),
                        "smooth_lambda": float(args.smooth_lambda),
                        "test_macro_f1": float(result.test_metrics["macro_f1"]),
                        "delta_vs_baseline": np.nan,
                        "delta_vs_t2a_default": np.nan,
                    }
                )
                diagnostics_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "operator_mode": str(result.operator_mode),
                        **{k: float(v) if v is not None else np.nan for k, v in result.diagnostics.items()},
                    }
                )

    per_seed_df = pd.DataFrame(per_seed_rows)
    for dataset in datasets:
        ds = per_seed_df[per_seed_df["dataset"] == dataset].copy()
        baseline_map = {
            int(r["seed"]): float(r["test_macro_f1"])
            for _, r in ds[ds["operator_mode"] == "baseline"].iterrows()
        }
        t2a_map = {
            int(r["seed"]): float(r["test_macro_f1"])
            for _, r in ds[ds["operator_mode"] == "t2a_default"].iterrows()
        }
        mask = per_seed_df["dataset"] == dataset
        per_seed_df.loc[mask, "delta_vs_baseline"] = [
            float(per_seed_df.loc[idx, "test_macro_f1"]) - float(baseline_map.get(int(per_seed_df.loc[idx, "seed"]), np.nan))
            for idx in per_seed_df[mask].index
        ]
        per_seed_df.loc[mask, "delta_vs_t2a_default"] = [
            float(per_seed_df.loc[idx, "test_macro_f1"]) - float(t2a_map.get(int(per_seed_df.loc[idx, "seed"]), np.nan))
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
        t2a_default = ds[ds["operator_mode"] == "t2a_default"]["test_macro_f1"].tolist()
        t2b_saliency = ds[ds["operator_mode"] == "t2b_saliency"]["test_macro_f1"].tolist()
        t2b_randomized = ds[ds["operator_mode"] == "t2b_randomized"]["test_macro_f1"].tolist()
        best_mode = "baseline"
        best_mean = _mean_std(baseline)[0]
        for mode_name, vals in [
            ("t2a_default", t2a_default),
            ("t2b_saliency", t2b_saliency),
            ("t2b_randomized", t2b_randomized),
        ]:
            mean_val = _mean_std(vals)[0]
            if mean_val > best_mean:
                best_mean = mean_val
                best_mode = mode_name
        summary_rows.append(
            {
                "dataset": str(dataset),
                "baseline_macro_f1_mean": _mean_std(baseline)[0],
                "baseline_macro_f1_std": _mean_std(baseline)[1],
                "baseline_macro_f1": _format_mean_std(baseline),
                "t2a_default_macro_f1_mean": _mean_std(t2a_default)[0],
                "t2a_default_macro_f1_std": _mean_std(t2a_default)[1],
                "t2a_default_macro_f1": _format_mean_std(t2a_default),
                "t2b_saliency_macro_f1_mean": _mean_std(t2b_saliency)[0],
                "t2b_saliency_macro_f1_std": _mean_std(t2b_saliency)[1],
                "t2b_saliency_macro_f1": _format_mean_std(t2b_saliency),
                "t2b_random_macro_f1_mean": _mean_std(t2b_randomized)[0],
                "t2b_random_macro_f1_std": _mean_std(t2b_randomized)[1],
                "t2b_random_macro_f1": _format_mean_std(t2b_randomized),
                "best_mode": str(best_mode),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    config_csv = os.path.join(out_root, "trajectory_pia_t2b0_config_table.csv")
    per_seed_csv = os.path.join(out_root, "trajectory_pia_t2b0_per_seed.csv")
    summary_csv = os.path.join(out_root, "trajectory_pia_t2b0_dataset_summary.csv")
    diagnostics_csv = os.path.join(out_root, "trajectory_pia_t2b0_diagnostics_summary.csv")
    config_df.to_csv(config_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    lines: List[str] = [
        "# Trajectory PIA T2b-0 Conclusion",
        "",
        "更新时间：2026-03-29",
        "",
        "本轮主比较对象：`baseline / t2a_default / t2b_saliency / t2b_randomized`。",
        "当前口径：`T2b-0` 是 fixed-rule local saliency probe，不是完整局部动力学算子。",
        "",
    ]
    for dataset in datasets:
        ds = summary_df[summary_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        row = ds.iloc[0]
        baseline_mean = float(row["baseline_macro_f1_mean"])
        t2a_mean = float(row["t2a_default_macro_f1_mean"])
        sal_mean = float(row["t2b_saliency_macro_f1_mean"])
        rnd_mean = float(row["t2b_random_macro_f1_mean"])
        sal_vs_baseline = "yes" if sal_mean > baseline_mean + 1e-9 else "not_yet"
        sal_vs_t2a = "yes" if sal_mean > t2a_mean + 1e-9 else "not_yet"
        sal_vs_random = "yes" if sal_mean > rnd_mean + 1e-9 else "not_yet"
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(f"- `baseline`: {row['baseline_macro_f1']}")
        lines.append(f"- `t2a_default`: {row['t2a_default_macro_f1']}")
        lines.append(f"- `t2b_saliency`: {row['t2b_saliency_macro_f1']}")
        lines.append(f"- `t2b_randomized`: {row['t2b_random_macro_f1']}")
        lines.append("")
        lines.append(f"- `t2b_saliency > baseline`：`{sal_vs_baseline}`")
        lines.append(f"- `t2b_saliency > t2a_default`：`{sal_vs_t2a}`")
        lines.append(f"- `t2b_saliency > randomized`：`{sal_vs_random}`")
        lines.append("")
    conclusion_md = os.path.join(out_root, "trajectory_pia_t2b0_conclusion.md")
    with open(conclusion_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[trajectory-pia-t2b0] wrote {config_csv}")
    print(f"[trajectory-pia-t2b0] wrote {per_seed_csv}")
    print(f"[trajectory-pia-t2b0] wrote {summary_csv}")
    print(f"[trajectory-pia-t2b0] wrote {diagnostics_csv}")
    print(f"[trajectory-pia-t2b0] wrote {conclusion_md}")


if __name__ == "__main__":
    main()
