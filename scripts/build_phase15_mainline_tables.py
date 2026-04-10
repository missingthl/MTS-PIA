#!/usr/bin/env python
"""Build formal Phase15 mainline freeze tables.

Outputs:
- main_performance_table.csv
- mechanism_diagnosis_table.csv
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


CORE_DATASETS = [
    "seed1",
    "seediv",
    "seedv",
    "har",
    "natops",
    "fingermovements",
    "selfregulationscp1",
]


PROTOCOL_INFO = {
    "seed1": (
        "seed_family_native",
        "SEED native protocol: per session first 9 trials train, last 6 trials test",
    ),
    "seediv": (
        "seed_family_native",
        "SEED_IV native protocol: per session first 16 trials train, last 8 trials test",
    ),
    "seedv": (
        "seed_family_native",
        "SEED_V native protocol: per session first 9 trials train, last 6 trials test",
    ),
    "har": (
        "fixed_split",
        "dataset-provided TRAIN/TEST split; z-space mainline keeps standard windowed SPD/tangent pipeline",
    ),
    "natops": (
        "fixed_split",
        "dataset-provided TRAIN/TEST split; z-space mainline keeps standard windowed SPD/tangent pipeline",
    ),
    "fingermovements": (
        "fixed_split",
        "dataset-provided TRAIN/TEST split; z-space mainline keeps standard windowed SPD/tangent pipeline",
    ),
    "selfregulationscp1": (
        "fixed_split",
        "dataset-provided TRAIN/TEST split; z-space mainline keeps standard windowed SPD/tangent pipeline",
    ),
}


def _stats(series: pd.Series) -> Tuple[float, float]:
    arr = series.to_numpy(dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def _load_minirocket_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        dataset = str(row["dataset"]).strip().lower()
        acc = None
        f1 = None
        if pd.notna(row.get("trial_acc")):
            acc = float(row["trial_acc"])
        elif pd.notna(row.get("test_acc")):
            acc = float(row["test_acc"])
        if pd.notna(row.get("trial_macro_f1")):
            f1 = float(row["trial_macro_f1"])
        elif pd.notna(row.get("test_macro_f1")):
            f1 = float(row["test_macro_f1"])
        rows.append(
            {
                "dataset": dataset,
                "minirocket_protocol_type": str(row.get("protocol_type", "")).strip(),
                "minirocket_acc": acc,
                "minirocket_f1": f1,
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["dataset"], keep="last")


def _best_method_label(values: Dict[str, float]) -> str:
    best_name = max(values.items(), key=lambda kv: float(kv[1]))[0]
    return best_name


def _remark_for_row(
    *,
    step_variant: str,
    baseline_f1: float,
    step1b_f1: float,
    step1b_gate_f1: float,
    flip_rate: float,
    delta_intrusion: float,
    n_seeds: int,
) -> str:
    eps = 0.005
    if step_variant == "step1b":
        if step1b_f1 > baseline_f1 + eps:
            return "局部正信号，方向库可用；当前仍需多 seed 验证（推断）" if n_seeds == 1 else "收益稳定，方向库质量可接受（推断）"
        if step1b_f1 < baseline_f1 - eps:
            if flip_rate > 0.02 or delta_intrusion > 0.0:
                return "更像方向质量问题，增强未转化为有效边界移动（推断）"
            return "收益不稳定，当前证据不足（推断）"
        return "接近 baseline，当前更像收益不稳定（推断）"

    if step1b_gate_f1 > step1b_f1 + eps:
        return "Gate 进一步过滤无效增强，当前为正信号（推断）" if n_seeds == 1 else "Gate 过滤有效，收益更稳（推断）"
    if step1b_gate_f1 < step1b_f1 - eps and step1b_f1 > baseline_f1 + eps:
        return "更像控制阈值偏严，Gate 把有效增强也过滤掉了（推断）"
    if step1b_gate_f1 < baseline_f1 - eps and step1b_f1 < baseline_f1 - eps:
        return "Gate 未能挽回方向质量问题，整体收益不足（推断）"
    return "相对 Step1B 无明显额外收益，当前证据不足（推断）"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mainline-root", type=str, default="out/phase15_mainline_freeze_20260319")
    parser.add_argument(
        "--minirocket-csv",
        type=str,
        default="out/raw_minirocket_official_protocol_20260318/official_protocol_status.csv",
    )
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    out_dir = args.out_dir or args.mainline_root
    os.makedirs(out_dir, exist_ok=True)

    df_mini = _load_minirocket_table(args.minirocket_csv)

    perf_rows: List[Dict[str, object]] = []
    mech_rows: List[Dict[str, object]] = []

    for dataset in CORE_DATASETS:
        perf_path = os.path.join(args.mainline_root, dataset, "summary_per_seed.csv")
        mech_path = os.path.join(args.mainline_root, dataset, "mechanism_per_seed.csv")
        protocol_type, protocol_note = PROTOCOL_INFO[dataset]

        if not (os.path.isfile(perf_path) and os.path.isfile(mech_path)):
            mini_row = df_mini[df_mini["dataset"] == dataset]
            if mini_row.empty:
                raise RuntimeError(f"Missing MiniROCKET official row for dataset={dataset}")
            mini = mini_row.iloc[0]
            perf_rows.append(
                {
                    "dataset": dataset,
                    "protocol_type": protocol_type,
                    "protocol_note": protocol_note,
                    "n_seeds": 0,
                    "baseline_acc": np.nan,
                    "baseline_acc_std": np.nan,
                    "baseline_f1": np.nan,
                    "baseline_f1_std": np.nan,
                    "step1b_acc": np.nan,
                    "step1b_acc_std": np.nan,
                    "step1b_f1": np.nan,
                    "step1b_f1_std": np.nan,
                    "step1b_gate_acc": np.nan,
                    "step1b_gate_acc_std": np.nan,
                    "step1b_gate_f1": np.nan,
                    "step1b_gate_f1_std": np.nan,
                    "minirocket_acc": float(mini["minirocket_acc"]),
                    "minirocket_f1": float(mini["minirocket_f1"]),
                    "best_zspace_method": "current_evidence_insufficient",
                    "best_method_on_dataset": "minirocket_only_currently_closed",
                    "relative_gain_vs_baseline": np.nan,
                    "relative_gain_vs_minirocket": np.nan,
                }
            )
            for step_variant in ["step1b", "step1b_gate"]:
                mech_rows.append(
                    {
                        "dataset": dataset,
                        "protocol_type": protocol_type,
                        "n_seeds": 0,
                        "step_variant": step_variant,
                        "flip_rate": np.nan,
                        "flip_rate_std": np.nan,
                        "margin_drop_median": np.nan,
                        "margin_drop_median_std": np.nan,
                        "knn_intrusion_rate": np.nan,
                        "knn_intrusion_rate_std": np.nan,
                        "real_knn_intrusion_rate": np.nan,
                        "real_knn_intrusion_rate_std": np.nan,
                        "delta_intrusion": np.nan,
                        "delta_intrusion_std": np.nan,
                        "n_aug_used_for_mech": np.nan,
                        "n_aug_used_for_mech_std": np.nan,
                        "dir_profile_summary": "current evidence insufficient",
                        "remark": "当前长 EEG 主线冻结结果尚未闭环，不写结论",
                    }
                )
            continue

        df_perf = pd.read_csv(perf_path)
        df_mech = pd.read_csv(mech_path)
        n_seeds = int(len(df_perf))

        baseline_acc_mean, baseline_acc_std = _stats(df_perf["baseline_acc"])
        baseline_f1_mean, baseline_f1_std = _stats(df_perf["baseline_f1"])
        step1b_acc_mean, step1b_acc_std = _stats(df_perf["step1b_acc"])
        step1b_f1_mean, step1b_f1_std = _stats(df_perf["step1b_f1"])
        step1b_gate_acc_mean, step1b_gate_acc_std = _stats(df_perf["step1b_gate_acc"])
        step1b_gate_f1_mean, step1b_gate_f1_std = _stats(df_perf["step1b_gate_f1"])

        best_zspace_values = {
            "baseline": baseline_f1_mean,
            "step1b": step1b_f1_mean,
            "step1b_gate": step1b_gate_f1_mean,
        }
        best_zspace_method = _best_method_label(best_zspace_values)
        best_zspace_f1 = float(best_zspace_values[best_zspace_method])

        mini_row = df_mini[df_mini["dataset"] == dataset]
        if mini_row.empty:
            raise RuntimeError(f"Missing MiniROCKET official row for dataset={dataset}")
        mini = mini_row.iloc[0]
        minirocket_acc = float(mini["minirocket_acc"])
        minirocket_f1 = float(mini["minirocket_f1"])

        best_method_values = {
            "baseline": baseline_f1_mean,
            "step1b": step1b_f1_mean,
            "step1b_gate": step1b_gate_f1_mean,
            "minirocket": minirocket_f1,
        }
        perf_rows.append(
            {
                "dataset": dataset,
                "protocol_type": str(df_perf["protocol_type"].iloc[0]),
                "protocol_note": str(df_perf["protocol_note"].iloc[0]),
                "n_seeds": n_seeds,
                "baseline_acc": baseline_acc_mean,
                "baseline_acc_std": baseline_acc_std,
                "baseline_f1": baseline_f1_mean,
                "baseline_f1_std": baseline_f1_std,
                "step1b_acc": step1b_acc_mean,
                "step1b_acc_std": step1b_acc_std,
                "step1b_f1": step1b_f1_mean,
                "step1b_f1_std": step1b_f1_std,
                "step1b_gate_acc": step1b_gate_acc_mean,
                "step1b_gate_acc_std": step1b_gate_acc_std,
                "step1b_gate_f1": step1b_gate_f1_mean,
                "step1b_gate_f1_std": step1b_gate_f1_std,
                "minirocket_acc": minirocket_acc,
                "minirocket_f1": minirocket_f1,
                "best_zspace_method": best_zspace_method,
                "best_method_on_dataset": _best_method_label(best_method_values),
                "relative_gain_vs_baseline": best_zspace_f1 - baseline_f1_mean,
                "relative_gain_vs_minirocket": best_zspace_f1 - minirocket_f1,
            }
        )

        for step_variant in ["step1b", "step1b_gate"]:
            df_v = df_mech[df_mech["step_variant"] == step_variant].copy()
            flip_mean, flip_std = _stats(df_v["flip_rate"])
            margin_mean, margin_std = _stats(df_v["margin_drop_median"])
            intr_mean, intr_std = _stats(df_v["knn_intrusion_rate"])
            real_intr_mean, real_intr_std = _stats(df_v["real_knn_intrusion_rate"])
            delta_intr_mean, delta_intr_std = _stats(df_v["delta_intrusion"])
            aug_mean, aug_std = _stats(df_v["n_aug_used_for_mech"])
            if n_seeds == 1:
                dir_summary = str(df_v["dir_profile_summary"].iloc[0])
            else:
                uniq = sorted(set(df_v["dir_profile_summary"].astype(str).tolist()))
                dir_summary = " | ".join(uniq[:3])
            mech_rows.append(
                {
                    "dataset": dataset,
                    "protocol_type": str(df_perf["protocol_type"].iloc[0]),
                    "n_seeds": n_seeds,
                    "step_variant": step_variant,
                    "flip_rate": flip_mean,
                    "flip_rate_std": flip_std,
                    "margin_drop_median": margin_mean,
                    "margin_drop_median_std": margin_std,
                    "knn_intrusion_rate": intr_mean,
                    "knn_intrusion_rate_std": intr_std,
                    "real_knn_intrusion_rate": real_intr_mean,
                    "real_knn_intrusion_rate_std": real_intr_std,
                    "delta_intrusion": delta_intr_mean,
                    "delta_intrusion_std": delta_intr_std,
                    "n_aug_used_for_mech": aug_mean,
                    "n_aug_used_for_mech_std": aug_std,
                    "dir_profile_summary": dir_summary,
                    "remark": _remark_for_row(
                        step_variant=step_variant,
                        baseline_f1=baseline_f1_mean,
                        step1b_f1=step1b_f1_mean,
                        step1b_gate_f1=step1b_gate_f1_mean,
                        flip_rate=flip_mean,
                        delta_intrusion=delta_intr_mean,
                        n_seeds=n_seeds,
                    ),
                }
            )

    perf_df = pd.DataFrame(perf_rows)
    mech_df = pd.DataFrame(mech_rows)
    perf_df.to_csv(os.path.join(out_dir, "main_performance_table.csv"), index=False)
    mech_df.to_csv(os.path.join(out_dir, "mechanism_diagnosis_table.csv"), index=False)


if __name__ == "__main__":
    main()
