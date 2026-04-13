#!/usr/bin/env python
"""Build focused diagnostic tables for the frozen Phase15 mainline.

Outputs:
- direction_health_table.csv
- gate_mechanism_delta_table.csv
- highdim_risk_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from statistics import pstdev
from typing import Dict, List

import pandas as pd


CLOSED_DATASETS = [
    "seed1",
    "har",
    "natops",
    "fingermovements",
    "selfregulationscp1",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_mech_rows(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, object]] = []
        for row in reader:
            rows.append(
                {
                    "direction_id": int(row["direction_id"]),
                    "usage": float(row["usage"]),
                    "accept_rate": float(row["accept_rate"]),
                    "flip_rate": float(row["flip_rate"]),
                    "margin_drop_median": float(row["margin_drop_median"]),
                    "n_gen": int(row["n_gen"]),
                    "n_acc": int(row["n_acc"]),
                    "n_flip_eval": int(row["n_flip_eval"]),
                }
            )
    return rows


def _dataset_base(seedfamily_root: str, fixedsplit_root: str, dataset: str) -> str:
    if dataset == "seed1":
        return os.path.join(seedfamily_root, dataset, "seed3")
    return os.path.join(fixedsplit_root, dataset, "seed3")


def _usage_entropy(usages: List[float]) -> float:
    return float(-sum(u * math.log(u + 1e-12) for u in usages))


def _worst_dir(rows: List[Dict[str, object]]) -> Dict[str, object]:
    return max(
        rows,
        key=lambda r: (
            float(r["flip_rate"]),
            -float(r["accept_rate"]),
            float(r["margin_drop_median"]),
        ),
    )


def _gate_diag_label(delta_trial_f1: float, delta_flip: float, delta_margin: float, delta_intrusion: float) -> str:
    mech_improves = (
        (delta_flip < -1e-6)
        or (delta_margin > 1e-6)
        or (delta_intrusion < -1e-6)
    )
    acc_improves = delta_trial_f1 > 1e-6
    if mech_improves and not acc_improves:
        return "a_mech_improves_but_accuracy_not"
    if mech_improves and acc_improves:
        return "c_mech_and_accuracy_improve"
    if (not mech_improves) and acc_improves:
        return "mixed_accuracy_up_mech_weak"
    return "b_no_clear_mech_improvement"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixedsplit-root",
        type=str,
        default="out/phase15_mainline_freeze_20260319_fixedsplit",
    )
    parser.add_argument(
        "--seedfamily-root",
        type=str,
        default="out/phase15_mainline_freeze_20260319_seedfamily",
    )
    parser.add_argument(
        "--formal-main-performance-csv",
        type=str,
        default="out/phase15_mainline_freeze_20260319_formal/main_performance_table.csv",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="out/phase15_mainline_diagnostics_20260319",
    )
    args = parser.parse_args()

    _ensure_dir(args.out_root)
    perf_df = pd.read_csv(args.formal_main_performance_csv)
    perf_map = {str(r["dataset"]): r for _, r in perf_df.iterrows()}

    direction_rows: List[Dict[str, object]] = []
    gate_rows: List[Dict[str, object]] = []
    highdim_rows: List[Dict[str, object]] = []

    for dataset in CLOSED_DATASETS:
        seed_dir = _dataset_base(args.seedfamily_root, args.fixedsplit_root, dataset)

        a_meta = _load_json(os.path.join(seed_dir, "A_baseline", "run_meta.json"))
        b_meta = _load_json(os.path.join(seed_dir, "B_step1b", "run_meta.json"))
        c_meta = _load_json(os.path.join(seed_dir, "C_step1b_gate", "run_meta.json"))
        a_metrics = _load_json(os.path.join(seed_dir, "A_baseline", "metrics.json"))
        b_metrics = _load_json(os.path.join(seed_dir, "B_step1b", "metrics.json"))
        c_metrics = _load_json(os.path.join(seed_dir, "C_step1b_gate", "metrics.json"))
        b_mech_rows = _load_mech_rows(os.path.join(seed_dir, "B_step1b", "mech_table.csv"))
        c_mech_rows = _load_mech_rows(os.path.join(seed_dir, "C_step1b_gate", "mech_table.csv"))

        for step_variant, meta, mech_rows in [
            ("step1b", b_meta, b_mech_rows),
            ("step1b_gate", c_meta, c_mech_rows),
        ]:
            usages = [float(r["usage"]) for r in mech_rows]
            accepts = [float(r["accept_rate"]) for r in mech_rows]
            flips = [float(r["flip_rate"]) for r in mech_rows]
            margins = [float(r["margin_drop_median"]) for r in mech_rows]
            worst = _worst_dir(mech_rows)
            aug = meta.get("augmentation", {})
            mix = aug.get("mixing_stats", {})
            direction_rows.append(
                {
                    "dataset": dataset,
                    "protocol_type": meta.get("protocol_type"),
                    "step_variant": step_variant,
                    "feature_dim": int(meta.get("feature_dim", 0)),
                    "k_dir": int(aug.get("k_dir", 0)),
                    "subset_size": int(aug.get("subset_size", 0)),
                    "aug_total_count": int(aug.get("aug_total_count", 0)),
                    "aug_per_trial_mean": float(aug.get("aug_per_trial_mean", 0.0)),
                    "aug_per_trial_std": float(aug.get("aug_per_trial_std", 0.0)),
                    "train_selected_aug_ratio": float(meta.get("train_selected_aug_ratio", 0.0)),
                    "final_accept_rate": float(meta.get("final_accept_rate", 1.0)),
                    "direction_usage_min": float(min(usages)),
                    "direction_usage_max": float(max(usages)),
                    "direction_usage_std": float(pstdev(usages)),
                    "direction_usage_entropy": _usage_entropy(usages),
                    "dirs_pos_margin": int(sum(v > 0 for v in margins)),
                    "dirs_neg_margin": int(sum(v < 0 for v in margins)),
                    "dirs_flip_ge_0p02": int(sum(v >= 0.02 for v in flips)),
                    "dirs_accept_lt_0p90": int(sum(v < 0.9 for v in accepts)),
                    "worst_dir_id": int(worst["direction_id"]),
                    "worst_dir_accept_rate": float(worst["accept_rate"]),
                    "worst_dir_flip_rate": float(worst["flip_rate"]),
                    "worst_dir_margin_drop": float(worst["margin_drop_median"]),
                    "mix_mean_abs_ai": float(mix.get("mean_abs_ai", 0.0)),
                    "mix_avg_subset_size": float(mix.get("avg_subset_size", 0.0)),
                }
            )

        b_mech = b_meta.get("mech", {})
        c_mech = c_meta.get("mech", {})
        gate = c_meta.get("gate_apply", {})
        delta_trial_f1 = float(c_metrics["trial_macro_f1"]) - float(b_metrics["trial_macro_f1"])
        delta_window_f1 = float(c_metrics["window_macro_f1"]) - float(b_metrics["window_macro_f1"])
        delta_flip = float(c_mech.get("flip_rate", 0.0)) - float(b_mech.get("flip_rate", 0.0))
        delta_margin = float(c_mech.get("margin_drop_median", 0.0)) - float(b_mech.get("margin_drop_median", 0.0))
        delta_intrusion = float(c_mech.get("knn_intrusion_rate", 0.0)) - float(b_mech.get("knn_intrusion_rate", 0.0))
        gate2 = gate.get("gate2", {})
        tau_src_y = gate2.get("tau_src_y", {})
        tau_vals = [float(v) for v in tau_src_y.values()]

        gate_rows.append(
            {
                "dataset": dataset,
                "protocol_type": c_meta.get("protocol_type"),
                "feature_dim": int(c_meta.get("feature_dim", 0)),
                "accept_rate_gate1": float(gate.get("accept_rate_gate1", 0.0)),
                "accept_rate_gate2": float(gate.get("accept_rate_gate2", 0.0)),
                "accept_rate_final": float(gate.get("accept_rate_final", 0.0)),
                "step1b_trial_f1": float(b_metrics["trial_macro_f1"]),
                "step1b_gate_trial_f1": float(c_metrics["trial_macro_f1"]),
                "delta_trial_f1": delta_trial_f1,
                "step1b_window_f1": float(b_metrics["window_macro_f1"]),
                "step1b_gate_window_f1": float(c_metrics["window_macro_f1"]),
                "delta_window_f1": delta_window_f1,
                "step1b_flip_rate": float(b_mech.get("flip_rate", 0.0)),
                "step1b_gate_flip_rate": float(c_mech.get("flip_rate", 0.0)),
                "delta_flip": delta_flip,
                "step1b_margin_drop_median": float(b_mech.get("margin_drop_median", 0.0)),
                "step1b_gate_margin_drop_median": float(c_mech.get("margin_drop_median", 0.0)),
                "delta_margin_drop_median": delta_margin,
                "step1b_knn_intrusion_rate": float(b_mech.get("knn_intrusion_rate", 0.0)),
                "step1b_gate_knn_intrusion_rate": float(c_mech.get("knn_intrusion_rate", 0.0)),
                "delta_knn_intrusion_rate": delta_intrusion,
                "step1b_real_knn_intrusion_rate": float(b_mech.get("real_knn_intrusion_rate", 0.0)),
                "step1b_gate_real_knn_intrusion_rate": float(c_mech.get("real_knn_intrusion_rate", 0.0)),
                "delta_real_knn_intrusion_rate": float(c_mech.get("real_knn_intrusion_rate", 0.0))
                - float(b_mech.get("real_knn_intrusion_rate", 0.0)),
                "gate_label": _gate_diag_label(delta_trial_f1, delta_flip, delta_margin, delta_intrusion),
                "gate2_tau_src_min": float(min(tau_vals)) if tau_vals else 0.0,
                "gate2_tau_src_max": float(max(tau_vals)) if tau_vals else 0.0,
                "gate2_tau_src_span": (float(max(tau_vals)) - float(min(tau_vals))) if tau_vals else 0.0,
            }
        )

        perf_row = perf_map[dataset]
        highdim_rows.append(
            {
                "dataset": dataset,
                "protocol_type": a_meta.get("protocol_type"),
                "feature_dim": int(a_meta.get("feature_dim", 0)),
                "train_count_trials": int(a_meta.get("train_count_trials", 0)),
                "test_count_trials": int(a_meta.get("test_count_trials", 0)),
                "total_train_windows_used": int(a_meta.get("total_train_windows_used", 0)),
                "per_trial_windows_mean_after_cap": float(a_meta.get("per_trial_windows_mean_after_cap", 0.0)),
                "per_trial_windows_std_after_cap": float(a_meta.get("per_trial_windows_std_after_cap", 0.0)),
                "baseline_trial_f1": float(a_metrics["trial_macro_f1"]),
                "baseline_window_f1": float(a_metrics["window_macro_f1"]),
                "baseline_trial_window_gap": float(a_metrics["trial_macro_f1"]) - float(a_metrics["window_macro_f1"]),
                "step1b_trial_f1": float(b_metrics["trial_macro_f1"]),
                "step1b_window_f1": float(b_metrics["window_macro_f1"]),
                "step1b_trial_window_gap": float(b_metrics["trial_macro_f1"]) - float(b_metrics["window_macro_f1"]),
                "step1b_gate_trial_f1": float(c_metrics["trial_macro_f1"]),
                "step1b_gate_window_f1": float(c_metrics["window_macro_f1"]),
                "step1b_gate_trial_window_gap": float(c_metrics["trial_macro_f1"]) - float(c_metrics["window_macro_f1"]),
                "minirocket_f1": float(perf_row["minirocket_f1"]),
                "relative_gap_vs_minirocket": float(perf_row["baseline_f1"]) - float(perf_row["minirocket_f1"]),
                "step1b_aug_ratio_selected": float(b_meta.get("train_selected_aug_ratio", 0.0)),
                "step1b_gate_aug_ratio_selected": float(c_meta.get("train_selected_aug_ratio", 0.0)),
                "gate2_tau_src_span": (float(max(tau_vals)) - float(min(tau_vals))) if tau_vals else 0.0,
            }
        )

    pd.DataFrame(direction_rows).to_csv(
        os.path.join(args.out_root, "direction_health_table.csv"),
        index=False,
    )
    pd.DataFrame(gate_rows).to_csv(
        os.path.join(args.out_root, "gate_mechanism_delta_table.csv"),
        index=False,
    )
    pd.DataFrame(highdim_rows).to_csv(
        os.path.join(args.out_root, "highdim_risk_summary.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
