#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ACT_ROOT = PROJECT_ROOT / "standalone_projects" / "ACT_ManifoldBridge"
PHASE1_ROOT = ACT_ROOT / "results" / "csta_external_baselines_phase1" / "resnet1d_s123"
PHASE2_ROOT = ACT_ROOT / "results" / "csta_external_baselines_phase2" / "resnet1d_s123"
SAMPLING_ROOT = ACT_ROOT / "results" / "csta_sampling_v1" / "resnet1d_s123"

SAMPLING_ARMS = [
    "csta_top1_current",
    "csta_topk_softmax_tau_0.05",
    "csta_topk_softmax_tau_0.10",
    "csta_topk_softmax_tau_0.20",
    "csta_topk_uniform_top5",
]


def _load_df(root: Path) -> pd.DataFrame:
    csv = root / "per_seed_external.csv"
    if not csv.is_file():
        return pd.DataFrame()
    return pd.read_csv(csv)


def _mean_for(df: pd.DataFrame, *, dataset: str, method: str, seed: int | None = None) -> float:
    if df.empty:
        return float("nan")
    sub = df[(df["dataset"] == dataset) & (df["method"] == method)].copy()
    if seed is not None:
        sub = sub[sub["seed"] == seed]
    if sub.empty:
        return float("nan")
    return float(sub["aug_f1"].mean())


def _paired_gain_vs_phase1_no_aug(df_sampling: pd.DataFrame, df_phase1: pd.DataFrame, method: str) -> float:
    gains: list[float] = []
    sub = df_sampling[df_sampling["method"] == method]
    for _, row in sub.iterrows():
        base = _mean_for(
            df_phase1,
            dataset=str(row["dataset"]),
            method="no_aug",
            seed=int(row["seed"]),
        )
        if np.isfinite(base):
            gains.append(float(row["aug_f1"]) - base)
    return float(np.mean(gains)) if gains else float("nan")


def _metric_mean(sub: pd.DataFrame, column: str) -> float:
    if column not in sub.columns:
        return float("nan")
    return float(pd.to_numeric(sub[column], errors="coerce").mean())


def main() -> None:
    df1 = _load_df(PHASE1_ROOT)
    df2 = _load_df(PHASE2_ROOT)
    dfs = _load_df(SAMPLING_ROOT)
    if dfs.empty:
        raise SystemExit(f"Sampling results not found at {SAMPLING_ROOT / 'per_seed_external.csv'}")
    if df1.empty:
        raise SystemExit(f"Phase 1 locked refs not found at {PHASE1_ROOT / 'per_seed_external.csv'}")

    SAMPLING_ROOT.mkdir(parents=True, exist_ok=True)
    datasets = sorted(dfs["dataset"].dropna().unique())

    identity_rows = []
    for dataset in datasets:
        phase1_no_aug = _mean_for(df1, dataset=dataset, method="no_aug")
        phase1_top1 = _mean_for(df1, dataset=dataset, method="csta_top1_current")
        sampling_top1 = _mean_for(dfs, dataset=dataset, method="csta_top1_current")
        identity_rows.append(
            {
                "dataset": dataset,
                "phase1_no_aug": phase1_no_aug,
                "phase1_top1": phase1_top1,
                "sampling_top1": sampling_top1,
                "top1_delta_sampling_minus_phase1": sampling_top1 - phase1_top1,
            }
        )
    identity = pd.DataFrame(identity_rows)
    identity.to_csv(SAMPLING_ROOT / "final_audit_identity.csv", index=False)

    perf_rows = []
    for method in SAMPLING_ARMS:
        sub = dfs[dfs["method"] == method].copy()
        if sub.empty:
            continue
        perf_rows.append(
            {
                "method": method,
                "mean_f1": float(sub["aug_f1"].mean()),
                "mean_gain_vs_phase1_no_aug": _paired_gain_vs_phase1_no_aug(dfs, df1, method),
                "gamma_requested_mean": _metric_mean(sub, "gamma_requested_mean"),
                "gamma_used_mean": _metric_mean(sub, "gamma_used_mean"),
                "safe_clip_rate": _metric_mean(sub, "safe_clip_rate"),
                "safe_radius_ratio_mean": _metric_mean(sub, "safe_radius_ratio_mean"),
                "template_usage_entropy": _metric_mean(sub, "template_usage_entropy"),
                "top_template_concentration": _metric_mean(sub, "top_template_concentration"),
                "transport_error_logeuc_mean": _metric_mean(sub, "transport_error_logeuc_mean"),
            }
        )
    perf = pd.DataFrame(perf_rows).sort_values("mean_f1", ascending=False)
    perf.to_csv(SAMPLING_ROOT / "final_audit_sampling_policy.csv", index=False)

    if perf.empty:
        raise SystemExit("No sampling policy rows found.")
    best_method = str(perf.iloc[0]["method"])
    best_f1 = float(perf.iloc[0]["mean_f1"])
    external_sources = [
        ("no_aug", df1),
        ("dba_sameclass", df1),
        ("wdba_sameclass", df2),
        ("spawner_sameclass_style", df2),
        ("raw_aug_window_slicing", df2),
        ("raw_mixup", df1),
    ]
    ext_rows = [
        {
            "method": best_method,
            "mean_f1": best_f1,
            "gap_sampling_best_minus_method": 0.0,
            "source": "sampling_v1",
        }
    ]
    for method, source_df in external_sources:
        sub = source_df[source_df["method"] == method] if not source_df.empty else pd.DataFrame()
        if sub.empty:
            continue
        mean_f1 = float(sub["aug_f1"].mean())
        ext_rows.append(
            {
                "method": method,
                "mean_f1": mean_f1,
                "gap_sampling_best_minus_method": best_f1 - mean_f1,
                "source": "phase1_locked" if source_df is df1 else "phase2_locked",
            }
        )
    ext = pd.DataFrame(ext_rows).sort_values("mean_f1", ascending=False)
    ext.to_csv(SAMPLING_ROOT / "final_audit_external_gap.csv", index=False)

    print("### Part 1: Identity Check (Sampling V1 vs Phase 1 Locked)")
    print(identity.to_string(index=False))
    print("\n### Part 2: Sampling Policy Ablation")
    print(perf.to_string(index=False))
    print("\n### Part 3: External Baseline Gap")
    print(ext.to_string(index=False))


if __name__ == "__main__":
    main()
