#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_ROOT = PROJECT_ROOT / "results" / "mba_vs_rc4_census_v1" / "resnet1d_sharedbudget_s123"
DEFAULT_LOCKED_ROOT = PROJECT_ROOT / "results" / "mba_vs_rc4_census_v1" / "resnet1d_sharedbudget_s123"
DEFAULT_STEP1_ROOT = PROJECT_ROOT / "results" / "mba_core_rc4_fused_step1" / "resnet1d_sharedbudget_s123"
DEFAULT_SPECTRAL_ROOT = PROJECT_ROOT / "results" / "mba_core_spectral_osf_step1" / "resnet1d_sharedbudget_s123"
DEFAULT_MULTITEMPLATE_ROOT = PROJECT_ROOT / "results" / "mba_core_multitemplate_osf_v1" / "resnet1d_sharedbudget_s123"
DEFAULT_REF_ROOT = PROJECT_ROOT / "results" / "paper_matrix_v2_final" / "phase_all" / "resnet1d"

LOCKED_ARMS = {"baseline_ce", "mba_core_lraes", "mba_feedback_lraes", "rc4_osf"}
MULTITEMPLATE_ARMS = {
    "mba_core_zpia_top1_pool",
    "mba_core_zpia_multidir_pool",
    "mba_core_rc4_multiz_fused_concat",
}
PROGRESSIVE_ARMS = [
    "progressive_zpia_only",
    "progressive_zpia_osf",
    "random_progressive_osf",
    "progressive_zpia_only_pure",
    "progressive_zpia_pure_cumulative",
    "progressive_zpia_exposure_matched",
    "progressive_zpia_osf_highdose",
]
METHOD_ARMS = [
    "baseline_ce",
    "mba_core_lraes",
    "mba_feedback_lraes",
    "rc4_osf",
    "mba_core_rc4_fused_concat",
    "mba_core_spectral_osf_concat",
    "mba_core_zpia_top1_pool",
    "mba_core_zpia_multidir_pool",
    "mba_core_rc4_multiz_fused_concat",
    *PROGRESSIVE_ARMS,
]


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_job_rows(root: Path) -> pd.DataFrame:
    jobs_path = root / "jobs_manifest.csv"
    if not jobs_path.is_file():
        raise FileNotFoundError(f"Missing jobs manifest: {jobs_path}")
    df = pd.read_csv(jobs_path)
    if df.empty:
        raise ValueError(f"No jobs found in {jobs_path}")
    return df


def _load_run_manifest(path: Path) -> Dict:
    if not path.is_file():
        return {}
    return _read_json(path)


def _load_actual_rows(root: Path, jobs_df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for _, job in jobs_df.iterrows():
        if str(job.get("status", "")) != "success":
            continue
        result_path = Path(job["results_path"])
        manifest = _load_run_manifest(Path(job["manifest_path"]))
        if not result_path.is_file():
            continue
        df = pd.read_csv(result_path)
        for _, row in df.iterrows():
            rec = row.to_dict()
            rec["arm"] = str(job["arm"])
            rec["job_status"] = str(job["status"])
            rec["physical_gpu_id"] = job.get("gpu_id")
            for key, value in manifest.items():
                if key not in rec:
                    rec[key] = value
            records.append(rec)
    if not records:
        raise ValueError(f"No actual result rows found under {root}")
    return pd.DataFrame(records)


def _build_baseline_rows(actual_df: pd.DataFrame, tol: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = actual_df.groupby(["dataset", "seed"], dropna=False)
    for (dataset, seed), g in grouped:
        base_vals = pd.to_numeric(g["base_f1"], errors="coerce").dropna()
        if base_vals.empty:
            continue
        span = float(base_vals.max() - base_vals.min())
        mean_base = float(base_vals.mean())
        template = g.iloc[0].to_dict()
        template.update(
            {
                "arm": "baseline_ce",
                "f1": mean_base,
                "base_f1": mean_base,
                "act_f1": mean_base,
                "gain": 0.0,
                "f1_gain_pct": 0.0,
                "baseline_source_count": int(len(base_vals)),
                "baseline_drift_span": span,
                "baseline_drift_flag": bool(span > tol),
                "source_arms": ",".join(sorted(g["arm"].astype(str).unique().tolist())),
            }
        )
        rows.append(template)
    return pd.DataFrame(rows)


def _standardize_actual_rows(actual_df: pd.DataFrame, tol: float) -> pd.DataFrame:
    actual_df = actual_df.copy()
    actual_df["f1"] = pd.to_numeric(actual_df["act_f1"], errors="coerce")
    actual_df["baseline_source_count"] = 1
    actual_df["baseline_drift_span"] = np.nan
    actual_df["baseline_drift_flag"] = False
    actual_df["source_arms"] = actual_df["arm"]
    baseline_rows = _build_baseline_rows(actual_df, tol)
    all_rows = pd.concat([actual_df, baseline_rows], ignore_index=True, sort=False)
    return all_rows


def _baseline_drift_report(actual_df: pd.DataFrame, tol: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = actual_df.groupby(["dataset", "seed"], dropna=False)
    for (dataset, seed), g in grouped:
        by_arm = {row["arm"]: float(row["base_f1"]) for _, row in g.iterrows()}
        vals = [v for v in by_arm.values() if np.isfinite(v)]
        span = float(max(vals) - min(vals)) if vals else np.nan
        rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "base_f1_mba_core_lraes": by_arm.get("mba_core_lraes", np.nan),
                "base_f1_mba_feedback_lraes": by_arm.get("mba_feedback_lraes", np.nan),
                "base_f1_rc4_osf": by_arm.get("rc4_osf", np.nan),
                "baseline_span": span,
                "baseline_drift_tol": tol,
                "baseline_drift_flag": bool(np.isfinite(span) and span > tol),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "seed"]).reset_index(drop=True)


def _mba_repro_check(actual_df: pd.DataFrame, ref_root: Path) -> pd.DataFrame:
    mba_df = actual_df[actual_df["arm"] == "mba_core_lraes"].copy()
    rows: List[Dict[str, object]] = []
    for _, row in mba_df.iterrows():
        dataset = str(row["dataset"])
        seed = int(row["seed"])
        ref_path = ref_root / dataset / f"s{seed}_lraes_safe" / "final_results.csv"
        ref_found = ref_path.is_file()
        hist_f1 = np.nan
        if ref_found:
            ref_df = pd.read_csv(ref_path)
            if not ref_df.empty and "act_f1" in ref_df.columns:
                hist_f1 = float(ref_df.iloc[0]["act_f1"])
        curr_f1 = float(row["act_f1"])
        delta = curr_f1 - hist_f1 if np.isfinite(hist_f1) else np.nan
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "reference_found": bool(ref_found),
                "historical_path": str(ref_path),
                "historical_mba_act_f1": hist_f1,
                "current_mba_core_act_f1": curr_f1,
                "delta_repro": delta,
                "abs_delta_repro": abs(delta) if np.isfinite(delta) else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "seed",
                "reference_found",
                "historical_path",
                "historical_mba_act_f1",
                "current_mba_core_act_f1",
                "delta_repro",
                "abs_delta_repro",
            ]
        )
    return pd.DataFrame(rows).sort_values(["dataset", "seed"]).reset_index(drop=True)


def _dataset_summary(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        per_seed_df.groupby(["dataset", "arm"], dropna=False)
        .agg(
            n=("f1", "count"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_gain=("gain", "mean"),
            std_gain=("gain", "std"),
        )
        .reset_index()
        .sort_values(["dataset", "arm"])
    )
    return summary


def _load_reference_per_seed_rows(root: Path, keep_arms: set[str]) -> pd.DataFrame:
    candidates = [
        root / "_summary" / "per_seed_arms.csv",
        root / "_summary" / "per_seed_arms_step1.csv",
    ]
    per_seed_path = next((path for path in candidates if path.is_file()), None)
    if per_seed_path is None:
        raise FileNotFoundError(f"Missing per-seed summary under {root}")
    df = pd.read_csv(per_seed_path)
    return df[df["arm"].astype(str).isin(keep_arms)].copy()


def _try_load_reference_per_seed_rows(root: Path, keep_arms: set[str]) -> pd.DataFrame:
    try:
        return _load_reference_per_seed_rows(root, keep_arms)
    except FileNotFoundError:
        return pd.DataFrame()


def _load_locked_per_seed_rows(locked_root: Path) -> pd.DataFrame:
    return _load_reference_per_seed_rows(locked_root, LOCKED_ARMS)


def _gap_attribution_view(dataset_summary: pd.DataFrame, baseline_drift_df: pd.DataFrame, mba_repro_df: pd.DataFrame) -> pd.DataFrame:
    pivot = dataset_summary.pivot(index="dataset", columns="arm", values="mean_f1").reset_index()
    for col in METHOD_ARMS:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["mba_core_vs_baseline"] = pivot["mba_core_lraes"] - pivot["baseline_ce"]
    pivot["mba_feedback_vs_mba_core"] = pivot["mba_feedback_lraes"] - pivot["mba_core_lraes"]
    pivot["rc4_vs_mba_feedback"] = pivot["rc4_osf"] - pivot["mba_feedback_lraes"]
    pivot["rc4_vs_mba_core"] = pivot["rc4_osf"] - pivot["mba_core_lraes"]
    pivot["mba_core_rc4_fused_vs_mba_core"] = pivot["mba_core_rc4_fused_concat"] - pivot["mba_core_lraes"]
    pivot["mba_core_rc4_fused_vs_rc4_osf"] = pivot["mba_core_rc4_fused_concat"] - pivot["rc4_osf"]
    pivot["spectral_vs_mba_core"] = pivot["mba_core_spectral_osf_concat"] - pivot["mba_core_lraes"]
    pivot["spectral_vs_rc4_osf"] = pivot["mba_core_spectral_osf_concat"] - pivot["rc4_osf"]
    pivot["spectral_vs_rc4_fused"] = pivot["mba_core_spectral_osf_concat"] - pivot["mba_core_rc4_fused_concat"]
    pivot["zpia_multidir_vs_zpia_top1"] = pivot["mba_core_zpia_multidir_pool"] - pivot["mba_core_zpia_top1_pool"]
    pivot["rc4_multiz_vs_mba_core"] = pivot["mba_core_rc4_multiz_fused_concat"] - pivot["mba_core_lraes"]
    pivot["rc4_multiz_vs_rc4_fused"] = (
        pivot["mba_core_rc4_multiz_fused_concat"] - pivot["mba_core_rc4_fused_concat"]
    )
    pivot["rc4_multiz_vs_spectral"] = (
        pivot["mba_core_rc4_multiz_fused_concat"] - pivot["mba_core_spectral_osf_concat"]
    )
    pivot["progressive_osf_vs_progressive_zpia_only"] = (
        pivot["progressive_zpia_osf"] - pivot["progressive_zpia_only"]
    )
    pivot["progressive_osf_vs_zpia_top1"] = (
        pivot["progressive_zpia_osf"] - pivot["mba_core_zpia_top1_pool"]
    )
    pivot["random_progressive_osf_vs_progressive_osf"] = (
        pivot["random_progressive_osf"] - pivot["progressive_zpia_osf"]
    )
    pivot["progressive_zpia_only_pure_vs_progressive_zpia_only"] = (
        pivot["progressive_zpia_only_pure"] - pivot["progressive_zpia_only"]
    )
    pivot["progressive_zpia_pure_cumulative_vs_zpia_top1"] = (
        pivot["progressive_zpia_pure_cumulative"] - pivot["mba_core_zpia_top1_pool"]
    )
    pivot["progressive_zpia_exposure_matched_vs_zpia_top1"] = (
        pivot["progressive_zpia_exposure_matched"] - pivot["mba_core_zpia_top1_pool"]
    )
    pivot["progressive_zpia_osf_highdose_vs_progressive_zpia_pure_cumulative"] = (
        pivot["progressive_zpia_osf_highdose"] - pivot["progressive_zpia_pure_cumulative"]
    )
    pivot["progressive_zpia_osf_highdose_vs_zpia_top1"] = (
        pivot["progressive_zpia_osf_highdose"] - pivot["mba_core_zpia_top1_pool"]
    )
    pivot["progressive_zpia_osf_highdose_vs_rc4_multiz"] = (
        pivot["progressive_zpia_osf_highdose"] - pivot["mba_core_rc4_multiz_fused_concat"]
    )

    drift_max = (
        baseline_drift_df.groupby("dataset", dropna=False)["baseline_span"].max().rename("baseline_drift_span_max").reset_index()
    )
    repro = (
        mba_repro_df.groupby("dataset", dropna=False)["abs_delta_repro"].mean().rename("mba_repro_abs_delta_mean").reset_index()
    )
    out = pivot.merge(drift_max, on="dataset", how="left").merge(repro, on="dataset", how="left")
    return out.sort_values("dataset").reset_index(drop=True)


def _mba_core_rc4_vs_locked(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = dataset_summary.pivot(index="dataset", columns="arm", values="mean_f1").reset_index()
    for col in ["baseline_ce", "mba_core_lraes", "mba_feedback_lraes", "rc4_osf", "mba_core_rc4_fused_concat"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["mba_core_rc4_fused_vs_mba_core"] = pivot["mba_core_rc4_fused_concat"] - pivot["mba_core_lraes"]
    pivot["mba_core_rc4_fused_vs_rc4_osf"] = pivot["mba_core_rc4_fused_concat"] - pivot["rc4_osf"]
    pivot["mba_core_rc4_fused_vs_mba_feedback"] = pivot["mba_core_rc4_fused_concat"] - pivot["mba_feedback_lraes"]
    return pivot.sort_values("dataset").reset_index(drop=True)


def _spectral_osf_vs_refs(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = dataset_summary.pivot(index="dataset", columns="arm", values="mean_f1").reset_index()
    for col in METHOD_ARMS:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["spectral_vs_mba_core_lraes"] = pivot["mba_core_spectral_osf_concat"] - pivot["mba_core_lraes"]
    pivot["spectral_vs_rc4_osf"] = pivot["mba_core_spectral_osf_concat"] - pivot["rc4_osf"]
    pivot["spectral_vs_rc4_fused"] = pivot["mba_core_spectral_osf_concat"] - pivot["mba_core_rc4_fused_concat"]
    return pivot.sort_values("dataset").reset_index(drop=True)


def _multitemplate_vs_refs(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = dataset_summary.pivot(index="dataset", columns="arm", values="mean_f1").reset_index()
    for col in METHOD_ARMS:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["zpia_multidir_vs_zpia_top1"] = pivot["mba_core_zpia_multidir_pool"] - pivot["mba_core_zpia_top1_pool"]
    pivot["rc4_multiz_vs_mba_core_lraes"] = pivot["mba_core_rc4_multiz_fused_concat"] - pivot["mba_core_lraes"]
    pivot["rc4_multiz_vs_rc4_osf"] = pivot["mba_core_rc4_multiz_fused_concat"] - pivot["rc4_osf"]
    pivot["rc4_multiz_vs_rc4_fused"] = (
        pivot["mba_core_rc4_multiz_fused_concat"] - pivot["mba_core_rc4_fused_concat"]
    )
    pivot["rc4_multiz_vs_spectral_osf"] = (
        pivot["mba_core_rc4_multiz_fused_concat"] - pivot["mba_core_spectral_osf_concat"]
    )
    pivot["progressive_osf_vs_progressive_zpia_only"] = (
        pivot["progressive_zpia_osf"] - pivot["progressive_zpia_only"]
    )
    pivot["progressive_osf_vs_zpia_top1"] = (
        pivot["progressive_zpia_osf"] - pivot["mba_core_zpia_top1_pool"]
    )
    pivot["random_progressive_osf_vs_progressive_osf"] = (
        pivot["random_progressive_osf"] - pivot["progressive_zpia_osf"]
    )
    return pivot.sort_values("dataset").reset_index(drop=True)


def _progressive_vs_refs(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = dataset_summary.pivot(index="dataset", columns="arm", values="mean_f1").reset_index()
    for col in METHOD_ARMS:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["progressive_zpia_only_vs_zpia_top1"] = (
        pivot["progressive_zpia_only"] - pivot["mba_core_zpia_top1_pool"]
    )
    pivot["progressive_zpia_osf_vs_progressive_zpia_only"] = (
        pivot["progressive_zpia_osf"] - pivot["progressive_zpia_only"]
    )
    pivot["progressive_zpia_osf_vs_zpia_top1"] = (
        pivot["progressive_zpia_osf"] - pivot["mba_core_zpia_top1_pool"]
    )
    pivot["progressive_zpia_osf_vs_rc4_multiz"] = (
        pivot["progressive_zpia_osf"] - pivot["mba_core_rc4_multiz_fused_concat"]
    )
    pivot["random_progressive_osf_vs_progressive_zpia_osf"] = (
        pivot["random_progressive_osf"] - pivot["progressive_zpia_osf"]
    )
    pivot["progressive_zpia_only_pure_vs_progressive_zpia_only"] = (
        pivot["progressive_zpia_only_pure"] - pivot["progressive_zpia_only"]
    )
    pivot["progressive_zpia_pure_cumulative_vs_zpia_top1"] = (
        pivot["progressive_zpia_pure_cumulative"] - pivot["mba_core_zpia_top1_pool"]
    )
    pivot["progressive_zpia_exposure_matched_vs_zpia_top1"] = (
        pivot["progressive_zpia_exposure_matched"] - pivot["mba_core_zpia_top1_pool"]
    )
    pivot["progressive_zpia_osf_highdose_vs_progressive_zpia_pure_cumulative"] = (
        pivot["progressive_zpia_osf_highdose"] - pivot["progressive_zpia_pure_cumulative"]
    )
    pivot["progressive_zpia_osf_highdose_vs_zpia_top1"] = (
        pivot["progressive_zpia_osf_highdose"] - pivot["mba_core_zpia_top1_pool"]
    )
    pivot["progressive_zpia_osf_highdose_vs_rc4_multiz"] = (
        pivot["progressive_zpia_osf_highdose"] - pivot["mba_core_rc4_multiz_fused_concat"]
    )
    return pivot.sort_values("dataset").reset_index(drop=True)


def _overall_summary_table(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    pivot_f1 = dataset_summary.pivot(index="dataset", columns="arm", values="mean_f1")
    arms = METHOD_ARMS
    pivot_f1 = pivot_f1[[arm for arm in arms if arm in pivot_f1.columns]]
    rows: List[Dict[str, object]] = []
    for arm in arms:
        if arm not in pivot_f1.columns:
            continue
        arm_vals = pd.to_numeric(pivot_f1[arm], errors="coerce")
        arm_mask = arm_vals.notna()
        baseline_vals = (
            pd.to_numeric(pivot_f1["baseline_ce"], errors="coerce")
            if "baseline_ce" in pivot_f1.columns
            else None
        )
        mba_core_vals = (
            pd.to_numeric(pivot_f1["mba_core_lraes"], errors="coerce")
            if "mba_core_lraes" in pivot_f1.columns
            else None
        )
        rc4_vals = (
            pd.to_numeric(pivot_f1["rc4_osf"], errors="coerce")
            if "rc4_osf" in pivot_f1.columns
            else None
        )
        baseline_mask = arm_mask & baseline_vals.notna() if baseline_vals is not None else pd.Series(False, index=pivot_f1.index)
        mba_core_mask = arm_mask & mba_core_vals.notna() if mba_core_vals is not None else pd.Series(False, index=pivot_f1.index)
        rc4_mask = arm_mask & rc4_vals.notna() if rc4_vals is not None else pd.Series(False, index=pivot_f1.index)
        rows.append(
            {
                "arm": arm,
                "n_datasets": int(arm_mask.sum()),
                "n_common_vs_baseline": int(baseline_mask.sum()),
                "n_common_vs_mba_core_lraes": int(mba_core_mask.sum()),
                "n_common_vs_rc4_osf": int(rc4_mask.sum()),
                "mean_f1_mean_over_datasets": float(arm_vals[arm_mask].mean()),
                "mean_gain_vs_baseline": (
                    float((arm_vals[baseline_mask] - baseline_vals[baseline_mask]).mean())
                    if baseline_mask.any()
                    else np.nan
                ),
                "win_count_vs_mba_core_lraes": (
                    int(((arm_vals[mba_core_mask] - mba_core_vals[mba_core_mask]) > 0).sum())
                    if mba_core_mask.any()
                    else 0
                ),
                "win_count_vs_rc4_osf": (
                    int(((arm_vals[rc4_mask] - rc4_vals[rc4_mask]) > 0).sum()) if rc4_mask.any() else 0
                ),
                "mean_delta_vs_mba_core_lraes": (
                    float((arm_vals[mba_core_mask] - mba_core_vals[mba_core_mask]).mean())
                    if mba_core_mask.any()
                    else np.nan
                ),
                "mean_delta_vs_rc4_osf": (
                    float((arm_vals[rc4_mask] - rc4_vals[rc4_mask]).mean()) if rc4_mask.any() else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _per_dataset_gap_table(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = dataset_summary.pivot(index="dataset", columns="arm", values="mean_f1").reset_index()
    for col in METHOD_ARMS:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["mba_core_lraes_minus_baseline"] = pivot["mba_core_lraes"] - pivot["baseline_ce"]
    pivot["mba_feedback_lraes_minus_mba_core_lraes"] = pivot["mba_feedback_lraes"] - pivot["mba_core_lraes"]
    pivot["rc4_osf_minus_mba_feedback_lraes"] = pivot["rc4_osf"] - pivot["mba_feedback_lraes"]
    pivot["mba_core_rc4_fused_concat_minus_mba_core_lraes"] = (
        pivot["mba_core_rc4_fused_concat"] - pivot["mba_core_lraes"]
    )
    pivot["mba_core_rc4_fused_concat_minus_rc4_osf"] = pivot["mba_core_rc4_fused_concat"] - pivot["rc4_osf"]
    pivot["mba_core_spectral_osf_concat_minus_mba_core_lraes"] = (
        pivot["mba_core_spectral_osf_concat"] - pivot["mba_core_lraes"]
    )
    pivot["mba_core_spectral_osf_concat_minus_mba_core_rc4_fused_concat"] = (
        pivot["mba_core_spectral_osf_concat"] - pivot["mba_core_rc4_fused_concat"]
    )
    pivot["zpia_multidir_pool_minus_zpia_top1_pool"] = (
        pivot["mba_core_zpia_multidir_pool"] - pivot["mba_core_zpia_top1_pool"]
    )
    pivot["mba_core_rc4_multiz_fused_concat_minus_mba_core_lraes"] = (
        pivot["mba_core_rc4_multiz_fused_concat"] - pivot["mba_core_lraes"]
    )
    pivot["mba_core_rc4_multiz_fused_concat_minus_mba_core_rc4_fused_concat"] = (
        pivot["mba_core_rc4_multiz_fused_concat"] - pivot["mba_core_rc4_fused_concat"]
    )
    pivot["mba_core_rc4_multiz_fused_concat_minus_mba_core_spectral_osf_concat"] = (
        pivot["mba_core_rc4_multiz_fused_concat"] - pivot["mba_core_spectral_osf_concat"]
    )
    pivot["progressive_zpia_osf_minus_progressive_zpia_only"] = (
        pivot["progressive_zpia_osf"] - pivot["progressive_zpia_only"]
    )
    pivot["progressive_zpia_osf_minus_mba_core_zpia_top1_pool"] = (
        pivot["progressive_zpia_osf"] - pivot["mba_core_zpia_top1_pool"]
    )
    pivot["random_progressive_osf_minus_progressive_zpia_osf"] = (
        pivot["random_progressive_osf"] - pivot["progressive_zpia_osf"]
    )
    pivot["progressive_zpia_only_pure_minus_progressive_zpia_only"] = (
        pivot["progressive_zpia_only_pure"] - pivot["progressive_zpia_only"]
    )
    pivot["progressive_zpia_pure_cumulative_minus_mba_core_zpia_top1_pool"] = (
        pivot["progressive_zpia_pure_cumulative"] - pivot["mba_core_zpia_top1_pool"]
    )
    pivot["progressive_zpia_exposure_matched_minus_mba_core_zpia_top1_pool"] = (
        pivot["progressive_zpia_exposure_matched"] - pivot["mba_core_zpia_top1_pool"]
    )
    pivot["progressive_zpia_osf_highdose_minus_progressive_zpia_pure_cumulative"] = (
        pivot["progressive_zpia_osf_highdose"] - pivot["progressive_zpia_pure_cumulative"]
    )
    pivot["progressive_zpia_osf_highdose_minus_mba_core_zpia_top1_pool"] = (
        pivot["progressive_zpia_osf_highdose"] - pivot["mba_core_zpia_top1_pool"]
    )
    pivot["progressive_zpia_osf_highdose_minus_mba_core_rc4_multiz_fused_concat"] = (
        pivot["progressive_zpia_osf_highdose"] - pivot["mba_core_rc4_multiz_fused_concat"]
    )
    keep_cols = [
        "dataset",
        "mba_core_lraes_minus_baseline",
        "mba_feedback_lraes_minus_mba_core_lraes",
        "rc4_osf_minus_mba_feedback_lraes",
        "mba_core_rc4_fused_concat_minus_mba_core_lraes",
        "mba_core_rc4_fused_concat_minus_rc4_osf",
        "mba_core_spectral_osf_concat_minus_mba_core_lraes",
        "mba_core_spectral_osf_concat_minus_mba_core_rc4_fused_concat",
        "zpia_multidir_pool_minus_zpia_top1_pool",
        "mba_core_rc4_multiz_fused_concat_minus_mba_core_lraes",
        "mba_core_rc4_multiz_fused_concat_minus_mba_core_rc4_fused_concat",
        "mba_core_rc4_multiz_fused_concat_minus_mba_core_spectral_osf_concat",
        "progressive_zpia_osf_minus_progressive_zpia_only",
        "progressive_zpia_osf_minus_mba_core_zpia_top1_pool",
        "random_progressive_osf_minus_progressive_zpia_osf",
        "progressive_zpia_only_pure_minus_progressive_zpia_only",
        "progressive_zpia_pure_cumulative_minus_mba_core_zpia_top1_pool",
        "progressive_zpia_exposure_matched_minus_mba_core_zpia_top1_pool",
        "progressive_zpia_osf_highdose_minus_progressive_zpia_pure_cumulative",
        "progressive_zpia_osf_highdose_minus_mba_core_zpia_top1_pool",
        "progressive_zpia_osf_highdose_minus_mba_core_rc4_multiz_fused_concat",
    ]
    out = pivot[keep_cols].copy()
    out = out.dropna(
        subset=[
            "mba_core_lraes_minus_baseline",
            "mba_feedback_lraes_minus_mba_core_lraes",
            "rc4_osf_minus_mba_feedback_lraes",
            "mba_core_rc4_fused_concat_minus_mba_core_lraes",
            "mba_core_rc4_fused_concat_minus_rc4_osf",
            "zpia_multidir_pool_minus_zpia_top1_pool",
            "mba_core_rc4_multiz_fused_concat_minus_mba_core_lraes",
            "mba_core_rc4_multiz_fused_concat_minus_mba_core_rc4_fused_concat",
        ]
    )
    return out.sort_values("dataset").reset_index(drop=True)


def _regime_taxonomy_table(dataset_summary: pd.DataFrame, tie_eps: float = 0.005) -> pd.DataFrame:
    pivot = dataset_summary.pivot(index="dataset", columns="arm", values="mean_f1").reset_index()
    for col in METHOD_ARMS:
        if col not in pivot.columns:
            pivot[col] = np.nan
    rows: List[Dict[str, object]] = []
    for _, row in pivot.iterrows():
        if (
            pd.isna(row["mba_core_lraes"])
            or pd.isna(row["mba_feedback_lraes"])
            or pd.isna(row["rc4_osf"])
            or pd.isna(row["mba_core_rc4_fused_concat"])
            or pd.isna(row["mba_core_zpia_top1_pool"])
            or pd.isna(row["mba_core_zpia_multidir_pool"])
            or pd.isna(row["mba_core_rc4_multiz_fused_concat"])
        ):
            continue
        ds = row["dataset"]
        mba_core = float(row["mba_core_lraes"])
        mba_feedback = float(row["mba_feedback_lraes"])
        rc4_osf = float(row["rc4_osf"])
        rc4_fused = float(row["mba_core_rc4_fused_concat"])
        spectral = float(row["mba_core_spectral_osf_concat"]) if not pd.isna(row["mba_core_spectral_osf_concat"]) else -np.inf
        zpia_top1 = float(row["mba_core_zpia_top1_pool"])
        zpia_multidir = float(row["mba_core_zpia_multidir_pool"])
        rc4_multiz = float(row["mba_core_rc4_multiz_fused_concat"])
        progressive_zpia = (
            float(row["progressive_zpia_only"]) if not pd.isna(row["progressive_zpia_only"]) else -np.inf
        )
        progressive_osf = (
            float(row["progressive_zpia_osf"]) if not pd.isna(row["progressive_zpia_osf"]) else -np.inf
        )
        random_progressive = (
            float(row["random_progressive_osf"]) if not pd.isna(row["random_progressive_osf"]) else -np.inf
        )
        progressive_pure = (
            float(row["progressive_zpia_only_pure"]) if not pd.isna(row["progressive_zpia_only_pure"]) else -np.inf
        )
        progressive_cumulative = (
            float(row["progressive_zpia_pure_cumulative"])
            if not pd.isna(row["progressive_zpia_pure_cumulative"])
            else -np.inf
        )
        progressive_exposure = (
            float(row["progressive_zpia_exposure_matched"])
            if not pd.isna(row["progressive_zpia_exposure_matched"])
            else -np.inf
        )
        progressive_highdose = (
            float(row["progressive_zpia_osf_highdose"])
            if not pd.isna(row["progressive_zpia_osf_highdose"])
            else -np.inf
        )
        delta_core = rc4_fused - mba_core
        delta_rc4 = rc4_fused - rc4_osf
        delta_spectral_core = spectral - mba_core if np.isfinite(spectral) else np.nan
        delta_spectral_rc4 = spectral - rc4_fused if np.isfinite(spectral) else np.nan
        delta_multiz_core = rc4_multiz - mba_core
        delta_multiz_rc4 = rc4_multiz - rc4_fused
        delta_zpia_multi = zpia_multidir - zpia_top1
        delta_progressive_osf = (
            progressive_osf - progressive_zpia if np.isfinite(progressive_osf) and np.isfinite(progressive_zpia) else np.nan
        )
        delta_progressive_top1 = progressive_osf - zpia_top1 if np.isfinite(progressive_osf) else np.nan
        delta_pure_progressive = (
            progressive_pure - progressive_zpia
            if np.isfinite(progressive_pure) and np.isfinite(progressive_zpia)
            else np.nan
        )
        delta_cumulative_top1 = (
            progressive_cumulative - zpia_top1 if np.isfinite(progressive_cumulative) else np.nan
        )
        delta_exposure_top1 = progressive_exposure - zpia_top1 if np.isfinite(progressive_exposure) else np.nan
        delta_highdose_cumulative = (
            progressive_highdose - progressive_cumulative
            if np.isfinite(progressive_highdose) and np.isfinite(progressive_cumulative)
            else np.nan
        )
        delta_highdose_top1 = progressive_highdose - zpia_top1 if np.isfinite(progressive_highdose) else np.nan
        feedback_best = max(mba_feedback, rc4_osf)
        if max(
            abs(delta_core),
            abs(delta_rc4),
            abs(delta_spectral_core) if np.isfinite(delta_spectral_core) else 0.0,
            abs(delta_spectral_rc4) if np.isfinite(delta_spectral_rc4) else 0.0,
            abs(delta_multiz_core),
            abs(delta_multiz_rc4),
        ) < tie_eps:
            regime = "saturated_or_near_tie"
        elif rc4_multiz > max(rc4_fused, spectral, mba_core, feedback_best) + tie_eps:
            regime = "rc4_multiz_dominant"
        elif progressive_osf > max(rc4_multiz, rc4_fused, zpia_top1, mba_core, feedback_best) + tie_eps:
            regime = "progressive_osf_dominant"
        elif progressive_highdose > max(rc4_multiz, rc4_fused, zpia_top1, mba_core, feedback_best) + tie_eps:
            regime = "progressive_osf_highdose_dominant"
        elif progressive_exposure > max(zpia_top1, rc4_multiz, mba_core, feedback_best) + tie_eps:
            regime = "progressive_exposure_matched_dominant"
        elif progressive_cumulative > max(zpia_top1, rc4_multiz, mba_core, feedback_best) + tie_eps:
            regime = "progressive_zpia_cumulative_dominant"
        elif progressive_zpia > max(zpia_top1, rc4_multiz, mba_core, feedback_best) + tie_eps:
            regime = "progressive_zpia_dominant"
        elif spectral > max(rc4_fused, mba_core, feedback_best) + tie_eps:
            regime = "spectral_osf_dominant"
        elif feedback_best > max(rc4_fused, mba_core) + tie_eps:
            regime = "feedback_beneficial_pathological"
        elif delta_core > tie_eps:
            regime = "rc4_fused_dominant"
        else:
            regime = "lraes_core_dominant"
        rows.append(
            {
                "dataset": ds,
                "regime": regime,
                "tie_eps": tie_eps,
                "best_arm": max(
                    [
                        ("baseline_ce", float(row["baseline_ce"])),
                        ("mba_core_lraes", mba_core),
                        ("mba_feedback_lraes", mba_feedback),
                        ("rc4_osf", rc4_osf),
                        ("mba_core_rc4_fused_concat", rc4_fused),
                        ("mba_core_spectral_osf_concat", spectral),
                        ("mba_core_zpia_top1_pool", zpia_top1),
                        ("mba_core_zpia_multidir_pool", zpia_multidir),
                        ("mba_core_rc4_multiz_fused_concat", rc4_multiz),
                        ("progressive_zpia_only", progressive_zpia),
                        ("progressive_zpia_osf", progressive_osf),
                        ("random_progressive_osf", random_progressive),
                        ("progressive_zpia_only_pure", progressive_pure),
                        ("progressive_zpia_pure_cumulative", progressive_cumulative),
                        ("progressive_zpia_exposure_matched", progressive_exposure),
                        ("progressive_zpia_osf_highdose", progressive_highdose),
                    ],
                    key=lambda kv: kv[1],
                )[0],
                "mba_core_rc4_fused_concat_minus_mba_core_lraes": delta_core,
                "mba_core_rc4_fused_concat_minus_rc4_osf": delta_rc4,
                "mba_core_spectral_osf_concat_minus_mba_core_lraes": delta_spectral_core,
                "mba_core_spectral_osf_concat_minus_mba_core_rc4_fused_concat": delta_spectral_rc4,
                "zpia_multidir_pool_minus_zpia_top1_pool": delta_zpia_multi,
                "mba_core_rc4_multiz_fused_concat_minus_mba_core_lraes": delta_multiz_core,
                "mba_core_rc4_multiz_fused_concat_minus_mba_core_rc4_fused_concat": delta_multiz_rc4,
                "progressive_zpia_osf_minus_progressive_zpia_only": delta_progressive_osf,
                "progressive_zpia_osf_minus_mba_core_zpia_top1_pool": delta_progressive_top1,
                "progressive_zpia_only_pure_minus_progressive_zpia_only": delta_pure_progressive,
                "progressive_zpia_pure_cumulative_minus_mba_core_zpia_top1_pool": delta_cumulative_top1,
                "progressive_zpia_exposure_matched_minus_mba_core_zpia_top1_pool": delta_exposure_top1,
                "progressive_zpia_osf_highdose_minus_progressive_zpia_pure_cumulative": delta_highdose_cumulative,
                "progressive_zpia_osf_highdose_minus_mba_core_zpia_top1_pool": delta_highdose_top1,
                "random_progressive_osf_minus_progressive_zpia_osf": (
                    random_progressive - progressive_osf
                    if np.isfinite(random_progressive) and np.isfinite(progressive_osf)
                    else np.nan
                ),
                "mba_feedback_lraes_minus_mba_core_lraes": mba_feedback - mba_core,
                "rc4_osf_minus_mba_feedback_lraes": rc4_osf - mba_feedback,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "regime",
                "tie_eps",
                "best_arm",
                "mba_core_rc4_fused_concat_minus_mba_core_lraes",
                "mba_core_rc4_fused_concat_minus_rc4_osf",
                "mba_core_spectral_osf_concat_minus_mba_core_lraes",
                "mba_core_spectral_osf_concat_minus_mba_core_rc4_fused_concat",
                "zpia_multidir_pool_minus_zpia_top1_pool",
                "mba_core_rc4_multiz_fused_concat_minus_mba_core_lraes",
                "mba_core_rc4_multiz_fused_concat_minus_mba_core_rc4_fused_concat",
                "progressive_zpia_osf_minus_progressive_zpia_only",
                "progressive_zpia_osf_minus_mba_core_zpia_top1_pool",
                "progressive_zpia_only_pure_minus_progressive_zpia_only",
                "progressive_zpia_pure_cumulative_minus_mba_core_zpia_top1_pool",
                "progressive_zpia_exposure_matched_minus_mba_core_zpia_top1_pool",
                "progressive_zpia_osf_highdose_minus_progressive_zpia_pure_cumulative",
                "progressive_zpia_osf_highdose_minus_mba_core_zpia_top1_pool",
                "random_progressive_osf_minus_progressive_zpia_osf",
                "mba_feedback_lraes_minus_mba_core_lraes",
                "rc4_osf_minus_mba_feedback_lraes",
            ]
        )
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the MBA vs RC-4 full matrix.")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    parser.add_argument("--locked-root", type=str, default=str(DEFAULT_LOCKED_ROOT))
    parser.add_argument("--step1-root", type=str, default=str(DEFAULT_STEP1_ROOT))
    parser.add_argument("--spectral-root", type=str, default=str(DEFAULT_SPECTRAL_ROOT))
    parser.add_argument("--multitemplate-root", type=str, default=str(DEFAULT_MULTITEMPLATE_ROOT))
    parser.add_argument("--ref-root", type=str, default=str(DEFAULT_REF_ROOT))
    parser.add_argument("--baseline-drift-tol", type=float, default=1e-4)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    locked_root = Path(args.locked_root).resolve()
    step1_root = Path(args.step1_root).resolve()
    spectral_root = Path(args.spectral_root).resolve()
    multitemplate_root = Path(args.multitemplate_root).resolve()
    ref_root = Path(args.ref_root).resolve()
    summary_root = root / "_summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    jobs_df = _load_job_rows(root)
    actual_df = _load_actual_rows(root, jobs_df)
    current_per_seed_df = _standardize_actual_rows(actual_df, args.baseline_drift_tol)
    ref_frames: List[pd.DataFrame] = []
    loaded_arms: set[str] = set()
    if locked_root != root:
        locked_per_seed_df = _try_load_reference_per_seed_rows(locked_root, LOCKED_ARMS)
        ref_frames.append(locked_per_seed_df)
        if not locked_per_seed_df.empty:
            loaded_arms.update(locked_per_seed_df["arm"].astype(str).unique().tolist())
    if step1_root != root:
        step1_per_seed_df = _try_load_reference_per_seed_rows(step1_root, {"mba_core_rc4_fused_concat"})
        ref_frames.append(step1_per_seed_df)
        if not step1_per_seed_df.empty:
            loaded_arms.update(step1_per_seed_df["arm"].astype(str).unique().tolist())
    if spectral_root != root:
        spectral_per_seed_df = _try_load_reference_per_seed_rows(
            spectral_root, {"mba_core_spectral_osf_concat"}
        )
        ref_frames.append(spectral_per_seed_df)
        if not spectral_per_seed_df.empty:
            loaded_arms.update(spectral_per_seed_df["arm"].astype(str).unique().tolist())
    if multitemplate_root != root:
        multitemplate_per_seed_df = _try_load_reference_per_seed_rows(
            multitemplate_root, MULTITEMPLATE_ARMS
        )
        ref_frames.append(multitemplate_per_seed_df)
        if not multitemplate_per_seed_df.empty:
            loaded_arms.update(multitemplate_per_seed_df["arm"].astype(str).unique().tolist())

    ref_frames = [df for df in ref_frames if not df.empty]
    if ref_frames:
        current_keep = current_per_seed_df[~current_per_seed_df["arm"].astype(str).isin(loaded_arms)].copy()
        per_seed_df = pd.concat(ref_frames + [current_keep], ignore_index=True, sort=False)
    else:
        per_seed_df = current_per_seed_df
    baseline_drift_df = _baseline_drift_report(actual_df, args.baseline_drift_tol)
    mba_repro_df = _mba_repro_check(actual_df, ref_root)
    dataset_summary_df = _dataset_summary(per_seed_df)
    gap_view_df = _gap_attribution_view(dataset_summary_df, baseline_drift_df, mba_repro_df)
    step1_compare_df = _mba_core_rc4_vs_locked(dataset_summary_df)
    spectral_compare_df = _spectral_osf_vs_refs(dataset_summary_df)
    multitemplate_compare_df = _multitemplate_vs_refs(dataset_summary_df)
    progressive_compare_df = _progressive_vs_refs(dataset_summary_df)
    overall_summary_df = _overall_summary_table(dataset_summary_df)
    per_dataset_gap_df = _per_dataset_gap_table(dataset_summary_df)
    regime_taxonomy_df = _regime_taxonomy_table(dataset_summary_df)

    per_seed_path = summary_root / "per_seed_arms.csv"
    dataset_summary_path = summary_root / "dataset_summary.csv"
    drift_path = summary_root / "baseline_drift_report.csv"
    repro_path = summary_root / "mba_repro_check.csv"
    gap_path = summary_root / "gap_attribution_view.csv"
    per_seed_step1_path = summary_root / "per_seed_arms_step1.csv"
    dataset_summary_step1_path = summary_root / "dataset_summary_step1.csv"
    step1_compare_path = summary_root / "mba_core_rc4_vs_locked.csv"
    step1_gap_path = summary_root / "step1_gap_view.csv"
    spectral_compare_path = summary_root / "spectral_osf_vs_refs.csv"
    multitemplate_compare_path = summary_root / "multi_template_osf_vs_refs.csv"
    progressive_compare_path = summary_root / "progressive_osf_vs_refs.csv"
    final_table1_path = summary_root / "table1_overall_mean_and_winrate.csv"
    final_table2_path = summary_root / "table2_per_dataset_gap_attribution.csv"
    final_table3_path = summary_root / "table3_regime_taxonomy.csv"

    per_seed_df.to_csv(per_seed_path, index=False)
    dataset_summary_df.to_csv(dataset_summary_path, index=False)
    baseline_drift_df.to_csv(drift_path, index=False)
    mba_repro_df.to_csv(repro_path, index=False)
    gap_view_df.to_csv(gap_path, index=False)
    per_seed_df.to_csv(per_seed_step1_path, index=False)
    dataset_summary_df.to_csv(dataset_summary_step1_path, index=False)
    step1_compare_df.to_csv(step1_compare_path, index=False)
    gap_view_df.to_csv(step1_gap_path, index=False)
    spectral_compare_df.to_csv(spectral_compare_path, index=False)
    multitemplate_compare_df.to_csv(multitemplate_compare_path, index=False)
    progressive_compare_df.to_csv(progressive_compare_path, index=False)
    overall_summary_df.to_csv(final_table1_path, index=False)
    per_dataset_gap_df.to_csv(final_table2_path, index=False)
    regime_taxonomy_df.to_csv(final_table3_path, index=False)

    print(per_seed_path)
    print(dataset_summary_path)
    print(drift_path)
    print(repro_path)
    print(gap_path)
    print(per_seed_step1_path)
    print(dataset_summary_step1_path)
    print(step1_compare_path)
    print(step1_gap_path)
    print(spectral_compare_path)
    print(multitemplate_compare_path)
    print(progressive_compare_path)
    print(final_table1_path)
    print(final_table2_path)
    print(final_table3_path)


if __name__ == "__main__":
    main()
