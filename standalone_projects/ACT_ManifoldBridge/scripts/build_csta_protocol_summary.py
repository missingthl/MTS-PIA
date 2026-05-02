#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.datasets import load_trials_for_dataset, make_trial_split  # noqa: E402


RAW_AUG_METHODS = {
    "raw_aug_jitter",
    "raw_aug_scaling",
    "raw_aug_timewarp",
    "raw_aug_magnitude_warping",
    "raw_aug_window_warping",
    "raw_aug_window_slicing",
}

PHASE2_METHODS = {
    "raw_aug_magnitude_warping",
    "raw_aug_window_warping",
    "raw_aug_window_slicing",
    "wdba_sameclass",
    "spawner_sameclass_style",
}

CSTA_METHODS = {
    "csta_top1_current",
    "csta_group_template_top",
    "csta_topk_softmax_tau_0.05",
    "csta_topk_softmax_tau_0.10",
    "csta_topk_softmax_tau_0.20",
    "csta_topk_uniform_top5",
}

DEFAULT_PHASE1_ROOT = PROJECT_ROOT / "results" / "csta_external_baselines_phase1" / "resnet1d_s123"
DEFAULT_PHASE2_ROOT = PROJECT_ROOT / "results" / "csta_external_baselines_phase2" / "resnet1d_s123"
DEFAULT_PHASE3_ROOT = PROJECT_ROOT / "results" / "csta_external_baselines_phase3" / "manifold_mixup_resnet1d_s123"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "results" / "csta_protocol_v1"


@dataclass(frozen=True)
class RankBundle:
    rank_rows: pd.DataFrame
    ranks: pd.DataFrame


def _read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if path.is_file():
        return pd.read_csv(path)
    if required:
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.DataFrame()


def _success(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "status" not in out.columns:
        out["status"] = "success"
    return out[out["status"].fillna("success").astype(str) == "success"].copy()


def _safe_float(value) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _first_non_null(values: pd.Series, default=np.nan):
    vals = values.dropna()
    if vals.empty:
        return default
    return vals.iloc[0]


def _method_family(method: str) -> str:
    if method == "no_aug":
        return "none"
    if method == "best_rawaug" or method.startswith("raw_aug_"):
        return "raw_time"
    if method == "raw_mixup":
        return "raw_mixup"
    if method in {"dba_sameclass", "wdba_sameclass"}:
        return "dtw_barycenter"
    if method == "spawner_sameclass_style":
        return "dtw_pattern_mix"
    if method == "raw_smote_flatten_balanced":
        return "flattened_raw"
    if method in {"random_cov_state", "pca_cov_state"}:
        return "covariance_state"
    if method.startswith("csta_"):
        return "covariance_template"
    if method == "manifold_mixup":
        return "hidden_state"
    return "other"


def _metadata_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "source_space",
        "label_mode",
        "backbone",
        "train_split_only",
        "uses_external_library",
        "library_name",
        "budget_matched",
        "selection_rule",
    ]
    available = [c for c in fields if c in rows.columns]
    if rows.empty:
        return pd.DataFrame(columns=["method", *fields])
    meta = rows.groupby("method", as_index=False).agg({c: _first_non_null for c in available})
    for col in fields:
        if col not in meta.columns:
            meta[col] = np.nan
    overrides = {
        "best_rawaug": {
            "source_space": "raw_time",
            "label_mode": "hard",
            "backbone": "resnet1d",
            "train_split_only": True,
            "uses_external_library": True,
            "library_name": "mixed_raw_aug",
            "budget_matched": True,
            "selection_rule": "validation_selected_raw_aug",
        }
    }
    for method, values in overrides.items():
        if method in set(meta["method"]):
            for key, value in values.items():
                meta.loc[meta["method"] == method, key] = value
        else:
            meta = pd.concat([meta, pd.DataFrame([{"method": method, **values}])], ignore_index=True)
    meta["method_family"] = meta["method"].map(_method_family)
    return meta


def _derive_best_rawaug(rows: pd.DataFrame) -> pd.DataFrame:
    raw = rows[rows["method"].isin(RAW_AUG_METHODS)].copy()
    if raw.empty:
        return pd.DataFrame(columns=rows.columns)
    best_rows: List[Dict[str, object]] = []
    for (dataset, seed), sub in raw.groupby(["dataset", "seed"], dropna=False):
        sort_cols = ["best_val_f1", "method"] if "best_val_f1" in sub.columns else ["aug_f1", "method"]
        ascending = [False, True]
        best = sub.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict()
        best["selected_method"] = best.get("method")
        best["method"] = "best_rawaug"
        best["source_space"] = "raw_time"
        best["label_mode"] = "hard"
        best["uses_external_library"] = True
        best["library_name"] = "mixed_raw_aug"
        best["budget_matched"] = True
        best["selection_rule"] = "validation_selected_raw_aug"
        best_rows.append(best)
    return pd.DataFrame(best_rows)


def _dataset_summary(rows: pd.DataFrame) -> pd.DataFrame:
    value_cols = ["method_elapsed_sec", "fallback_count", "actual_aug_ratio", "aug_count"]
    for col in value_cols:
        if col not in rows.columns:
            rows[col] = np.nan
    out = (
        rows.groupby(["dataset", "method"], as_index=False)
        .agg(
            macro_f1_mean=("aug_f1", "mean"),
            macro_f1_std_over_seeds=("aug_f1", "std"),
            gain_mean_existing=("gain", "mean"),
            actual_aug_ratio=("actual_aug_ratio", "mean"),
            aug_count=("aug_count", "mean"),
            method_elapsed_sec_mean=("method_elapsed_sec", "mean"),
            fallback_count_mean=("fallback_count", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["dataset", "method"])
        .reset_index(drop=True)
    )
    return out


def _rank_bundle(dataset_summary: pd.DataFrame, methods: Optional[Sequence[str]] = None) -> RankBundle:
    sub = dataset_summary.copy()
    if methods is not None:
        sub = sub[sub["method"].isin(methods)].copy()
    pivot = sub.pivot(index="dataset", columns="method", values="macro_f1_mean")
    common_methods = [c for c in pivot.columns if pivot[c].notna().all()]
    pivot = pivot[common_methods]
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    rows = []
    for method in ranks.columns:
        vals = ranks[method].dropna()
        rows.append(
            {
                "method": method,
                "n_rank_datasets": int(vals.shape[0]),
                "average_rank": float(vals.mean()) if not vals.empty else np.nan,
                "win_count": int((vals == 1.0).sum()) if not vals.empty else 0,
            }
        )
    return RankBundle(rank_rows=pd.DataFrame(rows), ranks=ranks)


def _external_methods(methods: Iterable[str]) -> List[str]:
    return sorted(m for m in set(methods) if m != "no_aug" and m not in CSTA_METHODS)


def _best_global_method(dataset_summary: pd.DataFrame, methods: Sequence[str]) -> Tuple[str, float]:
    sub = dataset_summary[dataset_summary["method"].isin(methods)].copy()
    if sub.empty:
        return "", float("nan")
    means = sub.groupby("method", as_index=False).agg(mean_f1=("macro_f1_mean", "mean"))
    means = means.sort_values(["mean_f1", "method"], ascending=[False, True])
    return str(means.iloc[0]["method"]), float(means.iloc[0]["mean_f1"])


def _best_by_dataset(dataset_summary: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    sub = dataset_summary[dataset_summary["method"].isin(methods)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["dataset", "best_external_method", "best_external_f1"])
    rows = []
    for dataset, g in sub.groupby("dataset", dropna=False):
        best = g.sort_values(["macro_f1_mean", "method"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "best_external_method": best["method"],
                "best_external_f1": float(best["macro_f1_mean"]),
            }
        )
    return pd.DataFrame(rows)


def _per_seed_negative_rates(rows: pd.DataFrame) -> pd.DataFrame:
    no_aug = rows[rows["method"] == "no_aug"][["dataset", "seed", "aug_f1"]].rename(columns={"aug_f1": "no_aug_f1"})
    joined = rows.merge(no_aug, on=["dataset", "seed"], how="left")
    joined["seed_gain_vs_no_aug"] = joined["aug_f1"] - joined["no_aug_f1"]
    return (
        joined.groupby("method", as_index=False)
        .agg(
            negative_transfer_rate_seed=("seed_gain_vs_no_aug", lambda x: float((x < 0).mean())),
            n_seed_comparisons=("seed_gain_vs_no_aug", "count"),
        )
        .reset_index(drop=True)
    )


def _build_overall_metrics(
    rows: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    meta: pd.DataFrame,
    dataset_best_external: pd.DataFrame,
    global_best_method: str,
    global_best_mean: float,
    combined_ranks: pd.DataFrame,
) -> pd.DataFrame:
    no_aug = dataset_summary[dataset_summary["method"] == "no_aug"][["dataset", "macro_f1_mean"]].rename(
        columns={"macro_f1_mean": "no_aug_f1"}
    )
    ds = dataset_summary.merge(no_aug, on="dataset", how="left")
    ds = ds.merge(dataset_best_external, on="dataset", how="left")
    ds["gain_vs_no_aug"] = ds["macro_f1_mean"] - ds["no_aug_f1"]
    ds["gap_vs_dataset_best_external"] = ds["macro_f1_mean"] - ds["best_external_f1"]

    seed_neg = _per_seed_negative_rates(rows)
    out = (
        ds.groupby("method", as_index=False)
        .agg(
            macro_f1_mean=("macro_f1_mean", "mean"),
            macro_f1_std_over_datasets=("macro_f1_mean", "std"),
            gain_vs_no_aug_mean=("gain_vs_no_aug", "mean"),
            negative_transfer_rate_dataset=("gain_vs_no_aug", lambda x: float((x < 0).mean())),
            gap_vs_dataset_best_external=("gap_vs_dataset_best_external", "mean"),
            n_datasets=("dataset", "nunique"),
        )
        .reset_index(drop=True)
    )
    out["global_best_external"] = global_best_method
    out["global_best_external_mean_f1"] = global_best_mean
    out["gap_vs_global_best_external"] = out["macro_f1_mean"] - global_best_mean
    out = out.merge(seed_neg, on="method", how="left")
    out = out.merge(combined_ranks, on="method", how="left")
    out = out.merge(meta, on="method", how="left")
    for col in ["budget_matched", "source_space", "label_mode"]:
        if col not in out.columns:
            out[col] = np.nan
    ordered = [
        "method",
        "method_family",
        "macro_f1_mean",
        "macro_f1_std_over_datasets",
        "gain_vs_no_aug_mean",
        "negative_transfer_rate_dataset",
        "negative_transfer_rate_seed",
        "gap_vs_global_best_external",
        "gap_vs_dataset_best_external",
        "average_rank",
        "win_count",
        "budget_matched",
        "source_space",
        "label_mode",
        "n_datasets",
        "global_best_external",
        "global_best_external_mean_f1",
    ]
    return out[[c for c in ordered if c in out.columns]].sort_values(
        ["average_rank", "macro_f1_mean"], ascending=[True, False], na_position="last"
    )


def _build_dataset_metrics(
    dataset_summary: pd.DataFrame,
    ranks: pd.DataFrame,
    dataset_best_external: pd.DataFrame,
) -> pd.DataFrame:
    no_aug = dataset_summary[dataset_summary["method"] == "no_aug"][["dataset", "macro_f1_mean"]].rename(
        columns={"macro_f1_mean": "no_aug_f1"}
    )
    out = dataset_summary.merge(no_aug, on="dataset", how="left")
    out = out.merge(dataset_best_external, on="dataset", how="left")
    rank_long = ranks.reset_index().melt(id_vars="dataset", var_name="method", value_name="rank_in_dataset")
    out = out.merge(rank_long, on=["dataset", "method"], how="left")
    out["is_dataset_winner"] = out["rank_in_dataset"] == 1.0
    out["gain_vs_no_aug"] = out["macro_f1_mean"] - out["no_aug_f1"]
    out["gap_vs_best_external_dataset"] = out["macro_f1_mean"] - out["best_external_f1"]
    ordered = [
        "dataset",
        "method",
        "macro_f1_mean",
        "macro_f1_std_over_seeds",
        "gain_vs_no_aug",
        "rank_in_dataset",
        "is_dataset_winner",
        "gap_vs_best_external_dataset",
        "actual_aug_ratio",
        "method_elapsed_sec_mean",
        "fallback_count_mean",
        "n_seeds",
        "best_external_method",
        "best_external_f1",
    ]
    return out[[c for c in ordered if c in out.columns]].sort_values(["dataset", "rank_in_dataset", "method"])


def _build_csta_vs_external(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = dataset_summary.pivot(index="dataset", columns="method", values="macro_f1_mean").reset_index()
    wanted = [
        "dataset",
        "csta_top1_current",
        "csta_group_template_top",
        "no_aug",
        "best_rawaug",
        "dba_sameclass",
        "wdba_sameclass",
        "spawner_sameclass_style",
        "random_cov_state",
        "pca_cov_state",
    ]
    out = pd.DataFrame()
    out["dataset"] = pivot["dataset"]
    for col in wanted[1:]:
        out[col] = pivot[col] if col in pivot.columns else np.nan
    csta = out["csta_top1_current"]
    for ref, suffix in [
        ("best_rawaug", "best_rawaug"),
        ("dba_sameclass", "dba"),
        ("wdba_sameclass", "wdba"),
        ("random_cov_state", "random_cov"),
        ("pca_cov_state", "pca_cov"),
    ]:
        out[f"csta_top1_minus_{suffix}"] = csta - out[ref]
    return out.sort_values("dataset")


def _build_average_rank_tables(
    phase1_rows: pd.DataFrame,
    phase2_rows: pd.DataFrame,
    main_dataset_summary: pd.DataFrame,
) -> pd.DataFrame:
    phase1_ds = _dataset_summary(phase1_rows)
    phase2_ds = _dataset_summary(phase2_rows)
    p1 = _rank_bundle(phase1_ds).rank_rows.rename(
        columns={"average_rank": "rank_phase1_only", "win_count": "win_count_phase1_only"}
    )
    p2 = _rank_bundle(phase2_ds).rank_rows.rename(
        columns={"average_rank": "rank_phase2_new_only", "win_count": "win_count_phase2_new_only"}
    )
    pc = _rank_bundle(main_dataset_summary).rank_rows.rename(
        columns={"average_rank": "rank_phase1_phase2_common", "win_count": "win_count_phase1_phase2_common"}
    )
    out = pc.merge(p1[["method", "rank_phase1_only", "win_count_phase1_only"]], on="method", how="outer")
    out = out.merge(p2[["method", "rank_phase2_new_only", "win_count_phase2_new_only"]], on="method", how="outer")
    return out.sort_values(["rank_phase1_phase2_common", "rank_phase1_only", "rank_phase2_new_only"], na_position="last")


def _build_negative_transfer(dataset_metrics: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    dataset_rates = (
        dataset_metrics.groupby("method", as_index=False)
        .agg(
            negative_transfer_rate_dataset=("gain_vs_no_aug", lambda x: float((x < 0).mean())),
            negative_transfer_dataset_count=("gain_vs_no_aug", lambda x: int((x < 0).sum())),
            n_dataset_comparisons=("gain_vs_no_aug", "count"),
        )
        .reset_index(drop=True)
    )
    seed_rates = _per_seed_negative_rates(rows)
    return dataset_rates.merge(seed_rates, on="method", how="left")


def _build_cost_pareto(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    out = (
        dataset_summary.groupby("method", as_index=False)
        .agg(
            macro_f1_mean=("macro_f1_mean", "mean"),
            method_elapsed_sec_mean=("method_elapsed_sec_mean", "mean"),
            method_elapsed_sec_std=("method_elapsed_sec_mean", "std"),
            actual_aug_ratio_mean=("actual_aug_ratio", "mean"),
            fallback_count_mean=("fallback_count_mean", "mean"),
        )
        .reset_index(drop=True)
    )
    out["cost_available"] = out["method_elapsed_sec_mean"].notna()
    out["method_elapsed_sec_per_aug_ratio"] = out["method_elapsed_sec_mean"] / out["actual_aug_ratio_mean"].replace(0, np.nan)
    out["method_family"] = out["method"].map(_method_family)
    return out.sort_values(["cost_available", "macro_f1_mean"], ascending=[False, False])


def _load_dataset_meta(datasets: Sequence[str], seeds: Sequence[int], val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    meta_rows: List[Dict[str, object]] = []
    split_rows: List[Dict[str, object]] = []
    for dataset in sorted(set(datasets)):
        try:
            trials = load_trials_for_dataset(dataset)
            x0 = np.asarray(trials[0].x)
            labels_all = np.asarray([t.y for t in trials], dtype=np.int64)
            counts_all = np.bincount(labels_all) if labels_all.size else np.asarray([])
            class_imbalance = (
                float(counts_all.max() / max(counts_all.min(), 1)) if counts_all.size and counts_all.min() > 0 else np.nan
            )
            first_seed_meta = {}
            for seed in seeds:
                train, test, val = make_trial_split(trials, seed=int(seed), val_ratio=float(val_ratio))
                train_labels = np.asarray([t.y for t in train], dtype=np.int64)
                test_labels = np.asarray([t.y for t in test], dtype=np.int64)
                val_labels = np.asarray([t.y for t in val], dtype=np.int64) if val else np.asarray([], dtype=np.int64)
                split_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "n_train": int(len(train)),
                        "n_val": int(len(val)),
                        "n_test": int(len(test)),
                        "n_classes_train": int(np.unique(train_labels).size) if train_labels.size else 0,
                        "n_classes_test": int(np.unique(test_labels).size) if test_labels.size else 0,
                        "n_classes_val": int(np.unique(val_labels).size) if val_labels.size else 0,
                    }
                )
                if not first_seed_meta:
                    first_seed_meta = {
                        "n_train": int(len(train)),
                        "n_val": int(len(val)),
                        "n_test": int(len(test)),
                    }
            meta_rows.append(
                {
                    "dataset": dataset,
                    "n_total": int(len(trials)),
                    "n_train": first_seed_meta.get("n_train", np.nan),
                    "n_val": first_seed_meta.get("n_val", np.nan),
                    "n_test": first_seed_meta.get("n_test", np.nan),
                    "C": int(x0.shape[0]) if x0.ndim >= 2 else np.nan,
                    "T": int(x0.shape[1]) if x0.ndim >= 2 else np.nan,
                    "n_classes": int(np.unique(labels_all).size),
                    "class_imbalance_ratio": class_imbalance,
                }
            )
        except Exception as exc:
            meta_rows.append(
                {
                    "dataset": dataset,
                    "n_total": np.nan,
                    "n_train": np.nan,
                    "n_val": np.nan,
                    "n_test": np.nan,
                    "C": np.nan,
                    "T": np.nan,
                    "n_classes": np.nan,
                    "class_imbalance_ratio": np.nan,
                    "load_error": str(exc),
                }
            )
    return pd.DataFrame(meta_rows), pd.DataFrame(split_rows)


def _build_regime_table(dataset_metrics: pd.DataFrame, dataset_meta: pd.DataFrame) -> pd.DataFrame:
    pivot = dataset_metrics.pivot(index="dataset", columns="method", values="macro_f1_mean").reset_index()
    best = (
        dataset_metrics[~dataset_metrics["method"].isin({"no_aug", *CSTA_METHODS})]
        .sort_values(["dataset", "macro_f1_mean", "method"], ascending=[True, False, True])
        .groupby("dataset", as_index=False)
        .first()[["dataset", "method", "macro_f1_mean"]]
        .rename(columns={"method": "best_external_method", "macro_f1_mean": "best_external_f1"})
    )
    csta_rank = dataset_metrics[dataset_metrics["method"] == "csta_top1_current"][
        ["dataset", "rank_in_dataset", "gain_vs_no_aug"]
    ].rename(columns={"rank_in_dataset": "csta_rank", "gain_vs_no_aug": "csta_gain"})
    out = dataset_meta.merge(best, on="dataset", how="left").merge(csta_rank, on="dataset", how="left")
    out["baseline_f1"] = out["dataset"].map(dict(zip(pivot["dataset"], pivot.get("no_aug", pd.Series(np.nan, index=pivot.index)))))
    out["csta_f1"] = out["dataset"].map(
        dict(zip(pivot["dataset"], pivot.get("csta_top1_current", pd.Series(np.nan, index=pivot.index))))
    )
    out["regime_label"] = ""
    ordered = [
        "dataset",
        "n_train",
        "n_test",
        "C",
        "T",
        "n_classes",
        "class_imbalance_ratio",
        "baseline_f1",
        "best_external_method",
        "best_external_f1",
        "csta_f1",
        "csta_rank",
        "csta_gain",
        "regime_label",
    ]
    return out[[c for c in ordered if c in out.columns]].sort_values("dataset")


def _build_phase3_refs(
    phase3_rows: pd.DataFrame,
    main_dataset_summary: pd.DataFrame,
    phase1_dataset_summary: pd.DataFrame,
    phase2_dataset_summary: pd.DataFrame,
    dataset_best_external: pd.DataFrame,
) -> pd.DataFrame:
    if phase3_rows.empty:
        return pd.DataFrame()
    phase3_ds = _dataset_summary(phase3_rows)
    p3 = phase3_ds.pivot(index="dataset", columns="method", values="macro_f1_mean")
    main_pivot = main_dataset_summary.pivot(index="dataset", columns="method", values="macro_f1_mean")
    phase1_external = _external_methods(phase1_dataset_summary["method"].unique())
    phase2_external = sorted(set(phase2_dataset_summary["method"]).intersection(PHASE2_METHODS))
    best_phase1_method, best_phase1_mean = _best_global_method(phase1_dataset_summary, phase1_external)
    best_phase2_method, best_phase2_mean = _best_global_method(phase2_dataset_summary, phase2_external)
    best_ds = dataset_best_external.set_index("dataset")
    rows = []
    for dataset in sorted(p3.index):
        no_aug = _safe_float(main_pivot.at[dataset, "no_aug"]) if "no_aug" in main_pivot.columns and dataset in main_pivot.index else np.nan
        csta = (
            _safe_float(main_pivot.at[dataset, "csta_top1_current"])
            if "csta_top1_current" in main_pivot.columns and dataset in main_pivot.index
            else np.nan
        )
        group = (
            _safe_float(main_pivot.at[dataset, "csta_group_template_top"])
            if "csta_group_template_top" in main_pivot.columns and dataset in main_pivot.index
            else np.nan
        )
        best_phase1_val = (
            _safe_float(main_pivot.at[dataset, best_phase1_method])
            if best_phase1_method in main_pivot.columns and dataset in main_pivot.index
            else np.nan
        )
        best_phase2_val = (
            _safe_float(main_pivot.at[dataset, best_phase2_method])
            if best_phase2_method in main_pivot.columns and dataset in main_pivot.index
            else np.nan
        )
        best_external_val = _safe_float(best_ds.at[dataset, "best_external_f1"]) if dataset in best_ds.index else np.nan
        mm = _safe_float(p3.at[dataset, "manifold_mixup"]) if "manifold_mixup" in p3.columns else np.nan
        rows.append(
            {
                "dataset": dataset,
                "no_aug": no_aug,
                "csta_top1_current": csta,
                "csta_group_template_top": group,
                "best_phase1_global_method": best_phase1_method,
                "best_phase1_global_mean_f1": best_phase1_mean,
                "best_phase1_global": best_phase1_val,
                "best_phase2_global_method": best_phase2_method,
                "best_phase2_global_mean_f1": best_phase2_mean,
                "best_phase2_global": best_phase2_val,
                "best_external_dataset_method": best_ds.at[dataset, "best_external_method"] if dataset in best_ds.index else "",
                "best_external_dataset": best_external_val,
                "manifold_mixup": mm,
                "manifold_mixup_minus_no_aug": mm - no_aug,
                "manifold_mixup_minus_csta_top1": mm - csta,
                "manifold_mixup_minus_best_phase2_global": mm - best_phase2_val,
            }
        )
    return pd.DataFrame(rows)


def _class_count_map(dataset: str, seed: int, val_ratio: float) -> Dict[int, int]:
    trials = load_trials_for_dataset(dataset)
    train, _, _ = make_trial_split(trials, seed=int(seed), val_ratio=float(val_ratio))
    labels = np.asarray([t.y for t in train], dtype=np.int64)
    return {int(c): int((labels == c).sum()) for c in np.unique(labels)}


def _build_wdba_audit(phase2_rows: pd.DataFrame, val_ratio: float) -> pd.DataFrame:
    rows = []
    wdba = phase2_rows[phase2_rows["method"] == "wdba_sameclass"].copy()
    for _, row in wdba.iterrows():
        dataset = str(row["dataset"])
        seed = int(row["seed"])
        k = int(row.get("wdba_k", 5) if not pd.isna(row.get("wdba_k", np.nan)) else 5)
        fallback_total = int(row.get("fallback_count", 0) if not pd.isna(row.get("fallback_count", np.nan)) else 0)
        try:
            class_counts = _class_count_map(dataset, seed, val_ratio)
        except Exception:
            class_counts = {}
        multiplier = int(round(float(row.get("actual_aug_ratio", 0) or 0)))
        with_replacement_total = 0
        for class_id, class_size in sorted(class_counts.items()):
            n_aug_requested = int(class_size * multiplier)
            with_replacement = int(n_aug_requested if class_size < k else 0)
            with_replacement_total += with_replacement
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "class_id": class_id,
                    "class_size": class_size,
                    "n_aug_requested": n_aug_requested,
                    "with_replacement_count": with_replacement,
                    "true_fallback_count": np.nan,
                    "fallback_reason": "with_replacement_due_to_class_size" if with_replacement else "none",
                    "valid_dba_count": np.nan,
                    "weights_entropy_mean": np.nan,
                    "dtw_distance_mean": np.nan,
                    "method_elapsed_sec": row.get("method_elapsed_sec", np.nan),
                    "audit_detail_available": False,
                }
            )
        true_fallback_total = max(fallback_total - with_replacement_total, 0)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "class_id": "__all__",
                "class_size": sum(class_counts.values()) if class_counts else np.nan,
                "n_aug_requested": int(row.get("aug_count", 0) if not pd.isna(row.get("aug_count", np.nan)) else 0),
                "with_replacement_count": with_replacement_total,
                "true_fallback_count": true_fallback_total,
                "fallback_reason": "replacement_plus_tau_or_barycenter_fallback"
                if true_fallback_total
                else "replacement_only_or_none",
                "valid_dba_count": int(row.get("aug_count", 0) if not pd.isna(row.get("aug_count", np.nan)) else 0)
                - true_fallback_total,
                "weights_entropy_mean": np.nan,
                "dtw_distance_mean": np.nan,
                "method_elapsed_sec": row.get("method_elapsed_sec", np.nan),
                "audit_detail_available": False,
            }
        )
    return pd.DataFrame(rows)


def _build_spawner_audit(phase2_rows: pd.DataFrame, val_ratio: float) -> pd.DataFrame:
    rows = []
    spawner = phase2_rows[phase2_rows["method"] == "spawner_sameclass_style"].copy()
    for _, row in spawner.iterrows():
        dataset = str(row["dataset"])
        seed = int(row["seed"])
        fallback_total = int(row.get("fallback_count", 0) if not pd.isna(row.get("fallback_count", np.nan)) else 0)
        try:
            class_counts = _class_count_map(dataset, seed, val_ratio)
        except Exception:
            class_counts = {}
        multiplier = int(round(float(row.get("actual_aug_ratio", 0) or 0)))
        scarcity_total = 0
        for class_id, class_size in sorted(class_counts.items()):
            n_aug_requested = int(class_size * multiplier)
            scarcity = int(n_aug_requested if class_size <= 1 else 0)
            scarcity_total += scarcity
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "class_id": class_id,
                    "class_size": class_size,
                    "n_aug_requested": n_aug_requested,
                    "same_class_scarcity_count": scarcity,
                    "dtw_path_true_fallback_count": np.nan,
                    "fallback_reason": "no_same_class_mate" if scarcity else "none",
                    "dtw_path_valid_rate": np.nan,
                    "noise_scale": row.get("spawner_noise_scale", np.nan),
                    "method_elapsed_sec": row.get("method_elapsed_sec", np.nan),
                    "audit_detail_available": False,
                }
            )
        true_fallback_total = max(fallback_total - scarcity_total, 0)
        aug_count = int(row.get("aug_count", 0) if not pd.isna(row.get("aug_count", np.nan)) else 0)
        valid_rate = 1.0 - (true_fallback_total / aug_count) if aug_count else np.nan
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "class_id": "__all__",
                "class_size": sum(class_counts.values()) if class_counts else np.nan,
                "n_aug_requested": aug_count,
                "same_class_scarcity_count": scarcity_total,
                "dtw_path_true_fallback_count": true_fallback_total,
                "fallback_reason": "dtw_path_exception_or_alignment_fallback" if true_fallback_total else "scarcity_only_or_none",
                "dtw_path_valid_rate": valid_rate,
                "noise_scale": row.get("spawner_noise_scale", np.nan),
                "method_elapsed_sec": row.get("method_elapsed_sec", np.nan),
                "audit_detail_available": False,
            }
        )
    return pd.DataFrame(rows)


def _build_mechanism_diagnostics(rows: pd.DataFrame) -> pd.DataFrame:
    csta = rows[rows["method"].isin(CSTA_METHODS)].copy()
    wanted = [
        "template_response_top1_mean",
        "template_response_gap_mean",
        "template_confidence_mean",
        "selected_template_entropy",
        "template_usage_entropy",
        "top_template_concentration",
        "safe_radius_ratio_mean",
        "safe_clip_rate",
        "gamma_requested_mean",
        "gamma_used_mean",
        "z_displacement_norm_mean",
        "transport_error_logeuc_mean",
        "aug_valid_rate",
        "feasible_rate",
        "selector_accept_rate",
        "fidelity_score_mean",
        "variety_score_mean",
        "fv_score_mean",
        "pre_filter_reject_count",
        "post_bridge_reject_count",
    ]
    meta_fields = [
        "operator_name",
        "dictionary_estimator",
        "activation_policy",
        "activation_scope",
        "activation_topk",
        "activation_tau",
        "safe_generator",
        "bridge_realizer",
    ]
    output_rows = []
    for (dataset, method), sub in csta.groupby(["dataset", "method"], dropna=False):
        out = {"dataset": dataset, "method": method}
        available_count = 0
        for col in wanted:
            source_col = col
            if col == "selected_template_entropy" and col not in sub.columns and "template_usage_entropy" in sub.columns:
                source_col = "template_usage_entropy"
            if source_col in sub.columns:
                value = pd.to_numeric(sub[source_col], errors="coerce").mean()
                out[col] = value
                if not pd.isna(value):
                    available_count += 1
            else:
                out[col] = np.nan
        if pd.isna(out.get("selected_template_entropy", np.nan)) and not pd.isna(out.get("template_usage_entropy", np.nan)):
            out["selected_template_entropy"] = out["template_usage_entropy"]
        for col in meta_fields:
            if col in sub.columns:
                out[col] = _first_non_null(sub[col])
            else:
                out[col] = np.nan
        out["diagnostic_available"] = available_count == len(wanted)
        out["diagnostic_partial_available"] = available_count > 0
        output_rows.append(out)
    return pd.DataFrame(output_rows)


def _build_hyperparam_template() -> pd.DataFrame:
    rows = [
        {
            "stage": "csta_tuning",
            "parameter": "gamma",
            "candidate_values": "0.05,0.10,0.20",
            "primary_readouts": "vs_no_aug,vs_wdba,vs_best_rawaug,negative_transfer_rate",
            "status": "planned_not_run",
        },
        {
            "stage": "csta_tuning",
            "parameter": "eta_safe",
            "candidate_values": "0.25,0.50,0.75,1.00",
            "primary_readouts": "safe_clip_rate,safe_radius_ratio_mean,vs_wdba",
            "status": "planned_not_run",
        },
        {
            "stage": "csta_tuning",
            "parameter": "k_dir",
            "candidate_values": "5,10,20",
            "primary_readouts": "template_usage_entropy,template_response_gap_mean,vs_random_pca_cov",
            "status": "planned_not_run",
        },
        {
            "stage": "csta_tuning",
            "parameter": "multiplier",
            "candidate_values": "2,5,10",
            "primary_readouts": "actual_aug_ratio,gain_vs_no_aug,negative_transfer_rate",
            "status": "planned_not_run",
        },
    ]
    return pd.DataFrame(rows)


def _write_validity_report(
    path: Path,
    *,
    global_best_method: str,
    global_best_mean: float,
    phase3_refs: pd.DataFrame,
    wdba_audit: pd.DataFrame,
    spawner_audit: pd.DataFrame,
) -> None:
    wdba_all = wdba_audit[wdba_audit["class_id"].astype(str) == "__all__"].copy()
    spawner_all = spawner_audit[spawner_audit["class_id"].astype(str) == "__all__"].copy()
    lines = [
        "# External Baseline Validity Report",
        "",
        "Phase 1/2 performance references were treated as locked; no training performance was rerun.",
        "",
        f"- Global best external method: `{global_best_method}`",
        f"- Global best external mean F1: `{global_best_mean:.6f}`" if math.isfinite(global_best_mean) else "- Global best external mean F1: `nan`",
        f"- Phase 3 reference rows: `{len(phase3_refs)}`",
        f"- Phase 3 refs have no all-NaN reference columns: `{not phase3_refs.drop(columns=['dataset'], errors='ignore').isna().all().any() if not phase3_refs.empty else False}`",
        "",
        "## wDBA Audit",
        "",
        "The historical `fallback_count` mixed class-size replacement with any true DBA/tau/barycenter fallback. "
        "This audit separates class-size replacement exactly from the train split and reports the remaining count as true fallback.",
        "",
    ]
    if not wdba_all.empty:
        lines.extend(
            [
                f"- Total with-replacement count: `{int(pd.to_numeric(wdba_all['with_replacement_count'], errors='coerce').fillna(0).sum())}`",
                f"- Total true fallback count estimate: `{int(pd.to_numeric(wdba_all['true_fallback_count'], errors='coerce').fillna(0).sum())}`",
                f"- Detail entropy/distance available: `{bool(wdba_all['audit_detail_available'].fillna(False).any())}`",
            ]
        )
    else:
        lines.append("- No wDBA rows found.")
    lines.extend(["", "## SPAWNER-style Audit", ""])
    if not spawner_all.empty:
        lines.extend(
            [
                f"- Total same-class scarcity count: `{int(pd.to_numeric(spawner_all['same_class_scarcity_count'], errors='coerce').fillna(0).sum())}`",
                f"- Total DTW-path true fallback estimate: `{int(pd.to_numeric(spawner_all['dtw_path_true_fallback_count'], errors='coerce').fillna(0).sum())}`",
                f"- Detail path-level audit available: `{bool(spawner_all['audit_detail_available'].fillna(False).any())}`",
            ]
        )
    else:
        lines.append("- No SPAWNER-style rows found.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(args) -> None:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    phase1 = _success(_read_csv(Path(args.phase1_root) / "per_seed_external.csv"))
    phase2 = _success(_read_csv(Path(args.phase2_root) / "per_seed_external.csv"))
    phase3 = _success(_read_csv(Path(args.phase3_root) / "per_seed_external.csv", required=False))

    phase1["phase"] = "phase1_locked"
    phase2["phase"] = "phase2_new"
    if not phase3.empty:
        phase3["phase"] = "phase3_case_study"

    phase12_actual = pd.concat([phase1, phase2], ignore_index=True, sort=False)
    best_rawaug = _derive_best_rawaug(phase12_actual)
    best_rawaug["phase"] = "derived_best_rawaug"
    main_rows = pd.concat([phase12_actual, best_rawaug], ignore_index=True, sort=False)

    meta = _metadata_from_rows(main_rows)
    dataset_summary = _dataset_summary(main_rows)
    phase1_dataset_summary = _dataset_summary(phase1)
    phase2_dataset_summary = _dataset_summary(phase2)

    external_methods = _external_methods(dataset_summary["method"].unique())
    global_best_method, global_best_mean = _best_global_method(dataset_summary, external_methods)
    dataset_best_external = _best_by_dataset(dataset_summary, external_methods)
    rank_bundle = _rank_bundle(dataset_summary)

    overall = _build_overall_metrics(
        main_rows,
        dataset_summary,
        meta,
        dataset_best_external,
        global_best_method,
        global_best_mean,
        rank_bundle.rank_rows,
    )
    dataset_metrics = _build_dataset_metrics(dataset_summary, rank_bundle.ranks, dataset_best_external)
    csta_vs_external = _build_csta_vs_external(dataset_summary)
    average_rank = _build_average_rank_tables(phase1, phase2, dataset_summary)
    negative_transfer = _build_negative_transfer(dataset_metrics, main_rows)
    cost_pareto = _build_cost_pareto(dataset_summary)

    datasets = sorted(set(main_rows["dataset"]).union(set(phase3["dataset"]) if not phase3.empty else set()))
    seeds = sorted(int(s) for s in pd.to_numeric(main_rows["seed"], errors="coerce").dropna().unique())
    dataset_meta, split_manifest = _load_dataset_meta(datasets, seeds, val_ratio=float(args.val_ratio))
    regime = _build_regime_table(dataset_metrics, dataset_meta)

    phase3_refs = _build_phase3_refs(
        phase3,
        dataset_summary,
        phase1_dataset_summary,
        phase2_dataset_summary,
        dataset_best_external,
    )
    wdba_audit = _build_wdba_audit(phase2, val_ratio=float(args.val_ratio))
    spawner_audit = _build_spawner_audit(phase2, val_ratio=float(args.val_ratio))
    mechanism = _build_mechanism_diagnostics(main_rows)
    hyperparams = _build_hyperparam_template()

    overall.to_csv(out_root / "protocol_overall_metrics.csv", index=False)
    dataset_metrics.to_csv(out_root / "protocol_dataset_metrics.csv", index=False)
    csta_vs_external.to_csv(out_root / "protocol_csta_vs_external.csv", index=False)
    average_rank.to_csv(out_root / "protocol_average_rank.csv", index=False)
    negative_transfer.to_csv(out_root / "protocol_negative_transfer.csv", index=False)
    cost_pareto.to_csv(out_root / "protocol_cost_pareto.csv", index=False)
    regime.to_csv(out_root / "protocol_regime_table.csv", index=False)
    phase3_refs.to_csv(out_root / "phase3_case_study_refs.csv", index=False)
    wdba_audit.to_csv(out_root / "wdba_audit.csv", index=False)
    spawner_audit.to_csv(out_root / "spawner_audit.csv", index=False)
    dataset_meta.to_csv(out_root / "dataset_meta.csv", index=False)
    split_manifest.to_csv(out_root / "split_manifest.csv", index=False)

    dataset_metrics[["dataset", "method", "gain_vs_no_aug"]].to_csv(out_root / "figure_delta_heatmap.csv", index=False)
    average_rank.to_csv(out_root / "figure_rank_plot.csv", index=False)
    cost_pareto.to_csv(out_root / "figure_cost_pareto.csv", index=False)
    mechanism.to_csv(out_root / "figure_csta_mechanism_diagnostics.csv", index=False)
    hyperparams.to_csv(out_root / "figure_hyperparam_template.csv", index=False)
    _write_validity_report(
        out_root / "external_baseline_validity_report.md",
        global_best_method=global_best_method,
        global_best_mean=global_best_mean,
        phase3_refs=phase3_refs,
        wdba_audit=wdba_audit,
        spawner_audit=spawner_audit,
    )

    print(f"Wrote CSTA protocol summaries to {out_root}")
    print(f"Global best external: {global_best_method} ({global_best_mean:.6f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified CSTA protocol summaries from locked external baseline roots.")
    parser.add_argument("--phase1-root", type=str, default=str(DEFAULT_PHASE1_ROOT))
    parser.add_argument("--phase2-root", type=str, default=str(DEFAULT_PHASE2_ROOT))
    parser.add_argument("--phase3-root", type=str, default=str(DEFAULT_PHASE3_ROOT))
    parser.add_argument("--out-root", type=str, default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
