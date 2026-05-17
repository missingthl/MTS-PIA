#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


CS_FLOW_COLLAPSE_REFERENCE = 0.976901

REF_METHODS = {
    "u5": "csta_topk_uniform_top5",
    "random_cov": "random_cov_state",
    "bank_random": "csta_template_random_within_bank",
    "latent_direct": "latent_residual_direct",
    "latent_flow": "latent_residual_flow",
    "wdba": "wdba_sameclass",
}

DIAG_FIELDS = [
    "task_utility_mean",
    "task_utility_std",
    "task_margin_mean",
    "task_margin_std",
    "task_bad_margin_mass",
    "task_wrong_pred_mass",
    "task_sampling_entropy",
    "task_sampling_effective_support",
    "task_kl_task_vs_geo",
    "task_guidance_fallback_rate",
    "task_invalid_candidate_rate",
    "task_warmup_train_loss_mean",
    "latent_train_mse_mean",
    "latent_train_pred_target_cosine_mean",
    "latent_train_cosine_loss_mean",
    "latent_generated_direction_pairwise_cosine_mean",
    "latent_unique_direction_ratio",
    "latent_effective_aug_multiplier",
    "latent_residual_effective_rank",
    "latent_residual_pairwise_cosine_mean",
    "bridge_success_rate",
    "gamma_used_ratio_mean",
    "gamma_used_ratio_mean_audit",
    "safe_clip_rate",
    "transport_error_logeuc_mean",
]


def _bootstrap_ci(data: List[float], n_bootstrap: int = 2000, seed: int = 24680) -> tuple[float, float]:
    if not data or len(data) < 2:
        return (np.nan, np.nan)
    arr = np.asarray(data, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_bootstrap)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _fmt(val: object) -> str:
    try:
        f = float(val)
    except Exception:
        return "n/a"
    if not np.isfinite(f):
        return "n/a"
    return f"{f:.6g}"


def _markdown_table(headers: List[str], rows: List[List[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def _first_finite(row: Dict[str, object], *keys: str) -> float:
    for key in keys:
        try:
            value = float(row.get(key, np.nan))
        except Exception:
            value = np.nan
        if np.isfinite(value):
            return value
    return np.nan


def _paired_stats(df: pd.DataFrame, method: str, ref: str, *, index_cols: List[str]) -> Dict[str, object]:
    common = df[df["method"].isin([method, ref])].pivot_table(
        index=index_cols, columns="method", values="aug_f1", aggfunc="mean"
    ).dropna()
    if common.empty or method not in common or ref not in common:
        return {
            "n": 0,
            "mean_delta": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "wilcoxon_p": np.nan,
        }
    delta = (common[method] - common[ref]).to_numpy(dtype=np.float64)
    wins = int(np.sum(delta > 1e-6))
    ties = int(np.sum(np.abs(delta) <= 1e-6))
    losses = int(np.sum(delta < -1e-6))
    ci_low, ci_high = _bootstrap_ci(delta.tolist())
    p_val = np.nan
    if len(delta) >= 2 and np.any(np.abs(delta) > 1e-12):
        try:
            p_val = float(wilcoxon(delta, zero_method="wilcox").pvalue)
        except ValueError:
            p_val = np.nan
    return {
        "n": int(len(delta)),
        "mean_delta": float(np.mean(delta)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "wilcoxon_p": p_val,
    }


def build_report(out_root: Path) -> None:
    source = out_root / "per_seed_external.csv"
    if not source.exists():
        raise FileNotFoundError(f"{source} not found")
    df = pd.read_csv(source)
    df = df[df["status"] == "success"].copy()
    if df.empty:
        raise RuntimeError(f"No success rows in {source}")

    def get_mean(method: str) -> float:
        vals = df[df["method"] == method]["aug_f1"].dropna()
        return float(vals.mean()) if not vals.empty else np.nan

    ref_f1s = {name: get_mean(method) for name, method in REF_METHODS.items()}
    rows = []
    for method in sorted(df["method"].unique()):
        m_df = df[df["method"] == method]
        f1s = m_df["aug_f1"].dropna().to_numpy(dtype=np.float64)
        if f1s.size == 0:
            continue
        low, high = _bootstrap_ci(f1s.tolist())
        row = {
            "method": method,
            "debug_probe": bool(method == "task_guided_residual_direct"),
            "mean_f1": float(np.mean(f1s)),
            "ci_low": low,
            "ci_high": high,
            "delta_vs_u5": float(np.mean(f1s) - ref_f1s["u5"]),
            "delta_vs_random_cov": float(np.mean(f1s) - ref_f1s["random_cov"]),
            "delta_vs_bank_random": float(np.mean(f1s) - ref_f1s["bank_random"]),
            "delta_vs_latent_direct": float(np.mean(f1s) - ref_f1s["latent_direct"]),
            "delta_vs_latent_flow": float(np.mean(f1s) - ref_f1s["latent_flow"]),
            "delta_vs_wdba": float(np.mean(f1s) - ref_f1s["wdba"]),
        }
        for field in DIAG_FIELDS:
            row[field] = float(pd.to_numeric(m_df[field], errors="coerce").mean()) if field in m_df else np.nan
        for ref_name, ref_method in REF_METHODS.items():
            seed_stats = _paired_stats(df, method, ref_method, index_cols=["dataset", "seed"])
            dataset_stats = _paired_stats(df, method, ref_method, index_cols=["dataset"])
            row[f"seed_wins_vs_{ref_name}"] = seed_stats["wins"]
            row[f"seed_ties_vs_{ref_name}"] = seed_stats["ties"]
            row[f"seed_losses_vs_{ref_name}"] = seed_stats["losses"]
            row[f"dataset_wins_vs_{ref_name}"] = dataset_stats["wins"]
            row[f"dataset_ties_vs_{ref_name}"] = dataset_stats["ties"]
            row[f"dataset_losses_vs_{ref_name}"] = dataset_stats["losses"]
            row[f"paired_delta_vs_{ref_name}_mean"] = seed_stats["mean_delta"]
            row[f"paired_delta_vs_{ref_name}_ci_low"] = seed_stats["ci_low"]
            row[f"paired_delta_vs_{ref_name}_ci_high"] = seed_stats["ci_high"]
            row[f"wilcoxon_p_vs_{ref_name}"] = seed_stats["wilcoxon_p"]
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("mean_f1", ascending=False)
    summary.to_csv(out_root / "task_guided_latent_residual_summary.csv", index=False)

    report = ["# Task-Guided Latent Residual Flow Report", ""]
    report.append(
        "`task_guided_residual_direct` is a debug-only probe for task-reweighted train-only residual targets; "
        "it is not a competing method or paper baseline."
    )
    report.append("")

    headers = [
        "Method",
        "Probe?",
        "Mean F1",
        "95% CI",
        "Delta U5",
        "Delta Random",
        "Delta Bank",
        "Delta Latent Flow",
        "Delta wDBA",
        "Seed W/T/L vs U5",
    ]
    table_rows = []
    for r in summary.to_dict("records"):
        table_rows.append(
            [
                r["method"],
                "yes" if r["debug_probe"] else "no",
                _fmt(r["mean_f1"]),
                f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]",
                _fmt(r["delta_vs_u5"]),
                _fmt(r["delta_vs_random_cov"]),
                _fmt(r["delta_vs_bank_random"]),
                _fmt(r["delta_vs_latent_flow"]),
                _fmt(r["delta_vs_wdba"]),
                f"{r['seed_wins_vs_u5']}/{r['seed_ties_vs_u5']}/{r['seed_losses_vs_u5']}",
            ]
        )
    report.append("## Leaderboard")
    report.append(_markdown_table(headers, table_rows))
    report.append("")

    diag_headers = [
        "Method",
        "Task Entropy",
        "Eff Support",
        "KL Task/Geo",
        "Bad Margin Mass",
        "Wrong Pred Mass",
        "Fallback",
        "Invalid",
        "Gen Cos",
        "Eff Mult",
    ]
    diag_rows = []
    for r in summary.to_dict("records"):
        if str(r["method"]).startswith("task_guided_"):
            diag_rows.append(
                [
                    r["method"],
                    _fmt(r.get("task_sampling_entropy")),
                    _fmt(r.get("task_sampling_effective_support")),
                    _fmt(r.get("task_kl_task_vs_geo")),
                    _fmt(r.get("task_bad_margin_mass")),
                    _fmt(r.get("task_wrong_pred_mass")),
                    _fmt(r.get("task_guidance_fallback_rate")),
                    _fmt(r.get("task_invalid_candidate_rate")),
                    _fmt(r.get("latent_generated_direction_pairwise_cosine_mean")),
                    _fmt(r.get("latent_effective_aug_multiplier")),
                ]
            )
    report.append("## Task Guidance Diagnostics")
    report.append(_markdown_table(diag_headers, diag_rows))
    report.append("")

    fit_headers = [
        "Method",
        "Train MSE",
        "Train Cos",
        "Utility Mean",
        "Margin Mean",
        "Warmup Loss",
        "Bridge Success",
        "Gamma Ratio",
        "Safe Clip",
        "Transport Err",
    ]
    fit_rows = []
    for r in summary.to_dict("records"):
        if str(r["method"]).startswith("task_guided_"):
            fit_rows.append(
                [
                    r["method"],
                    _fmt(r.get("latent_train_mse_mean")),
                    _fmt(r.get("latent_train_pred_target_cosine_mean")),
                    _fmt(r.get("task_utility_mean")),
                    _fmt(r.get("task_margin_mean")),
                    _fmt(r.get("task_warmup_train_loss_mean")),
                    _fmt(r.get("bridge_success_rate")),
                    _fmt(_first_finite(r, "gamma_used_ratio_mean", "gamma_used_ratio_mean_audit")),
                    _fmt(r.get("safe_clip_rate")),
                    _fmt(r.get("transport_error_logeuc_mean")),
                ]
            )
    report.append("## Fitting And Safety")
    report.append(_markdown_table(fit_headers, fit_rows))
    report.append("")

    report.append("## Interpretation Notes")
    task_flow = next((r for r in rows if r["method"] == "task_guided_latent_residual_flow"), None)
    task_direct = next((r for r in rows if r["method"] == "task_guided_residual_direct"), None)
    if task_flow:
        eff_support = float(task_flow.get("task_sampling_effective_support", np.nan))
        if np.isfinite(eff_support) and eff_support <= 1.5:
            report.append("- Task sampling effective support is near 1; treat this as selector-collapse risk before interpreting F1.")
        gen_cos = float(task_flow.get("latent_generated_direction_pairwise_cosine_mean", np.nan))
        if np.isfinite(gen_cos):
            if gen_cos >= CS_FLOW_COLLAPSE_REFERENCE:
                report.append("- Direction collapse remains comparable to or worse than CS-Flow reference.")
            else:
                report.append("- Direction collapse is below the CS-Flow reference; inspect whether F1 also improves.")
        report.append(
            f"- task_guided_latent_residual_flow deltas: vs latent_flow={_fmt(task_flow.get('delta_vs_latent_flow'))}, "
            f"vs U5={_fmt(task_flow.get('delta_vs_u5'))}, vs random_cov={_fmt(task_flow.get('delta_vs_random_cov'))}."
        )
    if task_direct and task_flow:
        report.append(
            f"- Direct probe gap (flow - direct): {_fmt(task_flow.get('delta_vs_latent_direct'))}. "
            "If direct improves but flow does not, inspect fitting before changing the task utility."
        )

    (out_root / "task_guided_latent_residual_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    build_report(Path(args.out_root))


if __name__ == "__main__":
    main()
