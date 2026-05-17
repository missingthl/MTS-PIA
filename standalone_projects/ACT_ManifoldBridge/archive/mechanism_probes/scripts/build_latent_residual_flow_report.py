#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


CS_FLOW_COLLAPSE_REFERENCE = 0.976901

LATENT_DIAG_FIELDS = [
    "latent_train_cosine_loss_mean",
    "latent_train_mse_mean",
    "latent_train_pred_target_cosine_mean",
    "latent_pred_velocity_norm_mean",
    "latent_pred_velocity_norm_std",
    "latent_target_velocity_norm_mean",
    "latent_target_velocity_norm_std",
    "latent_target_dist_mean",
    "latent_target_dist_std",
    "latent_target_sampling_entropy",
    "latent_target_sampling_entropy_by_class_mean",
    "latent_target_sampling_entropy_by_class_min",
    "latent_fallback_rate",
    "latent_residual_effective_rank",
    "latent_residual_pairwise_cosine_mean",
    "latent_generated_direction_pairwise_cosine_mean",
    "latent_unique_direction_ratio",
    "latent_effective_aug_multiplier",
    "bridge_success_rate",
    "gamma_used_ratio_mean",
    "safe_clip_rate",
    "transport_error_logeuc_mean",
]

REF_METHODS = {
    "u5": "csta_topk_uniform_top5",
    "random_cov": "random_cov_state",
    "bank_random": "csta_template_random_within_bank",
    "cs_flow": "cs_flow_single_step",
    "wdba": "wdba_sameclass",
    "direct_probe": "latent_residual_direct",
}


def _bootstrap_ci(data: List[float], n_bootstrap: int = 2000, seed: int = 12345) -> tuple[float, float]:
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
    if np.isnan(f):
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
    summary_rows = []
    for method in sorted(df["method"].unique()):
        m_df = df[df["method"] == method]
        f1s = m_df["aug_f1"].dropna().to_numpy(dtype=np.float64)
        if f1s.size == 0:
            continue
        low, high = _bootstrap_ci(f1s.tolist())
        row = {
            "method": method,
            "debug_probe": bool(method == "latent_residual_direct"),
            "mean_f1": float(np.mean(f1s)),
            "ci_low": low,
            "ci_high": high,
            "delta_vs_u5": float(np.mean(f1s) - ref_f1s["u5"]),
            "delta_vs_random_cov": float(np.mean(f1s) - ref_f1s["random_cov"]),
            "delta_vs_bank_random": float(np.mean(f1s) - ref_f1s["bank_random"]),
            "delta_vs_cs_flow": float(np.mean(f1s) - ref_f1s["cs_flow"]),
            "delta_vs_wdba": float(np.mean(f1s) - ref_f1s["wdba"]),
            "delta_vs_direct_probe": float(np.mean(f1s) - ref_f1s["direct_probe"]),
        }
        for field in LATENT_DIAG_FIELDS:
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
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("mean_f1", ascending=False)
    summary_df.to_csv(out_root / "latent_residual_flow_summary.csv", index=False)

    report = ["# Latent Residual Flow Pilot Report", ""]
    report.append(
        "`latent_residual_direct` is a debug probe for train-only same-class RBF residual targets, "
        "not a competing paper baseline."
    )
    report.append("")
    headers = [
        "Method",
        "Probe?",
        "Mean F1",
        "95% CI",
        "Δ vs U5",
        "Δ vs Random",
        "Δ vs Bank",
        "Δ vs CS-Flow",
        "Δ vs wDBA",
        "Seed W/T/L vs U5",
    ]
    rows = []
    for r in summary_df.to_dict("records"):
        rows.append(
            [
                r["method"],
                "yes" if r["debug_probe"] else "no",
                _fmt(r["mean_f1"]),
                f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]",
                _fmt(r["delta_vs_u5"]),
                _fmt(r["delta_vs_random_cov"]),
                _fmt(r["delta_vs_bank_random"]),
                _fmt(r["delta_vs_cs_flow"]),
                _fmt(r["delta_vs_wdba"]),
                f"{r['seed_wins_vs_u5']}/{r['seed_ties_vs_u5']}/{r['seed_losses_vs_u5']}",
            ]
        )
    report.append("## Leaderboard")
    report.append(_markdown_table(headers, rows))
    report.append("")

    report.append("## Latent Residual Diagnostics")
    diag_headers = [
        "Method",
        "Train MSE",
        "Train Cos",
        "Pred Vel Norm",
        "Residual Rank",
        "Residual Cos",
        "Generated Cos",
        "Effective Mult",
        "Fallback",
    ]
    diag_rows = []
    for r in summary_df.to_dict("records"):
        if str(r["method"]).startswith("latent_residual_"):
            diag_rows.append(
                [
                    r["method"],
                    _fmt(r.get("latent_train_mse_mean")),
                    _fmt(r.get("latent_train_pred_target_cosine_mean")),
                    _fmt(r.get("latent_pred_velocity_norm_mean")),
                    _fmt(r.get("latent_residual_effective_rank")),
                    _fmt(r.get("latent_residual_pairwise_cosine_mean")),
                    _fmt(r.get("latent_generated_direction_pairwise_cosine_mean")),
                    _fmt(r.get("latent_effective_aug_multiplier")),
                    _fmt(r.get("latent_fallback_rate")),
                ]
            )
    report.append(_markdown_table(diag_headers, diag_rows))
    report.append("")

    latent = next((r for r in summary_rows if r["method"] == "latent_residual_flow"), None)
    direct = next((r for r in summary_rows if r["method"] == "latent_residual_direct"), None)
    report.append("## Decision Notes")
    if latent:
        gen_cos = float(latent.get("latent_generated_direction_pairwise_cosine_mean", np.nan))
        if np.isfinite(gen_cos):
            if gen_cos < CS_FLOW_COLLAPSE_REFERENCE:
                report.append(
                    f"- Direction collapse is reduced vs CS-Flow reference "
                    f"({gen_cos:.6f} < {CS_FLOW_COLLAPSE_REFERENCE:.6f})."
                )
            else:
                report.append(
                    f"- Direction collapse is not reduced vs CS-Flow reference "
                    f"({gen_cos:.6f} >= {CS_FLOW_COLLAPSE_REFERENCE:.6f})."
                )
        report.append(
            "- Pilot7 gate should be considered only if latent_residual_flow beats CS-Flow, "
            "reduces collapse, and is competitive with random_cov or close to U5."
        )
        report.append(
            f"- latent_residual_flow Δ vs CS-Flow: {_fmt(latent.get('delta_vs_cs_flow'))}; "
            f"Δ vs U5: {_fmt(latent.get('delta_vs_u5'))}; "
            f"Δ vs random_cov: {_fmt(latent.get('delta_vs_random_cov'))}."
        )
    if direct and latent:
        report.append(
            f"- Direct probe gap (flow - direct): {_fmt(latent.get('delta_vs_direct_probe'))}. "
            "If direct is strong but flow is weak, inspect fitting diagnostics before changing target sampling."
        )

    (out_root / "latent_residual_flow_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    build_report(Path(args.out_root))


if __name__ == "__main__":
    main()
