#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


REF_METHODS = {
    "u5": "csta_topk_uniform_top5",
    "random_cov": "random_cov_state",
    "bank_random": "csta_template_random_within_bank",
    "latent_flow": "latent_residual_flow",
    "task_guided_flow": "task_guided_latent_residual_flow",
    "wdba": "wdba_sameclass",
}

DIAG_FIELDS = [
    "lc_valid_candidate_rate",
    "lc_no_valid_fallback_rate",
    "lc_bad_margin_mass",
    "lc_wrong_pred_mass",
    "lc_sampling_entropy",
    "lc_sampling_effective_support",
    "lc_kl_lc_vs_geo",
    "lc_margin_mean",
    "lc_margin_std",
    "lc_margin_target_mean",
    "lc_weight_top1_mass",
    "latent_generated_direction_pairwise_cosine_mean",
    "latent_effective_aug_multiplier",
    "latent_train_mse_mean",
    "latent_train_pred_target_cosine_mean",
    "bridge_success_rate",
    "gamma_used_ratio_mean",
    "gamma_used_ratio_mean_audit",
    "safe_clip_rate",
    "transport_error_logeuc_mean",
]


def _bootstrap_ci(data: List[float], n_bootstrap: int = 2000, seed: int = 13579) -> tuple[float, float]:
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


def _paired_stats(df: pd.DataFrame, method: str, ref: str, *, index_cols: List[str]) -> Dict[str, object]:
    common = df[df["method"].isin([method, ref])].pivot_table(
        index=index_cols, columns="method", values="aug_f1", aggfunc="mean"
    ).dropna()
    if common.empty or method not in common or ref not in common:
        return {"wins": 0, "ties": 0, "losses": 0, "mean_delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "wilcoxon_p": np.nan}
    delta = (common[method] - common[ref]).to_numpy(dtype=np.float64)
    ci_low, ci_high = _bootstrap_ci(delta.tolist())
    p_val = np.nan
    if len(delta) >= 2 and np.any(np.abs(delta) > 1e-12):
        try:
            p_val = float(wilcoxon(delta, zero_method="wilcox").pvalue)
        except ValueError:
            p_val = np.nan
    return {
        "wins": int(np.sum(delta > 1e-6)),
        "ties": int(np.sum(np.abs(delta) <= 1e-6)),
        "losses": int(np.sum(delta < -1e-6)),
        "mean_delta": float(np.mean(delta)),
        "ci_low": ci_low,
        "ci_high": ci_high,
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

    refs = {name: get_mean(method) for name, method in REF_METHODS.items()}
    rows = []
    for method in sorted(df["method"].unique()):
        m_df = df[df["method"] == method]
        f1s = m_df["aug_f1"].dropna().to_numpy(dtype=np.float64)
        if f1s.size == 0:
            continue
        low, high = _bootstrap_ci(f1s.tolist())
        row = {
            "method": method,
            "debug_probe": bool(method == "lc_residual_direct"),
            "mean_f1": float(np.mean(f1s)),
            "ci_low": low,
            "ci_high": high,
            "delta_vs_u5": float(np.mean(f1s) - refs["u5"]),
            "delta_vs_random_cov": float(np.mean(f1s) - refs["random_cov"]),
            "delta_vs_bank_random": float(np.mean(f1s) - refs["bank_random"]),
            "delta_vs_latent_flow": float(np.mean(f1s) - refs["latent_flow"]),
            "delta_vs_task_guided_flow": float(np.mean(f1s) - refs["task_guided_flow"]),
            "delta_vs_wdba": float(np.mean(f1s) - refs["wdba"]),
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
    summary.to_csv(out_root / "lc_latent_residual_summary.csv", index=False)

    report = ["# Label-Consistent Latent Residual Flow Report", ""]
    report.append("`lc_residual_direct` is a debug-only probe, not a competing paper baseline.")
    report.append("")
    headers = ["Method", "Probe?", "Mean F1", "95% CI", "Delta U5", "Delta Random", "Delta Bank", "Delta Task", "Seed W/T/L vs U5"]
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
                _fmt(r["delta_vs_task_guided_flow"]),
                f"{r['seed_wins_vs_u5']}/{r['seed_ties_vs_u5']}/{r['seed_losses_vs_u5']}",
            ]
        )
    report.append("## Leaderboard")
    report.append(_markdown_table(headers, table_rows))
    report.append("")

    diag_rows = []
    for r in summary.to_dict("records"):
        if str(r["method"]).startswith("lc_"):
            diag_rows.append(
                [
                    r["method"],
                    _fmt(r.get("lc_valid_candidate_rate")),
                    _fmt(r.get("lc_no_valid_fallback_rate")),
                    _fmt(r.get("lc_wrong_pred_mass")),
                    _fmt(r.get("lc_bad_margin_mass")),
                    _fmt(r.get("lc_sampling_effective_support")),
                    _fmt(r.get("lc_weight_top1_mass")),
                    _fmt(r.get("latent_generated_direction_pairwise_cosine_mean")),
                    _fmt(r.get("latent_effective_aug_multiplier")),
                ]
            )
    report.append("## LC Diagnostics")
    report.append(_markdown_table(["Method", "Valid Rate", "No-valid Fallback", "Wrong Mass", "Bad Margin", "Eff Support", "Top1 Mass", "Gen Cos", "Eff Mult"], diag_rows))
    report.append("")

    lc_flow = next((r for r in rows if r["method"] == "lc_latent_residual_flow"), None)
    if lc_flow:
        report.append("## Gate Notes")
        report.append(
            f"- Wrong-pred mass: {_fmt(lc_flow.get('lc_wrong_pred_mass'))}; "
            f"bad-margin mass: {_fmt(lc_flow.get('lc_bad_margin_mass'))}."
        )
        report.append(
            f"- Deltas: vs task-guided={_fmt(lc_flow.get('delta_vs_task_guided_flow'))}, "
            f"vs latent={_fmt(lc_flow.get('delta_vs_latent_flow'))}, "
            f"vs U5={_fmt(lc_flow.get('delta_vs_u5'))}, "
            f"vs random={_fmt(lc_flow.get('delta_vs_random_cov'))}."
        )
        if float(lc_flow.get("lc_sampling_effective_support", np.nan)) <= 1.5:
            report.append("- Effective support is near 1; treat as sampling collapse.")
    (out_root / "lc_latent_residual_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    build_report(Path(args.out_root))


if __name__ == "__main__":
    main()
