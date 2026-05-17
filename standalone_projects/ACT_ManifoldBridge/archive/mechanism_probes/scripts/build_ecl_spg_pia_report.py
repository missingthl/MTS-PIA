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
    "task_guided": "task_guided_latent_residual_flow",
    "lc_flow": "lc_latent_residual_flow",
    "wdba": "wdba_sameclass",
}

DIAG_FIELDS = [
    "spg_zhead_train_acc",
    "spg_grad_norm_mean",
    "spg_projected_grad_norm_mean",
    "spg_projection_energy",
    "spg_direction_pairwise_cosine_mean",
    "spg_effective_aug_multiplier",
    "spg_support_rank",
    "ecl_projection_energy_mean",
    "ecl_alpha_mean",
    "ecl_alignment_to_projected_gradient_mean",
    "ecl_direction_pairwise_cosine_mean",
    "ecl_effective_aug_multiplier",
    "ecl_support_rank",
    "ecl_support_noise_norm_mean",
    "ecl_fallback_rate",
    "bridge_success_rate",
    "gamma_used_ratio_mean",
    "gamma_used_ratio_mean_audit",
    "safe_clip_rate",
    "transport_error_logeuc_mean",
]


def _bootstrap_ci(data: List[float], n_bootstrap: int = 2000, seed: int = 9876) -> tuple[float, float]:
    if len(data) < 2:
        return (np.nan, np.nan)
    arr = np.asarray(data, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_bootstrap)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _fmt(v: object) -> str:
    try:
        f = float(v)
    except Exception:
        return "n/a"
    if not np.isfinite(f):
        return "n/a"
    return f"{f:.6g}"


def _first_finite(*values: object) -> object:
    for value in values:
        try:
            f = float(value)
        except Exception:
            continue
        if np.isfinite(f):
            return f
    return np.nan


def _table(headers: List[str], rows: List[List[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def _paired(df: pd.DataFrame, method: str, ref: str, index_cols: List[str]) -> Dict[str, object]:
    common = df[df.method.isin([method, ref])].pivot_table(index=index_cols, columns="method", values="aug_f1").dropna()
    if common.empty or method not in common or ref not in common:
        return {"wins": 0, "ties": 0, "losses": 0, "mean": np.nan, "low": np.nan, "high": np.nan, "p": np.nan}
    delta = (common[method] - common[ref]).to_numpy(dtype=np.float64)
    low, high = _bootstrap_ci(delta.tolist())
    p = np.nan
    if len(delta) >= 2 and np.any(np.abs(delta) > 1e-12):
        try:
            p = float(wilcoxon(delta, zero_method="wilcox").pvalue)
        except ValueError:
            p = np.nan
    return {
        "wins": int(np.sum(delta > 1e-6)),
        "ties": int(np.sum(np.abs(delta) <= 1e-6)),
        "losses": int(np.sum(delta < -1e-6)),
        "mean": float(np.mean(delta)),
        "low": low,
        "high": high,
        "p": p,
    }


def build_report(out_root: Path) -> None:
    source = out_root / "per_seed_external.csv"
    df = pd.read_csv(source)
    df = df[df.status == "success"].copy()
    refs = {name: float(df[df.method == method].aug_f1.mean()) for name, method in REF_METHODS.items()}
    rows = []
    for method in sorted(df.method.unique()):
        mdf = df[df.method == method]
        f1 = mdf.aug_f1.to_numpy(dtype=np.float64)
        low, high = _bootstrap_ci(f1.tolist())
        row = {
            "method": method,
            "debug_probe": method == "spg_pia_zhead_deterministic",
            "mean_f1": float(np.mean(f1)),
            "ci_low": low,
            "ci_high": high,
        }
        for ref_name, ref_method in REF_METHODS.items():
            row[f"delta_vs_{ref_name}"] = float(row["mean_f1"] - refs[ref_name])
            seed = _paired(df, method, ref_method, ["dataset", "seed"])
            dataset = _paired(df, method, ref_method, ["dataset"])
            row[f"seed_wtl_vs_{ref_name}"] = f"{seed['wins']}/{seed['ties']}/{seed['losses']}"
            row[f"dataset_wtl_vs_{ref_name}"] = f"{dataset['wins']}/{dataset['ties']}/{dataset['losses']}"
            row[f"wilcoxon_p_vs_{ref_name}"] = seed["p"]
        for field in DIAG_FIELDS:
            row[field] = float(pd.to_numeric(mdf[field], errors="coerce").mean()) if field in mdf else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("mean_f1", ascending=False)
    summary.to_csv(out_root / "ecl_spg_pia_summary.csv", index=False)

    report = ["# ECL-SPG-PIA Report", "", "`ecl_spg_pia_zhead_deterministic` is diagnostic only. SPG reference cosine is 0.991163."]
    table_rows = []
    for r in summary.to_dict("records"):
        table_rows.append([
            r["method"],
            "yes" if r["debug_probe"] else "no",
            _fmt(r["mean_f1"]),
            f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]",
            _fmt(r["delta_vs_u5"]),
            _fmt(r["delta_vs_random_cov"]),
            _fmt(r["delta_vs_bank_random"]),
            r["seed_wtl_vs_u5"],
        ])
    report.append("## Leaderboard")
    report.append(_table(["Method", "Probe?", "Mean F1", "95% CI", "Delta U5", "Delta Random", "Delta Bank", "Seed W/T/L vs U5"], table_rows))
    report.append("")
    diag_rows = []
    for r in summary.to_dict("records"):
        if str(r["method"]).startswith("spg_pia_") or str(r["method"]).startswith("ecl_spg_pia_"):
            diag_rows.append([
                r["method"],
                _fmt(r.get("spg_zhead_train_acc")),
                _fmt(_first_finite(r.get("ecl_projection_energy_mean"), r.get("spg_projection_energy"))),
                _fmt(r.get("ecl_alpha_mean")),
                _fmt(r.get("ecl_alignment_to_projected_gradient_mean")),
                _fmt(_first_finite(r.get("ecl_direction_pairwise_cosine_mean"), r.get("spg_direction_pairwise_cosine_mean"))),
                _fmt(_first_finite(r.get("ecl_effective_aug_multiplier"), r.get("spg_effective_aug_multiplier"))),
                _fmt(_first_finite(r.get("ecl_support_rank"), r.get("spg_support_rank"))),
                _fmt(r.get("bridge_success_rate")),
                _fmt(r.get("safe_clip_rate")),
            ])
    report.append("## SPG/ECL Diagnostics")
    report.append(_table(["Method", "z-head acc", "Proj Energy", "ECL alpha", "ECL align", "Dir Cos", "Eff Mult", "Support Rank", "Bridge", "Safe Clip"], diag_rows))
    (out_root / "ecl_spg_pia_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    build_report(Path(args.out_root))


if __name__ == "__main__":
    main()
