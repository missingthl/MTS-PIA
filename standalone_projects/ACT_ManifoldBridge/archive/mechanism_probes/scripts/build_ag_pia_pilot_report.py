#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_REFS = [
    "csta_topk_uniform_top5",
    "csta_template_random_within_bank",
    "random_cov_state",
    "pca_cov_state",
    "ag_target_direct",
    "ag_pia_single",
    "ag_pia_multihead5",
    "wdba_sameclass",
    "dba_sameclass",
]


AG_FIELDS = [
    "ag_target_effective_rank",
    "ag_target_pairwise_cosine_mean",
    "ag_target_norm_mean",
    "ag_target_norm_std",
    "ag_head_pairwise_cosine_mean",
    "ag_head_effective_rank",
    "ag_head_usage_entropy",
    "ag_operator_train_mse_mean",
    "ag_operator_train_cosine_mean",
    "ag_pred_target_cosine_mean",
    "ag_tangent_available_rate",
    "ag_fallback_rate",
    "ag_pos_dist_mean",
    "ag_neg_centroid_dist_mean",
]


def _wtl(df: pd.DataFrame, method: str, ref: str) -> Dict[str, int]:
    pivot = df.pivot_table(index=["dataset", "seed"], columns="method", values="aug_f1", aggfunc="mean")
    if method not in pivot or ref not in pivot:
        return {"wins": 0, "ties": 0, "losses": 0}
    delta = (pivot[method] - pivot[ref]).dropna()
    return {
        "wins": int((delta > 1e-12).sum()),
        "ties": int((np.abs(delta) <= 1e-12).sum()),
        "losses": int((delta < -1e-12).sum()),
    }


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append("" if not np.isfinite(val) else f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_report(out_root: Path) -> None:
    per_seed = out_root / "per_seed_external.csv"
    if not per_seed.is_file():
        raise FileNotFoundError(f"Missing per-seed file: {per_seed}")
    df = pd.read_csv(per_seed)
    ok = df[df["status"].fillna("success") == "success"].copy()
    if ok.empty:
        raise RuntimeError("No successful rows found.")
    methods = [m for m in DEFAULT_REFS if m in set(ok["method"])]
    summary_rows: List[Dict[str, object]] = []
    for method, sub in ok.groupby("method"):
        row: Dict[str, object] = {
            "method": method,
            "n_rows": int(len(sub)),
            "n_datasets": int(sub["dataset"].nunique()),
            "n_seeds": int(sub["seed"].nunique()),
            "mean_f1": float(sub["aug_f1"].mean()),
            "std_f1": float(sub["aug_f1"].std()) if len(sub) > 1 else np.nan,
        }
        for field in AG_FIELDS:
            if field in sub:
                vals = pd.to_numeric(sub[field], errors="coerce")
                row[field] = float(vals.mean()) if vals.notna().any() else np.nan
        for ref in ["csta_topk_uniform_top5", "csta_template_random_within_bank", "random_cov_state", "pca_cov_state"]:
            wtl = _wtl(ok, method, ref)
            row[f"wins_vs_{ref}"] = wtl["wins"]
            row[f"ties_vs_{ref}"] = wtl["ties"]
            row[f"losses_vs_{ref}"] = wtl["losses"]
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("mean_f1", ascending=False)
    summary.to_csv(out_root / "ag_pia_pilot_summary.csv", index=False)

    lines = [
        "# AG-PIA Pilot Report",
        "",
        f"Source: `{per_seed}`",
        "",
        "## Leaderboard",
        "",
        _markdown_table(summary[["method", "n_rows", "mean_f1", "std_f1"]]),
        "",
        "## AG Diagnostics",
        "",
    ]
    ag_cols = ["method"] + [f for f in AG_FIELDS if f in summary.columns]
    ag_summary = summary[summary["method"].astype(str).str.startswith("ag_")]
    if not ag_summary.empty:
        lines.append(_markdown_table(ag_summary[ag_cols]))
    else:
        lines.append("No AG-PIA rows found.")
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- `ag_target_direct` is debug-only and must not be presented as a paper baseline.",
            "- If direct target is strong but AG operator is weak, inspect operator train cosine/MSE before abandoning AG-PIA.",
            "- AG-PIA rows are CSTA internal methods, not external baselines.",
        ]
    )
    (out_root / "ag_pia_pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AG-PIA pilot summary/report.")
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.out_root)


if __name__ == "__main__":
    main()
