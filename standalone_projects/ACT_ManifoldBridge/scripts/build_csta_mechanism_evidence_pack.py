from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_ROOT = PROJECT_ROOT / "results" / "local_tangent_audit_v1" / "resnet1d_s123"
DEFAULT_SELECTOR_ROOT = PROJECT_ROOT / "results" / "csta_selector_ablation_v1" / "resnet1d_s123"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "results" / "csta_mechanism_evidence_v1" / "resnet1d_s123"


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _performance(selector_root: Path) -> pd.DataFrame:
    for name in [
        "pilot7_full_external_vs_csta_performance_table.csv",
        "pilot7_external_vs_csta_performance_table.csv",
        "selector_ablation_summary.csv",
    ]:
        df = _read(selector_root / name)
        if not df.empty:
            return df
    return pd.DataFrame()


def _per_seed_perf(selector_root: Path) -> pd.DataFrame:
    for name in ["per_seed_selector_with_refs.csv", "per_seed_selector_ablation.csv"]:
        df = _read(selector_root / name)
        if not df.empty:
            return df
    return pd.DataFrame()


def _mechanism_main(perf: pd.DataFrame, local_overall: pd.DataFrame, local_per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    perf_cols = set(perf.columns)
    for _, p in perf.iterrows() if not perf.empty else []:
        method = str(p.get("method", ""))
        if not method:
            continue
        out = {
            "method": method,
            "display_name": p.get("display_name", method),
            "family": p.get("family", ""),
            "mean_f1_over_datasets": p.get("mean_f1_over_datasets", p.get("mean_f1", np.nan)),
            "mean_gain_vs_no_aug": p.get("mean_gain_vs_no_aug", np.nan),
            "delta_vs_csta_uniform_top5": p.get("delta_vs_csta_uniform_top5", np.nan)
            if "delta_vs_csta_uniform_top5" in perf_cols
            else np.nan,
        }
        method_seed = local_per_seed[local_per_seed["method"] == method] if not local_per_seed.empty else pd.DataFrame()
        source_map = {
            "pia_selected": "selected_alignment_mean",
            "random_cov": "random_alignment_mean",
            "pca_cov": "pca_alignment_mean",
        }
        for source, col in source_map.items():
            # Dataset-Seed equal-weighted mean (Unified Evidence Layer Policy)
            # method_seed is already at per-seed granularity if the input CSV is per-seed
            out[f"{source}_tangent_alignment"] = (
                float(pd.to_numeric(method_seed[col], errors="coerce").mean()) if col in method_seed else np.nan
            )
        for col in [
            "selected_minus_random_alignment",
            "selected_minus_pca_alignment",
            "top1_alignment_mean",
            "top5_alignment_mean",
            "selected_alignment_rank_within_top5_mean",
            "selected_alignment_minus_top5_mean",
            "actual_candidate_audit_available",
        ]:
            out[col] = float(pd.to_numeric(method_seed[col], errors="coerce").mean()) if col in method_seed else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def _tradeoff(per_seed_perf: pd.DataFrame, local_per_seed: pd.DataFrame) -> pd.DataFrame:
    if local_per_seed.empty:
        return pd.DataFrame()
    keep = local_per_seed[
        local_per_seed["method"].isin(["csta_top1_current", "csta_topk_uniform_top5"])
    ].copy()
    perf_cols = ["dataset", "seed", "method", "aug_f1", "gain", "template_usage_entropy", "top_template_concentration"]
    if not per_seed_perf.empty and {"dataset", "seed", "method"}.issubset(per_seed_perf.columns):
        cols = [c for c in perf_cols if c in per_seed_perf.columns]
        keep = keep.merge(per_seed_perf[cols], on=["dataset", "seed", "method"], how="left")
    return keep[
        [
            c
            for c in [
                "dataset",
                "seed",
                "method",
                "aug_f1",
                "gain",
                "selected_alignment_mean",
                "selected_minus_random_alignment",
                "selected_minus_pca_alignment",
                "top1_alignment_mean",
                "top5_alignment_mean",
                "top5_alignment_std_mean",
                "selected_alignment_rank_within_top5_mean",
                "selected_alignment_minus_top5_mean",
                "template_usage_entropy",
                "top_template_concentration",
                "audit_source",
                "actual_candidate_audit_available",
            ]
            if c in keep.columns
        ]
    ]


def _write_report(out_root: Path, main: pd.DataFrame, tangent: pd.DataFrame, tradeoff: pd.DataFrame) -> None:
    lines = [
        "# CSTA Mechanism Evidence Pack",
        "",
        "This pack combines post-hoc local tangent alignment with pilot7 performance summaries.",
        "",
        "## Key Interpretation",
        "",
        "- Local tangent alignment is a mechanism diagnostic, not a selector or training component.",
        "- PIA directions should be interpreted as tangent-relevant proposal directions when they exceed random covariance directions.",
        "- Higher alignment alone is not sufficient utility evidence; UniformTop5 may trade peak alignment for high-response neighborhood diversity.",
        "",
    ]
    if not main.empty:
        csta = main[main["method"] == "csta_topk_uniform_top5"]
        top1 = main[main["method"] == "csta_top1_current"]
        if not csta.empty:
            lines.append(
                f"- CSTA UniformTop5 mean F1: {float(csta['mean_f1_over_datasets'].iloc[0]):.6f}; "
                f"PIA selected alignment: {float(csta['pia_selected_tangent_alignment'].iloc[0]):.6f}."
            )
        if not top1.empty:
            lines.append(
                f"- CSTA Top1 selected alignment: {float(top1['pia_selected_tangent_alignment'].iloc[0]):.6f}; "
                "compare this with performance to avoid overclaiming alignment as utility."
            )
    lines.extend(["", "## Files", ""])
    for name in [
        "mechanism_main_table.csv",
        "tangent_alignment_table.csv",
        "diversity_alignment_tradeoff.csv",
        "alignment_performance_correlation.csv",
    ]:
        lines.append(f"- `{name}`")
    (out_root / "mechanism_evidence_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CSTA mechanism evidence pack.")
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--selector-root", type=Path, default=DEFAULT_SELECTOR_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    local_overall = _read(Path(args.local_root) / "local_tangent_overall_summary.csv")
    local_per_seed = _read(Path(args.local_root) / "local_tangent_per_seed_summary.csv")
    corr = _read(Path(args.local_root) / "local_tangent_alignment_vs_performance.csv")
    perf = _performance(Path(args.selector_root))
    per_seed_perf = _per_seed_perf(Path(args.selector_root))

    main_table = _mechanism_main(perf, local_overall, local_per_seed)
    tradeoff = _tradeoff(per_seed_perf, local_per_seed)

    main_table.to_csv(out_root / "mechanism_main_table.csv", index=False)
    local_overall.to_csv(out_root / "tangent_alignment_table.csv", index=False)
    tradeoff.to_csv(out_root / "diversity_alignment_tradeoff.csv", index=False)
    corr.to_csv(out_root / "alignment_performance_correlation.csv", index=False)
    _write_report(out_root, main_table, local_overall, tradeoff)
    print(f"[mechanism-evidence] wrote {out_root}")


if __name__ == "__main__":
    main()
