from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "results" / "local_tangent_audit_v1" / "resnet1d_s123"
DEFAULT_SELECTOR_ROOT = PROJECT_ROOT / "results" / "csta_selector_ablation_v1" / "resnet1d_s123"
DEFAULT_PHASE1_ROOT = PROJECT_ROOT / "results" / "csta_external_baselines_phase1" / "resnet1d_s123"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _collect_run_summaries(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("* /s*/local_tangent_run_summary.csv".replace(" ", ""))):
        df = _read_csv(path)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _collect_candidate_rows(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("* /s*/local_tangent_candidate_audit.csv.gz".replace(" ", ""))):
        df = _read_csv(path)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _dataset_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    if per_seed.empty:
        return pd.DataFrame()
    num_cols = [
        "tangent_available_rate",
        "selected_alignment_mean",
        "random_alignment_mean",
        "pca_alignment_mean",
        "selected_minus_random_alignment",
        "selected_minus_pca_alignment",
        "selected_normal_leakage_mean",
        "random_normal_leakage_mean",
        "pca_normal_leakage_mean",
        "fallback_count",
        "insufficient_neighbor_count",
        "top1_alignment_mean",
        "top5_alignment_mean",
        "top5_alignment_std_mean",
        "selected_alignment_rank_within_top5_mean",
        "selected_alignment_minus_top5_mean",
        "top5_response_mean",
        "top5_response_std_mean",
    ]
    agg: Dict[str, Tuple[str, str]] = {"n_seeds": ("seed", "nunique")}
    for col in num_cols:
        if col in per_seed.columns:
            agg[f"{col}_mean"] = (col, "mean")
            agg[f"{col}_std"] = (col, "std")
    return per_seed.groupby(["dataset", "method"], as_index=False).agg(**agg)


def _overall_summary(candidates: pd.DataFrame, per_seed: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    df = candidates.copy()
    df["tangent_alignment"] = pd.to_numeric(df["tangent_alignment"], errors="coerce")
    df["normal_leakage"] = pd.to_numeric(df["normal_leakage"], errors="coerce")
    rows = []
    for (method, source), sub in df.groupby(["method", "direction_source"], dropna=False):
        anchors = sub.drop_duplicates(["dataset", "seed", "anchor_index"])
        
        # Dataset-Seed equal-weighted mean (Unified Evidence Layer Policy)
        # Each unique (dataset, seed) run contributes equally to the mean
        per_run_align = sub.groupby(["dataset", "seed"])["tangent_alignment"].mean()
        per_run_leak = sub.groupby(["dataset", "seed"])["normal_leakage"].mean()
        
        rows.append(
            {
                "method": method,
                "direction_source": source,
                "mean_tangent_alignment": float(per_run_align.mean()),
                "mean_normal_leakage": float(per_run_leak.mean()),
                "n_datasets": int(sub["dataset"].nunique()),
                "n_seeds": int(sub[["dataset", "seed"]].drop_duplicates().shape[0]),
                "available_rate": float(anchors["tangent_available"].mean()) if "tangent_available" in anchors else np.nan,
                "actual_candidate_audit_available_rate": float(anchors["actual_candidate_audit_available"].mean())
                if "actual_candidate_audit_available" in anchors
                else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if not per_seed.empty:
        diff = (
            per_seed.groupby("method", as_index=False)
            .agg(
                mean_selected_minus_random=("selected_minus_random_alignment", "mean"),
                mean_selected_minus_pca=("selected_minus_pca_alignment", "mean"),
            )
        )
        out = out.merge(diff, on="method", how="left")
    return out.sort_values(["method", "direction_source"]).reset_index(drop=True)


def _pearson(x: pd.Series, y: pd.Series) -> float:
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    common = pd.concat([xx, yy], axis=1).dropna()
    if common.shape[0] < 3:
        return np.nan
    a = common.iloc[:, 0].to_numpy(dtype=np.float64)
    b = common.iloc[:, 1].to_numpy(dtype=np.float64)
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(x: pd.Series, y: pd.Series) -> float:
    xx = pd.to_numeric(x, errors="coerce").rank(method="average")
    yy = pd.to_numeric(y, errors="coerce").rank(method="average")
    return _pearson(xx, yy)


def _load_performance(selector_root: Path, phase1_root: Path) -> pd.DataFrame:
    candidates = [
        selector_root / "per_seed_selector_with_refs.csv",
        selector_root / "per_seed_selector_ablation.csv",
        selector_root / "per_seed_external.csv",
    ]
    frames = []
    for path in candidates:
        df = _read_csv(path)
        if not df.empty and {"dataset", "seed", "method", "aug_f1"}.issubset(df.columns):
            frames.append(df)
            break
    phase1 = _read_csv(phase1_root / "per_seed_external.csv")
    if not phase1.empty and {"dataset", "seed", "method", "aug_f1"}.issubset(phase1.columns):
        frames.append(phase1[phase1["method"].isin(["no_aug", "random_cov_state", "pca_cov_state"])].copy())
    if not frames:
        return pd.DataFrame()
    perf = pd.concat(frames, ignore_index=True, sort=False)
    if "status" in perf.columns:
        perf = perf[perf["status"].fillna("success") == "success"].copy()
    return perf


def _alignment_vs_performance(per_seed: pd.DataFrame, selector_root: Path, phase1_root: Path) -> Tuple[pd.DataFrame, str]:
    if per_seed.empty:
        return pd.DataFrame(), "No local tangent per-seed summary available."
    perf = _load_performance(selector_root, phase1_root)
    if perf.empty:
        return pd.DataFrame(), "Performance files not found; skipped correlation."

    local = per_seed[per_seed["method"] == "csta_topk_uniform_top5"][
        ["dataset", "seed", "selected_alignment_mean", "selected_minus_random_alignment", "selected_minus_pca_alignment"]
    ].copy()
    pivot = perf.pivot_table(index=["dataset", "seed"], columns="method", values="aug_f1", aggfunc="mean").reset_index()
    merged = local.merge(pivot, on=["dataset", "seed"], how="left")
    if "csta_topk_uniform_top5" in merged and "no_aug" in merged:
        merged["csta_u5_gain_vs_no_aug"] = merged["csta_topk_uniform_top5"] - merged["no_aug"]
    if "csta_topk_uniform_top5" in merged and "random_cov_state" in merged:
        merged["csta_u5_delta_vs_random_cov_state"] = merged["csta_topk_uniform_top5"] - merged["random_cov_state"]
    if "csta_topk_uniform_top5" in merged and "pca_cov_state" in merged:
        merged["csta_u5_delta_vs_pca_cov_state"] = merged["csta_topk_uniform_top5"] - merged["pca_cov_state"]

    metric_cols = [
        "csta_u5_gain_vs_no_aug",
        "csta_u5_delta_vs_random_cov_state",
        "csta_u5_delta_vs_pca_cov_state",
    ]
    rows = []
    for align_col in ["selected_alignment_mean", "selected_minus_random_alignment", "selected_minus_pca_alignment"]:
        for metric in metric_cols:
            if align_col in merged and metric in merged:
                rows.append(
                    {
                        "alignment_metric": align_col,
                        "performance_metric": metric,
                        "n_pairs": int(merged[[align_col, metric]].dropna().shape[0]),
                        "pearson": _pearson(merged[align_col], merged[metric]),
                        "spearman": _spearman(merged[align_col], merged[metric]),
                    }
                )
    return pd.DataFrame(rows), ""


def _write_report(root: Path, per_seed: pd.DataFrame, dataset: pd.DataFrame, overall: pd.DataFrame, warning: str) -> None:
    lines = [
        "# Local Tangent Audit Report",
        "",
        "This is a post-hoc diagnostic for CSTA/PIA template directions. It does not change training or augmentation outputs.",
        "",
    ]
    if warning:
        lines.extend(["## Warnings", "", f"- {warning}", ""])
    lines.extend(
        [
            "## Coverage",
            "",
            f"- Per-seed rows: {len(per_seed)}",
            f"- Dataset-summary rows: {len(dataset)}",
            f"- Overall rows: {len(overall)}",
            "",
        ]
    )
    if not overall.empty:
        lines.extend(
            [
                "## Overall Alignment",
                "",
                "```csv",
                overall.to_csv(index=False).strip(),
                "```",
                "",
            ]
        )
    (root / "local_tangent_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local tangent audit summaries.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--selector-root", type=Path, default=DEFAULT_SELECTOR_ROOT)
    parser.add_argument("--phase1-root", type=Path, default=DEFAULT_PHASE1_ROOT)
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    per_seed = _collect_run_summaries(root)
    candidates = _collect_candidate_rows(root)
    dataset = _dataset_summary(per_seed)
    overall = _overall_summary(candidates, per_seed)
    corr, warning = _alignment_vs_performance(per_seed, Path(args.selector_root), Path(args.phase1_root))

    per_seed.to_csv(root / "local_tangent_per_seed_summary.csv", index=False)
    dataset.to_csv(root / "local_tangent_dataset_summary.csv", index=False)
    overall.to_csv(root / "local_tangent_overall_summary.csv", index=False)
    corr.to_csv(root / "local_tangent_alignment_vs_performance.csv", index=False)
    _write_report(root, per_seed, dataset, overall, warning)
    print(f"[local-tangent-summary] wrote {root}")


if __name__ == "__main__":
    main()
