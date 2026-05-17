from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = PROJECT_ROOT / "results" / "csta_direction_specificity_stress_v1" / "resnet1d_s123"
DEFAULT_PHASE1_ROOT = PROJECT_ROOT / "results" / "csta_external_baselines_phase1" / "resnet1d_s123"
DEFAULT_PHASE2_ROOT = PROJECT_ROOT / "results" / "csta_external_baselines_phase2" / "resnet1d_s123"

MAIN_METHODS = [
    "csta_topk_uniform_top5",
    "csta_top1_current",
    "csta_template_random_within_bank",
    "random_cov_state",
    "pca_cov_state",
]

REFERENCE_METHODS = [
    "dba_sameclass",
    "wdba_sameclass",
]

DIAGNOSTIC_FIELDS = [
    "template_response_top1_mean",
    "template_response_top5_mean",
    "template_response_gap_top1_top5_mean",
    "template_response_entropy_mean",
    "template_response_abs_mean",
    "selected_template_entropy",
    "template_usage_entropy",
    "top_template_concentration",
    "gamma_requested_mean",
    "gamma_used_mean",
    "gamma_used_ratio_mean",
    "pre_safe_displacement_norm_mean",
    "post_safe_displacement_norm_mean",
    "z_displacement_norm_mean",
    "safe_clip_rate",
    "safe_radius_ratio_mean",
    "transport_error_logeuc_mean",
    "candidate_audit_rows",
    "candidate_audit_available",
    "candidate_physics_ok",
    "aug_valid_rate",
]

DIAGNOSTIC_ALIASES = {
    "template_response_top1_mean": "template_response_top1_mean_audit",
    "template_response_top5_mean": "template_response_top5_mean_audit",
    "template_response_gap_top1_top5_mean": "template_response_gap_top1_top5_mean_audit",
    "template_response_entropy_mean": "template_response_entropy_mean_audit",
    "gamma_used_ratio_mean": "gamma_used_ratio_mean_audit",
    "pre_safe_displacement_norm_mean": "pre_safe_displacement_norm_mean_audit",
    "post_safe_displacement_norm_mean": "post_safe_displacement_norm_mean_audit",
}


def _read_per_seed(path: Path, *, phase: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df = df.copy()
    df["phase"] = phase
    return df


def _success(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    status = df["status"] if "status" in df.columns else pd.Series(["success"] * len(df), index=df.index)
    return df[status.fillna("success") == "success"].copy()


def _load_rows(out_root: Path, phase1_root: Path, phase2_root: Path, extra_roots: Sequence[str]) -> pd.DataFrame:
    frames = []
    frames.append(_read_per_seed(out_root / "per_seed_external.csv", phase="direction_specificity_main"))
    for item in extra_roots:
        if not str(item).strip():
            continue
        frames.append(_read_per_seed(Path(item) / "per_seed_external.csv", phase=f"extra:{Path(item).name}"))
    p1 = _read_per_seed(phase1_root / "per_seed_external.csv", phase="phase1_locked")
    if not p1.empty:
        frames.append(p1[p1["method"].isin([m for m in REFERENCE_METHODS if m != "wdba_sameclass"])].copy())
    p2 = _read_per_seed(phase2_root / "per_seed_external.csv", phase="phase2_locked")
    if not p2.empty:
        frames.append(p2[p2["method"].isin(["wdba_sameclass"])].copy())
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    ok = _success(df)
    if ok.empty:
        return pd.DataFrame()
    agg: Dict[str, tuple[str, str]] = {
        "macro_f1_mean": ("aug_f1", "mean"),
        "macro_f1_std": ("aug_f1", "std"),
        "gain_mean": ("gain", "mean"),
        "n_seeds": ("seed", "nunique"),
    }
    for field in DIAGNOSTIC_FIELDS:
        if field in ok.columns:
            agg[f"{field}_mean"] = (field, "mean")
    return ok.groupby(["dataset", "method"], as_index=False).agg(**agg)


def _overall_summary(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    if dataset_summary.empty:
        return pd.DataFrame()
    pivot = dataset_summary.pivot(index="dataset", columns="method", values="macro_f1_mean")
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    refs = {
        "uniform_top5": "csta_topk_uniform_top5",
        "bank_random": "csta_template_random_within_bank",
        "full_random": "random_cov_state",
        "top1": "csta_top1_current",
        "wdba": "wdba_sameclass",
        "dba": "dba_sameclass",
    }
    rows = []
    for method in sorted(pivot.columns):
        vals = pivot[method].dropna()
        rank_vals = ranks[method].dropna() if method in ranks else pd.Series(dtype=float)
        row = {
            "method": method,
            "n_datasets": int(vals.shape[0]),
            "mean_f1": float(vals.mean()) if not vals.empty else np.nan,
            "mean_rank": float(rank_vals.mean()) if not rank_vals.empty else np.nan,
            "win_count": int((rank_vals == 1.0).sum()) if not rank_vals.empty else 0,
        }
        for label, ref_method in refs.items():
            if ref_method not in pivot.columns or method == ref_method:
                row[f"mean_delta_vs_{label}"] = np.nan
                row[f"win_count_vs_{label}"] = np.nan
                continue
            common = pd.concat([pivot[method], pivot[ref_method]], axis=1, keys=["method", "ref"]).dropna()
            row[f"mean_delta_vs_{label}"] = float((common["method"] - common["ref"]).mean()) if not common.empty else np.nan
            row[f"win_count_vs_{label}"] = int((common["method"] > common["ref"]).sum()) if not common.empty else 0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mean_f1", "mean_rank"], ascending=[False, True])


def _diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    ok = _success(df)
    if ok.empty:
        return pd.DataFrame()
    group_cols = ["method"]
    if "eta_safe" in ok.columns:
        group_cols.append("eta_safe")
    rows = []
    for keys, sub in ok.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_rows"] = int(len(sub))
        row["n_datasets"] = int(sub["dataset"].nunique()) if "dataset" in sub else 0
        row["n_seeds"] = int(sub["seed"].nunique()) if "seed" in sub else 0
        for field in DIAGNOSTIC_FIELDS:
            vals = None
            if field in sub.columns:
                vals = pd.to_numeric(sub[field], errors="coerce")
            alias = DIAGNOSTIC_ALIASES.get(field)
            if vals is None or not vals.notna().any():
                if alias and alias in sub.columns:
                    vals = pd.to_numeric(sub[alias], errors="coerce")
            if vals is not None:
                row[field] = float(vals.mean()) if vals.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _completion(df: pd.DataFrame, methods: Sequence[str], datasets: Sequence[str], seeds: Sequence[int]) -> pd.DataFrame:
    rows = []
    for method in methods:
        sub = df[df["method"] == method] if not df.empty and "method" in df else pd.DataFrame()
        ok = _success(sub)
        rows.append(
            {
                "method": method,
                "expected_rows": int(len(datasets) * len(seeds)),
                "n_rows": int(len(sub)),
                "n_success": int(len(ok)),
                "n_failed": int((sub["status"] == "failed").sum()) if "status" in sub else 0,
                "n_datasets": int(sub["dataset"].nunique()) if "dataset" in sub else 0,
                "n_seeds": int(sub["seed"].nunique()) if "seed" in sub else 0,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_ci(delta: Sequence[float], *, seed: int = 123, n_boot: int = 5000) -> Dict[str, float]:
    arr = np.asarray([x for x in delta if np.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return {"mean_delta": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    boot = np.asarray([rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(int(n_boot))])
    return {
        "mean_delta": float(arr.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
    }


def _win_tie_loss(df: pd.DataFrame, methods: Iterable[str], *, tie_eps: float = 1e-9) -> pd.DataFrame:
    ok = _success(df)
    if ok.empty:
        return pd.DataFrame()
    pivot = ok.pivot_table(index=["dataset", "seed"], columns="method", values="aug_f1", aggfunc="mean")
    refs = ["csta_topk_uniform_top5", "csta_template_random_within_bank", "random_cov_state", "wdba_sameclass", "dba_sameclass"]
    rows = []
    for method in methods:
        if method not in pivot:
            continue
        for ref in refs:
            if ref not in pivot or method == ref:
                continue
            common = pd.concat([pivot[method], pivot[ref]], axis=1, keys=["method", "ref"]).dropna()
            delta = common["method"] - common["ref"]
            ci = _bootstrap_ci(delta.to_numpy())
            rows.append(
                {
                    "method": method,
                    "reference": ref,
                    "n_pairs": int(delta.shape[0]),
                    "mean_delta": ci["mean_delta"],
                    "median_delta": float(delta.median()) if not delta.empty else np.nan,
                    "win": int((delta > tie_eps).sum()),
                    "tie": int((delta.abs() <= tie_eps).sum()),
                    "loss": int((delta < -tie_eps).sum()),
                    "bootstrap_ci_low": ci["ci_low"],
                    "bootstrap_ci_high": ci["ci_high"],
                }
            )
    return pd.DataFrame(rows)


def _question_answer(overall: pd.DataFrame, diagnostics: pd.DataFrame) -> list[str]:
    def mean_f1(method: str) -> float:
        if overall.empty or method not in set(overall.get("method", [])):
            return np.nan
        return float(overall.loc[overall["method"] == method, "mean_f1"].iloc[0])

    def diag(method: str, field: str) -> float:
        if diagnostics.empty or method not in set(diagnostics.get("method", [])) or field not in diagnostics:
            return np.nan
        vals = diagnostics.loc[diagnostics["method"] == method, field]
        return float(vals.mean()) if vals.notna().any() else np.nan

    u5 = mean_f1("csta_topk_uniform_top5")
    bank = mean_f1("csta_template_random_within_bank")
    full = mean_f1("random_cov_state")
    top1 = mean_f1("csta_top1_current")
    response_gap = diag("csta_topk_uniform_top5", "template_response_gap_top1_top5_mean")
    u5_norm = diag("csta_topk_uniform_top5", "post_safe_displacement_norm_mean")
    bank_norm = diag("csta_template_random_within_bank", "post_safe_displacement_norm_mean")
    full_norm = diag("random_cov_state", "post_safe_displacement_norm_mean")
    return [
        "## Direction Specificity Questions",
        "",
        f"1. Bank-random close to U5? U5={u5:.6g}, bank-random={bank:.6g}, delta={bank - u5:.6g}.",
        f"2. Bank-random better than full random covariance? bank-random={bank:.6g}, full-random={full:.6g}, delta={bank - full:.6g}.",
        f"3. Top1 underperforms U5? top1={top1:.6g}, U5={u5:.6g}, delta_top1_minus_u5={top1 - u5:.6g}.",
        f"4. Response gaps flat? U5 mean top1-top5 response gap={response_gap:.6g}; inspect `direction_specificity_diagnostics.csv` before making geometric claims.",
        f"5. Safe-step shrinkage comparable? post-safe norms: U5={u5_norm:.6g}, bank-random={bank_norm:.6g}, full-random={full_norm:.6g}.",
        "6. Relaxed eta exposes direction specificity? Compare eta-tagged rows in `direction_specificity_diagnostics.csv` if stress roots were supplied.",
        "7. Claim boundary: this report should not claim manifold flatness or direction superiority unless performance, response-gap, and post-safe diagnostics jointly support it.",
    ]


def build(args) -> None:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    extra_roots = [x.strip() for x in str(args.extra_roots).split(",") if x.strip()]

    all_rows = _load_rows(out_root, Path(args.phase1_root), Path(args.phase2_root), extra_roots)
    main_rows = _read_per_seed(out_root / "per_seed_external.csv", phase="direction_specificity_main")
    if datasets:
        if not all_rows.empty and "dataset" in all_rows.columns:
            all_rows = all_rows[all_rows["dataset"].isin(datasets)].copy()
        if not main_rows.empty and "dataset" in main_rows.columns:
            main_rows = main_rows[main_rows["dataset"].isin(datasets)].copy()

    dataset_summary = _dataset_summary(all_rows)
    main_dataset_summary = _dataset_summary(main_rows)
    overall = _overall_summary(dataset_summary)
    diagnostics = _diagnostics(all_rows[all_rows["method"].isin(MAIN_METHODS)] if not all_rows.empty else all_rows)
    completion = _completion(main_rows, MAIN_METHODS, datasets, seeds)
    wtl = _win_tie_loss(all_rows, MAIN_METHODS)

    all_rows.to_csv(out_root / "per_seed_direction_specificity_with_refs.csv", index=False)
    main_dataset_summary.to_csv(out_root / "dataset_summary_external.csv", index=False)
    overall.to_csv(out_root / "overall_summary_external.csv", index=False)
    diagnostics.to_csv(out_root / "direction_specificity_diagnostics.csv", index=False)
    completion.to_csv(out_root / "direction_specificity_completion.csv", index=False)
    wtl.to_csv(out_root / "direction_specificity_win_tie_loss.csv", index=False)

    report = [
        "# Direction Specificity Stress Report",
        "",
        f"- out_root: `{out_root}`",
        f"- phase1_root: `{args.phase1_root}`",
        f"- phase2_root: `{args.phase2_root}`",
        f"- extra_roots: `{', '.join(extra_roots) if extra_roots else '(none)'}`",
        f"- rows_with_refs: `{len(all_rows)}`",
        "",
        "Generated files:",
        "- `per_seed_direction_specificity_with_refs.csv`",
        "- `dataset_summary_external.csv`",
        "- `overall_summary_external.csv`",
        "- `direction_specificity_diagnostics.csv`",
        "- `direction_specificity_completion.csv`",
        "- `direction_specificity_win_tie_loss.csv`",
        "",
        "Interpretation guardrail: distinguish full random covariance (`random_cov_state`) from random template sampling inside the PIA dictionary (`csta_template_random_within_bank`).",
        "",
    ]
    report.extend(_question_answer(overall, diagnostics))
    (out_root / "direction_specificity_stress_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Direction Specificity Stress Test summaries.")
    parser.add_argument("--out-root", type=str, default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--phase1-root", type=str, default=str(DEFAULT_PHASE1_ROOT))
    parser.add_argument("--phase2-root", type=str, default=str(DEFAULT_PHASE2_ROOT))
    parser.add_argument("--extra-roots", type=str, default="", help="Comma-separated additional result roots, e.g. eta stress roots.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="atrialfibrillation,ering,handmovementdirection,handwriting,japanesevowels,natops,racketsports",
    )
    parser.add_argument("--seeds", type=str, default="1,2,3")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
