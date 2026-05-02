from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BYARM_ROOT = PROJECT_ROOT / "results" / "csta_selector_ablation_v1" / "resnet1d_s123_byarm"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "results" / "csta_selector_ablation_v1" / "resnet1d_s123"
DEFAULT_PHASE1_ROOT = PROJECT_ROOT / "results" / "csta_external_baselines_phase1" / "resnet1d_s123"
DEFAULT_PHASE2_ROOT = PROJECT_ROOT / "results" / "csta_external_baselines_phase2" / "resnet1d_s123"

SELECTOR_METHODS = [
    "csta_topk_uniform_top5",
    "csta_fv_filter_top5",
    "csta_fv_score_top5",
    "csta_random_feasible_selector",
]

REFERENCE_METHODS = [
    "csta_top1_current",
    "random_cov_state",
    "pca_cov_state",
    "dba_sameclass",
    "wdba_sameclass",
]

DIAGNOSTIC_FIELDS = [
    "feasible_rate",
    "selector_accept_rate",
    "fv_score_mean",
    "fidelity_score_mean",
    "variety_score_mean",
    "relevance_score_mean",
    "safe_balance_score_mean",
    "template_usage_entropy",
    "top_template_concentration",
    "safe_radius_ratio_mean",
    "gamma_used_mean",
    "z_displacement_norm_mean",
    "transport_error_logeuc_mean",
    "post_bridge_reject_count",
    "aug_valid_rate",
    "pre_filter_reject_count",
    "reject_reason_zero_gamma",
    "reject_reason_safe_radius",
    "reject_reason_bridge_fail",
    "reject_reason_transport_error",
]


def _read_per_seed(path: Path, *, phase: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df = df.copy()
    df["phase"] = phase
    return df


def _load_selector_rows(byarm_root: Path) -> pd.DataFrame:
    frames = []
    for method in SELECTOR_METHODS:
        frames.append(_read_per_seed(byarm_root / method / "per_seed_external.csv", phase="selector_ablation"))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _load_reference_rows(phase1_root: Path, phase2_root: Path) -> pd.DataFrame:
    frames = []
    p1 = _read_per_seed(phase1_root / "per_seed_external.csv", phase="phase1_locked")
    if not p1.empty:
        frames.append(p1[p1["method"].isin([m for m in REFERENCE_METHODS if m != "wdba_sameclass"])].copy())
    p2 = _read_per_seed(phase2_root / "per_seed_external.csv", phase="phase2_locked")
    if not p2.empty:
        frames.append(p2[p2["method"].isin(["wdba_sameclass"])].copy())
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _success(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df.get("status", "success").fillna("success") == "success"].copy()


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
    rows = []
    uniform = pivot.get("csta_topk_uniform_top5")
    wdba = pivot.get("wdba_sameclass")
    for method in sorted(pivot.columns):
        vals = pivot[method].dropna()
        rank_vals = ranks[method].dropna() if method in ranks else pd.Series(dtype=float)
        out = {
            "method": method,
            "n_datasets": int(vals.shape[0]),
            "mean_f1": float(vals.mean()) if not vals.empty else np.nan,
            "mean_rank": float(rank_vals.mean()) if not rank_vals.empty else np.nan,
            "win_count": int((rank_vals == 1.0).sum()) if not rank_vals.empty else 0,
        }
        if uniform is not None and method != "csta_topk_uniform_top5":
            common = pd.concat([pivot[method], uniform], axis=1, keys=["method", "uniform"]).dropna()
            out["mean_delta_vs_uniform_top5"] = float((common["method"] - common["uniform"]).mean()) if not common.empty else np.nan
            out["win_count_vs_uniform_top5"] = int((common["method"] > common["uniform"]).sum()) if not common.empty else 0
        else:
            out["mean_delta_vs_uniform_top5"] = np.nan
            out["win_count_vs_uniform_top5"] = np.nan
        if wdba is not None:
            common = pd.concat([pivot[method], wdba], axis=1, keys=["method", "wdba"]).dropna()
            out["mean_delta_vs_wdba"] = float((common["method"] - common["wdba"]).mean()) if not common.empty else np.nan
            out["win_count_vs_wdba"] = int((common["method"] > common["wdba"]).sum()) if not common.empty else 0
        else:
            out["mean_delta_vs_wdba"] = np.nan
            out["win_count_vs_wdba"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["mean_f1", "mean_rank"], ascending=[False, True])


def _bootstrap_ci(vals: Sequence[float], *, seed: int = 123, n_boot: int = 5000) -> Dict[str, float]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean_delta": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    boot = np.asarray([rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(int(n_boot))])
    return {
        "mean_delta": float(arr.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
    }


def _win_tie_loss(per_seed: pd.DataFrame, compare_methods: Iterable[str], *, tie_eps: float = 1e-9) -> pd.DataFrame:
    ok = _success(per_seed)
    if ok.empty:
        return pd.DataFrame()
    pivot = ok.pivot_table(index=["dataset", "seed"], columns="method", values="aug_f1", aggfunc="mean")
    rows = []
    for method in compare_methods:
        if method not in pivot:
            continue
        for ref in ["csta_topk_uniform_top5", "wdba_sameclass", "dba_sameclass"]:
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


def _diagnostics(selector_rows: pd.DataFrame) -> pd.DataFrame:
    ok = _success(selector_rows)
    if ok.empty:
        return pd.DataFrame()
    rows = []
    for method, sub in ok.groupby("method"):
        out = {"method": method, "n_rows": int(len(sub))}
        for field in DIAGNOSTIC_FIELDS:
            if field in sub.columns:
                vals = pd.to_numeric(sub[field], errors="coerce")
                out[field] = float(vals.mean()) if vals.notna().any() else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def _dataset_comparison(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    if dataset_summary.empty:
        return pd.DataFrame()
    pivot = dataset_summary.pivot(index="dataset", columns="method", values="macro_f1_mean").reset_index()
    out = pivot.copy()
    for method in ["csta_fv_filter_top5", "csta_fv_score_top5", "csta_random_feasible_selector"]:
        if method in out and "csta_topk_uniform_top5" in out:
            out[f"{method}_minus_uniform_top5"] = out[method] - out["csta_topk_uniform_top5"]
        if method in out and "wdba_sameclass" in out:
            out[f"{method}_minus_wdba"] = out[method] - out["wdba_sameclass"]
    return out


def _completion(selector_rows: pd.DataFrame, datasets: Sequence[str], seeds: Sequence[int]) -> pd.DataFrame:
    rows = []
    for method in SELECTOR_METHODS:
        sub = selector_rows[selector_rows["method"] == method] if not selector_rows.empty else pd.DataFrame()
        rows.append(
            {
                "method": method,
                "expected_rows": int(len(datasets) * len(seeds)),
                "n_rows": int(len(sub)),
                "n_success": int((_success(sub)).shape[0]) if not sub.empty else 0,
                "n_failed": int((sub.get("status", pd.Series(dtype=str)) == "failed").sum()) if not sub.empty else 0,
                "n_datasets": int(sub["dataset"].nunique()) if "dataset" in sub else 0,
                "n_seeds": int(sub["seed"].nunique()) if "seed" in sub else 0,
            }
        )
    return pd.DataFrame(rows)


def build(args) -> None:
    byarm_root = Path(args.byarm_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    selector_rows = _load_selector_rows(byarm_root)
    refs = _load_reference_rows(Path(args.phase1_root), Path(args.phase2_root))
    all_rows = pd.concat([refs, selector_rows], ignore_index=True, sort=False) if not refs.empty else selector_rows.copy()

    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    completion = _completion(selector_rows, datasets, seeds)
    dataset_summary = _dataset_summary(all_rows)
    selector_dataset_summary = _dataset_summary(selector_rows)
    overall = _overall_summary(dataset_summary)
    diag = _diagnostics(selector_rows)
    compare = _dataset_comparison(dataset_summary)
    wtl = _win_tie_loss(all_rows, SELECTOR_METHODS)

    selector_rows.to_csv(out_root / "per_seed_selector_ablation.csv", index=False)
    all_rows.to_csv(out_root / "per_seed_selector_with_refs.csv", index=False)
    completion.to_csv(out_root / "selector_ablation_completion.csv", index=False)
    selector_dataset_summary.to_csv(out_root / "selector_ablation_dataset_summary.csv", index=False)
    overall.to_csv(out_root / "selector_ablation_summary.csv", index=False)
    diag.to_csv(out_root / "fv_selector_diagnostics.csv", index=False)
    compare.to_csv(out_root / "csta_vs_external_after_selector.csv", index=False)
    wtl.to_csv(out_root / "selector_win_tie_loss.csv", index=False)
    wtl.to_csv(out_root / "selector_bootstrap_ci.csv", index=False)

    report = [
        "# CSTA Selector Ablation Summary",
        "",
        f"- byarm_root: `{byarm_root}`",
        f"- phase1_root: `{args.phase1_root}`",
        f"- phase2_root: `{args.phase2_root}`",
        f"- selector_rows: `{len(selector_rows)}`",
        f"- total_rows_with_refs: `{len(all_rows)}`",
        "",
        "Generated files:",
        "- `selector_ablation_completion.csv`",
        "- `selector_ablation_summary.csv`",
        "- `selector_ablation_dataset_summary.csv`",
        "- `fv_selector_diagnostics.csv`",
        "- `csta_vs_external_after_selector.csv`",
        "- `selector_win_tie_loss.csv`",
        "- `selector_bootstrap_ci.csv`",
    ]
    (out_root / "selector_ablation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CSTA pre-bridge FV selector ablation summaries.")
    parser.add_argument("--byarm-root", type=str, default=str(DEFAULT_BYARM_ROOT))
    parser.add_argument("--out-root", type=str, default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--phase1-root", type=str, default=str(DEFAULT_PHASE1_ROOT))
    parser.add_argument("--phase2-root", type=str, default=str(DEFAULT_PHASE2_ROOT))
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
