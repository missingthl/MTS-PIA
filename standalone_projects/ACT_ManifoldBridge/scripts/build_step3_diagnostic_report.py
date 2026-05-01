#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


COMBO_RE = re.compile(r"^g(?P<gamma>[0-9.]+)_e(?P<eta>[0-9.]+)$")
DEFAULT_COMPARE_METHODS = [
    "no_aug",
    "csta_top1_current",
    "dba_sameclass",
    "wdba_sameclass",
]
SAFE_FIELDS = [
    "gamma_requested_mean",
    "gamma_used_mean",
    "safe_clip_rate",
    "safe_radius_ratio_mean",
    "gamma_zero_rate",
    "transport_error_logeuc_mean",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_csv(path)
    return pd.DataFrame()


def _combo_dirs(root: Path) -> List[Path]:
    out: List[Path] = []
    for path in sorted(root.iterdir() if root.is_dir() else []):
        if path.is_dir() and COMBO_RE.match(path.name):
            out.append(path)
    return out


def _parse_combo(name: str) -> Dict[str, float]:
    match = COMBO_RE.match(name)
    if not match:
        raise ValueError(f"Invalid Step3 combo directory name: {name}")
    return {
        "combo": name,
        "gamma": float(match.group("gamma")),
        "eta_safe": float(match.group("eta")),
    }


def _success_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    status = df["status"] if "status" in df.columns else "success"
    return df[status.fillna("success").astype(str) == "success"].copy()


def _load_step3(root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for combo_dir in _combo_dirs(root):
        per_seed = _read_csv(combo_dir / "per_seed_external.csv")
        if per_seed.empty:
            continue
        meta = _parse_combo(combo_dir.name)
        for key, value in meta.items():
            per_seed[key] = value
        frames.append(per_seed)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_refs(paths: List[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        if not path:
            continue
        csv_path = path / "per_seed_external.csv"
        df = _read_csv(csv_path)
        if not df.empty:
            df["ref_root"] = str(path)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _finite_has_issue(df: pd.DataFrame, cols: List[str]) -> Dict[str, bool]:
    out: Dict[str, bool] = {"has_nan": False, "has_inf": False}
    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        out["has_nan"] = bool(out["has_nan"] or vals.isna().any())
        out["has_inf"] = bool(out["has_inf"] or np.isinf(vals.to_numpy(dtype=float)).any())
    return out


def _build_completion(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for combo, sub in df.groupby("combo", dropna=False):
        meta = _parse_combo(str(combo))
        status = sub["status"].fillna("success").astype(str) if "status" in sub.columns else pd.Series(["success"] * len(sub))
        issue = _finite_has_issue(sub, ["aug_f1", *SAFE_FIELDS])
        rows.append(
            {
                **meta,
                "n_rows": int(len(sub)),
                "n_success": int((status == "success").sum()),
                "n_failed": int((status != "success").sum()),
                "n_datasets": int(sub["dataset"].nunique()) if "dataset" in sub.columns else 0,
                "n_seeds": int(sub["seed"].nunique()) if "seed" in sub.columns else 0,
                **issue,
            }
        )
    return pd.DataFrame(rows).sort_values(["gamma", "eta_safe"]) if rows else pd.DataFrame()


def _build_grid_summary(df: pd.DataFrame) -> pd.DataFrame:
    ok = _success_rows(df)
    if ok.empty:
        return pd.DataFrame()
    rows = []
    for combo, sub in ok.groupby("combo"):
        meta = _parse_combo(str(combo))
        dataset_means = sub.groupby("dataset")["aug_f1"].mean()
        rows.append(
            {
                **meta,
                "n_dataset_seed": int(len(sub)),
                "n_datasets": int(dataset_means.shape[0]),
                "mean_f1_over_datasets": float(dataset_means.mean()),
                "mean_f1_over_dataset_seed": float(pd.to_numeric(sub["aug_f1"], errors="coerce").mean()),
                "std_f1_over_datasets": float(dataset_means.std(ddof=1)) if len(dataset_means) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_f1_over_datasets", ascending=False)


def _build_safe_audit(df: pd.DataFrame) -> pd.DataFrame:
    ok = _success_rows(df)
    if ok.empty:
        return pd.DataFrame()
    rows = []
    for combo, sub in ok.groupby("combo"):
        meta = _parse_combo(str(combo))
        row = {**meta}
        for field in SAFE_FIELDS:
            vals = pd.to_numeric(sub[field], errors="coerce") if field in sub.columns else pd.Series(dtype=float)
            row[f"{field}_mean"] = float(vals.mean()) if not vals.empty else np.nan
            row[f"{field}_max"] = float(vals.max()) if not vals.empty else np.nan
        gamma_req = pd.to_numeric(sub.get("gamma_requested_mean", pd.Series(dtype=float)), errors="coerce")
        gamma_used = pd.to_numeric(sub.get("gamma_used_mean", pd.Series(dtype=float)), errors="coerce")
        if len(gamma_req) and len(gamma_used):
            row["gamma_used_gt_requested_count"] = int((gamma_used > gamma_req + 1e-9).sum())
        else:
            row["gamma_used_gt_requested_count"] = 0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["gamma", "eta_safe"]) if rows else pd.DataFrame()


def _external_method_means(refs: pd.DataFrame) -> pd.DataFrame:
    ok = _success_rows(refs)
    if ok.empty:
        return pd.DataFrame()
    methods = ok[~ok["method"].astype(str).str.startswith("csta_")].copy()
    methods = methods[methods["method"] != "no_aug"]
    if methods.empty:
        return pd.DataFrame()
    return (
        methods.groupby("method", as_index=False)
        .agg(mean_f1=("aug_f1", "mean"), n_dataset_seed=("aug_f1", "size"))
        .sort_values(["mean_f1", "method"], ascending=[False, True])
    )


def _best_combo(grid: pd.DataFrame) -> Optional[str]:
    if grid.empty:
        return None
    return str(grid.iloc[0]["combo"])


def _build_dataset_comparison(step3: pd.DataFrame, refs: pd.DataFrame, best_combo: str) -> pd.DataFrame:
    ok_refs = _success_rows(refs)
    best = _success_rows(step3)
    best = best[best["combo"] == best_combo].copy()
    if best.empty:
        return pd.DataFrame()
    best_dataset = best.groupby("dataset", as_index=False).agg(pia_final=("aug_f1", "mean"))
    ref_dataset = ok_refs.groupby(["dataset", "method"], as_index=False).agg(aug_f1=("aug_f1", "mean"))
    pivot = ref_dataset.pivot(index="dataset", columns="method", values="aug_f1").reset_index()
    out = best_dataset.merge(pivot, on="dataset", how="left")
    raw_cols = [c for c in out.columns if str(c).startswith("raw_aug_")]
    out["best_rawaug"] = out[raw_cols].max(axis=1) if raw_cols else np.nan
    external_cols = [
        c
        for c in out.columns
        if c not in {"dataset", "pia_final"}
        and not str(c).startswith("csta_")
        and c != "no_aug"
    ]
    out["dataset_best_external"] = out[external_cols].max(axis=1) if external_cols else np.nan
    out["dataset_best_external_method"] = (
        out[external_cols].idxmax(axis=1) if external_cols else ""
    )
    for ref in ["no_aug", "csta_top1_current", "best_rawaug", "dba_sameclass", "wdba_sameclass", "dataset_best_external"]:
        if ref in out.columns:
            out[f"pia_minus_{ref}"] = out["pia_final"] - out[ref]
    rank_cols = [c for c in external_cols if c in out.columns] + ["pia_final"]
    out["pia_rank_in_dataset"] = out[rank_cols].rank(axis=1, ascending=False, method="average")["pia_final"]
    return out.sort_values("dataset")


def _paired_delta(best: pd.DataFrame, refs: pd.DataFrame, ref_method: str) -> pd.Series:
    ref = refs[refs["method"] == ref_method][["dataset", "seed", "aug_f1"]].rename(columns={"aug_f1": "ref_f1"})
    cur = best[["dataset", "seed", "aug_f1"]].rename(columns={"aug_f1": "pia_f1"})
    merged = cur.merge(ref, on=["dataset", "seed"], how="inner")
    if merged.empty:
        return pd.Series(dtype=float)
    return pd.to_numeric(merged["pia_f1"], errors="coerce") - pd.to_numeric(merged["ref_f1"], errors="coerce")


def _wilcoxon_pvalue(delta: pd.Series) -> float:
    vals = delta.dropna().to_numpy(dtype=float)
    vals = vals[np.abs(vals) > 1e-12]
    if len(vals) == 0:
        return np.nan
    try:
        from scipy.stats import wilcoxon  # type: ignore

        return float(wilcoxon(vals).pvalue)
    except Exception:
        return np.nan


def _bootstrap_ci(delta: pd.Series, *, seed: int = 123, n_boot: int = 5000) -> Dict[str, float]:
    vals = delta.dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return {"ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    means = np.empty((int(n_boot),), dtype=np.float64)
    for i in range(int(n_boot)):
        sample = rng.choice(vals, size=len(vals), replace=True)
        means[i] = float(np.mean(sample))
    return {
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def _build_win_tie_loss(step3: pd.DataFrame, refs: pd.DataFrame, best_combo: str, compare_methods: List[str]) -> pd.DataFrame:
    best = _success_rows(step3)
    refs_ok = _success_rows(refs)
    best = best[best["combo"] == best_combo].copy()
    rows = []
    for method in compare_methods:
        delta = _paired_delta(best, refs_ok, method)
        vals = delta.dropna()
        ci = _bootstrap_ci(vals)
        rows.append(
            {
                "reference_method": method,
                "n_pairs": int(vals.shape[0]),
                "mean_delta": float(vals.mean()) if not vals.empty else np.nan,
                "median_delta": float(vals.median()) if not vals.empty else np.nan,
                "win_count": int((vals > 1e-12).sum()) if not vals.empty else 0,
                "tie_count": int((vals.abs() <= 1e-12).sum()) if not vals.empty else 0,
                "loss_count": int((vals < -1e-12).sum()) if not vals.empty else 0,
                "bootstrap_ci_low": ci["ci_low"],
                "bootstrap_ci_high": ci["ci_high"],
                "wilcoxon_p_value": _wilcoxon_pvalue(vals),
            }
        )
    return pd.DataFrame(rows)


def build_report(args: argparse.Namespace) -> None:
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)

    step3 = _load_step3(root)
    refs = _load_refs([Path(p) for p in args.ref_roots if p])
    if step3.empty:
        raise RuntimeError(f"No Step3 per_seed_external.csv files found under {root}")

    completion = _build_completion(step3)
    grid = _build_grid_summary(step3)
    safe = _build_safe_audit(step3)
    external = _external_method_means(refs)
    best_combo = _best_combo(grid)
    if best_combo is None:
        raise RuntimeError("No successful Step3 rows found.")
    dataset_comp = _build_dataset_comparison(step3, refs, best_combo)
    compare_methods = args.compare_methods.split(",") if args.compare_methods else DEFAULT_COMPARE_METHODS
    wtl = _build_win_tie_loss(step3, refs, best_combo, [m.strip() for m in compare_methods if m.strip()])

    completion.to_csv(out_dir / "step3_completion_audit.csv", index=False)
    grid.to_csv(out_dir / "step3_grid_summary.csv", index=False)
    safe.to_csv(out_dir / "step3_safe_audit.csv", index=False)
    dataset_comp.to_csv(out_dir / "step3_dataset_comparison.csv", index=False)
    external.to_csv(out_dir / "step3_external_reference_means.csv", index=False)
    wtl.to_csv(out_dir / "step3_win_tie_loss.csv", index=False)
    wtl.to_csv(out_dir / "step3_bootstrap_ci.csv", index=False)

    best_row = grid[grid["combo"] == best_combo].iloc[0].to_dict()
    leaderboard_rows = [
        {
            "method": f"PIA Uniform-Top5 ({best_combo})",
            "mean_f1": float(best_row["mean_f1_over_datasets"]),
            "source": "step3",
        }
    ]
    for _, row in external.iterrows():
        leaderboard_rows.append(
            {
                "method": str(row["method"]),
                "mean_f1": float(row["mean_f1"]),
                "source": "locked_external_refs",
            }
        )
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("mean_f1", ascending=False)
    leaderboard.to_csv(out_dir / "step3_external_leaderboard.csv", index=False)

    wdba = external[external["method"] == "wdba_sameclass"]
    wdba_text = "missing"
    if not wdba.empty:
        wdba_text = f"{float(wdba.iloc[0]['mean_f1']):.6f}"
    report = [
        "# CSTA Step3 Diagnostic Report",
        "",
        f"- result_status: `{args.result_status}`",
        f"- root: `{root}`",
        f"- best_combo: `{best_combo}`",
        f"- best_mean_f1_over_datasets: `{float(best_row['mean_f1_over_datasets']):.6f}`",
        f"- locked_wdba_mean_f1: `{wdba_text}`",
        "",
        "## Notes",
        "",
        "- If `result_status=stale`, these rows are retained for audit only and must not be used for eta-safe conclusions.",
        "- The final paper table should use a rerun generated after the eta-safe propagation fix.",
        "",
        "## Generated Files",
        "",
        "- `step3_completion_audit.csv`",
        "- `step3_grid_summary.csv`",
        "- `step3_safe_audit.csv`",
        "- `step3_dataset_comparison.csv`",
        "- `step3_external_leaderboard.csv`",
        "- `step3_win_tie_loss.csv`",
        "- `step3_bootstrap_ci.csv`",
    ]
    (out_dir / "step3_final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build formal CSTA Step3 diagnostic summary tables.")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument(
        "--ref-roots",
        type=str,
        nargs="*",
        default=[],
        help="Locked external roots containing per_seed_external.csv.",
    )
    parser.add_argument("--compare-methods", type=str, default=",".join(DEFAULT_COMPARE_METHODS))
    parser.add_argument("--result-status", type=str, default="candidate", choices=["candidate", "stale", "final"])
    args = parser.parse_args()
    build_report(args)


if __name__ == "__main__":
    main()
