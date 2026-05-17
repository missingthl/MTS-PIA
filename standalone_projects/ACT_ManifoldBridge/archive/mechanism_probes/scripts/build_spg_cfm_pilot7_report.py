"""
build_spg_cfm_pilot7_report.py
Generate spg_cfm_pilot7_report.md and spg_cfm_pilot7_summary.csv.

Required: results_dir/per_seed_external.csv with >=168 rows.
"""
import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def _bootstrap_ci(arr, stat=np.mean, n_boot=2000, ci=0.95, rng_seed=42):
    """Return (point_est, lo, hi) via bootstrap."""
    rng = np.random.default_rng(rng_seed)
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = [stat(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo = float(np.quantile(boots, (1 - ci) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))
    return float(stat(arr)), lo, hi


def _wtl_pair(df, target, baseline, col="aug_f1", tol=1e-4):
    """Return (W, T, L, mean_delta) for matched (dataset, seed) rows."""
    pivot = df.pivot_table(index=["dataset", "seed"], columns="method", values=col)
    if target not in pivot.columns or baseline not in pivot.columns:
        return 0, 0, 0, float("nan")
    delta = pivot[target] - pivot[baseline]
    w = int((delta > tol).sum())
    t = int((delta.abs() <= tol).sum())
    l = int((delta < -tol).sum())
    mean_delta = float(delta.mean())
    return w, t, l, mean_delta


def _wtl_by_dataset(df, target, baseline, col="aug_f1", tol=1e-4):
    """Return per-dataset W/T/L dict."""
    results = {}
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        pivot = sub.pivot_table(index="seed", columns="method", values=col)
        if target not in pivot.columns or baseline not in pivot.columns:
            results[ds] = (0, 0, 0)
            continue
        delta = pivot[target] - pivot[baseline]
        results[ds] = (
            int((delta > tol).sum()),
            int((delta.abs() <= tol).sum()),
            int((delta < -tol).sum()),
        )
    return results


def _diag_row(df, method, fields):
    sub = df[df["method"] == method]
    out = {}
    for f in fields:
        if f in sub.columns:
            vals = sub[f].dropna()
            out[f] = float(vals.mean()) if len(vals) > 0 else float("nan")
        else:
            out[f] = float("nan")
    return out


# ── main ─────────────────────────────────────────────────────────────────────

TARGET = "spg_cfm_one_step"

BASELINES = [
    "csta_topk_uniform_top5",
    "random_cov_state",
    "csta_template_random_within_bank",
    "spg_pia_zhead",
    "latent_residual_flow",
    "wdba_sameclass",
    "dba_sameclass",
]

DIAG_FIELDS = [
    "spg_cfm_generated_direction_pairwise_cosine_mean",  # direction diversity
    "spg_cfm_alignment_to_spg_mean",                     # alignment
    "spg_cfm_projection_energy_mean",                    # projection energy
    "spg_cfm_projection_energy_std",
    "spg_cfm_effective_aug_multiplier",                  # effective multiplier
    "bridge_success_rate",
    "safe_clip_rate",
    "gamma_used_ratio_mean",
    "spg_zhead_train_acc",
    "transport_error_logeuc_mean",
]

SPEED_FIELDS = [
    "augmentation_build_time_sec",
    "spg_cfm_generation_time_sec",
    "generation_time_per_aug_sample_ms",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--baseline", default="csta_topk_uniform_top5")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    csv_path = results_dir / "per_seed_external.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    # Validate row count
    all_methods = [TARGET] + [b for b in BASELINES if b in df["method"].unique()]
    df_all = df[df["method"].isin(all_methods)].copy()
    n_rows = len(df_all[df_all["method"] == TARGET])
    print(f"[Pilot7] target rows for {TARGET}: {n_rows}  (expected ≥21 for 7ds×3seeds)")

    # ── 1. Leaderboard ───────────────────────────────────────────────────────
    lb = df_all.groupby("method")["aug_f1"].agg(["mean", "std", "count"])
    lb = lb.sort_values("mean", ascending=False)

    # ── 2. W/T/L (seed-level) ────────────────────────────────────────────────
    wtl_rows = []
    for bl in BASELINES:
        if bl not in df["method"].unique():
            continue
        w, t, l, md = _wtl_pair(df, TARGET, bl)
        _, lo, hi = _bootstrap_ci(
            (df.pivot_table(index=["dataset", "seed"], columns="method", values="aug_f1")
               .get(TARGET, pd.Series(dtype=float)) -
             df.pivot_table(index=["dataset", "seed"], columns="method", values="aug_f1")
               .get(bl, pd.Series(dtype=float))).dropna().values
        )
        wtl_rows.append({
            "baseline": bl,
            "W": w, "T": t, "L": l,
            "mean_delta": round(md, 5),
            "CI_lo": round(lo, 5),
            "CI_hi": round(hi, 5),
        })
    wtl_df = pd.DataFrame(wtl_rows)

    # ── 3. W/T/L (dataset-level) ─────────────────────────────────────────────
    ds_wtl_rows = []
    for bl in BASELINES:
        if bl not in df["method"].unique():
            continue
        ds_wtl = _wtl_by_dataset(df, TARGET, bl)
        for ds, (w, t, l) in ds_wtl.items():
            ds_wtl_rows.append({"dataset": ds, "baseline": bl,
                                 "W": w, "T": t, "L": l, "W/T/L": f"{w}/{t}/{l}"})
    ds_wtl_df = pd.DataFrame(ds_wtl_rows)

    # ── 4. Dataset Breakdown ──────────────────────────────────────────────────
    ds_pivot = df_all.pivot_table(
        index="dataset", columns="method", values="aug_f1", aggfunc="mean"
    ).round(4)

    # ── 5. Diagnostics ────────────────────────────────────────────────────────
    diag = _diag_row(df, TARGET, DIAG_FIELDS)
    diag_df = pd.DataFrame([diag], index=[TARGET]).T

    # ── 6. Speed Audit ────────────────────────────────────────────────────────
    available_speed = [f for f in SPEED_FIELDS if f in df.columns]
    speed_df = df_all.groupby("method")[available_speed].mean(numeric_only=True)

    target_build = speed_df.loc[TARGET, "augmentation_build_time_sec"] \
        if TARGET in speed_df.index and "augmentation_build_time_sec" in speed_df.columns else float("nan")

    wdba_build = speed_df.loc["wdba_sameclass", "augmentation_build_time_sec"] \
        if "wdba_sameclass" in speed_df.index and "augmentation_build_time_sec" in speed_df.columns else float("nan")

    u5_build = speed_df.loc["csta_topk_uniform_top5", "augmentation_build_time_sec"] \
        if "csta_topk_uniform_top5" in speed_df.index and "augmentation_build_time_sec" in speed_df.columns else float("nan")

    rel_vs_wdba = target_build / wdba_build if not math.isnan(wdba_build) and wdba_build > 0 else float("nan")
    rel_vs_u5   = target_build / u5_build   if not math.isnan(u5_build)   and u5_build   > 0 else float("nan")

    speed_extra = pd.Series({
        "relative_speed_vs_wdba": round(rel_vs_wdba, 3) if not math.isnan(rel_vs_wdba) else "N/A (wdba no time)",
        "relative_speed_vs_u5":   round(rel_vs_u5, 3)   if not math.isnan(rel_vs_u5)   else "N/A (u5 no time)",
    }, name=TARGET)

    # ── 7. Save artifacts ────────────────────────────────────────────────────
    lb.to_csv(results_dir / "spg_cfm_pilot7_summary.csv")

    report_path = results_dir / "spg_cfm_pilot7_report.md"
    with open(report_path, "w") as f:
        f.write(f"# SPG-CFM One-Step Pilot7 Report\n\n")
        f.write(f"**Target**: `{TARGET}` | **Datasets**: 7 | **Seeds**: 3 | **Total rows expected**: 168\n\n")
        f.write(f"**Actual target rows**: {n_rows}\n\n")

        f.write("---\n\n## 1. Leaderboard (Mean Macro-F1)\n\n")
        f.write("```\n" + lb.to_string() + "\n```\n\n")

        f.write("## 2. Seed-Level W/T/L vs Baselines (with 95% Bootstrap CI on Δ)\n\n")
        f.write("```\n" + wtl_df.to_string(index=False) + "\n```\n\n")

        f.write("## 3. Dataset-Level W/T/L\n\n")
        if not ds_wtl_df.empty:
            for bl in BASELINES:
                sub = ds_wtl_df[ds_wtl_df["baseline"] == bl]
                if sub.empty:
                    continue
                f.write(f"### vs `{bl}`\n\n")
                f.write("```\n" + sub[["dataset", "W/T/L"]].to_string(index=False) + "\n```\n\n")

        f.write("## 4. Dataset Breakdown (Mean F1)\n\n")
        f.write("```\n" + ds_pivot.to_string() + "\n```\n\n")

        f.write("## 5. SPG-CFM Diagnostics\n\n")
        f.write("```\n" + diag_df.to_string() + "\n```\n\n")

        f.write("## 6. Speed Audit\n\n")
        f.write("```\n" + speed_df.to_string() + "\n```\n\n")
        f.write(f"```\n{speed_extra.to_string()}\n```\n\n")

    print(f"Report  → {report_path}")
    print(f"Summary → {results_dir / 'spg_cfm_pilot7_summary.csv'}")

    # Quick CLI output
    print("\n--- Leaderboard ---")
    print(lb.to_string())
    print("\n--- W/T/L (seed-level) ---")
    print(wtl_df.to_string(index=False))
    print("\n--- Diagnostics ---")
    print(diag_df.to_string())
    print("\n--- Speed (rel) ---")
    print(speed_extra.to_string())


if __name__ == "__main__":
    main()
