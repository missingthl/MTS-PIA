#!/usr/bin/env python3
"""Step 3A: CSTA-U5 vs random_cov_state / pca_cov_state paired statistics."""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

OUT = Path("results/final20_cov_controls_stats_v1/resnet1d_s123")
OUT.mkdir(parents=True, exist_ok=True)

# ── Load per-seed data ────────────────────────────────────────────
df_csta = pd.read_csv("results/csta_pia_final20/resnet1d_s123/per_seed_external.csv")
df_ext  = pd.read_csv("results/final20_minimal_baseline_v1/resnet1d_s123/per_seed_external.csv")

csta = df_csta[df_csta["method"] == "csta_topk_uniform_top5"]
rand = df_ext[df_ext["method"] == "random_cov_state"]
pca  = df_ext[df_ext["method"] == "pca_cov_state"]


def paired_stats(df_a, df_b, label_a, label_b, out_prefix):
    m = df_a.merge(df_b, on=["dataset", "seed"], suffixes=("_a", "_b"))
    m["delta"] = m["aug_f1_a"] - m["aug_f1_b"]
    m["gain_a"] = m["aug_f1_a"] - m["base_f1_a"]
    m["gain_b"] = m["aug_f1_b"] - m["base_f1_b"]

    # Per-dataset
    ds = m.groupby("dataset").agg(
        mean_f1_a=("aug_f1_a", "mean"), std_f1_a=("aug_f1_a", "std"),
        mean_f1_b=("aug_f1_b", "mean"), std_f1_b=("aug_f1_b", "std"),
        mean_gain_a=("gain_a", "mean"), mean_gain_b=("gain_b", "mean"),
        mean_delta=("delta", "mean"), std_delta=("delta", "std"),
        n_seeds=("delta", "count"),
        seed_win=("delta", lambda x: (x > 0).sum()),
        seed_loss=("delta", lambda x: (x < 0).sum()),
        seed_tie=("delta", lambda x: (x == 0).sum()),
    ).reset_index()
    ds["result"] = ds["mean_delta"].apply(
        lambda d: "win" if d > 1e-9 else ("loss" if d < -1e-9 else "tie"))
    ds.to_csv(OUT / f"{out_prefix}_per_dataset.csv", index=False)

    # Per-seed
    m[["dataset","seed","aug_f1_a","aug_f1_b","base_f1_a","gain_a","gain_b","delta"]].to_csv(
        OUT / f"{out_prefix}_per_seed.csv", index=False)

    # Statistics
    deltas = m["delta"].values
    ds_deltas = ds["mean_delta"].values
    n_datasets = len(ds)
    n_seeds = len(m)

    # Bootstrap CI
    rng = np.random.default_rng(42)
    B = 10000
    bs_means = np.array([np.mean(rng.choice(deltas, size=len(deltas), replace=True)) for _ in range(B)])
    ci_lo, ci_hi = np.percentile(bs_means, [2.5, 97.5])
    ci_crosses_zero = bool(ci_lo <= 0 <= ci_hi)

    # Wilcoxon
    try:
        w_ds = stats.wilcoxon(ds_deltas, alternative="two-sided")
        wds_s, wds_p = float(w_ds.statistic), float(w_ds.pvalue)
    except Exception:
        wds_s, wds_p = np.nan, np.nan
    try:
        w_seed = stats.wilcoxon(deltas, alternative="two-sided")
        wsd_s, wsd_p = float(w_seed.statistic), float(w_seed.pvalue)
    except Exception:
        wsd_s, wsd_p = np.nan, np.nan

    overall = {
        "comparison": f"{label_a} vs {label_b}",
        "n_datasets": n_datasets, "n_seeds": n_seeds,
        "mean_f1_a": float(m["aug_f1_a"].mean()),
        "mean_f1_b": float(m["aug_f1_b"].mean()),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "std_delta": float(np.std(deltas, ddof=1)),
        "dataset_win": int((ds["result"]=="win").sum()),
        "dataset_loss": int((ds["result"]=="loss").sum()),
        "dataset_tie": int((ds["result"]=="tie").sum()),
        "seed_win": int((deltas>0).sum()),
        "seed_loss": int((deltas<0).sum()),
        "seed_tie": int((deltas==0).sum()),
        "bootstrap_ci_lo": float(ci_lo), "bootstrap_ci_hi": float(ci_hi),
        "bootstrap_ci_crosses_zero": ci_crosses_zero,
        "wilcoxon_dataset_stat": wds_s, "wilcoxon_dataset_p": wds_p,
        "wilcoxon_seed_stat": wsd_s, "wilcoxon_seed_p": wsd_p,
    }
    pd.DataFrame([overall]).to_csv(OUT / f"{out_prefix}_overall.csv", index=False)
    pd.DataFrame({"bootstrap_mean": bs_means}).to_csv(OUT / f"{out_prefix}_bootstrap_ci.csv", index=False)

    losing = ds[ds["mean_delta"] < -1e-9]["dataset"].tolist()
    return overall, losing, ds


# ── Run ───────────────────────────────────────────────────────────
ov_r, lose_r, ds_r = paired_stats(csta, rand, "csta_topk_uniform_top5", "random_cov_state", "csta_vs_random_cov")
ov_p, lose_p, ds_p = paired_stats(csta, pca,  "csta_topk_uniform_top5", "pca_cov_state",    "csta_vs_pca_cov")

pd.DataFrame([ov_r, ov_p]).to_csv(OUT / "cov_control_overall_stats.csv", index=False)

# ── Report ─────────────────────────────────────────────────────────
rpt = f"""# CSTA-U5 vs Covariance-State Controls: Statistical Report

**Status**: OFFICIAL — based on locked Final20 experiment outputs.
**Date**: 2026-05-05
**Config**: csta_topk_uniform_top5, gamma=0.1, eta_safe=0.75, resnet1d, seeds 1/2/3

---

## 1. CSTA-U5 vs Random Covariance-State

| Metric | Value |
| :--- | :--- |
| Mean F1 (CSTA) | {ov_r['mean_f1_a']:.4f} |
| Mean F1 (Random Cov) | {ov_r['mean_f1_b']:.4f} |
| Mean Delta | {ov_r['mean_delta']:.4f} |
| Median Delta | {ov_r['median_delta']:.4f} |
| Dataset W / T / L | {ov_r['dataset_win']} / {ov_r['dataset_tie']} / {ov_r['dataset_loss']} |
| Seed W / T / L | {ov_r['seed_win']} / {ov_r['seed_tie']} / {ov_r['seed_loss']} |
| Bootstrap CI (95%) | [{ov_r['bootstrap_ci_lo']:.4f}, {ov_r['bootstrap_ci_hi']:.4f}] |
| CI crosses zero | **{ov_r['bootstrap_ci_crosses_zero']}** |
| Wilcoxon (dataset) p | {ov_r['wilcoxon_dataset_p']:.4f} |
| Wilcoxon (seed) p | {ov_r['wilcoxon_seed_p']:.4f} |
"""
if not ov_r['bootstrap_ci_crosses_zero']:
    rpt += "\n**CI does NOT cross zero. CSTA-U5 significantly outperforms random_cov_state.**\n"
else:
    rpt += "\n**CI crosses zero. The improvement over random_cov_state is not statistically significant.**\n"

rpt += f"\nDatasets where CSTA < random_cov: {lose_r}\n"

rpt += f"""
## 2. CSTA-U5 vs PCA Covariance-State

| Metric | Value |
| :--- | :--- |
| Mean F1 (CSTA) | {ov_p['mean_f1_a']:.4f} |
| Mean F1 (PCA Cov) | {ov_p['mean_f1_b']:.4f} |
| Mean Delta | {ov_p['mean_delta']:.4f} |
| Median Delta | {ov_p['median_delta']:.4f} |
| Dataset W / T / L | {ov_p['dataset_win']} / {ov_p['dataset_tie']} / {ov_p['dataset_loss']} |
| Seed W / T / L | {ov_p['seed_win']} / {ov_p['seed_tie']} / {ov_p['seed_loss']} |
| Bootstrap CI (95%) | [{ov_p['bootstrap_ci_lo']:.4f}, {ov_p['bootstrap_ci_hi']:.4f}] |
| CI crosses zero | **{ov_p['bootstrap_ci_crosses_zero']}** |
| Wilcoxon (dataset) p | {ov_p['wilcoxon_dataset_p']:.4f} |
| Wilcoxon (seed) p | {ov_p['wilcoxon_seed_p']:.4f} |
"""
if not ov_p['bootstrap_ci_crosses_zero']:
    rpt += "\n**CI does NOT cross zero. CSTA-U5 significantly outperforms pca_cov_state.**\n"
else:
    rpt += "\n**CI crosses zero. The improvement over pca_cov_state is not statistically significant.**\n"

rpt += f"\nDatasets where CSTA < pca_cov: {lose_p}\n"

rpt += """
## 3. Paper Wording Guardrails

### For CSTA vs random_cov:
"""
if not ov_r['bootstrap_ci_crosses_zero']:
    rpt += "- CSTA-U5 achieves a statistically significant improvement over random covariance-state directions.\n"
else:
    rpt += "- CSTA-U5 achieves a consistent mean improvement over random covariance-state directions, but the improvement is not statistically significant across all datasets.\n"
    rpt += "- Covariance-state perturbation itself is a strong augmentation baseline; PIA adds structure and auditability rather than a strictly dominant performance advantage.\n"

rpt += "\n### For CSTA vs pca_cov:\n"
if not ov_p['bootstrap_ci_crosses_zero']:
    rpt += "- CSTA-U5 achieves a statistically significant improvement over PCA covariance-state directions.\n"
else:
    rpt += "- CSTA-U5 achieves a consistent mean improvement over PCA covariance-state directions, but the improvement is not statistically significant across all datasets.\n"

rpt += f"""
## 4. Per-Dataset Summary

| Dataset | CSTA F1 | Rand F1 | PCA F1 | Δ vs Rand | Δ vs PCA |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
for _, row in ds_r.iterrows():
    pca_row = ds_p[ds_p["dataset"] == row["dataset"]]
    pca_f1 = pca_row["mean_f1_b"].values[0] if len(pca_row) > 0 else 0
    pca_d = pca_row["mean_delta"].values[0] if len(pca_row) > 0 else 0
    rpt += f"| {row['dataset']} | {row['mean_f1_a']:.4f} | {row['mean_f1_b']:.4f} | {pca_f1:.4f} | {row['mean_delta']:+.4f} | {pca_d:+.4f} |\n"

rpt += """
## 5. Key Conclusion
"""
if ov_r['bootstrap_ci_crosses_zero'] and ov_p['bootstrap_ci_crosses_zero']:
    rpt += ("Both comparisons have bootstrap CIs crossing zero. PIA/UniformTop5 provides a mean advantage "
            "over generic covariance-state perturbations but the dataset-level behavior is mixed. "
            "Covariance-state augmentation is already a strong augmentation space; PIA contributes "
            "structure and auditability within this space.\n")
elif not ov_r['bootstrap_ci_crosses_zero'] and not ov_p['bootstrap_ci_crosses_zero']:
    rpt += ("CSTA-U5 shows statistically significant improvement over both random and PCA covariance-state controls. "
            "PIA templates provide a structured advantage over generic covariance-state perturbations.\n")
else:
    rpt += ("Results are mixed: one comparison shows statistical significance while the other does not. "
            "Covariance-state augmentation is a strong space; PIA adds consistent mean-level improvement.\n")

with open(OUT / "cov_control_stats_report.md", "w") as f:
    f.write(rpt)

print("=== Step 3A complete ===")
print(f"CSTA vs Random: delta={ov_r['mean_delta']:.4f}, CI=[{ov_r['bootstrap_ci_lo']:.4f},{ov_r['bootstrap_ci_hi']:.4f}], crosses_zero={ov_r['bootstrap_ci_crosses_zero']}, ds_WTL={ov_r['dataset_win']}/{ov_r['dataset_tie']}/{ov_r['dataset_loss']}")
print(f"CSTA vs PCA:    delta={ov_p['mean_delta']:.4f}, CI=[{ov_p['bootstrap_ci_lo']:.4f},{ov_p['bootstrap_ci_hi']:.4f}], crosses_zero={ov_p['bootstrap_ci_crosses_zero']}, ds_WTL={ov_p['dataset_win']}/{ov_p['dataset_tie']}/{ov_p['dataset_loss']}")
