import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np

def calculate_wtl(df, target_method, baseline_method):
    pivot = df.pivot_table(index=["dataset", "seed"], columns="method", values="aug_f1")
    if target_method not in pivot.columns or baseline_method not in pivot.columns:
        return 0, 0, 0
    
    diff = pivot[target_method] - pivot[baseline_method]
    w = (diff > 1e-4).sum()
    t = (abs(diff) <= 1e-4).sum()
    l = (diff < -1e-4).sum()
    return int(w), int(t), int(l)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--baseline", type=str, default="csta_topk_uniform_top5", help="Primary baseline for W/T/L")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    csv_path = results_dir / "per_seed_external.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Filter for methods of interest
    targets = ["spg_cfm_one_step", "spg_cfm_k3"]
    baselines = [
        "csta_topk_uniform_top5",
        "random_cov_state",
        "csta_template_random_within_bank",
        "spg_pia_zhead",
        "latent_residual_flow",
        "wdba_sameclass",
        "dba_sameclass"
    ]
    all_methods = targets + [b for b in baselines if b in df["method"].unique()]
    df_filtered = df[df["method"].isin(all_methods)].copy()

    # 1. Leaderboard (Mean F1)
    leaderboard = df_filtered.groupby("method")["aug_f1"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    
    # 2. W/T/L vs Primary Baseline
    wtl_rows = []
    for t in targets:
        if t in df["method"].unique():
            w, t_cnt, l = calculate_wtl(df, t, args.baseline)
            wtl_rows.append({"target": t, "baseline": args.baseline, "W/T/L": f"{w}/{t_cnt}/{l}"})
    
    # 2b. k3 vs one_step
    if "spg_cfm_k3" in df["method"].unique() and "spg_cfm_one_step" in df["method"].unique():
        w, t_cnt, l = calculate_wtl(df, "spg_cfm_k3", "spg_cfm_one_step")
        wtl_rows.append({"target": "spg_cfm_k3", "baseline": "spg_cfm_one_step", "W/T/L": f"{w}/{t_cnt}/{l}"})
    
    wtl_df = pd.DataFrame(wtl_rows)

    # 3. SPG-CFM Diagnostics
    diag_fields = [
        "spg_cfm_steps",
        "spg_cfm_train_mse_mean",
        "spg_cfm_train_cosine_mean",
        "spg_cfm_train_pred_target_cosine_mean",
        "spg_cfm_generation_pred_target_cosine_mean",
        "spg_cfm_generated_direction_pairwise_cosine_mean",
        "spg_cfm_effective_aug_multiplier",
        "spg_cfm_alignment_to_spg_mean",
        "spg_cfm_projection_energy_mean",
        "spg_cfm_condition_norm_mean",
        "spg_zhead_train_acc",
        "gamma_used_ratio_mean",
        "safe_clip_rate",
        "bridge_success_rate"
    ]
    available_diags = [f for f in diag_fields if f in df.columns]
    cfm_diags = df[df["method"].isin(targets)].groupby("method")[available_diags].mean(numeric_only=True).T

    # 4. Speed Audit
    time_fields = [
        "augmentation_build_time_sec",
        "spg_cfm_zhead_time_sec",
        "spg_cfm_condition_time_sec",
        "spg_cfm_train_time_sec",
        "spg_cfm_generation_time_sec",
        "generation_time_per_aug_sample_ms"
    ]
    available_times = [f for f in time_fields if f in df.columns]
    speed_summary = df_filtered.groupby("method")[available_times].mean(numeric_only=True)
    
    # Relative Speed
    u5_time = speed_summary.loc["csta_topk_uniform_top5", "augmentation_build_time_sec"] if "csta_topk_uniform_top5" in speed_summary.index else np.nan
    wdba_time = speed_summary.loc["wdba_sameclass", "augmentation_build_time_sec"] if "wdba_sameclass" in speed_summary.index else np.nan
    
    speed_summary["rel_vs_u5"] = speed_summary["augmentation_build_time_sec"] / u5_time if pd.notna(u5_time) else np.nan
    speed_summary["rel_vs_wdba"] = speed_summary["augmentation_build_time_sec"] / wdba_time if pd.notna(wdba_time) else np.nan

    # Save CSV Summary
    summary_csv = results_dir / "spg_cfm_summary.csv"
    leaderboard.to_csv(summary_csv)
    print(f"Saved summary to {summary_csv}")

    # Generate Markdown Report
    report_md = results_dir / "spg_cfm_report.md"
    with open(report_md, "w") as f:
        f.write(f"# SPG-CFM Performance Report: {results_dir.name}\n\n")
        
        f.write("## 1. Leaderboard (Macro F1)\n\n")
        f.write("```\n" + leaderboard.to_string() + "\n```\n\n")
        
        f.write(f"## 2. W/T/L Summary\n\n")
        f.write("```\n" + wtl_df.to_string(index=False) + "\n```\n\n")
        
        f.write("## 3. SPG-CFM Diagnostics (Mean across datasets)\n\n")
        f.write("```\n" + cfm_diags.to_string() + "\n```\n\n")
        
        f.write("## 4. Speed Audit\n\n")
        f.write("```\n" + speed_summary.to_string() + "\n```\n\n")
        
        f.write("## 5. Dataset Breakdown\n\n")
        pivot = df_filtered.pivot_table(index="dataset", columns="method", values="aug_f1", aggfunc="mean").round(4)
        f.write("```\n" + pivot.to_string() + "\n```\n\n")

    print(f"Generated report at {report_md}")

    # Final CLI Output for Quick View
    print("\n--- Leaderboard ---")
    print(leaderboard.to_string())
    print("\n--- W/T/L Summary ---")
    print(wtl_df.to_string(index=False))
    print("\n--- Speed Audit ---")
    print(speed_summary.to_string())

if __name__ == "__main__":
    main()
