
import os
import sys
import pandas as pd
import numpy as np
import json

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def analyze_seed(seed):
    print(f"\n=== Analyzing Seed {seed} ===")
    base_dir = f"promoted_results/phase13f/step1/seed1/seed{seed}"
    path_s = f"{base_dir}/dcnet/spatial_trial_pred.csv"
    path_m = f"{base_dir}/manifold/manifold_trial_pred.csv"
    path_f = f"{base_dir}/fusion/fusion_trial_pred.csv"
    
    if not (os.path.exists(path_s) and os.path.exists(path_m) and os.path.exists(path_f)):
        print(f"Skipping Seed {seed}: Missing files.")
        return None

    df_s = pd.read_csv(path_s)
    df_m = pd.read_csv(path_m)
    df_f = pd.read_csv(path_f)
    
    # Merge
    # Rename cols to avoid collision
    df_s = df_s[["trial_id", "true_label"]].copy()
    df_s["pred_s"] = pd.read_csv(path_s)[["l0", "l1", "l2"]].values.argmax(axis=1) # Or use existing if predicted label col exists
    # Wait, Step 1 script saved logits? Let's check CSV cols.
    # Step 1 `s_test.to_csv` saved whatever `predict_split` returned.
    # `predict_split` returns `trial_id`, `true_label`, `l0`, `l1`, `l2`.
    # So we compute pred from logits.
    
    # Re-read to be safe
    raw_s = pd.read_csv(path_s)
    raw_m = pd.read_csv(path_m)
    raw_f = pd.read_csv(path_f)
    
    # Compute Preds
    raw_s["pred"] = np.argmax(raw_s[["l0", "l1", "l2"]].values, axis=1)
    raw_m["pred"] = np.argmax(raw_m[["l0", "l1", "l2"]].values, axis=1)
    # Fusion output has `pred_label` col directly
    
    # Merge
    merged = pd.merge(raw_s[["trial_id", "true_label", "pred"]], raw_m[["trial_id", "pred"]], on="trial_id", suffixes=("_s", "_m"))
    merged = pd.merge(merged, raw_f[["trial_id", "pred_label"]], on="trial_id")
    merged.rename(columns={"pred_label": "pred_f"}, inplace=True)
    
    # Verify Alignment
    if len(merged) != 270: # Expected Test Size for Seed1 (Seed 0 is 270? Check Step 1 report)
        # Step 1 report says 270.
        print(f"Warning: Merged length {len(merged)} != 270")
        
    # Correctness
    merged["S_ok"] = (merged["pred_s"] == merged["true_label"])
    merged["M_ok"] = (merged["pred_m"] == merged["true_label"])
    merged["F_ok"] = (merged["pred_f"] == merged["true_label"])
    
    # Buckets
    merged["bucket"] = "Unknown"
    merged.loc[merged.S_ok & merged.M_ok, "bucket"] = "Both Correct"
    merged.loc[merged.S_ok & (~merged.M_ok), "bucket"] = "Spatial Only"
    merged.loc[(~merged.S_ok) & merged.M_ok, "bucket"] = "Manifold Only" # Complementary
    merged.loc[(~merged.S_ok) & (~merged.M_ok), "bucket"] = "Neither Correct"
    
    # Counts
    counts = merged["bucket"].value_counts().to_dict()
    total = len(merged)
    
    print("\n[Buckets]")
    for k in ["Both Correct", "Spatial Only", "Manifold Only", "Neither Correct"]:
        c = counts.get(k, 0)
        print(f"  {k}: {c} ({c/total*100:.2f}%)")
        
    # Fusion Attribution
    # Corrected by Fusion: F_ok=True AND S_ok=False
    rescued = merged[(merged.F_ok) & (~merged.S_ok)]
    # Hurt by Fusion: F_ok=False AND S_ok=True
    lost = merged[(~merged.F_ok) & (merged.S_ok)]
    
    n_rescued = len(rescued)
    n_lost = len(lost)
    net = n_rescued - n_lost
    
    print(f"\n[Attribution]")
    print(f"  Rescued (F_ok & !S_ok): {n_rescued}")
    print(f"  Lost (S_ok & !F_ok): {n_lost}")
    print(f"  Net Gain: {net} ({net/total*100:.2f}%)")
    
    # Should match Delta S in report
    acc_s = merged["S_ok"].mean()
    acc_f = merged["F_ok"].mean()
    print(f"  Spatial Acc: {acc_s:.4f}")
    print(f"  Fusion Acc: {acc_f:.4f}")
    print(f"  Delta: {acc_f - acc_s:.4f}")
    
    # Save Detail
    out_dir = f"promoted_results/phase13f/step2/seed1/seed{seed}"
    ensure_dir(out_dir)
    merged.to_csv(f"{out_dir}/gain_attribution_detail.csv", index=False)
    
    return {
        "Seed": seed,
        "N": total,
        "Acc_S": acc_s,
        "Acc_M": merged["M_ok"].mean(),
        "Acc_F": acc_f,
        "Both_Ok": counts.get("Both Correct", 0),
        "S_Only": counts.get("Spatial Only", 0),
        "M_Only": counts.get("Manifold Only", 0),
        "Neither": counts.get("Neither Correct", 0),
        "Rescued": n_rescued,
        "Lost": n_lost,
        "Net_Gain_Count": net
    }

def main():
    seeds = [0, 4]
    stats = []
    
    report_lines = []
    report_lines.append("# Phase 13F Step 2: Fusion Gain Attribution\n")
    
    for seed in seeds:
        res = analyze_seed(seed)
        if res:
            stats.append(res)
            
    # Aggregate Report
    report_lines.append("## Summary Table\n")
    report_lines.append(f"| Seed | Acc S | Acc M | Acc F | Rescued | Lost | Net Gain | Complementary (M_Only) |")
    report_lines.append(f"| --- | --- | --- | --- | --- | --- | --- | --- |")
    
    for s in stats:
        report_lines.append(f"| {s['Seed']} | {s['Acc_S']:.4f} | {s['Acc_M']:.4f} | {s['Acc_F']:.4f} | {s['Rescued']} | {s['Lost']} | {s['Net_Gain_Count']} | {s['M_Only']} |")
        
    # Detailed text
    report_lines.append("\n## Detailed Breakdown\n")
    for s in stats:
        report_lines.append(f"### Seed {s['Seed']}")
        report_lines.append(f"- **Correctness Buckets**:")
        report_lines.append(f"  - Both Correct: {s['Both_Ok']}")
        report_lines.append(f"  - Spatial Only: {s['S_Only']} (Dominant)")
        report_lines.append(f"  - Manifold Only: {s['M_Only']} (Complementary)")
        report_lines.append(f"  - Neither: {s['Neither']}")
        report_lines.append(f"- **Fusion Impact**:")
        report_lines.append(f"  - Rescued ({s['Rescued']}): Trials where S failed but Fusion succeeded (enabled by M).")
        report_lines.append(f"  - Lost ({s['Lost']}): Trials where S succeeded but Fusion failed (noise from M).")
        report_lines.append(f"  - Net: +{s['Net_Gain_Count']} trials.")
        report_lines.append("")

    out_path = "promoted_results/phase13f/step2/seed1/GAIN_ATTRIBUTION_REPORT.md"
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w") as f:
        f.write("\n".join(report_lines))
        
    print(f"\nReport saved to {out_path}")
    
    # Save CSV
    df = pd.DataFrame(stats)
    df.to_csv("promoted_results/phase13f/step2/seed1/summary_gain_attribution.csv", index=False)

if __name__ == "__main__":
    main()
