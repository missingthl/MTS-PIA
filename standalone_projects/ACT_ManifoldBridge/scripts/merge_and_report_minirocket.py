import pandas as pd
import glob
import os
import numpy as np

def merge_and_report(root_dir, report_name):
    print(f"Merging results from {root_dir}...")
    # EXCLUDE already merged files
    csv_files = [f for f in glob.glob(os.path.join(root_dir, "**/per_seed_external.csv"), recursive=True) if "merged" not in f]
    
    if not csv_files:
        print("No CSV files found.")
        return
    
    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    
    # Save merged
    df.to_csv(os.path.join(root_dir, "per_seed_external_merged.csv"), index=False)
    
    pivot = df.pivot_table(index=['dataset', 'seed'], columns='method', values='aug_f1').reset_index()
    available_arms = [a for a in ['no_aug', 'csta_topk_uniform_top5'] if a in pivot.columns]
    if available_arms:
        pivot = pivot.dropna(subset=available_arms)
    
    # Audit
    with open(os.path.join(root_dir, report_name), 'w') as f:
        f.write(f"# MiniRocket {report_name.replace('.md', '').replace('_', ' ').title()}\n\n")
        f.write(f"- **Datasets (Total)**: {df['dataset'].nunique()}\n")
        f.write(f"- **Total Rows Found**: {len(df)}\n")
        f.write(f"- **Aligned Pairs (Complete)**: {len(pivot)}\n")
        f.write(f"- **Expected Pairs (20x3)**: 60\n\n")
        
        if 'no_aug' in pivot.columns and 'csta_topk_uniform_top5' in pivot.columns:
            pivot['delta'] = pivot['csta_topk_uniform_top5'] - pivot['no_aug']
            f.write("## Performance (CSTA-U5 vs No-Aug)\n")
            f.write(f"- **Mean Delta F1**: {pivot['delta'].mean():+.4f}\n")
            
            win = len(pivot[pivot['delta'] > 0.001])
            tie = len(pivot[np.abs(pivot['delta']) <= 0.001])
            loss = len(pivot[pivot['delta'] < -0.001])
            f.write(f"- **W/T/L (Seed-level)**: {win} Win / {tie} Tie / {loss} Loss\n\n")
            
            f.write("### Dataset-level Breakdown\n")
            ds_breakdown = pivot.groupby('dataset')['delta'].mean().sort_values(ascending=False)
            f.write("| Dataset | Mean Delta F1 |\n")
            f.write("| :--- | :--- |\n")
            for ds, val in ds_breakdown.items():
                f.write(f"| {ds} | {val:+.4f} |\n")

    print(f"Report generated: {report_name}")

def final_mechanism_audit(core_dir, batchb_dir, output_path):
    print("Performing Final Mechanism Audit...")
    core_csv = os.path.join(core_dir, "per_seed_external_merged.csv")
    batchb_csv = os.path.join(batchb_dir, "per_seed_external_merged.csv")
    
    df_core = pd.read_csv(core_csv)
    df_batchb = pd.read_csv(batchb_csv)
    
    df_core = df_core[df_core['method'].isin(['no_aug', 'csta_topk_uniform_top5'])]
    df_batchb = df_batchb[df_batchb['method'].isin(['random_cov_state', 'pca_cov_state'])]
    
    combined = pd.concat([df_core, df_batchb], ignore_index=True)
    pivot = combined.pivot_table(index=['dataset', 'seed'], columns='method', values='aug_f1').reset_index()
    pivot = pivot.dropna(subset=['no_aug', 'csta_topk_uniform_top5', 'random_cov_state', 'pca_cov_state'])
    
    with open(output_path, 'w') as f:
        f.write("# MiniRocket Final Mechanism Audit (Fidelity vs Random)\n\n")
        f.write(f"- **Aligned Pairs (Full Coverage)**: {len(pivot)}\n\n")
        
        pivot['gain_csta'] = pivot['csta_topk_uniform_top5'] - pivot['no_aug']
        pivot['gain_random'] = pivot['random_cov_state'] - pivot['no_aug']
        pivot['gain_pca'] = pivot['pca_cov_state'] - pivot['no_aug']
        
        f.write("## 1. Overall Comparison\n")
        f.write(f"| Metric | CSTA-U5 | Random Cov | PCA Cov |\n")
        f.write(f"| :--- | :--- | :--- | :--- |\n")
        f.write(f"| Mean Gain (F1) | {pivot['gain_csta'].mean():+.4f} | {pivot['gain_random'].mean():+.4f} | {pivot['gain_pca'].mean():+.4f} |\n")
        
        f.write("\n## 2. W/T/L vs CSTA\n")
        for ref in ['random_cov_state', 'pca_cov_state']:
            win = len(pivot[pivot['csta_topk_uniform_top5'] > pivot[ref] + 0.001])
            loss = len(pivot[pivot['csta_topk_uniform_top5'] < pivot[ref] - 0.001])
            tie = len(pivot) - win - loss
            f.write(f"- **CSTA vs {ref}**: {win} W / {tie} T / {loss} L\n")

if __name__ == "__main__":
    merge_and_report("results/minirocket_final20_core_v1", "minirocket_core_report.md")
    merge_and_report("results/minirocket_final20_batchB_v1", "minirocket_batchB_report.md")
    final_mechanism_audit("results/minirocket_final20_core_v1", "results/minirocket_final20_batchB_v1", "results/minirocket_final_mechanism_audit.md")
