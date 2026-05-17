import pandas as pd
import numpy as np
import os

def audit_and_report(csv_path, output_report_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Audit - Basic stats
    total_rows = len(df)
    
    # 2. Audit - Uniqueness
    unique_keys = df.set_index(['dataset', 'seed', 'method']).index.is_unique
    duplicate_count = total_rows - len(df.drop_duplicates(['dataset', 'seed', 'method']))
    
    # 3. Audit - Missing datasets/seeds/arms
    expected_datasets = [
        "handwriting","uwavegesturelibrary","ering","motorimagery","natops",
        "epilepsy","articularywordrecognition","har","japanesevowels","pendigits",
        "basicmotions","cricket","racketsports","ethanolconcentration","libras",
        "heartbeat","fingermovements","selfregulationscp2","atrialfibrillation",
        "handmovementdirection"
    ]
    expected_seeds = [1, 2, 3]
    expected_arms = ["no_aug", "csta_topk_uniform_top5"]
    
    missing = []
    for d in expected_datasets:
        for s in expected_seeds:
            for a in expected_arms:
                exists = df[(df['dataset'] == d) & (df['seed'] == s) & (df['method'] == a)]
                if len(exists) == 0:
                    missing.append(f"{d}_s{s}_{a}")
    
    # 4. Audit - Backbone and Arms
    wrong_backbone = df[df['backbone'] != 'moderntcn']['backbone'].unique()
    wrong_arms = df[~df['method'].isin(expected_arms)]['method'].unique()
    
    # 5. Audit - Status and NaN
    failed_rows = df[df['status'] != 'success']
    nan_f1 = df[df['aug_f1'].isna()]
    
    # 6. Audit - Hparams (if columns exist)
    # Note: hparams might be embedded in run_config or separate columns. 
    # Based on external_baseline_manifest, they should be in dedicated columns if logged.
    # We will check 'pia_gamma', 'eta_safe', 'multiplier', 'k_dir' if they exist.
    hparam_issues = []
    checks = {'pia_gamma': 0.1, 'eta_safe': 0.75, 'multiplier': 10, 'k_dir': 10}
    for col, val in checks.items():
        if col in df.columns:
            # We filter by CSTA rows only for these checks
            csta_df = df[df['method'] == 'csta_topk_uniform_top5']
            deviations = csta_df[~np.isclose(csta_df[col].astype(float), val)]
            if len(deviations) > 0:
                hparam_issues.append(f"{col} deviation: {deviations[col].unique()}")

    # Prepare Report Content
    with open(output_report_path, 'w') as f:
        f.write("# ModernTCN Final20 Robustness Audit & Report\n\n")
        
        f.write("## 1. Integrity Audit Results\n")
        f.write(f"- **Total Rows**: {total_rows} (Expected: 120 + header)\n")
        f.write(f"- **Uniqueness**: {'PASS' if unique_keys else 'FAIL'} (Duplicates: {duplicate_count})\n")
        f.write(f"- **Missing Rows**: {len(missing)} items missing\n")
        if missing:
            f.write(f"  - Missing: {missing}\n")
        f.write(f"- **Backbone Check**: {'PASS' if len(wrong_backbone) == 0 else 'FAIL'} (Found: {wrong_backbone})\n")
        f.write(f"- **Arms Check**: {'PASS' if len(wrong_arms) == 0 else 'FAIL'} (Found: {wrong_arms})\n")
        f.write(f"- **Status Check**: {'PASS' if len(failed_rows) == 0 else 'FAIL'} (Failed: {len(failed_rows)})\n")
        if not failed_rows.empty:
            f.write(f"  - Failed Rows: {failed_rows[['dataset', 'seed', 'method', 'fail_reason']].to_dict('records')}\n")
        f.write(f"- **NaN Check**: {'PASS' if len(nan_f1) == 0 else 'FAIL'} (NaN F1: {len(nan_f1)})\n")
        f.write(f"- **Hparam Audit**: {'PASS' if not hparam_issues else 'FAIL'}\n")
        if hparam_issues:
            f.write(f"  - Issues: {hparam_issues}\n")
        
        f.write("\n---\n\n")
        
        # Performance Analysis
        f.write("## 2. Core Performance Analysis (CSTA-U5 vs No-Aug)\n")
        f.write("Comparison Pairs: 60 (20 datasets x 3 seeds)\n\n")
        
        # Pivot for comparison
        pivot_df = df.pivot_table(index=['dataset', 'seed'], columns='method', values='aug_f1').reset_index()
        if 'no_aug' in pivot_df.columns and 'csta_topk_uniform_top5' in pivot_df.columns:
            pivot_df['delta'] = pivot_df['csta_topk_uniform_top5'] - pivot_df['no_aug']
            
            mean_delta = pivot_df['delta'].mean()
            win = len(pivot_df[pivot_df['delta'] > 0.001])
            tie = len(pivot_df[np.abs(pivot_df['delta']) <= 0.001])
            loss = len(pivot_df[pivot_df['delta'] < -0.001])
            
            f.write(f"- **Mean Delta (F1)**: {mean_delta:+.4f}\n")
            f.write(f"- **W/T/L (Threshold 0.001)**: {win} / {tie} / {loss}\n\n")
            
            f.write("### Dataset-level Performance Breakdown (Mean over 3 seeds)\n\n")
            ds_breakdown = pivot_df.groupby('dataset')['delta'].mean().sort_values(ascending=False)
            f.write("| Dataset | Mean Delta F1 |\n")
            f.write("| :--- | :--- |\n")
            for ds, d_val in ds_breakdown.items():
                f.write(f"| {ds} | {d_val:+.4f} |\n")
        else:
            f.write("ERROR: Missing one of the required methods for comparison.\n")

    print(f"Audit and Report generated at {output_report_path}")

if __name__ == "__main__":
    audit_and_report(
        "results/moderntcn_final20_robustness_v1/per_seed_external.csv",
        "results/moderntcn_final20_robustness_v1/moderntcn_csta_vs_noaug_report.md"
    )
