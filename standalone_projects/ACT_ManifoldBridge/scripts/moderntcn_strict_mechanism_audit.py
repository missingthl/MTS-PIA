import pandas as pd
import glob
import os
import numpy as np

def perform_moderntcn_mechanism_audit():
    print("Starting Strict Mechanism Audit for ModernTCN...")
    
    # 1. Load Core (Rebuilt in P0.5)
    core_path = "results/moderntcn_final20_robustness_v1/per_seed_external_REBUILT.csv"
    if not os.path.exists(core_path):
        print("Core data missing.")
        return
    df_core = pd.read_csv(core_path)
    
    # 2. Load Batch B
    batchb_files = glob.glob("results/moderntcn_final20_batchB_v1/**/per_seed_external.csv", recursive=True)
    df_batchb = pd.concat([pd.read_csv(f) for f in batchb_files], ignore_index=True)
    
    # 3. Join
    combined = pd.concat([df_core, df_batchb], ignore_index=True)
    pivot = combined.pivot_table(index=['dataset', 'seed'], columns='method', values='aug_f1').reset_index()
    
    # 4. Strict Shared-Pair Filter (4 arms required)
    required_arms = ['no_aug', 'csta_topk_uniform_top5', 'random_cov_state', 'pca_cov_state']
    shared = pivot.dropna(subset=required_arms).copy()
    
    # 5. Metrics
    shared['delta_csta'] = shared['csta_topk_uniform_top5'] - shared['no_aug']
    shared['delta_random'] = shared['random_cov_state'] - shared['no_aug']
    shared['delta_pca'] = shared['pca_cov_state'] - shared['no_aug']
    
    def get_wtl(df, col1, col2):
        delta = df[col1] - df[col2]
        w = len(delta[delta > 0.001])
        l = len(delta[delta < -0.001])
        t = len(df) - w - l
        return f"{w} / {t} / {l}"

    # 6. Report
    with open("results/moderntcn_final20_batchB_v1/moderntcn_mechanism_audit.md", "w") as f:
        f.write("# ModernTCN Mechanism Audit (Strict Shared-Pair)\n\n")
        f.write(f"- **Shared Pair N**: {len(shared)}\n")
        f.write(f"- **Datasets**: {shared['dataset'].nunique()}\n\n")
        
        f.write("## 1. Overall Performance Comparison\n")
        f.write("| Arm | Mean Delta (F1) | W/T/L (vs no_aug) |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| CSTA-U5 | {shared['delta_csta'].mean():+.4f} | {get_wtl(shared, 'csta_topk_uniform_top5', 'no_aug')} |\n")
        f.write(f"| Random Cov | {shared['delta_random'].mean():+.4f} | {get_wtl(shared, 'random_cov_state', 'no_aug')} |\n")
        f.write(f"| PCA Cov | {shared['delta_pca'].mean():+.4f} | {get_wtl(shared, 'pca_cov_state', 'no_aug')} |\n")
        
        f.write("\n## 2. Direct Head-to-Head (W/T/L vs CSTA)\n")
        f.write(f"- CSTA-U5 vs Random Cov: {get_wtl(shared, 'csta_topk_uniform_top5', 'random_cov_state')}\n")
        f.write(f"- CSTA-U5 vs PCA Cov: {get_wtl(shared, 'csta_topk_uniform_top5', 'pca_cov_state')}\n")
        
        f.write("\n## 3. Dataset-level Detail\n")
        f.write("| Dataset | Delta CSTA | Delta Random | Delta PCA |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        ds_stats = shared.groupby('dataset')[['delta_csta', 'delta_random', 'delta_pca']].mean().sort_values(by='delta_csta', ascending=False)
        for ds, row in ds_stats.iterrows():
            f.write(f"| {ds} | {row['delta_csta']:+.4f} | {row['delta_random']:+.4f} | {row['delta_pca']:+.4f} |\n")
        
        f.write("\n\n## 4. Negative Gain Datasets (CSTA)\n")
        neg = ds_stats[ds_stats['delta_csta'] < -0.001]
        if neg.empty:
            f.write("None\n")
        else:
            f.write(neg.index.tolist().__str__() + "\n")

    shared.to_csv("results/moderntcn_final20_batchB_v1/moderntcn_mechanism_audit.csv", index=False)
    print("Mechanism Audit Complete.")

if __name__ == "__main__":
    perform_moderntcn_mechanism_audit()
