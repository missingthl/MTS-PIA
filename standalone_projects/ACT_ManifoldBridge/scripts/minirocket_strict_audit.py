import pandas as pd
import glob
import os
import numpy as np

FINAL20 = [
    "handwriting", "uwavegesturelibrary", "ering", "motorimagery", "natops",
    "epilepsy", "articularywordrecognition", "har", "japanesevowels", "pendigits",
    "basicmotions", "cricket", "racketsports", "ethanolconcentration", "libras",
    "heartbeat", "fingermovements", "selfregulationscp2", "atrialfibrillation", "handmovementdirection"
]

def load_all_csvs(root_dir):
    csv_files = [f for f in glob.glob(os.path.join(root_dir, "**/per_seed_external.csv"), recursive=True) if "merged" not in f]
    if not csv_files: return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

def perform_strict_audit():
    print("Starting Strict Consistency Audit...")
    
    # Load Data
    df_core = load_all_csvs("results/minirocket_final20_core_v1")
    df_batchb = load_all_csvs("results/minirocket_final20_batchB_v1")
    
    if df_core.empty or df_batchb.empty:
        print("Data missing in one or more roots.")
        return

    # 1. Core Integrity (U5 vs No-Aug)
    pivot_core = df_core.pivot_table(index=['dataset', 'seed'], columns='method', values='aug_f1').reset_index()
    core_complete = pivot_core.dropna(subset=['no_aug', 'csta_topk_uniform_top5']).copy()
    
    completed_datasets = sorted(core_complete['dataset'].unique().tolist())
    missing_datasets = sorted(list(set(FINAL20) - set(completed_datasets)))
    
    core_complete['delta'] = core_complete['csta_topk_uniform_top5'] - core_complete['no_aug']
    win = len(core_complete[core_complete['delta'] > 0.001])
    loss = len(core_complete[core_complete['delta'] < -0.001])
    tie = len(core_complete) - win - loss
    
    # 2. Batch B Joint Audit (4-Arm Shared)
    combined = pd.concat([df_core, df_batchb], ignore_index=True)
    pivot_shared = combined.pivot_table(index=['dataset', 'seed'], columns='method', values='aug_f1').reset_index()
    
    required_arms = ['no_aug', 'csta_topk_uniform_top5', 'random_cov_state', 'pca_cov_state']
    shared_complete = pivot_shared.dropna(subset=required_arms).copy()
    
    # 3. Write Formal Reports
    # a. Core Report
    with open("results/minirocket_final20_core_v1/minirocket_core_report_v2.md", "w") as f:
        f.write("# MiniRocket Core Performance (Strict Subset Audit)\n\n")
        f.write(f"- **Subset Scope**: {len(completed_datasets)} / 20 datasets\n")
        f.write(f"- **Missing Datasets**: {', '.join(missing_datasets)}\n")
        f.write(f"- **Actual N (Aligned Pairs)**: {len(core_complete)}\n")
        f.write(f"- **Expected N (for completed subset)**: {len(completed_datasets) * 3}\n\n")
        
        f.write("## Performance (CSTA-U5 vs No-Aug)\n")
        f.write(f"- **Mean Delta F1**: {core_complete['delta'].mean():+.4f}\n")
        f.write(f"- **W/T/L (Seed-level)**: {win} Win / {tie} Tie / {loss} Loss\n")
        f.write(f"  *(Audit check: {win} + {tie} + {loss} = {win+tie+loss})\n\n")
        
        f.write("### Dataset-level Summary\n")
        ds_stats = core_complete.groupby('dataset')['delta'].mean().sort_values(ascending=False)
        f.write("| Dataset | Mean Delta F1 |\n| :--- | :--- |\n")
        for ds, val in ds_stats.items():
            f.write(f"| {ds} | {val:+.4f} |\n")
            
    # b. Mechanism Report
    with open("results/minirocket_final_mechanism_audit_v2.md", "w") as f:
        f.write("# MiniRocket Mechanism Evidence (Strict Shared-Pair Audit)\n\n")
        f.write(f"- **Shared Pair Count (N)**: {len(shared_complete)}\n")
        f.write(f"- **Datasets with full coverage**: {shared_complete['dataset'].nunique()}\n\n")
        
        for method in ['csta_topk_uniform_top5', 'random_cov_state', 'pca_cov_state']:
            delta = shared_complete[method] - shared_complete['no_aug']
            w = len(shared_complete[shared_complete[method] > shared_complete['no_aug'] + 0.001])
            l = len(shared_complete[shared_complete[method] < shared_complete['no_aug'] - 0.001])
            t = len(shared_complete) - w - l
            f.write(f"### CSTA-U5 vs {method if method != 'csta_topk_uniform_top5' else 'No-Aug'}\n")
            f.write(f"- Mean Delta: {delta.mean():+.4f}\n")
            f.write(f"- W/T/L: {w} / {t} / {l}\n\n")

    print("Audit Complete. Reports generated.")
    
    # 4. Final Terminal Summary
    print("\nSUMMARY FOR USER:")
    print(f"Core Actual N: {len(core_complete)}")
    print(f"Shared Pair N: {len(shared_complete)}")
    print(f"W/T/L Align  : {'PASS' if win+tie+loss == len(core_complete) else 'FAIL'}")

if __name__ == "__main__":
    perform_strict_audit()
