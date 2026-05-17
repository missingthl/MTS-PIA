import pandas as pd
import glob
import os

def rebuild():
    print("Rebuilding ModernTCN master CSV from atomic run files...")
    # Find all {dataset}_results.csv but ignore sweep_results.csv
    files = glob.glob("results/moderntcn_final20_robustness_v1/_csta_runs/**/*.csv", recursive=True)
    target_files = [f for f in files if "_results.csv" in f and "sweep" not in f]
    
    print(f"Found {len(target_files)} atomic result files.")
    
    final_rows = []
    for f in target_files:
        try:
            df = pd.read_csv(f)
            if df.empty: continue
            
            # Each file contains one dataset-seed results row
            res = df.iloc[0].to_dict()
            ds = res.get('dataset')
            seed = res.get('seed')
            status = res.get('status', 'success')
            
            # Row A: No-Aug
            final_rows.append({
                'dataset': ds,
                'seed': seed,
                'method': 'no_aug',
                'status': status,
                'aug_f1': res.get('base_f1')
            })
            
            # Row B: CSTA
            final_rows.append({
                'dataset': ds,
                'seed': seed,
                'method': 'csta_topk_uniform_top5',
                'status': status,
                'aug_f1': res.get('act_f1')
            })
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    master_df = pd.DataFrame(final_rows)

    out_path = "results/moderntcn_final20_robustness_v1/per_seed_external_REBUILT.csv"
    master_df.to_csv(out_path, index=False)
    print(f"Rebuilt Master CSV saved to {out_path}")
    
    # Audit for missing
    pivot = master_df.pivot_table(index=['dataset', 'seed'], columns='method', values='aug_f1')
    aligned = pivot.dropna(subset=['no_aug', 'csta_topk_uniform_top5'])
    print(f"Aligned Pairs: {len(aligned)} / 60")
    print("Missing Pairs (Seed-level):")
    missing = pivot[pivot.isnull().any(axis=1)]
    print(missing.index.tolist())

if __name__ == "__main__":
    rebuild()
