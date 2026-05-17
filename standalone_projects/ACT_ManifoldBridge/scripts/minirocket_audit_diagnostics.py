import pandas as pd
import glob
import os

def detailed_audit(root_dir):
    print(f"Starting Diagnostic Audit for: {root_dir}")
    csv_files = glob.glob(os.path.join(root_dir, "**/per_seed_external.csv"), recursive=True)
    
    all_rows = []
    for f in csv_files:
        all_rows.append(pd.read_csv(f))
    
    if not all_rows:
        print("ERROR: No data found.")
        return
        
    df = pd.concat(all_rows, ignore_index=True)
    
    # 1. Basic Stats
    expected_datasets = ["handwriting", "uwavegesturelibrary", "ering", "motorimagery", "natops", "epilepsy", "articularywordrecognition", "har", "japanesevowels", "pendigits", "basicmotions", "cricket", "racketsports", "ethanolconcentration", "libras", "heartbeat", "fingermovements", "selfregulationscp2", "atrialfibrillation", "handmovementdirection"]
    expected_seeds = [1, 2, 3]
    expected_arms = ["no_aug", "csta_topk_uniform_top5"]
    
    expected_total_rows = len(expected_datasets) * len(expected_seeds) * len(expected_arms)
    actual_total_rows = len(df)
    
    # 2. Duplicate Check
    duplicates = df[df.duplicated(subset=['dataset', 'seed', 'method'], keep=False)]
    
    # 3. Completeness Check
    missing_items = []
    found_pairs = 0
    missing_pairs = []
    
    pivot = df.pivot_table(index=['dataset', 'seed'], columns='method', values='aug_f1')
    
    for ds in expected_datasets:
        for s in expected_seeds:
            # Check individual arms
            for arm in expected_arms:
                mask = (df['dataset'] == ds) & (df['seed'] == s) & (df['method'] == arm)
                if len(df[mask]) == 0:
                    missing_items.append(f"{ds}-s{s}-{arm}")
            
            # Check pair completeness
            if (ds, s) in pivot.index:
                row = pivot.loc[(ds, s)]
                if not pd.isna(row.get('no_aug')) and not pd.isna(row.get('csta_topk_uniform_top5')):
                    found_pairs += 1
                else:
                    missing_pairs.append(f"{ds}-s{s}")
            else:
                missing_pairs.append(f"{ds}-s{s}")

    # 4. Filtered Check (NaNs in aug_f1)
    nan_rows = df[df['aug_f1'].isna()]

    # 5. Output Results
    print("\n" + "="*50)
    print("MINIROCKET CORE AUDIT RESULTS")
    print("="*50)
    print(f"Expected Pair Count (20x3)  : 60")
    print(f"Actual Pair Count (Complete): {found_pairs}")
    print(f"Total Rows Found            : {actual_total_rows} / {expected_total_rows}")
    print(f"Tie Threshold (used in Rpt) : 0.001")
    print("-" * 30)
    
    if duplicates.empty:
        print("Duplicate Pairs             : None")
    else:
        print(f"Duplicate Pairs             : {len(duplicates)} found!")
        print(duplicates[['dataset', 'seed', 'method']])

    print(f"Missing Individual Arms     : {len(missing_items)}")
    if missing_items:
        print(f"  First 5 missing: {missing_items[:5]}")

    print(f"Missing/Incomplete Pairs    : {len(missing_pairs)}")
    if missing_pairs:
        print(f"  Missing Pairs: {missing_pairs}")

    print(f"Filtered Pairs (NaN F1)     : {len(nan_rows)}")
    print("="*50)

if __name__ == "__main__":
    detailed_audit("results/minirocket_final20_core_v1")
