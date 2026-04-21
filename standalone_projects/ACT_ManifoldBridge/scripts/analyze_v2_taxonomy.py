import pandas as pd
import numpy as np
import os
import argparse

def classify_dataset_regime(row):
    """
    Classifies a dataset based on ACT v2 Taxonomy logic.
    """
    gain = row['f1_gain_pct_mean']
    conflict = row['host_conflict_rate_mean']
    fidelity = row['fidelity_score_mean_mean']
    
    # Define Conflict Regimes
    if 0.45 <= conflict <= 0.55:
        conflict_label = "near_orthogonal"
    elif conflict < 0.45:
        conflict_label = "synergy_biased"
    else:
        conflict_label = "conflict_biased"
        
    # Heuristic Regime Classification
    if gain > 2.0:
        if conflict_label == "near_orthogonal":
            # If gain is high and orthogonal, it's a boundary-shaping success
            regime = "boundary_shaping"
        else:
            regime = "representation_first"
    elif gain < -1.0:
        regime = "objective_mismatch"
    else:
        if row['base_f1_mean'] > 0.95:
            regime = "saturated"
        else:
            regime = "variance_open"
            
    return regime, conflict_label

def main(csv_path, out_dir):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Metrics to aggregate
    metrics = [
        'base_f1', 'act_f1', 'f1_gain_pct', 
        'host_conflict_rate', 'host_geom_cosine_mean', 
        'fidelity_score_mean', 'entropy_shift_mean' # Assuming entropy_shift exists or is added
    ]
    # Filter only available metrics
    available_metrics = [m for m in metrics if m in df.columns]
    
    stats = df.groupby('dataset')[available_metrics].agg(['mean', 'std']).reset_index()
    stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
    
    # Manual patching for entropy_shift if missing in CSV but implied in theory
    if 'entropy_shift_mean_mean' not in stats.columns:
        stats['entropy_shift_mean_mean'] = 0.0 # Placeholder
        
    regimes = []
    conflict_labels = []
    for _, row in stats.iterrows():
        r, c = classify_dataset_regime(row)
        regimes.append(r)
        conflict_labels.append(c)
        
    stats['v2_regime'] = regimes
    stats['conflict_regime'] = conflict_labels
    
    print("\n--- ACT v2 Taxonomy Summary ---")
    summary = stats[['dataset', 'f1_gain_pct_mean', 'host_conflict_rate_mean', 'v2_regime', 'conflict_regime']]
    print(summary.to_string(index=False))
    
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "v2_taxonomy_table.csv")
    stats.to_csv(report_path, index=False)
    
    # Generate Note
    note_path = os.path.join(out_dir, "taxonomy_note.md")
    with open(note_path, "w") as f:
        f.write("# ACT v2 Taxonomy Note\n")
        f.write("Generated from Reference Protocol sweep.\n\n")
        f.write("| Regime | Count | Description |\n")
        f.write("| :--- | :--- | :--- |\n")
        for r in stats['v2_regime'].unique():
            count = len(stats[stats['v2_regime'] == r])
            f.write(f"| {r} | {count} | Automatic classification based on gain and alignment |\n")
            
    print(f"\nFinal assets stored in {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/final_paper_sweep_v1/sweep_results.csv")
    parser.add_argument("--out-dir", type=str, default="results/v2_taxonomy_analysis")
    args = parser.parse_args()
    main(args.csv, args.out_dir)
