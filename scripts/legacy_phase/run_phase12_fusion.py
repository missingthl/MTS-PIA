#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/run_phase12_fusion.py

Performs Weighted Average Fusion on Spatial and Manifold predictions.
"""

import pandas as pd
import numpy as np
import sys
import os
import argparse

def fuse_predictions(spatial_path, manifold_path, output_dir="promoted_results/fusion"):
    print(f"Fusion: {spatial_path} + {manifold_path}")
    
    if not os.path.exists(spatial_path) or not os.path.exists(manifold_path):
        print("Error: Input files not found.")
        return
        
    df_s = pd.read_csv(spatial_path)
    df_m = pd.read_csv(manifold_path)
    
    # Ensure ID string
    df_s['trial_id'] = df_s['trial_id'].astype(str)
    df_m['trial_id'] = df_m['trial_id'].astype(str)
    
    # Merge
    merged = pd.merge(df_s, df_m, on='trial_id', suffixes=('_s', '_m'))
    
    if len(merged) == 0:
        print("Error: No overlapping trials found!")
        return

    # Check Label Consistency
    # Assuming 'true_label_s' available. If not, check 'true_label'.
    lbl_col_s = 'true_label_s' if 'true_label_s' in merged else 'true_label'
    if 'true_label_s' in merged and 'true_label_m' in merged:
        mismatch = (merged['true_label_s'] != merged['true_label_m']).sum()
        if mismatch > 0:
            print(f"Warning: {mismatch} label mismatches found! Using Spatial labels.")
            
    y_true = merged['true_label_s'].values
    
    # Extract Probas
    # Spatial cols: prob_0, prob_1, prob_2 (or prob_0_s, etc.)
    # Manifold cols: prob_0, prob_1, prob_2
    
    ps = merged[['prob_0_s', 'prob_1_s', 'prob_2_s']].values
    pm = merged[['prob_0_m', 'prob_1_m', 'prob_2_m']].values
    
    results = []
    
    print(f"Aligned {len(merged)} trials.")
    
    # Sweep Alpha
    for alpha in np.linspace(0, 1, 21): # 0.05 steps
        p_fused = alpha * ps + (1 - alpha) * pm
        preds = np.argmax(p_fused, axis=1)
        acc = np.mean(preds == y_true)
        results.append({"alpha": alpha, "acc": acc})
        
    df_res = pd.DataFrame(results)
    best = df_res.loc[df_res['acc'].idxmax()]
    
    print("\n--- Fusion Results ---")
    print(df_res.to_string(index=False))
    print(f"Best Alpha: {best['alpha']:.2f}, Acc: {best['acc']:.4f}")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_name = os.path.basename(spatial_path).replace("spatial", "fusion")
    df_res.to_csv(os.path.join(output_dir, out_name), index=False)
    
    return best['acc']

if __name__ == "__main__":
    # Example Usage
    # python scripts/run_phase12_fusion.py --spatial experiments/... --manifold promoted/...
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial", required=True)
    parser.add_argument("--manifold", required=True)
    args = parser.parse_args()
    
    fuse_predictions(args.spatial, args.manifold)
