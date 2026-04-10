#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/fusion_postprocess.py

Phase 13 A: Fusion Engine
- V0: Weighted Soft Vote
- V1: Conflict Arbitration (Margin-based)
"""

import pandas as pd
import numpy as np
import os
import json
import argparse

def calculate_metrics(df, pred_col, label_col='y_true'):
    acc = (df[pred_col] == df[label_col]).mean()
    return acc

def run_fusion(spatial_csv, manifold_csv, output_dir, seed=0):
    print(f"Loading Spatial: {spatial_csv}")
    print(f"Loading Manifold: {manifold_csv}")
    
    df_s = pd.read_csv(spatial_csv)
    df_m = pd.read_csv(manifold_csv)
    
    # Ensure String IDs
    df_s['trial_id'] = df_s['trial_id'].astype(str)
    df_m['trial_id'] = df_m['trial_id'].astype(str)
    
    # Standardize column names if needed (map unique names to p0..p2)
    # Assuming Standard Export: prob_0, prob_1, prob_2, y_true/true_label
    
    # Rename 'true_label' to 'y_true' if needed
    if 'true_label' in df_s.columns: df_s.rename(columns={'true_label': 'y_true'}, inplace=True)
    if 'true_label' in df_m.columns: df_m.rename(columns={'true_label': 'y_true'}, inplace=True)
    
    # Merge
    merged = pd.merge(df_s, df_m, on='trial_id', suffixes=('_s', '_m'))
    
    if len(merged) == 0:
        raise ValueError("No overlapping trials found!")
        
    print(f"Fused {len(merged)} trials.")
    
    # Check Label Consistency
    # Use spatial label as ground truth if match
    merged['y_true'] = merged['y_true_s']
    
    # Probas
    ps = merged[['prob_0_s', 'prob_1_s', 'prob_2_s']].values
    pm = merged[['prob_0_m', 'prob_1_m', 'prob_2_m']].values
    
    # Margins (if not present, recalc)
    if 'margin_s' in merged.columns:
        ms = merged['margin_s'].values
    else:
        # Calc margin: Top1 - Top2
        ps_sorted = np.sort(ps, axis=1)
        ms = ps_sorted[:, -1] - ps_sorted[:, -2]
        
    if 'margin_m' in merged.columns:
        mm = merged['margin_m'].values
    else:
        pm_sorted = np.sort(pm, axis=1)
        mm = pm_sorted[:, -1] - pm_sorted[:, -2]
        
    # --- Fusion V0: Weighted (w=0.5) ---
    w = 0.5
    p_v0 = (1-w)*ps + w*pm
    pred_v0 = np.argmax(p_v0, axis=1)
    
    merged['pred_v0'] = pred_v0
    acc_v0 = calculate_metrics(merged, 'pred_v0')
    
    # --- Fusion V1: Arbitration ---
    pred_s = np.argmax(ps, axis=1)
    pred_m = np.argmax(pm, axis=1)
    
    # Logic
    # If agree -> use s
    # If disagree -> use stream with higher margin
    
    agree_mask = (pred_s == pred_m)
    conflict_mask = ~agree_mask
    
    final_pred_v1 = np.zeros_like(pred_s)
    
    # Agree case
    final_pred_v1[agree_mask] = pred_s[agree_mask]
    
    # Conflict case
    # If Ms > Mm -> S, else M
    # Note: user said "Difference delta tuning later", initially 0.
    s_wins = (ms > mm)
    
    # Indices where Conflict AND S wins
    idx_s_wins = conflict_mask & s_wins
    # Indices where Conflict AND M wins
    idx_m_wins = conflict_mask & (~s_wins)
    
    final_pred_v1[idx_s_wins] = pred_s[idx_s_wins]
    final_pred_v1[idx_m_wins] = pred_m[idx_m_wins]
    
    merged['pred_v1'] = final_pred_v1
    acc_v1 = calculate_metrics(merged, 'pred_v1')
    
    # Save V0, V1
    os.makedirs(output_dir, exist_ok=True)
    merged[['trial_id', 'y_true', 'pred_v0']].to_csv(os.path.join(output_dir, "fusion_v0_pred.csv"), index=False)
    merged[['trial_id', 'y_true', 'pred_v1']].to_csv(os.path.join(output_dir, "fusion_v1_pred.csv"), index=False)
    
    # Analysis
    acc_s = (pred_s == merged['y_true']).mean()
    acc_m = (pred_m == merged['y_true']).mean()
    
    agree_rate = agree_mask.mean()
    conflict_rate = conflict_mask.mean()
    
    # Conflict Subset Analysis
    subset_conflict = merged[conflict_mask]
    if len(subset_conflict) > 0:
        conf_acc_s = (subset_conflict['pred_s_s'] if 'pred_s_s' in subset_conflict else pred_s[conflict_mask] == subset_conflict['y_true']).mean()
        # Actually simplest re-calculation
        conf_acc_s = (pred_s[conflict_mask] == subset_conflict['y_true']).mean()
        conf_acc_m = (pred_m[conflict_mask] == subset_conflict['y_true']).mean()
        conf_acc_v1 = (final_pred_v1[conflict_mask] == subset_conflict['y_true']).mean()
    else:
        conf_acc_s, conf_acc_m, conf_acc_v1 = 0.0, 0.0, 0.0
        
    summary = {
        "spatial_acc": acc_s,
        "manifold_acc": acc_m,
        "fusion_v0_acc": acc_v0,
        "fusion_v1_acc": acc_v1,
        "agree_rate": agree_rate,
        "conflict_rate": conflict_rate,
        "conflict_acc_spatial": conf_acc_s,
        "conflict_acc_manifold": conf_acc_m,
        "conflict_acc_v1": conf_acc_v1,
        "improvement_on_conflict": conf_acc_v1 - max(conf_acc_s, conf_acc_m)
    }
    
    print("\nFusion Summary:")
    print(json.dumps(summary, indent=2))
    
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial", required=True)
    parser.add_argument("--manifold", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    run_fusion(args.spatial, args.manifold, args.out_dir, args.seed)
