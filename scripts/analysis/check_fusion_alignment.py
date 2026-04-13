#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/analysis/check_fusion_alignment.py

Verifies that Spatial and Manifold prediction CSVs are aligned by Trial ID and Label.
"""

import pandas as pd
import sys
import os

def check_alignment(spatial_path, manifold_path):
    print(f"Checking alignment:\n  Spatial: {spatial_path}\n  Manifold: {manifold_path}")
    
    if not os.path.exists(spatial_path):
        print(f"FAIL: Spatial path not found: {spatial_path}")
        return False
    if not os.path.exists(manifold_path):
        print(f"FAIL: Manifold path not found: {manifold_path}")
        return False
        
    df_s = pd.read_csv(spatial_path)
    df_m = pd.read_csv(manifold_path)
    
    # Check Trial IDs
    s_ids = set(df_s['trial_id'].astype(str))
    m_ids = set(df_m['trial_id'].astype(str))
    
    intersect = s_ids.intersection(m_ids)
    
    print(f"Spatial Trials: {len(s_ids)}")
    print(f"Manifold Trials: {len(m_ids)}")
    print(f"Intersection: {len(intersect)}")
    
    if len(s_ids) != len(m_ids) or len(intersect) != len(s_ids):
        print("FAIL: Mismatch in Trial sets!")
        diff_s = s_ids - m_ids
        diff_m = m_ids - s_ids
        if diff_s: print(f"  Only in Spatial: {list(diff_s)[:5]}")
        if diff_m: print(f"  Only in Manifold: {list(diff_m)[:5]}")
        return False
        
    # Check Labels
    # Merge on trial_id
    df_s['trial_id'] = df_s['trial_id'].astype(str)
    df_m['trial_id'] = df_m['trial_id'].astype(str)
    
    merged = pd.merge(df_s, df_m, on='trial_id', suffixes=('_s', '_m'))
    
    # Check True Label consistency
    mismatch = merged[merged['true_label_s'] != merged['true_label_m']]
    if len(mismatch) > 0:
        print(f"FAIL: True Label mismatch on {len(mismatch)} trials!")
        print(mismatch[['trial_id', 'true_label_s', 'true_label_m']].head())
        return False
        
    print("PASS: Alignment Confirmed.")
    return True

if __name__ == "__main__":
    # Hardcoded paths based on LS output
    spatial_path = "experiments/phase9_fusion/preds/spatial_trial_preds_seed0.csv"
    manifold_path = "promoted_results/seed0_fold1_quick_check_preds_test_last_trial.csv"
    
    check_alignment(spatial_path, manifold_path)
