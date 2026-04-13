#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/run_phase12_validation.py

Phase 12.0 Validation Gate:
1. Dataset Consistency (SEED1 vs SEEDV)
2. Trial ID Alignment/Audit
3. Reshape Logic Check (Mocked due to missing torch)
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets.adapters import get_adapter

def log(msg, buffer):
    print(msg)
    buffer.append(msg)

def check_reshape_logic_mock(log_buffer):
    log("\n[C. Reshape Logic Mock Verification]", log_buffer)
    
    # 1. Simulate Input: Band-Major flattened
    # (Band, Channel) flat
    bands = 5
    chans = 62
    
    # Create data where value indicates Band and Channel
    # Band 0: 1000..1061
    # Band 1: 2000..2061
    flat_310 = np.zeros(bands * chans, dtype=int)
    for b in range(bands):
        for c in range(chans):
            idx = b * chans + c
            flat_310[idx] = (b + 1) * 1000 + c
            
    log(f"Simulated Input (Flat 310): B0C0={flat_310[0]}, B0C61={flat_310[61]}, B1C0={flat_310[62]}", log_buffer)
    
    # 2. Simulate Current Codebase Logic (BUGGY)
    # trial.reshape(T=1, 62, 5)
    try:
        current_reshape = flat_310.reshape(1, 62, 5) # Numpy default is row-major
        # Check Channel 0, Band 4 (Index [0,0,4])
        # Row 0 is [B0C0, B0C1, B0C2, B0C3, B0C4]
        val_buggy = current_reshape[0, 0, 4] # Should be B4C0 (5000), but is B0C4 (1004)
        
        log(f"Current Logic `reshape(T, 62, 5)` result at [0,0,4] (Gamma Ch0): {val_buggy}", log_buffer)
        if val_buggy == 1004:
            log("!! CONFIRMED BUG: Current logic selects Band 0 Channel 4 instead of Band 4 Channel 0.", log_buffer)
        else:
             log(f"Current Logic Result Unexpected: {val_buggy}", log_buffer)
    except Exception as e:
        log(f"Current Logic Error: {e}", log_buffer)

    # 3. Simulate Proposed Fix
    # reshape(T, 5, 62).transpose(0, 2, 1) -> (T, 62, 5)
    try:
        fixed_reshape = flat_310.reshape(1, 5, 62) # (Band, Channel)
        fixed_transpose = fixed_reshape.transpose(0, 2, 1) # (T, Channel, Band)
        
        val_fixed = fixed_transpose[0, 0, 4] # Gamma Ch0
        # fixed_reshape[0, 4, 0] -> Band 4 (row 4), Chan 0 (col 0) -> 5000.
        log(f"Proposed Fix `reshape(T, 5, 62).transpose(0, 2, 1)` result at [0,0,4]: {val_fixed}", log_buffer)
        
        if val_fixed == 5000:
            log("PASS: Proposed fix correctly selects Band 4 Channel 0.", log_buffer)
        else:
            log(f"Proposed Fix Failed? {val_fixed}", log_buffer)
            
    except Exception as e:
        log(f"Proposed Fix Error: {e}", log_buffer)
        
    return True

def main():
    log_buffer = []
    log("=== Phase 12.0 Validation Gate (No Torch) ===\n", log_buffer)
    
    # 1. Dataset Consistency
    log("[A. Dataset Consistency]", log_buffer)
    try:
        adapter = get_adapter("seed1")
        log(f"Adapter: {adapter.name}", log_buffer)
        log(f"Num Classes: {adapter.num_classes}", log_buffer)
    except Exception as e:
        log(f"Adapter Load Error: {e}", log_buffer)
        return

    # 2. Mock Logic Check
    check_reshape_logic_mock(log_buffer)
    
    # 3. Audit Check (E)
    log("\n[E. Audit Check]", log_buffer)
    try:
        # Load Folds
        mf_folds = adapter.get_manifold_trial_folds()
        # For Spatial, use defaults or just try-catch
        # Assuming defaults work or we pass minimal args
        sp_folds = adapter.get_spatial_folds_for_cnn(
             seed_de_root="data/SEED/SEED_EEG/ExtractedFeatures_1s",
             seed_de_var="de_LDS1"
        )
        
        mf_fold1 = mf_folds['fold1']
        sp_fold1 = sp_folds['fold1']
        
        mf_tr_ids = set(mf_fold1.trial_id_train)
        mf_te_ids = set(mf_fold1.trial_id_test)
        
        intersect = mf_tr_ids.intersection(mf_te_ids)
        log(f"Manifold Train/Test Intersection: {len(intersect)}", log_buffer)
        if len(intersect) == 0:
            log("PASS: No intersection.", log_buffer)
        else:
            log(f"FAIL: Intersection found {len(intersect)}", log_buffer)
            
        sp_te_ids = set(sp_fold1.trial_id_test) if sp_fold1.trial_id_test is not None else set()
        log(f"Spatial Test IDs: {len(sp_te_ids)} vs Manifold Test IDs {len(mf_te_ids)}", log_buffer)
        
        if len(mf_te_ids) > 0 and len(mf_te_ids) == len(sp_te_ids):
            log("PASS: Test ID counts match.", log_buffer)
        
    except Exception as e:
        log(f"Data Load Error: {e}", log_buffer)

    # Report
    with open("logs/phase12_validation_report.md", "w") as f:
        f.write("\n".join(log_buffer))
        
if __name__ == "__main__":
    main()
