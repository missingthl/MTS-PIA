
import sys
import os
import torch
import numpy as np
import pandas as pd
sys.path.append(os.getcwd())

from datasets.adapters import Seed1Adapter
from runners.manifold_shared_runner import SharedManifoldRunner

def process_trials(trials_list, y_list, id_list):
    """
    Process already-grouped trials.
    trials_list: List of (T, 310) arrays.
    y_list: (N,) labels.
    id_list: (N,) IDs.
    
    Returns:
        X_out: List of (62, T, 5) arrays (variable T)
        y_out: (N,)
        info_out: List of dicts
    """
    n_trials = len(trials_list)
    
    # 1. Determine Max T
    max_T = 0
    t_sizes = []
    for t in trials_list:
        max_T = max(max_T, t.shape[0])
        t_sizes.append(t.shape[0])
    
    print(f"Processing {n_trials} trials. Max T = {max_T}. Mean T={np.mean(t_sizes):.1f}")
    
    # 2. Allocate List
    X_out = []
    y_out = np.array(y_list, dtype=np.int64)
    info_out = []
    
    for i in range(n_trials):
        raw = trials_list[i] # (T_curr, 310)
        tid = id_list[i]
        
        try:
            # Reshape (T, 62, 5)
            x_reshaped = raw.reshape(raw.shape[0], 62, 5)
            x_permuted = x_reshaped.transpose(1, 0, 2) # (62, T, 5)
            
            # Store Variable T array
            X_out.append(x_permuted)
            
            # Metadata
            parts = str(tid).split('_')
            subj = parts[0] if len(parts) > 0 else 'u'
            sess = parts[1] if len(parts) > 1 else 'u'
            
            info_out.append({
                'trial_id': str(tid),
                'subject': subj,
                'session': sess,
                'split': 'unknown',
                'n_samples': x_permuted.shape[1]
            })
            
        except Exception as e:
            print(f"Error processing trial {i} ({tid}): {e}")
            raise e
            
    return X_out, y_out, info_out

def run_phase9_7():
    seeds = [0] # Minimal run first
    
    print("=== Phase 9.7: Shared-Backbone 5-Band Manifold ===")
    adapter = Seed1Adapter()
    
    for seed in seeds:
        print(f"\nProcessing SEED {seed}...")
        
        # Seeding
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        # Load Data
        print("Loading Data via Adapter...")
        _ = adapter.get_spatial_folds_for_cnn(
             seed_de_mode="official", 
             seed_de_root="data/SEED/SEED_EEG/ExtractedFeatures_1s", 
             seed_de_var="de_LDS1"
        )
        folds = adapter.get_manifold_trial_folds()
        fold = folds['fold1'] # This is TrialFoldData
        
        # Group Data (Already Grouped in FoldData)
        print("Processing Train Data...")
        # Use trials_train instead of X_train
        X_tr, y_tr, info_tr = process_trials(fold.trials_train, fold.y_trial_train, fold.trial_id_train)
        for d in info_tr: d['split'] = 'train'
            
        print("Processing Test Data...")
        X_te, y_te, info_te = process_trials(fold.trials_test, fold.y_trial_test, fold.trial_id_test)
        for d in info_te: d['split'] = 'test'
            
        # Config
        config = {
            'epochs': 50,
            'batch_size': 1, # BS=1 for variable T
            'lr_backbone': 5e-4, # Increased to encourage learning
            'lr_head': 2e-3,     # Increased
            'weight_decay': 1e-3, # Reduced decay
            'dropout': 0.25, # Reduced dropout
            'clip_grad_norm': 1.0
        }
        
        # Run
        runner = SharedManifoldRunner(seed=seed)
        metrics = runner.train_and_evaluate(
            train_data=(X_tr, y_tr, info_tr),
            test_data=(X_te, y_te, info_te),
            config=config
        )
        
        print(f"[Seed {seed}] Completed.")
        print(metrics)

if __name__ == "__main__":
    run_phase9_7()
