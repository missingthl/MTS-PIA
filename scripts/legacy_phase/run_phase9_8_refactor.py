
import sys
import os
import torch
import numpy as np
import pandas as pd
import json

# Add project root to path
sys.path.append(os.getcwd())

from datasets.adapters import Seed1Adapter
from runners.spatial_dcnet_torch import SpatialDCNetRunnerTorch

def run_phase9_8():
    seeds = [3, 4] # Run for missing seeds
    print("=== Phase 9.8: DCNet Refactor Verification ===")
    
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
        folds_spatial = adapter.get_spatial_folds_for_cnn(
             seed_de_mode="official", 
             seed_de_root="data/SEED/SEED_EEG/ExtractedFeatures_1s", 
             seed_de_var="de_LDS1"
        )
        fold_s = folds_spatial['fold1']
        
        print("Starting DCNet Training (Refactored)...")
        runner = SpatialDCNetRunnerTorch(
            num_classes=3,
            epochs=40, # Sufficient for convergence
            batch_size=128,
            learning_rate=1e-4,
            num_workers=4
        )
        
        metrics = runner.fit_predict(fold_s, f"seed{seed}_refactor")
        
        print(f"[Seed {seed}] Completed.")
        print(metrics)
        
        # Verify Accuracy
        acc = metrics['best_val_acc']
        print(f"Test Accuracy: {acc:.4f}")
        
        if acc < 0.8:
            print("WARNING: Accuracy below 80%. Refactor might have issues.")
        else:
            print("SUCCESS: Accuracy verified.")

if __name__ == "__main__":
    run_phase9_8()
