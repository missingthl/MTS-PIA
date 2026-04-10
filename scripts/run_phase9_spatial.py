
import sys
import os
import argparse
import torch
import numpy as np
sys.path.append(os.getcwd())

from datasets.adapters import Seed1Adapter
from runners.spatial_dcnet_torch import SpatialDCNetRunnerTorch

def run_export():
    print("=== Phase 9: Spatial Stream Export ===")
    adapter = Seed1Adapter()
    seeds = [0, 1, 2]
    
    for seed in seeds:
        print(f"\nRunning Spatial SEED {seed}...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Load Data
        folds = adapter.get_spatial_folds_for_cnn(
             seed_de_mode="official", 
             seed_de_root="data/SEED/SEED_EEG/ExtractedFeatures_1s", 
             seed_de_var="de_LDS1"
        )
        fold = folds['fold1']
        
        # Args
        runner = SpatialDCNetRunnerTorch(
            num_classes=3,
            epochs=40,
            batch_size=2048,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        # Passing seed in fold_name to trigger export
        res = runner.fit_predict(fold, fold_name=f"spatial_seed{seed}")
        
        # Export internal to runner writes directly to 'experiments/phase9_fusion/preds/spatial_trial_preds_seed{seed}.csv'

if __name__ == "__main__":
    run_export()
