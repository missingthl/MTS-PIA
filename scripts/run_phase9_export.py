
import sys
import os
import argparse
import torch
import numpy as np
sys.path.append(os.getcwd())

from datasets.adapters import Seed1Adapter
from runners.manifold_deep_runner import ManifoldDeepRunner
from runners.spatial_dcnet_torch import SpatialDCNetRunnerTorch

class MockArgs:
    def __init__(self, **kwargs):
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed_raw_debug = False
        self.out_prefix = "" 
        self.__dict__.update(kwargs) # Update overwrite defaults
    def __str__(self):
        return str(self.__dict__)

def run_export():
    seeds = [0, 1, 2]
    
    # --- Manifold Stream ---
    # Protocol: B2-like (Deep, Windowed, Trial Agg).
    # Need to ensure ManifoldDeepRunner exports to: experiments/phase9_fusion/preds/manifold_trial_preds_seed{seed}.csv
    # ManifoldDeepRunner uses 'out_prefix' to parse seed.
    # It hardcodes path to "experiments/phase8_rebaseline/reports/window_trial_preds_seed{seed}.csv" in current code.
    # I should change that path in runner or symlink?
    # Actually I should have modified the path in ManifoldDeepRunner to be generic or directed to Phase 9.
    # Or I can just copy the file after generation.
    # Wait, Step 2170 modified runner to "experiments/phase8_rebaseline...".
    # I didn't update it to Phase 9.
    # I will let it write to Phase 8 path and move it, or update runner again.
    # Updating runner is cleaner. I will pass a specific output dir via args? Runner doesn't use generic args for path.
    # I'll rely on copying/moving.
    
    # --- Manifold Stream ---
    # SKIPPED for Phase 9.5 Spatial Re-run
    adapter = Seed1Adapter()
    
    if False:
        print("=== Phase 9: Manifold Stream ===")
        # adapter = Seed1Adapter() # Moved up
        
        for seed in seeds:
            print(f"Running Manifold SEED {seed}...")
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Load Data
            _ = adapter.get_spatial_folds_for_cnn(
                 seed_de_mode="official", 
                 seed_de_root="data/SEED/SEED_EEG/ExtractedFeatures_1s", 
                 seed_de_var="de_LDS1"
            ) # Init adapter state
            folds = adapter.get_manifold_trial_folds()
            fold = folds['fold1']
            
            # Args
            # Batch Size 1 (Optimal from Phase 8.2)
            # Alpha 0.01 (Default)
            args = MockArgs(
                epochs=30, # Optimized: Plateaued at 30 in previous run
                batch_size=32,
                cov_alpha=0.01,
                hidden_dim=96,
                out_prefix=f"phase9_manifold_seed{seed}"
            )
            
            runner = ManifoldDeepRunner(args, num_classes=3)
            res = runner.fit_predict(fold, fold_name=f"manifold_seed{seed}")
            
            # Move file
            src = f"experiments/phase8_rebaseline/reports/window_trial_preds_seed{seed}.csv"
            dst = f"experiments/phase9_fusion/preds/manifold_trial_preds_seed{seed}.csv"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(src):
                import shutil
                shutil.copy(src, dst)
                print(f"Copied Manifold preds to {dst}")
            else:
                print(f"WARNING: Manifold pred file missing: {src}")
    
            # Move Train File
            src_tr = f"experiments/phase8_rebaseline/reports/window_train_preds_seed{seed}.csv"
            dst_tr = f"experiments/phase9_fusion/preds/manifold_train_preds_seed{seed}.csv"
            if os.path.exists(src_tr):
                import shutil
                shutil.copy(src_tr, dst_tr)
                print(f"Copied Manifold TRAIN preds to {dst_tr}")
            else:
                print(f"WARNING: Manifold TRAIN pred file missing: {src_tr}")
    
    # --- Spatial Stream ---
    print("\n=== Phase 9: Spatial Stream ===")
    # Spatial runner needs 'trial_id_test' in fold.
    # Seed1Adapter now provides it.
    
    for seed in seeds:
        print(f"Running Spatial SEED {seed}...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Load Data (Spatial)
        folds = adapter.get_spatial_folds_for_cnn(
             seed_de_mode="official", 
             seed_de_root="data/SEED/SEED_EEG/ExtractedFeatures_1s", 
             seed_de_var="de_LDS1"
        )
        fold = folds['fold1']
        
        # Args
        # DCNet Default Config
        runner = SpatialDCNetRunnerTorch(
            num_classes=3,
            epochs=40, # Standard
            batch_size=2048,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        # Passing seed in fold_name to trigger export
        res = runner.fit_predict(fold, fold_name=f"spatial_seed{seed}")
        
        # Export logic inside runner handles writing to 'experiments/phase9_fusion/preds/spatial_trial_preds_seed{seed}.csv'
        # So no copy needed if implementation matches plan.

if __name__ == "__main__":
    run_export()
