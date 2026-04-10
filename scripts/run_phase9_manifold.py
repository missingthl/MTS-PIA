
import sys
import os
import argparse
import torch
import numpy as np
sys.path.append(os.getcwd())

from datasets.adapters import Seed1Adapter
from runners.manifold_deep_runner import ManifoldDeepRunner

class MockArgs:
    def __init__(self, **kwargs):
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed_raw_debug = False
        self.out_prefix = "" 
        self.__dict__.update(kwargs)
    def __str__(self):
        return str(self.__dict__)

def run_export():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Specific seed to run (optional)")
    args_cli = parser.parse_args()
    
    print("=== Phase 9: Manifold Stream Export ===")
    adapter = Seed1Adapter()
    
    if args_cli.seed is not None:
        seeds = [args_cli.seed]
        print(f"Running Specific Seed: {seeds[0]}")
    else:
        seeds = [0, 1, 2] # Run all seeds
    
    # Load Data ONCE if possible?
    # Adapter methods reload. We will bear the cost (it worked before).
    
    for seed in seeds:
        print(f"\nRunning Manifold SEED {seed}...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Init Adapter (Required for Manifold runner internal state)
        _ = adapter.get_spatial_folds_for_cnn(
             seed_de_mode="official", 
             seed_de_root="data/SEED/SEED_EEG/ExtractedFeatures_1s", 
             seed_de_var="de_LDS1"
        )
        
        folds = adapter.get_manifold_trial_folds()
        fold = folds['fold1']
        
        # Args: Fast Mode (BS=32, Ep=30)
        args = MockArgs(
            epochs=30, 
            batch_size=32, 
            cov_alpha=0.01,
            hidden_dim=96,
            out_prefix=f"phase9_manifold_seed{seed}"
        )
        
        runner = ManifoldDeepRunner(args, num_classes=3)
        res = runner.fit_predict(fold, fold_name=f"manifold_seed{seed}")
        
        # Move files
        # Test Preds
        src = f"experiments/phase8_rebaseline/reports/window_trial_preds_seed{seed}.csv"
        dst = f"experiments/phase9_fusion/preds/manifold_trial_preds_seed{seed}.csv"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(src):
            import shutil
            shutil.copy(src, dst)
            print(f"Copied Manifold preds to {dst}")
        else:
            print(f"WARNING: Manifold pred file missing: {src}")

        # Train Preds
        src_tr = f"experiments/phase8_rebaseline/reports/window_train_preds_seed{seed}.csv"
        dst_tr = f"experiments/phase9_fusion/preds/manifold_train_preds_seed{seed}.csv"
        if os.path.exists(src_tr):
            import shutil
            shutil.copy(src_tr, dst_tr)
            print(f"Copied Manifold TRAIN preds to {dst_tr}")
        else:
            print(f"WARNING: Manifold TRAIN pred file missing: {src_tr}")

if __name__ == "__main__":
    run_export()
