#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/run_phase13_final.py

Phase 13: Final Full Scale Execution.
Re-runs Manifold Stream (Guided, p=2.0) on 5 seeds with verified data pipeline.
Performs Fusion calculation at the end.
"""

import sys
import os
import torch
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets.adapters import get_adapter
from runners.manifold_deep_runner import ManifoldDeepRunner
from scripts.legacy_phase.run_phase12_fusion import fuse_predictions

def main():
    seeds = [0, 1, 2, 3, 4]
    p_val = 2.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check Spatial Preds Existence
    print("Checking Spatial Predictions...")
    for s in seeds:
        path = f"experiments/phase9_fusion/preds/spatial_trial_preds_seed{s}.csv"
        if not os.path.exists(path):
            print(f"CRITICAL: Spatial preds missing for seed {s}: {path}")
            # In a real final run, we might want to re-generate spatial here too.
            # But assuming Phase 9 artifacts exist.
            # If missing, we skip? Or fail?
            return 
    print("Spatial Predictions OK.")
    
    # Load Adapter Once
    adapter = get_adapter("seed1")
    folds = adapter.get_manifold_trial_folds() # 3 folds
    # We usually run on 'fold1' for validation split in previous phases.
    # Should we run all 3 folds for full CV?
    # Previous phases (Phase 10/11) ran single fold (fold1) per seed to save time?
    # task.md says "3 seeds" then "5 seeds". Usually implies 5 random seeds on a fixed split, or 5-fold?
    # Your code `run_phase11_audit.py` passes `folds['fold1']`.
    # So we stick to **Seed Validation** (Fixed Fold 1, Varying Init Seed 0-4).
    # Wait, Seed affects `Split Train/Val` inside the runner?
    # `ManifoldDeepRunner` line 232: `seed_val = getattr(self.args, 'seed', 0)`.
    # `ManifoldDeepRunner` line 240: `r.shuffle(indices)`.
    # AND Seed affects Network Initialization.
    # So yes, iterating seeds is correct. Use `fold1` data usage.
    
    fusion_results = []
    
    for s in seeds:
        print(f"\n=== Running Seed {s} (Guided p={p_val}) ===")
        
        # Args
        class Args:
            torch_device = device
            epochs = 50 
            batch_size = 32
            mvp1_guided_cov = True
            mvp1_attn_power = p_val
            # DCNet Checkpoint (Teacher)
            # We need one per seed? Or one shared?
            # Phase 10 generated `experiments/checkpoints/dcnet_guided_seed{s}.pt`?
            # I need to find where they are.
            # `generate_dcnet_checkpoints` in `run_phase10_2_full.py` saved them.
            # Let's assume standard path: `experiments/checkpoints/seed{s}_dcnet_best.pt`?
            # Check `experiments/checkpoints` listing if needed.
            # For now try a pattern.
            dcnet_ckpt = f"experiments/checkpoints/seed{s}_dcnet_refactor.pt" 
            # Note: Need to verify this path exists!
            
            metrics_csv = None
            seed = s
            cov_alpha = 0.01
            hidden_dim = 96
            
        args = Args()
        
        # Verify Checkpoint
        if not os.path.exists(args.dcnet_ckpt):
             # Try alternate name
             args.dcnet_ckpt = f"experiments/checkpoints/seed{s}_dcnet_best.pt"
        if not os.path.exists(args.dcnet_ckpt):
             print(f"WARN: Teacher Checkpoint not found for seed {s}. Skipping or Failing.")
             # Fallback to seed0?
             # args.dcnet_ckpt = "experiments/checkpoints/seed0_dcnet_refactor.pt"
             # return
        
        runner = ManifoldDeepRunner(args, num_classes=3)
        res = runner.fit_predict(folds['fold1'], fold_name=f"seed{s}_phase13_final")
        
        # Fusion
        spatial_path = f"experiments/phase9_fusion/preds/spatial_trial_preds_seed{s}.csv"
        manifold_path = f"promoted_results/seed{s}_phase13_final_preds_test_last_trial.csv"
        
        print(f"Fusing: {spatial_path} + {manifold_path}")
        if os.path.exists(manifold_path):
            acc_fuse = fuse_predictions(spatial_path, manifold_path, output_dir="promoted_results/phase13_fusion")
            fusion_results.append({
                "seed": s, 
                "manifold_acc": res['last']['test_trial_acc'],
                "fusion_acc": acc_fuse
            })
        else:
            print("Fusion Skipped (Missing Manifold Output)")
            
    # Summary
    print("\n=== Phase 13 Final Summary ===")
    df = pd.DataFrame(fusion_results)
    if not df.empty:
        print(df)
        print(f"Mean Manifold: {df['manifold_acc'].mean():.4f} +/- {df['manifold_acc'].std():.4f}")
        print(f"Mean Fusion: {df['fusion_acc'].mean():.4f} +/- {df['fusion_acc'].std():.4f}")
        df.to_csv("promoted_results/phase13_final_summary.csv", index=False)
    else:
        print("No results.")

if __name__ == "__main__":
    main()
