#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/run_phase12_quick_fix_check.py

Quick run of Manifold Deep Runner (Seed 0) to verify:
1. It runs without error after the reshape fix.
2. It produces reasonable accuracy (sanity check).
"""

import sys
import os
import argparse
import pandas as pd
import torch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.adapters import get_adapter
from runners.manifold_deep_runner import ManifoldDeepRunner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Small epochs for quick check") # fast
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    # Mock args object for Runner
    class Args:
        torch_device = args.device
        epochs = args.epochs
        batch_size = 32
        mvp1_guided_cov = True # Test guided path too? Or just base?
        # If guided, we need dcnet_ckpt. 
        # Note: We found missing checkpoints before. Let's assume we have seed0 base.
        # Let's run BASELINE first to minimize dependency.
        # mvp1_guided_cov = False 
        dcnet_ckpt = None
        metrics_csv = None
        seed = args.seed
        cov_alpha = 0.01
        hidden_dim = 96
        
    # Baseline args
    args_base = Args()
    args_base.mvp1_guided_cov = False
    
    print(f"--- Quick Check (Seed {args.seed}) Baseline ---")
    
    # Load Data
    adapter = get_adapter("seed1")
    folds = adapter.get_manifold_trial_folds()
    
    # Run Fold 1
    runner = ManifoldDeepRunner(args_base, num_classes=3)
    res = runner.fit_predict(folds['fold1'], fold_name=f"seed{args.seed}_fold1_quick_check")
    
    print("\n--- Result ---")
    print(res)

if __name__ == "__main__":
    main()
