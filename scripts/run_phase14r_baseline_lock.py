#!/usr/bin/env python
"""Phase 14R Baseline Lock: Seeds 0-9.

Baselines:
1. Aligned (1s): sample + LR(lbfgs) + [C sweep] + [Agg sweep]
2. Peak (4s): sample + LR(lbfgs) + [C sweep] + [Agg sweep]

Note: Switched Peak from LinearSVC to LR based on Seed 4 results (LR=0.748 > LinearSVC=0.726).
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# reuse functions from rev2 script
from run_phase14r_step6b1_rev2 import run_single_seed_block_wise, SeedProcessedTrialDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--out-root", default="promoted_results/phase14r/step6_lock")
    args = parser.parse_args()
    
    # Configuration for Lock
    # We restrict the grid to only the necessary components
    cfg = {
        "processed_root": "data/SEED/SEED_EEG/Preprocessed_EEG",
        "stim_xlsx": "data/SEED/SEED_EEG/SEED_stimulation.xlsx",
        "out_root": args.out_root,
        "use_bands": 0, # Strict mode
        "bands": "delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50",
        "spd_eps": 1e-4,
        "hop": 1.0,
        "center": "logcenter", # Fixed
        "standardize": True,   # Fixed
        "max_iter": 1000,
        
        # Grid - The "Two Baselines"
        "windows": [1.0, 4.0],
        "cov_est": ["sample"],
        "clf": ["lr_saga"], # Implies lbfgs in our script
        "C_list": [0.1, 1.0, 10.0],
        "trial_agg": ["majority", "meanlogit"]
    }
    
    # Load Data
    print(f"Loading Dataset...")
    ds = SeedProcessedTrialDataset(cfg["processed_root"], cfg["stim_xlsx"])
    all_trials = sorted(list(ds), key=lambda x: x["trial_id_str"])
    print(f"Loaded {len(all_trials)} trials.")
    
    all_results = []
    
    for seed in args.seeds:
        print(f"Processing Seed {seed}...")
        # run_single_seed_block_wise handles the block loop and extraction
        res = run_single_seed_block_wise(seed, cfg, all_trials)
        all_results.extend(res)
        
    # Save
    os.makedirs(f"{cfg['out_root']}/seed1", exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(f"{cfg['out_root']}/seed1/summary.csv", index=False)
    print(f"Saved Lock Summary to {cfg['out_root']}/seed1/summary.csv")

if __name__ == "__main__":
    main()
