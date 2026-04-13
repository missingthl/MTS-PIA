#!/usr/bin/env python
"""Phase 15 N2: Feature Cache.

Goal:
- Pre-compute window tokens (Tangent Space at Identity) for all trials.
- Config: 1.0s window, Sample Covariance.
- Output: `data/_cache/phase15_n2_cache/` with `.pt` files per trial.
- This allows fast iteration for N2 without repeated bandpass/cov extraction.

Validation:
- No Pre-compute Bands (Safe Mode).
"""

import os
import sys
import numpy as np
import torch
import argparse
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets.seed_processed_trials import SeedProcessedTrialDataset
from manifold_raw.features import parse_band_spec
from scripts.legacy_phase.run_phase14r_step6b1_rev2 import extract_features_block, logm_spd, vec_utri

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/_cache/phase15_n2_cache")
    parser.add_argument("--processed-root", default="data/SEED/SEED_EEG/Preprocessed_EEG")
    args = parser.parse_args()
    
    # Config matching Aligned Baseline
    win_sec = 1.0
    hop_sec = 1.0
    est_mode = "sample"
    spd_eps = 1e-4
    bands_str = "delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50"
    bands = parse_band_spec(bands_str)
    stim_xlsx = "data/SEED/SEED_EEG/SEED_stimulation.xlsx"
    
    # Load DS
    print("Loading Dataset...")
    ds = SeedProcessedTrialDataset(args.processed_root, stim_xlsx)
    all_trials = sorted(list(ds), key=lambda x: x["trial_id_str"])
    print(f"Loaded {len(all_trials)} trials.")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Starting Cache Process...")
    print(f"Config: Win={win_sec}s, Est={est_mode}, Eps={spd_eps}")
    
    # Process one-by-one to save memory (though block is efficient too)
    # We can reuse extract_features_block passing [trial]
    
    for trial in tqdm(all_trials):
        tid = trial["trial_id_str"]
        out_path = os.path.join(args.out_dir, f"{tid}.pt")
        
        if os.path.exists(out_path):
            continue
            
        # Extract Covs (N, 62, 62)
        covs, y, _ = extract_features_block([trial], win_sec, hop_sec, est_mode, spd_eps, bands)
        
        if len(covs) == 0:
            print(f"WARNING: Trial {tid} produced 0 windows.")
            continue
            
        # Log Map (Tangent at Identity)
        # Note: We do NOT center here. Centering happens at runtime per seed split.
        covs_log = np.array([logm_spd(c, spd_eps) for c in covs], dtype=np.float32)
        
        # Vectorize (Upper Tri)
        feats = np.array([vec_utri(c) for c in covs_log], dtype=np.float32)
        
        # Save
        data = {
            "x": torch.from_numpy(feats),
            "y": int(y[0]), # Trials have single label
            "tid": tid
        }
        torch.save(data, out_path)
        
    print(f"Cache complete at {args.out_dir}")

if __name__ == "__main__":
    main()
