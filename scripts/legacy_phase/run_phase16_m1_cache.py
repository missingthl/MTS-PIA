#!/usr/bin/env python
"""Phase 16 M1: Factorized Token Cache.

Goal:
- Pre-compute disjoint per-band covariances for all trials.
- Config: 1.0s window, Sample Covariance.
- Dimensions: [T, 5, 62, 62] (5 bands).
- Output: `data/_cache/phase16_m1_cache/` with `.pt` files per trial.
"""

import os
import sys
import numpy as np
import torch
import argparse
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets.seed_processed_trials import SeedProcessedTrialDataset
from manifold_raw.features import parse_band_spec, window_slices, bandpass
from scripts.legacy_phase.run_phase14r_step6b1_rev2 import _cov_empirical

def extract_factorized_covs(trial, win_sec, hop_sec, spd_eps, bands):
    # 1. Bandpass all bands
    b_data = {b.name: bandpass(trial["x_trial"], trial["sfreq"], b) for b in bands}
    band_names = sorted(b_data.keys()) # delta, theta, alpha, beta, gamma
    
    first_band = b_data[band_names[0]]
    n_samples = first_band.shape[1]
    fs = trial["sfreq"]
    w_list = window_slices(n_samples, fs, win_sec, hop_sec)
    
    covs_per_window = []
    
    for s, e in w_list:
        band_covs = []
        for b_name in band_names:
            chunk = b_data[b_name][:, s:e].astype(np.float32)
            # Time-centering & standardization (reproduce Phase 15 logic)
            m = chunk.mean()
            sd = chunk.std() + 1e-6
            chunk = (chunk - m) / sd
            
            # Compute Covariance for THIS band
            # shape (62, T)
            # _cov_empirical expects (62, T)
            cov = _cov_empirical(chunk, spd_eps)
            band_covs.append(cov)
            
        covs_per_window.append(np.stack(band_covs)) # [5, 62, 62]
        
    return np.array(covs_per_window, dtype=np.float32) # [T, 5, 62, 62]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/_cache/phase16_m1_cache")
    parser.add_argument("--processed-root", default="data/SEED/SEED_EEG/Preprocessed_EEG")
    args = parser.parse_args()
    
    # Phase 15 Params
    win_sec = 1.0
    hop_sec = 1.0
    spd_eps = 1e-4
    bands_str = "delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50"
    bands = parse_band_spec(bands_str)
    stim_xlsx = "data/SEED/SEED_EEG/SEED_stimulation.xlsx"
    
    print("Loading Dataset...")
    ds = SeedProcessedTrialDataset(args.processed_root, stim_xlsx)
    all_trials = sorted(list(ds), key=lambda x: x["trial_id_str"])
    print(f"Loaded {len(all_trials)} trials.")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Starting Factorized Cache Process...")
    
    for trial in tqdm(all_trials):
        tid = trial["trial_id_str"]
        out_path = os.path.join(args.out_dir, f"{tid}.pt")
        
        if os.path.exists(out_path):
            continue
            
        # extract [T, 5, 62, 62]
        covs = extract_factorized_covs(trial, win_sec, hop_sec, spd_eps, bands)
        
        if len(covs) == 0:
            continue
            
        data = {
            "x_covs": torch.from_numpy(covs), # [T, 5, 62, 62]
            "y": int(trial["label"]),
            "tid": tid,
            "band_names": [b.name for b in bands]
        }
        torch.save(data, out_path)
        
    print(f"Cache complete at {args.out_dir}")

if __name__ == "__main__":
    main()
