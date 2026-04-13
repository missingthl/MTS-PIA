#!/usr/bin/env python
"""Phase 16 M1: Factorized Probe (Band-wise).

Goal:
- Verify signal content in each band (Delta, Theta, Alpha, Beta, Gamma) individually.
- Train Sklearn LR on `Vec(Logm(Cov_b))`.
- Metrics: Per-band Acc, Mean-Logit Ensemble Acc.
"""

import os
import sys
import glob
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.legacy_phase.run_phase14r_step6b1_rev2 import logm_spd, vec_utri

def load_band_features(cache_dir, spd_eps=1e-4):
    print("Loading Cache for Probe...", end=" ")
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    data = []
    for f in tqdm(files, desc="Loading"):
        d = torch.load(f)
        # d["x_covs"] is [T, 5, 62, 62]
        covs = d["x_covs"].numpy()
        
        # Build list of 5 feature sets
        band_feats = []
        for b in range(5):
            # Logm -> Vec
            c_log = np.array([logm_spd(c, spd_eps) for c in covs[:, b]], dtype=np.float32)
            bst = np.array([vec_utri(c) for c in c_log], dtype=np.float32)
            band_feats.append(bst)
            
        trial = {
            "bands_x": band_feats, # List of 5 [T, 1953] arrays
            "y": d["y"],
            "tid": d["tid"]
        }
        data.append(trial)
    data.sort(key=lambda x: x["tid"])
    print("Done.")
    return data

def run_seed_probe(seed, all_trials):
    print(f"Seed {seed} Probe...")
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_trials))
    n_train = int(0.8 * len(all_trials))
    train_trials = [all_trials[i] for i in indices[:n_train]]
    test_trials = [all_trials[i] for i in indices[n_train:]]
    
    y_train_flat = []
    for t in train_trials:
        y_train_flat.extend([t["y"]] * t["bands_x"][0].shape[0])
    y_train_flat = np.array(y_train_flat)
        
    band_names = ["delta", "theta", "alpha", "beta", "gamma"]
    res = {}
    
    clfs = []
    scalers = []
    
    # Train Per-Band
    for b in range(5):
        # Flatten
        X_train_list = [t["bands_x"][b] for t in train_trials]
        X_train_flat = np.concatenate(X_train_list, axis=0)
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_flat)
        
        clf = LogisticRegression(solver="lbfgs", multi_class="multinomial", C=1.0, max_iter=500, random_state=seed, n_jobs=4)
        clf.fit(X_train_s, y_train_flat)
        
        clfs.append(clf)
        scalers.append(scaler)
        
        # Evaluate
        X_test_list = [t["bands_x"][b] for t in test_trials]
        preds_all = []
        for i, t in enumerate(test_trials):
            x_s = scaler.transform(t["bands_x"][b])
            l = clf.predict(x_s)
            # Majority vote
            if len(l) > 0:
                maj = Counter(l).most_common(1)[0][0]
                preds_all.append(int(maj))
            else:
                preds_all.append(0)
                
        y_true = [t["y"] for t in test_trials]
        acc = np.mean(np.array(preds_all) == np.array(y_true))
        res[f"acc_{band_names[b]}"] = acc
        
    # Ensemble (Mean Logit)
    preds_ens = []
    y_true = [t["y"] for t in test_trials]
    
    for t in test_trials:
        # Sum logits across bands (and time)
        # Or mean logits across bands, then majority? 
        # Better: Mean logits across bands per window, then Mean over time? (N2b)
        
        logits_sum = 0
        for b in range(5):
             x_s = scalers[b].transform(t["bands_x"][b])
             l = clfs[b].decision_function(x_s) # [T, 3]
             logits_sum += l
             
        # Mean across bands
        logits_mean_band = logits_sum / 5.0
        
        # Mean across time (Trial Agg)
        logits_trial = logits_mean_band.mean(axis=0) # [3]
        pred = np.argmax(logits_trial)
        preds_ens.append(pred)
        
    res["acc_ensemble"] = np.mean(np.array(preds_ens) == np.array(y_true))
    res["seed"] = seed
    
    print(f"  {res}")
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data/_cache/phase16_m1_cache")
    parser.add_argument("--out-dir", default="promoted_results/phase16/m1_factorized")
    args = parser.parse_args()
    
    all_trials = load_band_features(args.cache_dir)
    results = []
    for seed in range(10):
        results.append(run_seed_probe(seed, all_trials))
        
    df = pd.DataFrame(results)
    df.to_csv(f"{args.out_dir}/probe_baselines.csv", index=False)
    print("Probes Saved.")

if __name__ == "__main__":
    main()
