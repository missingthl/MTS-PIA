#!/usr/bin/env python
"""Generate run_meta.json for N3c (Phase 15).

Recreates the training setup (Sklearn fit) to capture:
- Scaler statistics (mean, scale)
- Class mapping
- Hyperparameters
"""

import os
import sys
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.run_phase15_n2b_cache import load_all_n2_cache

def generate_meta():
    out_root = "promoted_results/phase15/n3c_report"
    n2_cache = "data/_cache/phase15_n2_cache"
    seeds = list(range(10))
    
    # Config from run_phase15_n3c_report.py
    cfg = {
        "epochs": 25,
        "lr_adapter": 1e-4,
        "lr_agg": 5e-4,
        "lr_linear": 1e-5,
        "dropout": 0.5,
        "weight_decay": 1e-2,
        "adapter_dim": 32,
        "eps": 0.05,
        "gate_reg": 1e-2,
        "batch_size": 16,
        "protocol": "80/20 Split (No Validation)"
    }
    
    print("Loading data...")
    all_trials = load_all_n2_cache(n2_cache)
    
    for seed in seeds:
        print(f"Processing Seed {seed}...")
        
        # Reproduce Split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_trials))
        n_train = int(0.8 * len(all_trials))
        train_idx = indices[:n_train]
        train_ds = [all_trials[i] for i in train_idx]
        
        # Fit Scaler
        X_train_list = [t["x"].numpy() for t in train_ds]
        y_train_list = [np.full(len(t["x"]), t["y"]) for t in train_ds]
        X_train_flat = np.concatenate(X_train_list, axis=0)
        y_train_flat = np.concatenate(y_train_list, axis=0)
        
        scaler = StandardScaler()
        scaler.fit(X_train_flat)
        
        # Fit LR (to check classes)
        clf = LogisticRegression(solver="lbfgs", multi_class="multinomial", C=1.0, max_iter=1000, random_state=seed)
        clf.fit(scaler.transform(X_train_flat), y_train_flat)
        
        # Meta Data
        meta = {
            "seed": seed,
            "classes": clf.classes_.tolist(),
            "n_classes": len(clf.classes_),
            "scaler_mean_sample": scaler.mean_[:5].tolist(), # Sample for verification
            "scaler_scale_sample": scaler.scale_[:5].tolist(),
            "scaler_mean_hash": hash(tuple(scaler.mean_)),
            "scaler_scale_hash": hash(tuple(scaler.scale_)),
            "config": cfg
        }
        
        seed_dir = f"{out_root}/seed{seed}"
        os.makedirs(seed_dir, exist_ok=True)
        
        with open(f"{seed_dir}/run_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
            
    print("Done.")

if __name__ == "__main__":
    generate_meta()
