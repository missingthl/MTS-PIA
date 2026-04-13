#!/usr/bin/env python
"""Phase 15 N2b: Logits Cache.

Goal:
- Train Baseline Window Classifier (Sklearn LR LBFGS) per seed.
- Predict Logits (Decision Function) for ALL trials.
- Cache logits [T, 3] per trial to `data/_cache/phase15_n2b_cache/seed{N}.pt`.

This ensures N2b starts from the exact same information as the Baseline.
"""

import os
import sys
import glob
import torch
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# --- Re-use Cache from N2 ---
# N2 cache contains "x" (Tangent Vector), "y", "tid".
# This saves us from re-doing covariance/logm.
# However, Standardization is split-dependent, so we do it here.

def load_all_n2_cache(cache_dir):
    print("Loading N2 Cache...", end=" ")
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    data = []
    for f in tqdm(files, desc="Loading"):
        # Torch load might be slow for 675 files, but okay.
        d = torch.load(f)
        data.append(d)
    # Sort by TID
    data.sort(key=lambda x: x["tid"])
    print(f"Loaded {len(data)} trials.")
    return data

def process_seed(seed: int, all_data: list, out_dir: str):
    print(f"\n=== Seed {seed} ===")
    
    # 1. Split (Step 6B2 Logic)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_data))
    n_train = int(0.8 * len(all_data))
    
    train_idx = indices[:n_train] # Train + Val (for N2b split later, here we train LR on full Train)
    test_idx = indices[n_train:]
    
    train_trials = [all_data[i] for i in train_idx]
    
    # 2. Prepare Train X/y
    # Flatten windows
    # x is (T, D)
    print("    Preparing Train Data...")
    X_train_list = [t["x"].numpy() for t in train_trials]
    y_train_list = [np.full(len(t["x"]), t["y"]) for t in train_trials]
    
    X_train_flat = np.concatenate(X_train_list, axis=0)
    y_train_flat = np.concatenate(y_train_list, axis=0)
    
    # 3. Standardize (Fit on Train)
    print("    Standardizing...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_flat)
    
    # 4. Train LR (LBFGS, C=1.0)
    print("    Training LR (LBFGS)...")
    clf = LogisticRegression(solver="lbfgs", multi_class="multinomial", C=1.0, 
                             max_iter=1000, random_state=seed, n_jobs=-1)
    clf.fit(X_train_s, y_train_flat)
    
    acc = clf.score(X_train_s, y_train_flat)
    print(f"    Train Window Acc: {acc:.4f}")
    
    # 5. Predict Logits for ALL trials
    # We need to standardize per trial using the Train scaler
    print("    Predicting Logits...")
    
    seed_cache = {}
    
    for trial in tqdm(all_data, desc=f"Seed {seed} Inference"):
        x = trial["x"].numpy()
        x_s = scaler.transform(x)
        
        # decision_function gives (T, 3) logits
        logits = clf.decision_function(x_s)
        
        seed_cache[trial["tid"]] = {
            "logits": torch.from_numpy(logits).float(), # [T, 3]
            "y": trial["y"],
            "tid": trial["tid"]
        }
        
    # Save
    out_path = os.path.join(out_dir, f"seed{seed}.pt")
    torch.save(seed_cache, out_path)
    print(f"    Saved to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n2-cache", default="data/_cache/phase15_n2_cache")
    parser.add_argument("--out-dir", default="data/_cache/phase15_n2b_cache")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    all_data = load_all_n2_cache(args.n2_cache)
    
    for seed in args.seeds:
        process_seed(seed, all_data, args.out_dir)

if __name__ == "__main__":
    main()
