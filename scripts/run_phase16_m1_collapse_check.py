#!/usr/bin/env python
"""Phase 16 M1: Collapse Check.

Goal:
- Verify that Phase 16 Factorized Tokens (Raw Covs) can reproduce Phase 15 Broadband Tokens.
- Logic: Compute Mean(Covs) -> Logm -> Vec.
- Run N3c Protocol (80/20, 25 epochs, SE-Adapter).
- Target: Mean Acc ~0.6674.
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.run_phase14r_step6b1_rev2 import logm_spd, vec_utri
from scripts.run_phase15_n3c_report import EndToEndModelSE, collate_trials

def load_factorized_cache(cache_dir, spd_eps=1e-4):
    print("Loading Factorized Cache...", end=" ")
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    data = []
    
    # Pre-allocation optimizations could be done, but lists are fine for 675 trials.
    for f in tqdm(files, desc="Loading"):
        d = torch.load(f)
        # d has "x_covs": [T, 5, 62, 62] (Raw Covs)
        
        # COLLAPSE LOGIC
        # 1. Mean over bands -> [T, 62, 62] (Approx Broadband Cov)
        # Note: Scaling cancel out via StandardScaler later (Logm shift).
        cov_broad = d["x_covs"].mean(dim=1).numpy()
        
        # 2. Logm
        # Note: Phase 15 used spd_eps inside logm_spd logic or extract logic?
        # extract_features_block puts eps in cov estimation.
        # logm_spd adds eps to eigenvalues.
        # We reused _cov_empirical(..., eps). So covs in cache are already regularized?
        # Yes, run_phase16_m1_cache.py calls _cov_empirical(..., spd_eps).
        # So we don't need to add it again? 
        # But cov_broad is a Mean of regularized covs. It is SPD.
        # logm_spd adds eps again. This matches Phase 15 run_phase15_n2_cache logic.
        cov_log = np.array([logm_spd(c, spd_eps) for c in cov_broad], dtype=np.float32)
        
        # 3. Vec Utri
        x_vec = np.array([vec_utri(c) for c in cov_log], dtype=np.float32)
        
        trial = {
            "x": torch.from_numpy(x_vec), # [T, 1953]
            "y": d["y"],
            "tid": d["tid"]
        }
        data.append(trial)
        
    data.sort(key=lambda x: x["tid"])
    print(f"Loaded {len(data)} trials.")
    return data

def run_seed(seed: int, all_trials: list, cfg: dict, device: torch.device):
    # Identical to N3c Logic
    print(f"\n=== Seed {seed} ===")
    
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_trials))
    n_train = int(0.8 * len(all_trials))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    train_ds = [all_trials[i] for i in train_idx]
    test_ds = [all_trials[i] for i in test_idx]
    
    # Train Ref LR
    X_train_list = [t["x"].numpy() for t in train_ds]
    y_train_list = [np.full(len(t["x"]), t["y"]) for t in train_ds]
    X_train_flat = np.concatenate(X_train_list, axis=0)
    y_train_flat = np.concatenate(y_train_list, axis=0)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_flat)
    
    clf = LogisticRegression(solver="lbfgs", multi_class="multinomial", C=1.0, max_iter=1000, random_state=seed)
    clf.fit(X_train_s, y_train_flat)
    
    # Model
    model = EndToEndModelSE(
        input_dim=1953, 
        n_classes=3,
        scaler_mean=scaler.mean_, 
        scaler_scale=scaler.scale_,
        adapter_dim=cfg["adapter_dim"],
        eps=cfg["eps"],
        dropout=cfg["dropout"]
    ).to(device)
    
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy(clf.coef_))
        model.head.bias.copy_(torch.from_numpy(clf.intercept_))
        
    optimizer = optim.AdamW([
        {"params": model.adapter.parameters(), "lr": cfg["lr_adapter"]},
        {"params": model.head.parameters(), "lr": cfg["lr_linear"]},
        {"params": model.attn_score.parameters(), "lr": cfg["lr_agg"]},
        {"params": model.classifier.parameters(), "lr": cfg["lr_agg"]}
    ], weight_decay=cfg["weight_decay"])
    
    criterion = nn.CrossEntropyLoss()
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_trials)
    
    # Baseline Acc (80/20) - Used for comparison drift check
    bl_preds = {}
    for t in test_ds:
        x_s = scaler.transform(t["x"].numpy())
        l = clf.decision_function(x_s)
        wins = np.argmax(l, axis=1)
        if len(wins) > 0:
            from collections import Counter
            maj = Counter(wins).most_common(1)[0][0]
            bl_preds[t["tid"]] = int(maj)
        else:
            bl_preds[t["tid"]] = 0
    
    bl_acc = sum(1 for t in test_ds if bl_preds[t["tid"]] == t["y"]) / len(test_ds)
    
    # Train Loop
    for epoch in range(cfg["epochs"]):
        model.train()
        for x, y, mask, _ in train_dl:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            out, _, _, s = model(x, mask)
            loss_ce = criterion(out, y)
            gate_dev = (s - 1.0)
            loss_gate = (gate_dev ** 2).mean() * cfg["gate_reg"]
            loss = loss_ce + loss_gate
            loss.backward()
            optimizer.step()

    # Final Eval
    model.eval()
    test_dl = DataLoader(test_ds, batch_size=cfg["batch_size"], collate_fn=collate_trials)
    corr = 0
    tot = 0
    with torch.no_grad():
        for x, y, mask, _ in test_dl:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            out, _, _, _ = model(x, mask)
            preds = out.argmax(1).cpu().numpy()
            for i, p in enumerate(preds):
                if p == y[i].item():
                    corr += 1
            tot += len(y)
            
    test_acc = corr / tot
    print(f"    Result: Acc={test_acc:.4f} (Baseline={bl_acc:.4f})")
    
    return {
        "seed": seed,
        "n3c_acc": test_acc,
        "baseline_acc": bl_acc,
        "delta": test_acc - bl_acc
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data/_cache/phase16_m1_cache")
    parser.add_argument("--out-dir", default="promoted_results/phase16/m1_factorized")
    args = parser.parse_args()
    
    # N3c Config
    cfg = {
        "epochs": 25,
        "lr_adapter": 1e-4, "lr_agg": 5e-4, "lr_linear": 1e-5,
        "dropout": 0.5, "weight_decay": 1e-2,
        "adapter_dim": 32, "eps": 0.05, "gate_reg": 1e-2,
        "batch_size": 16
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    
    all_trials = load_factorized_cache(args.cache_dir)
    
    results = []
    seeds = range(10)
    for seed in seeds:
        res = run_seed(seed, all_trials, cfg, device)
        results.append(res)
        
    df = pd.DataFrame(results)
    df.to_csv(f"{args.out_dir}/collapse_check.csv", index=False)
    
    mean_acc = df["n3c_acc"].mean()
    print(f"\nMean Acc: {mean_acc:.4f}")
    if abs(mean_acc - 0.6674) <= 1e-4:
        print("PASS: Reproducibility Verified.")
    else:
        print(f"FAIL: Mean Acc {mean_acc:.4f} != 0.6674")

if __name__ == "__main__":
    main()
