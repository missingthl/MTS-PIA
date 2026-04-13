#!/usr/bin/env python
"""Phase 15 N3a: Warm-Start Linear Fine-tuning + Safe Aggregator.

Goal:
- Warm-start Torch Linear Head with Sklearn LR (LBFGS) weights.
- Verify exact equivalence (logits matching).
- Fine-tune End-to-End (Low LR for Head, Higher for Aggregator).
- Improve performance while maintaining Strong Seed Protection.

Contract:
- 1.0s / Sample / LogCenter / Tangent (Phase 14R Step 6B2).
- Mean-Init Aggregator.
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
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Re-use N2 Cache (Raw Tangents)
from scripts.legacy_phase.run_phase15_n2b_cache import load_all_n2_cache

# --- Data & Collate ---

def collate_trials(batch):
    # Batch is list of dicts {"x": tensor(T, D), "y": int, "tid": str}
    xs = [item["x"] for item in batch]
    ys = torch.tensor([item["y"] for item in batch], dtype=torch.long)
    tids = [item["tid"] for item in batch]
    
    lengths = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    x_padded = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0.0)
    
    # Mask (True = Valid)
    mask = torch.arange(x_padded.size(1))[None, :] < lengths[:, None]
    
    return x_padded, ys, mask, tids

# --- Model ---

class WarmStartModel(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        # 1. Linear Head (To be warm-started)
        self.head = nn.Linear(input_dim, n_classes)
        
        # 2. Aggregator (Logits -> Attn -> Logits)
        self.attn_score = nn.Linear(n_classes, 1)
        # Init to 0 for Uniform Attn (Mean Pool behavior)
        nn.init.zeros_(self.attn_score.weight)
        nn.init.zeros_(self.attn_score.bias)
        
        self.classifier = nn.Linear(n_classes, n_classes)
        # Init to Identity (Pass-through)
        nn.init.eye_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # x: (B, T, D)
        
        # 1. Window Logits
        # Collapsed dimensions for Linear: (B*T, D)
        b, t, d = x.shape
        x_flat = x.view(-1, d)
        logits_flat = self.head(x_flat) # (B*T, C)
        logits = logits_flat.view(b, t, -1) # (B, T, C)
        
        # 2. Aggregation
        # Scores
        scores = self.attn_score(logits).squeeze(-1) # (B, T)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1) # (B, T)
        
        # Pooling
        # (B, T, 1) * (B, T, C) -> Sum -> (B, C)
        ctx = (weights.unsqueeze(-1) * logits).sum(dim=1)
        
        # 3. Final Projection (Identity init)
        ctx = self.dropout(ctx)
        out = self.classifier(ctx)
        
        return out, weights, logits

# --- Equivalence Check ---

def check_equivalence(model, clf, scaler, val_trials, device):
    """Verify Torch Model matches Sklearn LR at Init."""
    print("    Verifying Equivalence (A1-A4)...")
    
    model.eval()
    
    # 1. Check Weights
    # Sklearn: (n_classes, n_features)
    # Torch: (out_features, in_features)
    sk_W = clf.coef_
    sk_b = clf.intercept_
    th_W = model.head.weight.detach().cpu().numpy()
    th_b = model.head.bias.detach().cpu().numpy()
    
    diff_W = np.abs(sk_W - th_W).max()
    diff_b = np.abs(sk_b - th_b).max()
    
    print(f"      Weight Diff: {diff_W:.2e}")
    print(f"      Bias Diff: {diff_b:.2e}")
    
    if diff_W > 1e-5 or diff_b > 1e-5:
        print("      FATAL: Weights mismatch!")
        # return False
    
    # 2. Check Logits on batch
    # Prepare batch
    sample_trial = val_trials[0]
    x_raw = sample_trial["x"].numpy() # (T, D)
    x_scaled = scaler.transform(x_raw) # Sklearn scale
    
    # Sklearn Logits
    sk_logits = clf.decision_function(x_scaled)
    
    # Torch Logits (Manual inputs)
    x_t = torch.from_numpy(x_scaled).float().to(device).unsqueeze(0) # (1, T, D)
    mask = torch.ones(1, x_t.size(1), dtype=torch.bool).to(device)
    
    with torch.no_grad():
        _, _, th_logits_seq = model(x_t, mask)
        th_logits = th_logits_seq.squeeze(0).cpu().numpy()
        
    logit_diff = np.abs(sk_logits - th_logits).max()
    print(f"      Logit Diff: {logit_diff:.2e}")
    
    if logit_diff > 1e-4:
        print("      FATAL: Logit mismatch!")
        print(f"      Sklearn Sample: {sk_logits[0]}")
        print(f"      Torch Sample: {th_logits[0]}")
        return False
        
    print("      PASS: Equivalence verified.")
    return True

# --- Training Loop ---

def run_seed(seed: int, all_trials: List[Dict], cfg: Dict, device: torch.device):
    print(f"\n=== Seed {seed} ===")
    
    # 1. Split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_trials))
    n_train_full = int(0.8 * len(all_trials))
    train_full_idx = indices[:n_train_full]
    test_idx = indices[n_train_full:]
    
    n_train = int(0.8 * len(train_full_idx))
    train_idx = train_full_idx[:n_train]
    val_idx = train_full_idx[n_train:]
    
    train_trials = [all_trials[i] for i in train_idx]
    val_trials = [all_trials[i] for i in val_idx]
    test_trials = [all_trials[i] for i in test_idx]
    
    # 2. Prepare Sklearn Training Data (Flattened)
    print("    Training Warm-Start LR...")
    X_train_list = [t["x"].numpy() for t in train_trials]
    y_train_list = [np.full(len(t["x"]), t["y"]) for t in train_trials]
    X_train_flat = np.concatenate(X_train_list, axis=0)
    y_train_flat = np.concatenate(y_train_list, axis=0)
    
    # 3. Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_flat)
    
    # 4. Train LR
    clf = LogisticRegression(solver="lbfgs", multi_class="multinomial", C=1.0, max_iter=1000, random_state=seed)
    clf.fit(X_train_s, y_train_flat)
    
    # 5. Apply Scaler to Datasets (In-place modification of x tensor? No, better copy or transform on fly)
    # The collate fn expects 'x' tensor. Let's pre-transform dataset.
    def transform_ds(ds):
        new_ds = []
        for t in ds:
            x_np = t["x"].numpy()
            x_s = scaler.transform(x_np)
            new_ds.append({
                "x": torch.from_numpy(x_s).float(),
                "y": t["y"],
                "tid": t["tid"]
            })
        return new_ds

    train_ds = transform_ds(train_trials)
    val_ds = transform_ds(val_trials)
    test_ds = transform_ds(test_trials)
    
    # 6. Init Torch Model
    model = WarmStartModel(input_dim=1953, n_classes=3, dropout=cfg["dropout"]).to(device)
    
    # Copy Weights
    # Check classes order
    # print(f"    Classes: {clf.classes_}") 
    # Assumes classes are [0, 1, 2]. If not, mapping needed.
    # SeedProcessedTrialDataset guarantees 0,1,2.
    
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy(clf.coef_))
        model.head.bias.copy_(torch.from_numpy(clf.intercept_))
        
    # Verify
    if not check_equivalence(model, clf, scaler, val_trials, device):
        print("ABORTING SEED: Eq Check Failed")
        return None
        
    # 7. Training
    # Stage 1: Freeze Head, Train Aggregator
    # Actually User requested "Two-stage fine-tuning".
    # Or Joint with differential LR.
    # "B2) Joint fine-tuning... Differential LR"
    
    # Optimizer Groups
    params = [
        {"params": model.head.parameters(), "lr": cfg["lr_head"]},
        {"params": model.attn_score.parameters(), "lr": cfg["lr_agg"]},
        {"params": model.classifier.parameters(), "lr": cfg["lr_agg"]}
    ]
    optimizer = optim.AdamW(params, weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_trials)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], collate_fn=collate_trials)
    
    best_val_acc = 0.0
    best_state = None
    patience = 0
    
    # Baseline Acc for Flip Calculation
    # We use Sklearn window predictions aggregated by Majority
    bl_preds = {}
    for t in test_trials:
        x_s = scaler.transform(t["x"].numpy())
        logits = clf.decision_function(x_s)
        win_preds = np.argmax(logits, axis=1)
        if len(win_preds) > 0:
            from collections import Counter
            maj = Counter(win_preds).most_common(1)[0][0]
        else:
            maj = 0
        bl_preds[t["tid"]] = int(maj)
        
    bl_acc = sum([1 for t in test_trials if bl_preds[t["tid"]] == t["y"]]) / len(test_trials)
    print(f"    Baseline Acc: {bl_acc:.4f}")

    for epoch in range(cfg["epochs"]):
        model.train()
        for x, y, mask, _ in train_dl:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            out, _, _ = model(x, mask)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
        # Val
        model.eval()
        corr = 0
        tot = 0
        with torch.no_grad():
            for x, y, mask, _ in val_dl:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                out, _, _ = model(x, mask)
                corr += (out.argmax(1) == y).sum().item()
                tot += y.size(0)
        val_acc = corr / tot
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= cfg["patience"]:
                break
                
    # Test
    model.load_state_dict(best_state)
    model.eval()
    
    test_dl = DataLoader(test_ds, batch_size=cfg["batch_size"], collate_fn=collate_trials)
    
    n3a_corr = 0
    flips = 0
    t_tot = 0
    
    with torch.no_grad():
        for x, y, mask, tids in test_dl:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            out, _, _ = model(x, mask)
            preds = out.argmax(1).cpu().numpy()
            
            for i, p in enumerate(preds):
                tid = tids[i]
                y_true = y[i].item()
                if p == y_true:
                    n3a_corr += 1
                
                bl_p = bl_preds[tid]
                if p != bl_p:
                    flips += 1
                    
                t_tot += 1
                
    n3a_acc = n3a_corr / t_tot
    flip_rate = flips / t_tot
    
    print(f"    Result: TestAcc={n3a_acc:.4f} (Delta={n3a_acc - bl_acc:.4f}) FlipRate={flip_rate:.2%}")
    
    return {
        "seed": seed,
        "n3a_acc": n3a_acc,
        "baseline_aligned": bl_acc,
        "delta": n3a_acc - bl_acc,
        "flip_rate": flip_rate
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n2-cache", default="data/_cache/phase15_n2_cache")
    parser.add_argument("--out-root", default="promoted_results/phase15/n3a_warmstart")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    
    # Hyperparams
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr-head", type=float, default=1e-5) # Vibration
    parser.add_argument("--lr-agg", type=float, default=1e-4) # Learning (Conservative)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    
    args = parser.parse_args()
    cfg = vars(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(f"{args.out_root}/seed1", exist_ok=True)
    
    # Load Data
    all_data = load_all_n2_cache(args.n2_cache)
    
    results = []
    for seed in args.seeds:
        res = run_seed(seed, all_data, cfg, device)
        if res:
            results.append(res)
            
    df = pd.DataFrame(results)
    df.to_csv(f"{args.out_root}/seed1/summary_by_seed.csv", index=False)
    print(f"Saved to {args.out_root}/seed1/summary_by_seed.csv")

if __name__ == "__main__":
    main()
