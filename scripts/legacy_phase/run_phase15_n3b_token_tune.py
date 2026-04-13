#!/usr/bin/env python
"""Phase 15 N3b: Token Tuning with Safe Residual Adapter.

Goal:
- End-to-End tuning of Tangent Vectors.
- Uses Zero-Init Residual Adapter to ensure Epoch 0 Equivalence.
- Frozen Scaler + Warm-Started Linear + Mean-Init Aggregator.
- Metric: Trial Accuracy (vs Locked Baseline).
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
from scripts.legacy_phase.run_phase15_n2b_cache import load_all_n2_cache

# --- Data ---

def collate_trials(batch):
    xs = [item["x"] for item in batch]
    ys = torch.tensor([item["y"] for item in batch], dtype=torch.long)
    tids = [item["tid"] for item in batch]
    lengths = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    x_padded = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0.0)
    mask = torch.arange(x_padded.size(1))[None, :] < lengths[:, None]
    return x_padded, ys, mask, tids

# --- Modules ---

class SafeScaler(nn.Module):
    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        super().__init__()
        # Register as buffers (not parameters, so no gradients, state_dict saved)
        self.register_buffer("mean", torch.from_numpy(mean).float())
        self.register_buffer("scale", torch.from_numpy(scale).float())
        
    def forward(self, x):
        # x: (B, T, D)
        return (x - self.mean) / (self.scale + 1e-9)

class ZeroInitAdapter(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.act = nn.GELU()
        
        # Norm? User says "No LayerNorm output... If used, inside bottleneck"
        self.norm = nn.LayerNorm(bottleneck_dim) 
        
        self.up = nn.Linear(bottleneck_dim, input_dim)
        # Zero Init UP to ensure identity at start
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, T, D)
        residual = x
        out = self.down(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.up(out) # Initially 0
        return residual + out

class EndToEndModel(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, scaler_mean, scaler_scale, 
                 adapter_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        
        # 1. Adapter (Residual)
        self.adapter = ZeroInitAdapter(input_dim, adapter_dim)
        
        # 2. Frozen Scaler
        self.scaler = SafeScaler(scaler_mean, scaler_scale)
        
        # 3. Linear Head (Warm-Start target)
        self.head = nn.Linear(input_dim, n_classes)
        
        # 4. Aggregator (Mean-Init)
        self.attn_score = nn.Linear(n_classes, 1)
        nn.init.zeros_(self.attn_score.weight)
        nn.init.zeros_(self.attn_score.bias)
        self.classifier = nn.Linear(n_classes, n_classes)
        nn.init.eye_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # x: (B, T, D) Raw Tangents
        
        # Adapter
        x_adapt = self.adapter(x)
        
        # Scaler
        x_scaled = self.scaler(x_adapt)
        
        # Linear Head (Applied per token)
        b, t, d = x_scaled.shape
        logits_seq = self.head(x_scaled) # (B, T, C)
        
        # Aggregation
        scores = self.attn_score(logits_seq).squeeze(-1) # (B, T)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1) # (B, T)
        
        ctx = (weights.unsqueeze(-1) * logits_seq).sum(dim=1) # (B, C)
        
        ctx = self.dropout(ctx)
        out = self.classifier(ctx)
        
        return out, weights, logits_seq, x_adapt

# --- Safety & Training ---

def check_equivalence(model, clf, val_trials, device):
    print("    Safety Check: Epoch 0 Equivalence...")
    model.eval()
    
    # 1. Standardize Sklearn Input
    # Get scaler stats from model (should match what generated clf)
    # But wait, clf was trained on scaled data.
    # We verify that model.scaler matches the scaler used for clf (passed in init).
    # Then we check logits.
    
    t = val_trials[0]
    x_raw = t["x"].numpy() # (T, D)
    x_torch = t["x"].unsqueeze(0).to(device) # (1, T, D)
    mask = torch.ones(1, x_raw.shape[0], dtype=torch.bool).to(device)
    
    with torch.no_grad():
        out, weights, logits_seq, x_adapt = model(x_torch, mask)
        
    # Check A: Adapter is Identity
    diff_adapt = np.abs(x_adapt.cpu().numpy() - x_torch.cpu().numpy()).max()
    print(f"      Adapter Diff: {diff_adapt:.2e}")
    if diff_adapt > 1e-6:
        print("      FATAL: Adapter not identity!")
        return False
        
    # Check B: Logits match Sklearn
    # Sklearn prediction needs scaling
    scaler_mean = model.scaler.mean.cpu().numpy()
    scaler_scale = model.scaler.scale.cpu().numpy()
    x_scaled_np = (x_raw - scaler_mean) / (scaler_scale + 1e-9)
    sk_logits = clf.decision_function(x_scaled_np)
    
    th_logits = logits_seq.squeeze(0).cpu().numpy()
    
    diff_logits = np.abs(sk_logits - th_logits).max()
    print(f"      Logits Diff: {diff_logits:.2e}")
    if diff_logits > 1e-4:
        print(f"      FATAL: Logits mismatch! Max diff {diff_logits}")
        return False
        
    # Check C: Aggregation is Mean
    # Mean Logits manually
    sk_mean = sk_logits.mean(axis=0)
    th_out = out.squeeze(0).cpu().numpy()
    # Note: Torch Attn might be slightly different from arithmetic mean due to softmax logic
    # But with Attn Weights~Uniform, it should be close.
    # init.zeros -> weights = 1/T exactly.
    diff_agg = np.abs(sk_mean - th_out).max()
    print(f"      Agg vs Mean: {diff_agg:.2e}") 
    if diff_agg > 1e-4:
         print("      FATAL: Aggregator not mean!")
         return False
         
    print("      PASS: Safety Check Verified.")
    return True

def run_seed(seed: int, all_trials: List, cfg: Dict, device: torch.device):
    print(f"\n=== Seed {seed} ===")
    
    # Split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_trials))
    n_train_full = int(0.8 * len(all_trials))
    train_full = indices[:n_train_full]
    test_idx = indices[n_train_full:]
    n_train = int(0.8 * len(train_full))
    train_idx = train_full[:n_train]
    val_idx = train_full[n_train:]
    
    train_ds = [all_trials[i] for i in train_idx]
    val_ds = [all_trials[i] for i in val_idx]
    test_ds = [all_trials[i] for i in test_idx]
    
    # 1. Train Reference Sklearn LR
    print("    Training Reference LR (LBFGS)...")
    # Prepare Data
    X_train_list = [t["x"].numpy() for t in train_ds]
    y_train_list = [np.full(len(t["x"]), t["y"]) for t in train_ds]
    X_train_flat = np.concatenate(X_train_list, axis=0)
    y_train_flat = np.concatenate(y_train_list, axis=0)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_flat)
    
    clf = LogisticRegression(solver="lbfgs", multi_class="multinomial", C=1.0, max_iter=1000, random_state=seed)
    clf.fit(X_train_s, y_train_flat)
    
    # 2. Build Model
    model = EndToEndModel(
        input_dim=1953, 
        n_classes=3,
        scaler_mean=scaler.mean_, 
        scaler_scale=scaler.scale_,
        adapter_dim=cfg["adapter_dim"],
        dropout=cfg["dropout"]
    ).to(device)
    
    # Warm Start Linear
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy(clf.coef_))
        model.head.bias.copy_(torch.from_numpy(clf.intercept_))
        
    init_head_w = model.head.weight.clone().detach()
    init_head_b = model.head.bias.clone().detach()
        
    # Safety Check
    if not check_equivalence(model, clf, val_ds, device):
        return None
        
    # 3. Training
    # Differential LR
    optimizer = optim.AdamW([
        {"params": model.adapter.parameters(), "lr": cfg["lr_adapter"]},
        {"params": model.head.parameters(), "lr": cfg["lr_linear"]},
        {"params": model.attn_score.parameters(), "lr": cfg["lr_agg"]},
        {"params": model.classifier.parameters(), "lr": cfg["lr_agg"]}
    ], weight_decay=cfg["weight_decay"])
    
    criterion = nn.CrossEntropyLoss()
    
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_trials)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], collate_fn=collate_trials)
    
    # Baseline Prediction (for Flip Calc)
    bl_preds = {}
    for t in test_ds:
        x_s = scaler.transform(t["x"].numpy())
        l = clf.decision_function(x_s)
        # Majority
        wins = np.argmax(l, axis=1)
        if len(wins) > 0:
            from collections import Counter
            maj = Counter(wins).most_common(1)[0][0]
            bl_preds[t["tid"]] = int(maj)
        else:
            bl_preds[t["tid"]] = 0
            
    bl_acc = sum(1 for t in test_ds if bl_preds[t["tid"]] == t["y"]) / len(test_ds)
    print(f"    Baseline Acc: {bl_acc:.4f}")
    
    best_val_acc = 0.0
    best_state = None
    patience = 0
    best_epoch = 0
    
    for epoch in range(cfg["epochs"]):
        model.train()
        for x, y, mask, _ in train_dl:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            out, _, _, _ = model(x, mask)
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
                out, _, _, _ = model(x, mask)
                corr += (out.argmax(1) == y).sum().item()
                tot += y.size(0)
        val_acc = corr / tot
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= cfg["patience"]:
                break
                
    # Final Eval
    model.load_state_dict(best_state)
    model.eval()
    
    test_dl = DataLoader(test_ds, batch_size=cfg["batch_size"], collate_fn=collate_trials)
    corr = 0
    tot = 0
    flips = 0
    with torch.no_grad():
        for x, y, mask, tids in test_dl:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            out, _, _, _ = model(x, mask)
            preds = out.argmax(1).cpu().numpy()
            
            for i, p in enumerate(preds):
                if p == y[i].item():
                    corr += 1
                if p != bl_preds[tids[i]]:
                    flips += 1
                tot += 1
                
    test_acc = corr / tot
    flip_rate = flips / tot
    
    # Drift Check
    curr_head_w = model.head.weight.detach()
    drift_nm = torch.norm(curr_head_w - init_head_w).item()
    
    print(f"    Result: Acc={test_acc:.4f} (Delta={test_acc - bl_acc:.4f}) Flip={flip_rate:.2%} Drift={drift_nm:.4f}")
    
    return {
        "seed": seed,
        "n3b_acc": test_acc,
        "baseline_aligned": bl_acc,
        "delta": test_acc - bl_acc,
        "flip_rate": flip_rate,
        "drift": drift_nm,
        "best_epoch": best_epoch
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n2-cache", default="data/_cache/phase15_n2_cache")
    parser.add_argument("--out-root", default="promoted_results/phase15/n3b_token_tune")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    
    # Hyperparams
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr-adapter", type=float, default=5e-4)
    parser.add_argument("--lr-agg", type=float, default=5e-4) # Slightly reduced from 1e-3
    parser.add_argument("--lr-linear", type=float, default=1e-5) # Tiny
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--adapter-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    
    args = parser.parse_args()
    cfg = vars(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(f"{args.out_root}/seed1", exist_ok=True)
    
    all_trials = load_all_n2_cache(args.n2_cache)
    
    results = []
    for seed in args.seeds:
        res = run_seed(seed, all_trials, cfg, device)
        if res:
            results.append(res)
            
    df = pd.DataFrame(results)
    df.to_csv(f"{args.out_root}/summary.csv", index=False)
    print(f"Saved to {args.out_root}/summary.csv")

if __name__ == "__main__":
    main()
