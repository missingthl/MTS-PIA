#!/usr/bin/env python
"""Phase 15 N3c: Report-Track Fair Compare (80/20).

Goal:
- Evaluate SE-Adapter (N3b-rev1) on FULL 80/20 split.
- No Validation set (mimic Baseline protocol).
- Fixed Epochs (25).
- Metric: Last Epoch Test Accuracy.
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.run_phase15_n2b_cache import load_all_n2_cache

# --- Data ---

def collate_trials(batch):
    xs = [item["x"] for item in batch]
    ys = torch.tensor([item["y"] for item in batch], dtype=torch.long)
    tids = [item["tid"] for item in batch]
    lengths = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    x_padded = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0.0)
    mask = torch.arange(x_padded.size(1))[None, :] < lengths[:, None]
    return x_padded, ys, mask, tids

# --- Modules (Same as N3b-rev1) ---

class SafeScaler(nn.Module):
    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(mean).float())
        self.register_buffer("scale", torch.from_numpy(scale).float())
        
    def forward(self, x):
        return (x - self.mean) / (self.scale + 1e-9)

class SEGatedAdapter(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int = 32, eps: float = 0.05, dropout: float = 0.1):
        super().__init__()
        self.eps = eps
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, input_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        mask_f = mask.float().unsqueeze(-1)
        x_masked = x * mask_f
        nums = x_masked.sum(dim=1)
        denoms = mask_f.sum(dim=1).clamp(min=1e-9)
        z = nums / denoms
        g = self.down(z)
        g = self.act(g)
        g = self.dropout(g)
        g = self.up(g)
        s = 1.0 + self.eps * torch.tanh(g)
        x_out = x * s.unsqueeze(1)
        return x_out, s

class EndToEndModelSE(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, scaler_mean, scaler_scale, 
                 adapter_dim: int = 32, eps: float = 0.05, dropout: float = 0.5):
        super().__init__()
        self.adapter = SEGatedAdapter(input_dim, adapter_dim, eps, dropout=0.1)
        self.scaler = SafeScaler(scaler_mean, scaler_scale)
        self.head = nn.Linear(input_dim, n_classes)
        self.attn_score = nn.Linear(n_classes, 1)
        nn.init.zeros_(self.attn_score.weight)
        nn.init.zeros_(self.attn_score.bias)
        self.classifier = nn.Linear(n_classes, n_classes)
        nn.init.eye_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x_adapt, s = self.adapter(x, mask)
        x_scaled = self.scaler(x_adapt)
        logits_seq = self.head(x_scaled)
        scores = self.attn_score(logits_seq).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        ctx = (weights.unsqueeze(-1) * logits_seq).sum(dim=1)
        ctx = self.dropout(ctx)
        out = self.classifier(ctx)
        return out, weights, logits_seq, s

# --- Training ---

def run_seed(seed: int, all_trials: List, cfg: Dict, device: torch.device):
    print(f"\n=== Seed {seed} ===")
    
    # 80/20 Split (No Val)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_trials))
    n_train = int(0.8 * len(all_trials))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    train_ds = [all_trials[i] for i in train_idx]
    test_ds = [all_trials[i] for i in test_idx]
    
    # Train Ref LR (on 80% train)
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
        
    init_head_w = model.head.weight.clone().detach()
    
    # Optimizer
    optimizer = optim.AdamW([
        {"params": model.adapter.parameters(), "lr": cfg["lr_adapter"]},
        {"params": model.head.parameters(), "lr": cfg["lr_linear"]},
        {"params": model.attn_score.parameters(), "lr": cfg["lr_agg"]},
        {"params": model.classifier.parameters(), "lr": cfg["lr_agg"]}
    ], weight_decay=cfg["weight_decay"])
    
    criterion = nn.CrossEntropyLoss()
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_trials)
    
    # Baseline Acc (80/20)
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
    print(f"    Baseline Acc (80/20): {bl_acc:.4f}")
    
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

    # Final Eval (Last Epoch)
    model.eval()
    test_dl = DataLoader(test_ds, batch_size=cfg["batch_size"], collate_fn=collate_trials)
    corr = 0
    tot = 0
    flips = 0
    gate_devs_test = []
    
    with torch.no_grad():
        for x, y, mask, tids in test_dl:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            out, _, _, s = model(x, mask)
            preds = out.argmax(1).cpu().numpy()
            for i, p in enumerate(preds):
                if p == y[i].item():
                    corr += 1
                if p != bl_preds[tids[i]]:
                    flips += 1
                tot += 1
            gate_devs_test.append((s - 1.0).abs().cpu().numpy())
            
    test_acc = corr / tot
    flip_rate = flips / tot
    drift_nm = torch.norm(model.head.weight.detach() - init_head_w).item()
    
    all_devs = np.concatenate(gate_devs_test, axis=0).flatten()
    final_gate_stats = {
        "mean_abs": np.mean(all_devs),
        "p95_abs": np.percentile(all_devs, 95),
        "max_abs": np.max(all_devs)
    }

    print(f"    Result: Acc={test_acc:.4f} (Delta={test_acc - bl_acc:.4f}) Flip={flip_rate:.2%} GateMean={final_gate_stats['mean_abs']:.4f}")

    return {
        "seed": seed,
        "n3c_acc": test_acc,
        "baseline80_acc": bl_acc,
        "delta_vs_baseline": test_acc - bl_acc,
        "flip_rate": flip_rate,
        "drift": drift_nm,
        "gate_mean_abs": final_gate_stats['mean_abs'],
        "gate_p95_abs": final_gate_stats['p95_abs'],
        "gate_max_abs": final_gate_stats['max_abs']
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n2-cache", default="data/_cache/phase15_n2_cache")
    parser.add_argument("--out-root", default="promoted_results/phase15/n3c_report")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr-adapter", type=float, default=1e-4) 
    parser.add_argument("--lr-agg", type=float, default=5e-4)
    parser.add_argument("--lr-linear", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--adapter-dim", type=int, default=32)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--gate-reg", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=16)
    
    args = parser.parse_args()
    cfg = vars(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(f"{args.out_root}", exist_ok=True)
    all_trials = load_all_n2_cache(args.n2_cache)
    
    results = []
    for seed in args.seeds:
        res = run_seed(seed, all_trials, cfg, device)
        if res: results.append(res)
            
    df = pd.DataFrame(results)
    df.to_csv(f"{args.out_root}/summary_by_seed.csv", index=False)
    print(f"Saved to {args.out_root}/summary_by_seed.csv")

if __name__ == "__main__":
    main()
