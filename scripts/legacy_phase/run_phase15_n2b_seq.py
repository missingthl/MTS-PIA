#!/usr/bin/env python
"""Phase 15 N2b: Learnable Temporal Aggregation (Logits-First, Mean-Init).

Goal:
- Train Attention Pooling on Logits.
- Initialize to Mean-Pooling (approx MeanLogit logic).
- Reduce regression on Strong Seeds while gaining on Weak Seeds.
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
from typing import Dict, List, Tuple

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# --- Dataset (Logits) ---

class LogitsDataset(Dataset):
    def __init__(self, trials: List[Dict]):
        self.trials = trials
        
    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, idx):
        return self.trials[idx]

def collate_logits(batch):
    # Batch is list of dicts {"logits": tensor(T, 3), "y": int, "tid": str}
    xs = [item["logits"] for item in batch] # [(T, 3)]
    ys = torch.tensor([item["y"] for item in batch], dtype=torch.long)
    tids = [item["tid"] for item in batch]
    
    lengths = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    x_padded = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0.0) # (B, T, 3)
    
    # Mask (True = Valid)
    mask = torch.arange(x_padded.size(1))[None, :] < lengths[:, None]
    
    return x_padded, ys, mask, tids

# --- Model ---

class LogitAttnPooling(nn.Module):
    def __init__(self, n_classes: int = 3, dropout: float = 0.5):
        super().__init__()
        # Input is (B, T, n_classes)
        
        # Attention Scorer: Linear(n_classes -> 1)
        # We want init to be 0 for uniform attention
        self.attn_score = nn.Linear(n_classes, 1)
        nn.init.zeros_(self.attn_score.weight)
        nn.init.zeros_(self.attn_score.bias)
        
        # Output Projector: Linear(n_classes -> n_classes)
        # We want init to be Identity to pass mean logits through
        self.classifier = nn.Linear(n_classes, n_classes)
        nn.init.eye_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # x: (B, T, 3)
        
        # Scores
        scores = self.attn_score(x).squeeze(-1) # (B, T)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1) # (B, T)
        
        # Weighted Sum (Pooling)
        # (B, T, 1) * (B, T, 3) -> (B, T, 3) -> sum -> (B, 3)
        ctx = (weights.unsqueeze(-1) * x).sum(dim=1)
        
        ctx = self.dropout(ctx)
        logits = self.classifier(ctx)
        
        return logits, weights

# --- Training ---

def run_seed(seed: int, logits_file: str, cfg: Dict, device: torch.device):
    print(f"\n=== Seed {seed} ===")
    
    # Load
    t_dict = torch.load(logits_file) # dict {tid: data}
    all_trials = sorted(list(t_dict.values()), key=lambda x: x["tid"])
    
    # Split (Same seed logic)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_trials))
    n_train_full = int(0.8 * len(all_trials))
    train_full = indices[:n_train_full]
    test_idx = indices[n_train_full:]
    
    n_train = int(0.8 * len(train_full))
    train_idx = train_full[:n_train]
    val_idx = train_full[n_train:]
    
    train_set = [all_trials[i] for i in train_idx]
    val_set = [all_trials[i] for i in val_idx]
    test_set = [all_trials[i] for i in test_idx]  # Use this for final perf
    
    # Baseline (Majority Vote on Fixed Logits? Or Mean Logit?)
    # The cached logits are from the window classifier.
    # So we can compute the baseline performance directly from these cached logits!
    # Baseline Aligned used Majority Vote of Window Predictions.
    # Window Prediction = argmax(logits).
    
    def calc_baseline_acc(dataset, mode="majority"):
        corr = 0
        for t in dataset:
            logits = t["logits"].numpy()
            y_true = t["y"]
            win_preds = np.argmax(logits, axis=1) # [T]
            
            if mode == "majority":
                from collections import Counter
                # Handle case where win_preds is empty (though T>0 usually)
                if len(win_preds) == 0:
                    pred = 0
                else:
                    pred = Counter(win_preds).most_common(1)[0][0]
            elif mode == "meanlogit":
                pred = np.argmax(np.mean(logits, axis=0))
            
            if pred == y_true:
                corr += 1
        return corr / len(dataset)
        
    bl_maj_acc = calc_baseline_acc(test_set, "majority")
    
    # DataLoaders
    train_dl = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_logits)
    val_dl = DataLoader(val_set, batch_size=cfg["batch_size"], collate_fn=collate_logits)
    test_dl = DataLoader(test_set, batch_size=cfg["batch_size"], collate_fn=collate_logits)
    
    # Model
    model = LogitAttnPooling(n_classes=3, dropout=cfg["dropout"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_state = None
    patience = 0
    best_epoch = 0
    
    # Calculate Initial Zero-Shot Performance (Should act like MeanLogit)
    model.eval()
    with torch.no_grad():
        c = 0
        tot = 0
        for x, y, mask, _ in test_dl:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            out, _ = model(x, mask)
            c += (out.argmax(1) == y).sum().item()
            tot += y.size(0)
        init_acc = c / tot
    print(f"    Init (MeanLogit) Test Acc: {init_acc:.4f} (Baseline Maj: {bl_maj_acc:.4f})")
    
    for epoch in range(cfg["epochs"]):
        model.train()
        for x, y, mask, _ in train_dl:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            out, _ = model(x, mask)
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
                out, _ = model(x, mask)
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
                
    # Final Test
    model.load_state_dict(best_state)
    model.eval()
    
    preds_n2b = []
    y_trues = []
    tids_list = []
    
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, mask, tids in test_dl:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            out, _ = model(x, mask)
            p = out.argmax(1)
            correct += (p == y).sum().item()
            total += y.size(0)
            
            preds_n2b.extend(p.cpu().tolist())
            y_trues.extend(y.cpu().tolist())
            tids_list.extend(tids)
            
    n2b_acc = correct / total
    
    # Flip Analysis
    flips = 0
    flip_wins = 0
    flip_losses = 0
    
    # We need trial-level baseline predictions for exact flip counting
    # Recalculate baseline per trial
    bl_preds = {}
    for t in test_set:
        logits = t["logits"].numpy()
        # Majority Vote
        win_preds = np.argmax(logits, axis=1)
        if len(win_preds) == 0:
            maj = 0
        else:
            from collections import Counter
            maj = Counter(win_preds).most_common(1)[0][0]
        bl_preds[t["tid"]] = int(maj)
        
    for i, tid in enumerate(tids_list):
        n2b_p = preds_n2b[i]
        bl_p = bl_preds[tid]
        y_p = y_trues[i]
        
        if n2b_p != bl_p:
            flips += 1
            if n2b_p == y_p:
                flip_wins += 1
            elif bl_p == y_p:
                flip_losses += 1
                
    flip_rate = flips / total
    flip_gain = flip_wins - flip_losses
    
    print(f"    Result: TestAcc={n2b_acc:.4f} (Delta={n2b_acc - bl_maj_acc:.4f})")
    print(f"    Flips: Rate={flip_rate:.2%}, Wins={flip_wins}, Losses={flip_losses}, Net={flip_gain}")
    
    return {
        "seed": seed,
        "n2b_acc": n2b_acc,
        "baseline_aligned": bl_maj_acc,
        "delta_vs_baseline": n2b_acc - bl_maj_acc,
        "init_acc": init_acc,
        "best_epoch": best_epoch,
        "flip_rate": flip_rate,
        "flip_net": flip_gain
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data/_cache/phase15_n2b_cache")
    parser.add_argument("--out-root", default="promoted_results/phase15/n2b_seq")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    
    # Config
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4) # Conservative
    parser.add_argument("--weight-decay", type=float, default=1e-1) # Strong anchor to Identity/Mean
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16) # Smaller batch for stability
    parser.add_argument("--patience", type=int, default=10)
    
    args = parser.parse_args()
    cfg = vars(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(f"{args.out_root}/seed1", exist_ok=True)
    
    results = []
    for seed in args.seeds:
        f = os.path.join(args.cache_dir, f"seed{seed}.pt")
        if not os.path.exists(f):
            print(f"Missing cache for seed {seed}")
            continue
        res = run_seed(seed, f, cfg, device)
        results.append(res)
        
    df = pd.DataFrame(results)
    df.to_csv(f"{args.out_root}/seed1/summary_by_seed.csv", index=False)
    print(f"Saved to {args.out_root}/seed1/summary_by_seed.csv")

if __name__ == "__main__":
    main()
