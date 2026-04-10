#!/usr/bin/env python
"""Phase 15 N2: Learnable Trial Aggregation.

Goal:
- Train Attention Pooling model on frozen tangent features.
- Compare trial-level accuracy vs Aligned Baseline.

Protocol:
- Load cached features.
- Iterate Seeds 0-9.
- Split: Train (64%), Val (16%), Test (20%). (Using 80/20 train/test, then 80/20 train/val inside train).
- Standardize: Fit on Train. Apply to Val/Test.
- Model: Project(1953->d) -> Attention(d) -> Classify(3).
- Train: SGD/AdamW. Early stopping on Val Acc.
- Eval: Test Acc vs Baseline.
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
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Dataset ---

class TrialDataset(Dataset):
    def __init__(self, trials: List[Dict]):
        self.trials = trials
        
    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, idx):
        return self.trials[idx]

def collate_trials(batch):
    # Batch is list of dicts {"x": tensor(L, D), "y": int, "tid": str}
    # Pad sequences
    xs = [item["x"] for item in batch]
    ys = torch.tensor([item["y"] for item in batch], dtype=torch.long)
    tids = [item["tid"] for item in batch]
    
    # Pad
    lengths = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    x_padded = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0.0)
    
    # Mask (True where valid, False where padding - wait, Transformer mask usually True for mask?)
    # For simple attention pooling manual impl, we just use lengths or mask (1 for valid).
    mask = torch.arange(x_padded.size(1))[None, :] < lengths[:, None]
    
    return x_padded, ys, mask, tids

# --- Model ---

class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attn_scores = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, x, mask):
        # x: (B, L, input_dim)
        # mask: (B, L) boolean, True = Valid
        
        h = self.project(x) # (B, L, H)
        
        # Attention
        scores = self.attn_scores(h).squeeze(-1) # (B, L)
        scores = scores.masked_fill(~mask, -1e9) # Mask padding
        weights = torch.softmax(scores, dim=1) # (B, L)
        
        # Weighted Sum
        # (B, L, 1) * (B, L, H) -> (B, L, H) -> sum(1) -> (B, H)
        ctx = (weights.unsqueeze(-1) * h).sum(dim=1)
        
        logits = self.classifier(ctx)
        return logits, weights

# --- Training ---

def train_one_seed(seed: int, cache_dir: str, cfg: Dict, device: torch.device):
    print(f"\n=== Seed {seed} ===")
    
    # 1. Load All Trials
    # Need to know total list to split deterministically matching Step 6B2
    # We load everything into memory (675 trials * L * 1953 * 4 bytes ~ 600MB? Easy)
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    all_data = []
    for f in files:
        all_data.append(torch.load(f))
        
    # Sort by TID to ensure deterministic strict order before RNG split
    all_data.sort(key=lambda x: x["tid"])
    
    # 2. Split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_data))
    n_train_full = int(0.8 * len(all_data))
    
    train_full_idx = indices[:n_train_full]
    test_idx = indices[n_train_full:]
    
    # Sub-split Train into Train/Val
    n_train = int(0.8 * len(train_full_idx))
    train_idx = train_full_idx[:n_train]
    val_idx = train_full_idx[n_train:]
    
    train_set = [all_data[i] for i in train_idx]
    val_set = [all_data[i] for i in val_idx]
    test_set = [all_data[i] for i in test_idx]
    
    print(f"    Split: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    
    # 3. Standardize
    print("    Standardizing...")
    # Collect all train vectors to fit scaler
    # Flatten list of tensors
    # x is (L, D)
    all_train_vecs = torch.cat([t["x"] for t in train_set], dim=0).numpy()
    scaler = StandardScaler()
    scaler.fit(all_train_vecs)
    
    del all_train_vecs # Free mem
    
    def apply_standardize(t_list):
        for item in t_list:
            # item["x"] is torch tensor.
            # Convert to numpy, transform, back to torch
            n = item["x"].numpy()
            n_s = scaler.transform(n)
            item["x"] = torch.from_numpy(n_s).float()
            
    apply_standardize(train_set)
    apply_standardize(val_set)
    apply_standardize(test_set)
    
    # 4. DataLoaders
    train_dl = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_trials)
    val_dl = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_trials)
    test_dl = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_trials)
    
    # 5. Model
    model = AttentionPooling(input_dim=1953, hidden_dim=cfg["hidden_dim"], n_classes=3, dropout=cfg["dropout"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    
    # 6. Loop
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(cfg["epochs"]):
        # Train
        model.train()
        for x, y, mask, _ in train_dl:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            logits, _ = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
        # Val
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, mask, _ in val_dl:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                logits, _ = model(x, mask)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        val_acc = correct / total if total > 0 else 0
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= cfg["patience"]:
            break
            
    # 7. Test
    model.load_state_dict(best_state)
    model.eval()
    t_correct = 0
    t_total = 0
    with torch.no_grad():
        for x, y, mask, _ in test_dl:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits, _ = model(x, mask)
            preds = logits.argmax(dim=1)
            t_correct += (preds == y).sum().item()
            t_total += y.size(0)
            
    test_acc = t_correct / t_total
    print(f"    Result: ValAcc={best_val_acc:.4f} (Ep {best_epoch}), TestAcc={test_acc:.4f}")
    
    return {
        "seed": seed,
        "n2_acc": test_acc,
        "best_epoch": best_epoch,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data/_cache/phase15_n2_cache")
    parser.add_argument("--out-root", default="promoted_results/phase15/n2_seq")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    
    # Model Hyperparams
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4) # Slightly stronger reg
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)
    
    args = parser.parse_args()
    cfg = vars(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(f"{args.out_root}/seed1", exist_ok=True)
    
    results = []
    
    for seed in args.seeds:
        res = train_one_seed(seed, args.cache_dir, cfg, device)
        results.append(res)
        
    # Save Summary
    df = pd.DataFrame(results)
    df.to_csv(f"{args.out_root}/seed1/summary_by_seed.csv", index=False)
    print(f"Saved to {args.out_root}/seed1/summary_by_seed.csv")

if __name__ == "__main__":
    main()
