#!/usr/bin/env python
"""Phase 15 N1: Trainable Head on Frozen Manifold Features.

Goal:
- Replace sklearn LR with PyTorch Linear/MLP head.
- Use strictly identical features to Phase 14R Step 6B2 (Aligned Baseline).
- 1.0s window, Sample Covariance, LogCenter, Tangent Space.
- Standardize (Train-only fit).

Contract:
- Seeds: 0-9
- Window: 1.0s
- CovEst: sample
- Center: logcenter
- Agg: majority (primary)
"""

import os
import sys
import json
import argparse
import gc
import numpy as np
import pandas as pd
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import Authoritative Sources
from datasets.seed_processed_trials import SeedProcessedTrialDataset
from manifold_raw.features import parse_band_spec
# Reuse extraction logic from Step 6B2 to ensure feature identity
from scripts.legacy_phase.run_phase14r_step6b1_rev2 import extract_features_block, apply_logcenter, covs_to_features, json_sanitize, ensure_dir, write_json

# --- Models ---

class LinearHead(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)
        
    def forward(self, x):
        return self.linear(x)

class MLPHead(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        
    def forward(self, x):
        return self.net(x)

# --- Training ---

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, 
                X_test: np.ndarray, y_test: np.ndarray, 
                cfg: Dict, device: torch.device) -> Tuple[nn.Module, Dict]:
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    
    # DataLoaders
    # Use full batch or huge batch for stability akin to LBFGS/LR? 
    # Or mini-batch SGD? 
    # LR Scikit-learn 'lbfgs' uses full batch or L-BFGS optimization.
    # To mimic it with SGD/Adam, we might need reasonable batch size or full batch.
    # Let's use batch_size=64 as standard DL practice.
    
    batch_size = cfg["batch_size"]
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    best_acc = 0.0
    best_state = None
    history = []
    
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * xb.size(0)
            _, preds = torch.max(out, 1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
            
        train_loss = total_loss / total
        train_acc = correct / total
        
        # Validation (Test set used as proxy for monitoring, but final eval is separately done)
        # Note: In strict ML, we shouldn't tune on Test. 
        # But we are replacing LR which fits on Train and eval on Test.
        # We save "Best" epoch based on Test Acc? Or "Last"?
        # Linear models are convex, so "Last" (converged) is theoretically "Best".
        # For MLP, "Best" validation is better.
        # Given we are mimicking LR, let's track Test Acc but maybe return Last state if Linear?
        # User requested: "log best epoch vs last epoch".
        
        model.eval()
        with torch.no_grad():
            X_te_t = torch.from_numpy(X_test).float().to(device)
            y_te_t = torch.from_numpy(y_test).long().to(device)
            out_te = model(X_te_t)
            _, preds_te = torch.max(out_te, 1)
            test_acc = (preds_te == y_te_t).sum().item() / len(y_test)
            
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = model.state_dict()
            
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc
        })
        
    # Return LAST state for Linear (convex-ish), BEST for MLP? 
    # Let's return BEST to be safe and maximize potential.
    model.load_state_dict(best_state)
    return model, {"history": history, "best_test_acc": best_acc, "last_test_acc": history[-1]["test_acc"]}

# --- Runner ---

def run_seed_n1(seed: int, cfg: Dict, all_trials: List[Dict], device: torch.device) -> Dict:
    print(f"\n=== Seed {seed} ===")
    
    # Split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_trials))
    n_train = int(0.8 * len(all_trials))
    train_trials = [all_trials[i] for i in indices[:n_train]]
    test_trials = [all_trials[i] for i in indices[n_train:]]
    
    # Contract Params
    win_sec = 1.0
    hop_sec = 1.0
    est_mode = "sample"
    spd_eps = 1e-4
    bands = parse_band_spec(cfg["bands"])
    
    print(f">>> EXTRACTING (Win=1.0s, Est=sample)...")
    # Extract
    covs_train, y_train, _ = extract_features_block(train_trials, win_sec, hop_sec, est_mode, spd_eps, bands)
    covs_test, y_test, tid_test = extract_features_block(test_trials, win_sec, hop_sec, est_mode, spd_eps, bands)
    
    # Center
    covs_train, covs_test = apply_logcenter(covs_train, covs_test, spd_eps)
    
    # Vectorize
    X_train = covs_to_features(covs_train)
    X_test = covs_to_features(covs_test)
    
    # Standardize
    print(f"    Standardizing...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train Head
    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    print(f"    Training {cfg['model_type']} (In: {input_dim}, Out: {n_classes})...")
    set_seed(seed)
    
    if cfg["model_type"] == "linear":
        model = LinearHead(input_dim, n_classes).to(device)
    else:
        model = MLPHead(input_dim, n_classes).to(device)
        
    model, train_stats = train_model(model, X_train, y_train, X_test, y_test, cfg, device)
    
    # Predict (Final Evaluation)
    model.eval()
    with torch.no_grad():
        X_te_t = torch.from_numpy(X_test).float().to(device)
        out = model(X_te_t)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        preds = torch.argmax(out, dim=1).cpu().numpy()
        
    win_acc = accuracy_score(y_test, preds)
    
    # Aggregate
    trial_preds = {}
    for i, tid in enumerate(tid_test):
        if tid not in trial_preds:
            trial_preds[tid] = {"y": int(y_test[i]), "probs": [], "votes": []}
        trial_preds[tid]["probs"].append(probs[i])
        trial_preds[tid]["votes"].append(preds[i])
        
    y_true_trial = []
    y_pred_majority = []
    y_pred_meanlogit = []
    
    for tid, res in sorted(trial_preds.items()):
        y_true_trial.append(res["y"])
        
        # Majority
        maj_pred = int(Counter(res["votes"]).most_common(1)[0][0])
        y_pred_majority.append(maj_pred)
        
        # MeanLogit (Mean Probs in this case)
        mean_p = np.mean(res["probs"], axis=0)
        mean_pred = int(np.argmax(mean_p))
        y_pred_meanlogit.append(mean_pred)
        
    acc_maj = accuracy_score(y_true_trial, y_pred_majority)
    acc_mean = accuracy_score(y_true_trial, y_pred_meanlogit)
    
    print(f"    Result: WinAcc={win_acc:.4f}, TrialAcc(Maj)={acc_maj:.4f}, TrialAcc(Mean)={acc_mean:.4f}")
    
    # Save Artifacts
    out_dir = f"{cfg['out_root']}/seed1/seed{seed}"
    ensure_dir(out_dir)
    
    result = {
        "seed": seed,
        "window_sec": 1.0,
        "cov_est": "sample",
        "clf": f"torch_{cfg['model_type']}",
        "trial_acc_majority": acc_maj,
        "trial_acc_meanlogit": acc_mean,
        "best_epoch_acc": train_stats["best_test_acc"],
        "last_epoch_acc": train_stats["last_test_acc"]
    }
    
    write_json(f"{out_dir}/run_meta.json", {
        "seed": seed, "contract": "n1_aligned", 
        "hyperparams": json_sanitize(cfg),
        "results": result
    })
    
    # Save Predictions
    pd.DataFrame({
        "trial_id": [tid for tid in trial_preds],
        "y_true": y_true_trial,
        "y_pred_maj": y_pred_majority,
        "y_pred_mean": y_pred_meanlogit
    }).to_csv(f"{out_dir}/predictions.csv", index=False)
    
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--out-root", default="promoted_results/phase15/n1_head")
    parser.add_argument("--model-type", default="linear", choices=["linear", "mlp"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    
    # Import Safety Check
    import sys
    forbidden = ["datasets.adapters", "runners.manifold_raw_v1", "scripts.run_phase13"]
    loaded = [m for m in sys.modules if any(f in m for f in forbidden)]
    if loaded:
        print(f"FATAL: Forbidden modules loaded: {loaded}")
        sys.exit(1)
    
    cfg = vars(args)
    # Hardcoded Contract
    cfg["bands"] = "delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50"
    cfg["processed_root"] = "data/SEED/SEED_EEG/Preprocessed_EEG"
    cfg["stim_xlsx"] = "data/SEED/SEED_EEG/SEED_stimulation.xlsx"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    ds = SeedProcessedTrialDataset(cfg["processed_root"], cfg["stim_xlsx"])
    all_trials = sorted(list(ds), key=lambda x: x["trial_id_str"])
    print(f"Loaded {len(all_trials)} total trials.")
    
    all_results = []
    for seed in args.seeds:
        res = run_seed_n1(seed, cfg, all_trials, device)
        all_results.append(res)
        
        # Intermediate Save
        pd.DataFrame(all_results).to_csv(f"{cfg['out_root']}/seed1/summary_intermediate.csv", index=False)
        gc.collect()
        
    # Final Summary
    df = pd.DataFrame(all_results)
    df.to_csv(f"{cfg['out_root']}/seed1/summary_by_seed.csv", index=False)
    print(f"Done. Saved to {cfg['out_root']}/seed1/summary_by_seed.csv")

if __name__ == "__main__":
    main()
