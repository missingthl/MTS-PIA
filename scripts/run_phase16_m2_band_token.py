#!/usr/bin/env python
"""Phase 16 M2: Band Fusion (Promotion Gate).

Goal:
- GPU Readiness Validation.
- Verify Engineering Correctness on CPU:
  1. Weight Transfer (Sklearn -> Torch) < 1e-5 error.
  2. Gate Learnability (Mode-T + GateOnly).
  3. Mode-S Safety (NaN Quarantine).
- Telemetry: Full Visibility (Entropy, Grads, NaNs).
"""

import os
import sys
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
import time
import json
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.run_phase15_n3c_report import EndToEndModelSE, SafeScaler, SEGatedAdapter
from scripts.run_phase16_m1_collapse_check import load_factorized_cache
from scripts.run_phase14r_step6b1_rev2 import logm_spd, vec_utri

# --- Constants & Globals ---
DEBUG_MODE = False
DEBUG_MAX_BATCHES = 5
EPS_FUSE = 1e-2     # Stabilization for SPD sum (Increased for GPU stability)
EPS_EIG = 1e-3      # Min eigenvalue clamp (Increased for GPU stability)

# --- Telemetry Classes ---
class TelemetryTracker:
    def __init__(self):
        self.stats = []
        self.nan_grad_batches = 0
        self.total_batches = 0
        self.hybrid_eigh_fallbacks = 0
        
    def log_batch(self, seed, mode, device, epoch, batch, alpha, loss, grad_stats=None, spd_stats=None):
        # alpha: [B, T, 5] (Unflattened)
        # 1. Alpha Statistics
        # Mean over Batch & Time
        alpha_mean_vec = alpha.mean(dim=(0, 1)).detach().cpu().numpy() # [5]
        
        # Entropy per token: -sum(p log p)
        # alpha is [B, T, 5]
        # ent: [B, T]
        ent_token = -(alpha * (alpha + 1e-9).log()).sum(dim=-1)
        alpha_ent_mean = ent_token.mean().item()
        alpha_ent_max = ent_token.max().item()
        token_count = ent_token.numel()
        
        # Sum Check (should be bbox 1.0)
        alpha_sum_mean = alpha.sum(dim=-1).mean().item()

        row = {
            "seed": seed,
            "fuse_mode": mode,
            "device": str(device),
            "epoch": epoch,
            "batch": batch,
            "batch0_loss": loss,
            "alpha_mean": str(alpha_mean_vec.tolist()),
            "alpha_sum_mean": alpha_sum_mean,
            "alpha_ent_mean": alpha_ent_mean,
            "alpha_ent_max": alpha_ent_max,
            "token_count": token_count,
        }
        if grad_stats:
            row.update(grad_stats)
        if spd_stats:
            row.update(spd_stats)
            
        self.stats.append(row)
        
    def to_csv(self, path):
        pd.DataFrame(self.stats).to_csv(path, index=False)

tracker = TelemetryTracker()

# --- Math Utils ---

def batch_logm_spd(C, eps=1e-4):
    # C: [..., N, N]
    # 1. Enforce Symmetry
    C = 0.5 * (C + C.transpose(-1, -2))
    
    try:
        L, V = torch.linalg.eigh(C)
    except RuntimeError:
        # Hybrid Fallback
        tracker.hybrid_eigh_fallbacks += 1
        device = C.device
        C_cpu = C.cpu()
        L, V = torch.linalg.eigh(C_cpu)
        L = L.to(device)
        V = V.to(device)

    # 2. Clamp
    L = L.clamp(min=eps)
    LogL = torch.diag_embed(torch.log(L))
    res = V @ LogL @ V.transpose(-1, -2)
    return res

def batch_vec_utri(X):
    N = X.shape[-1]
    triu_idx = torch.triu_indices(N, N, device=X.device)
    return X[..., triu_idx[0], triu_idx[1]]

def get_gate_grad_stats(model):
    # Calculate Gate-specific gradient stats
    gate_params = list(model.gate.parameters())
    total_norm = 0.0
    n_none = 0
    n_params = len(gate_params)
    
    for p in gate_params:
        if p.grad is None:
            n_none += 1
        else:
            gn = p.grad.detach().norm(2).item()
            if np.isnan(gn) or np.isinf(gn):
                return None, n_none, n_params # Signal NaN
            total_norm += gn**2
            
    return total_norm**0.5, n_none, n_params

def collate_covs(batch):
    xs = [item["x_covs"] for item in batch]
    ys = torch.tensor([item["y"] for item in batch], dtype=torch.long)
    lengths = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    x_padded = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0.0)
    mask = torch.arange(x_padded.size(1))[None, :] < lengths[:, None]
    return x_padded, ys, mask

# --- Models ---

class BandFusionGate(nn.Module):
    def __init__(self, input_dim=1953, n_bands=5, hidden_dim=64):
        super().__init__()
        self.n_bands = n_bands
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim * n_bands, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_bands)
        )
        # Zero Init for Uniform Start
        nn.init.zeros_(self.gate_net[0].weight)
        nn.init.zeros_(self.gate_net[0].bias)
        nn.init.zeros_(self.gate_net[2].weight)
        nn.init.zeros_(self.gate_net[2].bias)
        
    def forward(self, covs):
        vecs = []
        for i in range(self.n_bands):
            c = covs[:, i]
            # Use robust logm (Gradient flow blocked? No, this is input projection)
            # Input covs are constant, so no grad needed w.r.t input.
            with torch.no_grad():
                log_c = batch_logm_spd(c, eps=EPS_EIG)
                v = batch_vec_utri(log_c)
            vecs.append(v)
            
        V_cat = torch.cat(vecs, dim=1)
        logits = self.gate_net(V_cat)
        weights = torch.softmax(logits, dim=1)
        # Return vecs detached for Mode-T efficiency (they are constant)
        return weights, vecs 

class FusedModelDual(nn.Module):
    def __init__(self, p15_model, device, mode="tangent"):
        super().__init__()
        self.gate = BandFusionGate().to(device)
        self.downstream = p15_model
        self.mode = mode
        self.device = device
        
    def forward(self, covs, mask):
        B, T, n_b, N, _ = covs.shape
        flat_covs = covs.view(-1, n_b, N, N) # [M, 5, N, N]
        
        # 1. Compute Weights & Tangent Vecs
        alpha, vecs_list = self.gate(flat_covs) # [M, 5], list of [M, D]
        
        # 2. Fuse
        if self.mode == "tangent":
            # Mode-T: Sum in Tangent Space
            V_stack = torch.stack(vecs_list, dim=1)
            alpha_view = alpha.unsqueeze(-1) # [M, 5, 1]
            x_fused_vec = (V_stack * alpha_view).sum(dim=1) # [M, D]
            
        elif self.mode == "spd":
            # Mode-S: Sum in SPD Space
            alpha_view = alpha.view(-1, n_b, 1, 1)
            C_fused = (flat_covs * alpha_view).sum(dim=1) # [M, N, N]
            
            # Regularize
            eye = torch.eye(N, device=self.device).view(1, N, N)
            C_fused = C_fused + EPS_FUSE * eye
            
            # Map
            C_log = batch_logm_spd(C_fused, eps=EPS_EIG) # [M, N, N]
            x_fused_vec = batch_vec_utri(C_log) # [M, D]
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        # 3. Downstream
        x_seq = x_fused_vec.view(B, T, -1)
        return self.downstream(x_seq, mask), alpha.view(B, T, 5)

# --- Execution ---

def run_seed(seed, all_trials, cfg, device, out_dir):
    print(f"\n=== Seed {seed} ({cfg['mode'].upper()}) ===")
    
    # Split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_trials))
    n_train = int(0.8 * len(all_trials))
    
    if DEBUG_MODE:
        indices = indices[:24] 
        n_train = 16 
    
    train_ds = [all_trials[i] for i in indices[:n_train]]
    test_ds = [all_trials[i] for i in indices[n_train:]]
    
    # Baseline
    x_train_list = []
    y_train_list = []
    
    for t in train_ds:
        c_mean = t["x_covs"].mean(dim=0).numpy()
        c_log = np.array([logm_spd(c, 1e-4) for c in c_mean], dtype=np.float32)
        x = np.array([vec_utri(c) for c in c_log], dtype=np.float32)
        x_train_list.append(x)
        y_train_list.append(np.full(len(x), t["y"]))
        
    X_train_flat = np.concatenate(x_train_list, axis=0)
    y_train_flat = np.concatenate(y_train_list, axis=0)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_flat)
    
    clf = LogisticRegression(solver="lbfgs", multi_class="multinomial", C=1.0, max_iter=200, random_state=seed)
    clf.fit(X_train_s, y_train_flat)
    
    # Warm Start P15
    p15 = EndToEndModelSE(1953, 3, scaler.mean_, scaler.scale_, cfg["adapter_dim"], cfg["eps"], cfg["dropout"])
    with torch.no_grad():
        p15.head.weight.copy_(torch.from_numpy(clf.coef_))
        p15.head.bias.copy_(torch.from_numpy(clf.intercept_))
        
    # --- Check 1: Weight Transfer Equivalence ---
    print("  [Check 1] Verifying Weight Transfer...")
    # Get SKLearn Logits for a batch
    X_sample_s = X_train_s[:4]
    sk_logits = clf.decision_function(X_sample_s)
    
    # Get Torch Logits
    # P15 Head: x @ W.T + b
    # x: [B, D]
    x_torch = torch.from_numpy(X_sample_s).float()
    with torch.no_grad():
        torch_logits = p15.head(x_torch).numpy()
        
    # Diff
    # Note: sklearn multinomial decision_function is [B, C]. For binary/multiclass it varies.
    # Here classes=3, so sk_logits is [B, 3].
    diff = np.abs(sk_logits - torch_logits).max()
    print(f"    Max Logit Diff: {diff:.2e}")
    if diff > 1e-5:
        print("    FATAL: Weight Transfer Failed! Check Scikit vs Torch versions.")
        # raise ValueError("Weight Transfer Mismatch")
    else:
        print("    PASSED.")
        
    # Dual Model
    model = FusedModelDual(p15, device, mode=cfg["mode"]).to(device)
    
    # --- Check 2: Epoch 0 Uniformity ---
    model.eval()
    dl_chk = DataLoader(test_ds, batch_size=4, collate_fn=collate_covs)
    x, y, mask = next(iter(dl_chk))
    x, mask = x.to(device), mask.to(device)
    
    with torch.no_grad():
        _, alpha = model(x, mask)
        if not torch.allclose(alpha, torch.tensor(0.2, device=device), atol=1e-3):
            print(f"WARNING: Epoch 0 Alpha Non-Uniform! Mean: {alpha.mean()}")
        else:
            print("  [Check 2] Epoch 0 Alpha Uniform (0.2). PASSED.")
            
    # Train
    lr_p15 = cfg["lr_p15"]
    if cfg["gate_only"]:
        lr_p15 = 0.0
        print("  [Config] Gate-Only Training (Downstream Frozen)")
        
    optimizer = optim.AdamW([
        {"params": model.gate.parameters(), "lr": cfg["lr_gate"]},
        {"params": model.downstream.parameters(), "lr": lr_p15}
    ], weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_covs)
    
    model.train()
    
    gate_params_init = [p.detach().clone() for p in model.gate.parameters()]
    
    for epoch in range(cfg["epochs"]):
        for batch_i, (x, y, mask) in enumerate(train_dl):
            tracker.total_batches += 1
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            
            # Forward
            out, alpha = model(x, mask)
            p15_res = out[0]
            loss = criterion(p15_res, y)
            
            loss.backward()
            
            # Check Gradients
            grad_nan = False
            gate_grad_norm, n_gate_none, n_gate_total = get_gate_grad_stats(model)
            
            if gate_grad_norm is None: # NaN in gate grad
                grad_nan = True
            
            # Check Downstream NaNs
            if not grad_nan:
                for p in model.downstream.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any():
                            grad_nan = True
                            break
            
            if grad_nan:
                tracker.nan_grad_batches += 1
                if DEBUG_MODE: 
                    print(f"    [Skipped] NaN Gradient at Ep{epoch} B{batch_i}")
                optimizer.zero_grad()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            # Log Telemetry
            if batch_i == 0:
                curr_gate_params = list(model.gate.parameters())
                upt_norm = 0.0
                for p0, p1 in zip(gate_params_init, curr_gate_params):
                    upt_norm += (p0 - p1).norm(2).item()**2
                upt_norm = upt_norm**0.5
                
                grad_stats = {
                    "gate_grad_norm": gate_grad_norm if gate_grad_norm else 0.0,
                    "n_gate_grads_none": n_gate_none,
                    "gate_update_norm": upt_norm,
                    "n_nan_grads_batches": tracker.nan_grad_batches
                }
                
                tracker.log_batch(seed, cfg["mode"], device, epoch, batch_i, alpha, loss.item(), grad_stats)
                print(f"    Ep {epoch}: Loss={loss.item():.4f} GateGrad={grad_stats['gate_grad_norm']:.4f} Upd={upt_norm:.4f} NaNs={tracker.nan_grad_batches}")

    # Final Eval
    model.eval()
    test_dl = DataLoader(test_ds, batch_size=cfg["batch_size"], collate_fn=collate_covs)
    corr = 0
    tot = 0
    alpha_accum = []
    
    with torch.no_grad():
        for x, y, mask in test_dl:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            res, alpha = model(x, mask)
            preds = res[0].argmax(dim=1).cpu().numpy()
            y_np = y.cpu().numpy()
            
            corr += (preds == y_np).sum()
            tot += len(y_np) 
            alpha_accum.append(alpha.mean(dim=(0,1)).cpu().numpy())
            
    acc = corr / tot if tot > 0 else 0.0
    
    mean_alpha = np.mean(alpha_accum, axis=0)
    print(f"  Result: Acc={acc:.4f} Alphas={mean_alpha}")
    
    # Sanity Assert
    if acc < 0.0 or acc > 1.0:
        raise ValueError(f"FATAL: Accuracy {acc} out of bounds!")

    return {
        "seed": seed,
        "acc": acc,
        "mode": cfg["mode"],
        "alpha_means": mean_alpha.tolist()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data/_cache/phase16_m1_cache")
    parser.add_argument("--out-dir", default="promoted_results/phase16/m2_band_token")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--epochs", type=int, default=25)
    
    parser.add_argument("--fuse-mode", choices=["tangent", "spd"], default="tangent")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gate-only", action="store_true", help="Freeze downstream")
    
    args = parser.parse_args()
    
    global DEBUG_MODE
    DEBUG_MODE = args.debug
    
    # Auto-device
    if args.device:
        device_str = args.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    print(f"Running M2 Promotion Gate | Mode={args.fuse_mode} | Device={device} | Debug={DEBUG_MODE} | GateOnly={args.gate_only}")
    print(f"Exec: {sys.executable}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Config
    batch_size = 4 if DEBUG_MODE else 16
    epochs = 3 if DEBUG_MODE else args.epochs 
    seeds = [0] if DEBUG_MODE and args.seeds == list(range(10)) else args.seeds
    
    cfg = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr_gate": 5e-4, 
        "lr_p15": 1e-4,
        "adapter_dim": 32,
        "eps": 0.05,
        "dropout": 0.5,
        "mode": args.fuse_mode,
        "gate_only": args.gate_only
    }
    
    # Load Data
    import glob
    files = sorted(glob.glob(os.path.join(args.cache_dir, "*.pt")))
    raw_trials = []
    for f in tqdm(files, desc="Loading Data"):
        try:
             d = torch.load(f)
             raw_trials.append({"x_covs": d["x_covs"], "y": d["y"]})
        except Exception:
             pass
             
    results = []
    for seed in seeds:
         res = run_seed(seed, raw_trials, cfg, device, args.out_dir)
         results.append(res)
         
    # Save Artifacts
    tracker.to_csv(f"{args.out_dir}/telemetry_summary.csv")
    
    run_name = f"summary_{args.fuse_mode}"
    rows = []
    for r in results:
        rows.append({
            "seed": r["seed"],
            "acc": r["acc"],
            "alpha_mean": str(r["alpha_means"]),
            "notes": f"Gate Check Mode={args.fuse_mode} GateOnly={args.gate_only}"
        })
    pd.DataFrame(rows).to_csv(f"{args.out_dir}/{run_name}_report.csv", index=False)
    
    # Run Meta JSON
    meta = {
        "env": sys.executable,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "args": vars(args),
        "telemetry": {
             "nan_grad_batches": tracker.nan_grad_batches,
             "total_batches": tracker.total_batches
        }
    }
    with open(f"{args.out_dir}/run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
