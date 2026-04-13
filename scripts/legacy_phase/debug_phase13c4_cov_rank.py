
import os
import sys
import json
import torch
import numpy as np
import argparse
import traceback

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets.adapters import get_adapter
from runners.manifold_deep_runner import TrialDataset
from models.band_activation import BandScalarGateV1
from models.spdnet import CovPoolWeightedBandsV1

def ensure_contiguous(x):
    if not x.is_contiguous():
        return x.contiguous()
    return x

def compute_diagnostics(cov, eps, label):
    # cov: (B, C, C)
    B, C, _ = cov.shape
    
    # 1. Eigenvalues
    # Use eigh for symmetric
    try:
        vals, _ = torch.linalg.eigh(cov)
        vals = vals.clamp(min=1e-9) # Avoid negatives for log
    except Exception as e:
        return {"error": str(e)}

    # Percentiles
    vals_flat = vals.detach().cpu().numpy().flatten()
    qs = np.percentile(vals_flat, [1, 5, 50, 95, 99])
    
    # 2. Count <= 10*eps
    # Post-eps values: max(vals, eps).
    # We want to check raw values relative to eps
    # But usually we analyze the matrix *after* regularization?
    # Prompt says: "eigenvalues... (pre-eps and post-eps)"
    # "count of eigenvalues <= 10*eps (post-eps)"
    
    vals_post = torch.clamp(vals, min=eps)
    count_low = (vals_post <= 10*eps).sum().item() / B
    
    # 3. Effective Rank (Entropy)
    # p = val / sum(val)
    # H = -sum(p log p)
    # Rank = exp(H)
    vals_sum = vals_post.sum(dim=1, keepdim=True)
    p = vals_post / vals_sum
    entropy = -(p * torch.log(p)).sum(dim=1)
    eff_rank = torch.exp(entropy).mean().item()
    
    # 4. Condition Number
    # max / min
    cond = (vals_post.max(dim=1)[0] / vals_post.min(dim=1)[0])
    cond_p95 = np.percentile(cond.detach().cpu().numpy(), 95)
    
    # 5. Eps Dominance
    # ||eps*I||_F / ||C||_F
    # eps*I norm = sqrt(C * eps^2) = eps * sqrt(C)
    # For C=62, eps*sqrt(62)
    eps_norm = eps * np.sqrt(C)
    c_norm = torch.norm(cov, dim=(1,2)).mean().item()
    eps_dom = eps_norm / (c_norm + 1e-9)
    
    return {
        "p01": qs[0], "p05": qs[1], "p50": qs[2], "p95": qs[3], "p99": qs[4],
        "count_low_avg": count_low,
        "eff_rank": eff_rank,
        "cond_p95": cond_p95,
        "eps_dominance": eps_dom
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="seed1")
    parser.add_argument("--bands_mode", type=str, default="all5_timecat")
    parser.add_argument("--spd_eps", type=float, default=0.001)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    
    print(f"DEBUG: Starting Covariance Rank Analysis for Seed={args.seed}")
    
    # 1. Load Data
    adapter = get_adapter(args.dataset)
    folds = adapter.get_manifold_trial_folds()
    # fold1 for training data
    # Create Dataset (guided=True implies return_5band=True in this codebase version)
    # logic from Runner: if mvp1_guided_cov: self.return_5band = True
    # Create Dataset (guided=True implies return_5band=True in this codebase version)
    fold = folds['fold1']
    if hasattr(fold, 'trials_train'):
        trials = fold.trials_train
        labels = fold.y_trial_train
    else:
        # Fallback if tuple
        trials = fold[0]
        labels = fold[1]
        
    dset = TrialDataset(trials, labels, band_idx=4, bands_mode=args.bands_mode, 
                        return_5band=True, window_len=24, stride=12) 
    
    # Get Batch
    loader = torch.utils.data.DataLoader(dset, batch_size=8, shuffle=False)
    Xb, yb, tib = next(iter(loader))
    
    print(f"Data Loaded: {Xb.shape} Stride:{Xb.stride()} Contig:{Xb.is_contiguous()}")
    
    # 2. Setup Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gate = BandScalarGateV1(n_bands=5).to(device).double()
    # Random weights are fine for RANK analysis (structural check)
    cov_pool = CovPoolWeightedBandsV1(eps=args.spd_eps).to(device).double()
    
    Xb = Xb.to(device).double()
    
    results = {}
    results["inputs"] = {
        "x_shape": list(Xb.shape),
        "stride": list(Xb.stride()),
        "contiguous": Xb.is_contiguous(),
        "eps": args.spd_eps
    }
    
    # === PATH 1: TimeCat Control (Simulation) ===
    # Control logic: Permute(0, 2, 1, 3) -> Reshape(B, 62, -1) -> Cov -> Eps
    # Xb: (B, 5, 62, T) from dataset (due to previous fix in Dataset __getitem__)
    # Note: Dataset returns (5, 62, T) now? 
    # Let's verify what Dataset returns. The previous inspection said [8, 5, 62, 24].
    
    # TimeCat logic: 
    # (B, 5, 62, T) -> (B, 62, 5, T) -> (B, 62, 5*T)
    X_control = Xb.permute(0, 2, 1, 3).reshape(Xb.shape[0], 62, -1)
    # Compute Cov
    B, C, T_total = X_control.shape
    mean = X_control.mean(dim=2, keepdim=True)
    Xc = X_control - mean
    cov_control = torch.bmm(Xc, Xc.transpose(1, 2)) / (T_total - 1)
    # Add eps
    ident = torch.eye(C, device=device).unsqueeze(0)
    cov_control_eps = cov_control + args.spd_eps * ident
    
    results["control_timecat"] = compute_diagnostics(cov_control, args.spd_eps, "control")
    results["control_timecat"]["T_eff"] = T_total
    
    # === PATH 2: BandGate (Current) ===
    # Logic from Runner: Xb.permute(0, 3, 2, 1) -> Gate -> CovPool
    # Wait, my "fix" in runner added .contiguous(). 
    # But here we want the raw behavior first to see if it matters?
    # "Step 1... Reproduce EXACTLY... Ensure any .view is replaced... if needed"
    # The prompt says "Step 3 — Add a minimal 'contiguity fix' probe...". So Step 2 is standard.
    
    # Standard:
    X_gate_in = Xb.permute(0, 3, 2, 1) # (B, 5, 62, 24) -> (B, 24, 62, 5) ? 
    # No, Dataset returns (B, 5, 62, T). 
    # Wait, Inspection said Shape Matches Expected ([B, 5, 62, 24]).
    # Checks runner code: Xb_perm = Xb.permute(0, 3, 2, 1).contiguous().
    # Xb from loader is (B, 5, 62, T).
    # Runner: `if self.bands_mode == "all5_timecat": ... if use_band_gate: Xb_perm = Xb.permute(0, 3, 2, 1)` -> (B, T, 62, 5).
    # Wait, `band_gate_model` expects (B, 5, 62, T)?
    # Let's check `band_activation.py`: `def forward(self, x): # x: (B, 5, 62, T)`.
    # OK, so `Xb` from Dataset is (B, 5, 62, T).
    # Runner does: `Xb_perm = Xb.permute(0, 3, 2, 1)`. That makes it (B, T, 62, 5).
    # Then passes to `band_gate_model`?
    # IF Runner passes `(B, T, 62, 5)` to `BandScalarGate` which expects `(B, 5, 62, T)`, that's a HUGE BUG.
    # PROBE output said: `x_shape: [8, 5, 62, 24]`.
    # This implies `Xb_perm` was `[8, 5, 62, 24]`.
    # So `Xb` must have been `[8, 24, 62, 5]`?
    # Let's check Dataset `__getitem__`.
    # Dataset `all5_timecat`:
    # returns `x = x_temp.permute(2, 1, 0)` -> (5, 62, T).
    # DataLoader stacks to (B, 5, 62, T).
    # Runner: `Xb` is (B, 5, 62, T).
    # Runner: `Xb_perm = Xb.permute(0, 3, 2, 1)` -> (B, T, 62, 5).
    # PROBE output said `[8, 5, 62, 24]`.
    # This contradicts.
    # UNLESS, `T` is 5? And `n_bands` is 24? No.
    # Let's re-read Inspection Report.
    # | x_shape | [8, 5, 62, 24] | Probe |
    # | stride | [7440, 1, 5, 310] | Probe |
    # If shape is [8, 5, 62, 24], then `permute(0, 3, 2, 1)` on `(B, 5, 62, 24)` would produce `(B, 24, 62, 5)`.
    # So if Probe printed `[8, 5, 62, 24]`, then `Xb_perm` IS `[8, 5, 62, 24]`.
    # This means `permute(0, 3, 2, 1)` RESULTED in `[8, 5, 62, 24]`.
    # That implies Input `Xb` was `[8, 24, 62, 5]`.
    # Check Dataset again.
    # `x = x_temp.permute(2, 1, 0)` -> (5, 62, T).
    # Wait, earlier in Dataset: `t_reshaped = t.reshape(t.shape[0], 5, 62).transpose(0, 2, 1)` -> (T, 62, 5).
    # `x_temp` (tensor) is (T, 62, 5).
    # `permute(2, 1, 0)` -> (5, 62, T).
    # So Dataset returns `(5, 62, T)`. Batch -> `(B, 5, 62, T)`.
    # So `Xb` is `(B, 5, 62, T)`.
    # Runner line: `Xb_perm = Xb.permute(0, 3, 2, 1)`.
    # Output: `(B, T, 62, 5)`.
    # PROBE SAYS: `[8, 5, 62, 24]`.
    # Contradiction unless `T=5` and `5=24`. Impossible.
    # OR: Runner Code `Xb_perm = Xb.permute(...)` ?
    # Let's look at `ManifoldDeepRunner` code again (via memory or verify).
    
    # Actually, verify Runner code NOW in thought.
    # Last edit: `Xb_perm = Xb.permute(0, 3, 2, 1).contiguous()`
    # If `Xb` is `(B, 5, 62, T)`, then `Xb_perm` is `(B, T, 62, 5)`.
    # Probe printed `Xb_perm.shape`.
    # If Probe says `[8, 5, 62, 24]`, then `T` MUST be 5, and `n_bands` MUST be 24?
    # Or `Xb` was `(B, 24, 62, 5)` coming from Dataset?
    # If Dataset returns `(T, 62, 5)`... 
    # `TrialDataset` code:
    # `x_temp = self._apply_norm(x_temp)` (T, 62, 5).
    # `x = x_temp.permute(2, 1, 0).numpy()` -> (5, 62, T).
    # So `Xb` is (B, 5, 62, T).
    # T=24? (Window length). `all5_timecat` means `T_eff = 5*window`.
    # If `Xb` is `(B, 5, 62, 24)`, then `permute(0, 3, 2, 1)` -> `(B, 24, 62, 5)`.
    
    # HYPOTHESIS: I misread the Probe output or the Permutation in Runner?
    # Inspecting previous turn output:
    # `[PROBE] Dumped input_probe.json: torch.Size([8, 5, 62, 24])`
    # And JSON content: `"x_shape": [8, 5, 62, 24]`.
    # THIS IMPLIES `Xb_perm` IS `[8, 5, 62, 24]`.
    # IF `Xb_perm` is `[8, 5, 62, 24]` AND `Xb` was `(B, 5, 62, 24)`, THEN `permute(0, 3, 2, 1)` would clearly NOT be Identity.
    # Wait, `permute(0, 3, 2, 1)` swaps dim 1 and 3.
    # `(B, A, B, C)` -> `(B, C, B, A)`.
    # If result is `(8, 5, 62, 24)`, then Input must be `(8, 24, 62, 5)`.
    # So `Xb` coming from Loader WAS `(8, 24, 62, 5)`.
    # This means Dataset returned `(24, 62, 5)`.
    # Let's check Dataset `__getitem__` again.
    # `x_temp` (T, 62, 5).
    # `x = x_temp.permute(2, 1, 0)` -> (5, 62, T).
    # I replaced this in Step 6110.
    # Maybe Step 6110 didn't apply correctly?
    # Or maybe the `else` block executed?
    # `if self.bands_mode == "all5_timecat": ... x = x_temp.permute(2, 1, 0).numpy()`
    # If `Xb` is `(8, 24, 62, 5)`, it means it returned `(T, 62, 5)`.
    # That means it SKIPPED my `permute(2, 1, 0)` change?
    
    # THIS IS A MAJOR FINDING if true.
    # "Layout/reshape bug" / "Dataset Mismatch".
    
    # For the debug script, I will simply trust `Xb` from Dataset and apply the SAME transformation as Runner used `Xb.permute(0, 3, 2, 1)`.
    # This will reproduce the "wrong" shape if it was wrong.
    
    # Path 2 (BandGate) in Script:
    X_gate = Xb.permute(0, 3, 2, 1)
    # If this results in (B, 5, 62, T), then Gate works.
    # If Output is (B, 5, 62, T), Gate (5->1) works on dim 1.
    # If Output is (B, T, 62, 5), Gate (5->1) works on dim 1 => Reduces T?
    # `BandScalarGateV1`: `var = x.var(dim=(2, 3))`.
    # Input (B, C1, C2, C3). Var over (C2, C3). Result (B, C1).
    # If Input is (B, 5, 62, T): Var over (62, T). Result (B, 5). Correct.
    # If Input is (B, T, 62, 5): Var over (62, 5). Result (B, T). 
    # Then MLP(T) -> T weights?
    # But `MLP` is `nn.Linear(n_bands, ...)`. `n_bands=5`.
    # If Input to MLP is (B, T=24), and expects 5, it will crash shapes mismatch `8x24` vs `5x16`.
    # Wait! The earlier error WAS `mat1 and mat2 shapes cannot be multiplied (8x24 and 5x16)`.
    # This proves `Xb` was (B, T=24, 62, 5).
    # So I tried to fix it by permuting `Xb`. 
    # If I fixed it by `Xb.permute(0, 3, 2, 1)`, and `Xb` was `(B, 24, 62, 5)`, result is `(B, 5, 62, 24)`.
    # This passes shape check.
    # So the *Fix* worked structurally.
    
    # But does `(B, 5, 62, 24)` make sense?
    # Dim 1=5 (Bands). Dim 2=62 (Channels). Dim 3=24 (Time).
    # `CovPoolWeighted`: `x.mean(dim=3)` -> Mean over Time. `cov` over Time.
    # `(B, 5, 62, 24)`. Cov over 24 samples.
    # Rank of Cov(24 samples) is min(62, 23)=23.
    # Then we sum 5 rank-23 matrices.
    # Max rank <= 5*23 = 115 (capped at 62).
    # So it *could* be full rank.
    # But effectively it might be lower.
    
    # Debug script logic pokračuje:
    # We will simulate this path.
    # We need to compute C_final and check Rank.
    
    w, _ = gate(X_gate) # (B, 5)
    cov_gate = cov_pool(X_gate, w) 
    
    results["bandgate"] = compute_diagnostics(cov_gate, args.spd_eps, "bandgate")
    
    # === Path 3: Forced Contiguous ===
    X_gate_contig = ensure_contiguous(X_gate)
    cov_gate_c = cov_pool(X_gate_contig, w)
    results["bandgate_contiguous"] = compute_diagnostics(cov_gate_c, args.spd_eps, "bandgate_c")
    
    # === Path 4: Gate Semantics ===
    # Gate on Signal: w * X
    # w: (B, 5). X: (B, 5, 62, T).
    w_uns = w.unsqueeze(2).unsqueeze(3) # (B, 5, 1, 1)
    # The math: sqrt(w) * X for covariance to be w*Cov?
    # `CovPoolWeightedBandsV1` does `sum(w * Cov(X))`.
    # $Cov(\sqrt{w} X) = (\sqrt{w})^2 Cov(X) = w Cov(X)$.
    # So equivalent signal modulation is $\sqrt{w} X$.
    X_mod = torch.sqrt(w_uns) * X_gate_contig
    # Now compute Cov of X_mod.
    # But `CovPoolWeightedBandsV1` sums covariances of bands.
    # `Cov(X_mod)`: if we treat (5, T) as dimensions?
    # No, we want `sum_b Cov(X_mod_b)`.
    # We can just use `CovPoolWeightedBandsV1` with weights=1?
    # `cov_pool(X_mod, ones)`.
    ones = torch.ones_like(w)
    cov_sig = cov_pool(X_mod, ones)
    
    # Compare cov_sig vs cov_gate
    diff = torch.norm(cov_sig - cov_gate_c, p='fro')
    norm = torch.norm(cov_sig, p='fro')
    ratio = diff / (norm + 1e-9)
    results["gate_semantics"] = {"fro_diff_ratio": ratio.item()}
    
    # Save
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"DEBUG: Saved to {args.out}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
