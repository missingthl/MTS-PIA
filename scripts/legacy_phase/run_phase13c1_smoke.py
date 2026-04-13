
import sys
import os
import shutil
import json
import pandas as pd
import torch
import subprocess
import argparse
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets.adapters import get_adapter
from runners.manifold_deep_runner import ManifoldDeepRunner

BASE_DIR = "promoted_results/phase13c1/smoke"
SPATIAL_CS_PATH = "promoted_results/phase13/seed0/spatial_trial_pred.csv"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def run_smoke_case(adapter, mode, seed=0):
    print(f"\n=== Smoke Test: {mode} ===")
    run_dir = os.path.join(BASE_DIR, mode)
    ensure_dir(run_dir)
    
    # Config
    class Args:
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        epochs = 3
        batch_size = 8
        mvp1_guided_cov = False # Mode 'single' and 'all5' are usually unguided baseline or guided? 
        # User said "p=2.0 (if guided)". 
        # But for 'all5_timecat', guidance (spatial attention) is tricky if we just concat?
        # The user Plan B: "all5_timecat: concatenate... input 62x(T*5)". 
        # Guidance usually weights bands using DCNet attention. 
        # If we time-concat, we might not use guidance or we use it before concat?
        # User spec: "Input mode only changes dataset output... keep DeepSPDClassifier unchanged"
        # If I enable guidance, `ManifoldDeepRunner` computes saliency on 5-band input.
        # But 'all5_timecat' dataset mode changes output to flattened. 
        # `ManifoldDeepRunner` guidance logic expects `Xb` to be 5-band? 
        # In `TrialaDataset` for `all5_timecat`, I return (62, T*5). This is 2D (or 3D with batch).
        # My `ManifoldDeepRunner` update for `all5_timecat` in `train` loop: 
        # "if self.mvp1_guided_cov: A62 = self._compute_saliency(Xb.float())"
        # `_compute_saliency` expects `(B, Win, 62, 5)`.
        # But `all5_timecat` returns `(62, Win*5)`.
        # So `all5_timecat` is incompatible with `mvp1_guided_cov` as currently implemented?
        # The user PR says "Priority Plan B: Time-Concatenation for Rank".
        # It doesn't explicitly say "Guided".
        # But user said "p=2.0 (if guided)". implies it might be guided?
        # If I use `all5_timecat`, I likely should use UNGUIDED to check rank improvement first.
        # Let's assume UNGUIDED for `all5_timecat` smoke test unless specified.
        # Wait, if I use `bands_mode='single'`, I can use guided.
        # If `bands_mode='all5_timecat'`, `ManifoldDeepRunner` logic will crash if `guided=True` because shape mismatch.
        # I will set `guided=False` for this smoke test to focus on Rank/Stability.
        
        mvp1_guided_cov = False 
        mvp1_attn_power = 2.0
        seed = 0
        dcnet_ckpt = "experiments/checkpoints/seedv_spatial_torch_seed0_phase13_spatial_fallback.pt"
        metrics_csv = None
        cov_alpha = 0.01
        hidden_dim = 96
        bands_mode = mode
        spd_eps = 1e-3
        
    args = Args()
    
    # Run
    folds = adapter.get_manifold_trial_folds()
    runner = ManifoldDeepRunner(args, num_classes=3)
    fold_name = f"smoke_{mode}"
    
    # Cleanup previous logs
    if os.path.exists(f"promoted_results/{fold_name}_lambda_stats.csv"):
        os.remove(f"promoted_results/{fold_name}_lambda_stats.csv")
    
    try:
        res = runner.fit_predict(folds['fold1'], fold_name=fold_name)
    except Exception as e:
        print(f"Run Error: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    # Check outputs
    expected_csv = f"promoted_results/{fold_name}_preds_test_last_trial.csv"
    target_csv = os.path.join(run_dir, "manifold_trial_pred.csv")
    target_csv = os.path.join(run_dir, "manifold_trial_pred.csv")
    if os.path.exists(expected_csv):
        shutil.copy(expected_csv, target_csv)
        
    # Copy Window CSV
    expected_win = f"promoted_results/{fold_name}_preds_test_last_window.csv"
    target_win = os.path.join(run_dir, "manifold_window_pred.csv")
    if os.path.exists(expected_win):
        shutil.copy(expected_win, target_win)
    
    # Meta
    metrics_source = f"promoted_results/{fold_name}_metrics.json"
    meta_path = os.path.join(run_dir, "export_meta.json")
    if os.path.exists(metrics_source):
        shutil.copy(metrics_source, meta_path)
    elif "metadata" in res:
         # Fallback
         with open(meta_path, "w") as f:
             json.dump(res['metadata'], f, indent=2)
            
    # Gen Report
    cmd = [
        sys.executable, "scripts/analysis/gen_single_run_report.py",
        "--run_dir", run_dir,
        "--spatial_csv", SPATIAL_CS_PATH,
        "--manifold_csv", target_csv
    ]
    subprocess.call(cmd)
    
    # Read Stats
    meta = res['metadata']
    lambda_stats = pd.read_csv(f"promoted_results/{fold_name}_lambda_stats.csv")
    l_p05 = lambda_stats['l_p05'].iloc[-1]
    
    return {
        "mode": mode,
        "T": 24, # Window len hardcoded in runner
        "T_eff": meta['T_eff'],
        "spd_eps": meta['spd_eps'],
        "lambda_min_p05": l_p05,
        "acc": res['last']['test_trial_acc']
    }

def main():
    ensure_dir(BASE_DIR)
    adapter = get_adapter("seed1")
    
    results = []
    
    # 1. Single
    r1 = run_smoke_case(adapter, "single")
    if r1: results.append(r1)
    
    # 2. All5
    r2 = run_smoke_case(adapter, "all5_timecat")
    if r2: results.append(r2)
    
    # Summary
    if results:
        df = pd.DataFrame(results)
        print("\n=== Smoke Test Summary ===")
        print(df.to_string())
        
        # Write Report
        with open(os.path.join(BASE_DIR, "smoke_report.md"), "w") as f:
             f.write("# Phase 13C-1 Smoke Test\n\n")
             # Manual table
             f.write("| mode | T | T_eff | spd_eps | lambda_min_p05 | acc |\n")
             f.write("|---|---|---|---|---|---|\n")
             for _, row in df.iterrows():
                 f.write(f"| {row['mode']} | {row['T']} | {row['T_eff']} | {row['spd_eps']} | {row['lambda_min_p05']:.6f} | {row['acc']:.4f} |\n")
             
        # Validation
        # Check T_eff
        row_all5 = df[df['mode'] == 'all5_timecat'].iloc[0]
        if row_all5['T_eff'] == row_all5['T'] * 5:
            print("PASS: T_eff check for all5_timecat")
        else:
            print(f"FAIL: T_eff check. Got {row_all5['T_eff']}, expected {row_all5['T']*5}")
            sys.exit(2)
            
        sys.exit(0)
    else:
        sys.exit(3)

if __name__ == "__main__":
    main()
