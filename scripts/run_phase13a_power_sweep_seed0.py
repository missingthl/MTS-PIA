
import sys
import os
import shutil
import json
import pandas as pd
import torch
import subprocess
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.adapters import get_adapter
from runners.manifold_deep_runner import ManifoldDeepRunner

SWEEP_ROOT = "promoted_results/phase13/seed0/power_sweep"
SPATIAL_CS_PATH = "promoted_results/phase13/seed0/spatial_trial_pred.csv"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def run_single_power(adapter, p, spatial_csv):
    print(f"\n=== Power Sweep: p={p} ===")
    
    run_dir = os.path.join(SWEEP_ROOT, f"p{p}")
    ensure_dir(run_dir)
    
    # 1. Manifold Config
    class Args:
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        epochs = 50
        batch_size = 32
        mvp1_guided_cov = True
        mvp1_attn_power = float(p) # Ensure float
        seed = 0
        dcnet_ckpt = "experiments/checkpoints/seedv_spatial_torch_seed0_phase13a_spatial_fallback.pt"
        metrics_csv = None
        cov_alpha = 0.01
        hidden_dim = 96
        
    args = Args()
    
    # Fallback Ckpt check
    if not os.path.exists(args.dcnet_ckpt):
         alt = "experiments/checkpoints/seed0_dcnet_refactor.pt"
         if os.path.exists(alt):
              args.dcnet_ckpt = alt
         else:
              print(f"CRITICAL: Teacher Checkpoint missing: {args.dcnet_ckpt}")
              sys.exit(3)
              
    # 2. Run
    folds = adapter.get_manifold_trial_folds()
    runner = ManifoldDeepRunner(args, num_classes=3)
    
    fold_name = f"phase13a_sweep_p{p}_seed0"
    res = runner.fit_predict(folds['fold1'], fold_name=fold_name)
    
    # 3. Move/Rename Output
    expected_csv = f"promoted_results/{fold_name}_preds_test_last_trial.csv"
    target_csv = os.path.join(run_dir, f"manifold_pred_p{p}.csv")
    
    if os.path.exists(expected_csv):
        shutil.copy(expected_csv, target_csv)
    else:
        print(f"Error: Output CSV missing: {expected_csv}")
        # Mark fail but continue? Exit 2.
        sys.exit(2)
        
    # 4. Write Meta
    meta = {
        "seed": 0,
        "guided": True,
        "power": p,
        "epochs": 50,
        "test_trial_acc": res['last']['test_trial_acc']
    }
    with open(os.path.join(run_dir, "export_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        
    # 5. Generate Report
    cmd = [
        sys.executable, "scripts/gen_single_run_report.py",
        "--run_dir", run_dir,
        "--spatial_csv", spatial_csv,
        "--manifold_csv", target_csv
    ]
    ret = subprocess.call(cmd)
    if ret != 0:
        print(f"Warning: Report generation failed for p={p}")
        # sys.exit(2) # Don't hard exit whole sweep? 
        return False
        
    return True

def main():
    powers = [0, 0.5, 1, 2, 4]
    
    adapter = get_adapter("seed1")
    
    if not os.path.exists(SPATIAL_CS_PATH):
        print(f"Critical: Spatial CSV not found at {SPATIAL_CS_PATH}")
        sys.exit(3)
    
    failed = False
    for p in powers:
        ok = run_single_power(adapter, p, SPATIAL_CS_PATH)
        if not ok:
            failed = True
            
    # Aggregate Summary
    print("\n=== generating Summary ===")
    results = []
    for p in powers:
        diag_path = os.path.join(SWEEP_ROOT, f"p{p}", "diagnostics.json")
        if os.path.exists(diag_path):
            with open(diag_path, 'r') as f:
                d = json.load(f)
                d['p'] = p
                results.append(d)
        else:
             print(f"Warning: Missing diagnostics for p={p}")
             
    if results:
        df = pd.DataFrame(results)
        summary_path = os.path.join(SWEEP_ROOT, "seed0_power_sweep_summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"Summary saved: {summary_path}")
        print(df.to_markdown())
        
    if failed:
        sys.exit(2)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
