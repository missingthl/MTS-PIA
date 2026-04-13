
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

# Constants
BASE_PROMOTED = "promoted_results/phase13"
SWEEP_ROOT = os.path.join(BASE_PROMOTED, "power_sweep")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def get_spatial_csv(seed):
    return os.path.join(BASE_PROMOTED, f"seed{seed}", "spatial_trial_pred.csv")

def run_single_experiment(adapter, seed, p, epochs, stage, spatial_csv):
    print(f"\n[{stage}] Run: Seed={seed}, p={p}, Ep={epochs}")
    
    run_dir = os.path.join(SWEEP_ROOT, f"seed{seed}", f"{stage}", f"p{p}")
    ensure_dir(run_dir)
    
    # 1. Config
    ep_val = int(epochs)
    seed_val = int(seed)
    power_val = float(p)
    
    class Args:
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        epochs = ep_val
        batch_size = 32
        mvp1_guided_cov = True
        mvp1_attn_power = power_val
        seed = seed_val
        # Use Fallback Ckpt logic
        dcnet_ckpt = f"experiments/checkpoints/seedv_spatial_torch_seed{seed}_phase13_spatial_fallback.pt"
        metrics_csv = None
        cov_alpha = 0.01
        hidden_dim = 96
        
    args = Args()
    
    # Checkpoint Validation
    if not os.path.exists(args.dcnet_ckpt):
         alt = f"experiments/checkpoints/seedv_spatial_torch_seed{seed}_refactor.pt"
         if os.path.exists(alt):
              args.dcnet_ckpt = alt
         else:
              print(f"CRITICAL: Teacher Checkpoint missing: {args.dcnet_ckpt}")
              return None

    # 2. Run
    # Warning: Re-instantiating runner repeatedly might accumulate generic logic overhead, but fine for sweep.
    # Note: Adapter is always "seed1" (dataset name).
    folds = adapter.get_manifold_trial_folds()
    
    runner = ManifoldDeepRunner(args, num_classes=3)
    fold_name = f"phase13a_sweep_s{seed}_{stage}_p{p}"
    
    try:
        res = runner.fit_predict(folds['fold1'], fold_name=fold_name)
    except Exception as e:
        print(f"Run Error: {e}")
        return None
        
    # 3. Artifact Management
    expected_csv = f"promoted_results/{fold_name}_preds_test_last_trial.csv"
    target_csv = os.path.join(run_dir, f"manifold_pred.csv")
    
    if os.path.exists(expected_csv):
        shutil.copy(expected_csv, target_csv)
    else:
        print(f"Error: Output CSV missing: {expected_csv}")
        return None
        
    # 4. Meta
    meta = {
        "seed": seed,
        "guided": True,
        "power": p,
        "stage": stage,
        "epochs": epochs,
        "test_trial_acc": res['last']['test_trial_acc']
    }
    with open(os.path.join(run_dir, "export_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        
    # 5. Report via Generator
    cmd = [
        sys.executable, "scripts/analysis/gen_single_run_report.py",
        "--run_dir", run_dir,
        "--spatial_csv", spatial_csv,
        "--manifold_csv", target_csv
    ]
    ret = subprocess.call(cmd)
    
    # 6. Load Diagnostics for Return
    diag_path = os.path.join(run_dir, "diagnostics.json")
    if os.path.exists(diag_path):
        with open(diag_path, 'r') as f:
            d = json.load(f)
            d.update({"seed": seed, "stage": stage, "epochs": epochs, "p": p})
            return d
            
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,4")
    parser.add_argument("--powers", default="0,0.5,1,2,4")
    parser.add_argument("--stage1_epochs", type=int, default=10)
    parser.add_argument("--stage2_epochs", type=int, default=50)
    parser.add_argument("--topk", type=int, default=2)
    args = parser.parse_args()
    
    seeds = [int(x) for x in args.seeds.split(",")]
    powers = [float(x) for x in args.powers.split(",")]
    
    adapter = get_adapter("seed1")
    
    # === Stage 1: Screening ===
    print("=== Stage 1: Screening ===")
    stage1_results = []
    
    for seed in seeds:
        spatial_csv = get_spatial_csv(seed)
        if not os.path.exists(spatial_csv):
            print(f"Error: Spatial CSV missing for Seed {seed}")
            continue
            
        for p in powers:
            res = run_single_experiment(adapter, seed, p, args.stage1_epochs, "stage1", spatial_csv)
            if res:
                stage1_results.append(res)
                
    if not stage1_results:
        print("Stage 1 failed completely.")
        sys.exit(2)
        
    df1 = pd.DataFrame(stage1_results)
    df1.to_csv(os.path.join(SWEEP_ROOT, "stage1_summary.csv"), index=False)
    print("\nStage 1 Summary:")
    print(df1.to_string())
    
    # === Selection ===
    print("\n=== Selection (Top K) ===")
    stage2_candidates = []
    
    for seed in seeds:
        sub = df1[df1["seed"] == seed]
        if sub.empty: continue
        
        # Rank by manifold_acc
        sub = sub.sort_values("manifold_acc", ascending=False)
        top = sub.head(args.topk)
        print(f"Seed {seed} Top {args.topk}: p={top['p'].tolist()}")
        
        for _, row in top.iterrows():
            stage2_candidates.append((seed, row['p']))
            
    # === Stage 2: Confirmation ===
    print("\n=== Stage 2: Confirmation ===")
    stage2_results = []
    
    for seed, p in stage2_candidates:
        spatial_csv = get_spatial_csv(seed)
        res = run_single_experiment(adapter, seed, p, args.stage2_epochs, "stage2", spatial_csv)
        if res:
            res["validated"] = True # Mark as full run
            stage2_results.append(res)
            
    if stage2_results:
        df2 = pd.DataFrame(stage2_results)
        # Final Summary
        final_df = pd.concat([df1, df2], ignore_index=True)
        out_path = os.path.join(SWEEP_ROOT, "power_sweep_summary.csv")
        final_df.to_csv(out_path, index=False)
        print(f"\nFinal Summary saved: {out_path}")
        print(df2.to_string())
        
        sys.exit(0)
    else:
        print("Stage 2 failed.")
        sys.exit(2)

if __name__ == "__main__":
    main()
