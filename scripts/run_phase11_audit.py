
import sys
import os
import pandas as pd
import json
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runners.manifold_deep_runner import ManifoldDeepRunner
from datasets.adapters import Seed1Adapter

def load_or_run(seed, guided, power):
    mode_str = "guided" if guided else "baseline"
    run_name = f"seed{seed}_phase11_{mode_str}_p{power:.1f}"
    
    # Check all artifacts
    metrics_path = f"promoted_results/{run_name}_metrics.json"
    split_path = f"promoted_results/{run_name}_split.json"
    status_path = f"promoted_results/{run_name}_status.txt"
    
    if os.path.exists(metrics_path) and os.path.exists(split_path) and os.path.exists(status_path):
        print(f"[{run_name}] Found existing artifacts. Loading...")
        with open(metrics_path, "r") as f:
            return json.load(f)
    
    print(f"\n=== Running {run_name} (Seed={seed} Guided={guided} Power={power}) ===")
    
    class Args:
        pass
    
    args = Args()
    args.seed = seed
    args.epochs = 50
    args.batch_size = 8
    args.torch_device = "cuda"
    
    # MVP1 Args
    args.mvp1_guided_cov = guided
    args.mvp1_attn_method = "grad"
    args.mvp1_attn_power = power
    
    # Checkpoint Path (Refactored DCNet)
    if guided:
        args.dcnet_ckpt = f"experiments/checkpoints/seedv_spatial_torch_seed{seed}_refactor.pt"
    else:
        args.dcnet_ckpt = None
        
    args.metrics_csv = None 
    
    # Load Data (Seed1Adapter)
    adapter = Seed1Adapter()
    folds = adapter.get_manifold_trial_folds()
    fold = folds['fold1']
    
    runner = ManifoldDeepRunner(args, num_classes=3)
    results = runner.fit_predict(fold, run_name)
    
    return results

def main():
    if not os.path.exists("promoted_results"):
        os.makedirs("promoted_results")
        
    runs_to_exec = []
    
    # Main Results: Seeds 0-4, Base + Guided(2.0)
    for seed in [0, 1, 2, 3, 4]:
        runs_to_exec.append({"seed": seed, "guided": False, "power": 0.0, "type": "main"})
        runs_to_exec.append({"seed": seed, "guided": True, "power": 2.0, "type": "main"})
        
    # Ablation: Seeds 0-1, Powers [0.5, 1.0, 4.0]
    for seed in [0, 1]:
        for p in [0.5, 1.0, 4.0]:
            runs_to_exec.append({"seed": seed, "guided": True, "power": p, "type": "ablation"})
            
    all_results = []
    audit_summary = {}
    
    for r in runs_to_exec:
        seed = r['seed']
        guided = r['guided']
        power = r['power']
        
        try:
            res = load_or_run(seed, guided, power)
            
            # Flatten for CSV
            row = {
                "seed": seed,
                "guided": guided,
                "power": power,
                "type": r['type'],
                "epoch_best": res['best']['epoch'],
                "win_acc_best": res['best']['test_win_acc'],
                "trial_acc_best": res['best']['test_trial_acc'],
                "trial_acc_last": res['last']['test_trial_acc'],
                "audit_passed": res.get('metadata', {}).get('audit_passed', False)
            }
            all_results.append(row)
            
            # Audit Summary Collection
            mode_str = "guided" if guided else "baseline"
            run_name = f"seed{seed}_phase11_{mode_str}_p{power:.1f}"
            split_path = f"promoted_results/{run_name}_split.json"
            if os.path.exists(split_path):
                with open(split_path, "r") as f:
                    split_data = json.load(f)
                    audit_summary[run_name] = {
                        "intersect": split_data.get("intersection_count", -1),
                        "sample": split_data.get("intersection_sample", [])
                    }
                    if split_data.get("intersection_count", 0) > 0:
                        print(f"CRITICAL: Run {run_name} failed audit!")
                        sys.exit(1)
                        
        except Exception as e:
            print(f"Error running {seed}-{guided}-{power}: {e}")
            sys.exit(1)

    # Save Aggregate
    df = pd.DataFrame(all_results)
    df.to_csv("promoted_results/summary_table_phase11.csv", index=False)
    print("Saved summary_table_phase11.csv")
    
    with open("promoted_results/audit_summary_seed0_4.json", "w") as f:
        json.dump(audit_summary, f, indent=2)
    print("Saved audit_summary_seed0_4.json")

if __name__ == "__main__":
    main()
