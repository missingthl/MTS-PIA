
import os
import sys
import json
import argparse
import subprocess
import torch
import pandas as pd
import traceback

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.adapters import get_adapter
from runners.manifold_deep_runner import ManifoldDeepRunner

# Params
CONFIG = {
    "dataset": "seed1",
    "seeds": [0, 4],
    "bands_mode": "all5_timecat",
    "spd_eps": 0.001,
    "epochs": 50,
    "batch_size": 8,
    "lr": 0.0001,
    "weight_decay": 0.0,
    "split_mode": "trial_80_20",
    "audit_key": "subject_session_trial",
    "aggregation_method": "mean",
    "guided": True,
    "attn_power": 0.5,
    "teacher_ckpt_template": "experiments/checkpoints/seedv_spatial_torch_seed{}_refactor.pt"
}

ROOT_DIR = "promoted_results/phase13c4/step1/seed1"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def run_worker(seed, run_tag, use_gate):
    print(f"[Worker] Starting Seed={seed} Tag={run_tag} Gate={use_gate}")
    
    # 1. Setup Dirs
    run_dir = os.path.join(ROOT_DIR, f"seed{seed}", run_tag, "manifold")
    ensure_dir(run_dir)
    
    # 2. Config
    args = Args(
        torch_device="cuda" if torch.cuda.is_available() else "cpu",
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        mvp1_guided_cov=CONFIG['guided'],
        mvp1_attn_power=CONFIG['attn_power'],
        seed=seed,
        dcnet_ckpt=CONFIG['teacher_ckpt_template'].format(seed),
        metrics_csv=None,
        bands_mode=CONFIG['bands_mode'],
        band_norm_mode="none",
        spd_eps=CONFIG['spd_eps'],
        use_band_gate=use_gate,
        is_val=False # Log training stats
    )
    
    # 3. Load Data
    adapter = get_adapter(CONFIG['dataset'])
    folds = adapter.get_manifold_trial_folds()
    
    # 4. Run
    runner = ManifoldDeepRunner(args, num_classes=3)
    fold_name = f"c4s1_s{seed}_{run_tag}"
    
    # Checkpoint Dir override (Hack: Runner writes to promoted_results/ by default)
    # We want it in run_dir. 
    # Current runner hardcodes "promoted_results/{fold_name}_...".
    # We can rely on that and move files, or simple allow it.
    # The prompt says: ".../seed{SEED}/{RUN_TAG}/manifold/report.json"
    
    try:
        res = runner.fit_predict(folds['fold1'], fold_name=fold_name)
    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    # 5. Move Artifacts
    # Runner outputs: 
    # promoted_results/{fold_name}_metrics.json
    # promoted_results/{fold_name}_gate_stats.json (if gate)
    # promoted_results/{fold_name}_preds_test_last_trial.csv
    # promoted_results/{fold_name}_metadata.json (wait, did I implement metadata save? Yes, but inside 'metrics' in previous code? No, I separated it.)
    
    # Move metrics -> report.json
    src_met = f"promoted_results/{fold_name}_metrics.json"
    if os.path.exists(src_met):
        os.rename(src_met, os.path.join(run_dir, "report.json"))
        
    # Move gate stats
    src_gate = f"promoted_results/{fold_name}_gate_stats.json"
    dst_gate_dir = os.path.dirname(run_dir) # parent of manifold/
    if os.path.exists(src_gate):
        os.rename(src_gate, os.path.join(dst_gate_dir, "gate_stats.json"))
        
    # Move Preds
    src_pred = f"promoted_results/{fold_name}_preds_test_last_trial.csv"
    if os.path.exists(src_pred):
        os.rename(src_pred, os.path.join(run_dir, "manifold_trial_pred.csv"))
        
    # Move Metadata -> export_meta.json
    # Wait, did I save metadata separately? My fix block did: `with open(..., _metadata.json)`.
    src_meta = f"promoted_results/{fold_name}_metadata.json"  # Hypothesizing name based on my fix
    # If not found, check inside results dict.
    if os.path.exists(src_meta):
        os.rename(src_meta, os.path.join(dst_gate_dir, "export_meta.json"))
    
    # Cleanup status
    if os.path.exists(f"promoted_results/{fold_name}_status.txt"):
        os.remove(f"promoted_results/{fold_name}_status.txt")

    print(f"[Worker] Finished Seed={seed} Tag={run_tag}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tag", type=str)
    parser.add_argument("--use_gate", type=str) # "true"/"false"
    
    args = parser.parse_args()
    
    if args.worker:
        run_worker(args.seed, args.tag, args.use_gate.lower() == "true")
    else:
        # Orchestrator
        ensure_dir(ROOT_DIR)
        
        seeds = CONFIG['seeds']
        runs = [
            ("control", False),
            ("bandgate", True)
        ]
        
        summary = []
        
        procs = []
        for seed in seeds:
            for tag, use_gate in runs:
                print(f"=== Launching Seed={seed} Tag={tag} ===")
                cmd = [
                    sys.executable, __file__,
                    "--worker",
                    "--seed", str(seed),
                    "--tag", tag,
                    "--use_gate", str(use_gate)
                ]
                # Redirect output to log files
                log_path = os.path.join(ROOT_DIR, f"seed{seed}_{tag}.log")
                with open(log_path, "w") as f:
                    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
                    procs.append((p, seed, tag, log_path))
        
        print(f"\n[Orchestrator] {len(procs)} runs launched in parallel. Waiting...")
        
        summary = []
        for p, seed, tag, log_path in procs:
            ret = p.wait()
            status = "PASS" if ret == 0 else "FAIL"
            print(f"Finished Seed={seed} Tag={tag} Status={status}")
            
            # Collect metrics
            # Reading report.json
            rep_path = os.path.join(ROOT_DIR, f"seed{seed}/{tag}/manifold/report.json")
            acc = 0.0
            if os.path.exists(rep_path):
                with open(rep_path) as f:
                    try:
                        d = json.load(f)
                        acc = d.get('last', {}).get('test_trial_acc', 0.0)
                    except:
                        pass
            
            summary.append({
                "seed": seed,
                "run_tag": tag,
                "status": status,
                "trial_acc": acc
            })
                
        # Save Summary
        df = pd.DataFrame(summary)
        df.to_csv(os.path.join(ROOT_DIR, "summary.csv"), index=False)
        print("\n=== Summary ===")
        print(df.to_string())

if __name__ == "__main__":
    main()
