
import os
import sys
import torch
import numpy as np
import random
import argparse
import json
import traceback
import subprocess

# Setup Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from runners.trial_prototype_mdm_runner import TrialPrototypeMDMRunner
from datasets.adapters import get_adapter

CONFIG = {
    "dataset": "seed1",
    "bands_mode": "all5_timecat",
    "band_norm_mode": "per_band_global_z",
    "split_mode": "trial_80_20",
    "audit_key": "subject_session_trial",
    
    # Model
    "matrix_mode": "corr",
    "subject_centering": False,
    "global_centering": True,
    "use_roi_pooling": False,
    "guided": False,
    "gate": False,
    "spd_eps": 1e-3,
    "epochs": 1, 
    "batch_size": 32,
    "lr": 1e-4,
    
    # Base Path
    "rel_path_base_fmt": "phase13e/step6/seed1/seed{}/trial_proto_mdm_logeuclid/manifold"
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_worker(seed):
    print(f"\n=== Starting Phase 13E Step 6 Trial Proto MDM (Seed {seed}) ===")
    set_seed(seed)
    
    # 1. Args
    class Args:
        pass
    args = Args()
    
    args.dataset_name = CONFIG['dataset']
    args.dataset = CONFIG['dataset'] 
    args.seed = seed
    
    args.bands_mode = CONFIG['bands_mode']
    args.band_norm_mode = CONFIG['band_norm_mode']
    args.matrix_mode = CONFIG['matrix_mode']
    args.subject_centering = CONFIG['subject_centering']
    args.global_centering = CONFIG['global_centering']
    args.mvp1_guided_cov = CONFIG['guided']
    args.use_band_gate = CONFIG['gate']
    args.use_roi_pooling = CONFIG['use_roi_pooling']
    args.epochs = CONFIG['epochs']
    args.batch_size = CONFIG['batch_size']
    args.spd_eps = CONFIG['spd_eps']
    args.lr = CONFIG['lr']
    args.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    rel_path = CONFIG['rel_path_base_fmt'].format(seed)
    out_dir = f"promoted_results/{rel_path}"
    os.makedirs(out_dir, exist_ok=True)
    args.metrics_csv = os.path.join(out_dir, "manifold_window_pred.csv") # Placeholder
    
    # Dummy Ckpt
    ckpt_dir = os.path.join("experiments/checkpoints", rel_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    args.dcnet_ckpt = None
    
    # 2. Data
    adapter = get_adapter(args.dataset)
    folds = adapter.get_manifold_trial_folds()
    
    # 3. Run
    runner = TrialPrototypeMDMRunner(args, num_classes=3)
    fold_name = f"{rel_path}/report"
    
    runner.fit_predict(folds['fold1'], fold_name=fold_name)
    
    # 4. Generate Report (Placeholder script, maybe reuse step 5 but update path?)
    # We will generate report separately or update script?
    # User asked for "gen_single_run_report" usage.
    # But that script expects window-level metrics.json structure usually.
    # TrialPrototypeRunner saves `trial_acc` in report_metrics.json.
    # Standard report generator might fail or show N/A for window stuff.
    # Let's try running it.
    
    print(f"Generating Report for Seed {seed}...")
    cmd = [
        sys.executable,
        "scripts/analysis/gen_single_run_report.py",
        "--root", out_dir
    ]
    
    try:
        subprocess.check_call(cmd)
        print(f"Report Generated for Seed {seed}.")
    except Exception as e:
        print(f"Report generation failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    
    try:
        run_worker(args.seed)
    except Exception as e:
        traceback.print_exc()
        print(f"Seed {args.seed} Failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
