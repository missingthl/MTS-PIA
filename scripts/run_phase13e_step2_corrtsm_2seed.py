
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runners.manifold_deep_runner import ManifoldDeepRunner
from datasets.adapters import get_adapter

CONFIG = {
    "dataset": "seed1",
    "seeds": [0, 4],
    "bands_mode": "all5_timecat",
    "band_norm_mode": "per_band_global_z",
    "split_mode": "trial_80_20",
    "audit_key": "subject_session_trial",
    
    # Model
    "matrix_mode": "corr",
    "use_roi_pooling": False,
    "guided": False,
    "gate": False,
    "spd_eps": 1e-3,
    "epochs": 10,
    "batch_size": 8,
    "lr": 1e-4, # Verified in Step 1
    
    # Base Path
    "rel_path_base_fmt": "phase13e/step2/seed1/seed{}/corr_tsm_manifold_only/manifold"
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_worker(seed):
    print(f"\n=== Starting Phase 13E Step 2 Validation (Seed {seed}) ===")
    set_seed(seed)
    
    # 1. Args
    class Args:
        pass
    args = Args()
    
    args.dataset_name = CONFIG['dataset']
    args.dataset = CONFIG['dataset'] # For consistency
    args.seed = seed # Explicitly pass seed for splitting logic
    
    args.bands_mode = CONFIG['bands_mode']
    args.band_norm_mode = CONFIG['band_norm_mode']
    args.matrix_mode = CONFIG['matrix_mode']
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
    args.metrics_csv = os.path.join(out_dir, "manifold_window_pred.csv")
    
    # Dummy Ckpt
    ckpt_dir = os.path.join("experiments/checkpoints", rel_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    args.dcnet_ckpt = None
    
    # 2. Data
    adapter = get_adapter(args.dataset)
    folds = adapter.get_manifold_trial_folds()
    
    # 3. Run
    runner = ManifoldDeepRunner(args, num_classes=3)
    fold_name = f"{rel_path}/report"
    
    # This writes:
    # promoted_results/{fold_name}_metrics.json
    # promoted_results/{fold_name}_diagnostics.json
    runner.fit_predict(folds['fold1'], fold_name=fold_name)
    
    # 4. Generate Report
    print(f"Generating Report for Seed {seed}...")
    cmd = [
        sys.executable,
        "scripts/gen_single_run_report.py",
        "--root", out_dir # Root folder
    ]
    # Note: gen_single_run_report logic for finding files
    # It constructs "report_metrics.json" inside root?
    # Wait, Runner saves to `promoted_results/{fold_name}_metrics.json`.
    # fold_name = "phase13e/step2/seed1/seed0/corr_tsm_manifold_only/manifold/report"
    # So file is "promoted_results/phase13e/.../manifold/report_metrics.json"
    # Root passed to script should be "promoted_results/phase13e/.../manifold"
    # And script looks for "report_metrics.json" inside root.
    # This matches.
    
    subprocess.check_call(cmd)
    print(f"Report Generated for Seed {seed}.")

def main():
    for seed in CONFIG['seeds']:
        try:
            run_worker(seed)
        except Exception as e:
            traceback.print_exc()
            print(f"Seed {seed} Failed.")
            sys.exit(1)

if __name__ == "__main__":
    main()
