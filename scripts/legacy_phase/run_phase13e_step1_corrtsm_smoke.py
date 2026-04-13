
import os
import sys
import torch
import numpy as np
import random
import argparse
import json
import traceback

# Setup Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from runners.manifold_deep_runner import ManifoldDeepRunner
from datasets.adapters import get_adapter

CONFIG = {
    # Data
    "dataset": "seed1",
    "bands_mode": "all5_timecat", # T ~ 120
    "band_norm_mode": "per_band_global_z", 
    "split_mode": "trial_80_20",
    "audit_key": "subject_session_trial",
    
    # Model
    "matrix_mode": "corr",            # NEW
    "use_roi_pooling": False,         # 62 channels
    "guided": False,
    "gate": False,
    
    # Training
    "epochs": 10,
    "batch_size": 8,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "spd_eps": 1e-3,
    
    # Paths
    "rel_path_base": "phase13e/step1/seed1/seed0/corr_tsm_manifold_only/manifold"
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_worker(seed):
    print(f"--- Starting Phase 13E Step 1 Smoke (Seed {seed}) ---")
    set_seed(seed)
    
    class Args:
        pass
    args = Args()
    
    # Map Config
    args.dataset_name = CONFIG['dataset']
    # args.bands_mode etc needed for dataset adapter? 
    # get_adapter(dataset_name) handles defaults usually, or we pass params to get_adapter if generic.
    # Looking at run_phase13d_step2a_debug.py, it constructs args then calls get_adapter(args.dataset).
    # wait, get_adapter(args.dataset) in debug script line 65 just takes string.
    # The adapter internal logic might rely on global constants or defaults. 
    # But ManifoldDeepRunner needs args.bands_mode.
    
    args.dataset = CONFIG['dataset'] # For consistency
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
    
    # Output Dir
    out_dir = f"promoted_results/{CONFIG['rel_path_base']}"
    os.makedirs(out_dir, exist_ok=True)
    args.metrics_csv = os.path.join(out_dir, "manifold_window_pred.csv")
    
    # Checkpoint Dir (Dummy)
    ckpt_dir = os.path.join("experiments/checkpoints", CONFIG['rel_path_base'])
    os.makedirs(ckpt_dir, exist_ok=True)
    args.dcnet_ckpt = None
    
    # Load Data
    print(f"Loading Adapter for {args.dataset}...")
    adapter = get_adapter(args.dataset)
    # The adapter uses 'all5_timecat' by default for SEED usually? 
    # Or we need to ensure adapter returns what we expect. 
    # verify adapter uses CONFIG['bands_mode'].
    # The debug script didn't pass bands_mode to get_adapter, but passed args to Runner.
    # Assumption: Adapter returns raw trials (list of ndarrays) and Runner performs stacking/windowing?
    # Actually ManifoldDeepRunner.fit_predict(fold) takes data.
    # Adapter.get_manifold_trial_folds() returns splits.
    
    folds = adapter.get_manifold_trial_folds()
    print("Data Loaded.")
    
    # Run
    runner = ManifoldDeepRunner(args, num_classes=3)
    
    # fold_name for report generation path
    # Runner saves to: promoted_results/{fold_name}_diagnostics.json
    # We want: promoted_results/{CONFIG['rel_path_base']}/report_diagnostics.json
    # So fold_name = f"{CONFIG['rel_path_base']}/report"
    
    fold_name = f"{CONFIG['rel_path_base']}/report"
    
    # We use 'fold1' as standard single fold for this smoke
    res = runner.fit_predict(folds['fold1'], fold_name=fold_name)
    
    print("Smoke Run Finished.")
    
    # Generate Report
    print("Generating Report...")
    # metrics_json = .../report_metrics.json (saved by runner)
    # diag_json = .../report_diagnostics.json
    # probe_json = ... ? Runner saves probes to report_probe_*.json? 
    # We need to verify what probes are saved.
    
    # Command to gen report
    # python scripts/analysis/gen_single_run_report.py --metrics ... --diagnostics ... --out ...
    
    import subprocess
    cmd = [
        sys.executable,
        "scripts/analysis/gen_single_run_report.py",
        "--metrics", f"promoted_results/{fold_name}_metrics.json",
        "--diagnostics", f"promoted_results/{fold_name}_diagnostics.json",
        "--out", f"promoted_results/{CONFIG['rel_path_base']}/SINGLE_RUN_REPORT.md"
    ]
    subprocess.check_call(cmd)
    print("Report Generated.")


if __name__ == "__main__":
    try:
        run_worker(0)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
