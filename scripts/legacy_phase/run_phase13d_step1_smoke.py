
import os
import sys
import argparse
import subprocess
import json
import traceback

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Configuration
CONFIG = {
    "dataset": "seed1", # Only seed1 for now
    "seeds": [0], # Smoke test single seed
    "bands_mode": "all5_timecat",
    "band_norm_mode": "manual_per_band_time",
    "band_activation_mode": "input_gate_v1",
    "use_band_gate": True,
    "mvp1_guided_cov": True,
    "mvp1_attn_power": 0.5,
    "epochs": 10,  # Smoke test
    "batch_size": 8,
    "spd_eps": 0.001,
    "aggregation_method": "mean",
    "split_mode": "trial_80_20",
    "output_root": "promoted_results/phase13d/step1"
}

def run_worker(seed, tag, use_gate):
    """
    Run worker script for a single configuration.
    """
    import pandas as pd
    from runners.manifold_deep_runner import ManifoldDeepRunner
    from datasets.adapters import get_adapter

    # 1. Setup Args
    class Args:
        def __init__(self):
            # Base
            self.dataset = CONFIG['dataset']
            self.seed = seed
            self.bands_mode = CONFIG['bands_mode']
            self.band_norm_mode = CONFIG['band_norm_mode']
            self.band_activation_mode = CONFIG['band_activation_mode']
            self.use_band_gate = use_gate
            self.mvp1_guided_cov = CONFIG['mvp1_guided_cov']
            self.mvp1_attn_power = CONFIG['mvp1_attn_power']
            self.epochs = CONFIG['epochs']
            self.batch_size = CONFIG['batch_size']
            self.spd_eps = CONFIG['spd_eps']
            self.torch_device = "cuda"
            self.metrics_csv = None # Handled manually
            self.is_val = False
            self.dcnet_ckpt = "experiments/checkpoints/seedv_spatial_torch_seed0_refactor.pt" # Locked template

    args = Args()
    
    # 2. Prepare Output Dir
    out_dir = f"{CONFIG['output_root']}/{args.dataset}/seed{seed}/{tag}/manifold"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"[Worker] Starting Seed={seed} Tag={tag} Gate={use_gate}")
    print(f"Output: {out_dir}")
    
    # Ensure checkpoint dir exists matching rel_path structure
    rel_path_dir = f"phase13d/step1/{args.dataset}/seed{seed}/{tag}/manifold"
    ckpt_dir = os.path.join("experiments/checkpoints", rel_path_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 3. Load Data
    adapter = get_adapter(args.dataset)
    folds = adapter.get_manifold_trial_folds()
    
    # 4. Run Manifold Deep Runner
    runner = ManifoldDeepRunner(args, num_classes=3)
    
    # We run 'fit_predict' which handles splitting and training
    # We pass the fold name to save results in the right dir
    # Runner saves to "promoted_results/{fold_name}_metrics.json" etc.
    # We need to hack the path prefix inside the runner or rely on relative paths?
    # Runner code uses: f"promoted_results/{fold_name}_metrics.json"
    # So we set fold_name to the relative path from project root 'promoted_results/'?
    # No, Runner: `with open(f"promoted_results/{fold_name}_metrics.json", "w")`
    # So if we want `promoted_results/phase13d/step1/...`, fold_name should be `phase13d/step1/...`
    
    rel_path = f"phase13d/step1/{args.dataset}/seed{seed}/{tag}/manifold/report" # Using 'report' as basename
    # Check if directory exists (runner assumes `promoted_results` exists, but subdirs?)
    # Runner doesn't create dirs. We created `out_dir` which is `promoted_results/...`
    # So we need to ensure the path corresponds.
    
    # Actually, let's just use `rel_path` without `promoted_results/` prefix if runner adds it?
    # Runner: `open(f"promoted_results/{fold_name}_metrics.json")`
    # So YES, `fold_name` is the suffix.
    res = runner.fit_predict(folds['fold1'], fold_name=rel_path)
    
    print(f"[Worker] Finished Seed={seed} Tag={tag}")
    return res

if __name__ == "__main__":
    try:
        run_worker(0, "norm_inputgate", True)
    except Exception as e:
        traceback.print_exc()
        # Save failure log
        fail_path = f"{CONFIG['output_root']}/{CONFIG['dataset']}/seed0/norm_inputgate/FAILURE_LOG.txt"
        os.makedirs(os.path.dirname(fail_path), exist_ok=True)
        with open(fail_path, "w") as f:
            f.write(traceback.format_exc())
        sys.exit(1)
