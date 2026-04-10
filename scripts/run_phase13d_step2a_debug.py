
import os
import sys
import argparse
import traceback

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configuration
CONFIG = {
    "dataset": "seed1",
    "seeds": [0],
    "bands_mode": "all5_timecat",
    "band_norm_mode": "per_band_global_z", 
    "use_band_gate": False, 
    "mvp1_guided_cov": False, 
    "use_roi_pooling": True, # ROI Pooling On
    "mvp1_attn_power": 0.0,
    "epochs": 1, # DEBUG: Just 1 epoch
    "batch_size": 8,
    "spd_eps": 0.001,
    "output_root": "promoted_results/phase13d/step2a"
}

def run_worker(seed, tag):
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
            self.use_band_gate = CONFIG['use_band_gate']
            self.use_roi_pooling = CONFIG['use_roi_pooling']
            self.mvp1_guided_cov = CONFIG['mvp1_guided_cov']
            self.mvp1_attn_power = CONFIG['mvp1_attn_power']
            self.epochs = CONFIG['epochs']
            self.batch_size = CONFIG['batch_size']
            self.spd_eps = CONFIG['spd_eps']
            self.torch_device = "cuda"
            self.metrics_csv = None 
            self.is_val = False
            self.dcnet_ckpt = None 

    args = Args()
    
    # 2. Output
    out_dir = f"{CONFIG['output_root']}/{args.dataset}/seed{seed}/{tag}/manifold"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"[Worker] Starting Seed={seed} Tag={tag} Mode=ROI Pooling (Debug 1 Ep)")
    print(f"Output: {out_dir}")
    
    # Checkpoint Dir
    rel_path_dir = f"phase13d/step2a/{args.dataset}/seed{seed}/{tag}/manifold"
    ckpt_dir = os.path.join("experiments/checkpoints", rel_path_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 3. Load Data
    adapter = get_adapter(args.dataset)
    folds = adapter.get_manifold_trial_folds()
    
    # 4. Run
    runner = ManifoldDeepRunner(args, num_classes=3)
    
    # Fold Name = Suffix for outputs
    rel_path = f"{rel_path_dir}/report"
    res = runner.fit_predict(folds['fold1'], fold_name=rel_path)
    
    print(f"[Worker] Finished Seed={seed}")
    return res

if __name__ == "__main__":
    try:
        run_worker(0, "roi_manifold_only")
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
