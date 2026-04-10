
import sys
import os
import torch
import numpy as np
import argparse

# Add project root to path
sys.path.append(os.getcwd())

from datasets.adapters import Seed1Adapter
from runners.manifold_deep_runner import ManifoldDeepRunner
from runners.spatial_dcnet_torch import DCNetTorch # Ensure importable

def run_mvp1():
    print("=== Phase 10: DCNet-Guided Manifold (MVP1) ===")
    
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window_len", type=int, default=24)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--guided", action="store_true")
    parser.add_argument("--power", type=float, default=0.5)
    args = parser.parse_args()
    
    # Set manual args for runner
    args.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    args.mvp1_guided_cov = args.guided
    args.mvp1_attn_method = "grad"
    args.mvp1_attn_power = args.power
    
    # Checkpoint Path (Refactored DCNet)
    args.dcnet_ckpt = f"experiments/checkpoints/seedv_spatial_torch_seed{args.seed}_refactor.pt"
    
    print(f"Config: Seed={args.seed}, Guided={args.guided}, Epochs={args.epochs}")
    print(f"DCNet Checkpoint: {args.dcnet_ckpt}")
    
    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load Data
    adapter = Seed1Adapter()
    folds = adapter.get_manifold_trial_folds()
    fold = folds['fold1']
    
    # Run
    runner = ManifoldDeepRunner(args, num_classes=3)
    if args.guided:
        mode_str = f"guided_p{args.power}"
    else:
        mode_str = "baseline"
    
    metrics = runner.fit_predict(fold, f"seed{args.seed}_mvp1_{mode_str}")
    
    print(f"Final Metrics: {metrics}")

if __name__ == "__main__":
    run_mvp1()
