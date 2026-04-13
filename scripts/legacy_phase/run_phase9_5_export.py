
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runners.manifold_multiband_runner import MultiBandRunner
from datasets.adapters import Seed1Adapter

class MockArgs:
    def __init__(self, seed, mode, epochs, batch_size):
        self.seed = seed
        self.dataset = 'seed'
        self.model = 'spdnet_multiband'
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = 1e-4
        self.weight_decay = 1e-3
        self.torch_device = "cuda"
        self.out_prefix = f"experiments/phase9_5_multiband/seed{seed}"
        self.check_inputs = True

def run_seed(seed, mode_str):
    print(f"\n=== Running Phase 9.5 Manifold5 Export | Seed {seed} | Mode {mode_str} ===")
    
    # Mode config
    if mode_str == 'sanity':
        bs = 1 # Variable trial length requires BS=1
        eps = 5
    else: # report
        bs = 1
        eps = 50
        
    args = MockArgs(seed, mode_str, eps, bs)
    
    # Data
    adapter = Seed1Adapter()
    folds = adapter.get_spatial_folds_for_cnn(
        seed_de_root="data/SEED/SEED_EEG/ExtractedFeatures_1s",
        seed_de_var="de_LDS",
        seed_de_mode="official",
        seed_freeze_align=True
    )
    
    # Run
    # official mode usually has fold1
    fold = folds['fold1']
    runner = MultiBandRunner(args)
    runner.fit_predict(fold, 'fold1', seed)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['sanity', 'report'], default='sanity', help="sanity (fast) or report (stable)")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma separated seeds")
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    
    for seed in seeds:
        run_seed(seed, args.mode)
