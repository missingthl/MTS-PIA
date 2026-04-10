
import argparse
import sys
import os
import pandas as pd
import json
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runners.manifold_deep_runner import ManifoldDeepRunner
from datasets.adapters import TrialFoldData

def run_experiment(seed, guided, power):
    print(f"\n=== Running Seed={seed} Guided={guided} Power={power} ===")
    
    class Args:
        pass
    
    args = Args()
    args.seed = seed
    args.epochs = 50
    args.batch_size = 8
    args.torch_device = "cuda" # Assume GPU
    # MVP1 Args
    args.mvp1_guided_cov = guided
    args.mvp1_attn_method = "grad"
    args.mvp1_attn_power = power
    
    # Checkpoint Path (Refactored DCNet)
    args.dcnet_ckpt = f"experiments/checkpoints/seedv_spatial_torch_seed{seed}_refactor.pt"
    args.metrics_csv = None # Handled locally
    
    # Load Data
    # Assuming SEED1 subject-dependent
    # from pia_unified_demo import load_data <- Removed
    
    # We need to replicate load_data logic or just manual load
    # Quick fix: Use Adapters directly? 
    # Actually manifold runner usually takes a 'fold' object.
    # Let's recreate the data loading logic briefly.
    
    # Data params (Standard)
    # T=24, Hop=12
    root_dir = "data/SEED/SEED_EEG/ExtractedFeatures_1s"
    
    # We need to use the standard loading flow to ensure consistency
    # Let's rely on cached processing if possible? 
    # Or just re-implement the Adapter call.
    # Data Loading
    from datasets.adapters import Seed1Adapter
    adapter = Seed1Adapter()
    
    # Use standard windowing if controlled by adapter or custom
    # Seed1Adapter has get_manifold_trial_folds which returns pre-processed folds
    # But we might need custom T_window?
    # Usually Seed1Adapter uses default config.
    # Let's check get_manifold_trial_folds arguments if possible, or assume defaults.
    # pia_unified_demo uses adapter.create_trial_fold for custom T.
    # Seed1Adapter might not have create_trial_fold implemented?
    # run_mvp1_guided.py used adapter.get_manifold_trial_folds().
    # Let's stick to what worked: get_manifold_trial_folds().
    # But run_mvp1_guided had args for window_len/stride but didn't pass them to adapter!
    # It seems Seed1Adapter uses fixed windowing? 
    # Let's check datasets/adapters.py for Seed1Adapter implementation details.
    
    folds = adapter.get_manifold_trial_folds()
    fold = folds['fold1']
    
    runner = ManifoldDeepRunner(args, num_classes=3)
    mode_str = "guided" if guided else "baseline"
    run_name = f"seed{seed}_mvp1_{mode_str}"
    
    metrics = runner.fit_predict(fold, run_name)
    return metrics

def load_or_run(seed, guided, power):
    mode_str = "guided" if guided else "baseline"
    run_name = f"seed{seed}_mvp1_{mode_str}"
    metrics_path = f"promoted_results/{run_name}_metrics.json"
    
    if os.path.exists(metrics_path):
        print(f"[{run_name}] Found existing metrics. Loading...")
        with open(metrics_path, "r") as f:
            return json.load(f)
    else:
        return run_experiment(seed, guided, power)

def main():
    if not os.path.exists("promoted_results"):
        os.makedirs("promoted_results")
        
    seeds = [0, 1, 2]
    results_list = []
    
    for seed in seeds:
        # Baseline
        print(f"--- Seed {seed} Baseline ---")
        m_base = load_or_run(seed, guided=False, power=0.0)
        
        # Guided (p=2.0)
        print(f"--- Seed {seed} Guided ---")
        m_guide = load_or_run(seed, guided=True, power=2.0)
        
        # Record
        row = {
            'seed': seed,
            'base_best_epoch': m_base['best']['epoch'],
            'base_best_win_acc': m_base['best']['test_win_acc'],
            'base_best_trial_acc': m_base['best']['test_trial_acc'],
            'base_best_wc': m_base['best']['test_wrong_conf'],
            'base_last_trial_acc': m_base['last']['test_trial_acc'],
            
            'guide_best_epoch': m_guide['best']['epoch'],
            'guide_best_win_acc': m_guide['best']['test_win_acc'],
            'guide_best_trial_acc': m_guide['best']['test_trial_acc'],
            'guide_best_wc': m_guide['best']['test_wrong_conf'],
            'guide_last_trial_acc': m_guide['last']['test_trial_acc'],
        }
        
        # Deltas
        row['delta_trial_acc'] = row['guide_best_trial_acc'] - row['base_best_trial_acc']
        
        results_list.append(row)
        
        # Incremental Save
        pd.DataFrame(results_list).to_csv("promoted_results/summary_table.csv", index=False)

if __name__ == "__main__":
    main()
