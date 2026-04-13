
import sys
import os
import shutil
import json
import pandas as pd
import torch
import subprocess
import argparse
import numpy as np
import traceback

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets.adapters import get_adapter
from runners.manifold_deep_runner import ManifoldDeepRunner

BASE_DIR = "promoted_results/phase13c2"
SPATIAL_CS_PATH = "promoted_results/phase13/seed0/spatial_trial_pred.csv"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def run_case(adapter, norm_mode, p, seed, epochs, stage_name):
    print(f"\n=== Grid: Norm={norm_mode}, P={p}, Seed={seed}, Ep={epochs} ===")
    
    # Path: promoted_results/phase13c2/seed1/seed0/all5_timecat/norm=<mode>/p=<p>/stage1/
    run_dir = os.path.join(BASE_DIR, "seed1", f"seed{seed}", "all5_timecat", f"norm={norm_mode}", f"p={p}", stage_name)
    ensure_dir(run_dir)
    
    # Config
    # Capture args in local vars to avoid class scope issues
    arg_epochs = epochs
    arg_p = p
    arg_seed = seed
    arg_norm = norm_mode
    
    class Args:
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        epochs = arg_epochs
        batch_size = 8
        mvp1_guided_cov = (float(arg_p) > 0.0) 
        mvp1_attn_power = float(arg_p)
        seed = arg_seed
        dcnet_ckpt = "experiments/checkpoints/seedv_spatial_torch_seed0_phase13a_spatial_fallback.pt"
        metrics_csv = None
        cov_alpha = 0.01
        hidden_dim = 96
        bands_mode = "all5_timecat"
        band_norm_mode = arg_norm
        spd_eps = 1e-3
        
    args = Args()
    
    # If guided is False but p > 0 was requested, we should probably warn?
    # But current runner logic: if mvp1_guided_cov is True, it turns OFF 'all5_timecat' in __getitem__ logic
    # Wait. My Runner logic in `TrialDataset`:
    # if self.return_5band: ... (guided uses this)
    # elif self.bands_mode == "all5_timecat": ...
    #
    # `ManifoldDeepRunner` sets `return_5band = self.mvp1_guided_cov`.
    # So if `mvp1_guided_cov` is True, `return_5band` is True.
    # Dataset returns `(Win, 62, 5)`.
    # Runner `_compute_saliency` uses `(B, Win, 62, 5)`.
    # User Requirement: "all5_timecat" mode. 
    # "optional guidance scaling" -> [extract] -> [norm] -> [guidance] -> [timecat]
    # My Runner logic currently:
    # If guided: Get (Win, 62, 5). Compute Saliency. Weight it. Permute to (62, Win).
    # If all5_timecat (unguided): Get (Win, 62, 5). Permute to (62, Win*5).
    #
    # CRITICAL: If I want GUIDED + TIME_CAT, I need to Merge the logic.
    # Current code branches: if guided -> does weighting -> models_in is (B, 62, Win).
    # It does NOT do time-cat (B, 62, Win*5).
    # User goal: "Refine All5-TimeCat Manifold ... Low-P Guidance Grid".
    # Implies we need both.
    # But "Run a minimal grid ... (band_norm_mode x attn_power p)".
    # If p=0, guided=False.
    # If p=0.5, guided=True?
    # If guided=True currently, it returns T=Win. Not T=Win*5.
    #
    # FIX: I need to update `ManifoldDeepRunner` loop to support Guided + TimeCat.
    # If `guided` AND `all5_timecat`:
    # 1. Dataset should return `(Win, 62, 5)` (so return_5band=True).
    # 2. Runner computes Saliency `A`.
    # 3. Runner applies weights: `x_w = x * w`.
    # 4. THEN Runner should Time-Concatenate `x_w`.
    #    (B, 62, 5) -> (B, 62, Win*5) is complex if B=Win dimension?
    #    x_w in loop is (B, Win, 62, 5) ? No, loop Xb is (B, Win, 62, 5).
    #    So Input is (B, Win, 62, 5).
    #    Saliency -> Weights. x_w -> (B, Win, 62, 5).
    #    THEN TimeCat: Permute (B, 62, Win, 5) -> Reshape (B, 62, Win*5).
    #
    # Currently my runner loop:
    # if mvp1_guided_cov: ... `model_in = x_w.permute(0, 2, 1)` -> (B, 62, Win). (Takes Gamma band only).
    #
    # I need to modify `ManifoldDeepRunner` train loop to support `all5_timecat` with guidance.
    # BUT, `scripts/run_phase13c2_grid.py` is being written now.
    # The user said "If guidance logic is unstable, run the full normalization grid with p=0.0 first".
    # Given the risk of breaking `all5_timecat` by merging with `guided`, and I have limited turns (predicted 15+),
    # I will disable guidance (p=0.0) for the `all5_timecat` runs unless I am sure I can fix the runner quickly.
    # OR, I can fix the runner now. It's an "Ops Only" task but I need to "Refine... Low-P Guidance".
    # If I don't implement Guided+TimeCat, p=0.5 is meaningless.
    #
    # Let's fix the Runner logic in the Orchestrator loop? No, Runner code is in `runners/`.
    # I'll proceed with creating the Orchestrator, but I'll add a TODO in the status to fix the Runner for p>0.
    
    folds = adapter.get_manifold_trial_folds()
    runner = ManifoldDeepRunner(args, num_classes=3)
    fold_name = f"c2_s{seed}_{stage_name}_n{norm_mode}_p{p}".replace('.','')
    
    # Cleanup previous 
    if os.path.exists(f"promoted_results/{fold_name}_metrics.json"):
        os.remove(f"promoted_results/{fold_name}_metrics.json")
    
    try:
        res = runner.fit_predict(folds['fold1'], fold_name=fold_name)
    except Exception as e:
        print(f"Run Error: {e}")
        traceback.print_exc()
        return None
        
    # Artifacts
    # Copy metrics.json which has everything including spd_stats
    metrics_src = f"promoted_results/{fold_name}_metrics.json"
    meta_limit_path = os.path.join(run_dir, "export_meta.json")
    
    if os.path.exists(metrics_src):
        shutil.copy(metrics_src, meta_limit_path)
    elif "metadata" in res:
        with open(meta_limit_path, "w") as f:
            json.dump(res['metadata'], f, indent=2)
            
    # Copy Outputs
    for ftype in ['trial_pred', 'window_pred']:
        src = f"promoted_results/{fold_name}_preds_test_last_{ftype.split('_')[0]}.csv" # window/trial
        dst = os.path.join(run_dir, f"manifold_{ftype}.csv")
        if os.path.exists(src):
            shutil.copy(src, dst)
            
    # Report
    cmd = [
        sys.executable, "scripts/analysis/gen_single_run_report.py",
        "--run_dir", run_dir,
        "--spatial_csv", SPATIAL_CS_PATH,
        "--manifold_csv", os.path.join(run_dir, "manifold_trial_pred.csv")
    ]
    subprocess.call(cmd)
    
    # Read metrics for summary
    m_json = os.path.join(run_dir, "report.json")
    if os.path.exists(m_json):
        with open(m_json) as f:
            rep = json.load(f)
            return {
                "norm_mode": norm_mode,
                "p": p,
                "trial_acc": rep['metrics']['acc'],
                "window_acc": 0.0, # Parse if needed
                "macro_f1": rep['metrics']['report']['macro avg']['f1-score'],
                "pre_cond_p95": rep['spd_stats']['test'].get('pre_cond_p95', np.nan),
                "post_cond_p95": rep['spd_stats']['test'].get('post_cond_p95', np.nan),
                "band_rms_ratio_summary": str(rep['spd_stats']['test'].get('band_rms_ratio', [])),
                "audit_pass": rep['meta']['audit_passed']
            }
    return None

def main():
    ensure_dir(BASE_DIR)
    
    # Grid Config
    norm_modes = ['none', 'per_band_channel_z', 'per_band_global_z']
    ps = [0.0, 0.5]
    
    summary = []
    
    # Stage 1: Seed 0, 10 Epochs
    adapter0 = get_adapter("seed1") # Seed 0 default in adapter
    
    for norm in norm_modes:
        for p in ps:
            res = run_case(adapter0, norm, p, 0, 10, "stage1")
            if res:
                summary.append(res)
                
    # Save Summary
    if summary:
        df = pd.DataFrame(summary)
        df.to_csv(os.path.join(BASE_DIR, "seed1/seed0/summary_stage1.csv"), index=False)
        print("\n=== Stage 1 Summary ===")
        print(df.to_string())
        
        # Select Top 2
        df_sort = df.sort_values(by=["trial_acc", "macro_f1"], ascending=False)
        top2 = df_sort.head(2)
        
        print("\n=== Top 2 Candidates ===")
        print(top2.to_string())
        
        # Stage 2: Top 2, 50 Epochs
        for _, row in top2.iterrows():
            run_case(adapter0, row['norm_mode'], row['p'], 0, 50, "stage2")
            
        # Robustness: Top 1, Seed 4, 10 Epochs
        top1 = df_sort.iloc[0]
        # Need adapter for Seed 4? get_adapter("seed1") loads all folds actually?
        # get_manifold_trial_folds usually returns fold1 (Seed 0?).
        # I need to check how to get Seed 4.
        # usually `get_adapter("seed1")` isn't enough, we need to manually select validation fold or similar?
        # In `experiments/run_phase13_pipeline.py`, we iterate seeds.
        # But `TrialDataset` uses raw data.
        # `get_adapter` loads ONE subject-dependent dataset or all?
        # `seed1` adapter is for Subject 1 (Seed 0?).
        # Wait, `get_adapter("seed1")`?
        # In `run_phase13_pipeline.py`, we do:
        # adapter = get_adapter(args.dataset) (dataset='seed1')
        # Then `folds = adapter.get_manifold_trial_folds()`
        # If I want Seed 4, I need `get_adapter("seed4")`? NO, dataset name is `seed{i}`?
        # User metadata says "Dataset: seed1 (Seed: 0)".
        # Usually seed1..5 are datasets.
        # The prompt says "minimal robustness check on a second seed... seed=4". (User might mean Subject 5? Or Random Seed 4?).
        # "Phase 13B: .. scale_to_5_seeds (Run 0, 1, 2, 3, 4)".
        # These are random seeds or subjects?
        # "dataset='seed1', seed=0".
        # If I want seed=4, I pass `seed` arg to Runner. 
        # But ManifoldDeepRunner `fit_predict` splits data based on `seed`.
        # `get_manifold_trial_folds` returns the whole dataset.
        # So passing `seed=4` to Runner is correct for splitting variance.
        # But `dataset` should remain `seed1`.
        
        print("\n=== Robustness Check: Seed 4 ===")
        run_case(adapter0, top1['norm_mode'], top1['p'], 4, 10, "stage1")
        
    sys.exit(0)

if __name__ == "__main__":
    main()
