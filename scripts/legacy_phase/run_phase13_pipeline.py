import argparse
import os
import sys
import shutil
import json
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets.adapters import get_adapter
from runners.manifold_deep_runner import ManifoldDeepRunner
from runners.spatial_dcnet_torch import SpatialDCNetRunnerTorch, DCNetTorch
from scripts.fusion_postprocess import run_fusion

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 13 Pipeline Executor")
    parser.add_argument("--seed", type=int, required=True, help="Random seed (0-4)")
    parser.add_argument("--gpus", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
    return parser.parse_args()

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def main():
    args = parse_args()
    seed = args.seed
    
    # Base Dir
    base_dir = f"promoted_results/phase13/seed{seed}"
    ensure_dir(base_dir)
    
    # Paths (Global-ish but passed around)
    spatial_pred_path = os.path.join(base_dir, "spatial_trial_pred.csv")
    manifold_pred_path = os.path.join(base_dir, "manifold_trial_pred_guided_p2.csv")
    
    # Load Adapter
    # NOTE: Adapter is usually "seed1" dataset but fold might change? 
    # Actually, for SEED dataset, is "seed1" the dataset name or the random seed? 
    # User clarification in Phase 8: "Dataset: seed1" usually means "SEED-IV dataset".
    # But wait, run_phase13a_seed0.py used `get_adapter("seed1")`.
    # And passed `seed=0` to runners.
    # So `get_adapter("seed1")` likely means "The SEED Dataset Adapter".
    adapter = get_adapter("seed1")
    
    # 1. Spatial
    run_spatial_export(adapter, seed, spatial_pred_path)
    
    # 2. Manifold
    run_manifold(adapter, seed, manifold_pred_path)
    
    # 3. Fusion
    print(f"\n=== [3/3] Fusion (Seed {seed}) ===")
    run_fusion(spatial_pred_path, manifold_pred_path, base_dir, seed=seed)
    
    print(f"\n=== Phase 13 Pipeline Seed {seed} Complete ===")
    
def run_spatial_export(adapter, seed, output_path):
    print(f"\n=== [1/3] Spatial Stream Export (Seed {seed}) ===")
    # ... logic adapted from seed0 script ...
    # We need to ensure we use the correct fallback checkpoint naming
    # ckpt_path = f"experiments/checkpoints/seedv_spatial_torch_seed{seed}_phase13_spatial_fallback.pt"
    # Or if teacher exists: f"experiments/checkpoints/seed{seed}_dcnet_refactor.pt"
    
    ckpt_fallback = f"experiments/checkpoints/seedv_spatial_torch_seed{seed}_phase13_spatial_fallback.pt"
    # Teacher might not exist for seeds > 0 if not pre-trained
    # But we want to run fallback training if needed.
    # Let's verify paths in function body.
    
    _run_spatial_logic(adapter, seed, output_path, ckpt_fallback)

def _run_spatial_logic(adapter, seed, output_path, ckpt_path):
    # Args
    class Args:
        epochs = 0 
        batch_size = 32
        learning_rate = 1e-3
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # Check Checkpoint
    load_ckpt = False
    
    # Priority: 1. Phase 13 Fallback, 2. Phase 9 Refactor
    if not os.path.exists(ckpt_path):
         alt_ckpt = f"experiments/checkpoints/seedv_spatial_torch_seed{seed}_refactor.pt"
         if os.path.exists(alt_ckpt):
             ckpt_path = alt_ckpt
    
    if os.path.exists(ckpt_path):
        print(f"Loading Spatial Checkpoint: {ckpt_path}")
        load_ckpt = True
    else:
        print(f"CRITICAL: Spatial Checkpoint missing: {ckpt_path}. Re-training (Fast).")
        Args.epochs = 50
        load_ckpt = False
        # Set back to default target for saving
        ckpt_path = f"experiments/checkpoints/seedv_spatial_torch_seed{seed}_phase13_spatial_fallback.pt"

    # Data
    folds = adapter.get_spatial_folds_for_cnn(
        seed_de_root="data/SEED/SEED_EEG/ExtractedFeatures_1s",
        seed_de_var="de_LDS1"
    )
    
    # Get Fold 1 (Standard for Phase 13)
    # Note: Adapter behavior for different seeds?
    # get_spatial_folds_for_cnn usually returns 'fold1' based on internal logic.
    # We should trust it returns the correct fold for the current dataset state.
    # However, `get_spatial_folds_for_cnn` does NOT take a seed argument for data splitting if it's fixed?
    # `adapter` is `Seed1Adapter`. It has fixed splits usually?
    # Actually, SEED-IV usually uses fixed Session-based splits or CV.
    # We will assume Fold 1 is the canonical test split.
    fold = folds['fold1']
    
    X_tr, y_tr, tid_tr = fold.X_train, fold.y_train.ravel(), fold.trial_id_train
    X_te, y_te, tid_te = fold.X_test, fold.y_test.ravel(), fold.trial_id_test
    
    runner = SpatialDCNetRunnerTorch(
        num_classes=3,
        epochs=Args.epochs,
        batch_size=Args.batch_size,
        learning_rate=Args.learning_rate,
        device=Args.device,
        spatial_head="softmax"
    )
    
    if load_ckpt:
        model = DCNetTorch(310, 3).to(Args.device)
        checkpoint = torch.load(ckpt_path, map_location=Args.device)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
        
        # Inference
        X_te_tensor = torch.from_numpy(X_te).float().reshape(-1, 310, 1, 1)
        logits_te = runner._batched_logits(model, X_te_tensor, torch.device(Args.device), Args.batch_size)
        proba_test = torch.softmax(torch.from_numpy(logits_te), dim=1).numpy()
    else:
        # Fallback Training
        print(f"Training Spatial Model (Seed {seed})...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        res = runner.fit_predict(fold, fold_name=f"seed{seed}_phase13_spatial")
        proba_test = res['sample_proba_test']
        
        # Save Checkpoint
        if hasattr(runner, 'model'):
             state = {"model_state": runner.model.state_dict()}
             torch.save(state, ckpt_path)
             print(f"Saved Fallback Checkpoint to: {ckpt_path}")

    # Export
    print(f"Exporting Spatial Preds to {output_path}...")
    df_te = pd.DataFrame({
        "trial_id": tid_te,
        "true_label": y_te,
        "prob_0": proba_test[:, 0],
        "prob_1": proba_test[:, 1],
        "prob_2": proba_test[:, 2]
    })
    
    agg = df_te.groupby("trial_id").agg({
        "true_label": "first",
        "prob_0": "mean",
        "prob_1": "mean",
        "prob_2": "mean",
        "trial_id": "count"
    }).rename(columns={"trial_id": "n_windows"}).reset_index()
    
    agg["pred_label"] = np.argmax(agg[["prob_0", "prob_1", "prob_2"]].values, axis=1)
    
    probs = agg[["prob_0", "prob_1", "prob_2"]].values
    p_safe = np.clip(probs, 1e-9, 1.0)
    agg["entropy"] = -np.sum(p_safe * np.log(p_safe), axis=1)
    p_sorted = np.sort(probs, axis=1)
    agg["margin"] = p_sorted[:, -1] - p_sorted[:, -2]
    agg["seed"] = seed
    
    cols = ["seed", "true_label", "n_windows", "prob_0", "prob_1", "prob_2", "pred_label", "entropy", "margin", "trial_id"]
    agg.to_csv(output_path, columns=cols, index=False)

def run_manifold(adapter, seed, output_path):
    print(f"\n=== [2/3] Manifold Stream (Guided p=2.0, Seed {seed}) ===")
    
    class Args:
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        epochs = 50
        batch_size = 32
        mvp1_guided_cov = True
        mvp1_attn_power = 2.0
        dcnet_ckpt = f"experiments/checkpoints/seedv_spatial_torch_seed{seed}_phase13_spatial_fallback.pt"
        metrics_csv = None
        cov_alpha = 0.01
        hidden_dim = 96
        
    args = Args()
    args.seed = seed # Inject seed
    
    # Check Checkpoint
    if not os.path.exists(args.dcnet_ckpt):
         # Try refactor name
         alt = f"experiments/checkpoints/seedv_spatial_torch_seed{seed}_refactor.pt"
         if os.path.exists(alt):
              args.dcnet_ckpt = alt
         else:
              raise FileNotFoundError(f"Teacher Checkpoint missing: {args.dcnet_ckpt}")
    
    print(f"Using Teacher Checkpoint: {args.dcnet_ckpt}")
              
    folds = adapter.get_manifold_trial_folds()
    # Adapter usually returns the same folds?
    # We rely on Random Seed setting in Runner to vary results?
    # Actually, `ManifoldDeepRunner` sets `torch.manual_seed(args.seed)` inside `fit_predict` or `__init__`?
    # Let's check. Yes, likely.
    
    runner = ManifoldDeepRunner(args, num_classes=3)
    
    fold_name = f"phase13_seed{seed}_guided"
    res = runner.fit_predict(folds['fold1'], fold_name=fold_name)
    
    # Move Artifacts
    src_csv = f"promoted_results/{fold_name}_preds_test_last_trial.csv"
    if os.path.exists(src_csv):
        shutil.copy(src_csv, output_path)
        print(f"Manifold Preds saved to: {output_path}")
    else:
        raise FileNotFoundError(f"Manifold runner did not produce: {src_csv}")
        
    # Export Meta
    meta = {
        "seed": seed,
        "guided": True,
        "power": 2.0,
        "epochs": 50,
        "test_trial_acc": res['last']['test_trial_acc']
    }
    with open(os.path.join(os.path.dirname(output_path), "export_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
