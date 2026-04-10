
import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import traceback
import argparse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Setup Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runners.spatial_dcnet_torch import SpatialDCNetRunnerTorch, DCNetTorch
from runners.manifold_deep_runner import ManifoldDeepRunner, TrialDataset
from models.spdnet import DeepSPDClassifier
from models.prototype_mdm import logm_spd, expm_sym
from datasets.adapters import get_adapter

# Configuration
CONFIG = {
    "seeds": [0, 4],
    "spatial_ckpt_fmt": "experiments/checkpoints/seedv_spatial_torch_seed{}_refactor.pt",
    "manifold_ckpt_fmt": "experiments/checkpoints/phase13e/step4/seed1/seed{}/global_centered_corr_tsm/manifold/report_last.pt",
    "manifold_args": {
        "dataset": "seed1",
        "bands_mode": "all5_timecat",
        "band_norm_mode": "per_band_global_z",
        "matrix_mode": "corr",
        "global_centering": True,
        "spd_eps": 1e-3,
        "guided": False, # Step 4 config
        "gate": False,
        "use_roi_pooling": False
    },
    "spatial_args": {
        "dataset": "seed1"
    }
}

class MockArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def get_spatial_logits(seed, adapter):
    print(f"  [Spatial] Loading Checkpoint for Seed {seed}...")
    ckpt_path = CONFIG["spatial_ckpt_fmt"].format(seed)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Spatial Ckpt not found: {ckpt_path}")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Model
    model = DCNetTorch(310, 3).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 2. Data
    folds = adapter.get_spatial_folds_for_cnn(seed_de_root="data/SEED/SEED_EEG/ExtractedFeatures_1s", seed_de_var="de_LDS1")
    fold = folds['fold1']
    
    # Helper to predict
    def predict_split(X, tid, y):
        X_tensor = torch.from_numpy(X).float().reshape(-1, 310, 1, 1)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=2048, shuffle=False)
        
        logits_list = []
        with torch.no_grad():
            for (bx,) in loader:
                out = model(bx.to(device))
                logits_list.append(out.cpu().numpy())
        window_logits = np.concatenate(logits_list, axis=0)
        
        # Aggregate to Trial
        df = pd.DataFrame(window_logits, columns=["l0", "l1", "l2"])
        df["trial_id"] = tid
        df["true_label"] = y
        
        # Mean Logits per trial
        agg = df.groupby("trial_id").agg({
            "true_label": "first",
            "l0": "mean",
            "l1": "mean",
            "l2": "mean"
        }).reset_index()
        
        # Sort by trial_id for alignment
        agg = agg.sort_values("trial_id").reset_index(drop=True)
        return agg

    print("  [Spatial] Predicting Train...")
    train_agg = predict_split(fold.X_train, fold.trial_id_train, fold.y_train.ravel())
    print("  [Spatial] Predicting Test...")
    test_agg = predict_split(fold.X_test, fold.trial_id_test, fold.y_test.ravel())
    
    return train_agg, test_agg

def get_manifold_logits(seed, adapter):
    print(f"  [Manifold] Loading Checkpoint for Seed {seed}...")
    ckpt_path = CONFIG["manifold_ckpt_fmt"].format(seed)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Manifold Ckpt not found: {ckpt_path}")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    m_cfg = CONFIG["manifold_args"]
    args = MockArgs(
        dataset=m_cfg["dataset"],
        bands_mode=m_cfg["bands_mode"],
        band_norm_mode=m_cfg["band_norm_mode"],
        matrix_mode=m_cfg["matrix_mode"],
        global_centering=m_cfg["global_centering"],
        spd_eps=m_cfg["spd_eps"],
        mvp1_guided_cov=m_cfg["guided"],
        use_band_gate=m_cfg["gate"],
        use_roi_pooling=m_cfg["use_roi_pooling"],
        epochs=0, 
        batch_size=32,
        lr=1e-4,
        torch_device=device
    )
    
    runner = ManifoldDeepRunner(args, num_classes=3)
    
    # 2. Data Loop needed to build model (input dim)
    folds = adapter.get_manifold_trial_folds()
    fold = folds['fold1']
    
    # We need to construct loaders similarly to fit_predict to get dimensions and ref covariance
    # But wait, fit_predict calculates C_ref.
    # We MUST load C_ref if we use global centering.
    # It was saved in `promoted_results/.../global_ref_spd.npy` or `global_ref.pt`?
    # Step 4 saved `global_ref_spd.npy`.
    # ManifoldDeepRunner computes it on the fly.
    # If we just load the model weights, C_ref isn't in there (it's data preprocessing).
    # We MUST re-compute C_ref on Training Data exactly as Step 4 did.
    # Step 4 used `fit_predict`.
    
    # Let's Re-Do Setup from fit_predict
    X_tr_list = fold.trials_train
    y_tr_list = fold.y_trial_train
    # Need to split? 
    # Step 4 fit_predict splits train/val.
    # And C_ref is computed on the TRAIN SUBSET (80%).
    # We must reproduce this split to match the trained model's C_ref assumption.
    
    n_train_total = len(X_tr_list)
    n_sub_train = int(0.8 * n_train_total)
    perm = np.random.RandomState(seed).permutation(n_train_total) # Use fixed seed
    tr_indices_sub = perm[:n_sub_train]
    
    X_train_sub = [X_tr_list[i] for i in tr_indices_sub]
    y_train_sub = [y_tr_list[i] for i in tr_indices_sub]
    
    window_len = 24
    stride = 12
    return_5band = (m_cfg["bands_mode"] == "all5_timecat")
    dset_norm = "none" if m_cfg["band_norm_mode"] in ["manual_per_band_time", "per_band_global_z"] else m_cfg["band_norm_mode"]
    
    train_sub_dset = TrialDataset(X_train_sub, y_train_sub, band_idx=4, window_len=window_len, stride=stride, return_5band=return_5band, bands_mode=m_cfg["bands_mode"], band_norm_mode=dset_norm)
    train_sub_loader = DataLoader(train_sub_dset, batch_size=32, shuffle=True)
    
    # Compute C_ref
    print("  [Manifold] Re-computing Global Template (Train Subset)...")
    runner.log_ref_global = runner._compute_global_log_euclidean_mean(train_sub_loader, "dummy")
    
    # Now Build Model
    # Need input dim
    # Get a batch
    for xb, _, _ in train_sub_loader:
        if xb.ndim == 4:
            # Logic from runner
            xb_perm = xb.permute(0, 3, 2, 1).contiguous()
            if m_cfg["band_norm_mode"] == "per_band_global_z":
                  pass # Preproc handled in dataset output? No, in runner loop usually.
                  # TrialDataset returns raw windows if norm mode is special?
                  # Actually TrialDataset implementation applies simple transforms. 
                  # ManifoldDeepRunner applies complex `per_band_global_z` logic inside loop if 4D.
            model_in_dim = 62 # Fixed for this
        else:
            model_in_dim = xb.shape[-1] # if matrix
        break
        
    # We know dim is 62
    runner.model = DeepSPDClassifier(n_channels=62, deep_layers=2, n_classes=3, output_dim=32, cov_eps=m_cfg["spd_eps"], hidden_dim=96).to(device)
    
    # Load Weights
    cp = torch.load(ckpt_path, map_location=device)
    runner.model.load_state_dict(cp)
    runner.model.eval()
    
    # Feature Helper
    def predict_manifold(X_list, y_list, tids):
        dset = TrialDataset(X_list, y_list, band_idx=4, window_len=window_len, stride=stride, return_5band=return_5band, bands_mode=m_cfg["bands_mode"], band_norm_mode=dset_norm, trial_ids=tids)
        loader = DataLoader(dset, batch_size=64, shuffle=False)
        
        logits = []
        labels = []
        ids = []
        
        with torch.no_grad():
            for xb, yb, tib in loader:
                xb = xb.to(device).double()
                
                # Preproc 4D -> Matrix
                if xb.ndim == 4:
                     xb_perm = xb.permute(0, 3, 2, 1).contiguous()
                     if m_cfg["band_norm_mode"] == "per_band_global_z":
                         mean = xb_perm.mean(dim=(2, 3), keepdim=True)
                         std = xb_perm.std(dim=(2, 3), keepdim=True) + 1e-6
                         xb_perm = (xb_perm - mean) / std
                     model_in = xb_perm.permute(0, 2, 1, 3).reshape(xb.size(0), 62, -1)
                else:
                     model_in = xb
                
                mat = runner._compute_matrix_from_input(model_in)
                mat = mat + torch.eye(mat.size(1), device=device).double() * runner.spd_eps
                
                # Global Center
                if runner.global_centering:
                    log_C = logm_spd(mat) # Need import? ManifoldDeepRunner might not have it exposed?
                    # ManifoldDeepRunner imports it internally or defines it.
                    # Wait, runner has it as method? No.
                    # It calls `log_C = ...` inside loop.
                    # I need to replicate logic or use `runner.model.forward`?
                    # `runner.model` is `DeepSPDClassifier`. It takes matrices.
                    # Centering happens BEFORE model.
                    pass
                
                # Replicate Centering
                # We need logm_spd. It's in `models/prototype_mdm` or `runners/manifold_deep_runner`?
                # Step 4 added it to `runners/manifold_deep_runner`? No, Step 5 created `models/prototype_mdm`. 
                # Step 4 script likely embedded it or used a library.
                # Let's import from `models.prototype_mdm` since we made it.
                # Local import removed (global used)
                
                log_C = logm_spd(mat)
                log_ref = runner.log_ref_global
                if log_ref.ndim == 2: log_ref = log_ref.unsqueeze(0)
                log_centered = log_C - log_ref
                C_centered = expm_sym(log_centered).float() # Model expects float usually
                
                out = runner.model(C_centered)
                logits.append(out.cpu().numpy())
                labels.append(yb.numpy())
                ids.append(tib.numpy()) # tib is numeric index
                
        logits = np.concatenate(logits, axis=0)
        labels = np.concatenate(labels, axis=0)
        indices = np.concatenate(ids, axis=0)
        
        # Map indices back to string IDs
        # tids is numpy array of strings
        trial_ids_str = [tids[i] for i in indices]
        
        df = pd.DataFrame(logits, columns=["l0", "l1", "l2"])
        df["trial_id"] = trial_ids_str
        df["true_label"] = labels
        
        agg = df.groupby("trial_id").agg({
            "true_label": "first",
            "l0": "mean", # Mean logits aggregation
            "l1": "mean",
            "l2": "mean"
        }).reset_index()
        
        agg = agg.sort_values("trial_id").reset_index(drop=True)
        return agg

    print("  [Manifold] Predicting Train...")
    # Passing full train set
    train_agg = predict_manifold(fold.trials_train, fold.y_trial_train, fold.trial_id_train)
    print("  [Manifold] Predicting Test...")
    test_agg = predict_manifold(fold.trials_test, fold.y_trial_test, fold.trial_id_test)
    
    return train_agg, test_agg

def run_seed(seed):
    print(f"\n=== Processing Seed {seed} ===")
    adapter = get_adapter("seed1")
    
    s_train, s_test = get_spatial_logits(seed, adapter)
    m_train, m_test = get_manifold_logits(seed, adapter)
    
    # Align & Check Integrity
    print(f"  [Fusion] Aligning...")
    
    def check_align(df1, df2, name):
        if len(df1) != len(df2):
            raise ValueError(f"{name} length mismatch: {len(df1)} vs {len(df2)}")
        if not df1["trial_id"].equals(df2["trial_id"]):
            raise ValueError(f"{name} trial_id mismatch")
        if not df1["true_label"].equals(df2["true_label"]):
            l1 = df1["true_label"].astype(int)
            l2 = df2["true_label"].astype(int)
            if not l1.equals(l2):
                 raise ValueError(f"{name} label mismatch")
    
    check_align(s_train, m_train, "Train")
    check_align(s_test, m_test, "Test")
    
    # Prepare Fusion Data
    # Stack Logits: [S_logits, M_logits] -> 6 features
    X_train = np.hstack([s_train[["l0", "l1", "l2"]].values, m_train[["l0", "l1", "l2"]].values])
    y_train = s_train["true_label"].values.astype(int)
    
    X_test = np.hstack([s_test[["l0", "l1", "l2"]].values, m_test[["l0", "l1", "l2"]].values])
    y_test = s_test["true_label"].values.astype(int)
    
    # Train Linear Fusion
    print("  [Fusion] Training Linear Head...")
    clf = LogisticRegression(random_state=seed, solver='lbfgs', multi_class='multinomial')
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Baseline Accs
    acc_s = accuracy_score(y_test, np.argmax(s_test[["l0", "l1", "l2"]].values, axis=1))
    acc_m = accuracy_score(y_test, np.argmax(m_test[["l0", "l1", "l2"]].values, axis=1))
    
    print(f"  [Result] Spatial: {acc_s:.4f} | Manifold: {acc_m:.4f} | Fusion: {acc:.4f}")
    
    # Save Outputs
    out_dir = f"promoted_results/phase13f/step1/seed1/seed{seed}/fusion"
    ensure_dir(out_dir)
    ensure_dir(f"promoted_results/phase13f/step1/seed1/seed{seed}/dcnet")
    ensure_dir(f"promoted_results/phase13f/step1/seed1/seed{seed}/manifold")

    s_test.to_csv(f"promoted_results/phase13f/step1/seed1/seed{seed}/dcnet/spatial_trial_pred.csv", index=False)
    m_test.to_csv(f"promoted_results/phase13f/step1/seed1/seed{seed}/manifold/manifold_trial_pred.csv", index=False)
    
    # Fusion Predictions
    f_df = pd.DataFrame({
        "trial_id": s_test["trial_id"],
        "true_label": y_test,
        "pred_label": y_pred,
        "logit0": 0, "logit1": 0, "logit2": 2 # Dummy or get proba
    })
    probas = clf.predict_proba(X_test)
    f_df["prob_0"] = probas[:, 0]
    f_df["prob_1"] = probas[:, 1]
    f_df["prob_2"] = probas[:, 2]
    
    f_df.to_csv(f"{out_dir}/fusion_trial_pred.csv", index=False)
    
    # Single Run Report
    with open(f"{out_dir}/SINGLE_RUN_REPORT.md", "w") as f:
        f.write(f"# Fusion Report Seed {seed}\n")
        f.write(f"- Spatial Acc: {acc_s:.4f}\n")
        f.write(f"- Manifold Acc: {acc_m:.4f}\n")
        f.write(f"- Fusion Acc: {acc:.4f}\n")
        f.write(f"- Align Check: PASS\n")
        
    return {
        "Seed": seed,
        "Spatial Acc": acc_s,
        "Manifold Acc": acc_m,
        "Fusion Acc": acc, 
        "Delta S": acc - acc_s,
        "Delta M": acc - acc_m
    }

def main():
    results = []
    for seed in CONFIG["seeds"]:
        try:
            res = run_seed(seed)
            results.append(res)
        except Exception as e:
            traceback.print_exc()
            print(f"Seed {seed} Failed.")
            
    # Summary
    df = pd.DataFrame(results)
    out = "promoted_results/phase13f/step1/seed1/summary.csv"
    ensure_dir(os.path.dirname(out))
    df.to_csv(out, index=False)
    
    # Experiment MD
    md_path = "promoted_results/phase13f/step1/EXPERIMENT_REPORT.md"
    with open(md_path, "w") as f:
        f.write("# Phase 13F Step 1: Late Fusion (Logit Stacking)\n\n")
        f.write(f"| {' | '.join(df.columns)} |\n")
        f.write(f"| {' | '.join(['---']*len(df.columns))} |\n")
        for _, row in df.iterrows():
            vals = [f"{x:.4f}" if isinstance(x, float) else str(x) for x in row.values]
            f.write(f"| {' | '.join(vals)} |\n")
            
    print(f"Done. Report at {md_path}")

if __name__ == "__main__":
    main()
