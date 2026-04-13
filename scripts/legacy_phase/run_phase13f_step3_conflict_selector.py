
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import json
import traceback
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Setup Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from runners.spatial_dcnet_torch import DCNetTorch
from runners.manifold_deep_runner import ManifoldDeepRunner, TrialDataset
from models.spdnet import DeepSPDClassifier
from models.prototype_mdm import logm_spd, expm_sym
from datasets.adapters import get_adapter

# Configuration (Identical to Step 1)
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
        "guided": False, 
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

def write_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# ---------------------------------------------------------
# Logit Extraction Logic (Copied from Step 1 for Consistency)
# ---------------------------------------------------------

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
    
    folds = adapter.get_manifold_trial_folds()
    fold = folds['fold1']
    
    # Re-compute C_ref on Training Subset (80%) logic matching Step 4
    X_tr_list = fold.trials_train
    y_tr_list = fold.y_trial_train
    
    n_train_total = len(X_tr_list)
    n_sub_train = int(0.8 * n_train_total)
    perm = np.random.RandomState(seed).permutation(n_train_total) 
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
    
    # Build Model
    runner.model = DeepSPDClassifier(n_channels=62, deep_layers=2, n_classes=3, output_dim=32, cov_eps=m_cfg["spd_eps"], hidden_dim=96).to(device)
    
    # Load Weights
    cp = torch.load(ckpt_path, map_location=device)
    runner.model.load_state_dict(cp)
    runner.model.eval()
    
    # Predict Helper
    def predict_manifold(X_list, y_list, tids):
        dset = TrialDataset(X_list, y_list, band_idx=4, window_len=window_len, stride=stride, return_5band=return_5band, bands_mode=m_cfg["bands_mode"], band_norm_mode=dset_norm, trial_ids=tids)
        loader = DataLoader(dset, batch_size=64, shuffle=False)
        
        logits = []
        labels = []
        ids = []
        
        with torch.no_grad():
            for xb, yb, tib in loader:
                xb = xb.to(device).double()
                
                # Preproc
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
                log_C = logm_spd(mat)
                log_ref = runner.log_ref_global
                if log_ref.ndim == 2: log_ref = log_ref.unsqueeze(0)
                log_centered = log_C - log_ref
                C_centered = expm_sym(log_centered).float()
                
                out = runner.model(C_centered)
                logits.append(out.cpu().numpy())
                labels.append(yb.numpy())
                ids.append(tib.numpy())
                
        logits = np.concatenate(logits, axis=0)
        labels = np.concatenate(labels, axis=0)
        indices = np.concatenate(ids, axis=0)
        
        trial_ids_str = [tids[i] for i in indices]
        
        df = pd.DataFrame(logits, columns=["l0", "l1", "l2"])
        df["trial_id"] = trial_ids_str
        df["true_label"] = labels
        
        agg = df.groupby("trial_id").agg({
            "true_label": "first",
            "l0": "mean",
            "l1": "mean",
            "l2": "mean"
        }).reset_index()
        
        agg = agg.sort_values("trial_id").reset_index(drop=True)
        return agg

    print("  [Manifold] Predicting Train...")
    train_agg = predict_manifold(fold.trials_train, fold.y_trial_train, fold.trial_id_train)
    print("  [Manifold] Predicting Test...")
    test_agg = predict_manifold(fold.trials_test, fold.y_trial_test, fold.trial_id_test)
    
    return train_agg, test_agg

# ---------------------------------------------------------
# Step 3 Fusion Logic
# ---------------------------------------------------------

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

def run_seed(seed, out_root):
    print(f"\n=== Processing Seed {seed} ===")
    adapter = get_adapter("seed1")
    
    s_train, s_test = get_spatial_logits(seed, adapter)
    m_train, m_test = get_manifold_logits(seed, adapter)
    
    # Align
    check_align(s_train, m_train, "Train")
    check_align(s_test, m_test, "Test")
    print(f"  [Fusion] Alignment OK.")
    
    # -------------------------------------------------------
    # 1. Prepare Data for Selector (Train on Conflict Samples)
    # -------------------------------------------------------
    
    # Function to extract conflict data
    def process_split(df_s, df_m):
        # Predictions
        pred_s = np.argmax(df_s[["l0", "l1", "l2"]].values, axis=1)
        pred_m = np.argmax(df_m[["l0", "l1", "l2"]].values, axis=1)
        y_true = df_s["true_label"].values.astype(int)
        
        # Flags
        is_agree = (pred_s == pred_m)
        is_conflict = (pred_s != pred_m)
        
        s_ok = (pred_s == y_true)
        m_ok = (pred_m == y_true)
        
        # Selector Target (Only for Conflict samples)
        # 1 if M correct (Trust M), 0 if S correct (Trust S)
        # If both correct (impossible in conflict), or both wrong, specific handling?
        # Target:
        #  If M_ok and !S_ok -> 1
        #  If S_ok and !M_ok -> 0
        #  If Both Wrong -> ? (Undefined, exclude from training?)
        # For training, we only want to learn from solvable conflicts.
        
        # Features: Combined Logits
        X_combined = np.hstack([df_s[["l0", "l1", "l2"]].values, df_m[["l0", "l1", "l2"]].values])
        
        return {
            "pred_s": pred_s,
            "pred_m": pred_m,
            "y_true": y_true,
            "is_conflict": is_conflict,
            "s_ok": s_ok,
            "m_ok": m_ok,
            "X": X_combined,
            "ids": df_s["trial_id"].values
        }

    tr = process_split(s_train, m_train)
    te = process_split(s_test, m_test)
    
    # -------------------------------------------------------
    # Evidence (A): Dataset Counts
    # -------------------------------------------------------
    n_train_total_trials = int(len(tr["y_true"]))
    n_test_total_trials = int(len(te["y_true"]))
    n_conflict_train = int(tr["is_conflict"].sum())
    n_conflict_test = int(te["is_conflict"].sum())
    n_agree_train = int(n_train_total_trials - n_conflict_train)
    n_agree_test = int(n_test_total_trials - n_conflict_test)

    # Select Training Samples for Selector
    # Condition: Conflict AND Exactly One Correct
    train_mask = tr["is_conflict"] & (tr["s_ok"] ^ tr["m_ok"]) # XOR
    
    X_sel_train = tr["X"][train_mask]
    # y=1 if M_ok (implies !S_ok due to XOR), y=0 if S_ok (implies !M_ok)
    y_sel_train = tr["m_ok"][train_mask].astype(int) 
    
    # -------------------------------------------------------
    # Evidence (B): Supervision Availability on TRAIN Conflicts
    # -------------------------------------------------------
    n_supervised_train_conflict = int(train_mask.sum())
    label1_count = int(y_sel_train.sum())  # 1 = trust_manifold
    label0_count = int(n_supervised_train_conflict - label1_count)  # 0 = trust_spatial
    single_class_flag = bool(n_supervised_train_conflict > 0 and (label0_count == 0 or label1_count == 0))

    print(
        f"  [Selector] Training Size: {n_supervised_train_conflict} samples "
        f"out of {n_conflict_train} conflicts (Total Train: {n_train_total_trials})"
    )

    # -------------------------------------------------------
    # 1. Train Selector (or force fallback)
    # -------------------------------------------------------
    selector = None
    selector_trained_flag = False
    fallback_reason = ""  # "single_class" | "no_supervised_conflicts" | "other" (empty string if trained)

    if n_supervised_train_conflict == 0:
        fallback_reason = "no_supervised_conflicts"
        print("  [Selector] Fallback: no supervised train conflicts. Defaulting to Spatial on conflicts.")
    elif single_class_flag:
        fallback_reason = "single_class"
        print(
            f"  [Selector] Fallback: single-class supervision "
            f"(label0_count={label0_count}, label1_count={label1_count}). Defaulting to Spatial on conflicts."
        )
    else:
        try:
            selector = LogisticRegression(random_state=seed, solver="lbfgs")
            selector.fit(X_sel_train, y_sel_train)
            selector_trained_flag = True
            print("  [Selector] Trained.")
        except Exception:
            fallback_reason = "other"
            selector_trained_flag = False
            selector = None
            print("  [Selector] Fallback: training failed. Defaulting to Spatial on conflicts.")
            traceback.print_exc()
    
    # -------------------------------------------------------
    # 2. Inference on Test Set
    # -------------------------------------------------------
    
    # Logic:
    # If Agree: use Pred S
    # If Conflict:
    #    Predict Trust (Prob(Trust M))
    #    If Prob > 0.5: use Pred M
    #    Else: use Pred S
    
    prob_trust_m = np.zeros(len(te["y_true"]))
    
    # Inference Loop (Vectorized)
    # Default selection: Spatial (0)
    trust_decisions = np.zeros(len(te["y_true"]), dtype=int) # 0=Trust S, 1=Trust M
    
    if selector_trained_flag:
        # Predict on ALL, filter later
        # Actually only need on conflict
        # But for array alignment, predict all or mask
        # Let's predict on all for simplicity of indexing
        trust_probas = selector.predict_proba(te["X"])[:, 1] # Prob class 1 (M)
        prob_trust_m = trust_probas
        trust_decisions = (trust_probas > 0.5).astype(int)
    
    # Fusion Rule
    y_pred_fusion = np.zeros_like(te["pred_s"])
    
    # Case 1: Agree
    mask_agree = ~te["is_conflict"]
    y_pred_fusion[mask_agree] = te["pred_s"][mask_agree]
    
    # Case 2: Conflict
    mask_conflict = te["is_conflict"]
    
    # Apply Trust
    # If trust_decisions == 1 (Trust M) -> M
    # If trust_decisions == 0 (Trust S) -> S
    
    # We can compute elementwise
    # But mask_conflict filters.
    
    # Final Calculation
    # Start with S
    y_pred_fusion = te["pred_s"].copy()
    
    # Overwrite where Conflict AND Trust M
    mask_override = mask_conflict & (trust_decisions == 1)
    y_pred_fusion[mask_override] = te["pred_m"][mask_override]
    
    # Metrics
    acc_s = accuracy_score(te["y_true"], te["pred_s"])
    acc_m = accuracy_score(te["y_true"], te["pred_m"])
    acc_f = accuracy_score(te["y_true"], y_pred_fusion)
    
    print(f"  [Result] Spatial: {acc_s:.4f} | Manifold: {acc_m:.4f} | Fusion (Conflict): {acc_f:.4f}")
    
    # Attribution Calculation
    f_ok = (y_pred_fusion == te["y_true"])
    s_ok = te["s_ok"]
    
    rescued = (f_ok & ~s_ok).sum()
    lost = (~f_ok & s_ok).sum()
    net = rescued - lost

    complementary_count = int((~te["s_ok"] & te["m_ok"]).sum())
    
    # Conflict Analysis
    n_conflicts = mask_conflict.sum()
    n_agreements = mask_agree.sum()
    # Accuracy in Agreement
    acc_agree = accuracy_score(te["y_true"][mask_agree], y_pred_fusion[mask_agree])
    # Accuracy in Conflict
    acc_conflict = 0
    if n_conflicts > 0:
        acc_conflict = accuracy_score(te["y_true"][mask_conflict], y_pred_fusion[mask_conflict])
        
    print(f"  [Attribution] Rescued: {rescued}, Lost: {lost}, Net: {net}")
    print(f"  [Conflict] N={n_conflicts}, Acc={acc_conflict:.4f}")

    # -------------------------------------------------------
    # Evidence (C): Fallback behavior on TEST conflicts
    # -------------------------------------------------------
    if selector_trained_flag:
        fallback_rate_test_conflict = 0.0
    else:
        if n_conflict_test == 0:
            fallback_rate_test_conflict = 0.0
        else:
            # In fallback mode, we default to Spatial on all conflicts.
            fallback_rate_test_conflict = float((mask_conflict & (trust_decisions == 0)).sum()) / float(n_conflict_test)
    
    # Save Results
    out_dir = os.path.join(out_root, "seed1", f"seed{seed}")
    ensure_dir(out_dir)
    
    # CSV
    out_df = pd.DataFrame({
        "trial_id": te["ids"],
        "true_label": te["y_true"],
        "pred_s": te["pred_s"],
        "pred_m": te["pred_m"],
        "is_conflict": te["is_conflict"].astype(int),
        "trust_m_prob": prob_trust_m,
        "selected_expert": np.where(mask_override, "Manifold", "Spatial"),
        "pred_fusion": y_pred_fusion,
        "is_correct": f_ok.astype(int)
    })
    out_df.to_csv(f"{out_dir}/fusion_pred.csv", index=False)

    # Evidence JSON (A–D)
    evidence = {
        "seed": int(seed),
        # (A) Dataset counts
        "n_train_total_trials": int(n_train_total_trials),
        "n_test_total_trials": int(n_test_total_trials),
        "n_conflict_train": int(n_conflict_train),
        "n_conflict_test": int(n_conflict_test),
        "n_agree_train": int(n_agree_train),
        "n_agree_test": int(n_agree_test),
        # (B) Supervision availability
        "n_supervised_train_conflict": int(n_supervised_train_conflict),
        "label0_count": int(label0_count),  # trust_spatial
        "label1_count": int(label1_count),  # trust_manifold
        "single_class_flag": bool(single_class_flag),
        # (C) Fallback behavior
        "selector_trained_flag": bool(selector_trained_flag),
        "fallback_reason": fallback_reason,
        "fallback_rate_test_conflict": float(fallback_rate_test_conflict),
        # (D) Final metrics (Step 2 schema-aligned)
        "acc_spatial": float(acc_s),
        "acc_manifold": float(acc_m),
        "acc_fusion": float(acc_f),
        "rescued": int(rescued),
        "lost": int(lost),
        "net_gain": int(net),
        "complementary_count": int(complementary_count),
    }
    write_json(os.path.join(out_dir, "selector_evidence.json"), evidence)

    return evidence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_root",
        type=str,
        default="promoted_results/phase13f/step3",
        help="Output root directory (e.g., promoted_results/phase13f/step3a).",
    )
    args = parser.parse_args()

    results = []
    for seed in CONFIG["seeds"]:
        try:
            res = run_seed(seed, out_root=args.out_root)
            results.append(res)
        except Exception:
            traceback.print_exc()
            print(f"Seed {seed} Failed.")
            
    # Summary
    df = pd.DataFrame(results)
    out = os.path.join(args.out_root, "seed1", "summary.csv")
    ensure_dir(os.path.dirname(out))
    df.to_csv(out, index=False)
    
    # Experiment Report
    md_path = os.path.join(args.out_root, "seed1", "EXPERIMENT_REPORT.md")
    with open(md_path, "w") as f:
        f.write("# Phase 13F Step 3a: Conflict-Selector Evidence Chain (Quantification)\n\n")

        cols = [
            "seed",
            "n_train_total_trials",
            "n_test_total_trials",
            "n_conflict_train",
            "n_conflict_test",
            "n_agree_train",
            "n_agree_test",
            "n_supervised_train_conflict",
            "label0_count",
            "label1_count",
            "single_class_flag",
            "selector_trained_flag",
            "fallback_reason",
            "fallback_rate_test_conflict",
            "acc_spatial",
            "acc_manifold",
            "acc_fusion",
            "rescued",
            "lost",
            "net_gain",
            "complementary_count",
        ]

        df_out = df.copy()
        for c in cols:
            if c not in df_out.columns:
                df_out[c] = np.nan
        df_out = df_out[cols]

        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, r in df_out.iterrows():
            row = []
            for c in cols:
                v = r[c]
                if c in {"acc_spatial", "acc_manifold", "acc_fusion", "fallback_rate_test_conflict"} and pd.notna(v):
                    row.append(f"{float(v):.4f}")
                elif isinstance(v, (bool, np.bool_)):
                    row.append("True" if bool(v) else "False")
                elif pd.isna(v):
                    row.append("")
                else:
                    row.append(str(int(v)) if isinstance(v, (int, np.integer)) else str(v))
            f.write("| " + " | ".join(row) + " |\n")

    print(f"\nDone. Report at {md_path}")

if __name__ == "__main__":
    main()
