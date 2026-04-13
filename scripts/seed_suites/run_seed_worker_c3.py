
import os
import sys
import json
import shutil
import argparse
import numpy as np
import pandas as pd
import torch
import traceback
from pathlib import Path
from sklearn.metrics import accuracy_score

# Add project root to sys.path
sys.path.append(os.getcwd())

# Lazy imports inside functions to minimize overhead/conflicts
from datasets.adapters import get_adapter

# --- constants ---
PHASE_ID = "phase13c3"
CONFIG_PATH = f"promoted_results/{PHASE_ID}/config_locked.json"
BASE_OUTPUT_DIR = f"promoted_results/{PHASE_ID}"
TEACHER_CKPT_TEMPLATE = "experiments/checkpoints/seedv_spatial_torch_seed{}_refactor.pt"

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

class ManifoldArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

# --- Stage Functions ---

def run_spatial_stage(seed, seed_dir):
    from runners.spatial_dcnet_torch import SpatialDCNetRunnerTorch, DCNetTorch
    
    output_path = f"{seed_dir}/fusion/spatial_trial_pred.csv"
    ensure_dir(os.path.dirname(output_path))
    
    print(f"[Worker] Running Spatial Inference for Seed {seed}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = TEACHER_CKPT_TEMPLATE.format(seed)
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Spatial Checkpoint missing: {ckpt_path}")
        
    adapter = get_adapter("seed1")
    folds = adapter.get_spatial_folds_for_cnn(
        seed_de_root="data/SEED/SEED_EEG/ExtractedFeatures_1s",
        seed_de_var="de_LDS1"
    )
    fold = folds['fold1']
    X_te, y_te, tid_te = fold.X_test, fold.y_test.ravel(), fold.trial_id_test
    
    model = DCNetTorch(310, 3).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    runner = SpatialDCNetRunnerTorch(num_classes=3, epochs=0, device=device)
    X_te_tensor = torch.from_numpy(X_te).float().reshape(-1, 310, 1, 1)
    
    try:
        logits_te = runner._batched_logits(model, X_te_tensor, torch.device(device), 2048)
        proba_test = torch.softmax(torch.from_numpy(logits_te), dim=1).numpy()
    except AttributeError:
        preds_list = []
        with torch.no_grad():
            dl = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_te_tensor), 
                batch_size=2048, shuffle=False
            )
            for (bx,) in dl:
                out = model(bx.to(device))
                preds_list.append(torch.softmax(out, dim=1).cpu().numpy())
        proba_test = np.concatenate(preds_list, axis=0)

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
    print(f"[Worker] Spatial Preds exported to: {output_path}")

def run_manifold_stage(seed, seed_dir, config):
    from runners.manifold_deep_runner import ManifoldDeepRunner
    
    print(f"[Worker] Running Manifold Training for Seed {seed}...")
    ensure_dir(f"{seed_dir}/manifold")
    
    m_args = ManifoldArgs(
        torch_device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        mvp1_guided_cov=config['guided'],
        mvp1_attn_power=config['attn_power'],
        bands_mode=config['bands_mode'],
        band_norm_mode=config['band_norm_mode'],
        spd_eps=config['spd_eps'],
        dcnet_ckpt=TEACHER_CKPT_TEMPLATE.format(seed),
        metrics_csv=None 
    )
    runner = ManifoldDeepRunner(m_args, num_classes=3)
    fold_name = f"phase13c3_seed{seed}_guided"
    adapter = get_adapter("seed1")
    folds = adapter.get_manifold_trial_folds()
    
    res = runner.fit_predict(folds['fold1'], fold_name=fold_name)
    
    src_pred = f"promoted_results/{fold_name}_preds_test_last_trial.csv"
    dst_pred = f"{seed_dir}/manifold/manifold_trial_pred.csv"
    shutil.move(src_pred, dst_pred)
    
    with open(f"{seed_dir}/manifold/report.json", "w") as f:
        json.dump(res.get('last',{}), f, indent=2)
    print(f"[Worker] Manifold Training complete.")

def compute_fusion_metrics(m_df, s_df):
    m_probs = m_df[['prob_0', 'prob_1', 'prob_2']].values
    s_probs = s_df[['prob_0', 'prob_1', 'prob_2']].values
    y_true = m_df['true_label'].values
    m_pred = np.argmax(m_probs, axis=1)
    s_pred = np.argmax(s_probs, axis=1)
    agree_mask = (m_pred == s_pred)
    conflict_mask = ~agree_mask
    acc_m = accuracy_score(y_true, m_pred)
    acc_s = accuracy_score(y_true, s_pred)
    
    if np.sum(conflict_mask) > 0:
        acc_m_conflict = accuracy_score(y_true[conflict_mask], m_pred[conflict_mask])
        acc_s_conflict = accuracy_score(y_true[conflict_mask], s_pred[conflict_mask])
    else:
        acc_m_conflict = 0.0
        acc_s_conflict = 0.0
        
    w = 0.5
    f0_probs = (1-w)*s_probs + w*m_probs
    f0_pred = np.argmax(f0_probs, axis=1)
    acc_f0 = accuracy_score(y_true, f0_pred)
    
    return {
        "spatial_acc": acc_s,
        "manifold_acc": acc_m,
        "fusion_v0_acc": acc_f0,
        "agree_rate": np.mean(agree_mask),
        "conflict_rate": np.mean(conflict_mask),
        "spatial_acc_conflict": acc_s_conflict,
        "manifold_acc_conflict": acc_m_conflict,
        "n_samples": len(y_true)
    }

def run_fusion_stage(seed, seed_dir):
    print(f"[Worker] Running Fusion Audit for Seed {seed}...")
    spatial_csv = f"{seed_dir}/fusion/spatial_trial_pred.csv"
    manifold_csv = f"{seed_dir}/manifold/manifold_trial_pred.csv"
    
    if not os.path.exists(spatial_csv) or not os.path.exists(manifold_csv):
        raise FileNotFoundError("Missing inputs for fusion stage")
        
    m_df = pd.read_csv(manifold_csv)
    s_df = pd.read_csv(spatial_csv)
    
    # Audit
    m_trials = set(m_df['trial_id'].unique())
    s_trials = set(s_df['trial_id'].unique())
    if m_trials != s_trials:
        msg = f"ALIGNMENT FAIL: Trial Mismatch M={len(m_trials)} S={len(s_trials)}"
        with open(f"{seed_dir}/INTEGRITY_FAIL.txt", "w") as f: f.write(msg)
        raise ValueError(msg)
        
    m_df = m_df.sort_values('trial_id').reset_index(drop=True)
    s_df = s_df.sort_values('trial_id').reset_index(drop=True)
    if not m_df['trial_id'].equals(s_df['trial_id']):
        raise ValueError("Sort mismatch")
        
    stats = compute_fusion_metrics(m_df, s_df)
    with open(f"{seed_dir}/INTEGRITY_PASS.txt", "w") as f: f.write("PASS")
    
    with open(f"{seed_dir}/c3_worker_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[Worker] Fusion Audit Complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["spatial", "manifold", "fusion"])
    args = parser.parse_args()
    seed = args.seed
    
    config = load_config()
    
    seed_dir = f"{BASE_OUTPUT_DIR}/seed1/seed{seed}"
    ensure_dir(seed_dir)
    
    try:
        if args.mode == "spatial":
            run_spatial_stage(seed, seed_dir)
        elif args.mode == "manifold":
            # Clean CUDA before manifold
            torch.cuda.empty_cache()
            run_manifold_stage(seed, seed_dir, config)
        elif args.mode == "fusion":
            run_fusion_stage(seed, seed_dir)
            
        sys.exit(0)
        
    except Exception as e:
        print(f"=== Worker Seed {seed} Mode {args.mode} Failed ===")
        traceback.print_exc()
        # Only write FAIL file in fusion stage or if critical?
        # Actually, Orchestrator catches exit code.
        # But writing to file helps diagnostics.
        with open(f"{seed_dir}/INTEGRITY_FAIL.txt", "w") as f: f.write(f"Mode {args.mode}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
