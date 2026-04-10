
import os
import sys
import json
import random
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.seed_raw_cnt import build_eeg62_view, load_one_raw
from datasets.seed_raw_trials import build_trial_index, slice_raw_trials
from manifold_raw.features import BandSpec, bandpass, parse_band_spec, window_slices, logmap_spd, vec_utri

CONFIG = {
    "seed": 0,
    "seed_raw_root": "data/SEED/SEED_EEG/SEED_RAW_EEG",
    "raw_backend": "cnt",
    "window_sec": 4.0,
    "hop_sec": 1.0,
    "bands": "delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50",
    "spd_eps": 1e-4, # Increased for stability
    "max_iter": 50, # Keep 50 for speed in smoke, but valid Counter import
    "out_root": "promoted_results/phase14r/step3/rawcov_global_center_4s1s"
}

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def write_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _parse_cnt_name(cnt_path):
    base = os.path.splitext(os.path.basename(cnt_path))[0]
    parts = base.split("_")
    if len(parts) < 2: return -1, -1, parts[0]
    return int(parts[0]), int(parts[1]), parts[0]

def _sorted_cnt_files(raw_root):
    paths = [str(p) for p in Path(raw_root).iterdir() if p.suffix.lower() == ".cnt"]
    return sorted(paths, key=lambda p: _parse_cnt_name(p))

def _trial_id(t):
    return f"{t.subject}_s{t.session}_t{t.trial}"

def _deterministic_split(trials, seed):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(trials))
    n_train = int(0.8 * len(trials))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    return [trials[i] for i in train_idx], [trials[i] for i in test_idx]

def _cov_from_cat(x_cat, eps):
    # Time de-mean per window
    x_c = x_cat - x_cat.mean(axis=1, keepdims=True)
    denom = max(1, x_c.shape[1] - 1)
    cov = (x_c @ x_c.T) / float(denom)
    cov = 0.5 * (cov + cov.T)
    cov = cov + np.eye(cov.shape[0], dtype=cov.dtype) * eps
    return cov

def process_trials_to_cov(trials, bands, raw_cache, cfg):
    cov_list = []
    y_list = []
    tid_list = []
    
    # Group by CNT
    by_cnt = {}
    for t in trials:
        by_cnt.setdefault(t["cnt_path"], []).append(t)
        
    for cnt_path, t_rows in sorted(by_cnt.items()):
        # Load Raw
        if cnt_path in raw_cache:
            raw62 = raw_cache[cnt_path]
        else:
            raw = load_one_raw(cnt_path, backend=cfg["raw_backend"], preload=False)
            raw62, _ = build_eeg62_view(raw, "data/SEED/channel_62_pos.locs")
            raw_cache[cnt_path] = raw62
            
        t_objs = [r["trial_obj"] for r in t_rows]
        segments = slice_raw_trials(raw62, t_objs, trial_offset_sec=0.0)
        seg_map = {f"{m['subject']}_s{m['session']}_t{m['trial']}": seg for seg, m in segments}
        
        fs = float(raw62.info["sfreq"])
        
        for r in t_rows:
            tid = r["trial_id"]
            if tid not in seg_map: continue
            seg = seg_map[tid]
            
            # Bandpass Full Trial
            # (Optimization: Bandpass once for the whole CNT is better but harder with slice logic)
            # We bandpass the sliced trial.
            band_data = {b.name: bandpass(seg, fs, b) for b in bands}
            
            # Windowing
            n_samples = seg.shape[1]
            w_list = window_slices(n_samples, fs, cfg["window_sec"], cfg["hop_sec"])
            
            for s, e in w_list:
                # Concat bands: (C, T) -> (C*5, T)? Or (5, C, T)?
                # Standard manifold: usually per-band covariance or concatenated channels.
                # "C2. Cov construction ... epsI" implies single covariance matrix.
                # Usually we concat channels: (62*5, T).
                # Wait, huge matrix (310x310). 
                # Let's check prompt "x in R^{B x 62 x T}".
                # Prompt says: "Data input: x e R^{B x 62 x T}".
                # "C2. Cov... C = XX^T". This implies 62x62 Covariance.
                # So we compute Cov *per band* and stack? Or just one band?
                # "B1. Input signal: processed raw EEG".
                # "A3. Preprocess ... bandpass/notch".
                # If we have 5 bands, do we have 5 covariances?
                # Prompt: "RawCov-Manifold ... win=4s ... Global Log-Euclid ... Shallow TSM".
                # TSM usually handles SPD matrices. 
                # If we have 5 bands, we typically have a set of SPDs.
                # Let's assume we maintain **separate** covariances for each band (5 x 62x62) or stack channel-wise?
                # Stacking channel-wise (310x310) is heavy.
                # "Riemannian Procrustes Analysis" paper uses block-diagonal or separate.
                # Given "RawCov" implication (Single Covariance on Raw?), maybe just Broad Band?
                # But `bands` config is present in previous step.
                # Let's see: `features.py` had `bands` config.
                # Step 1 script concatenated bands: `x_cat = np.concatenate(band_windows, axis=1) # (62, T*5)`.
                # This results in 62x62 Covariance (Cov of 62 channels, with 5x samples).
                # This is "Covariance Pooling across frequencies".
                # Let's stick to this as it produces 62x62.
                
                band_chunks = []
                for b in bands:
                    # Normalize band?
                    chunk = band_data[b.name][:, s:e]
                    # Global Z per band?
                    # mean/std
                    m = chunk.mean()
                    sd = chunk.std() + 1e-6
                    chunk = (chunk - m) / sd
                    band_chunks.append(chunk)
                
                # Concatenate along Time (dim 1)
                x_cat = np.concatenate(band_chunks, axis=1) # (62, T_win * 5)
                
                cov = _cov_from_cat(x_cat, cfg["spd_eps"])
                cov_list.append(cov)
                y_list.append(r["label"])
                tid_list.append(tid)
                
    return np.array(cov_list), np.array(y_list), tid_list

def main():
    cfg = CONFIG
    out_dir = cfg["out_root"]
    ensure_dir(out_dir)
    
    # 1. Load Index
    root = cfg["seed_raw_root"]
    cnt_files = _sorted_cnt_files(root)
    time_txt = os.path.join(root, "time.txt")
    if not os.path.exists(time_txt): time_txt = "data/SEED/SEED_EEG/time.txt"
    stim_xlsx = "data/SEED/SEED_EEG/SEED_stimulation.xlsx"
    
    trials_all = []
    for cnt_path in cnt_files:
        curr = build_trial_index(cnt_path, time_txt, stim_xlsx)
        for t in curr:
            trials_all.append({
                "trial_id": _trial_id(t),
                "label": int(t.label),
                "trial_obj": t,
                "cnt_path": cnt_path
            })
    trials_all.sort(key=lambda x: x["trial_id"])
    
    # 2. Split
    train_rows, test_rows = _deterministic_split(trials_all, cfg["seed"])
    
    # 3. Feature Extraction (Covariances)
    bands = parse_band_spec(cfg["bands"])
    raw_cache = {}
    
    print("Extracting Train Covariances...")
    cov_train, y_train, tid_train = process_trials_to_cov(train_rows, bands, raw_cache, cfg)
    print(f"Train: {cov_train.shape}")
    
    print("Extracting Test Covariances...")
    cov_test, y_test, tid_test = process_trials_to_cov(test_rows, bands, raw_cache, cfg)
    print(f"Test: {cov_test.shape}")
    
    # 4. Global Log-Euclid Centering
    # C3. "Strictly use TRAIN stats"
    print("Applying Global Log-Euclid Centering...")
    
    # Map to Tangent Space (LogMap)
    # T_i = logm(C_i)
    # We computed C_i. Now we compute T_i.
    
    # Note: Logic: T_centered = Log(C) - Log(C_bar) 
    # where Log(C_bar) = Mean(Log(C_train))
    
    def to_tangent(covs, eps):
        ts = []
        for c in covs:
            # logmap_spd returns (V * log(S)) @ V.T
            # This is Log(C)
            t = logmap_spd(c, eps)
            ts.append(t)
        return np.array(ts)
        
    t_train = to_tangent(cov_train, cfg["spd_eps"])
    
    # Compute Geometric Mean in Log Domain (Arithmetic Mean of Tangents)
    t_bar = np.mean(t_train, axis=0) # (62, 62)
    
    # Centering (Transport to Identity roughly)
    t_train_centered = t_train - t_bar
    
    # Apply to Test
    t_test = to_tangent(cov_test, cfg["spd_eps"])
    t_test_centered = t_test - t_bar
    
    # Vectorize (Upper Triangle)
    def vectorize(ts):
        vs = []
        for t in ts:
            vs.append(vec_utri(t))
        return np.array(vs)
        
    X_train_vec = vectorize(t_train_centered)
    X_test_vec = vectorize(t_test_centered)
    
    # 5. Classifier (Shallow TSM)
    # Standardize first? SVM/LR likes standardized features.
    scaler = StandardScaler()
    X_train_final = scaler.fit_transform(X_train_vec)
    X_test_final = scaler.transform(X_test_vec)
    
    print("Training Classifier...")
    clf = LogisticRegression(
        max_iter=cfg["max_iter"], 
        multi_class='multinomial', 
        solver='lbfgs', 
        random_state=cfg["seed"]
    )
    clf.fit(X_train_final, y_train)
    
    # Predictions
    y_pred_win = clf.predict(X_test_final)
    y_proba_win = clf.predict_proba(X_test_final)
    win_acc = np.mean(y_pred_win == y_test)
    
    # Trial Aggregation (Mean Proba)
    trial_preds = {}
    for tid, y, prob in zip(tid_test, y_test, y_proba_win):
        if tid not in trial_preds:
            trial_preds[tid] = {"y": y, "probs": []}
        trial_preds[tid]["probs"].append(prob)
        
    correct_trial = 0
    total_trial = 0
    for tid, res in trial_preds.items():
        mean_p = np.mean(res["probs"], axis=0)
        pred = np.argmax(mean_p)
        if pred == res["y"]:
            correct_trial += 1
        total_trial += 1
        
    trial_acc = correct_trial / total_trial if total_trial > 0 else 0
    
    print(f"Results: Win Acc={win_acc:.4f}, Trial Acc={trial_acc:.4f}")
    
    # Diagnostics
    # Check for collapse
    pred_counts = Counter(y_pred_win)
    print(f"Prediction Dist: {pred_counts}")
    
    metrics = {
        "win_acc": win_acc,
        "trial_acc": trial_acc,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "pred_counts": dict(pred_counts)
    }
    write_json(f"{out_dir}/report_metrics.json", metrics)
    
    # Report Markdown
    with open(f"{out_dir}/SINGLE_RUN_REPORT.md", "w") as f:
        f.write("# Phase 14R Step 3: RawCov + Global Centering Report\n\n")
        f.write(f"- Window: {cfg['window_sec']}s / Hop: {cfg['hop_sec']}s\n")
        f.write(f"- Global Centering: Log-Euclid Mean on Train\n")
        f.write(f"- Classifier: Logistic Regression (max_iter={cfg['max_iter']})\n")
        f.write(f"- **Window Acc**: {win_acc:.4f}\n")
        f.write(f"- **Trial Acc**: {trial_acc:.4f}\n")
        f.write(f"- Prediction Distribution: {dict(pred_counts)}\n")

if __name__ == "__main__":
    main()
