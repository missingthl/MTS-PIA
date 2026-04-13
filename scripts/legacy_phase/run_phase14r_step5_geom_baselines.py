#!/usr/bin/env python
"""Phase 14R Step 5: Riemannian Alignment + Tangent Space Baselines.

Geometry-aware methods to improve ProcessedCov beyond Step 4c (~71-75%).

Normalization modes:
- none: raw SPD
- logcenter: Log-Euclid mean centering (Step 4c)
- ra: Riemannian Alignment via whitening

Classifier sweep:
- LogisticRegression with C in {0.1, 1.0, 10.0}
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Tuple
from scipy.linalg import sqrtm, logm, inv

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets.seed_processed_trials import SeedProcessedTrialDataset
from manifold_raw.features import parse_band_spec, window_slices, bandpass

DEFAULT_CONFIG = {
    "processed_root": "data/SEED/SEED_EEG/Preprocessed_EEG",
    "stim_xlsx": "data/SEED/SEED_EEG/SEED_stimulation.xlsx",
    "window_sec": 4.0,
    "hop_sec": 1.0,
    "bands": "delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50",
    "spd_eps": 1e-4,
    "out_root_base": "promoted_results/phase14r/step5/seed1"
}

# --- Utilities ---

def json_sanitize(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives."""
    if isinstance(obj, dict):
        return {json_sanitize(k): json_sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Counter):
        return {int(k): int(v) for k, v in obj.items()}
    else:
        return obj

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def write_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(json_sanitize(obj), f, indent=2)

def check_collapse(pred_counts: Dict[int, int], threshold: float = 0.95) -> Dict:
    total = sum(pred_counts.values())
    if total == 0:
        return {"collapsed": True, "max_ratio": 1.0, "max_class": -1}
    max_count = max(pred_counts.values())
    max_class = max(pred_counts, key=pred_counts.get)
    ratio = max_count / total
    return {
        "collapsed": ratio >= threshold,
        "max_ratio": round(ratio, 4),
        "max_class": int(max_class)
    }

# --- SPD Operations ---

def regularize_spd(cov: np.ndarray, eps: float) -> np.ndarray:
    """Symmetrize and add eps*I."""
    cov = 0.5 * (cov + cov.T)
    cov = cov + np.eye(cov.shape[0], dtype=cov.dtype) * eps
    return cov

def logm_spd(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Log-map of SPD matrix."""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, eps)
    log_vals = np.log(vals)
    return (vecs * log_vals) @ vecs.T

def invsqrtm_spd(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Inverse square root of SPD matrix."""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, eps)
    inv_sqrt_vals = 1.0 / np.sqrt(vals)
    return (vecs * inv_sqrt_vals) @ vecs.T

def vec_utri(mat: np.ndarray) -> np.ndarray:
    """Vectorize upper-triangular (including diagonal)."""
    idx = np.triu_indices(mat.shape[0])
    return mat[idx]

def compute_mean_spd_logeuclidean(covs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Log-Euclidean mean of SPD matrices."""
    log_covs = np.array([logm_spd(c, eps) for c in covs])
    mean_log = np.mean(log_covs, axis=0)
    # Expm to get back to SPD (not needed for centering, but for RA)
    vals, vecs = np.linalg.eigh(mean_log)
    return (vecs * np.exp(vals)) @ vecs.T

# --- Covariance Extraction ---

def _cov_from_cat(x_cat: np.ndarray, eps: float) -> np.ndarray:
    x_c = x_cat - x_cat.mean(axis=1, keepdims=True)
    denom = max(1, x_c.shape[1] - 1)
    cov = (x_c @ x_c.T) / float(denom)
    return regularize_spd(cov, eps)

def extract_covs_from_trials(trials: List[Dict], bands, cfg: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract covariance matrices from trials."""
    cov_list = []
    y_list = []
    tid_list = []
    
    for i, t in enumerate(trials):
        if (i+1) % 20 == 0:
            print(f"  Processing trial {i+1}/{len(trials)}...", end="\r", flush=True)
            
        tid = t["trial_id_str"]
        x_raw = t["x_trial"]
        fs = t["sfreq"]
        label = t["label"]
        
        band_data = {}
        for b in bands:
            band_data[b.name] = bandpass(x_raw, fs, b)
            
        n_samples = x_raw.shape[1]
        w_list = window_slices(n_samples, fs, cfg["window_sec"], cfg["hop_sec"])
        
        for s, e in w_list:
            band_chunks = []
            for b in bands:
                chunk = band_data[b.name][:, s:e]
                m = chunk.mean()
                sd = chunk.std() + 1e-6
                chunk = (chunk - m) / sd
                band_chunks.append(chunk)

            x_cat = np.concatenate(band_chunks, axis=1)
            cov = _cov_from_cat(x_cat, cfg["spd_eps"])
            cov_list.append(cov)
            y_list.append(label)
            tid_list.append(tid)
            
    print(f"  Done processing {len(trials)} trials.           ")
    return np.array(cov_list), np.array(y_list), tid_list

# --- Normalization ---

def apply_normalization(covs_train: np.ndarray, covs_test: np.ndarray, 
                        mode: str, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Apply normalization to covs. Returns (normed_train, normed_test)."""
    
    if mode == "none":
        return covs_train, covs_test
    
    elif mode == "logcenter":
        # Log-Euclid centering (Step 4c style)
        log_train = np.array([logm_spd(c, eps) for c in covs_train])
        mean_log = np.mean(log_train, axis=0)
        
        normed_train = log_train - mean_log
        
        log_test = np.array([logm_spd(c, eps) for c in covs_test])
        normed_test = log_test - mean_log
        
        # These are already in tangent space (symmetric matrices)
        return normed_train, normed_test
    
    elif mode == "ra":
        # Riemannian Alignment via whitening
        mean_spd = compute_mean_spd_logeuclidean(covs_train, eps)
        inv_sqrt_mean = invsqrtm_spd(mean_spd, eps)
        
        # Whiten: C_ra = invsqrt(mean) @ C @ invsqrt(mean)
        normed_train = np.array([inv_sqrt_mean @ c @ inv_sqrt_mean for c in covs_train])
        normed_test = np.array([inv_sqrt_mean @ c @ inv_sqrt_mean for c in covs_test])
        
        # Then project to tangent space at Identity
        normed_train = np.array([logm_spd(c, eps) for c in normed_train])
        normed_test = np.array([logm_spd(c, eps) for c in normed_test])
        
        return normed_train, normed_test
    
    else:
        raise ValueError(f"Unknown norm_mode: {mode}")

# --- Feature Extraction ---

def covs_to_features(covs: np.ndarray) -> np.ndarray:
    """Convert covariance matrices (possibly already in tangent space) to feature vectors."""
    return np.array([vec_utri(c) for c in covs])

# --- Experiment Runner ---

def run_variant(seed: int, norm_mode: str, C_val: float, 
                covs_train: np.ndarray, y_train: np.ndarray, tid_train: List[str],
                covs_test: np.ndarray, y_test: np.ndarray, tid_test: List[str],
                cfg: Dict) -> Dict:
    """Run a single variant and return metrics."""
    
    eps = cfg["spd_eps"]
    
    # Apply normalization
    normed_train, normed_test = apply_normalization(covs_train, covs_test, norm_mode, eps)
    
    # Convert to features
    X_train = covs_to_features(normed_train)
    X_test = covs_to_features(normed_test)
    
    # StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Classifier
    clf = LogisticRegression(
        C=C_val,
        max_iter=1000,
        solver='lbfgs',
        random_state=seed
    )
    clf.fit(X_train, y_train)
    
    # Window-level predictions
    y_pred_win = clf.predict(X_test)
    y_proba_win = clf.predict_proba(X_test)
    win_acc = float(accuracy_score(y_test, y_pred_win))
    
    # Confusion matrix (window)
    cm_win = confusion_matrix(y_test, y_pred_win, labels=[0, 1, 2])
    
    # Trial aggregation
    trial_preds = {}
    for tid, y, prob in zip(tid_test, y_test, y_proba_win):
        if tid not in trial_preds:
            trial_preds[tid] = {"y": int(y), "probs": []}
        trial_preds[tid]["probs"].append(prob)
    
    y_true_trial = []
    y_pred_trial = []
    trial_ids = []
    for tid, res in sorted(trial_preds.items()):
        mean_p = np.mean(res["probs"], axis=0)
        pred = int(np.argmax(mean_p))
        y_true_trial.append(res["y"])
        y_pred_trial.append(pred)
        trial_ids.append(tid)
    
    trial_acc = float(accuracy_score(y_true_trial, y_pred_trial))
    macro_f1 = float(f1_score(y_true_trial, y_pred_trial, average='macro'))
    
    # Confusion matrix (trial)
    cm_trial = confusion_matrix(y_true_trial, y_pred_trial, labels=[0, 1, 2])
    
    # Pred counts
    pred_counts_win = Counter(y_pred_win)
    pred_counts_trial = Counter(y_pred_trial)
    
    # Collapse check
    collapse_win = check_collapse(pred_counts_win)
    collapse_trial = check_collapse(pred_counts_trial)
    
    return {
        "seed": seed,
        "norm_mode": norm_mode,
        "C": C_val,
        "win_acc": win_acc,
        "trial_acc": trial_acc,
        "macro_f1": macro_f1,
        "cm_win": cm_win,
        "cm_trial": cm_trial,
        "pred_counts_win": dict(pred_counts_win),
        "pred_counts_trial": dict(pred_counts_trial),
        "collapse_window": collapse_win["collapsed"],
        "collapse_trial": collapse_trial["collapsed"],
        "y_true_trial": y_true_trial,
        "y_pred_trial": y_pred_trial,
        "trial_ids": trial_ids
    }

def run_single_seed(seed: int, cfg: Dict, all_trials: List[Dict], args) -> List[Dict]:
    """Run all variants for a single seed."""
    
    print(f"\n{'='*60}")
    print(f"Running Seed {seed}")
    print(f"{'='*60}")
    
    # Split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_trials))
    n_train = int(0.8 * len(all_trials))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    train_rows = [all_trials[i] for i in train_idx]
    test_rows = [all_trials[i] for i in test_idx]
    
    print(f"Split: {len(train_rows)} Train, {len(test_rows)} Test")
    
    # Extract covariances
    bands = parse_band_spec(cfg["bands"])
    
    print("Extracting Train Covariances...")
    covs_train, y_train, tid_train = extract_covs_from_trials(train_rows, bands, cfg)
    
    print("Extracting Test Covariances...")
    covs_test, y_test, tid_test = extract_covs_from_trials(test_rows, bands, cfg)
    
    # Run all variants
    results = []
    norm_modes = ["none", "logcenter", "ra"]
    C_values = [0.1, 1.0, 10.0]
    
    for norm_mode in norm_modes:
        for C_val in C_values:
            variant_id = f"{norm_mode}_C{C_val}"
            print(f"  Running variant: {variant_id}...", end=" ")
            
            try:
                res = run_variant(
                    seed, norm_mode, C_val,
                    covs_train, y_train, tid_train,
                    covs_test, y_test, tid_test,
                    cfg
                )
                print(f"Win={res['win_acc']:.4f}, Trial={res['trial_acc']:.4f}")
                
                # Save artifacts
                out_dir = f"{cfg['out_root_base']}/seed{seed}/{variant_id}"
                ensure_dir(out_dir)
                
                # Metrics
                metrics = {
                    "seed": res["seed"],
                    "norm_mode": res["norm_mode"],
                    "C": res["C"],
                    "win_acc": res["win_acc"],
                    "trial_acc": res["trial_acc"],
                    "macro_f1": res["macro_f1"],
                    "collapse_window": res["collapse_window"],
                    "collapse_trial": res["collapse_trial"]
                }
                write_json(f"{out_dir}/metrics.json", metrics)
                
                # Confusion matrices
                pd.DataFrame(res["cm_win"], index=[0,1,2], columns=[0,1,2]).to_csv(f"{out_dir}/confusion_window.csv")
                pd.DataFrame(res["cm_trial"], index=[0,1,2], columns=[0,1,2]).to_csv(f"{out_dir}/confusion_trial.csv")
                
                # Pred counts
                write_json(f"{out_dir}/pred_counts_window.json", res["pred_counts_win"])
                write_json(f"{out_dir}/pred_counts_trial.json", res["pred_counts_trial"])
                
                # ytrue_ypred_trial
                pd.DataFrame({
                    "trial_id": res["trial_ids"],
                    "y_true": res["y_true_trial"],
                    "y_pred": res["y_pred_trial"]
                }).to_csv(f"{out_dir}/ytrue_ypred_trial.csv", index=False)
                
                results.append(metrics)
                
            except Exception as e:
                print(f"FAILED: {e}")
                results.append({
                    "seed": seed,
                    "norm_mode": norm_mode,
                    "C": C_val,
                    "win_acc": -1,
                    "trial_acc": -1,
                    "macro_f1": -1,
                    "collapse_window": True,
                    "collapse_trial": True,
                    "error": str(e)
                })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Phase 14R Step 5: Geometry Baselines")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 4], help="Seeds to run")
    parser.add_argument("--window_sec", type=float, default=4.0)
    parser.add_argument("--hop_sec", type=float, default=1.0)
    parser.add_argument("--sfreq", type=float, default=200.0)
    args = parser.parse_args()
    
    cfg = DEFAULT_CONFIG.copy()
    cfg["window_sec"] = args.window_sec
    cfg["hop_sec"] = args.hop_sec
    
    print(f"Loading Processed Dataset from {cfg['processed_root']}...")
    ds = SeedProcessedTrialDataset(cfg["processed_root"], cfg["stim_xlsx"])
    all_trials = sorted(list(ds), key=lambda x: x["trial_id_str"])
    print(f"Loaded {len(all_trials)} trials.")
    
    all_results = []
    for seed in args.seeds:
        seed_results = run_single_seed(seed, cfg, all_trials, args)
        all_results.extend(seed_results)
    
    # Summary CSV
    summary_path = f"{cfg['out_root_base']}/summary.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    
    # Print best per seed
    print("\n" + "="*60)
    print("Best Variants per Seed:")
    print("="*60)
    for seed in args.seeds:
        seed_df = df[df["seed"] == seed]
        if len(seed_df) > 0 and "trial_acc" in seed_df.columns:
            best = seed_df.loc[seed_df["trial_acc"].idxmax()]
            print(f"Seed {seed}: {best['norm_mode']}_C{best['C']} -> Trial={best['trial_acc']:.4f}, Win={best['win_acc']:.4f}")

if __name__ == "__main__":
    main()
