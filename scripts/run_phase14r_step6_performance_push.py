#!/usr/bin/env python
"""Phase 14R Step 6: ProcessedCov Performance Push.

Goal:
1. Validate 1s window viability for DCNet.
2. Improve covariance estimation with OAS shrinkage.
3. Push performance with Geometry-aware classifiers.

Sweeps:
- Window: [1.0, 2.0, 4.0]
- Cov Estimator: [empirical, oas]
- Norm Mode: [logcenter, ra]
- Classifier: LogisticRegression(C=1.0, solver=saga)
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
from sklearn.covariance import OAS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.seed_processed_trials import SeedProcessedTrialDataset
from manifold_raw.features import parse_band_spec, window_slices, bandpass

DEFAULT_CONFIG = {
    "processed_root": "data/SEED/SEED_EEG/Preprocessed_EEG",
    "stim_xlsx": "data/SEED/SEED_EEG/SEED_stimulation.xlsx",
    "bands": "delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50",
    "spd_eps": 1e-4,
    "out_root_base": "promoted_results/phase14r/step6/seed1"
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
    # Expm to get back to SPD
    vals, vecs = np.linalg.eigh(mean_log)
    return (vecs * np.exp(vals)) @ vecs.T

# --- Covariance Extraction ---

def _cov_empirical(x_cat: np.ndarray, eps: float) -> np.ndarray:
    x_c = x_cat - x_cat.mean(axis=1, keepdims=True)
    denom = max(1, x_c.shape[1] - 1)
    cov = (x_c @ x_c.T) / float(denom)
    return regularize_spd(cov, eps)

def _cov_oas(x_cat: np.ndarray, eps: float) -> np.ndarray:
    # OAS assumes samples are rows, features are cols
    # x_cat is (n_features, n_samples)
    try:
        oa = OAS(assume_centered=False)
        oa.fit(x_cat.T)
        return regularize_spd(oa.covariance_, eps)
    except Exception:
        # Fallback if OAS fails (singular etc)
        return _cov_empirical(x_cat, eps)


def extract_covs_from_precomputed(trials_data: List[Dict], 
                                  win_sec: float, hop_sec: float, 
                                  est_mode: str, spd_eps: float) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract covariance matrices from pre-computed band data."""
    cov_list = []
    y_list = []
    tid_list = []
    
    print(f"  Extracting {est_mode} covs (win={win_sec}s)...", end=" ", flush=True)
    
    for i, t in enumerate(trials_data):
        # t has keys: tid, label, fs, bands (dict of arrays)
        # x_raw shape needed? No, we have bands.
        # But we need n_samples to slice.
        # Assume bands['delta'] has shape (n_channels, n_samples)
        
        n_samples = t["bands"][list(t["bands"].keys())[0]].shape[1]
        fs = t["fs"]
        
        w_list = window_slices(n_samples, fs, win_sec, hop_sec)
        
        # Pre-allocate array for speed?
        # Actually list append is fine for now.
        
        for s, e in w_list:
            band_chunks = []
            # band order matters? 
            # We should iterate sorted keys or specific list.
            # let's assume bands passed in are robust.
            # But t["bands"] is a dict.
            # We need a fixed order.
            # Let's trust the pre-computation order.
            
            for b_name, b_data in t["bands"].items():
                chunk = b_data[:, s:e]
                m = chunk.mean()
                sd = chunk.std() + 1e-6
                chunk = (chunk - m) / sd
                band_chunks.append(chunk)

            x_cat = np.concatenate(band_chunks, axis=1)
            
            if est_mode == "empirical":
                cov = _cov_empirical(x_cat, spd_eps)
            elif est_mode == "oas":
                cov = _cov_oas(x_cat, spd_eps)
            else:
                raise ValueError(f"Unknown cov estimator: {est_mode}")
                
            cov_list.append(cov)
            y_list.append(t["label"])
            tid_list.append(t["tid"])
            
    print(f"Done. {len(cov_list)} windows.")
    return np.array(cov_list), np.array(y_list), tid_list

from joblib import Parallel, delayed

def precompute_trial(t, bands_spec):
    """Helper for parallel execution."""
    tid = t["trial_id_str"]
    x_raw = t["x_trial"]
    fs = t["sfreq"]
    label = t["label"]
    
    b_data = {}
    for b in bands_spec:
        b_data[b.name] = bandpass(x_raw, fs, b)
        
    return {
        "tid": tid,
        "label": label,
        "fs": fs,
        "bands": b_data
    }

def precompute_bands(trials: List[Dict], bands_spec) -> List[Dict]:
    """Pre-compute bandpassed data for all trials in parallel."""
    print(f"  Pre-computing band filters (n={len(trials)})...", end=" ", flush=True)
    
    processed = Parallel(n_jobs=4, verbose=1)(
        delayed(precompute_trial)(t, bands_spec) for t in trials
    )
            
    print(f"Done.")
    return processed

def run_single_seed(seed: int, cfg: Dict, all_trials: List[Dict], args) -> List[Dict]:
    """Run all sweep combinations for a single seed."""
    
    print(f"\n{'='*60}")
    print(f"Running Seed {seed}")
    print(f"{'='*60}")
    
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_trials))
    n_train = int(0.8 * len(all_trials))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    train_rows = [all_trials[i] for i in train_idx]
    test_rows = [all_trials[i] for i in test_idx]
    
    bands = parse_band_spec(cfg["bands"])
    
    # Pre-compute bands
    print("Pre-computing Train...")
    train_data = precompute_bands(train_rows, bands)
    print("Pre-computing Test...")
    test_data = precompute_bands(test_rows, bands)
    
    results = []
    
    # Sweep Parameters
    windows = [4.0, 2.0, 1.0] # Run 4s first as baseline
    est_modes = ["empirical", "oas"]
    norm_modes = ["logcenter", "ra"]
    
    for win_sec in windows:
        for est_mode in est_modes:
            
            # Extract Covariances from precomputed
            covs_train, y_train, tid_train = extract_covs_from_precomputed(
                train_data, win_sec, args.hop_sec, est_mode, cfg["spd_eps"]
            )
            covs_test, y_test, tid_test = extract_covs_from_precomputed(
                test_data, win_sec, args.hop_sec, est_mode, cfg["spd_eps"]
            )
            
            for norm_mode in norm_modes:
                variant_id = f"win{int(win_sec)}s_{est_mode}_{norm_mode}"
                print(f"  > Variant: {variant_id}...", end=" ")
                
                try:
                    res = run_variant(
                        seed, win_sec, est_mode, norm_mode,
                        covs_train, y_train, tid_train,
                        covs_test, y_test, tid_test,
                        cfg
                    )
                    
                    print(f"Trial Acc={res['trial_acc']:.4f}")
                    
                    # Save artifacts per variant
                    out_dir = f"{cfg['out_root_base']}/seed{seed}/{variant_id}"
                    ensure_dir(out_dir)
                    write_json(f"{out_dir}/metrics.json", res)
                    
                    results.append(res)
                    
                except Exception as e:
                    print(f"FAILED: {e}")
                    import traceback
                    traceback.print_exc()

    return results

def apply_normalization(covs_train: np.ndarray, covs_test: np.ndarray, 
                        mode: str, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Apply normalization to covs. Returns (normed_train, normed_test)."""
    
    if mode == "logcenter":
        # Log-Euclid centering
        log_train = np.array([logm_spd(c, eps) for c in covs_train])
        mean_log = np.mean(log_train, axis=0)
        
        normed_train = log_train - mean_log
        
        log_test = np.array([logm_spd(c, eps) for c in covs_test])
        normed_test = log_test - mean_log
        
        return normed_train, normed_test
    
    elif mode == "ra":
        # Riemannian Alignment via whitening
        mean_spd = compute_mean_spd_logeuclidean(covs_train, eps)
        inv_sqrt_mean = invsqrtm_spd(mean_spd, eps)
        
        # Whiten
        normed_train = np.array([inv_sqrt_mean @ c @ inv_sqrt_mean for c in covs_train])
        normed_test = np.array([inv_sqrt_mean @ c @ inv_sqrt_mean for c in covs_test])
        
        # Project to tangent space at Identity
        normed_train = np.array([logm_spd(c, eps) for c in normed_train])
        normed_test = np.array([logm_spd(c, eps) for c in normed_test])
        
        return normed_train, normed_test
    
    else:
        raise ValueError(f"Unknown norm_mode: {mode}")

# --- Feature Extraction ---

def covs_to_features(covs: np.ndarray) -> np.ndarray:
    return np.array([vec_utri(c) for c in covs])

# --- Experiment Runner ---

def run_variant(seed: int, win_sec: float, est_mode: str, norm_mode: str,
                covs_train: np.ndarray, y_train: np.ndarray, tid_train: List[str],
                covs_test: np.ndarray, y_test: np.ndarray, tid_test: List[str],
                cfg: Dict) -> Dict:
    """Run a single variant and return metrics."""
    
    eps = cfg["spd_eps"]
    
    # Normalization
    normed_train, normed_test = apply_normalization(covs_train, covs_test, norm_mode, eps)
    
    # Feature Extraction
    X_train = covs_to_features(normed_train)
    X_test = covs_to_features(normed_test)
    
    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Classifier (Fixed C=1.0 based on prior sweep, solver=saga for robustness)
    clf = LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver='saga',
        random_state=seed,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred_win = clf.predict(X_test)
    y_proba_win = clf.predict_proba(X_test)
    win_acc = float(accuracy_score(y_test, y_pred_win))
    
    cm_win = confusion_matrix(y_test, y_pred_win, labels=[0, 1, 2])
    
    # Trial Aggregation
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
    
    cm_trial = confusion_matrix(y_true_trial, y_pred_trial, labels=[0, 1, 2])
    
    collapse_win = check_collapse(Counter(y_pred_win))
    collapse_trial = check_collapse(Counter(y_pred_trial))
    
    return {
        "seed": seed,
        "window_sec": win_sec,
        "est_mode": est_mode,
        "norm_mode": norm_mode,
        "win_acc": win_acc,
        "trial_acc": trial_acc,
        "macro_f1": macro_f1,
        "cm_win": cm_win.tolist(),
        "cm_trial": cm_trial.tolist(),
        "collapse_window": collapse_win["collapsed"],
        "collapse_trial": collapse_trial["collapsed"],
        "n_train_wins": len(y_train),
        "n_test_wins": len(y_test)
    }



def main():
    parser = argparse.ArgumentParser(description="Phase 14R Step 6: ProcessedCov Performance Push")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 4], help="Seeds to run")
    parser.add_argument("--hop_sec", type=float, default=1.0)
    args = parser.parse_args()
    
    cfg = DEFAULT_CONFIG.copy()
    
    print(f"Loading Processed Dataset from {cfg['processed_root']}...")
    ds = SeedProcessedTrialDataset(cfg["processed_root"], cfg["stim_xlsx"])
    all_trials = sorted(list(ds), key=lambda x: x["trial_id_str"])
    print(f"Loaded {len(all_trials)} trials.")
    
    all_results = []
    for seed in args.seeds:
        seed_results = run_single_seed(seed, cfg, all_trials, args)
        all_results.extend(seed_results)
    
    # Summary
    summary_path = f"{cfg['out_root_base']}/summary.csv"
    # flattening the conf mats for csv is messy, drop 'em
    simple_results = [{k:v for k,v in r.items() if not k.startswith("cm_")} for r in all_results]
    df = pd.DataFrame(simple_results)
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    
    # Report bests
    print("\n" + "="*60)
    print("Top Configurations per Seed (by Trial Acc):")
    print("="*60)
    for seed in args.seeds:
        sdf = df[df["seed"] == seed]
        if not sdf.empty:
            s_sorted = sdf.sort_values("trial_acc", ascending=False)
            print(f"\nSEED {seed}:")
            print(s_sorted[["window_sec", "est_mode", "norm_mode", "trial_acc", "win_acc"]].head(5))

if __name__ == "__main__":
    main()
