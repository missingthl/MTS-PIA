#!/usr/bin/env python
"""Phase 14R Step 6: ProcessedCov Performance Push (Promoted Ops Run).

Goal:
- Push ProcessedCov baseline performance.
- Evaluate Covariance Estimators (Sample, OAS, LW).
- Sweep Window Lengths (1s, 2s, 4s).
- Compare Classifiers (LR-SAGA, LinearSVC).
- Generate "Ops Only" artifacts.
"""

import os
import sys
import json
import argparse
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Tuple
from scipy.linalg import sqrtm, logm, inv

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.covariance import OAS, LedoitWolf
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.seed_processed_trials import SeedProcessedTrialDataset
from manifold_raw.features import parse_band_spec, window_slices, bandpass

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

def vec_utri(mat: np.ndarray) -> np.ndarray:
    """Vectorize upper-triangular (including diagonal)."""
    idx = np.triu_indices(mat.shape[0])
    return mat[idx]

# --- Covariance Estimation ---

def _cov_empirical(x_cat: np.ndarray, eps: float) -> np.ndarray:
    # x_cat: (n_features, n_samples)
    x_c = x_cat - x_cat.mean(axis=1, keepdims=True)
    denom = max(1, x_c.shape[1] - 1)
    cov = (x_c @ x_c.T) / float(denom)
    return regularize_spd(cov, eps)

def _cov_oas(x_cat: np.ndarray, eps: float) -> np.ndarray:
    # OAS assumes samples are rows
    try:
        oa = OAS(assume_centered=False)
        oa.fit(x_cat.T)
        return regularize_spd(oa.covariance_, eps)
    except Exception:
        return _cov_empirical(x_cat, eps)

def _cov_lw(x_cat: np.ndarray, eps: float) -> np.ndarray:
    # LedoitWolf assumes samples are rows
    try:
        lw = LedoitWolf(assume_centered=False)
        lw.fit(x_cat.T)
        return regularize_spd(lw.covariance_, eps)
    except Exception:
        return _cov_empirical(x_cat, eps)

# --- Feature Pipeline ---

def extract_covs_from_precomputed(trials_data: List[Dict], 
                                  win_sec: float, hop_sec: float, 
                                  est_mode: str, spd_eps: float) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    """Extract covariance matrices from pre-computed band data."""
    cov_list = []
    y_list = []
    tid_list = []
    win_idx_list = []
    
    print(f"  Extracting {est_mode} covs (win={win_sec}s)...", end=" ", flush=True)
    
    for i, t in enumerate(trials_data):
        # t: {tid, label, fs, bands}
        # Assuming sorted band keys order is consistent (delta, theta, alpha, beta, gamma)
        band_names = sorted(t["bands"].keys())
        first_band = t["bands"][band_names[0]]
        
        n_samples = first_band.shape[1]
        fs = t["fs"]
        w_list = window_slices(n_samples, fs, win_sec, hop_sec)
        
        for w_idx, (s, e) in enumerate(w_list):
            band_chunks = []
            for b_name in band_names:
                # Time-centering within window
                chunk = t["bands"][b_name][:, s:e]
                m = chunk.mean() 
                sd = chunk.std() + 1e-6
                chunk = (chunk - m) / sd
                band_chunks.append(chunk)

            x_cat = np.concatenate(band_chunks, axis=1)
            
            if est_mode == "sample":
                cov = _cov_empirical(x_cat, spd_eps)
            elif est_mode == "oas":
                cov = _cov_oas(x_cat, spd_eps)
            elif est_mode == "ledoitwolf":
                cov = _cov_lw(x_cat, spd_eps)
            else:
                raise ValueError(f"Unknown cov estimator: {est_mode}")
                
            cov_list.append(cov)
            y_list.append(t["label"])
            tid_list.append(t["tid"])
            win_idx_list.append(w_idx)
            
    print(f"Done. {len(cov_list)} windows.")
    return np.array(cov_list), np.array(y_list), tid_list, win_idx_list

def apply_logcenter(covs_train: np.ndarray, covs_test: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Log-Euclidean Global Centering (Train only mean)."""
    log_train = np.array([logm_spd(c, eps) for c in covs_train])
    mean_log = np.mean(log_train, axis=0)
    
    start_train = log_train - mean_log
    
    log_test = np.array([logm_spd(c, eps) for c in covs_test])
    start_test = log_test - mean_log
    
    return start_train, start_test

def covs_to_features(covs: np.ndarray) -> np.ndarray:
    return np.array([vec_utri(c) for c in covs])

# --- Experiment Runner ---

def run_variant(seed: int, win_sec: float, est_mode: str, clf_name: str, C_val: float, agg_mode: str,
                X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray, tid_test: List[str],
                cfg: Dict, output_dir: str) -> Dict:
    
    # Standardize
    if cfg["standardize"]:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
    else:
        X_train_s, X_test_s = X_train, X_test
        
    # Classifier
    if clf_name == "lr_saga":
        # Use lbfgs instead of saga - much faster for medium-sized data
        clf = LogisticRegression(solver="lbfgs", multi_class="multinomial", penalty="l2",
                                 C=C_val, max_iter=cfg["max_iter"], tol=1e-4, 
                                 random_state=seed, n_jobs=-1)
    elif clf_name == "linear_svc":
        clf = LinearSVC(C=C_val, max_iter=cfg["max_iter"], random_state=seed, dual="auto")
    else:
        raise ValueError(f"Unknown classifier: {clf_name}")
        
    # Fit & Predict
    with ignore_warnings(category=ConvergenceWarning):
        clf.fit(X_train_s, y_train)
        
    y_pred_win = clf.predict(X_test_s)
    
    # Probs for aggregation
    if hasattr(clf, "predict_proba"):
        y_proba_win = clf.predict_proba(X_test_s)
    elif hasattr(clf, "decision_function"):
        # Sigmoid calibration or just softmax on decision function?
        # For LinearSVC, decision_function is (n_samples, n_classes)
        # We can use simple softmax for aggregation purposes or voting
        d = clf.decision_function(X_test_s)
        # Softmax
        e_d = np.exp(d - np.max(d, axis=1, keepdims=True))
        y_proba_win = e_d / e_d.sum(axis=1, keepdims=True)
    else:
        # Fallback to one-hot for voting
        n_classes = len(np.unique(y_train))
        y_proba_win = np.zeros((len(y_pred_win), n_classes))
        for i, yp in enumerate(y_pred_win):
            y_proba_win[i, int(yp)] = 1.0

    win_acc = float(accuracy_score(y_test, y_pred_win))
    
    # Aggregation
    trial_preds = {}
    for i, tid in enumerate(tid_test):
        if tid not in trial_preds:
            trial_preds[tid] = {"y": int(y_test[i]), "probs": [], "votes": []}
        trial_preds[tid]["probs"].append(y_proba_win[i])
        trial_preds[tid]["votes"].append(y_pred_win[i])
        
    y_true_trial = []
    y_pred_trial = []
    trial_ids = []
    
    for tid, res in sorted(trial_preds.items()):
        y_true_trial.append(res["y"])
        trial_ids.append(tid)
        
        if agg_mode == "meanlogit":
            mean_p = np.mean(res["probs"], axis=0)
            pred = int(np.argmax(mean_p))
        elif agg_mode == "majority":
            pred = int(Counter(res["votes"]).most_common(1)[0][0])
        else:
            pred = int(res["votes"][0]) 
            
        y_pred_trial.append(pred)
        
    trial_acc = float(accuracy_score(y_true_trial, y_pred_trial))
    
    # Artifacts
    cm_win = confusion_matrix(y_test, y_pred_win, labels=[0, 1, 2])
    cm_trial = confusion_matrix(y_true_trial, y_pred_trial, labels=[0, 1, 2])
    
    collapse_win = check_collapse(Counter(y_pred_win))
    collapse_trial = check_collapse(Counter(y_pred_trial))
    
    # Files
    # Handle n_iter_ for convergence check - LR has array, LinearSVC has int
    if hasattr(clf, "n_iter_"):
        n_iter = clf.n_iter_[0] if isinstance(clf.n_iter_, np.ndarray) else clf.n_iter_
        converged = bool(n_iter < cfg["max_iter"])
    else:
        converged = True
    write_json(f"{output_dir}/metrics.json", {
        "win_acc": win_acc,
        "trial_acc": trial_acc,
        "collapse_window": collapse_win,
        "collapse_trial": collapse_trial,
        "converged": converged
    })
    
    pd.DataFrame(cm_win).to_csv(f"{output_dir}/confusion_window.csv", index=False)
    pd.DataFrame(cm_trial).to_csv(f"{output_dir}/confusion_trial.csv", index=False)
    
    pd.DataFrame({"trial_id": trial_ids, "y_true": y_true_trial, "y_pred": y_pred_trial}).to_csv(
        f"{output_dir}/ytrue_ypred_trial.csv", index=False)
        
    # Run meta
    write_json(f"{output_dir}/run_meta.json", {
        "seed": seed, "window": win_sec, "cov_est": est_mode, "clf": clf_name, 
        "C": C_val, "agg": agg_mode, "n_train_win": len(y_train)
    })
    
    return {
        "seed": seed, "window_sec": win_sec, "hop_sec": cfg["hop_sec"],
        "cov_est": est_mode, "center": cfg["center"], "clf": clf_name,
        "C": C_val, "agg": agg_mode, "standardize": cfg["standardize"],
        "win_acc": win_acc, "trial_acc": trial_acc, 
        "collapse_win": collapse_win["collapsed"],
        "collapse_trial": collapse_trial["collapsed"]
    }

def precompute_bands(trials: List[Dict], bands_spec) -> List[Dict]:
    """Sequential pre-compute to be safe on RAM."""
    processed = []
    print("  Pre-computing bands...", end=" ", flush=True)
    for i, t in enumerate(trials):
        if i % 100 == 0: print(f"{i}...", end=" ", flush=True)
        b_data = {b.name: bandpass(t["x_trial"], t["sfreq"], b) for b in bands_spec}
        processed.append({
            "tid": t["trial_id_str"], "label": t["label"], 
            "fs": t["sfreq"], "bands": b_data
        })
    print("Done.")
    return processed

def run_single_seed(seed: int, cfg: Dict, all_trials: List[Dict]) -> List[Dict]:
    print(f"\nRunning Seed {seed}")
    
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_trials))
    n_train = int(0.8 * len(all_trials))
    train_data = [all_trials[i] for i in indices[:n_train]]
    test_data = [all_trials[i] for i in indices[n_train:]]
    
    bands = parse_band_spec(cfg["bands"])
    
    # Pre-compute
    train_pre = precompute_bands(train_data, bands)
    test_pre = precompute_bands(test_data, bands)
    
    seed_results = []
    
    # Loops
    for win_sec in cfg["windows"]:
        for est_mode in cfg["cov_est"]:
            
            # Extract
            covs_train, y_train, _, _ = extract_covs_from_precomputed(
                train_pre, win_sec, cfg["hop_sec"], est_mode, cfg["spd_eps"])
            covs_test, y_test, tid_test, _ = extract_covs_from_precomputed(
                test_pre, win_sec, cfg["hop_sec"], est_mode, cfg["spd_eps"])
            
            # Center (LogCenter only for now)
            if cfg["center"] == "logcenter":
                covs_train, covs_test = apply_logcenter(covs_train, covs_test, cfg["spd_eps"])
                
            # Vectorize
            X_train = covs_to_features(covs_train)
            X_test = covs_to_features(covs_test)
            
            for clf_name in cfg["clf"]:
                for C_val in cfg["C_list"]:
                    for agg in cfg["trial_agg"]:
                        
                        variant_name = f"w{int(win_sec)}s_{est_mode}_{clf_name}_C{C_val}_{agg}"
                        out_dir = f"{cfg['out_root']}/seed1/seed{seed}/{variant_name}"
                        
                        print(f"  > {variant_name}...", end=" ")
                        
                        try:
                            res = run_variant(
                                seed, win_sec, est_mode, clf_name, C_val, agg,
                                X_train, y_train, X_test, y_test, tid_test,
                                cfg, out_dir
                            )
                            print(f"Trial Acc: {res['trial_acc']:.4f}")
                            seed_results.append(res)
                        except Exception as e:
                            print(f"FAILED: {e}")
                            import traceback
                            traceback.print_exc()
                            
    return seed_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 4])
    parser.add_argument("--processed-root", default="data/SEED/SEED_EEG/Preprocessed_EEG")
    parser.add_argument("--sfreq", type=float, default=200)
    parser.add_argument("--windows", nargs="+", type=float, required=True)
    parser.add_argument("--hop", type=float, default=1.0)
    parser.add_argument("--cov-est", nargs="+", default=["sample", "oas", "ledoitwolf"])
    parser.add_argument("--center", default="logcenter")
    parser.add_argument("--clf", nargs="+", default=["lr_saga", "linear_svc"])
    parser.add_argument("--C-list", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--standardize", default="on")
    parser.add_argument("--trial-agg", nargs="+", default=["meanlogit", "majority"])
    
    args = parser.parse_args()
    
    cfg = vars(args)
    cfg["bands"] = "delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50"
    cfg["spd_eps"] = 1e-4
    cfg["out_root"] = "promoted_results/phase14r/step6"
    cfg["stim_xlsx"] = "data/SEED/SEED_EEG/SEED_stimulation.xlsx"
    cfg["standardize"] = (cfg["standardize"] == "on")
    cfg["hop_sec"] = cfg["hop"]  # Map 'hop' argument to 'hop_sec' for internal code
    
    # Load
    print(f"Loading {cfg['processed_root']}...")
    ds = SeedProcessedTrialDataset(cfg["processed_root"], cfg["stim_xlsx"])
    all_trials = sorted(list(ds), key=lambda x: x["trial_id_str"])
    print(f"Loaded {len(all_trials)} trials.")
    
    all_results = []
    for seed in args.seeds:
        res = run_single_seed(seed, cfg, all_trials)
        all_results.extend(res)
        
    # Summary
    df = pd.DataFrame(all_results)
    df.to_csv(f"{cfg['out_root']}/seed1/summary.csv", index=False)
    print(f"Saved summary to {cfg['out_root']}/seed1/summary.csv")

if __name__ == "__main__":
    main()
