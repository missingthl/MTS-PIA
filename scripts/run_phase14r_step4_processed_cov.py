#!/usr/bin/env python
"""Phase 14R Step 4c: ProcessedCov Model with Full Evidence Chain.

Features:
- JSON sanitization (numpy types -> Python natives)
- Confusion matrices (window + trial)
- Collapse check with explicit JSON output
- ytrue_ypred_trial.csv
- CLI: --seeds, --max_iter, --standardize
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.seed_processed_trials import SeedProcessedTrialDataset
from manifold_raw.features import parse_band_spec, window_slices, logmap_spd, vec_utri, bandpass

DEFAULT_CONFIG = {
    "processed_root": "data/SEED/SEED_EEG/Preprocessed_EEG",
    "stim_xlsx": "data/SEED/SEED_EEG/SEED_stimulation.xlsx",
    "window_sec": 4.0,
    "hop_sec": 1.0,
    "bands": "delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50",
    "spd_eps": 1e-4,
    "out_root_base": "promoted_results/phase14r/step4c/seed1"
}

# --- Utilities ---

def json_sanitize(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        # Convert keys to int if they are numeric numpy types, else str
        sanitized = {}
        for k, v in obj.items():
            if isinstance(k, (np.integer,)):
                new_key = int(k)
            elif isinstance(k, (np.floating,)):
                new_key = float(k)
            else:
                new_key = k
            sanitized[new_key] = json_sanitize(v)
        return sanitized
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
    sanitized = json_sanitize(obj)
    with open(path, "w") as f:
        json.dump(sanitized, f, indent=2)

def check_collapse(pred_counts: Dict[int, int], threshold: float = 0.95) -> Dict:
    """Return collapse check results."""
    total = sum(pred_counts.values())
    if total == 0:
        return {"collapsed": True, "max_ratio": 1.0, "max_class": -1}
    max_count = max(pred_counts.values())
    max_class = max(pred_counts, key=pred_counts.get)
    ratio = max_count / total
    return {
        "collapsed": ratio >= threshold,
        "max_ratio": round(ratio, 4),
        "max_class": int(max_class),
        "counts": {int(k): int(v) for k, v in pred_counts.items()}
    }

def _cov_from_cat(x_cat, eps):
    x_c = x_cat - x_cat.mean(axis=1, keepdims=True)
    denom = max(1, x_c.shape[1] - 1)
    cov = (x_c @ x_c.T) / float(denom)
    cov = 0.5 * (cov + cov.T)
    cov = cov + np.eye(cov.shape[0], dtype=cov.dtype) * eps
    return cov

def process_trials_to_cov(trials, bands, cfg):
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

def run_single_seed(seed: int, cfg: Dict, all_trials: List[Dict], args) -> Dict:
    """Run full pipeline for a single seed. Returns metrics dict."""
    out_dir = f"{cfg['out_root_base']}/seed{seed}"
    ensure_dir(out_dir)
    
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
    
    # Feature Extraction
    bands = parse_band_spec(cfg["bands"])
    
    print("Extracting Train Covariances...")
    cov_train, y_train, tid_train = process_trials_to_cov(train_rows, bands, cfg)
    
    print("Extracting Test Covariances...")
    cov_test, y_test, tid_test = process_trials_to_cov(test_rows, bands, cfg)
    
    # Global Log-Euclid Centering
    print("Applying Global Log-Euclid Centering...")
    
    def to_tangent(covs, eps):
        return np.array([logmap_spd(c, eps) for c in covs])
        
    t_train = to_tangent(cov_train, cfg["spd_eps"])
    t_bar = np.mean(t_train, axis=0)
    t_train_centered = t_train - t_bar
    
    t_test = to_tangent(cov_test, cfg["spd_eps"])
    t_test_centered = t_test - t_bar
    
    def vectorize(ts):
        return np.array([vec_utri(t) for t in ts])
        
    X_train_vec = vectorize(t_train_centered)
    X_test_vec = vectorize(t_test_centered)
    
    # Optional StandardScaler
    if args.standardize:
        print("Applying StandardScaler...")
        scaler = StandardScaler()
        X_train_final = scaler.fit_transform(X_train_vec)
        X_test_final = scaler.transform(X_test_vec)
    else:
        X_train_final = X_train_vec
        X_test_final = X_test_vec
    
    # Classifier
    print(f"Training Classifier (max_iter={args.max_iter})...")
    clf = LogisticRegression(
        max_iter=args.max_iter, 
        solver='lbfgs', 
        random_state=seed
    )
    clf.fit(X_train_final, y_train)
    
    # Convergence info
    n_iter = clf.n_iter_[0] if hasattr(clf, 'n_iter_') else -1
    converged = n_iter < args.max_iter
    print(f"  Converged: {converged} (n_iter={n_iter}/{args.max_iter})")
    
    # Window-level Predictions
    y_pred_win = clf.predict(X_test_final)
    y_proba_win = clf.predict_proba(X_test_final)
    win_acc = float(np.mean(y_pred_win == y_test))
    
    # Window Confusion Matrix
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
    
    trial_acc = float(np.mean(np.array(y_pred_trial) == np.array(y_true_trial)))
    
    # Trial Confusion Matrix
    cm_trial = confusion_matrix(y_true_trial, y_pred_trial, labels=[0, 1, 2])
    
    # Pred Counts
    pred_counts_win = Counter(y_pred_win)
    pred_counts_trial = Counter(y_pred_trial)
    
    # Collapse Check
    collapse_win = check_collapse(pred_counts_win)
    collapse_trial = check_collapse(pred_counts_trial)
    
    print(f"Results: Win Acc={win_acc:.4f}, Trial Acc={trial_acc:.4f}")
    print(f"Window Pred Counts: {dict(pred_counts_win)}")
    print(f"Trial Pred Counts: {dict(pred_counts_trial)}")
    print(f"Collapse (Win): {collapse_win['collapsed']} (ratio={collapse_win['max_ratio']:.3f})")
    print(f"Collapse (Trial): {collapse_trial['collapsed']} (ratio={collapse_trial['max_ratio']:.3f})")
    
    # --- Save Artifacts ---
    
    # Metrics JSON
    metrics = {
        "seed": seed,
        "win_acc": win_acc,
        "trial_acc": trial_acc,
        "n_train_trials": len(train_rows),
        "n_test_trials": len(test_rows),
        "n_train_windows": int(len(y_train)),
        "n_test_windows": int(len(y_test)),
        "max_iter": args.max_iter,
        "standardize": args.standardize,
        "n_iter": int(n_iter),
        "converged": converged
    }
    write_json(f"{out_dir}/report_metrics.json", metrics)
    
    # Confusion Matrices (CSV)
    pd.DataFrame(cm_win, index=[0,1,2], columns=[0,1,2]).to_csv(f"{out_dir}/confusion_window.csv")
    pd.DataFrame(cm_trial, index=[0,1,2], columns=[0,1,2]).to_csv(f"{out_dir}/confusion_trial.csv")
    
    # Pred Counts JSON
    write_json(f"{out_dir}/pred_counts_window.json", dict(pred_counts_win))
    write_json(f"{out_dir}/pred_counts_trial.json", dict(pred_counts_trial))
    
    # Collapse Check JSON
    write_json(f"{out_dir}/collapse_check.json", {
        "window": collapse_win,
        "trial": collapse_trial
    })
    
    # ytrue_ypred_trial.csv
    pd.DataFrame({
        "trial_id": trial_ids,
        "y_true": y_true_trial,
        "y_pred": y_pred_trial
    }).to_csv(f"{out_dir}/ytrue_ypred_trial.csv", index=False)
    
    # Report MD
    with open(f"{out_dir}/SINGLE_RUN_REPORT.md", "w") as f:
        f.write(f"# Phase 14R Step 4c: ProcessedCov Report (Seed {seed})\n\n")
        f.write(f"- **Window Acc**: {win_acc:.4f}\n")
        f.write(f"- **Trial Acc**: {trial_acc:.4f}\n")
        f.write(f"- Converged: {converged} (n_iter={n_iter}/{args.max_iter})\n")
        f.write(f"- Standardize: {args.standardize}\n\n")
        f.write(f"## Prediction Histograms\n")
        f.write(f"- Window: {json_sanitize(dict(pred_counts_win))}\n")
        f.write(f"- Trial: {json_sanitize(dict(pred_counts_trial))}\n\n")
        f.write(f"## Collapse Check\n")
        f.write(f"- Window: collapsed={collapse_win['collapsed']}, max_ratio={collapse_win['max_ratio']:.3f}\n")
        f.write(f"- Trial: collapsed={collapse_trial['collapsed']}, max_ratio={collapse_trial['max_ratio']:.3f}\n\n")
        f.write("## Confusion Matrix (Trial)\n")
        f.write("```\n")
        f.write(f"      Pred 0  Pred 1  Pred 2\n")
        for i, row in enumerate(cm_trial):
            f.write(f"True {i}:  {row[0]:5d}   {row[1]:5d}   {row[2]:5d}\n")
        f.write("```\n")
    
    # Return for aggregation
    metrics["pred_counts_window"] = json_sanitize(dict(pred_counts_win))
    metrics["pred_counts_trial"] = json_sanitize(dict(pred_counts_trial))
    metrics["collapse_window"] = collapse_win["collapsed"]
    metrics["collapse_trial"] = collapse_trial["collapsed"]
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Phase 14R Step 4c: ProcessedCov Model")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 4], help="Seeds to run")
    parser.add_argument("--max_iter", type=int, default=500, help="LogisticRegression max_iter")
    parser.add_argument("--standardize", type=int, default=1, help="Apply StandardScaler (0 or 1)")
    args = parser.parse_args()
    args.standardize = bool(args.standardize)
    
    cfg = DEFAULT_CONFIG.copy()
    
    print(f"Loading Processed Dataset from {cfg['processed_root']}...")
    ds = SeedProcessedTrialDataset(cfg["processed_root"], cfg["stim_xlsx"])
    all_trials = sorted(list(ds), key=lambda x: x["trial_id_str"])
    print(f"Loaded {len(all_trials)} trials.")
    
    results = []
    for seed in args.seeds:
        m = run_single_seed(seed, cfg, all_trials, args)
        results.append(m)
    
    # Summary CSV
    summary_path = f"{cfg['out_root_base']}/summary.csv"
    df = pd.DataFrame(results)
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    print(df[["seed", "win_acc", "trial_acc", "converged", "collapse_window", "collapse_trial"]].to_string(index=False))

if __name__ == "__main__":
    main()
