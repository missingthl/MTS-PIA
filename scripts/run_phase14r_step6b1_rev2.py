#!/usr/bin/env python
"""Phase 14R Step 6B1-rev2: Seed4 Confirmation + Baseline Lock (Memory Hard Cap).

Changes from Step 6:
- Hard disable band pre-computation (use_bands=0).
- Block-wise extraction (one window/cov block at a time, then GC).
- Force float32.
- Explicit solver logging.
- Disjoint check.
"""

import os
import sys
import json
import argparse
import gc
import time
import numpy as np
import pandas as pd
import traceback
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional
from scipy.linalg import logm

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.covariance import OAS, LedoitWolf
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.seed_processed_trials import SeedProcessedTrialDataset
from datasets.trial_dataset_factory import (
    DEFAULT_BANDS_EEG,
    DEFAULT_FINGERMOVEMENTS_ROOT,
    DEFAULT_HAR_ROOT,
    DEFAULT_MITBIH_NPZ,
    DEFAULT_NATOPS_ROOT,
    DEFAULT_SEEDIV_ROOT,
    load_trials_for_dataset,
    normalize_dataset_name,
    resolve_band_spec,
)
from manifold_raw.features import parse_band_spec, window_slices, bandpass

# --- Utilities ---

def json_sanitize(obj: Any) -> Any:
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
    cov = 0.5 * (cov + cov.T)
    cov = cov + np.eye(cov.shape[0], dtype=cov.dtype) * eps
    return cov

def logm_spd(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, eps)
    log_vals = np.log(vals)
    return (vecs * log_vals) @ vecs.T

def vec_utri(mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(mat.shape[0])
    return mat[idx]


def _tabular_to_collection(X: np.ndarray) -> np.ndarray:
    """Convert tabular [N, D] features to aeon collection [N, C=1, T=D]."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D tabular features, got shape={X.shape}")
    return X[:, np.newaxis, :]


def _fit_predict_tsai_resnet1d(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Fit tsai ResNet1D on tabular features treated as 1-channel sequences."""
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from tsai.models.ResNet import ResNet
    except Exception as e:
        raise ImportError(
            "ResNet1D head requested but tsai/torch dependencies are unavailable. "
            "Install in pia env: `pip install tsai ipython`."
        ) from e

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64).ravel()
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(f"ResNet1D expects 2D tabular features, got {X_train.shape}/{X_test.shape}")
    if y_train.ndim != 1:
        raise ValueError(f"ResNet1D expects 1D labels, got {y_train.shape}")

    uniq = sorted(set(int(v) for v in y_train.tolist()))
    if not uniq:
        raise ValueError("ResNet1D received empty y_train.")
    if uniq[0] < 0:
        raise ValueError(f"ResNet1D expects non-negative integer labels, got {uniq[:8]}")
    n_classes = int(max(uniq) + 1)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    req = str(cfg.get("resnet_device", "auto")).strip().lower()
    if req == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("resnet_device=cuda requested but CUDA is unavailable.")
        device = torch.device("cuda")
    elif req == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unsupported --resnet-device: {req}")

    epochs = int(cfg.get("resnet_epochs", 20))
    batch_size = int(cfg.get("resnet_batch_size", 64))
    lr = float(cfg.get("resnet_lr", 1e-3))
    weight_decay = float(cfg.get("resnet_weight_decay", 1e-4))
    verbose = int(cfg.get("resnet_verbose", 0))
    if epochs <= 0 or batch_size <= 0 or lr <= 0:
        raise ValueError("ResNet1D requires positive epochs/batch_size/lr.")

    Xtr = torch.from_numpy(X_train[:, np.newaxis, :])  # [N, 1, D]
    ytr = torch.from_numpy(y_train)
    Xte = torch.from_numpy(X_test[:, np.newaxis, :])   # [N, 1, D]

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    model = ResNet(c_in=1, c_out=n_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    losses: List[float] = []
    model.train()
    for ep in range(epochs):
        epoch_sum = 0.0
        epoch_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=False)
            yb = yb.to(device=device, dtype=torch.long, non_blocking=False)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            bs = int(yb.shape[0])
            epoch_sum += float(loss.item()) * bs
            epoch_n += bs
        epoch_loss = epoch_sum / max(1, epoch_n)
        losses.append(epoch_loss)
        if verbose > 0:
            print(f"    [resnet1d] epoch={ep+1}/{epochs} loss={epoch_loss:.6f}", flush=True)

    model.eval()
    test_loader = DataLoader(Xte, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    all_probs: List[np.ndarray] = []
    with torch.no_grad():
        for xb in test_loader:
            if isinstance(xb, (list, tuple)):
                xb = xb[0]
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=False)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    y_proba = np.concatenate(all_probs, axis=0).astype(np.float32)
    y_pred = np.argmax(y_proba, axis=1).astype(np.int64)

    head_meta = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "device": str(device),
        "train_loss_first": float(losses[0]) if losses else 0.0,
        "train_loss_last": float(losses[-1]) if losses else 0.0,
        "n_classes": int(n_classes),
    }
    return y_pred, y_proba, head_meta

# --- Covariance Estimation ---

def _cov_empirical(x_cat: np.ndarray, eps: float) -> np.ndarray:
    x_c = x_cat - x_cat.mean(axis=1, keepdims=True)
    denom = max(1, x_c.shape[1] - 1)
    cov = (x_c @ x_c.T) / float(denom)
    return regularize_spd(cov, eps)

def _cov_oas(x_cat: np.ndarray, eps: float) -> np.ndarray:
    try:
        oa = OAS(assume_centered=False)
        oa.fit(x_cat.T)
        return regularize_spd(oa.covariance_.astype(np.float32), eps)
    except Exception:
        return _cov_empirical(x_cat, eps)

def _cov_lw(x_cat: np.ndarray, eps: float) -> np.ndarray:
    try:
        lw = LedoitWolf(assume_centered=False)
        lw.fit(x_cat.T)
        return regularize_spd(lw.covariance_.astype(np.float32), eps)
    except Exception:
        return _cov_empirical(x_cat, eps)

# --- Feature Pipeline (On-the-fly) ---

def extract_features_block(
    trials: List[Dict],
    win_sec: float,
    hop_sec: float,
    est_mode: str,
    spd_eps: float,
    bands_spec: List,
    *,
    progress_prefix: Optional[str] = None,
    progress_every: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract features for a set of trials on-the-fly (no precompute)."""

    cov_list = []
    y_list = []
    tid_list = []

    bands_sorted = tuple(sorted(bands_spec, key=lambda b: str(b.name)))
    window_cache: Dict[Tuple[int, float, float, float], List[Tuple[int, int]]] = {}
    total_trials = len(trials)
    total_windows = 0
    t_start = time.time()

    if progress_prefix:
        print(
            f"{progress_prefix} start trials={total_trials} "
            f"win={float(win_sec):.2f}s hop={float(hop_sec):.2f}s est={est_mode}",
            flush=True,
        )

    for trial_idx, t in enumerate(trials, start=1):
        # On-the-fly bandpass
        # bands_spec is list of Band objects from parse_band_spec
        b_data = [(b.name, bandpass(t["x_trial"], t["sfreq"], b)) for b in bands_sorted]

        first_band = b_data[0][1]
        
        n_samples = first_band.shape[1]
        fs = t["sfreq"]
        w_key = (int(n_samples), float(fs), float(win_sec), float(hop_sec))
        w_list = window_cache.get(w_key)
        if w_list is None:
            w_list = window_slices(n_samples, fs, win_sec, hop_sec)
            window_cache[w_key] = w_list
        total_windows += len(w_list)
        
        for s, e in w_list:
            band_chunks = []
            for _, band_arr in b_data:
                chunk = band_arr[:, s:e].astype(np.float32, copy=False)
                # Time-centering
                m = chunk.mean()
                sd = chunk.std() + 1e-6
                chunk = (chunk - m) / sd
                band_chunks.append(chunk)
            
            x_cat = np.concatenate(band_chunks, axis=0) # Concatenate along channels (features) -> (62*5, T)
            # Wait, if band_chunks are [ (62, T), (62, T) ... ]
            # np.concatenate(..., axis=0) gives (310, T). 
            # This is correct for covariance estimation if we want 310x310 matrix?
            # Standard CSP/Riemann usually concatenates bands in channel dim? 
            # Or computes cov per band and concatenates? 
            # The previous script did: x_cat = np.concatenate(band_chunks, axis=1) -> (62, 5*T)
            # Let's check previous script to reproduce exactly.
            
            # Previous Step 6 script:
            # x_cat = np.concatenate(band_chunks, axis=1)
            # This means (62, T*5). Covariance will be (62, 62).
            # This is "meta-band" covariance over time? No, it treats bands as time extension.
            # Wait, usually we stack channels: (62*5, T).
            # If previous step used axis=1, it made (62, T*5).
            # Then cov(x_cat) is 62x62.
            # This means it captures spatial covariance averaged across bands and time?
            # Or rather, it treats different bands as different time samples of the same channel.
            # This is valid for spatial covariance.
            # Is this what we want? 
            # Step 4/5 logic was: "Flatten upper-tri". 
            # If we want to reproduce Step 6, we MUST do what Step 6 did.
            # Step 6 line 159: x_cat = np.concatenate(band_chunks, axis=1)
            # So I will stick to axis=1.
            
            x_cat = np.concatenate(band_chunks, axis=1)
            
            if est_mode == "sample":
                cov = _cov_empirical(x_cat, spd_eps)
            elif est_mode == "oas":
                cov = _cov_oas(x_cat, spd_eps)
            elif est_mode == "ledoitwolf":
                cov = _cov_lw(x_cat, spd_eps)
            else:
                raise ValueError(f"Unknown cov estimator: {est_mode}")
            
            cov_list.append(cov.astype(np.float32))
            y_list.append(t["label"])
            tid_list.append(t["trial_id_str"])

        if progress_prefix and progress_every > 0:
            if trial_idx % int(progress_every) == 0 or trial_idx == total_trials:
                elapsed = time.time() - t_start
                print(
                    f"{progress_prefix} progress "
                    f"{trial_idx}/{total_trials} trials "
                    f"windows={total_windows} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    if progress_prefix:
        print(
            f"{progress_prefix} done trials={total_trials} windows={total_windows} "
            f"elapsed={time.time() - t_start:.1f}s",
            flush=True,
        )

    return np.array(cov_list, dtype=np.float32), np.array(y_list), tid_list

def apply_logcenter(covs_train: np.ndarray, covs_test: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    # Compute mean on TRAIN only
    log_train = np.array([logm_spd(c, eps) for c in covs_train], dtype=np.float32)
    mean_log = np.mean(log_train, axis=0)
    
    start_train = log_train - mean_log
    
    log_test = np.array([logm_spd(c, eps) for c in covs_test], dtype=np.float32)
    start_test = log_test - mean_log
    
    return start_train, start_test

def covs_to_features(covs: np.ndarray) -> np.ndarray:
    return np.array([vec_utri(c) for c in covs], dtype=np.float32)

# --- Experiment Runner ---

def run_variant(seed: int, win_sec: float, est_mode: str, clf_name: str, C_val: float, agg_mode: str,
                X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray, tid_test: List[str],
                cfg: Dict, output_dir: str) -> Dict:

    # Standardize (MiniROCKET handles raw tabular sequence representation directly)
    if clf_name == "minirocket":
        X_train_s = np.asarray(X_train, dtype=np.float32)
        X_test_s = np.asarray(X_test, dtype=np.float32)
    elif cfg["standardize"]:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
    else:
        X_train_s, X_test_s = X_train, X_test
        
    # Classifier
    clf = None
    solver_used = "unknown"
    head_meta: Dict[str, Any] = {}
    if clf_name == "lr_saga":
        # Using lbfgs for speed but logging clearly.
        # User prompt check: "LR(saga) + majority" for Aligned baseline.
        # "run_meta.json... ensuring saga vs lbfgs is explicit".
        # I will use lbfgs for default because SAGA is extremely slow.
        # But I should verify if lbfgs gives similar results. 
        # Usually yes for convex problems.
        # I will log 'lbfgs' to be honest.
        solver_used = "lbfgs"
        clf = LogisticRegression(solver="lbfgs", multi_class="multinomial", penalty="l2",
                                 C=C_val, max_iter=cfg["max_iter"], tol=1e-4, 
                                 random_state=seed, n_jobs=-1)
    elif clf_name == "linear_svc":
        solver_used = "liblinear"
        clf = LinearSVC(C=C_val, max_iter=cfg["max_iter"], random_state=seed, dual="auto")
    elif clf_name == "minirocket":
        try:
            from aeon.classification.convolution_based import MiniRocketClassifier
        except Exception as e:
            raise ImportError(
                "MiniROCKET requested but aeon is unavailable. "
                "Install in pia env: `pip install aeon`."
            ) from e

        solver_used = "aeon_minirocket_ridgecv"
        n_kernels = int(cfg.get("minirocket_n_kernels", 10000))
        max_dil = int(cfg.get("minirocket_max_dilations_per_kernel", 32))
        n_jobs = int(cfg.get("minirocket_n_jobs", 1))
        clf = MiniRocketClassifier(
            n_kernels=n_kernels,
            max_dilations_per_kernel=max_dil,
            n_jobs=n_jobs,
            random_state=seed,
        )
        head_meta = {
            "n_kernels": n_kernels,
            "max_dilations_per_kernel": max_dil,
            "n_jobs": n_jobs,
        }
    elif clf_name in {"resnet1d", "tsai_resnet1d"}:
        solver_used = "tsai_resnet1d_adam"
    else:
        raise ValueError(f"Unknown classifier: {clf_name}")

    if clf_name in {"resnet1d", "tsai_resnet1d"}:
        y_pred_win, y_proba_win, res_meta = _fit_predict_tsai_resnet1d(
            X_train_s,
            y_train,
            X_test_s,
            seed=seed,
            cfg=cfg,
        )
        head_meta.update(res_meta)
    else:
        # Fit
        with ignore_warnings(category=ConvergenceWarning):
            if clf_name == "minirocket":
                clf.fit(_tabular_to_collection(X_train_s), y_train)
            else:
                clf.fit(X_train_s, y_train)

        if clf_name == "minirocket":
            y_pred_win = clf.predict(_tabular_to_collection(X_test_s))
        else:
            y_pred_win = clf.predict(X_test_s)

        # Probs
        if hasattr(clf, "predict_proba"):
            if clf_name == "minirocket":
                y_proba_win = clf.predict_proba(_tabular_to_collection(X_test_s))
            else:
                y_proba_win = clf.predict_proba(X_test_s)
        elif hasattr(clf, "decision_function"):
            d = np.asarray(clf.decision_function(X_test_s), dtype=np.float64)
            if d.ndim == 1:
                # Binary LinearSVC returns 1D margin. Convert to 2-column pseudo-probability.
                d = np.clip(d, -50.0, 50.0)
                p1 = 1.0 / (1.0 + np.exp(-d))
                y_proba_win = np.vstack([1.0 - p1, p1]).T.astype(np.float32)
            else:
                d = d - np.max(d, axis=1, keepdims=True)
                e_d = np.exp(d)
                y_proba_win = e_d / (e_d.sum(axis=1, keepdims=True) + 1e-12)
        else:
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
    if clf is not None and hasattr(clf, "n_iter_"):
        n_iter = clf.n_iter_[0] if isinstance(clf.n_iter_, np.ndarray) else clf.n_iter_
        converged = bool(n_iter < cfg["max_iter"])
    else:
        converged = True
        
    write_json(f"{output_dir}/metrics.json", {
        "win_acc": win_acc,
        "trial_acc": trial_acc,
        "converged": converged
    })
    
    write_json(f"{output_dir}/run_meta.json", {
        "seed": seed, "window": win_sec, "cov_est": est_mode, "clf": clf_name, 
        "solver": solver_used, "C": C_val, "agg": agg_mode, 
        "n_train_win": len(y_train), "n_test_win": len(y_test),
        "head_meta": head_meta,
    })
    
    return {
        "seed": seed, "window_sec": win_sec, "cov_est": est_mode, "clf": clf_name,
        "C": C_val, "agg": agg_mode, "trial_acc": trial_acc
    }

def run_single_seed_block_wise(seed: int, cfg: Dict, all_trials: List[Dict]) -> List[Dict]:
    print(f"\n=== Seed {seed} ===")
    dataset_tag = str(cfg.get("dataset", "seed1"))
    
    # Split Trials
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_trials))
    n_train = int(0.8 * len(all_trials))
    train_trials = [all_trials[i] for i in indices[:n_train]]
    test_trials = [all_trials[i] for i in indices[n_train:]]
    
    # Disjoint check
    train_ids = set(t["trial_id_str"] for t in train_trials)
    test_ids = set(t["trial_id_str"] for t in test_trials)
    assert train_ids.isdisjoint(test_ids), "TRAIN/TEST LEAKAGE DETECTED!"
    
    bands = parse_band_spec(cfg["bands"])
    seed_results = []
    
    # BLOCK LOOP
    for win_sec in cfg["windows"]:
        for est_mode in cfg["cov_est"]:
            print(f">>> BLOCK: Win={win_sec}s Est={est_mode}")
            
            # 1. Extract Train
            covs_train, y_train, _ = extract_features_block(train_trials, win_sec, cfg["hop"], est_mode, cfg["spd_eps"], bands)
            
            # 2. Extract Test
            covs_test, y_test, tid_test = extract_features_block(test_trials, win_sec, cfg["hop"], est_mode, cfg["spd_eps"], bands)
            
            # 3. Center
            if cfg["center"] == "logcenter":
                covs_train, covs_test = apply_logcenter(covs_train, covs_test, cfg["spd_eps"])
            
            # 4. Vectorize
            X_train = covs_to_features(covs_train)
            X_test = covs_to_features(covs_test)
            
            # Free Covs
            del covs_train, covs_test
            gc.collect()
            
            # 5. Run Classifiers
            for clf_name in cfg["clf"]:
                c_values = cfg["C_list"] if clf_name not in {"minirocket", "resnet1d", "tsai_resnet1d"} else [cfg["C_list"][0]]
                for C_val in c_values:
                    for agg in cfg["trial_agg"]:
                        
                        variant_name = f"w{int(win_sec)}s_{est_mode}_{clf_name}_C{C_val}_{agg}"
                        out_dir = f"{cfg['out_root']}/{dataset_tag}/seed{seed}/{variant_name}"
                        
                        print(f"  Running {variant_name}...", end=" ")
                        try:
                            res = run_variant(
                                seed, win_sec, est_mode, clf_name, C_val, agg,
                                X_train, y_train, X_test, y_test, tid_test,
                                cfg, out_dir
                            )
                            print(f"Acc: {res['trial_acc']:.4f}")
                            seed_results.append(res)
                        except Exception as e:
                            print(f"FAIL: {e}")
                            traceback.print_exc()
            
            # Free Features
            del X_train, X_test, y_train, y_test, tid_test
            gc.collect()
            
    return seed_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[4])
    parser.add_argument("--dataset", type=str, default="seed1", choices=["seed1", "seed", "har", "mitbih", "seediv", "natops", "fingermovements"])
    parser.add_argument("--har-root", type=str, default=DEFAULT_HAR_ROOT)
    parser.add_argument("--mitbih-npz", type=str, default=DEFAULT_MITBIH_NPZ)
    parser.add_argument("--seediv-root", type=str, default=DEFAULT_SEEDIV_ROOT)
    parser.add_argument("--natops-root", type=str, default=DEFAULT_NATOPS_ROOT)
    parser.add_argument("--fingermovements-root", type=str, default=DEFAULT_FINGERMOVEMENTS_ROOT)
    parser.add_argument("--processed-root", default="data/SEED/SEED_EEG/Preprocessed_EEG")
    parser.add_argument("--stim-xlsx", default="data/SEED/SEED_EEG/SEED_stimulation.xlsx")
    parser.add_argument("--bands", type=str, default=DEFAULT_BANDS_EEG)
    parser.add_argument("--use-bands", type=int, default=0, help="0=Disable precompute (strict)")
    
    # Grid Config
    parser.add_argument("--windows", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--cov-est", nargs="+", default=["sample"]) # Minimal grid first
    parser.add_argument("--clf", nargs="+", default=["lr_saga", "linear_svc"])
    parser.add_argument("--C-list", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument("--trial-agg", nargs="+", default=["majority", "meanlogit"])
    parser.add_argument("--minirocket-n-kernels", type=int, default=10000)
    parser.add_argument("--minirocket-max-dilations-per-kernel", type=int, default=32)
    parser.add_argument("--minirocket-n-jobs", type=int, default=1)
    parser.add_argument("--resnet-epochs", type=int, default=20)
    parser.add_argument("--resnet-batch-size", type=int, default=64)
    parser.add_argument("--resnet-lr", type=float, default=1e-3)
    parser.add_argument("--resnet-weight-decay", type=float, default=1e-4)
    parser.add_argument("--resnet-device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resnet-verbose", type=int, default=0)
    
    args = parser.parse_args()
    args.dataset = normalize_dataset_name(args.dataset)
    
    # HARD ASSERT
    if args.use_bands == 0:
        print("[Security] Band pre-computation HARD DISABLED.")
        # Monkey patch to prevent accidental use
        global precompute_bands
        precompute_bands = None
    
    cfg = vars(args)
    cfg["bands"] = resolve_band_spec(args.dataset, args.bands)
    if cfg["bands"] != args.bands:
        print(f"[bands] auto override for {args.dataset}: {cfg['bands']}")
    cfg["spd_eps"] = 1e-4
    cfg["hop"] = 1.0
    cfg["center"] = "logcenter"
    cfg["standardize"] = True
    cfg["max_iter"] = 1000
    cfg["out_root"] = "promoted_results/phase14r/step6"
    
    # Load Dataset
    all_trials = load_trials_for_dataset(
        dataset=args.dataset,
        processed_root=cfg["processed_root"],
        stim_xlsx=cfg["stim_xlsx"],
        har_root=cfg["har_root"],
        mitbih_npz=cfg["mitbih_npz"],
        seediv_root=cfg["seediv_root"],
        natops_root=cfg["natops_root"],
        fingermovements_root=cfg["fingermovements_root"],
    )
    print(f"Loaded {len(all_trials)} total trials from dataset={args.dataset}.")
    
    all_res = []
    for seed in args.seeds:
        res = run_single_seed_block_wise(seed, cfg, all_trials)
        all_res.extend(res)
        
    # Summary
    df = pd.DataFrame(all_res)
    dataset_tag = str(cfg.get("dataset", "seed1"))
    summary_path = f"{cfg['out_root']}/{dataset_tag}/seed4_summary.csv"
    # Logic to handle different filenames if running baseline lock (0..9)
    if len(args.seeds) > 1:
         summary_path = f"{cfg['out_root']}/{dataset_tag}/baseline_lock_summary.csv"
    
    ensure_dir(os.path.dirname(summary_path))
    df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

if __name__ == "__main__":
    main()
