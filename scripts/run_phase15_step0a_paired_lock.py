#!/usr/bin/env python
"""Phase 15 Step 0A-rev1: Paired Wrapper + Metrics/Meta Lock (LinearSVC).

Non-invasive wrapper over Mainline-B feature pipeline:
- Reuses extract_features_block / apply_logcenter / covs_to_features
- Runs paired A/B/C under the exact same split:
  A: baseline
  B: PIA (train-only)
  C: PIA + manifold gate (train-only)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIA.augment import PIADirectionalAffineAugmenter
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
from manifold_raw.features import parse_band_spec
from run_phase14r_step6b1_rev2 import (
    apply_logcenter,
    covs_to_features,
    ensure_dir,
    extract_features_block,
    json_sanitize,
    write_json,
)


def _ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        s = str(v)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _stable_tid_hash(tid: str) -> int:
    h = hashlib.sha256(str(tid).encode("utf-8")).hexdigest()[:16]
    return int(h, 16) & 0x7FFFFFFF


def _split_hash(train_ids: List[str], test_ids: List[str]) -> str:
    payload = json.dumps(
        {"train": [str(x) for x in train_ids], "test": [str(x) for x in test_ids]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_trial_split(all_trials: List[Dict], seed: int) -> Tuple[List[Dict], List[Dict], Dict[str, object]]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_trials))
    n_train = int(0.8 * len(all_trials))
    train_trials = [all_trials[i] for i in idx[:n_train]]
    test_trials = [all_trials[i] for i in idx[n_train:]]

    train_ids = [str(t["trial_id_str"]) for t in train_trials]
    test_ids = [str(t["trial_id_str"]) for t in test_trials]
    if not set(train_ids).isdisjoint(set(test_ids)):
        raise RuntimeError("Split leakage: train/test trial_id overlap.")

    split_meta = {
        "split_hash": _split_hash(train_ids, test_ids),
        "train_count_trials": len(train_ids),
        "test_count_trials": len(test_ids),
        "train_trial_ids": train_ids,
        "test_trial_ids": test_ids,
    }
    return train_trials, test_trials, split_meta


def _per_trial_window_counts(tids: np.ndarray) -> Dict[str, int]:
    c: Dict[str, int] = defaultdict(int)
    for t in tids:
        c[str(t)] += 1
    return dict(c)


def _count_stats(counts: Dict[str, int]) -> Dict[str, float]:
    if not counts:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    vals = np.asarray(list(counts.values()), dtype=np.float64)
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _apply_window_cap(
    X: np.ndarray,
    y: np.ndarray,
    tid: np.ndarray,
    cap_k: int,
    seed: int,
    *,
    is_aug: Optional[np.ndarray] = None,
    policy: str = "random",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], float]:
    is_aug_arr = (
        np.zeros((len(y),), dtype=bool)
        if is_aug is None
        else np.asarray(is_aug, dtype=bool).ravel()
    )
    if is_aug_arr.shape[0] != len(y):
        raise ValueError("is_aug must have same length as y.")

    valid_policies = {"random", "balanced_real_aug", "prefer_real", "prefer_aug"}
    if policy not in valid_policies:
        raise ValueError(f"Unknown cap sampling policy: {policy}")

    def _empty_aug_ratio(arr: np.ndarray) -> float:
        return float(np.mean(arr.astype(np.float64))) if arr.size else 0.0

    if cap_k <= 0:
        counts = _per_trial_window_counts(tid)
        return X, y, tid, is_aug_arr, counts, _empty_aug_ratio(is_aug_arr)

    keep_idx: List[int] = []
    tid_arr = np.asarray(tid)
    for trial_id in sorted(_ordered_unique(tid_arr.tolist())):
        idx = np.where(tid_arr == trial_id)[0]
        if idx.size <= cap_k:
            sel = idx
        else:
            rs = np.random.RandomState(seed ^ _stable_tid_hash(trial_id))
            aug_flags = is_aug_arr[idx]
            real_idx = idx[~aug_flags]
            aug_idx = idx[aug_flags]

            if policy == "random":
                sel = rs.choice(idx, size=cap_k, replace=False)
            elif policy == "balanced_real_aug":
                k_real_target = cap_k // 2
                k_aug_target = cap_k - k_real_target
                k_real = min(k_real_target, real_idx.size)
                k_aug = min(k_aug_target, aug_idx.size)

                parts: List[np.ndarray] = []
                if k_real > 0:
                    parts.append(rs.choice(real_idx, size=k_real, replace=False))
                if k_aug > 0:
                    parts.append(rs.choice(aug_idx, size=k_aug, replace=False))
                sel_seed = np.concatenate(parts) if parts else np.empty((0,), dtype=np.int64)

                rem = cap_k - sel_seed.size
                if rem > 0:
                    # Fill from whichever pool still has unused samples.
                    used = set(sel_seed.tolist())
                    rem_pool = np.asarray([i for i in idx.tolist() if i not in used], dtype=np.int64)
                    fill = rs.choice(rem_pool, size=rem, replace=False)
                    sel = np.concatenate([sel_seed, fill])
                else:
                    sel = sel_seed
            elif policy == "prefer_real":
                k_real = min(cap_k, real_idx.size)
                parts = []
                if k_real > 0:
                    parts.append(rs.choice(real_idx, size=k_real, replace=False))
                rem = cap_k - k_real
                if rem > 0:
                    parts.append(rs.choice(aug_idx, size=rem, replace=False))
                sel = np.concatenate(parts) if parts else np.empty((0,), dtype=np.int64)
            elif policy == "prefer_aug":
                k_aug = min(cap_k, aug_idx.size)
                parts = []
                if k_aug > 0:
                    parts.append(rs.choice(aug_idx, size=k_aug, replace=False))
                rem = cap_k - k_aug
                if rem > 0:
                    parts.append(rs.choice(real_idx, size=rem, replace=False))
                sel = np.concatenate(parts) if parts else np.empty((0,), dtype=np.int64)
            else:
                raise RuntimeError(f"Unhandled cap sampling policy: {policy}")
            sel = np.sort(sel)
        keep_idx.extend(sel.tolist())

    keep = np.asarray(keep_idx, dtype=np.int64)
    counts = _per_trial_window_counts(tid_arr[keep])
    is_aug_keep = is_aug_arr[keep]
    aug_ratio = _empty_aug_ratio(is_aug_keep)
    return X[keep], y[keep], tid_arr[keep], is_aug_keep, counts, aug_ratio


@dataclass
class PiaAugConfig:
    multiplier: int
    gamma: float
    gamma_jitter: float
    n_iters: int
    activation: str
    bias_update_mode: str
    C_repr: float
    seed: int


def _build_pia_aug_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tid_train: np.ndarray,
    cfg: PiaAugConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    tid_arr = np.asarray(tid_train)
    y_arr = np.asarray(y_train).astype(int).ravel()
    trial_ids = sorted(_ordered_unique(tid_arr.tolist()))

    aug_X_parts: List[np.ndarray] = []
    aug_y_parts: List[np.ndarray] = []
    aug_tid_parts: List[np.ndarray] = []
    gamma_list: List[float] = []
    recon_last: List[float] = []
    aug_count_per_trial: Dict[str, int] = {}

    for tid in trial_ids:
        idx = np.where(tid_arr == tid)[0]
        X_tid = X_train[idx]
        y_tid = y_arr[idx]
        added = 0
        for m in range(max(0, int(cfg.multiplier))):
            rs = np.random.RandomState(cfg.seed + m * 1009 + _stable_tid_hash(tid))
            if cfg.gamma_jitter > 0:
                g_lo = cfg.gamma * (1.0 - cfg.gamma_jitter)
                g_hi = cfg.gamma * (1.0 + cfg.gamma_jitter)
                gamma = float(rs.uniform(g_lo, g_hi))
            else:
                gamma = float(cfg.gamma)

            aug = PIADirectionalAffineAugmenter(
                gamma=gamma,
                n_iters=int(cfg.n_iters),
                activation=cfg.activation,
                bias_update_mode=cfg.bias_update_mode,
                C_repr=float(cfg.C_repr),
            )
            X_aug = aug.fit_transform(X_tid)
            st = aug.state()
            recon = st.get("recon_err")
            if isinstance(recon, list) and recon:
                recon_last.append(float(recon[-1]))

            aug_X_parts.append(np.asarray(X_aug, dtype=np.float32))
            aug_y_parts.append(np.asarray(y_tid, dtype=np.int64))
            aug_tid_parts.append(np.asarray([tid] * len(idx)))
            gamma_list.extend([gamma] * len(idx))
            added += len(idx)
        aug_count_per_trial[tid] = int(added)

    if aug_X_parts:
        X_aug_all = np.vstack(aug_X_parts).astype(np.float32)
        y_aug_all = np.concatenate(aug_y_parts).astype(np.int64)
        tid_aug_all = np.concatenate(aug_tid_parts)
    else:
        X_aug_all = np.empty((0, X_train.shape[1]), dtype=np.float32)
        y_aug_all = np.empty((0,), dtype=np.int64)
        tid_aug_all = np.empty((0,), dtype=object)

    g = np.asarray(gamma_list, dtype=np.float64) if gamma_list else np.asarray([], dtype=np.float64)
    aug_vals = np.asarray(list(aug_count_per_trial.values()), dtype=np.float64) if aug_count_per_trial else np.asarray([], dtype=np.float64)
    meta = {
        "aug_total_count": int(len(y_aug_all)),
        "aug_count_per_trial": aug_count_per_trial,
        "aug_per_trial_mean": float(np.mean(aug_vals)) if aug_vals.size else 0.0,
        "aug_per_trial_std": float(np.std(aug_vals)) if aug_vals.size else 0.0,
        "gamma_min": float(np.min(g)) if g.size else 0.0,
        "gamma_mean": float(np.mean(g)) if g.size else 0.0,
        "gamma_std": float(np.std(g)) if g.size else 0.0,
        "gamma_max": float(np.max(g)) if g.size else 0.0,
        "pia_lambda_or_C_repr": float(cfg.C_repr),
        "pia_n_iters": int(cfg.n_iters),
        "pia_activation": cfg.activation,
        "pia_bias_update_mode": cfg.bias_update_mode,
        "recon_last_mean": float(np.mean(recon_last)) if recon_last else 0.0,
        "recon_last_std": float(np.std(recon_last)) if recon_last else 0.0,
    }
    return X_aug_all, y_aug_all, tid_aug_all, meta


def _fit_gate_from_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    q: float,
) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[str, object]]:
    y = np.asarray(y_train).astype(int).ravel()
    classes = sorted(np.unique(y).tolist())

    mu: Dict[int, np.ndarray] = {}
    tau: Dict[int, float] = {}
    dist_pool: Dict[int, np.ndarray] = {}
    for c in classes:
        Xc = X_train[y == c]
        if Xc.shape[0] == 0:
            continue
        muc = np.mean(Xc, axis=0)
        d = np.linalg.norm(Xc - muc[None, :], axis=1)
        mu[int(c)] = muc
        tau[int(c)] = float(np.percentile(d, q))
        dist_pool[int(c)] = d

    meta = {
        "gate_metric": "tangent_l2_to_class_center",
        "gate_percentile_q": float(q),
        "tau_y": {str(k): float(v) for k, v in tau.items()},
        "train_dist_summary_by_class": {
            str(k): {
                "min": float(np.min(v)),
                "median": float(np.median(v)),
                "p95": float(np.percentile(v, 95)),
                "max": float(np.max(v)),
            }
            for k, v in dist_pool.items()
        },
    }
    return mu, tau, meta


def _apply_gate(
    X_aug: np.ndarray,
    y_aug: np.ndarray,
    tid_aug: np.ndarray,
    mu: Dict[int, np.ndarray],
    tau: Dict[int, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    if X_aug.shape[0] == 0:
        return X_aug, y_aug, tid_aug, {
            "accept_rate": 0.0,
            "accepted_count": 0,
            "rejected_count": 0,
            "accepted_dist_summary": {"min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0},
            "rejected_dist_summary": {"min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0},
        }

    y = np.asarray(y_aug).astype(int).ravel()
    dists = np.zeros((len(y),), dtype=np.float64)
    keep = np.zeros((len(y),), dtype=bool)

    for i in range(len(y)):
        cls = int(y[i])
        muc = mu.get(cls)
        tauc = tau.get(cls)
        if muc is None or tauc is None:
            dists[i] = np.inf
            keep[i] = False
            continue
        di = float(np.linalg.norm(X_aug[i] - muc))
        dists[i] = di
        keep[i] = di <= tauc

    accepted = dists[keep]
    rejected = dists[~keep]

    def _summ(v: np.ndarray) -> Dict[str, float]:
        if v.size == 0:
            return {"min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
        return {
            "min": float(np.min(v)),
            "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)),
            "max": float(np.max(v)),
        }

    meta = {
        "accept_rate": float(np.mean(keep)),
        "accepted_count": int(np.sum(keep)),
        "rejected_count": int(np.sum(~keep)),
        "accepted_dist_summary": _summ(accepted),
        "rejected_dist_summary": _summ(rejected),
    }
    return X_aug[keep], y_aug[keep], tid_aug[keep], meta


def _aggregate_trials(
    y_true_win: np.ndarray,
    y_pred_win: np.ndarray,
    scores_win: np.ndarray,
    tid_win: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true_win).astype(int).ravel()
    y_pred = np.asarray(y_pred_win).astype(int).ravel()
    scores = np.asarray(scores_win, dtype=np.float64)
    tids = np.asarray(tid_win)

    out_true: List[int] = []
    out_pred: List[int] = []

    for tid in sorted(_ordered_unique(tids.tolist())):
        idx = np.where(tids == tid)[0]
        yy = y_true[idx]
        pp = y_pred[idx]
        ss = scores[idx]
        out_true.append(int(yy[0]))
        if mode == "majority":
            out_pred.append(int(Counter(pp.tolist()).most_common(1)[0][0]))
        elif mode == "meanlogit":
            out_pred.append(int(np.argmax(np.mean(ss, axis=0))))
        else:
            raise ValueError(f"Unknown aggregation mode: {mode}")
    return np.asarray(out_true, dtype=int), np.asarray(out_pred, dtype=int)


def _fit_eval_linearsvc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tid_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tid_test: np.ndarray,
    *,
    seed: int,
    cap_k: int,
    cap_seed: int,
    cap_sampling_policy: str,
    linear_c: float,
    class_weight: Optional[str],
    max_iter: int,
    agg_mode: str,
    is_aug_train: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    X_cap, y_cap, tid_cap, is_aug_cap, cap_counts, selected_aug_ratio = _apply_window_cap(
        np.asarray(X_train, dtype=np.float32),
        np.asarray(y_train).astype(int).ravel(),
        np.asarray(tid_train),
        cap_k=int(cap_k),
        seed=int(cap_seed),
        is_aug=is_aug_train,
        policy=cap_sampling_policy,
    )

    scaler = StandardScaler()
    X_cap_s = scaler.fit_transform(X_cap)
    X_te_s = scaler.transform(np.asarray(X_test, dtype=np.float32))

    cw = None if class_weight in {None, "", "none"} else class_weight
    clf = LinearSVC(
        C=float(linear_c),
        class_weight=cw,
        max_iter=int(max_iter),
        random_state=int(seed),
        dual="auto",
    )
    clf.fit(X_cap_s, y_cap)

    y_pred_win = clf.predict(X_te_s)
    scores_win = clf.decision_function(X_te_s)
    if scores_win.ndim == 1:
        scores_win = np.vstack([-scores_win, scores_win]).T

    y_true_trial, y_pred_trial = _aggregate_trials(
        y_true_win=np.asarray(y_test).astype(int).ravel(),
        y_pred_win=y_pred_win,
        scores_win=scores_win,
        tid_win=np.asarray(tid_test),
        mode=agg_mode,
    )

    trial_acc = float(accuracy_score(y_true_trial, y_pred_trial))
    trial_macro_f1 = float(f1_score(y_true_trial, y_pred_trial, average="macro"))
    metrics = {
        "trial_acc": trial_acc,
        "trial_macro_f1": trial_macro_f1,
        "aggregation_mode": agg_mode,
        "window_acc": float(accuracy_score(np.asarray(y_test).astype(int).ravel(), y_pred_win)),
        "window_macro_f1": float(f1_score(np.asarray(y_test).astype(int).ravel(), y_pred_win, average="macro")),
        "trial_confusion_matrix": confusion_matrix(y_true_trial, y_pred_trial).tolist(),
    }

    cap_stats = _count_stats(cap_counts)
    meta = {
        "total_train_windows_used": int(len(y_cap)),
        "per_trial_windows_after_cap": cap_counts,
        "per_trial_windows_mean_after_cap": cap_stats["mean"],
        "per_trial_windows_std_after_cap": cap_stats["std"],
        "per_trial_windows_min_after_cap": cap_stats["min"],
        "per_trial_windows_max_after_cap": cap_stats["max"],
        "cap_sampling_policy": cap_sampling_policy,
        "train_selected_aug_ratio": float(selected_aug_ratio),
        "train_selected_aug_count": int(np.sum(is_aug_cap)),
        "train_selected_real_count": int(len(is_aug_cap) - np.sum(is_aug_cap)),
        "feature_dim": int(X_cap.shape[1]),
        "clf_params": {
            "model": "LinearSVC",
            "C": float(linear_c),
            "class_weight": cw,
            "max_iter": int(max_iter),
            "dual": "auto",
            "random_state": int(seed),
        },
    }
    return metrics, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--dataset", type=str, default="seed1", choices=["seed1", "seed", "har", "mitbih", "seediv", "natops", "fingermovements"])
    parser.add_argument("--har-root", type=str, default=DEFAULT_HAR_ROOT)
    parser.add_argument("--mitbih-npz", type=str, default=DEFAULT_MITBIH_NPZ)
    parser.add_argument("--seediv-root", type=str, default=DEFAULT_SEEDIV_ROOT)
    parser.add_argument("--natops-root", type=str, default=DEFAULT_NATOPS_ROOT)
    parser.add_argument("--fingermovements-root", type=str, default=DEFAULT_FINGERMOVEMENTS_ROOT)
    parser.add_argument("--processed-root", type=str, default="data/SEED/SEED_EEG/Preprocessed_EEG")
    parser.add_argument("--stim-xlsx", type=str, default="data/SEED/SEED_EEG/SEED_stimulation.xlsx")
    parser.add_argument("--out-root", type=str, default="out/phase15_step0a")

    # Mainline-B aligned feature config
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--hop-sec", type=float, default=1.0)
    parser.add_argument("--cov-est", type=str, default="sample", choices=["sample", "oas", "ledoitwolf"])
    parser.add_argument("--spd-eps", type=float, default=1e-4)
    parser.add_argument(
        "--bands",
        type=str,
        default=DEFAULT_BANDS_EEG,
    )
    parser.add_argument("--aggregation-mode", type=str, default="majority", choices=["majority", "meanlogit"])

    # Fixed classifier config
    parser.add_argument("--linear-c", type=float, default=1.0)
    parser.add_argument("--linear-class-weight", type=str, default="none")
    parser.add_argument("--linear-max-iter", type=int, default=1000)

    # Mandatory sample-weight control
    parser.add_argument("--window-cap-k", type=int, default=0, help="<=0 means auto from percentile.")
    parser.add_argument("--window-cap-percentile", type=float, default=75.0)
    parser.add_argument(
        "--cap-sampling-policy",
        type=str,
        default="random",
        choices=["random", "balanced_real_aug", "prefer_real", "prefer_aug"],
    )

    # PIA config (B/C)
    parser.add_argument("--pia-multiplier", type=int, default=1)
    parser.add_argument("--pia-gamma", type=float, default=0.2)
    parser.add_argument("--pia-gamma-jitter", type=float, default=0.0)
    parser.add_argument("--pia-n-iters", type=int, default=3)
    parser.add_argument("--pia-activation", type=str, default="sine")
    parser.add_argument("--pia-bias-update-mode", type=str, default="residual")
    parser.add_argument("--pia-c-repr", type=float, default=1.0)

    # Gate config (C)
    parser.add_argument("--gate-percentile", type=float, default=95.0)

    parser.add_argument("--split-preview-n", type=int, default=5)
    args = parser.parse_args()
    args.dataset = normalize_dataset_name(args.dataset)

    if args.pia_multiplier < 0:
        raise ValueError("--pia-multiplier must be >= 0")
    if args.window_cap_percentile <= 0 or args.window_cap_percentile > 100:
        raise ValueError("--window-cap-percentile must be in (0,100].")

    # 1) Load + split once (paired lock)
    all_trials = load_trials_for_dataset(
        dataset=args.dataset,
        processed_root=args.processed_root,
        stim_xlsx=args.stim_xlsx,
        har_root=args.har_root,
        mitbih_npz=args.mitbih_npz,
        seediv_root=args.seediv_root,
        natops_root=args.natops_root,
        fingermovements_root=args.fingermovements_root,
    )
    train_trials, test_trials, split_meta = _make_trial_split(all_trials, seed=int(args.seed))
    print(
        f"[split] dataset={args.dataset} trials total={len(all_trials)} "
        f"train={len(train_trials)} test={len(test_trials)}"
    )
    print(f"[split] split_hash={split_meta['split_hash']}")

    # 2) Feature extraction once (A/B/C share split, test untouched)
    bands_spec = resolve_band_spec(args.dataset, args.bands)
    if bands_spec != args.bands:
        print(f"[bands] auto override for {args.dataset}: {bands_spec}")
    bands = parse_band_spec(bands_spec)
    covs_train, y_train, tid_train = extract_features_block(
        train_trials, args.window_sec, args.hop_sec, args.cov_est, args.spd_eps, bands
    )
    covs_test, y_test, tid_test = extract_features_block(
        test_trials, args.window_sec, args.hop_sec, args.cov_est, args.spd_eps, bands
    )
    covs_train_lc, covs_test_lc = apply_logcenter(covs_train, covs_test, args.spd_eps)
    X_train_base = covs_to_features(covs_train_lc).astype(np.float32)
    X_test = covs_to_features(covs_test_lc).astype(np.float32)
    y_train_base = np.asarray(y_train).astype(int).ravel()
    y_test = np.asarray(y_test).astype(int).ravel()
    tid_train = np.asarray(tid_train)
    tid_test = np.asarray(tid_test)
    print(f"[feat] train_windows={len(y_train_base)} test_windows={len(y_test)} feat_dim={X_train_base.shape[1]}")

    # Fixed K from original TRAIN window counts (paired constant across A/B/C)
    base_counts = _per_trial_window_counts(tid_train)
    if args.window_cap_k > 0:
        cap_k = int(args.window_cap_k)
    else:
        cap_k = max(1, int(np.percentile(np.asarray(list(base_counts.values()), dtype=np.float64), args.window_cap_percentile)))
    print(f"[cap] K={cap_k} (percentile={args.window_cap_percentile if args.window_cap_k <= 0 else 'manual'})")

    seed_dir = os.path.join(args.out_root, f"seed{args.seed}")
    cond_dirs = {
        "A_baseline": os.path.join(seed_dir, "A_baseline"),
        "B_pia": os.path.join(seed_dir, "B_pia"),
        "C_pia_gate": os.path.join(seed_dir, "C_pia_gate"),
    }
    for d in cond_dirs.values():
        ensure_dir(d)

    common_meta = {
        "seed": int(args.seed),
        "split_hash": split_meta["split_hash"],
        "train_count_trials": int(split_meta["train_count_trials"]),
        "test_count_trials": int(split_meta["test_count_trials"]),
        "train_trial_ids_preview": split_meta["train_trial_ids"][: max(0, int(args.split_preview_n))],
        "test_trial_ids_preview": split_meta["test_trial_ids"][: max(0, int(args.split_preview_n))],
        "window_cap_K": int(cap_k),
        "cap_sampling_policy": args.cap_sampling_policy,
        "feature_pipeline": {
            "window_sec": float(args.window_sec),
            "hop_sec": float(args.hop_sec),
            "cov_est": args.cov_est,
            "spd_eps": float(args.spd_eps),
            "center": "logcenter_train_only",
            "vectorize": "upper_triangle",
            "bands": bands_spec,
        },
        "dataset": args.dataset,
        "aggregation_mode": args.aggregation_mode,
        "test_augmentation": "disabled",
    }

    summary_rows: List[Dict[str, object]] = []

    # A) Baseline
    metrics_a, train_meta_a = _fit_eval_linearsvc(
        X_train_base,
        y_train_base,
        tid_train,
        X_test,
        y_test,
        tid_test,
        seed=int(args.seed),
        cap_k=cap_k,
        cap_seed=int(args.seed) + 11,
        cap_sampling_policy=args.cap_sampling_policy,
        linear_c=float(args.linear_c),
        class_weight=args.linear_class_weight,
        max_iter=int(args.linear_max_iter),
        agg_mode=args.aggregation_mode,
        is_aug_train=np.zeros((len(y_train_base),), dtype=bool),
    )
    write_json(os.path.join(cond_dirs["A_baseline"], "metrics.json"), metrics_a)
    write_json(
        os.path.join(cond_dirs["A_baseline"], "run_meta.json"),
        {
            **common_meta,
            **train_meta_a,
            "condition": "A_baseline",
        },
    )
    summary_rows.append(
        {
            "condition": "A_baseline",
            "acc": metrics_a["trial_acc"],
            "macro_f1": metrics_a["trial_macro_f1"],
            "train_windows": train_meta_a["total_train_windows_used"],
            "aug_total": 0,
            "accept_rate": 1.0,
            "cap_sampling_policy": args.cap_sampling_policy,
            "train_selected_aug_ratio": train_meta_a["train_selected_aug_ratio"],
            "split_hash": common_meta["split_hash"],
        }
    )
    print(f"[A] acc={metrics_a['trial_acc']:.4f} macro_f1={metrics_a['trial_macro_f1']:.4f}")

    # Build shared PIA candidates for B/C from original TRAIN only
    pia_cfg = PiaAugConfig(
        multiplier=int(args.pia_multiplier),
        gamma=float(args.pia_gamma),
        gamma_jitter=float(args.pia_gamma_jitter),
        n_iters=int(args.pia_n_iters),
        activation=args.pia_activation,
        bias_update_mode=args.pia_bias_update_mode,
        C_repr=float(args.pia_c_repr),
        seed=int(args.seed),
    )
    X_aug, y_aug, tid_aug, aug_meta = _build_pia_aug_candidates(
        X_train=X_train_base, y_train=y_train_base, tid_train=tid_train, cfg=pia_cfg
    )
    print(f"[PIA] candidates={len(y_aug)} gamma_mean={aug_meta['gamma_mean']:.4f}")

    # B) PIA (train-only)
    X_train_b = np.vstack([X_train_base, X_aug]) if len(y_aug) else X_train_base.copy()
    y_train_b = np.concatenate([y_train_base, y_aug]) if len(y_aug) else y_train_base.copy()
    tid_train_b = np.concatenate([tid_train, tid_aug]) if len(y_aug) else tid_train.copy()
    is_aug_b = (
        np.concatenate(
            [
                np.zeros((len(y_train_base),), dtype=bool),
                np.ones((len(y_aug),), dtype=bool),
            ]
        )
        if len(y_aug)
        else np.zeros((len(y_train_base),), dtype=bool)
    )

    metrics_b, train_meta_b = _fit_eval_linearsvc(
        X_train_b,
        y_train_b,
        tid_train_b,
        X_test,
        y_test,
        tid_test,
        seed=int(args.seed),
        cap_k=cap_k,
        cap_seed=int(args.seed) + 17,
        cap_sampling_policy=args.cap_sampling_policy,
        linear_c=float(args.linear_c),
        class_weight=args.linear_class_weight,
        max_iter=int(args.linear_max_iter),
        agg_mode=args.aggregation_mode,
        is_aug_train=is_aug_b,
    )
    write_json(os.path.join(cond_dirs["B_pia"], "metrics.json"), metrics_b)
    write_json(
        os.path.join(cond_dirs["B_pia"], "run_meta.json"),
        {
            **common_meta,
            **train_meta_b,
            "condition": "B_pia",
            "augmentation": aug_meta,
        },
    )
    summary_rows.append(
        {
            "condition": "B_pia",
            "acc": metrics_b["trial_acc"],
            "macro_f1": metrics_b["trial_macro_f1"],
            "train_windows": train_meta_b["total_train_windows_used"],
            "aug_total": aug_meta["aug_total_count"],
            "accept_rate": 1.0,
            "cap_sampling_policy": args.cap_sampling_policy,
            "train_selected_aug_ratio": train_meta_b["train_selected_aug_ratio"],
            "split_hash": common_meta["split_hash"],
        }
    )
    print(f"[B] acc={metrics_b['trial_acc']:.4f} macro_f1={metrics_b['trial_macro_f1']:.4f}")

    # C) PIA + manifold gate (train-only gate calibration)
    mu_y, tau_y, gate_fit_meta = _fit_gate_from_train(
        X_train=X_train_base, y_train=y_train_base, q=float(args.gate_percentile)
    )
    X_aug_keep, y_aug_keep, tid_aug_keep, gate_apply_meta = _apply_gate(
        X_aug=X_aug, y_aug=y_aug, tid_aug=tid_aug, mu=mu_y, tau=tau_y
    )
    X_train_c = np.vstack([X_train_base, X_aug_keep]) if len(y_aug_keep) else X_train_base.copy()
    y_train_c = np.concatenate([y_train_base, y_aug_keep]) if len(y_aug_keep) else y_train_base.copy()
    tid_train_c = np.concatenate([tid_train, tid_aug_keep]) if len(y_aug_keep) else tid_train.copy()
    is_aug_c = (
        np.concatenate(
            [
                np.zeros((len(y_train_base),), dtype=bool),
                np.ones((len(y_aug_keep),), dtype=bool),
            ]
        )
        if len(y_aug_keep)
        else np.zeros((len(y_train_base),), dtype=bool)
    )

    metrics_c, train_meta_c = _fit_eval_linearsvc(
        X_train_c,
        y_train_c,
        tid_train_c,
        X_test,
        y_test,
        tid_test,
        seed=int(args.seed),
        cap_k=cap_k,
        cap_seed=int(args.seed) + 23,
        cap_sampling_policy=args.cap_sampling_policy,
        linear_c=float(args.linear_c),
        class_weight=args.linear_class_weight,
        max_iter=int(args.linear_max_iter),
        agg_mode=args.aggregation_mode,
        is_aug_train=is_aug_c,
    )
    write_json(os.path.join(cond_dirs["C_pia_gate"], "metrics.json"), metrics_c)
    write_json(
        os.path.join(cond_dirs["C_pia_gate"], "run_meta.json"),
        {
            **common_meta,
            **train_meta_c,
            "condition": "C_pia_gate",
            "augmentation": aug_meta,
            "gate_fit": gate_fit_meta,
            "gate_apply": gate_apply_meta,
        },
    )
    summary_rows.append(
        {
            "condition": "C_pia_gate",
            "acc": metrics_c["trial_acc"],
            "macro_f1": metrics_c["trial_macro_f1"],
            "train_windows": train_meta_c["total_train_windows_used"],
            "aug_total": aug_meta["aug_total_count"],
            "accept_rate": gate_apply_meta["accept_rate"],
            "cap_sampling_policy": args.cap_sampling_policy,
            "train_selected_aug_ratio": train_meta_c["train_selected_aug_ratio"],
            "split_hash": common_meta["split_hash"],
        }
    )
    print(
        f"[C] acc={metrics_c['trial_acc']:.4f} macro_f1={metrics_c['trial_macro_f1']:.4f} "
        f"accept_rate={gate_apply_meta['accept_rate']:.4f}"
    )

    # Split hash lock check
    split_hashes = [r["split_hash"] for r in summary_rows]
    if len(set(split_hashes)) != 1:
        raise RuntimeError(f"Split hash mismatch across A/B/C: {split_hashes}")

    paired_df = pd.DataFrame(summary_rows)
    paired_csv = os.path.join(args.out_root, f"phase15_step0a_seed{args.seed}_paired.csv")
    ensure_dir(os.path.dirname(paired_csv))
    paired_df.to_csv(paired_csv, index=False)

    print(f"[done] A/B/C dirs:")
    for k, d in cond_dirs.items():
        print(f"  - {k}: {d}")
    print(f"[done] paired_csv={paired_csv}")


if __name__ == "__main__":
    main()
