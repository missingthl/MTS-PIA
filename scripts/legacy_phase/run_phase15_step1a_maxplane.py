#!/usr/bin/env python
"""Phase 15 Step 1A: Max-Plane (Tangent PCA) Feedback to PIA + Gate.

Paired conditions under one split (per seed):
- A: baseline
- C0: PIA + Gate1
- C1: Max-Plane sampling + Gate1
- C2: PIA -> Max-Plane projection feedback + Gate1

Optional Gate2 (source-distance) is implemented via flags.
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
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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
from scripts.legacy_phase.run_phase14r_step6b1_rev2 import (
    apply_logcenter,
    covs_to_features,
    ensure_dir,
    extract_features_block,
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


def _summary_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _apply_window_cap(
    X: np.ndarray,
    y: np.ndarray,
    tid: np.ndarray,
    cap_k: int,
    seed: int,
    *,
    is_aug: Optional[np.ndarray] = None,
    policy: str = "balanced_real_aug",
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

    def _aug_ratio(arr: np.ndarray) -> float:
        return float(np.mean(arr.astype(np.float64))) if arr.size else 0.0

    if cap_k <= 0:
        counts = _per_trial_window_counts(tid)
        return X, y, tid, is_aug_arr, counts, _aug_ratio(is_aug_arr)

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
                seed_sel = np.concatenate(parts) if parts else np.empty((0,), dtype=np.int64)
                rem = cap_k - seed_sel.size
                if rem > 0:
                    used = set(seed_sel.tolist())
                    rem_pool = np.asarray([i for i in idx.tolist() if i not in used], dtype=np.int64)
                    fill = rs.choice(rem_pool, size=rem, replace=False)
                    sel = np.concatenate([seed_sel, fill])
                else:
                    sel = seed_sel
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
    return X[keep], y[keep], tid_arr[keep], is_aug_keep, counts, _aug_ratio(is_aug_keep)


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    tid_arr = np.asarray(tid_train)
    y_arr = np.asarray(y_train).astype(int).ravel()
    trial_ids = sorted(_ordered_unique(tid_arr.tolist()))

    aug_X_parts: List[np.ndarray] = []
    aug_y_parts: List[np.ndarray] = []
    aug_tid_parts: List[np.ndarray] = []
    aug_src_parts: List[np.ndarray] = []
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
                seed=int(cfg.seed + m * 1009 + _stable_tid_hash(tid)),
            )
            X_aug = np.asarray(aug.fit_transform(X_tid), dtype=np.float32)
            st = aug.state()
            recon = st.get("recon_err")
            if isinstance(recon, list) and recon:
                recon_last.append(float(recon[-1]))

            if X_aug.shape[0] != X_tid.shape[0]:
                raise RuntimeError("PIA candidate shape mismatch with source windows.")

            aug_X_parts.append(X_aug)
            aug_y_parts.append(np.asarray(y_tid, dtype=np.int64))
            aug_tid_parts.append(np.asarray([tid] * len(idx)))
            aug_src_parts.append(np.asarray(X_tid, dtype=np.float32))
            gamma_list.extend([gamma] * len(idx))
            added += len(idx)
        aug_count_per_trial[tid] = int(added)

    if aug_X_parts:
        X_aug_all = np.vstack(aug_X_parts).astype(np.float32)
        y_aug_all = np.concatenate(aug_y_parts).astype(np.int64)
        tid_aug_all = np.concatenate(aug_tid_parts)
        src_aug_all = np.vstack(aug_src_parts).astype(np.float32)
    else:
        X_aug_all = np.empty((0, X_train.shape[1]), dtype=np.float32)
        y_aug_all = np.empty((0,), dtype=np.int64)
        tid_aug_all = np.empty((0,), dtype=object)
        src_aug_all = np.empty((0, X_train.shape[1]), dtype=np.float32)

    g = np.asarray(gamma_list, dtype=np.float64) if gamma_list else np.asarray([], dtype=np.float64)
    aug_vals = (
        np.asarray(list(aug_count_per_trial.values()), dtype=np.float64)
        if aug_count_per_trial
        else np.asarray([], dtype=np.float64)
    )
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
    return X_aug_all, y_aug_all, tid_aug_all, src_aug_all, meta


def _fit_gate1_from_train(
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
        mu[int(c)] = np.asarray(muc, dtype=np.float32)
        tau[int(c)] = float(np.percentile(d, q))
        dist_pool[int(c)] = d
    meta = {
        "gate1_metric": "tangent_l2_to_class_center",
        "gate1_percentile_q": float(q),
        "tau_y": {str(k): float(v) for k, v in tau.items()},
        "train_dist_summary_by_class": {
            str(k): _summary_stats(v) for k, v in dist_pool.items()
        },
    }
    return mu, tau, meta


def _apply_gates(
    X_aug: np.ndarray,
    y_aug: np.ndarray,
    tid_aug: np.ndarray,
    src_aug: np.ndarray,
    *,
    mu_y: Dict[int, np.ndarray],
    tau_y: Dict[int, float],
    enable_gate2: bool,
    gate2_q_src: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    if X_aug.shape[0] == 0:
        return (
            X_aug,
            y_aug,
            tid_aug,
            src_aug,
            {
                "accept_rate_gate1": 0.0,
                "accept_rate_gate2": 1.0,
                "accept_rate_final": 0.0,
                "accepted_count": 0,
                "rejected_count": 0,
                "gate1_dist_accepted_summary": _summary_stats(np.asarray([], dtype=np.float64)),
                "gate1_dist_rejected_summary": _summary_stats(np.asarray([], dtype=np.float64)),
            },
        )

    y = np.asarray(y_aug).astype(int).ravel()
    d_center = np.zeros((len(y),), dtype=np.float64)
    keep1 = np.zeros((len(y),), dtype=bool)
    for i, cls in enumerate(y.tolist()):
        muc = mu_y.get(int(cls))
        tauc = tau_y.get(int(cls))
        if muc is None or tauc is None:
            d_center[i] = np.inf
            keep1[i] = False
            continue
        di = float(np.linalg.norm(X_aug[i] - muc))
        d_center[i] = di
        keep1[i] = di <= tauc

    gate2_meta: Dict[str, object] = {
        "enabled": bool(enable_gate2),
        "q_src": float(gate2_q_src),
        "tau_src_y": {},
        "src_dist_accepted_summary": _summary_stats(np.asarray([], dtype=np.float64)),
        "src_dist_rejected_summary": _summary_stats(np.asarray([], dtype=np.float64)),
    }
    if enable_gate2:
        d_src = np.linalg.norm(np.asarray(X_aug, dtype=np.float32) - np.asarray(src_aug, dtype=np.float32), axis=1)
        tau_src_y: Dict[int, float] = {}
        for cls in sorted(np.unique(y).tolist()):
            ds = d_src[y == cls]
            if ds.size == 0:
                continue
            tau_src_y[int(cls)] = float(np.percentile(ds, gate2_q_src))
        keep2 = np.asarray(
            [d_src[i] <= tau_src_y.get(int(y[i]), -np.inf) for i in range(len(y))],
            dtype=bool,
        )
        gate2_meta = {
            "enabled": True,
            "q_src": float(gate2_q_src),
            "tau_src_y": {str(k): float(v) for k, v in tau_src_y.items()},
            "src_dist_accepted_summary": _summary_stats(d_src[keep2]),
            "src_dist_rejected_summary": _summary_stats(d_src[~keep2]),
        }
    else:
        keep2 = np.ones((len(y),), dtype=bool)

    keep = keep1 & keep2
    meta = {
        "accept_rate_gate1": float(np.mean(keep1)),
        "accept_rate_gate2": float(np.mean(keep2)),
        "accept_rate_final": float(np.mean(keep)),
        "accepted_count": int(np.sum(keep)),
        "rejected_count": int(np.sum(~keep)),
        "gate1_dist_accepted_summary": _summary_stats(d_center[keep]),
        "gate1_dist_rejected_summary": _summary_stats(d_center[~keep]),
        "gate2": gate2_meta,
    }
    return X_aug[keep], y_aug[keep], tid_aug[keep], src_aug[keep], meta


@dataclass
class MaxPlaneModel:
    k: int
    mu_y: Dict[int, np.ndarray]
    U_y: Dict[int, np.ndarray]
    sigma_y: Dict[int, float]
    effective_k_y: Dict[int, int]
    evr_topk_y: Dict[int, List[float]]
    fit_count_y: Dict[int, int]


def _chi_quantile_empirical(k: int, p: float, seed: int = 0, n: int = 200000) -> float:
    rs = np.random.RandomState(seed + 7919 * int(k))
    z = rs.normal(size=(n, max(1, int(k))))
    r = np.sqrt(np.sum(z * z, axis=1))
    return float(np.percentile(r, 100.0 * float(p)))


def _fit_max_plane(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    k: int,
    sigma_mode: str,
    gate_tau_y: Optional[Dict[int, float]],
    sigma_target_accept: float,
    seed: int,
) -> Tuple[MaxPlaneModel, Dict[str, object]]:
    y = np.asarray(y_train).astype(int).ravel()
    classes = sorted(np.unique(y).tolist())
    d = int(X_train.shape[1])

    mu_y: Dict[int, np.ndarray] = {}
    U_y: Dict[int, np.ndarray] = {}
    sigma_y: Dict[int, float] = {}
    effective_k_y: Dict[int, int] = {}
    evr_topk_y: Dict[int, List[float]] = {}
    fit_count_y: Dict[int, int] = {}

    for cls in classes:
        Xc = np.asarray(X_train[y == cls], dtype=np.float32)
        n_c = int(Xc.shape[0])
        fit_count_y[int(cls)] = n_c
        if n_c == 0:
            continue
        muc = np.mean(Xc, axis=0).astype(np.float32)
        centered = Xc - muc[None, :]
        n_comp = int(min(max(1, k), centered.shape[0], centered.shape[1]))
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=int(seed + cls * 131))
        pca.fit(centered)
        U = np.asarray(pca.components_.T, dtype=np.float32)  # [d, n_comp]
        evr = [float(v) for v in pca.explained_variance_ratio_.tolist()]

        rad = np.linalg.norm(centered, axis=1)
        if sigma_mode == "s1":
            sig = float(np.median(rad) / np.sqrt(max(1, n_comp)))
        elif sigma_mode == "s2":
            tau_c = 0.0 if gate_tau_y is None else float(gate_tau_y.get(int(cls), 0.0))
            q_chi = _chi_quantile_empirical(n_comp, sigma_target_accept, seed=seed + cls)
            sig = float(tau_c / max(q_chi, 1e-8))
        else:
            raise ValueError(f"Unknown sigma_mode: {sigma_mode}")
        if not np.isfinite(sig) or sig <= 0:
            sig = float(max(1e-4, np.std(rad) / np.sqrt(max(1, n_comp))))

        mu_y[int(cls)] = muc
        U_y[int(cls)] = U
        sigma_y[int(cls)] = sig
        effective_k_y[int(cls)] = int(n_comp)
        evr_topk_y[int(cls)] = evr

    model = MaxPlaneModel(
        k=int(k),
        mu_y=mu_y,
        U_y=U_y,
        sigma_y=sigma_y,
        effective_k_y=effective_k_y,
        evr_topk_y=evr_topk_y,
        fit_count_y=fit_count_y,
    )
    meta = {
        "k": int(k),
        "sigma_mode": sigma_mode,
        "sigma_target_accept": float(sigma_target_accept),
        "sigma_y": {str(c): float(v) for c, v in sigma_y.items()},
        "effective_k_y": {str(c): int(v) for c, v in effective_k_y.items()},
        "explained_variance_ratio_topk": {str(c): [float(x) for x in evr_topk_y[c]] for c in evr_topk_y},
        "pca_fit_count_y": {str(c): int(v) for c, v in fit_count_y.items()},
    }
    return model, meta


def _build_plane_aug_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tid_train: np.ndarray,
    model: MaxPlaneModel,
    *,
    multiplier: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train).astype(int).ravel()
    tid = np.asarray(tid_train)
    classes = sorted(np.unique(y).tolist())

    aug_X_parts: List[np.ndarray] = []
    aug_y_parts: List[np.ndarray] = []
    aug_tid_parts: List[np.ndarray] = []
    aug_src_parts: List[np.ndarray] = []
    aug_count_per_trial: Dict[str, int] = defaultdict(int)

    for cls in classes:
        idx = np.where(y == cls)[0]
        if idx.size == 0:
            continue
        U = model.U_y.get(int(cls))
        mu = model.mu_y.get(int(cls))
        sigma = float(model.sigma_y.get(int(cls), 0.0))
        if U is None or mu is None:
            continue
        k_eff = int(U.shape[1])
        X_src_c = X[idx]
        y_c = y[idx]
        tid_c = tid[idx]

        for m in range(max(0, int(multiplier))):
            rs = np.random.RandomState(int(seed + cls * 1009 + m * 65537))
            if k_eff > 0:
                eps = rs.normal(loc=0.0, scale=sigma, size=(idx.size, k_eff)).astype(np.float32)
                X_aug_c = mu[None, :] + np.asarray(eps @ U.T, dtype=np.float32)
            else:
                X_aug_c = np.repeat(mu[None, :], idx.size, axis=0).astype(np.float32)
            aug_X_parts.append(X_aug_c)
            aug_y_parts.append(np.asarray(y_c, dtype=np.int64))
            aug_tid_parts.append(np.asarray(tid_c))
            aug_src_parts.append(np.asarray(X_src_c, dtype=np.float32))
            for t in tid_c.tolist():
                aug_count_per_trial[str(t)] += 1

    if aug_X_parts:
        X_aug = np.vstack(aug_X_parts).astype(np.float32)
        y_aug = np.concatenate(aug_y_parts).astype(np.int64)
        tid_aug = np.concatenate(aug_tid_parts)
        src_aug = np.vstack(aug_src_parts).astype(np.float32)
    else:
        X_aug = np.empty((0, X.shape[1]), dtype=np.float32)
        y_aug = np.empty((0,), dtype=np.int64)
        tid_aug = np.empty((0,), dtype=object)
        src_aug = np.empty((0, X.shape[1]), dtype=np.float32)

    vals = (
        np.asarray(list(aug_count_per_trial.values()), dtype=np.float64)
        if aug_count_per_trial
        else np.asarray([], dtype=np.float64)
    )
    sig_vals = np.asarray(list(model.sigma_y.values()), dtype=np.float64) if model.sigma_y else np.asarray([], dtype=np.float64)
    meta = {
        "aug_total_count": int(len(y_aug)),
        "aug_count_per_trial": dict(aug_count_per_trial),
        "aug_per_trial_mean": float(np.mean(vals)) if vals.size else 0.0,
        "aug_per_trial_std": float(np.std(vals)) if vals.size else 0.0,
        "sigma_min": float(np.min(sig_vals)) if sig_vals.size else 0.0,
        "sigma_mean": float(np.mean(sig_vals)) if sig_vals.size else 0.0,
        "sigma_std": float(np.std(sig_vals)) if sig_vals.size else 0.0,
        "sigma_max": float(np.max(sig_vals)) if sig_vals.size else 0.0,
        "plane_multiplier": int(multiplier),
    }
    return X_aug, y_aug, tid_aug, src_aug, meta


def _project_to_max_plane(X_in: np.ndarray, y_in: np.ndarray, model: MaxPlaneModel) -> Tuple[np.ndarray, Dict[str, object]]:
    X = np.asarray(X_in, dtype=np.float32)
    y = np.asarray(y_in).astype(int).ravel()
    out = np.empty_like(X, dtype=np.float32)
    residual = np.zeros((len(y),), dtype=np.float64)

    for cls in sorted(np.unique(y).tolist()):
        idx = np.where(y == cls)[0]
        if idx.size == 0:
            continue
        U = model.U_y.get(int(cls))
        mu = model.mu_y.get(int(cls))
        if U is None or mu is None:
            out[idx] = X[idx]
            residual[idx] = 0.0
            continue
        centered = X[idx] - mu[None, :]
        if U.shape[1] > 0:
            proj = (centered @ U) @ U.T
        else:
            proj = np.zeros_like(centered)
        out[idx] = mu[None, :] + proj
        residual[idx] = np.linalg.norm(centered - proj, axis=1)

    meta = {
        "projection_residual_summary": _summary_stats(residual),
    }
    return out.astype(np.float32), meta


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
    progress_prefix: Optional[str] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    if progress_prefix:
        print(f"{progress_prefix} cap_start train_windows={len(y_train)}", flush=True)
    X_cap, y_cap, tid_cap, is_aug_cap, cap_counts, selected_aug_ratio = _apply_window_cap(
        np.asarray(X_train, dtype=np.float32),
        np.asarray(y_train).astype(int).ravel(),
        np.asarray(tid_train),
        cap_k=int(cap_k),
        seed=int(cap_seed),
        is_aug=is_aug_train,
        policy=cap_sampling_policy,
    )
    if progress_prefix:
        print(f"{progress_prefix} cap_done kept_windows={len(y_cap)}", flush=True)

    if progress_prefix:
        print(f"{progress_prefix} scale_start", flush=True)
    scaler = StandardScaler()
    X_cap_s = scaler.fit_transform(X_cap)
    X_te_s = scaler.transform(np.asarray(X_test, dtype=np.float32))
    if progress_prefix:
        print(
            f"{progress_prefix} scale_done train_shape={tuple(X_cap_s.shape)} "
            f"test_shape={tuple(X_te_s.shape)}",
            flush=True,
        )

    cw = None if class_weight in {None, "", "none"} else class_weight
    clf = LinearSVC(
        C=float(linear_c),
        class_weight=cw,
        max_iter=int(max_iter),
        random_state=int(seed),
        dual="auto",
    )
    if progress_prefix:
        print(f"{progress_prefix} fit_start", flush=True)
    clf.fit(X_cap_s, y_cap)
    if progress_prefix:
        print(f"{progress_prefix} fit_done", flush=True)

    if progress_prefix:
        print(f"{progress_prefix} predict_start", flush=True)
    y_pred_win = clf.predict(X_te_s)
    scores_win = clf.decision_function(X_te_s)
    if scores_win.ndim == 1:
        scores_win = np.vstack([-scores_win, scores_win]).T
    if progress_prefix:
        print(f"{progress_prefix} predict_done", flush=True)

    y_true_trial, y_pred_trial = _aggregate_trials(
        y_true_win=np.asarray(y_test).astype(int).ravel(),
        y_pred_win=y_pred_win,
        scores_win=scores_win,
        tid_win=np.asarray(tid_test),
        mode=agg_mode,
    )

    metrics = {
        "trial_acc": float(accuracy_score(y_true_trial, y_pred_trial)),
        "trial_macro_f1": float(f1_score(y_true_trial, y_pred_trial, average="macro")),
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


def _parse_k_list(text: str) -> List[int]:
    vals = []
    for tok in str(text).split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(int(t))
    uniq = sorted(set(vals))
    if not uniq:
        raise ValueError("k list is empty.")
    if any(v <= 0 for v in uniq):
        raise ValueError("k must be positive.")
    return uniq


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
    parser.add_argument("--out-root", type=str, default="out/phase15_step1a")

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

    parser.add_argument("--linear-c", type=float, default=1.0)
    parser.add_argument("--linear-class-weight", type=str, default="none")
    parser.add_argument("--linear-max-iter", type=int, default=1000)

    parser.add_argument("--window-cap-k", type=int, default=120)
    parser.add_argument(
        "--cap-sampling-policy",
        type=str,
        default="balanced_real_aug",
        choices=["random", "balanced_real_aug", "prefer_real", "prefer_aug"],
    )

    parser.add_argument("--pia-multiplier", type=int, default=1)
    parser.add_argument("--pia-gamma", type=float, default=0.10)
    parser.add_argument("--pia-gamma-jitter", type=float, default=0.0)
    parser.add_argument("--pia-n-iters", type=int, default=2)
    parser.add_argument("--pia-activation", type=str, default="sine")
    parser.add_argument("--pia-bias-update-mode", type=str, default="residual")
    parser.add_argument("--pia-c-repr", type=float, default=1.0)

    parser.add_argument("--gate-percentile", type=float, default=95.0)
    parser.add_argument("--enable-gate2", action="store_true")
    parser.add_argument("--gate2-q-src", type=float, default=95.0)

    parser.add_argument("--k-list", type=str, default="4")
    parser.add_argument("--sigma-mode", type=str, default="s1", choices=["s1", "s2"])
    parser.add_argument("--sigma-target-accept", type=float, default=0.95)
    parser.add_argument("--plane-multiplier", type=int, default=1)

    parser.add_argument("--split-preview-n", type=int, default=5)
    args = parser.parse_args()
    args.dataset = normalize_dataset_name(args.dataset)

    if args.pia_multiplier < 0:
        raise ValueError("--pia-multiplier must be >= 0")
    if args.window_cap_k <= 0:
        raise ValueError("--window-cap-k must be > 0 in Step1A lock.")
    if args.gate_percentile <= 0 or args.gate_percentile > 100:
        raise ValueError("--gate-percentile must be in (0,100].")
    if args.gate2_q_src <= 0 or args.gate2_q_src > 100:
        raise ValueError("--gate2-q-src must be in (0,100].")
    k_list = _parse_k_list(args.k_list)

    # 1) Load + split once
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

    # 2) Feature extraction once
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
    print(f"[cap] K={args.window_cap_k} policy={args.cap_sampling_policy}")

    # Gate1 calibration on TRAIN only
    mu_gate1, tau_gate1, gate1_fit_meta = _fit_gate1_from_train(
        X_train=X_train_base, y_train=y_train_base, q=float(args.gate_percentile)
    )

    # Shared PIA candidates for C0/C2
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
    X_pia, y_pia, tid_pia, src_pia, pia_meta = _build_pia_aug_candidates(
        X_train=X_train_base,
        y_train=y_train_base,
        tid_train=tid_train,
        cfg=pia_cfg,
    )
    print(f"[PIA] candidates={len(y_pia)} gamma_mean={pia_meta['gamma_mean']:.4f}")

    X_c0_keep, y_c0_keep, tid_c0_keep, src_c0_keep, c0_gate_meta = _apply_gates(
        X_aug=X_pia,
        y_aug=y_pia,
        tid_aug=tid_pia,
        src_aug=src_pia,
        mu_y=mu_gate1,
        tau_y=tau_gate1,
        enable_gate2=bool(args.enable_gate2),
        gate2_q_src=float(args.gate2_q_src),
    )
    print(f"[C0] pre_gate={len(y_pia)} post_gate={len(y_c0_keep)} accept={c0_gate_meta['accept_rate_final']:.4f}")

    common_meta = {
        "seed": int(args.seed),
        "split_hash": split_meta["split_hash"],
        "train_count_trials": int(split_meta["train_count_trials"]),
        "test_count_trials": int(split_meta["test_count_trials"]),
        "train_trial_ids_preview": split_meta["train_trial_ids"][: max(0, int(args.split_preview_n))],
        "test_trial_ids_preview": split_meta["test_trial_ids"][: max(0, int(args.split_preview_n))],
        "window_cap_K": int(args.window_cap_k),
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

    cap_seed = int(args.seed) + 41

    # A baseline (shared for every k)
    metrics_a, train_meta_a = _fit_eval_linearsvc(
        X_train_base,
        y_train_base,
        tid_train,
        X_test,
        y_test,
        tid_test,
        seed=int(args.seed),
        cap_k=int(args.window_cap_k),
        cap_seed=cap_seed,
        cap_sampling_policy=args.cap_sampling_policy,
        linear_c=float(args.linear_c),
        class_weight=args.linear_class_weight,
        max_iter=int(args.linear_max_iter),
        agg_mode=args.aggregation_mode,
        is_aug_train=np.zeros((len(y_train_base),), dtype=bool),
    )

    # C0 baseline augmenter (shared for every k)
    X_train_c0 = np.vstack([X_train_base, X_c0_keep]) if len(y_c0_keep) else X_train_base.copy()
    y_train_c0 = np.concatenate([y_train_base, y_c0_keep]) if len(y_c0_keep) else y_train_base.copy()
    tid_train_c0 = np.concatenate([tid_train, tid_c0_keep]) if len(y_c0_keep) else tid_train.copy()
    is_aug_c0 = (
        np.concatenate(
            [
                np.zeros((len(y_train_base),), dtype=bool),
                np.ones((len(y_c0_keep),), dtype=bool),
            ]
        )
        if len(y_c0_keep)
        else np.zeros((len(y_train_base),), dtype=bool)
    )
    metrics_c0, train_meta_c0 = _fit_eval_linearsvc(
        X_train_c0,
        y_train_c0,
        tid_train_c0,
        X_test,
        y_test,
        tid_test,
        seed=int(args.seed),
        cap_k=int(args.window_cap_k),
        cap_seed=cap_seed,
        cap_sampling_policy=args.cap_sampling_policy,
        linear_c=float(args.linear_c),
        class_weight=args.linear_class_weight,
        max_iter=int(args.linear_max_iter),
        agg_mode=args.aggregation_mode,
        is_aug_train=is_aug_c0,
    )
    print(f"[A] acc={metrics_a['trial_acc']:.4f} f1={metrics_a['trial_macro_f1']:.4f}")
    print(f"[C0] acc={metrics_c0['trial_acc']:.4f} f1={metrics_c0['trial_macro_f1']:.4f}")

    # Per-k runs for C1/C2 + writing all A/C0/C1/C2
    for k in k_list:
        model, plane_meta = _fit_max_plane(
            X_train=X_train_base,
            y_train=y_train_base,
            k=int(k),
            sigma_mode=args.sigma_mode,
            gate_tau_y=tau_gate1,
            sigma_target_accept=float(args.sigma_target_accept),
            seed=int(args.seed),
        )

        # C1: max-plane sampling
        X_c1, y_c1, tid_c1, src_c1, c1_aug_meta = _build_plane_aug_candidates(
            X_train=X_train_base,
            y_train=y_train_base,
            tid_train=tid_train,
            model=model,
            multiplier=int(args.plane_multiplier),
            seed=int(args.seed) + 1701,
        )
        X_c1_keep, y_c1_keep, tid_c1_keep, src_c1_keep, c1_gate_meta = _apply_gates(
            X_aug=X_c1,
            y_aug=y_c1,
            tid_aug=tid_c1,
            src_aug=src_c1,
            mu_y=mu_gate1,
            tau_y=tau_gate1,
            enable_gate2=bool(args.enable_gate2),
            gate2_q_src=float(args.gate2_q_src),
        )

        X_train_c1 = np.vstack([X_train_base, X_c1_keep]) if len(y_c1_keep) else X_train_base.copy()
        y_train_c1 = np.concatenate([y_train_base, y_c1_keep]) if len(y_c1_keep) else y_train_base.copy()
        tid_train_c1 = np.concatenate([tid_train, tid_c1_keep]) if len(y_c1_keep) else tid_train.copy()
        is_aug_c1 = (
            np.concatenate(
                [
                    np.zeros((len(y_train_base),), dtype=bool),
                    np.ones((len(y_c1_keep),), dtype=bool),
                ]
            )
            if len(y_c1_keep)
            else np.zeros((len(y_train_base),), dtype=bool)
        )
        metrics_c1, train_meta_c1 = _fit_eval_linearsvc(
            X_train_c1,
            y_train_c1,
            tid_train_c1,
            X_test,
            y_test,
            tid_test,
            seed=int(args.seed),
            cap_k=int(args.window_cap_k),
            cap_seed=cap_seed,
            cap_sampling_policy=args.cap_sampling_policy,
            linear_c=float(args.linear_c),
            class_weight=args.linear_class_weight,
            max_iter=int(args.linear_max_iter),
            agg_mode=args.aggregation_mode,
            is_aug_train=is_aug_c1,
        )

        # C2: PIA -> max-plane projection feedback
        X_pia_fb, c2_fb_meta = _project_to_max_plane(X_pia, y_pia, model)
        X_c2_keep, y_c2_keep, tid_c2_keep, src_c2_keep, c2_gate_meta = _apply_gates(
            X_aug=X_pia_fb,
            y_aug=y_pia,
            tid_aug=tid_pia,
            src_aug=src_pia,
            mu_y=mu_gate1,
            tau_y=tau_gate1,
            enable_gate2=bool(args.enable_gate2),
            gate2_q_src=float(args.gate2_q_src),
        )
        X_train_c2 = np.vstack([X_train_base, X_c2_keep]) if len(y_c2_keep) else X_train_base.copy()
        y_train_c2 = np.concatenate([y_train_base, y_c2_keep]) if len(y_c2_keep) else y_train_base.copy()
        tid_train_c2 = np.concatenate([tid_train, tid_c2_keep]) if len(y_c2_keep) else tid_train.copy()
        is_aug_c2 = (
            np.concatenate(
                [
                    np.zeros((len(y_train_base),), dtype=bool),
                    np.ones((len(y_c2_keep),), dtype=bool),
                ]
            )
            if len(y_c2_keep)
            else np.zeros((len(y_train_base),), dtype=bool)
        )
        metrics_c2, train_meta_c2 = _fit_eval_linearsvc(
            X_train_c2,
            y_train_c2,
            tid_train_c2,
            X_test,
            y_test,
            tid_test,
            seed=int(args.seed),
            cap_k=int(args.window_cap_k),
            cap_seed=cap_seed,
            cap_sampling_policy=args.cap_sampling_policy,
            linear_c=float(args.linear_c),
            class_weight=args.linear_class_weight,
            max_iter=int(args.linear_max_iter),
            agg_mode=args.aggregation_mode,
            is_aug_train=is_aug_c2,
        )
        print(
            f"[k={k}] C1 acc={metrics_c1['trial_acc']:.4f} f1={metrics_c1['trial_macro_f1']:.4f} "
            f"accept={c1_gate_meta['accept_rate_final']:.4f} | "
            f"C2 acc={metrics_c2['trial_acc']:.4f} f1={metrics_c2['trial_macro_f1']:.4f} "
            f"accept={c2_gate_meta['accept_rate_final']:.4f}"
        )

        k_dir = os.path.join(args.out_root, f"k{k}")
        seed_dir = os.path.join(k_dir, f"seed{args.seed}")
        cond_dirs = {
            "A_baseline": os.path.join(seed_dir, "A_baseline"),
            "C0_pia_gate": os.path.join(seed_dir, "C0_pia_gate"),
            "C1_plane_gate": os.path.join(seed_dir, "C1_plane_gate"),
            "C2_pia_plane_gate": os.path.join(seed_dir, "C2_pia_plane_gate"),
        }
        for d in cond_dirs.values():
            ensure_dir(d)

        # Write condition artifacts
        write_json(os.path.join(cond_dirs["A_baseline"], "metrics.json"), metrics_a)
        write_json(
            os.path.join(cond_dirs["A_baseline"], "run_meta.json"),
            {**common_meta, **train_meta_a, "condition": "A_baseline", "k": int(k)},
        )

        write_json(os.path.join(cond_dirs["C0_pia_gate"], "metrics.json"), metrics_c0)
        write_json(
            os.path.join(cond_dirs["C0_pia_gate"], "run_meta.json"),
            {
                **common_meta,
                **train_meta_c0,
                "condition": "C0_pia_gate",
                "k": int(k),
                "augmentation": pia_meta,
                "gate1_fit": gate1_fit_meta,
                "gate_apply": c0_gate_meta,
            },
        )

        write_json(os.path.join(cond_dirs["C1_plane_gate"], "metrics.json"), metrics_c1)
        write_json(
            os.path.join(cond_dirs["C1_plane_gate"], "run_meta.json"),
            {
                **common_meta,
                **train_meta_c1,
                "condition": "C1_plane_gate",
                "k": int(k),
                "max_plane": plane_meta,
                "augmentation": c1_aug_meta,
                "gate1_fit": gate1_fit_meta,
                "gate_apply": c1_gate_meta,
            },
        )

        write_json(os.path.join(cond_dirs["C2_pia_plane_gate"], "metrics.json"), metrics_c2)
        write_json(
            os.path.join(cond_dirs["C2_pia_plane_gate"], "run_meta.json"),
            {
                **common_meta,
                **train_meta_c2,
                "condition": "C2_pia_plane_gate",
                "k": int(k),
                "max_plane": plane_meta,
                "augmentation": {
                    **pia_meta,
                    "feedback": "pia_project_to_max_plane",
                    **c2_fb_meta,
                },
                "gate1_fit": gate1_fit_meta,
                "gate_apply": c2_gate_meta,
            },
        )

        rows = [
            {
                "condition": "A_baseline",
                "acc": metrics_a["trial_acc"],
                "macro_f1": metrics_a["trial_macro_f1"],
                "accept_rate": 1.0,
                "split_hash": split_meta["split_hash"],
                "k": int(k),
            },
            {
                "condition": "C0_pia_gate",
                "acc": metrics_c0["trial_acc"],
                "macro_f1": metrics_c0["trial_macro_f1"],
                "accept_rate": c0_gate_meta["accept_rate_final"],
                "split_hash": split_meta["split_hash"],
                "k": int(k),
            },
            {
                "condition": "C1_plane_gate",
                "acc": metrics_c1["trial_acc"],
                "macro_f1": metrics_c1["trial_macro_f1"],
                "accept_rate": c1_gate_meta["accept_rate_final"],
                "split_hash": split_meta["split_hash"],
                "k": int(k),
            },
            {
                "condition": "C2_pia_plane_gate",
                "acc": metrics_c2["trial_acc"],
                "macro_f1": metrics_c2["trial_macro_f1"],
                "accept_rate": c2_gate_meta["accept_rate_final"],
                "split_hash": split_meta["split_hash"],
                "k": int(k),
            },
        ]
        paired_df = pd.DataFrame(rows)
        paired_csv = os.path.join(k_dir, f"phase15_step1a_seed{args.seed}_paired.csv")
        ensure_dir(os.path.dirname(paired_csv))
        paired_df.to_csv(paired_csv, index=False)
        print(f"[done-k] k={k} paired_csv={paired_csv}")


if __name__ == "__main__":
    main()
