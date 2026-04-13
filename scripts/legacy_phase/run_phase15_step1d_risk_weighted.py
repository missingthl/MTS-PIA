#!/usr/bin/env python
"""Phase 15 Step 1D: Direction-risk weighted sampling (class-conditional).

Locked setting:
- k_dir=5, subset_size=1, gamma=0.10
- Gate1(q=95) + Gate2(q_src=90)
- balanced_real_aug cap K=120

Conditions:
- A_baseline
- C_uniform
- C_w_global
- C_w_class
- C_trunc_class
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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
from scripts.legacy_phase.run_phase15_step1a_maxplane import (
    _apply_gates,
    _fit_eval_linearsvc,
    _fit_gate1_from_train,
    _make_trial_split,
)
from scripts.legacy_phase.run_phase15_step1b_multidir_matrix import _build_direction_bank_d1


def _stable_tid_hash(tid: str) -> int:
    h = hashlib.sha256(str(tid).encode("utf-8")).hexdigest()[:16]
    return int(h, 16) & 0x7FFFFFFF


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


def _parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for tok in str(text).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(int(t))
    out = sorted(set(out))
    if not out:
        raise ValueError("int list cannot be empty")
    return out


def _parse_seed_list(text: str) -> List[int]:
    return _parse_int_list(text)


def _parse_methods(text: str) -> List[str]:
    allowed = {"uniform", "w_global", "w_class", "trunc_class"}
    out = [t.strip() for t in str(text).split(",") if t.strip()]
    if not out:
        out = ["uniform", "w_global", "w_class", "trunc_class"]
    for m in out:
        if m not in allowed:
            raise ValueError(f"Unknown method: {m}")
    return out


def _quantile_stats(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _softmax_neg_beta(risk: np.ndarray, beta: float) -> np.ndarray:
    x = -float(beta) * np.asarray(risk, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    s = np.sum(ex)
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(ex) / max(1, ex.size)
    return ex / s


def _probs_entropy_norm(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    h = -np.sum(p * np.log(p))
    return float(h / np.log(max(2, p.size)))


def _sample_dirs_by_policy(
    rs: np.random.RandomState,
    y_row: np.ndarray,
    *,
    k_dir: int,
    method: str,
    probs_global: Optional[np.ndarray],
    probs_class: Optional[Dict[int, np.ndarray]],
) -> np.ndarray:
    n = int(y_row.shape[0])
    dirs = np.zeros((n,), dtype=np.int64)
    if method == "uniform":
        dirs = rs.randint(0, k_dir, size=n, dtype=np.int64)
        return dirs
    if method == "w_global":
        pg = probs_global if probs_global is not None else np.ones((k_dir,), dtype=np.float64) / k_dir
        dirs = rs.choice(k_dir, size=n, replace=True, p=pg).astype(np.int64)
        return dirs

    # class-conditional methods
    if probs_class is None:
        raise ValueError("probs_class is required for class-conditional methods")
    for cls in sorted(np.unique(y_row).tolist()):
        idx = np.where(y_row == cls)[0]
        pc = probs_class.get(int(cls))
        if pc is None:
            pc = np.ones((k_dir,), dtype=np.float64) / k_dir
        dirs[idx] = rs.choice(k_dir, size=idx.size, replace=True, p=pc).astype(np.int64)
    return dirs


def _build_aug_candidates_policy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tid_train: np.ndarray,
    *,
    direction_bank: np.ndarray,
    gamma: float,
    multiplier: int,
    method: str,
    probs_global: Optional[np.ndarray],
    probs_class: Optional[Dict[int, np.ndarray]],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    k_dir = int(direction_bank.shape[0])
    tid_arr = np.asarray(tid_train)
    y_arr = np.asarray(y_train).astype(int).ravel()
    trial_ids = sorted(_ordered_unique(tid_arr.tolist()))

    aug_x_parts: List[np.ndarray] = []
    aug_y_parts: List[np.ndarray] = []
    aug_tid_parts: List[np.ndarray] = []
    aug_src_parts: List[np.ndarray] = []
    aug_count_per_trial: Dict[str, int] = {}

    sel_count_global = np.zeros((k_dir,), dtype=np.int64)
    sel_count_by_class: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros((k_dir,), dtype=np.int64))

    for tid in trial_ids:
        idx = np.where(tid_arr == tid)[0]
        X_tid = np.asarray(X_train[idx], dtype=np.float32)
        y_tid = np.asarray(y_arr[idx], dtype=np.int64)
        added = 0
        for m in range(max(0, int(multiplier))):
            rs = np.random.RandomState(int(seed + m * 1009 + _stable_tid_hash(tid)))
            dirs = _sample_dirs_by_policy(
                rs,
                y_tid,
                k_dir=k_dir,
                method=method,
                probs_global=probs_global,
                probs_class=probs_class,
            )
            signs = rs.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=dirs.shape[0], replace=True)
            w = direction_bank[dirs] * signs[:, None]
            X_aug = (X_tid + float(gamma) * w).astype(np.float32)

            aug_x_parts.append(X_aug)
            aug_y_parts.append(y_tid.copy())
            aug_tid_parts.append(np.asarray([tid] * len(idx)))
            aug_src_parts.append(X_tid.copy())
            added += len(idx)

            binc = np.bincount(dirs, minlength=k_dir)
            sel_count_global += binc.astype(np.int64)
            for cls in np.unique(y_tid):
                cidx = np.where(y_tid == cls)[0]
                cb = np.bincount(dirs[cidx], minlength=k_dir)
                sel_count_by_class[int(cls)] += cb.astype(np.int64)

        aug_count_per_trial[tid] = int(added)

    if aug_x_parts:
        X_aug_all = np.vstack(aug_x_parts).astype(np.float32)
        y_aug_all = np.concatenate(aug_y_parts).astype(np.int64)
        tid_aug_all = np.concatenate(aug_tid_parts)
        src_aug_all = np.vstack(aug_src_parts).astype(np.float32)
    else:
        X_aug_all = np.empty((0, X_train.shape[1]), dtype=np.float32)
        y_aug_all = np.empty((0,), dtype=np.int64)
        tid_aug_all = np.empty((0,), dtype=object)
        src_aug_all = np.empty((0, X_train.shape[1]), dtype=np.float32)

    total_sel = int(np.sum(sel_count_global))
    top1_freq = float(np.max(sel_count_global) / max(1, total_sel))
    p_emp = sel_count_global.astype(np.float64) / max(1, np.sum(sel_count_global))
    entropy_norm = _probs_entropy_norm(p_emp)
    aug_vals = np.asarray(list(aug_count_per_trial.values()), dtype=np.float64) if aug_count_per_trial else np.asarray([], dtype=np.float64)

    meta = {
        "aug_total_count": int(len(y_aug_all)),
        "aug_count_per_trial": aug_count_per_trial,
        "aug_per_trial_mean": float(np.mean(aug_vals)) if aug_vals.size else 0.0,
        "aug_per_trial_std": float(np.std(aug_vals)) if aug_vals.size else 0.0,
        "gamma": float(gamma),
        "sampling_method": method,
        "direction_select_count_global": {str(i): int(v) for i, v in enumerate(sel_count_global.tolist())},
        "direction_select_top1_freq": float(top1_freq),
        "direction_select_entropy_norm": float(entropy_norm),
        "direction_select_count_by_class": {
            str(c): {str(i): int(v) for i, v in enumerate(arr.tolist())}
            for c, arr in sel_count_by_class.items()
        },
    }
    return X_aug_all, y_aug_all, tid_aug_all, src_aug_all, meta


def _probe_direction_risks(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    direction_bank: np.ndarray,
    gamma: float,
    alpha: float,
    metric_quantile: float,
    n_probe: int,
    n_min: int,
    mu_y: Dict[int, np.ndarray],
    tau_y: Dict[int, float],
    gate2_q_src: float,
    seed: int,
    fallback_inflate: float,
) -> Dict[str, object]:
    y_arr = np.asarray(y_train).astype(int).ravel()
    classes = sorted(np.unique(y_arr).tolist())
    k_dir = int(direction_bank.shape[0])

    # class -> dir -> stats
    risk_cls_dir: Dict[int, Dict[int, float]] = defaultdict(dict)
    accept_rate_cls_dir: Dict[int, Dict[int, float]] = defaultdict(dict)
    accepted_count_cls_dir: Dict[int, Dict[int, int]] = defaultdict(dict)
    p95_src_cls_dir: Dict[int, Dict[int, float]] = defaultdict(dict)
    p95_cls_cls_dir: Dict[int, Dict[int, float]] = defaultdict(dict)
    fallback_cls_dir: Dict[int, Dict[int, bool]] = defaultdict(dict)

    # for global aggregation
    d_src_acc_by_dir: Dict[int, List[np.ndarray]] = defaultdict(list)
    d_cls_acc_by_dir: Dict[int, List[np.ndarray]] = defaultdict(list)
    d_src_all_by_dir: Dict[int, List[np.ndarray]] = defaultdict(list)
    d_cls_all_by_dir: Dict[int, List[np.ndarray]] = defaultdict(list)
    acc_cnt_dir: Dict[int, int] = defaultdict(int)
    tot_cnt_dir: Dict[int, int] = defaultdict(int)

    for cls in classes:
        idx_cls = np.where(y_arr == cls)[0]
        if idx_cls.size == 0:
            continue
        mu = mu_y.get(int(cls))
        tau = tau_y.get(int(cls))
        if mu is None or tau is None:
            continue

        for i in range(k_dir):
            rs = np.random.RandomState(int(seed + cls * 10007 + i * 131))
            rep = idx_cls.size < n_probe
            pick = rs.choice(idx_cls, size=min(n_probe, idx_cls.size) if not rep else n_probe, replace=rep)
            X_src = np.asarray(X_train[pick], dtype=np.float32)
            n = int(X_src.shape[0])
            signs = rs.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=n, replace=True)
            w = direction_bank[i][None, :] * signs[:, None]
            X_aug = (X_src + float(gamma) * w).astype(np.float32)

            d_src = np.linalg.norm(X_aug - X_src, axis=1).astype(np.float64)
            d_cls = np.linalg.norm(X_aug - np.asarray(mu, dtype=np.float32)[None, :], axis=1).astype(np.float64)
            keep1 = d_cls <= float(tau)
            tau_src = float(np.percentile(d_src, gate2_q_src))
            keep2 = d_src <= tau_src
            keep = keep1 & keep2

            d_src_acc = d_src[keep]
            d_cls_acc = d_cls[keep]
            n_acc = int(np.sum(keep))
            acc_rate = float(n_acc / max(1, n))

            q = float(metric_quantile)
            if n_acc >= int(n_min):
                p_src = float(np.percentile(d_src_acc, q))
                p_cls = float(np.percentile(d_cls_acc, q))
                fallback = False
            else:
                p_src = float(np.mean(d_src)) * float(fallback_inflate)
                p_cls = float(np.mean(d_cls)) * float(fallback_inflate)
                fallback = True
            r = float(alpha) * p_src + (1.0 - float(alpha)) * p_cls

            risk_cls_dir[int(cls)][int(i)] = float(r)
            accept_rate_cls_dir[int(cls)][int(i)] = float(acc_rate)
            accepted_count_cls_dir[int(cls)][int(i)] = int(n_acc)
            p95_src_cls_dir[int(cls)][int(i)] = float(p_src)
            p95_cls_cls_dir[int(cls)][int(i)] = float(p_cls)
            fallback_cls_dir[int(cls)][int(i)] = bool(fallback)

            d_src_all_by_dir[int(i)].append(d_src)
            d_cls_all_by_dir[int(i)].append(d_cls)
            if n_acc > 0:
                d_src_acc_by_dir[int(i)].append(d_src_acc)
                d_cls_acc_by_dir[int(i)].append(d_cls_acc)
            acc_cnt_dir[int(i)] += int(n_acc)
            tot_cnt_dir[int(i)] += int(n)

    risk_global: Dict[int, float] = {}
    accept_rate_global: Dict[int, float] = {}
    fallback_global: Dict[int, bool] = {}
    p_src_global: Dict[int, float] = {}
    p_cls_global: Dict[int, float] = {}
    for i in range(k_dir):
        dsa = np.concatenate(d_src_acc_by_dir[i]) if d_src_acc_by_dir[i] else np.asarray([], dtype=np.float64)
        dca = np.concatenate(d_cls_acc_by_dir[i]) if d_cls_acc_by_dir[i] else np.asarray([], dtype=np.float64)
        dsall = np.concatenate(d_src_all_by_dir[i]) if d_src_all_by_dir[i] else np.asarray([], dtype=np.float64)
        dcall = np.concatenate(d_cls_all_by_dir[i]) if d_cls_all_by_dir[i] else np.asarray([], dtype=np.float64)
        q = float(metric_quantile)
        if dsa.size >= int(n_min):
            ps = float(np.percentile(dsa, q))
            pc = float(np.percentile(dca, q))
            fb = False
        else:
            ps = float(np.mean(dsall)) * float(fallback_inflate) if dsall.size else 1e6
            pc = float(np.mean(dcall)) * float(fallback_inflate) if dcall.size else 1e6
            fb = True
        risk_global[i] = float(alpha) * ps + (1.0 - float(alpha)) * pc
        accept_rate_global[i] = float(acc_cnt_dir[i] / max(1, tot_cnt_dir[i]))
        fallback_global[i] = fb
        p_src_global[i] = ps
        p_cls_global[i] = pc

    return {
        "risk_global": {str(i): float(v) for i, v in risk_global.items()},
        "accept_rate_global": {str(i): float(v) for i, v in accept_rate_global.items()},
        "p95_src_global": {str(i): float(v) for i, v in p_src_global.items()},
        "p95_cls_global": {str(i): float(v) for i, v in p_cls_global.items()},
        "fallback_global": {str(i): bool(v) for i, v in fallback_global.items()},
        "risk_class": {
            str(c): {str(i): float(v) for i, v in d.items()} for c, d in risk_cls_dir.items()
        },
        "accept_rate_class": {
            str(c): {str(i): float(v) for i, v in d.items()} for c, d in accept_rate_cls_dir.items()
        },
        "accepted_count_class": {
            str(c): {str(i): int(v) for i, v in d.items()} for c, d in accepted_count_cls_dir.items()
        },
        "p95_src_class": {
            str(c): {str(i): float(v) for i, v in d.items()} for c, d in p95_src_cls_dir.items()
        },
        "p95_cls_class": {
            str(c): {str(i): float(v) for i, v in d.items()} for c, d in p95_cls_cls_dir.items()
        },
        "fallback_class": {
            str(c): {str(i): bool(v) for i, v in d.items()} for c, d in fallback_cls_dir.items()
        },
    }


def _probs_from_risk_global(
    risk_global: Dict[str, float],
    *,
    beta: float,
    k_dir: int,
) -> np.ndarray:
    r = np.asarray([float(risk_global.get(str(i), 1e6)) for i in range(k_dir)], dtype=np.float64)
    return _softmax_neg_beta(r, beta=beta)


def _probs_from_risk_class(
    risk_class: Dict[str, Dict[str, float]],
    *,
    beta: float,
    k_dir: int,
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for cls_s, d in risk_class.items():
        r = np.asarray([float(d.get(str(i), 1e6)) for i in range(k_dir)], dtype=np.float64)
        out[int(cls_s)] = _softmax_neg_beta(r, beta=beta)
    return out


def _probs_from_risk_trunc_class(
    risk_class: Dict[str, Dict[str, float]],
    *,
    k_dir: int,
    trunc_m: int,
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    m = int(max(0, trunc_m))
    for cls_s, d in risk_class.items():
        r = np.asarray([float(d.get(str(i), 1e6)) for i in range(k_dir)], dtype=np.float64)
        order = np.argsort(r)  # low risk first
        keep = np.ones((k_dir,), dtype=bool)
        if m > 0:
            drop = order[-min(m, k_dir):]
            keep[drop] = False
        if not np.any(keep):
            keep[:] = True
        p = keep.astype(np.float64)
        p = p / np.sum(p)
        out[int(cls_s)] = p
    return out


def _write_condition(cond_dir: str, metrics: Dict, run_meta: Dict) -> None:
    ensure_dir(cond_dir)
    write_json(os.path.join(cond_dir, "metrics.json"), metrics)
    write_json(os.path.join(cond_dir, "run_meta.json"), run_meta)


def _run_condition(
    *,
    X_train_base: np.ndarray,
    y_train_base: np.ndarray,
    tid_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tid_test: np.ndarray,
    X_aug: np.ndarray,
    y_aug: np.ndarray,
    tid_aug: np.ndarray,
    src_aug: np.ndarray,
    seed: int,
    cap_k: int,
    cap_seed: int,
    cap_policy: str,
    linear_c: float,
    class_weight: str,
    max_iter: int,
    agg_mode: str,
    mu_gate1: Dict[int, np.ndarray],
    tau_gate1: Dict[int, float],
    gate2_q_src: float,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    X_keep, y_keep, tid_keep, _, gate_meta = _apply_gates(
        X_aug=X_aug,
        y_aug=y_aug,
        tid_aug=tid_aug,
        src_aug=src_aug,
        mu_y=mu_gate1,
        tau_y=tau_gate1,
        enable_gate2=True,
        gate2_q_src=float(gate2_q_src),
    )
    X_train = np.vstack([X_train_base, X_keep]) if len(y_keep) else X_train_base.copy()
    y_train = np.concatenate([y_train_base, y_keep]) if len(y_keep) else y_train_base.copy()
    tid_train_cond = np.concatenate([tid_train, tid_keep]) if len(y_keep) else tid_train.copy()
    is_aug = (
        np.concatenate(
            [
                np.zeros((len(y_train_base),), dtype=bool),
                np.ones((len(y_keep),), dtype=bool),
            ]
        )
        if len(y_keep)
        else np.zeros((len(y_train_base),), dtype=bool)
    )

    metrics, train_meta = _fit_eval_linearsvc(
        X_train,
        y_train,
        tid_train_cond,
        X_test,
        y_test,
        tid_test,
        seed=int(seed),
        cap_k=int(cap_k),
        cap_seed=int(cap_seed),
        cap_sampling_policy=cap_policy,
        linear_c=float(linear_c),
        class_weight=class_weight,
        max_iter=int(max_iter),
        agg_mode=agg_mode,
        is_aug_train=is_aug,
    )
    return metrics, train_meta, gate_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,3,4")
    parser.add_argument("--methods", type=str, default="uniform,w_global,w_class,trunc_class")
    parser.add_argument("--out-root", type=str, default="out/phase15_step1d_risk")
    parser.add_argument("--dataset", type=str, default="seed1", choices=["seed1", "seed", "har", "mitbih", "seediv", "natops", "fingermovements"])
    parser.add_argument("--har-root", type=str, default=DEFAULT_HAR_ROOT)
    parser.add_argument("--mitbih-npz", type=str, default=DEFAULT_MITBIH_NPZ)
    parser.add_argument("--seediv-root", type=str, default=DEFAULT_SEEDIV_ROOT)
    parser.add_argument("--natops-root", type=str, default=DEFAULT_NATOPS_ROOT)
    parser.add_argument("--fingermovements-root", type=str, default=DEFAULT_FINGERMOVEMENTS_ROOT)
    parser.add_argument("--processed-root", type=str, default="data/SEED/SEED_EEG/Preprocessed_EEG")
    parser.add_argument("--stim-xlsx", type=str, default="data/SEED/SEED_EEG/SEED_stimulation.xlsx")

    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--hop-sec", type=float, default=1.0)
    parser.add_argument("--cov-est", type=str, default="sample", choices=["sample", "oas", "ledoitwolf"])
    parser.add_argument("--spd-eps", type=float, default=1e-4)
    parser.add_argument(
        "--bands",
        type=str,
        default=DEFAULT_BANDS_EEG,
    )

    parser.add_argument("--window-cap-k", type=int, default=120)
    parser.add_argument("--cap-sampling-policy", type=str, default="balanced_real_aug")
    parser.add_argument("--aggregation-mode", type=str, default="majority")
    parser.add_argument("--linear-c", type=float, default=1.0)
    parser.add_argument("--linear-class-weight", type=str, default="none")
    parser.add_argument("--linear-max-iter", type=int, default=1000)

    parser.add_argument("--k-dir", type=int, default=5)
    parser.add_argument("--subset-size", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--pia-multiplier", type=int, default=1)
    parser.add_argument("--pia-n-iters", type=int, default=2)
    parser.add_argument("--pia-activation", type=str, default="sine")
    parser.add_argument("--pia-bias-update-mode", type=str, default="residual")
    parser.add_argument("--pia-c-repr", type=float, default=1.0)

    parser.add_argument("--gate1-q", type=float, default=95.0)
    parser.add_argument("--gate2-q-src", type=float, default=90.0)

    parser.add_argument("--risk-alpha", type=float, default=0.7)
    parser.add_argument("--risk-beta", type=float, default=1.0)
    parser.add_argument("--risk-quantile", type=float, default=95.0)
    parser.add_argument("--n-probe", type=int, default=500)
    parser.add_argument("--n-min", type=int, default=30)
    parser.add_argument("--trunc-m", type=int, default=1)
    parser.add_argument("--fallback-inflate", type=float, default=1.1)
    parser.add_argument("--split-preview-n", type=int, default=5)
    args = parser.parse_args()
    args.dataset = normalize_dataset_name(args.dataset)

    seeds = _parse_seed_list(args.seeds)
    methods = _parse_methods(args.methods)
    if args.k_dir != 5 or args.subset_size != 1:
        raise ValueError("Step1D lock requires --k-dir 5 and --subset-size 1.")
    if args.window_cap_k <= 0:
        raise ValueError("--window-cap-k must be > 0.")

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
    bands_spec = resolve_band_spec(args.dataset, args.bands)
    if bands_spec != args.bands:
        print(f"[bands] auto override for {args.dataset}: {bands_spec}")
    bands = parse_band_spec(bands_spec)

    print(
        f"[setup] seeds={seeds} methods={methods} "
        f"kdir={args.k_dir} s={args.subset_size} gamma={args.gamma}"
    )

    all_rows: List[Dict[str, object]] = []
    protocol_issues: List[str] = []

    for seed in seeds:
        print(f"[seed={seed}] start")
        train_trials, test_trials, split_meta = _make_trial_split(all_trials, seed=int(seed))
        print(
            f"[seed={seed}] split train={len(train_trials)} test={len(test_trials)} "
            f"hash={split_meta['split_hash']}"
        )

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
        print(
            f"[seed={seed}] feat train_windows={len(y_train_base)} "
            f"test_windows={len(y_test)} dim={X_train_base.shape[1]}"
        )

        mu_gate1, tau_gate1, gate1_fit_meta = _fit_gate1_from_train(
            X_train=X_train_base,
            y_train=y_train_base,
            q=float(args.gate1_q),
        )

        bank_seed = int(seed * 10000 + args.k_dir * 113 + 17)
        direction_bank, bank_meta = _build_direction_bank_d1(
            X_train=X_train_base,
            k_dir=int(args.k_dir),
            seed=bank_seed,
            n_iters=int(args.pia_n_iters),
            activation=args.pia_activation,
            bias_update_mode=args.pia_bias_update_mode,
            c_repr=float(args.pia_c_repr),
        )

        risk_probe = _probe_direction_risks(
            X_train=X_train_base,
            y_train=y_train_base,
            direction_bank=direction_bank,
            gamma=float(args.gamma),
            alpha=float(args.risk_alpha),
            metric_quantile=float(args.risk_quantile),
            n_probe=int(args.n_probe),
            n_min=int(args.n_min),
            mu_y=mu_gate1,
            tau_y=tau_gate1,
            gate2_q_src=float(args.gate2_q_src),
            seed=int(seed + 9973),
            fallback_inflate=float(args.fallback_inflate),
        )

        probs_global = _probs_from_risk_global(
            risk_global=risk_probe["risk_global"],
            beta=float(args.risk_beta),
            k_dir=int(args.k_dir),
        )
        probs_class = _probs_from_risk_class(
            risk_class=risk_probe["risk_class"],
            beta=float(args.risk_beta),
            k_dir=int(args.k_dir),
        )
        probs_trunc = _probs_from_risk_trunc_class(
            risk_class=risk_probe["risk_class"],
            k_dir=int(args.k_dir),
            trunc_m=int(args.trunc_m),
        )

        cap_seed = int(seed) + 41
        metrics_a, train_meta_a = _fit_eval_linearsvc(
            X_train_base,
            y_train_base,
            tid_train,
            X_test,
            y_test,
            tid_test,
            seed=int(seed),
            cap_k=int(args.window_cap_k),
            cap_seed=cap_seed,
            cap_sampling_policy=args.cap_sampling_policy,
            linear_c=float(args.linear_c),
            class_weight=args.linear_class_weight,
            max_iter=int(args.linear_max_iter),
            agg_mode=args.aggregation_mode,
            is_aug_train=np.zeros((len(y_train_base),), dtype=bool),
        )

        common_meta = {
            "seed": int(seed),
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
            "direction_bank": {
                **bank_meta,
                "subset_size": int(args.subset_size),
            },
            "risk_config": {
                "alpha": float(args.risk_alpha),
                "beta": float(args.risk_beta),
                "metric": f"p{int(args.risk_quantile)}",
                "N_probe": int(args.n_probe),
                "N_min": int(args.n_min),
                "trunc_m": int(args.trunc_m),
                "fallback_inflate": float(args.fallback_inflate),
            },
            "risk_tables": risk_probe,
            "risk_probs": {
                "global": {str(i): float(v) for i, v in enumerate(probs_global.tolist())},
                "class": {
                    str(c): {str(i): float(v) for i, v in enumerate(arr.tolist())}
                    for c, arr in probs_class.items()
                },
                "trunc_class": {
                    str(c): {str(i): float(v) for i, v in enumerate(arr.tolist())}
                    for c, arr in probs_trunc.items()
                },
            },
        }

        seed_dir = os.path.join(args.out_root, f"seed{seed}")
        cond_dirs = {
            "A_baseline": os.path.join(seed_dir, "A_baseline"),
            "C_uniform": os.path.join(seed_dir, "C_uniform"),
            "C_w_global": os.path.join(seed_dir, "C_w_global"),
            "C_w_class": os.path.join(seed_dir, "C_w_class"),
            "C_trunc_class": os.path.join(seed_dir, "C_trunc_class"),
        }
        _write_condition(
            cond_dirs["A_baseline"],
            metrics_a,
            {
                **common_meta,
                **train_meta_a,
                "condition": "A_baseline",
            },
        )

        method_to_cond = {
            "uniform": "C_uniform",
            "w_global": "C_w_global",
            "w_class": "C_w_class",
            "trunc_class": "C_trunc_class",
        }
        cond_results: Dict[str, Dict[str, object]] = {
            "A_baseline": {
                "metrics": metrics_a,
                "gate_meta": {"accept_rate_final": 1.0},
            }
        }

        for method in methods:
            if method == "uniform":
                pg = None
                pc = None
            elif method == "w_global":
                pg = probs_global
                pc = None
            elif method == "w_class":
                pg = None
                pc = probs_class
            elif method == "trunc_class":
                pg = None
                pc = probs_trunc
            else:
                raise RuntimeError(f"Unknown method: {method}")

            X_aug, y_aug, tid_aug, src_aug, aug_meta = _build_aug_candidates_policy(
                X_train=X_train_base,
                y_train=y_train_base,
                tid_train=tid_train,
                direction_bank=direction_bank,
                gamma=float(args.gamma),
                multiplier=int(args.pia_multiplier),
                method=method if method != "trunc_class" else "w_class",
                probs_global=pg,
                probs_class=pc,
                seed=int(seed + 100000 + args.k_dir * 101 + args.subset_size * 7 + len(method)),
            )

            metrics_c, train_meta_c, gate_meta_c = _run_condition(
                X_train_base=X_train_base,
                y_train_base=y_train_base,
                tid_train=tid_train,
                X_test=X_test,
                y_test=y_test,
                tid_test=tid_test,
                X_aug=X_aug,
                y_aug=y_aug,
                tid_aug=tid_aug,
                src_aug=src_aug,
                seed=int(seed),
                cap_k=int(args.window_cap_k),
                cap_seed=cap_seed,
                cap_policy=args.cap_sampling_policy,
                linear_c=float(args.linear_c),
                class_weight=args.linear_class_weight,
                max_iter=int(args.linear_max_iter),
                agg_mode=args.aggregation_mode,
                mu_gate1=mu_gate1,
                tau_gate1=tau_gate1,
                gate2_q_src=float(args.gate2_q_src),
            )

            cond = method_to_cond[method]
            _write_condition(
                cond_dirs[cond],
                metrics_c,
                {
                    **common_meta,
                    **train_meta_c,
                    "condition": cond,
                    "augmentation": aug_meta,
                    "gate1_fit": gate1_fit_meta,
                    "gate_apply": gate_meta_c,
                    "gate2_config": {"enabled": True, "q_src": float(args.gate2_q_src)},
                    "final_accept_rate": float(gate_meta_c["accept_rate_final"]),
                },
            )
            cond_results[cond] = {"metrics": metrics_c, "gate_meta": gate_meta_c}
            print(
                f"[seed={seed}][{cond}] f1={metrics_c['trial_macro_f1']:.4f} "
                f"delta_vs_A={metrics_c['trial_macro_f1']-metrics_a['trial_macro_f1']:+.4f} "
                f"accept={gate_meta_c['accept_rate_final']:.3f}"
            )

        # Seed-level paired csv (wide)
        wide_row = {
            "seed": int(seed),
            "split_hash": split_meta["split_hash"],
            "A_f1": metrics_a["trial_macro_f1"],
            "A_acc": metrics_a["trial_acc"],
        }
        for cond in ["C_uniform", "C_w_global", "C_w_class", "C_trunc_class"]:
            if cond not in cond_results:
                continue
            mc = cond_results[cond]["metrics"]
            wide_row[f"{cond}_f1"] = mc["trial_macro_f1"]
            wide_row[f"{cond}_acc"] = mc["trial_acc"]
            wide_row[f"{cond}_minus_A_f1"] = mc["trial_macro_f1"] - metrics_a["trial_macro_f1"]
            wide_row[f"{cond}_accept"] = cond_results[cond]["gate_meta"]["accept_rate_final"]

        paired_csv = os.path.join(seed_dir, f"phase15_step1d_seed{seed}_paired.csv")
        ensure_dir(os.path.dirname(paired_csv))
        pd.DataFrame([wide_row]).to_csv(paired_csv, index=False)

        for cond in ["C_uniform", "C_w_global", "C_w_class", "C_trunc_class"]:
            if cond not in cond_results:
                continue
            m = cond_results[cond]["metrics"]
            all_rows.append(
                {
                    "seed": int(seed),
                    "split_hash": split_meta["split_hash"],
                    "condition": cond,
                    "A_f1": metrics_a["trial_macro_f1"],
                    "A_acc": metrics_a["trial_acc"],
                    "f1": m["trial_macro_f1"],
                    "acc": m["trial_acc"],
                    "delta_vs_A_f1": m["trial_macro_f1"] - metrics_a["trial_macro_f1"],
                    "delta_vs_A_acc": m["trial_acc"] - metrics_a["trial_acc"],
                    "accept_rate_final": cond_results[cond]["gate_meta"]["accept_rate_final"],
                }
            )

    summary_df = pd.DataFrame(all_rows).sort_values(["condition", "seed"]).reset_index(drop=True)
    summary_csv = os.path.join(args.out_root, "summary", "phase15_step1d_summary_long.csv")
    ensure_dir(os.path.dirname(summary_csv))
    summary_df.to_csv(summary_csv, index=False)

    agg_rows: List[Dict[str, object]] = []
    if not summary_df.empty:
        for cond in sorted(summary_df["condition"].unique().tolist()):
            d = summary_df[summary_df["condition"] == cond]
            for metric in ["f1", "acc", "delta_vs_A_f1", "delta_vs_A_acc", "accept_rate_final"]:
                arr = d[metric].to_numpy(dtype=float)
                agg_rows.append(
                    {
                        "condition": cond,
                        "metric": metric,
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr, ddof=1) if arr.size > 1 else 0.0),
                    }
                )
    agg_df = pd.DataFrame(agg_rows)
    agg_csv = os.path.join(args.out_root, "summary", "phase15_step1d_agg.csv")
    agg_df.to_csv(agg_csv, index=False)

    report = {
        "seeds": seeds,
        "methods": methods,
        "summary_csv": summary_csv,
        "agg_csv": agg_csv,
        "protocol_issues": protocol_issues,
    }
    report_path = os.path.join(args.out_root, "summary", "phase15_step1d_report.json")
    write_json(report_path, report)
    print(f"[done] summary_csv={summary_csv}")
    print(f"[done] agg_csv={agg_csv}")
    print(f"[done] report={report_path}")


if __name__ == "__main__":
    main()
