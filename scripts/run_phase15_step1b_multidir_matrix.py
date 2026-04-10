#!/usr/bin/env python
"""Phase 15 Step 1B: Multi-Direction PIA + Gate1+Gate2 matrix runner.

Paired conditions per seed under one locked split:
- A: baseline
- C0: single-direction PIA + Gate1 + Gate2 (q_src fixed)
- Ck: multi-direction PIA + Gate1 + Gate2

Grid:
- k_dir in --k-dir-list
- subset_size in --subset-size-list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIA.telm2 import TELM2Config, TELM2Transformer
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
    write_json,
)
from scripts.run_phase15_step1a_maxplane import (
    PiaAugConfig,
    _apply_gates,
    _build_pia_aug_candidates,
    _fit_eval_linearsvc,
    _fit_gate1_from_train,
    _make_trial_split,
)


@dataclass(frozen=True)
class MultiDirSetting:
    k_dir: int
    subset_size: int

    @property
    def tag(self) -> str:
        return f"kdir{self.k_dir}_s{self.subset_size}"


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


def _parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for tok in str(text).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(int(t))
    out = sorted(set(out))
    if not out:
        raise ValueError("integer list cannot be empty")
    return out


def _parse_seed_list(text: str) -> List[int]:
    return _parse_int_list(text)


def _summary_stats(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"min": 0.0, "mean": 0.0, "std": 0.0, "max": 0.0}
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "max": float(np.max(arr)),
    }


def _gate_keep_mask(
    X_aug: np.ndarray,
    y_aug: np.ndarray,
    src_aug: np.ndarray,
    *,
    mu_y: Dict[int, np.ndarray],
    tau_y: Dict[int, float],
    enable_gate2: bool,
    gate2_q_src: float,
) -> np.ndarray:
    y = np.asarray(y_aug).astype(int).ravel()
    if y.size == 0:
        return np.zeros((0,), dtype=bool)

    keep1 = np.zeros((len(y),), dtype=bool)
    for i, cls in enumerate(y.tolist()):
        muc = mu_y.get(int(cls))
        tauc = tau_y.get(int(cls))
        if muc is None or tauc is None:
            keep1[i] = False
            continue
        keep1[i] = float(np.linalg.norm(X_aug[i] - muc)) <= float(tauc)

    if not enable_gate2:
        return keep1

    d_src = np.linalg.norm(np.asarray(X_aug, dtype=np.float32) - np.asarray(src_aug, dtype=np.float32), axis=1)
    tau_src_y: Dict[int, float] = {}
    for cls in sorted(np.unique(y).tolist()):
        ds = d_src[y == cls]
        if ds.size == 0:
            continue
        tau_src_y[int(cls)] = float(np.percentile(ds, float(gate2_q_src)))

    keep2 = np.asarray(
        [d_src[i] <= tau_src_y.get(int(y[i]), -np.inf) for i in range(len(y))],
        dtype=bool,
    )
    return keep1 & keep2


def _ensure_2d_scores(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim == 1:
        return np.vstack([-s, s]).T
    return s


def _true_class_margin(scores: np.ndarray, y_true: np.ndarray, classes: np.ndarray) -> np.ndarray:
    s = _ensure_2d_scores(scores)
    y = np.asarray(y_true).astype(int).ravel()
    classes_arr = np.asarray(classes).astype(int).ravel()
    class_to_idx = {int(c): i for i, c in enumerate(classes_arr.tolist())}
    n, c = s.shape
    out = np.zeros((n,), dtype=np.float64)
    for i in range(n):
        yi = int(y[i])
        ci = class_to_idx.get(yi)
        if ci is None:
            continue
        s_true = float(s[i, ci])
        if c <= 1:
            out[i] = s_true
            continue
        s_other = float(np.max(np.delete(s[i], ci)))
        out[i] = s_true - s_other
    return out


def _compute_mech_metrics(
    *,
    X_train_real: np.ndarray,
    y_train_real: np.ndarray,
    X_aug_generated: np.ndarray,
    y_aug_generated: np.ndarray,
    X_aug_accepted: np.ndarray,
    y_aug_accepted: np.ndarray,
    X_src_accepted: np.ndarray,
    dir_generated: np.ndarray,
    dir_accepted: np.ndarray,
    seed: int,
    linear_c: float,
    class_weight: Optional[str],
    linear_max_iter: int,
    knn_k: int,
    max_aug_for_mech: int,
    max_real_knn_ref: int,
    max_real_knn_query: int,
    progress_prefix: Optional[str] = None,
) -> Dict[str, object]:
    Xr = np.asarray(X_train_real, dtype=np.float32)
    yr = np.asarray(y_train_real).astype(int).ravel()
    Xg = np.asarray(X_aug_generated, dtype=np.float32)
    yg = np.asarray(y_aug_generated).astype(int).ravel()
    Xa = np.asarray(X_aug_accepted, dtype=np.float32)
    ya = np.asarray(y_aug_accepted).astype(int).ravel()
    Xs = np.asarray(X_src_accepted, dtype=np.float32)
    dir_g = np.asarray(dir_generated).astype(int).ravel()
    dir_a = np.asarray(dir_accepted).astype(int).ravel()

    if not (len(Xg) == len(yg) == len(dir_g)):
        raise ValueError("Mechanism metric input mismatch on generated arrays.")
    if not (len(Xa) == len(ya) == len(Xs) == len(dir_a)):
        raise ValueError("Mechanism metric input mismatch on accepted arrays.")

    cw = None if class_weight in {None, "", "none"} else class_weight
    if progress_prefix:
        print(
            f"{progress_prefix} start real={len(yr)} aug_gen={len(yg)} aug_acc={len(ya)}",
            flush=True,
        )
    scaler = StandardScaler()
    if progress_prefix:
        print(f"{progress_prefix} scale_fit_start", flush=True)
    Xr_s = scaler.fit_transform(Xr)
    if progress_prefix:
        print(f"{progress_prefix} scale_fit_done", flush=True)
    clf_ref = LinearSVC(
        C=float(linear_c),
        class_weight=cw,
        max_iter=int(linear_max_iter),
        random_state=int(seed),
        dual="auto",
    )
    if progress_prefix:
        print(f"{progress_prefix} clf_ref_fit_start", flush=True)
    clf_ref.fit(Xr_s, yr)
    if progress_prefix:
        print(f"{progress_prefix} clf_ref_fit_done", flush=True)

    n_acc = int(len(ya))
    rng_seed = int(seed + 8093)
    rs = np.random.RandomState(rng_seed)
    if n_acc <= 0:
        eval_idx = np.asarray([], dtype=np.int64)
    elif n_acc > int(max_aug_for_mech):
        eval_idx = np.sort(rs.choice(n_acc, size=int(max_aug_for_mech), replace=False))
    else:
        eval_idx = np.arange(n_acc, dtype=np.int64)

    Xa_eval = Xa[eval_idx] if eval_idx.size else np.empty((0, Xr.shape[1]), dtype=np.float32)
    Xs_eval = Xs[eval_idx] if eval_idx.size else np.empty((0, Xr.shape[1]), dtype=np.float32)
    ya_eval = ya[eval_idx] if eval_idx.size else np.empty((0,), dtype=np.int64)
    dir_eval = dir_a[eval_idx] if eval_idx.size else np.empty((0,), dtype=np.int64)

    if eval_idx.size:
        if progress_prefix:
            print(f"{progress_prefix} margin_eval_start eval={len(eval_idx)}", flush=True)
        src_scores = clf_ref.decision_function(scaler.transform(Xs_eval))
        aug_scores = clf_ref.decision_function(scaler.transform(Xa_eval))
        src_margin = _true_class_margin(src_scores, ya_eval, clf_ref.classes_)
        aug_margin = _true_class_margin(aug_scores, ya_eval, clf_ref.classes_)
        flip = (src_margin >= 0.0) != (aug_margin >= 0.0)
        margin_delta = aug_margin - src_margin
        flip_rate = float(np.mean(flip))
        margin_drop_median = float(np.median(margin_delta))
    else:
        flip = np.asarray([], dtype=bool)
        margin_delta = np.asarray([], dtype=np.float64)
        flip_rate = 0.0
        margin_drop_median = 0.0

    n_real = int(len(yr))
    if int(max_real_knn_ref) > 0 and n_real > int(max_real_knn_ref):
        ref_idx = np.sort(rs.choice(n_real, size=int(max_real_knn_ref), replace=False))
    else:
        ref_idx = np.arange(n_real, dtype=np.int64)
    Xr_knn = Xr[ref_idx]
    yr_knn = yr[ref_idx]

    k_eff = int(min(max(1, int(knn_k)), max(1, len(yr_knn))))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    if progress_prefix:
        print(
            f"{progress_prefix} knn_fit_start ref={len(yr_knn)} k={k_eff}",
            flush=True,
        )
    nn.fit(Xr_knn)
    if progress_prefix:
        print(f"{progress_prefix} knn_fit_done", flush=True)

    if eval_idx.size and len(yr_knn) > 0:
        if progress_prefix:
            print(f"{progress_prefix} knn_aug_query_start", flush=True)
        nn_idx = nn.kneighbors(Xa_eval, return_distance=False)
        y_nb = yr_knn[nn_idx]
        purity_each = np.mean(y_nb == ya_eval[:, None], axis=1).astype(np.float64)
        intrusion_rate = float(np.mean(1.0 - purity_each))
    else:
        intrusion_rate = 0.0
    purity = float(1.0 - intrusion_rate)

    if len(yr_knn) <= 1:
        real_intrusion_rate = 0.0
        real_query_idx_local = np.asarray([], dtype=np.int64)
    else:
        n_ref = int(len(yr_knn))
        if int(max_real_knn_query) > 0 and n_ref > int(max_real_knn_query):
            real_query_idx_local = np.sort(rs.choice(n_ref, size=int(max_real_knn_query), replace=False))
        else:
            real_query_idx_local = np.arange(n_ref, dtype=np.int64)

        X_real_q = Xr_knn[real_query_idx_local]
        y_real_q = yr_knn[real_query_idx_local]
        k_self = int(min(n_ref, k_eff + 1))
        if progress_prefix:
            print(
                f"{progress_prefix} knn_real_query_start query={len(real_query_idx_local)}",
                flush=True,
            )
        nn_idx_real = nn.kneighbors(X_real_q, n_neighbors=k_self, return_distance=False)
        if k_self >= 2:
            nn_idx_real_use = nn_idx_real[:, 1 : 1 + k_eff]
            y_real_nb = yr_knn[nn_idx_real_use]
            purity_real = np.mean(y_real_nb == y_real_q[:, None], axis=1).astype(np.float64)
            real_intrusion_rate = float(np.mean(1.0 - purity_real))
        else:
            real_intrusion_rate = 0.0

    real_purity = float(1.0 - real_intrusion_rate)
    delta_intrusion = float(intrusion_rate - real_intrusion_rate)
    if progress_prefix:
        print(f"{progress_prefix} done", flush=True)

    max_dir = -1
    if dir_g.size:
        max_dir = max(max_dir, int(np.max(dir_g)))
    if dir_a.size:
        max_dir = max(max_dir, int(np.max(dir_a)))
    if dir_eval.size:
        max_dir = max(max_dir, int(np.max(dir_eval)))

    dir_profile: Dict[str, Dict[str, object]] = {}
    total_gen = int(len(dir_g))
    for i in range(max_dir + 1):
        n_gen_i = int(np.sum(dir_g == i))
        n_acc_i = int(np.sum(dir_a == i))
        usage = float(n_gen_i / total_gen) if total_gen > 0 else 0.0
        accept_rate_i = float(n_acc_i / n_gen_i) if n_gen_i > 0 else 0.0
        eval_mask_i = dir_eval == i
        if np.any(eval_mask_i):
            flip_rate_i = float(np.mean(flip[eval_mask_i]))
            margin_drop_i = float(np.median(margin_delta[eval_mask_i]))
            n_flip_eval = int(np.sum(eval_mask_i))
        else:
            flip_rate_i = 0.0
            margin_drop_i = 0.0
            n_flip_eval = 0
        dir_profile[str(i)] = {
            "usage": usage,
            "accept_rate": accept_rate_i,
            "flip_rate": flip_rate_i,
            "margin_drop_median": margin_drop_i,
            "n_gen": n_gen_i,
            "n_acc": n_acc_i,
            "n_flip_eval": n_flip_eval,
        }

    return {
        "flip_rate": float(flip_rate),
        "margin_drop_median": float(margin_drop_median),
        "knn_intrusion_rate_k20": float(intrusion_rate) if int(knn_k) == 20 else None,
        "knn_purity_k20": float(purity) if int(knn_k) == 20 else None,
        "knn_intrusion_rate": float(intrusion_rate),
        "knn_purity": float(purity),
        "real_knn_intrusion_rate_k20": float(real_intrusion_rate) if int(knn_k) == 20 else None,
        "real_knn_purity_k20": float(real_purity) if int(knn_k) == 20 else None,
        "real_knn_intrusion_rate": float(real_intrusion_rate),
        "real_knn_purity": float(real_purity),
        "delta_intrusion_k20": float(delta_intrusion) if int(knn_k) == 20 else None,
        "delta_intrusion": float(delta_intrusion),
        "knn_k": int(k_eff),
        "dir_profile": dir_profile,
        "mech_sample_sizes": {
            "n_train_real_windows": int(len(yr)),
            "n_real_knn_ref": int(len(yr_knn)),
            "n_real_knn_query": int(len(real_query_idx_local)),
            "n_aug_generated": int(len(yg)),
            "n_aug_accepted": int(len(ya)),
            "n_aug_used_for_mech": int(len(eval_idx)),
        },
        "mech_rng_seed": int(rng_seed),
        "flip_margin_definition": "true_class_margin=(score_y-max_other); flip=sign_change(margin)",
        "knn_space": "z_raw",
    }


def _build_direction_bank_d1(
    X_train: np.ndarray,
    *,
    k_dir: int,
    seed: int,
    n_iters: int,
    activation: str,
    bias_update_mode: str,
    c_repr: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Build deterministic direction bank from TELM2 template rows (D1)."""
    cfg = TELM2Config(
        r_dimension=int(k_dir),
        n_iters=int(n_iters),
        activation=activation,
        bias_update_mode=bias_update_mode,
        C_repr=float(c_repr),
        enable_repr_learning=True,
        seed=int(seed),
    )
    telm = TELM2Transformer(cfg).fit(np.asarray(X_train, dtype=np.float64))
    arts = telm.get_artifacts()
    W = np.asarray(arts.W, dtype=np.float64)  # [k_dir, d]
    w_mean = np.mean(W, axis=0, keepdims=True)
    Wc = W - w_mean

    Wn = np.zeros_like(Wc, dtype=np.float64)
    for i in range(Wc.shape[0]):
        vec = Wc[i]
        nrm = float(np.linalg.norm(vec))
        if not np.isfinite(nrm) or nrm <= 1e-12:
            vec = W[i]
            nrm = float(np.linalg.norm(vec))
        if not np.isfinite(nrm) or nrm <= 1e-12:
            vec = np.zeros_like(vec)
            vec[0] = 1.0
            nrm = 1.0
        Wn[i] = vec / nrm

    row_norms = np.linalg.norm(Wn, axis=1)
    recon = np.asarray(getattr(arts, "recon_err", []), dtype=np.float64)
    meta = {
        "bank_source": "D1_telm2_templates_centered",
        "k_dir": int(k_dir),
        "seed": int(seed),
        "n_iters": int(n_iters),
        "activation": activation,
        "bias_update_mode": bias_update_mode,
        "c_repr": float(c_repr),
        "direction_norm_stats": _summary_stats(row_norms),
        "recon_last": float(recon[-1]) if recon.size else 0.0,
        "recon_mean": float(np.mean(recon)) if recon.size else 0.0,
        "recon_std": float(np.std(recon)) if recon.size else 0.0,
    }
    return np.asarray(Wn, dtype=np.float32), meta


def _sample_subset_indices(
    rs: np.random.RandomState,
    n_rows: int,
    *,
    k_dir: int,
    subset_size: int,
) -> np.ndarray:
    s = int(min(max(1, subset_size), k_dir))
    if s == 1:
        return rs.randint(0, k_dir, size=(n_rows, 1), dtype=np.int64)
    if s == 2:
        first = rs.randint(0, k_dir, size=(n_rows,), dtype=np.int64)
        second = rs.randint(0, k_dir - 1, size=(n_rows,), dtype=np.int64)
        second = np.where(second >= first, second + 1, second)
        return np.stack([first, second], axis=1).astype(np.int64)
    sel = np.empty((n_rows, s), dtype=np.int64)
    for i in range(n_rows):
        sel[i] = rs.choice(k_dir, size=s, replace=False)
    return sel


def _build_multidir_aug_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tid_train: np.ndarray,
    *,
    direction_bank: np.ndarray,
    subset_size: int,
    gamma: float,
    multiplier: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    tid_arr = np.asarray(tid_train)
    y_arr = np.asarray(y_train).astype(int).ravel()
    k_dir = int(direction_bank.shape[0])
    trial_ids = sorted(_ordered_unique(tid_arr.tolist()))

    aug_X_parts: List[np.ndarray] = []
    aug_y_parts: List[np.ndarray] = []
    aug_tid_parts: List[np.ndarray] = []
    aug_src_parts: List[np.ndarray] = []
    aug_dir_parts: List[np.ndarray] = []
    aug_count_per_trial: Dict[str, int] = {}
    dir_pick_count = np.zeros((k_dir,), dtype=np.int64)

    abs_ai_sum = 0.0
    ai_count = 0
    subset_size_sum = 0.0
    subset_obs = 0

    for tid in trial_ids:
        idx = np.where(tid_arr == tid)[0]
        if idx.size == 0:
            aug_count_per_trial[tid] = 0
            continue

        X_tid = np.asarray(X_train[idx], dtype=np.float32)
        y_tid = np.asarray(y_arr[idx], dtype=np.int64)
        added = 0
        for m in range(max(0, int(multiplier))):
            rs = np.random.RandomState(int(seed + m * 1009 + _stable_tid_hash(tid)))
            sel = _sample_subset_indices(
                rs,
                int(X_tid.shape[0]),
                k_dir=k_dir,
                subset_size=int(subset_size),
            )  # [N, s]
            s_eff = int(sel.shape[1])
            a = rs.normal(loc=0.0, scale=1.0, size=(X_tid.shape[0], s_eff)).astype(np.float32)
            a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)

            W_sel = direction_bank[sel]  # [N, s, d]
            w_mix = np.sum(a[:, :, None] * W_sel, axis=1).astype(np.float32)  # [N, d]
            X_aug = (X_tid + float(gamma) * w_mix).astype(np.float32)

            aug_X_parts.append(X_aug)
            aug_y_parts.append(y_tid.copy())
            aug_tid_parts.append(np.asarray([tid] * len(idx)))
            aug_src_parts.append(X_tid.copy())
            aug_dir_parts.append(sel[:, 0].astype(np.int64).copy())  # primary direction id
            added += len(idx)

            binc = np.bincount(sel.reshape(-1), minlength=k_dir)
            dir_pick_count += binc.astype(np.int64)
            abs_ai_sum += float(np.sum(np.abs(a)))
            ai_count += int(a.size)
            subset_size_sum += float(s_eff * X_tid.shape[0])
            subset_obs += int(X_tid.shape[0])

        aug_count_per_trial[tid] = int(added)

    if aug_X_parts:
        X_aug_all = np.vstack(aug_X_parts).astype(np.float32)
        y_aug_all = np.concatenate(aug_y_parts).astype(np.int64)
        tid_aug_all = np.concatenate(aug_tid_parts)
        src_aug_all = np.vstack(aug_src_parts).astype(np.float32)
        dir_aug_all = np.concatenate(aug_dir_parts).astype(np.int64)
    else:
        X_aug_all = np.empty((0, X_train.shape[1]), dtype=np.float32)
        y_aug_all = np.empty((0,), dtype=np.int64)
        tid_aug_all = np.empty((0,), dtype=object)
        src_aug_all = np.empty((0, X_train.shape[1]), dtype=np.float32)
        dir_aug_all = np.empty((0,), dtype=np.int64)

    aug_vals = (
        np.asarray(list(aug_count_per_trial.values()), dtype=np.float64)
        if aug_count_per_trial
        else np.asarray([], dtype=np.float64)
    )
    total_picks = int(np.sum(dir_pick_count))
    pick_frac = (
        {str(i): float(c / total_picks) for i, c in enumerate(dir_pick_count.tolist())}
        if total_picks > 0
        else {str(i): 0.0 for i in range(k_dir)}
    )

    meta = {
        "aug_total_count": int(len(y_aug_all)),
        "aug_count_per_trial": aug_count_per_trial,
        "aug_per_trial_mean": float(np.mean(aug_vals)) if aug_vals.size else 0.0,
        "aug_per_trial_std": float(np.std(aug_vals)) if aug_vals.size else 0.0,
        "gamma": float(gamma),
        "k_dir": int(k_dir),
        "subset_size": int(subset_size),
        "mixing_stats": {
            "mean_abs_ai": float(abs_ai_sum / max(1, ai_count)),
            "avg_subset_size": float(subset_size_sum / max(1, subset_obs)),
            "direction_pick_fraction": pick_frac,
        },
    }
    return X_aug_all, y_aug_all, tid_aug_all, src_aug_all, dir_aug_all, meta


def _write_condition(cond_dir: str, metrics: Dict, run_meta: Dict) -> None:
    ensure_dir(cond_dir)
    write_json(os.path.join(cond_dir, "metrics.json"), metrics)
    write_json(os.path.join(cond_dir, "run_meta.json"), run_meta)
    mech = run_meta.get("mech", {})
    if isinstance(mech, dict):
        dir_profile = mech.get("dir_profile", {})
        if isinstance(dir_profile, dict) and dir_profile:
            rows: List[Dict[str, object]] = []
            for k, v in sorted(dir_profile.items(), key=lambda kv: int(kv[0])):
                row = {"direction_id": int(k)}
                if isinstance(v, dict):
                    row.update(v)
                rows.append(row)
            pd.DataFrame(rows).to_csv(os.path.join(cond_dir, "mech_table.csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,4")
    parser.add_argument("--k-dir-list", type=str, default="3,5,8")
    parser.add_argument("--subset-size-list", type=str, default="1,2")
    parser.add_argument("--out-root", type=str, default="out/phase15_step1b_multidir")
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

    parser.add_argument("--pia-multiplier", type=int, default=1)
    parser.add_argument("--pia-gamma", type=float, default=0.10)
    parser.add_argument("--pia-gamma-jitter", type=float, default=0.0)
    parser.add_argument("--pia-n-iters", type=int, default=2)
    parser.add_argument("--pia-activation", type=str, default="sine")
    parser.add_argument("--pia-bias-update-mode", type=str, default="residual")
    parser.add_argument("--pia-c-repr", type=float, default=1.0)

    parser.add_argument("--gate1-q", type=float, default=95.0)
    parser.add_argument("--gate2-q-src", type=float, default=90.0)
    parser.add_argument("--mech-knn-k", type=int, default=20)
    parser.add_argument("--mech-max-aug-for-metrics", type=int, default=20000)
    parser.add_argument("--mech-max-real-knn-ref", type=int, default=30000)
    parser.add_argument("--mech-max-real-knn-query", type=int, default=3000)
    parser.add_argument("--split-preview-n", type=int, default=5)
    args = parser.parse_args()
    args.dataset = normalize_dataset_name(args.dataset)

    seeds = _parse_seed_list(args.seeds)
    k_dirs = _parse_int_list(args.k_dir_list)
    subset_sizes = _parse_int_list(args.subset_size_list)
    settings = [MultiDirSetting(k_dir=k, subset_size=s) for k in k_dirs for s in subset_sizes]

    if args.window_cap_k <= 0:
        raise ValueError("--window-cap-k must be > 0 for Step1B lock.")
    if not (0 < float(args.gate1_q) <= 100):
        raise ValueError("--gate1-q must be in (0,100].")
    if not (0 < float(args.gate2_q_src) <= 100):
        raise ValueError("--gate2-q-src must be in (0,100].")
    if int(args.mech_knn_k) <= 0:
        raise ValueError("--mech-knn-k must be > 0.")
    if int(args.mech_max_aug_for_metrics) <= 0:
        raise ValueError("--mech-max-aug-for-metrics must be > 0.")
    if int(args.mech_max_real_knn_ref) <= 0:
        raise ValueError("--mech-max-real-knn-ref must be > 0.")
    if int(args.mech_max_real_knn_query) <= 0:
        raise ValueError("--mech-max-real-knn-query must be > 0.")

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
        f"[setup] seeds={seeds} settings={[st.tag for st in settings]} "
        f"gate1_q={args.gate1_q} gate2_q_src={args.gate2_q_src}"
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

        pia_cfg = PiaAugConfig(
            multiplier=int(args.pia_multiplier),
            gamma=float(args.pia_gamma),
            gamma_jitter=float(args.pia_gamma_jitter),
            n_iters=int(args.pia_n_iters),
            activation=args.pia_activation,
            bias_update_mode=args.pia_bias_update_mode,
            C_repr=float(args.pia_c_repr),
            seed=int(seed),
        )
        X_pia, y_pia, tid_pia, src_pia, pia_meta = _build_pia_aug_candidates(
            X_train=X_train_base,
            y_train=y_train_base,
            tid_train=tid_train,
            cfg=pia_cfg,
        )
        X_c0_keep, y_c0_keep, tid_c0_keep, src_c0_keep, c0_gate_meta = _apply_gates(
            X_aug=X_pia,
            y_aug=y_pia,
            tid_aug=tid_pia,
            src_aug=src_pia,
            mu_y=mu_gate1,
            tau_y=tau_gate1,
            enable_gate2=True,
            gate2_q_src=float(args.gate2_q_src),
        )
        keep_c0 = _gate_keep_mask(
            X_pia,
            y_pia,
            src_pia,
            mu_y=mu_gate1,
            tau_y=tau_gate1,
            enable_gate2=True,
            gate2_q_src=float(args.gate2_q_src),
        )
        if int(np.sum(keep_c0)) != int(len(y_c0_keep)):
            raise RuntimeError(
                f"C0 gate keep mismatch: mask={int(np.sum(keep_c0))} vs apply={int(len(y_c0_keep))}"
            )
        dir_c0_gen = np.zeros((len(y_pia),), dtype=np.int64)
        dir_c0_keep = dir_c0_gen[keep_c0]
        mech_c0 = _compute_mech_metrics(
            X_train_real=X_train_base,
            y_train_real=y_train_base,
            X_aug_generated=X_pia,
            y_aug_generated=y_pia,
            X_aug_accepted=X_c0_keep,
            y_aug_accepted=y_c0_keep,
            X_src_accepted=src_c0_keep,
            dir_generated=dir_c0_gen,
            dir_accepted=dir_c0_keep,
            seed=int(seed),
            linear_c=float(args.linear_c),
            class_weight=args.linear_class_weight,
            linear_max_iter=int(args.linear_max_iter),
            knn_k=int(args.mech_knn_k),
            max_aug_for_mech=int(args.mech_max_aug_for_metrics),
            max_real_knn_ref=int(args.mech_max_real_knn_ref),
            max_real_knn_query=int(args.mech_max_real_knn_query),
        )
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
            seed=int(seed),
            cap_k=int(args.window_cap_k),
            cap_seed=cap_seed,
            cap_sampling_policy=args.cap_sampling_policy,
            linear_c=float(args.linear_c),
            class_weight=args.linear_class_weight,
            max_iter=int(args.linear_max_iter),
            agg_mode=args.aggregation_mode,
            is_aug_train=is_aug_c0,
        )
        print(f"[seed={seed}] C0 candidates={len(y_pia)} post_gate={len(y_c0_keep)}")

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
        }

        bank_cache: Dict[int, Tuple[np.ndarray, Dict[str, object]]] = {}

        for st in settings:
            if st.k_dir not in bank_cache:
                bank_seed = int(seed * 10000 + st.k_dir * 113 + 17)
                bank_cache[st.k_dir] = _build_direction_bank_d1(
                    X_train=X_train_base,
                    k_dir=int(st.k_dir),
                    seed=bank_seed,
                    n_iters=int(args.pia_n_iters),
                    activation=args.pia_activation,
                    bias_update_mode=args.pia_bias_update_mode,
                    c_repr=float(args.pia_c_repr),
                )

            direction_bank, bank_meta = bank_cache[st.k_dir]
            X_ck, y_ck, tid_ck, src_ck, dir_ck, ck_aug_meta = _build_multidir_aug_candidates(
                X_train=X_train_base,
                y_train=y_train_base,
                tid_train=tid_train,
                direction_bank=direction_bank,
                subset_size=int(st.subset_size),
                gamma=float(args.pia_gamma),
                multiplier=int(args.pia_multiplier),
                seed=int(seed + 100000 + st.k_dir * 101 + st.subset_size * 7),
            )
            X_ck_keep, y_ck_keep, tid_ck_keep, src_ck_keep, ck_gate_meta = _apply_gates(
                X_aug=X_ck,
                y_aug=y_ck,
                tid_aug=tid_ck,
                src_aug=src_ck,
                mu_y=mu_gate1,
                tau_y=tau_gate1,
                enable_gate2=True,
                gate2_q_src=float(args.gate2_q_src),
            )
            keep_ck = _gate_keep_mask(
                X_ck,
                y_ck,
                src_ck,
                mu_y=mu_gate1,
                tau_y=tau_gate1,
                enable_gate2=True,
                gate2_q_src=float(args.gate2_q_src),
            )
            if int(np.sum(keep_ck)) != int(len(y_ck_keep)):
                raise RuntimeError(
                    f"Ck gate keep mismatch: mask={int(np.sum(keep_ck))} vs apply={int(len(y_ck_keep))}"
                )
            dir_ck_keep = np.asarray(dir_ck, dtype=np.int64)[keep_ck]
            mech_ck = _compute_mech_metrics(
                X_train_real=X_train_base,
                y_train_real=y_train_base,
                X_aug_generated=X_ck,
                y_aug_generated=y_ck,
                X_aug_accepted=X_ck_keep,
                y_aug_accepted=y_ck_keep,
                X_src_accepted=src_ck_keep,
                dir_generated=dir_ck,
                dir_accepted=dir_ck_keep,
                seed=int(seed),
                linear_c=float(args.linear_c),
                class_weight=args.linear_class_weight,
                linear_max_iter=int(args.linear_max_iter),
                knn_k=int(args.mech_knn_k),
                max_aug_for_mech=int(args.mech_max_aug_for_metrics),
                max_real_knn_ref=int(args.mech_max_real_knn_ref),
                max_real_knn_query=int(args.mech_max_real_knn_query),
            )

            X_train_ck = np.vstack([X_train_base, X_ck_keep]) if len(y_ck_keep) else X_train_base.copy()
            y_train_ck = np.concatenate([y_train_base, y_ck_keep]) if len(y_ck_keep) else y_train_base.copy()
            tid_train_ck = np.concatenate([tid_train, tid_ck_keep]) if len(y_ck_keep) else tid_train.copy()
            is_aug_ck = (
                np.concatenate(
                    [
                        np.zeros((len(y_train_base),), dtype=bool),
                        np.ones((len(y_ck_keep),), dtype=bool),
                    ]
                )
                if len(y_ck_keep)
                else np.zeros((len(y_train_base),), dtype=bool)
            )
            metrics_ck, train_meta_ck = _fit_eval_linearsvc(
                X_train_ck,
                y_train_ck,
                tid_train_ck,
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
                is_aug_train=is_aug_ck,
            )

            setting_dir = os.path.join(args.out_root, st.tag, f"seed{seed}")
            cond_dirs = {
                "A_baseline": os.path.join(setting_dir, "A_baseline"),
                "C0_pia_gate": os.path.join(setting_dir, "C0_pia_gate"),
                "Ck_multidir_gate": os.path.join(setting_dir, "Ck_multidir_gate"),
            }

            _write_condition(
                cond_dirs["A_baseline"],
                metrics_a,
                {
                    **common_meta,
                    **train_meta_a,
                    "condition": "A_baseline",
                    "setting": st.tag,
                },
            )
            _write_condition(
                cond_dirs["C0_pia_gate"],
                metrics_c0,
                {
                    **common_meta,
                    **train_meta_c0,
                    "condition": "C0_pia_gate",
                    "setting": st.tag,
                    "augmentation": pia_meta,
                    "gate1_fit": gate1_fit_meta,
                    "gate_apply": c0_gate_meta,
                    "gate2_config": {"enabled": True, "q_src": float(args.gate2_q_src)},
                    "final_accept_rate": float(c0_gate_meta["accept_rate_final"]),
                    "mech": mech_c0,
                },
            )
            _write_condition(
                cond_dirs["Ck_multidir_gate"],
                metrics_ck,
                {
                    **common_meta,
                    **train_meta_ck,
                    "condition": "Ck_multidir_gate",
                    "setting": st.tag,
                    "augmentation": ck_aug_meta,
                    "direction_bank": {
                        **bank_meta,
                        "subset_size": int(st.subset_size),
                    },
                    "mixing_stats": ck_aug_meta.get("mixing_stats", {}),
                    "gate1_fit": gate1_fit_meta,
                    "gate_apply": ck_gate_meta,
                    "gate2_config": {"enabled": True, "q_src": float(args.gate2_q_src)},
                    "final_accept_rate": float(ck_gate_meta["accept_rate_final"]),
                    "mech": mech_ck,
                },
            )

            paired_rows = [
                {
                    "condition": "A_baseline",
                    "acc": metrics_a["trial_acc"],
                    "macro_f1": metrics_a["trial_macro_f1"],
                    "accept_rate": 1.0,
                    "split_hash": split_meta["split_hash"],
                    "setting": st.tag,
                },
                {
                    "condition": "C0_pia_gate",
                    "acc": metrics_c0["trial_acc"],
                    "macro_f1": metrics_c0["trial_macro_f1"],
                    "accept_rate": c0_gate_meta["accept_rate_final"],
                    "split_hash": split_meta["split_hash"],
                    "setting": st.tag,
                },
                {
                    "condition": "Ck_multidir_gate",
                    "acc": metrics_ck["trial_acc"],
                    "macro_f1": metrics_ck["trial_macro_f1"],
                    "accept_rate": ck_gate_meta["accept_rate_final"],
                    "split_hash": split_meta["split_hash"],
                    "setting": st.tag,
                },
            ]
            paired_csv = os.path.join(args.out_root, st.tag, f"phase15_step1b_seed{seed}_paired.csv")
            ensure_dir(os.path.dirname(paired_csv))
            pd.DataFrame(paired_rows).to_csv(paired_csv, index=False)

            all_rows.append(
                {
                    "seed": int(seed),
                    "setting": st.tag,
                    "k_dir": int(st.k_dir),
                    "subset_size": int(st.subset_size),
                    "split_hash": split_meta["split_hash"],
                    "A_f1": metrics_a["trial_macro_f1"],
                    "C0_f1": metrics_c0["trial_macro_f1"],
                    "Ck_f1": metrics_ck["trial_macro_f1"],
                    "A_acc": metrics_a["trial_acc"],
                    "C0_acc": metrics_c0["trial_acc"],
                    "Ck_acc": metrics_ck["trial_acc"],
                    "delta_Ck_minus_C0_f1": metrics_ck["trial_macro_f1"] - metrics_c0["trial_macro_f1"],
                    "delta_Ck_minus_A_f1": metrics_ck["trial_macro_f1"] - metrics_a["trial_macro_f1"],
                    "delta_Ck_minus_C0_acc": metrics_ck["trial_acc"] - metrics_c0["trial_acc"],
                    "delta_Ck_minus_A_acc": metrics_ck["trial_acc"] - metrics_a["trial_acc"],
                    "Ck_accept_rate_final": ck_gate_meta["accept_rate_final"],
                    "Ck_accept_rate_gate1": ck_gate_meta["accept_rate_gate1"],
                }
            )
            print(
                f"[seed={seed}][{st.tag}] A={metrics_a['trial_macro_f1']:.4f} "
                f"C0={metrics_c0['trial_macro_f1']:.4f} Ck={metrics_ck['trial_macro_f1']:.4f} "
                f"Ck-C0={metrics_ck['trial_macro_f1']-metrics_c0['trial_macro_f1']:+.4f} "
                f"accept={ck_gate_meta['accept_rate_final']:.3f}"
            )

    summary_df = pd.DataFrame(all_rows).sort_values(["setting", "seed"]).reset_index(drop=True)
    summary_csv = os.path.join(args.out_root, "summary", "phase15_step1b_summary.csv")
    ensure_dir(os.path.dirname(summary_csv))
    summary_df.to_csv(summary_csv, index=False)

    agg_rows: List[Dict[str, object]] = []
    if not summary_df.empty:
        for setting in sorted(summary_df["setting"].unique().tolist()):
            d = summary_df[summary_df["setting"] == setting]
            for metric in [
                "A_f1",
                "C0_f1",
                "Ck_f1",
                "A_acc",
                "C0_acc",
                "Ck_acc",
                "delta_Ck_minus_C0_f1",
                "delta_Ck_minus_A_f1",
                "delta_Ck_minus_C0_acc",
                "delta_Ck_minus_A_acc",
                "Ck_accept_rate_final",
            ]:
                arr = d[metric].to_numpy(dtype=float)
                agg_rows.append(
                    {
                        "setting": setting,
                        "metric": metric,
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr, ddof=1) if arr.size > 1 else 0.0),
                    }
                )
    agg_df = pd.DataFrame(agg_rows)
    agg_csv = os.path.join(args.out_root, "summary", "phase15_step1b_agg.csv")
    agg_df.to_csv(agg_csv, index=False)

    report = {
        "seeds": seeds,
        "settings": [st.tag for st in settings],
        "summary_csv": summary_csv,
        "agg_csv": agg_csv,
        "protocol_issues": protocol_issues,
    }
    report_path = os.path.join(args.out_root, "summary", "phase15_step1b_report.json")
    write_json(report_path, report)
    print(f"[done] summary_csv={summary_csv}")
    print(f"[done] agg_csv={agg_csv}")
    print(f"[done] report={report_path}")


if __name__ == "__main__":
    main()
