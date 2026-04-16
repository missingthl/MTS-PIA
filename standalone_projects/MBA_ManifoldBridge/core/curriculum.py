from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple


def _stable_tid_hash(tid: object) -> int:
    return abs(hash(str(tid))) % 1_000_003


def _minmax_norm(x: np.ndarray, *, constant_fill: float = 0.5) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64).ravel()
    if xx.size == 0: return np.asarray([], dtype=np.float64)
    xmin, xmax = float(np.min(xx)), float(np.max(xx))
    if abs(xmax - xmin) <= 1e-12:
        return np.full(xx.shape, float(constant_fill), dtype=np.float64)
    return (xx - xmin) / (xmax - xmin)


def active_direction_probs(gamma_by_dir: np.ndarray, *, freeze_eps: float) -> np.ndarray:
    g = np.asarray(gamma_by_dir, dtype=np.float64).ravel()
    active = g > float(freeze_eps)
    if not np.any(active):
        return np.full(g.shape, 1.0 / float(max(1, len(g))), dtype=np.float64)
    probs = np.zeros_like(g, dtype=np.float64)
    probs[active] = 1.0 / float(np.sum(active))
    return probs


def build_curriculum_aug_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tid_train: np.ndarray,
    *,
    direction_bank: np.ndarray,
    direction_probs: np.ndarray,
    gamma_by_dir: np.ndarray,
    multiplier: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Build augmented latent candidates based on curriculum budgets."""
    tid_arr = np.asarray(tid_train)
    y_arr = np.asarray(y_train).astype(int).ravel()
    k_dir = int(direction_bank.shape[0])
    probs = direction_probs / float(np.sum(direction_probs) + 1e-12)
    gammas = np.asarray(gamma_by_dir, dtype=np.float64).ravel()

    aug_X, aug_y, aug_tid, aug_src, aug_dir, aug_gamma = [], [], [], [], [], []
    tids = sorted(list(set(tid_arr.tolist())))

    for tid in tids:
        idx = np.where(tid_arr == tid)[0]
        X_tid, y_tid = X_train[idx], y_arr[idx]
        
        for m in range(max(0, int(multiplier))):
            rs = np.random.RandomState(int(seed + m * 1009 + _stable_tid_hash(tid)))
            dir_ids = rs.choice(k_dir, size=len(idx), replace=True, p=probs)
            signs = rs.choice([-1.0, 1.0], size=len(idx))
            g_vec = gammas[dir_ids]
            
            X_aug = X_tid + g_vec[:, None] * signs[:, None] * direction_bank[dir_ids]
            
            aug_X.append(X_aug)
            aug_y.append(y_tid)
            aug_tid.append([tid] * len(idx))
            aug_src.append(X_tid)
            aug_dir.append(dir_ids)
            aug_gamma.append(g_vec)

    if not aug_X:
        return (np.empty((0, X_train.shape[1]), dtype=np.float32), 
                np.empty((0,), dtype=np.int64), np.empty((0,), dtype=object),
                np.empty((0, X_train.shape[1]), dtype=np.float32), 
                np.empty((0,), dtype=np.int64), {})

    return (np.vstack(aug_X).astype(np.float32), 
            np.concatenate(aug_y).astype(np.int64), 
            np.concatenate(aug_tid), 
            np.vstack(aug_src).astype(np.float32), 
            np.concatenate(aug_dir).astype(np.int64), 
            {"aug_total_count": len(np.concatenate(aug_y))})


def update_direction_budget(
    *,
    gamma_before: np.ndarray,
    margin_by_dir: Dict[int, float],
    flip_by_dir: Dict[int, float],
    intrusion_by_dir: Dict[int, float],
    expand_factor: float,
    shrink_factor: float,
    gamma_max: float,
    freeze_eps: float,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Update curriculum budget based on direction health metrics."""
    k_dir = len(gamma_before)
    dir_ids = list(range(k_dir))
    margin = np.asarray([float(margin_by_dir.get(i, 0.0)) for i in dir_ids])
    flip = np.asarray([float(flip_by_dir.get(i, 0.0)) for i in dir_ids])
    intrusion = np.asarray([float(intrusion_by_dir.get(i, 0.0)) for i in dir_ids])

    # Safety score: higher is safer
    safety = (
        _minmax_norm(margin) + 
        (1.0 - _minmax_norm(flip)) + 
        (1.0 - _minmax_norm(intrusion))
    ) / 3.0

    gamma_after = gamma_before.copy()
    states = {}
    
    q_expand = np.quantile(safety, 0.75) if safety.size else 0.5
    q_shrink = np.quantile(safety, 0.35) if safety.size else 0.5

    for i in dir_ids:
        g0 = gamma_before[i]
        if g0 <= float(freeze_eps):
            gamma_after[i] = 0.0; states[i] = "freeze"; continue

        if safety[i] >= q_expand:
            gamma_after[i] = min(gamma_max, g0 * expand_factor); states[i] = "expand"
        elif safety[i] <= q_shrink:
            g1 = g0 * shrink_factor
            gamma_after[i] = g1 if g1 > freeze_eps else 0.0
            states[i] = "shrink" if g1 > freeze_eps else "freeze"
        else:
            states[i] = "hold"

    return gamma_after, states
