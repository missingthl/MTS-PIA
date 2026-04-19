from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple
from sklearn.neighbors import KDTree


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


def estimate_local_manifold_margins(
    X_train_z: np.ndarray, 
    y_train: np.ndarray
) -> np.ndarray:
    """
    Efficiently estimates the Local Manifold Margin for all samples.
    Calculates the distance to the nearest neighbor of a DIFFERENT class 
    using a KDTree for O(N log N) performance.
    """
    y_arr = np.asarray(y_train).astype(int).ravel()
    classes = np.unique(y_arr)
    margins = np.zeros(len(y_arr), dtype=np.float64)
    
    # Pre-build trees for each class to find nearest-neighbor of DIFFERENT class
    class_trees = {c: KDTree(X_train_z[y_arr == c]) for c in classes}
    
    for c in classes:
        idx_c = np.where(y_arr == c)[0]
        pts_c = X_train_z[idx_c]
        
        # Min distance to ANY other class
        min_dists = np.full(len(idx_c), np.inf)
        for other_c in classes:
            if c == other_c: continue
            dist, _ = class_trees[other_c].query(pts_c, k=1)
            min_dists = np.minimum(min_dists, dist.ravel())
        
        margins[idx_c] = min_dists
        
    return margins


def apply_safe_step_constraint(
    gamma: float, 
    direction_norm: float, 
    d_min: float, 
    eta: float = 0.5
) -> Tuple[float, float]:
    """
    Prop 2: Safe Region enforcement.
    Clips gamma to ensure the geometric perturbation doesn't overwhelm the 
    Local Manifold Margin, maintaining a healthy Signal-to-Noise ratio 
    between perturbation strength and decision boundary abundance.
    """
    # Constraint: gamma * ||u|| < eta * d_min
    max_step = eta * d_min / (direction_norm + 1e-12)
    safe_gamma = min(float(gamma), float(max_step))
    ratio = safe_gamma / (gamma + 1e-12)
    return safe_gamma, ratio


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
    eta_safe: float = 0.5,  # Safety coefficient
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Enhanced Augmentation with Proposition 2: Safe Region enforcement.
    """
    tid_arr = np.asarray(tid_train)
    y_arr = np.asarray(y_train).astype(int).ravel()
    X_z_all = X_train  # Assuming X_train here is the latent Z if we are in latent space
    
    actual_k = int(direction_bank.shape[0])
    if actual_k == 0:
        return (X_train.astype(np.float32), y_train.astype(np.int64), tid_train,
                X_train.astype(np.float32), np.zeros(len(y_train), dtype=np.int64),
                {"aug_total_count": 0, "status": "fallback_no_bank"})

    p_vec = np.asarray(direction_probs).ravel()
    # (Handling probs and gammas remains similar to original for robustness)
    if p_vec.size != actual_k:
        if p_vec.size < actual_k:
            p_vec = np.pad(p_vec, (0, actual_k - p_vec.size), mode='constant', constant_values=0.0)
        else:
            p_vec = p_vec[:actual_k]
    
    p_sum = np.sum(p_vec)
    probs = p_vec / p_sum if p_sum > 1e-12 else np.full((actual_k,), 1.0 / actual_k)
    
    gammas = np.asarray(gamma_by_dir, dtype=np.float64).ravel()
    if gammas.size != actual_k:
        if gammas.size < actual_k:
            gammas = np.pad(gammas, (0, actual_k - gammas.size), mode='edge')
        else:
            gammas = gammas[:actual_k]

    # Pre-calculate margins for the entire training set (O(N log N))
    X_z_all = X_train
    all_margins = estimate_local_manifold_margins(X_z_all, y_arr)
    
    # Map TIDs to global indices for distance lookup
    tid_to_idx = {tid: i for i, tid in enumerate(tid_arr)}

    aug_X, aug_y, aug_tid, aug_src, aug_dir, aug_gamma = [], [], [], [], [], []
    tids = sorted(list(set(tid_arr.tolist())))
    
    safe_ratios = []
    margins_diagnostic = []

    for tid in tids:
        idx_global = tid_to_idx[tid]
        X_sample, y_sample = X_z_all[idx_global], y_arr[idx_global]
        d_min = all_margins[idx_global]
        margins_diagnostic.append(d_min)
        
        rs = np.random.RandomState(int(seed + idx_global * 1009 + _stable_tid_hash(tid)))
        
        for m in range(max(0, int(multiplier))):
            dir_id = rs.choice(actual_k, p=probs)
            sign = rs.choice([-1.0, 1.0])
            g0 = gammas[dir_id]
            u_k = direction_bank[dir_id]
            u_norm = np.linalg.norm(u_k)
            
            # Apply Proposition 2: SafeStep Constraint (Manifold Margin Buffer)
            g_safe, ratio = apply_safe_step_constraint(g0, u_norm, d_min, eta=eta_safe)
            safe_ratios.append(ratio)
            
            x_aug = X_sample + g_safe * sign * u_k
            
            aug_X.append(x_aug[None, :])
            aug_y.append(y_sample)
            aug_tid.append(tid)
            aug_src.append(X_sample[None, :])
            aug_dir.append(dir_id)
            aug_gamma.append(g_safe)

    if not aug_X:
        return (np.empty((0, X_train.shape[1]), dtype=np.float32), 
                np.empty((0,), dtype=np.int64), np.empty((0,), dtype=object),
                np.empty((0, X_train.shape[1]), dtype=np.float32), 
                np.empty((0,), dtype=np.int64), {})

    meta = {
        "aug_total_count": len(aug_y),
        "safe_radius_ratio_mean": float(np.mean(safe_ratios)),
        "safe_radius_ratio_min": float(np.min(safe_ratios)),
        "manifold_margin_mean": float(np.mean(margins_diagnostic))
    }

    return (np.vstack(aug_X).astype(np.float32), 
            np.array(aug_y).astype(np.int64), 
            np.array(aug_tid), 
            np.vstack(aug_src).astype(np.float32), 
            np.array(aug_dir).astype(np.int64), 
            meta)


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
