from __future__ import annotations

from typing import Optional

import numpy as np

from core.curriculum import find_same_class_knn_neighbors

FV_SELECTOR_MODES = {"fv_filter_top5", "fv_score_top5", "random_feasible_selector"}
FV_TOP_K = 5
FV_SAFE_RATIO_TARGET = 0.75


def prepare_template_neighbor_indices(
    *,
    mode: str,
    X_train_z: np.ndarray,
    y_arr: np.ndarray,
    group_size: int,
    seed: int,
) -> Optional[np.ndarray]:
    if not (mode.startswith("group_") or mode == "sameclass_zmix"):
        return None
    if mode == "group_top_random_sameclass":
        neighbor_indices = np.zeros((len(y_arr), group_size), dtype=np.int64)
        for c in np.unique(y_arr):
            idx_c = np.where(y_arr == c)[0]
            for idx_global in idx_c:
                rng = np.random.default_rng(int(idx_global) + int(seed))
                neighbor_indices[idx_global] = rng.choice(idx_c, size=group_size, replace=True)
        return neighbor_indices
    return find_same_class_knn_neighbors(X_train_z, y_arr, k=group_size)


def select_template_ids_for_policy(
    *,
    mode: str,
    idx: int,
    pairs: int,
    X_train_z: np.ndarray,
    zpia_bank: np.ndarray,
    seed: int,
    neighbor_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    k = zpia_bank.shape[0]
    if mode == "random":
        rng = np.random.default_rng(int(idx) + int(seed))
        return rng.choice(np.arange(k), size=(pairs,), replace=False)
    if mode == "fixed":
        return np.arange(pairs) % k
    if mode == "group_random":
        if neighbor_indices is None:
            raise ValueError("group_random requires prepared neighbor_indices.")
        group_ids = neighbor_indices[idx]
        group_seed = int(np.sum(group_ids)) + int(seed)
        rng = np.random.default_rng(group_seed)
        return rng.choice(np.arange(k), size=(pairs,), replace=False)
    if mode == "group_top" or mode == "group_top_random_sameclass":
        if neighbor_indices is None:
            raise ValueError(f"{mode} requires prepared neighbor_indices.")
        group_ids = neighbor_indices[idx]
        z_G = np.mean(X_train_z[group_ids], axis=0)
        responses = np.abs(np.asarray(z_G, dtype=np.float64) @ zpia_bank.T)
        order = np.lexsort((np.arange(k), -responses))
        return order[:pairs]
    if mode == "group_avg_response":
        if neighbor_indices is None:
            raise ValueError("group_avg_response requires prepared neighbor_indices.")
        group_ids = neighbor_indices[idx]
        group_pts = X_train_z[group_ids]
        group_responses = np.abs(np.asarray(group_pts, dtype=np.float64) @ zpia_bank.T)
        mean_responses = np.mean(group_responses, axis=0)
        order = np.lexsort((np.arange(k), -mean_responses))
        return order[:pairs]
    if mode.startswith("topk_softmax_tau_"):
        tau = float(mode.split("_")[-1])
        top_k_num = 5
        responses = np.abs(np.asarray(X_train_z[idx], dtype=np.float64) @ zpia_bank.T)
        top_indices = np.lexsort((np.arange(k), -responses))[:top_k_num]
        top_responses = responses[top_indices]
        logits = top_responses / max(float(tau), 1e-12)
        logits = logits - float(np.max(logits))
        exp_r = np.exp(logits)
        probs = exp_r / np.sum(exp_r)
        rng = np.random.default_rng(int(idx) + int(seed))
        return rng.choice(top_indices, size=(pairs,), replace=True, p=probs)
    if mode.startswith("topk_uniform_top"):
        top_k_num = int(mode.split("top")[-1])
        responses = np.abs(np.asarray(X_train_z[idx], dtype=np.float64) @ zpia_bank.T)
        top_indices = np.lexsort((np.arange(k), -responses))[:top_k_num]
        rng = np.random.default_rng(int(idx) + int(seed))
        return rng.choice(top_indices, size=(pairs,), replace=True)
    responses = np.abs(np.asarray(X_train_z[idx], dtype=np.float64) @ zpia_bank.T)
    order = np.lexsort((np.arange(k), -responses))
    return order[:pairs]
