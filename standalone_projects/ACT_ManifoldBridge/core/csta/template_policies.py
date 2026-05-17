from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from core.curriculum import find_same_class_knn_neighbors

FV_SELECTOR_MODES = {"fv_filter_top5", "fv_score_top5", "random_feasible_selector"}
FV_TOP_K = 5
FV_SAFE_RATIO_TARGET = 0.75


def uses_class_residual_response(direction_meta: Optional[Dict[str, object]]) -> bool:
    return bool(direction_meta and direction_meta.get("response_centering") == "class_residual")


def build_response_class_means(
    X_train_z: np.ndarray,
    y_arr: Optional[np.ndarray],
    direction_meta: Optional[Dict[str, object]],
) -> Optional[Dict[int, np.ndarray]]:
    """Build class means only for direction banks that request residual responses."""
    if not uses_class_residual_response(direction_meta) or y_arr is None:
        return None
    X = np.asarray(X_train_z, dtype=np.float64)
    y = np.asarray(y_arr, dtype=np.int64).ravel()
    return {int(c): np.mean(X[y == c], axis=0) for c in np.unique(y)}


def response_vector_for_anchor(
    *,
    idx: int,
    X_train_z: np.ndarray,
    y_arr: Optional[np.ndarray] = None,
    direction_meta: Optional[Dict[str, object]] = None,
    response_class_means: Optional[Dict[int, np.ndarray]] = None,
) -> np.ndarray:
    """Return the anchor vector used for template-response ranking.

    AO-PIA banks are defined with class-residual responses. TELM2/zPIA and the
    standard controls keep the raw covariance-state anchor.
    """
    z = np.asarray(X_train_z[idx], dtype=np.float64)
    if not uses_class_residual_response(direction_meta) or y_arr is None:
        return z
    means = response_class_means or build_response_class_means(X_train_z, y_arr, direction_meta) or {}
    class_id = int(np.asarray(y_arr, dtype=np.int64).ravel()[idx])
    mean = means.get(class_id)
    return z - mean if mean is not None else z


def template_responses_for_anchor(
    *,
    idx: int,
    X_train_z: np.ndarray,
    zpia_bank: np.ndarray,
    y_arr: Optional[np.ndarray] = None,
    direction_meta: Optional[Dict[str, object]] = None,
    response_class_means: Optional[Dict[int, np.ndarray]] = None,
) -> np.ndarray:
    z_response = response_vector_for_anchor(
        idx=idx,
        X_train_z=X_train_z,
        y_arr=y_arr,
        direction_meta=direction_meta,
        response_class_means=response_class_means,
    )
    return np.abs(np.asarray(z_response, dtype=np.float64) @ np.asarray(zpia_bank, dtype=np.float64).T)


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
    y_arr: Optional[np.ndarray] = None,
    direction_meta: Optional[Dict[str, object]] = None,
    response_class_means: Optional[Dict[int, np.ndarray]] = None,
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
        if uses_class_residual_response(direction_meta) and y_arr is not None:
            means = response_class_means or build_response_class_means(X_train_z, y_arr, direction_meta) or {}
            class_id = int(np.asarray(y_arr, dtype=np.int64).ravel()[idx])
            if class_id in means:
                z_G = z_G - means[class_id]
        responses = np.abs(np.asarray(z_G, dtype=np.float64) @ zpia_bank.T)
        order = np.lexsort((np.arange(k), -responses))
        return order[:pairs]
    if mode == "group_avg_response":
        if neighbor_indices is None:
            raise ValueError("group_avg_response requires prepared neighbor_indices.")
        group_ids = neighbor_indices[idx]
        group_pts = X_train_z[group_ids]
        if uses_class_residual_response(direction_meta) and y_arr is not None:
            means = response_class_means or build_response_class_means(X_train_z, y_arr, direction_meta) or {}
            centered = []
            y_flat = np.asarray(y_arr, dtype=np.int64).ravel()
            for gid, z_val in zip(group_ids, group_pts):
                centered.append(np.asarray(z_val, dtype=np.float64) - means.get(int(y_flat[int(gid)]), 0.0))
            group_pts = np.asarray(centered, dtype=np.float64)
        group_responses = np.abs(np.asarray(group_pts, dtype=np.float64) @ zpia_bank.T)
        mean_responses = np.mean(group_responses, axis=0)
        order = np.lexsort((np.arange(k), -mean_responses))
        return order[:pairs]
    if mode.startswith("topk_softmax_tau_"):
        tau = float(mode.split("_")[-1])
        top_k_num = 5
        responses = template_responses_for_anchor(
            idx=idx,
            X_train_z=X_train_z,
            zpia_bank=zpia_bank,
            y_arr=y_arr,
            direction_meta=direction_meta,
            response_class_means=response_class_means,
        )
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
        responses = template_responses_for_anchor(
            idx=idx,
            X_train_z=X_train_z,
            zpia_bank=zpia_bank,
            y_arr=y_arr,
            direction_meta=direction_meta,
            response_class_means=response_class_means,
        )
        top_indices = np.lexsort((np.arange(k), -responses))[:top_k_num]
        rng = np.random.default_rng(int(idx) + int(seed))
        return rng.choice(top_indices, size=(pairs,), replace=True)
    responses = template_responses_for_anchor(
        idx=idx,
        X_train_z=X_train_z,
        zpia_bank=zpia_bank,
        y_arr=y_arr,
        direction_meta=direction_meta,
        response_class_means=response_class_means,
    )
    order = np.lexsort((np.arange(k), -responses))
    return order[:pairs]
