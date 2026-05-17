from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from core.curriculum import estimate_local_manifold_margins

from .materialize import materialize_z_aug_out
from .state import TrialRecord


AG_PIA_METHODS = {"ag_target_direct", "ag_pia_single", "ag_pia_multihead5"}


@dataclass(frozen=True)
class AGPIAConfig:
    k_pos: int = 5
    k_neg: int = 5
    lambda_tangent: float = 0.5
    lambda_inter: float = 0.25
    hidden_dim: int = 0
    ridge: float = 1e-3
    activation: str = "tanh"
    eps: float = 1e-12


def _unit(v: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
    v = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm <= eps:
        return np.zeros_like(v, dtype=np.float64), norm
    return (v / norm).astype(np.float64), norm


def _effective_rank(X: np.ndarray, eps: float = 1e-12) -> float:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 3:
        X = X.reshape(-1, X.shape[-1])
    if X.size == 0 or X.shape[0] <= 1:
        return 0.0
    s = np.linalg.svd(X - np.mean(X, axis=0, keepdims=True), compute_uv=False)
    energy = s * s
    total = float(np.sum(energy))
    if total <= eps:
        return 0.0
    p = energy / total
    return float(np.exp(-np.sum(p * np.log(p + eps))))


def _pairwise_cosine_mean(X: np.ndarray, eps: float = 1e-12) -> float:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 3:
        X = X.reshape(-1, X.shape[-1])
    if X.shape[0] <= 1:
        return np.nan
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, eps)
    sims = Xn @ Xn.T
    iu = np.triu_indices(X.shape[0], k=1)
    return float(np.mean(sims[iu])) if len(iu[0]) else np.nan


def _local_tangent_project(
    *,
    Z: np.ndarray,
    neighbor_ids: np.ndarray,
    vector: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, bool]:
    if neighbor_ids.size < 2:
        return np.asarray(vector, dtype=np.float64), False
    pts = np.asarray(Z[neighbor_ids], dtype=np.float64)
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    if float(np.linalg.norm(centered)) <= eps:
        return np.asarray(vector, dtype=np.float64), False
    try:
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.asarray(vector, dtype=np.float64), False
    energy = s * s
    if float(np.sum(energy)) <= eps:
        return np.asarray(vector, dtype=np.float64), False
    cumulative = np.cumsum(energy) / max(float(np.sum(energy)), eps)
    dim = int(np.searchsorted(cumulative, 0.90) + 1)
    dim = max(1, min(dim, vt.shape[0], 10))
    U = vt[:dim].T
    projected = U @ (U.T @ np.asarray(vector, dtype=np.float64))
    return projected.astype(np.float64), True


def build_ag_pia_targets(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    heads: int,
    seed: int,
    cfg: AGPIAConfig | None = None,
) -> Dict[str, object]:
    """Construct train-only AG-PIA augmentation target directions."""
    cfg = cfg or AGPIAConfig()
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).ravel()
    n, z_dim = Z.shape
    rng = np.random.default_rng(int(seed) + 1729)
    targets = np.zeros((int(heads), n, z_dim), dtype=np.float64)
    row_meta: List[List[Dict[str, object]]] = [[{} for _ in range(n)] for _ in range(int(heads))]

    for i in range(n):
        dists = np.linalg.norm(Z - Z[i], axis=1)
        same = np.where(y == y[i])[0]
        same = same[same != i]
        diff = np.where(y != y[i])[0]
        pos_order = same[np.argsort(dists[same])] if same.size else same
        neg_order = diff[np.argsort(dists[diff])] if diff.size else diff
        pos_ids = pos_order[: int(cfg.k_pos)]
        neg_ids = neg_order[: int(cfg.k_neg)]
        neg_centroid = np.mean(Z[neg_ids], axis=0) if neg_ids.size else Z[i]
        inter_vec = neg_centroid - Z[i]
        neg_dist = float(np.linalg.norm(inter_vec))

        for h in range(int(heads)):
            fallback_flag = False
            fallback_reason = ""
            if pos_order.size > h:
                pos_vec = Z[pos_order[h]] - Z[i]
            elif pos_ids.size:
                pos_vec = np.mean(Z[pos_ids], axis=0) - Z[i]
                fallback_flag = True
                fallback_reason = "insufficient_pos_neighbors"
            else:
                pos_vec = np.zeros((z_dim,), dtype=np.float64)
                fallback_flag = True
                fallback_reason = "no_same_class_neighbor"

            tan_vec, tangent_available = _local_tangent_project(
                Z=Z,
                neighbor_ids=pos_ids,
                vector=pos_vec,
                eps=float(cfg.eps),
            )
            g_raw = (
                (1.0 - float(cfg.lambda_tangent)) * pos_vec
                + float(cfg.lambda_tangent) * tan_vec
                - float(cfg.lambda_inter) * inter_vec
            )
            g, g_norm = _unit(g_raw, eps=float(cfg.eps))
            if g_norm <= float(cfg.eps):
                g, g_norm = _unit(pos_vec, eps=float(cfg.eps))
                fallback_flag = True
                fallback_reason = fallback_reason or "zero_ag_field_use_pos"
            if g_norm <= float(cfg.eps):
                rand = rng.normal(size=(z_dim,))
                g, g_norm = _unit(rand, eps=float(cfg.eps))
                fallback_flag = True
                fallback_reason = "zero_ag_field_use_seeded_random"

            targets[h, i] = g
            row_meta[h][i] = {
                "ag_target_norm": float(g_norm),
                "ag_tangent_available": bool(tangent_available),
                "ag_pos_neighbor_count": int(pos_ids.size),
                "ag_neg_neighbor_count": int(neg_ids.size),
                "ag_pos_dist": float(np.linalg.norm(pos_vec)),
                "ag_neg_centroid_dist": float(neg_dist),
                "ag_fallback_flag": bool(fallback_flag),
                "ag_fallback_reason": str(fallback_reason),
            }

    target_norms = np.asarray(
        [float(row_meta[h][i]["ag_target_norm"]) for h in range(int(heads)) for i in range(n)],
        dtype=np.float64,
    )
    return {
        "targets": targets.astype(np.float64),
        "row_meta": row_meta,
        "summary": {
            "ag_target_effective_rank": _effective_rank(targets),
            "ag_target_pairwise_cosine_mean": _pairwise_cosine_mean(targets),
            "ag_target_norm_mean": float(np.mean(target_norms)) if target_norms.size else np.nan,
            "ag_target_norm_std": float(np.std(target_norms)) if target_norms.size else np.nan,
            "ag_tangent_available_rate": float(
                np.mean([float(row_meta[h][i]["ag_tangent_available"]) for h in range(int(heads)) for i in range(n)])
            )
            if n and heads
            else 0.0,
            "ag_fallback_rate": float(
                np.mean([float(row_meta[h][i]["ag_fallback_flag"]) for h in range(int(heads)) for i in range(n)])
            )
            if n and heads
            else 0.0,
            "ag_pos_dist_mean": float(
                np.mean([float(row_meta[h][i]["ag_pos_dist"]) for h in range(int(heads)) for i in range(n)])
            )
            if n and heads
            else np.nan,
            "ag_neg_centroid_dist_mean": float(
                np.mean([float(row_meta[h][i]["ag_neg_centroid_dist"]) for h in range(int(heads)) for i in range(n)])
            )
            if n and heads
            else np.nan,
        },
    }


def _activation(X: np.ndarray, name: str) -> np.ndarray:
    if str(name) == "tanh":
        return np.tanh(X)
    if str(name) == "sigmoid":
        return 1.0 / (1.0 + np.exp(-X))
    return X


def fit_ag_pia_operator(
    Z: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    cfg: AGPIAConfig | None = None,
) -> Dict[str, object]:
    """Fit random-feature ridge operators from Z to target directions."""
    cfg = cfg or AGPIAConfig()
    Z = np.asarray(Z, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    heads, n, z_dim = targets.shape
    hidden_dim = int(cfg.hidden_dim) if int(cfg.hidden_dim) > 0 else max(64, 2 * int(z_dim))
    rng = np.random.default_rng(int(seed) + 9103)
    W = rng.normal(scale=1.0 / np.sqrt(max(float(z_dim), 1.0)), size=(hidden_dim, z_dim))
    b = rng.normal(scale=0.1, size=(hidden_dim,))
    H = _activation(Z @ W.T + b[None, :], str(cfg.activation))
    lhs = H.T @ H + float(cfg.ridge) * np.eye(hidden_dim, dtype=np.float64)
    betas = np.zeros((heads, hidden_dim, z_dim), dtype=np.float64)
    preds = np.zeros((heads, n, z_dim), dtype=np.float64)
    mses: List[float] = []
    cosines: List[float] = []
    for h in range(heads):
        rhs = H.T @ targets[h]
        beta = np.linalg.solve(lhs, rhs)
        pred = H @ beta
        pred_unit = np.asarray([_unit(v, eps=float(cfg.eps))[0] for v in pred], dtype=np.float64)
        betas[h] = beta
        preds[h] = pred_unit
        mses.append(float(np.mean((pred_unit - targets[h]) ** 2)))
        cos = np.sum(pred_unit * targets[h], axis=1) / (
            np.maximum(np.linalg.norm(pred_unit, axis=1), cfg.eps) * np.maximum(np.linalg.norm(targets[h], axis=1), cfg.eps)
        )
        cosines.append(float(np.mean(cos)))
    return {
        "W": W,
        "b": b,
        "betas": betas,
        "pred_dirs": preds,
        "summary": {
            "ag_operator_train_mse_mean": float(np.mean(mses)) if mses else np.nan,
            "ag_operator_train_cosine_mean": float(np.mean(cosines)) if cosines else np.nan,
            "ag_hidden_dim": int(hidden_dim),
            "ag_ridge": float(cfg.ridge),
            "ag_activation": str(cfg.activation),
        },
    }


def _head_diagnostics(pred_dirs: np.ndarray, head_ids: List[int]) -> Dict[str, object]:
    pred_dirs = np.asarray(pred_dirs, dtype=np.float64)
    heads = pred_dirs.shape[0]
    head_means = np.mean(pred_dirs, axis=1) if pred_dirs.size else np.empty((0, 0))
    counts = np.bincount(np.asarray(head_ids, dtype=np.int64), minlength=heads).astype(np.float64) if head_ids else np.zeros((heads,))
    probs = counts / max(float(counts.sum()), 1.0)
    return {
        "ag_head_pairwise_cosine_mean": _pairwise_cosine_mean(head_means),
        "ag_head_effective_rank": _effective_rank(head_means),
        "ag_head_usage_entropy": float(-np.sum(probs * np.log(probs + 1e-12))) if probs.size else np.nan,
    }


def build_ag_pia_aug_out(
    *,
    args,
    seed: int,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    method: str,
) -> Dict[str, object]:
    """Build AG-PIA candidates and realize them through the existing bridge."""
    if method not in AG_PIA_METHODS:
        raise ValueError(f"Unknown AG-PIA method: {method}")
    if str(getattr(args, "template_selection", "top_response")) != "top_response":
        raise ValueError("AG-PIA is a direction operator and must not use template-selection modes.")

    heads = 5 if method == "ag_pia_multihead5" else 1
    cfg = AGPIAConfig(
        k_pos=int(getattr(args, "ag_k_pos", 5)),
        k_neg=int(getattr(args, "ag_k_neg", 5)),
        lambda_tangent=float(getattr(args, "ag_lambda_tangent", 0.5)),
        lambda_inter=float(getattr(args, "ag_lambda_inter", 0.25)),
        hidden_dim=int(getattr(args, "ag_hidden_dim", 0)),
        ridge=float(getattr(args, "ag_ridge", 1e-3)),
        activation=str(getattr(args, "ag_activation", "tanh")),
    )
    target_out = build_ag_pia_targets(X_train_z, y_train, heads=heads, seed=seed, cfg=cfg)
    targets = np.asarray(target_out["targets"], dtype=np.float64)
    row_meta = target_out["row_meta"]
    if method == "ag_target_direct":
        pred_dirs = targets.copy()
        operator_summary = {
            "ag_operator_train_mse_mean": 0.0,
            "ag_operator_train_cosine_mean": 1.0,
            "ag_hidden_dim": 0,
            "ag_ridge": float(cfg.ridge),
            "ag_activation": "direct_target",
        }
    else:
        op = fit_ag_pia_operator(X_train_z, targets, seed=seed, cfg=cfg)
        pred_dirs = np.asarray(op["pred_dirs"], dtype=np.float64)
        operator_summary = dict(op["summary"])

    y_arr = np.asarray(y_train, dtype=np.int64).ravel()
    tid_arr = np.asarray([record.tid for record in train_recs], dtype=object)
    margins = estimate_local_manifold_margins(np.asarray(X_train_z, dtype=np.float64), y_arr)
    eta_safe = None if bool(getattr(args, "disable_safe_step", False)) else float(getattr(args, "eta_safe", 0.75))
    gamma_requested = float(getattr(args, "pia_gamma", 0.1))
    multiplier = int(getattr(args, "multiplier", 10))
    rng = np.random.default_rng(int(seed) + 32011)
    z_aug: List[np.ndarray] = []
    y_aug: List[int] = []
    tid_aug: List[object] = []
    rows: List[Dict[str, object]] = []
    head_ids_used: List[int] = []
    eps = float(cfg.eps)

    for i in range(len(X_train_z)):
        for c in range(multiplier):
            head_id = 0 if heads == 1 else int(rng.integers(0, heads))
            direction, pred_norm = _unit(pred_dirs[head_id, i], eps=eps)
            target = targets[head_id, i]
            target_norm = float(row_meta[head_id][i].get("ag_target_norm", np.linalg.norm(target)))
            pred_target_cos = float(np.dot(direction, target) / max(float(np.linalg.norm(target)), eps))
            direction_norm = float(np.linalg.norm(direction))
            d_min = float(margins[i])
            if eta_safe is None:
                gamma_used = gamma_requested
                safe_upper_bound = float("inf")
                safe_radius_ratio = 1.0
            else:
                safe_upper_bound = float(eta_safe) * d_min / (direction_norm + eps)
                gamma_used = min(gamma_requested, safe_upper_bound)
                safe_radius = float(eta_safe) * d_min
                safe_radius_ratio = float(abs(gamma_used) * direction_norm / (safe_radius + eps)) if safe_radius > 0 else 0.0
            W_i = (gamma_used * direction).astype(np.float32)
            z_aug.append((np.asarray(X_train_z[i], dtype=np.float64) + W_i).astype(np.float32))
            y_aug.append(int(y_arr[i]))
            tid_aug.append(tid_arr[i])
            head_ids_used.append(head_id)
            meta_i = dict(row_meta[head_id][i])
            rows.append(
                {
                    "anchor_index": int(i),
                    "tid": tid_arr[i],
                    "class_id": int(y_arr[i]),
                    "candidate_order": int(c),
                    "slot_index": int(len(rows)),
                    "direction_source": "ag_pia_operator",
                    "template_id": -1,
                    "template_rank": -1,
                    "template_sign": np.nan,
                    "template_response_abs": np.nan,
                    "selected_template_rank": -1,
                    "selected_template_response_abs": np.nan,
                    "direction_id": -1,
                    "ag_head_id": int(head_id),
                    "ag_target_norm": float(target_norm),
                    "ag_pred_norm": float(pred_norm),
                    "ag_pred_target_cosine": pred_target_cos,
                    **meta_i,
                    "gamma_requested": float(gamma_requested),
                    "gamma_used": float(gamma_used),
                    "gamma_used_ratio": float(gamma_used / gamma_requested) if abs(gamma_requested) > eps else np.nan,
                    "direction_norm": direction_norm,
                    "pre_safe_displacement_norm": float(abs(gamma_requested) * direction_norm),
                    "post_safe_displacement_norm": float(np.linalg.norm(W_i)),
                    "z_displacement_norm": float(np.linalg.norm(W_i)),
                    "safe_upper_bound": float(safe_upper_bound),
                    "safe_radius_ratio": float(safe_radius_ratio),
                    "manifold_margin": d_min,
                    "is_clipped": float(gamma_requested > safe_upper_bound + 1e-9),
                    "selection_stage": "ag_pia_operator",
                    "selector_name": method,
                    "feasible_flag": 1.0,
                    "selector_accept_flag": 1.0,
                }
            )

    flat_preds = pred_dirs.reshape(-1, pred_dirs.shape[-1])
    ag_summary = {
        **dict(target_out["summary"]),
        **operator_summary,
        **_head_diagnostics(pred_dirs, head_ids_used),
        "ag_pred_target_cosine_mean": float(np.mean([r["ag_pred_target_cosine"] for r in rows])) if rows else np.nan,
        "ag_fallback_rate": float(np.mean([float(r.get("ag_fallback_flag", False)) for r in rows])) if rows else 0.0,
        "ag_pred_norm_mean": float(np.mean(np.linalg.norm(flat_preds, axis=1))) if flat_preds.size else np.nan,
    }
    direction_meta = {
        "bank_source": "ag_pia_operator",
        "direction_source": "ag_pia_operator",
        "operator_source": method,
        "ag_heads": int(heads),
        **ag_summary,
    }
    return materialize_z_aug_out(
        z_aug=np.stack(z_aug).astype(np.float32) if z_aug else np.empty((0, X_train_z.shape[1]), dtype=np.float32),
        y_aug=np.asarray(y_aug, dtype=np.int64),
        tid_aug=np.asarray(tid_aug, dtype=object),
        audit_rows=rows,
        train_recs=train_recs,
        mean_log=mean_log,
        direction_bank_meta=direction_meta,
        effective_k=int(heads),
        eta_safe=eta_safe,
        algo_name=method,
        engine_id=method,
        extra_meta={
            **ag_summary,
            "selection_stage": "ag_pia_operator",
            "selector_name": method,
            "effective_k_ag_heads": int(heads),
            "multi_template_pairs": 0,
            "template_usage_entropy": 0.0,
            "top_template_concentration": 0.0,
        },
    )
