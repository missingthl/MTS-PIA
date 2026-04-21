from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors


@dataclass
class FisherPIAConfig:
    knn_k: int = 20
    interior_quantile: float = 0.7
    boundary_quantile: float = 0.3
    hetero_k: int = 3
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    eta: float = 1.0
    rho: float = 1e-6


@dataclass(frozen=True)
class LRAESConfig:
    beta: float = 0.5
    reg_lambda: float = 1e-4
    top_k_per_class: int = 3
    rank_tol: float = 1e-8
    eig_pos_eps: float = 1e-9


def _canonicalize_axis(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64).ravel().copy()
    nrm = float(np.linalg.norm(x))
    if nrm <= 1e-12:
        out = np.zeros_like(x); out[0] = 1.0; return out
    x /= nrm
    idx = int(np.argmax(np.abs(x)))
    if x[idx] < 0.0: x *= -1.0
    return x


def _scatter_from_center(x: np.ndarray, center: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64)
    cc = np.asarray(center, dtype=np.float64).ravel()
    d = int(cc.size)
    if xx.size == 0:
        return np.zeros((d, d), dtype=np.float64)
    diff = xx - cc[None, :]
    return (diff.T @ diff) / max(1, int(xx.shape[0]))


def compute_fisher_pia_terms(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    cfg: FisherPIAConfig,
) -> Tuple[Dict[int, Dict[str, object]], Dict[str, object]]:
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train).astype(int).ravel()
    classes = sorted(np.unique(y).tolist())
    
    d = int(X.shape[1])
    k_eff = int(min(max(1, int(cfg.knn_k)), max(1, X.shape[0] - 1)))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(X)
    nn_idx = nn.kneighbors(X, return_distance=False)[:, 1:]
    
    purity = np.mean(y[nn_idx] == y[:, None], axis=1).astype(np.float64)
    mu_by_class = {int(c): np.mean(X[y == c], axis=0).astype(np.float64) for c in classes}

    terms: Dict[int, Dict[str, object]] = {}
    total = float(max(1, X.shape[0]))
    for cls in classes:
        cls_i = int(cls)
        idx = np.where(y == cls_i)[0]
        Xy = X[idx]
        mu_y = mu_by_class[cls_i]
        purity_y = purity[idx]

        q_boundary = float(np.quantile(purity_y, float(cfg.boundary_quantile)))
        q_interior = float(np.quantile(purity_y, float(cfg.interior_quantile)))
        boundary_local = purity_y <= q_boundary
        interior_local = purity_y >= q_interior

        S_W = _scatter_from_center(Xy, mu_y)
        S_expand = _scatter_from_center(Xy[interior_local], mu_y) if np.any(interior_local) else S_W.copy()

        S_B = np.zeros((d, d), dtype=np.float64)
        for other in classes:
            if int(other) == cls_i: continue
            dm = mu_y - mu_by_class[int(other)]
            S_B += np.outer(dm, dm)

        S_risk = np.zeros((d, d), dtype=np.float64)
        risk_vectors = []
        boundary_idx = idx[boundary_local]
        for gi in boundary_idx.tolist():
            nb = nn_idx[gi]
            hetero = nb[y[nb] != cls_i]
            if hetero.size <= 0: continue
            nu_i = np.mean(X[hetero[:int(cfg.hetero_k)]], axis=0)
            dv = X[gi] - nu_i
            S_risk += np.outer(dv, dv)
            risk_vectors.append((nu_i - X[gi]).astype(np.float64))
        
        if risk_vectors: S_risk /= float(len(risk_vectors))

        terms[cls_i] = {
            "mu_y": mu_y,
            "S_expand": S_expand,
            "S_B": S_B,
            "S_W": S_W,
            "S_risk": S_risk,
            "boundary_to_hetero_vectors": np.vstack(risk_vectors) if risk_vectors else np.empty((0, d)),
            "class_weight": float(len(idx) / total),
        }

    meta = {"knn_k": k_eff, "classes": classes}
    return terms, meta


def build_lraes_direction_bank(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    k_dir: int,
    fisher_cfg: FisherPIAConfig,
    lraes_cfg: LRAESConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Advanced Local Region Adaptive Expansion eigensolver."""
    candidates = _collect_lraes_candidates(
        X_train,
        y_train,
        fisher_cfg=fisher_cfg,
        lraes_cfg=lraes_cfg,
    )

    # Sort candidates by global weighted score
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    bank = np.vstack([c["axis"] for c in candidates[:k_dir]]).astype(np.float32)
    return bank, {"bank_source": "lraes", "k_dir": bank.shape[0]}


def _collect_lraes_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    fisher_cfg: FisherPIAConfig,
    lraes_cfg: LRAESConfig,
) -> List[Dict[str, object]]:
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train).astype(int).ravel()
    class_terms, _ = compute_fisher_pia_terms(X, y, cfg=fisher_cfg)
    d = int(X.shape[1])
    I = np.eye(d)

    candidates: List[Dict[str, object]] = []
    for cls in sorted(class_terms.keys()):
        t = class_terms[cls]
        M = (t["S_expand"] + lraes_cfg.reg_lambda * I) - lraes_cfg.beta * (t["S_risk"] + lraes_cfg.reg_lambda * I)
        M = 0.5 * (M + M.T)

        eigvals, eigvecs = np.linalg.eigh(M)
        order = np.argsort(eigvals)[::-1]

        for rank in range(min(lraes_cfg.top_k_per_class, d)):
            idx = order[rank]
            eigval = float(eigvals[idx])
            candidates.append({
                "axis": _canonicalize_axis(eigvecs[:, idx]),
                "eigval": eigval,
                "score": float(t["class_weight"] * eigval),
                "class_id": int(cls),
                "axis_rank": int(rank),
            })
    return candidates


def build_lraes_class_basis_bank(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    k_dir: int,
    fisher_cfg: FisherPIAConfig,
    lraes_cfg: LRAESConfig,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[str, object]]:
    """
    ACL path: retain class-conditional top-k eigendirections instead of flattening
    them into a single global bank.
    """
    candidates = _collect_lraes_candidates(
        X_train,
        y_train,
        fisher_cfg=fisher_cfg,
        lraes_cfg=lraes_cfg,
    )
    y = np.asarray(y_train).astype(int).ravel()
    classes = sorted(np.unique(y).tolist())

    bank: Dict[int, Dict[str, np.ndarray]] = {}
    for cls in classes:
        cls_candidates = [c for c in candidates if int(c["class_id"]) == int(cls)]
        cls_candidates = sorted(cls_candidates, key=lambda x: x["score"], reverse=True)
        kept = cls_candidates[: max(0, int(k_dir))]

        if kept:
            axes = np.vstack([c["axis"] for c in kept]).astype(np.float32)
            eigvals = np.asarray([c["eigval"] for c in kept], dtype=np.float32)
            axis_scores = np.asarray([c["score"] for c in kept], dtype=np.float32)
            axis_ranks = np.asarray([c["axis_rank"] for c in kept], dtype=np.int64)
        else:
            d = int(X_train.shape[1])
            axes = np.empty((0, d), dtype=np.float32)
            eigvals = np.empty((0,), dtype=np.float32)
            axis_scores = np.empty((0,), dtype=np.float32)
            axis_ranks = np.empty((0,), dtype=np.int64)

        bank[int(cls)] = {
            "axes": axes,
            "eigvals": eigvals,
            "axis_scores": axis_scores,
            "class_id": np.asarray([int(cls)], dtype=np.int64),
            "axis_ranks": axis_ranks,
        }

    meta = {
        "bank_source": "lraes_class_basis",
        "k_dir": int(k_dir),
        "classes": classes,
    }
    return bank, meta


def build_pia_direction_bank(
    X_train: np.ndarray,
    k_dir: int = 5,
    seed: int = 42,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Randomized baseline for comparison."""
    rs = np.random.RandomState(seed)
    d = X_train.shape[1]
    W = rs.randn(k_dir, d).astype(np.float32)
    W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
    return W, {"bank_source": "random", "k_dir": k_dir}
