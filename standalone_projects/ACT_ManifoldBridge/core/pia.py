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


def build_zpia_direction_bank(
    X_train_z: np.ndarray,
    k_dir: int = 10,
    seed: int = 42,
    *,
    telm2_n_iters: int = 3,
    telm2_c_repr: float = 1.0,
    telm2_activation: str = "sine",
    telm2_bias_update_mode: str = "residual",
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Build native Log-Euclidean template directions with TELM2.

    ``zpia`` keeps the original MBA realization path intact. TELM2 only learns
    template axes in the train-only Log-Euclidean ``z`` cloud; every returned
    row is L2-normalized so ``pia_gamma`` remains comparable to LRAES.
    """
    try:
        from PIA.telm2 import TELM2Config, TELM2Transformer
    except ModuleNotFoundError:
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from PIA.telm2 import TELM2Config, TELM2Transformer

    X = np.asarray(X_train_z, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X_train_z must be a 2D matrix of Log-Euclidean states.")
    if X.shape[0] <= 0 or X.shape[1] <= 0:
        raise ValueError("X_train_z must have at least one row and one feature.")
    if not np.isfinite(X).all():
        raise ValueError("X_train_z contains NaN or inf values.")

    k = int(k_dir)
    if k <= 0:
        raise ValueError("k_dir must be positive for zpia.")

    cfg = TELM2Config(
        r_dimension=k,
        n_iters=int(telm2_n_iters),
        C_repr=float(telm2_c_repr),
        activation=str(telm2_activation),
        orthogonalize=True,
        enable_repr_learning=True,
        bias_update_mode=str(telm2_bias_update_mode),
        seed=int(seed),
    )
    transformer = TELM2Transformer(cfg).fit(X)
    artifacts = transformer.get_artifacts()
    W_raw = np.asarray(artifacts.W, dtype=np.float64)
    expected_shape = (k, int(X.shape[1]))
    if W_raw.shape != expected_shape:
        raise ValueError(f"TELM2 W shape mismatch: expected {expected_shape}, got {W_raw.shape}.")
    if not np.isfinite(W_raw).all():
        raise ValueError("TELM2 W contains NaN or inf values.")

    W_centered = W_raw - np.mean(W_raw, axis=0, keepdims=True)
    bank = np.zeros_like(W_centered, dtype=np.float64)
    fallback_row_count = 0
    for row_idx in range(k):
        row = np.asarray(W_centered[row_idx], dtype=np.float64).copy()
        norm = float(np.linalg.norm(row))
        if (not np.isfinite(norm)) or norm < 1e-12:
            row = np.asarray(W_raw[row_idx], dtype=np.float64).copy()
            norm = float(np.linalg.norm(row))
            fallback_row_count += 1
        if (not np.isfinite(norm)) or norm < 1e-12:
            row = np.zeros((X.shape[1],), dtype=np.float64)
            row[row_idx % X.shape[1]] = 1.0
            norm = 1.0
        row = row / (norm + 1e-12)
        bank[row_idx] = _canonicalize_axis(row)

    row_norms = np.linalg.norm(bank, axis=1)
    if (not np.isfinite(bank).all()) or (not np.isfinite(row_norms).all()):
        raise ValueError("zpia direction bank contains NaN or inf values after normalization.")
    if not np.allclose(row_norms, 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("zpia direction rows are not L2-normalized.")

    recon = np.asarray(list(artifacts.recon_err), dtype=np.float64)
    recon_finite = recon[np.isfinite(recon)]
    if recon_finite.size == 0:
        recon_last = recon_mean = recon_std = 0.0
    else:
        recon_last = float(recon_finite[-1])
        recon_mean = float(np.mean(recon_finite))
        recon_std = float(np.std(recon_finite))

    meta: Dict[str, object] = {
        "bank_source": "zpia_telm2",
        "k_dir": int(k),
        "z_dim": int(X.shape[1]),
        "n_train": int(X.shape[0]),
        "n_train_lt_z_dim": bool(int(X.shape[0]) < int(X.shape[1])),
        "row_norm_min": float(np.min(row_norms)),
        "row_norm_max": float(np.max(row_norms)),
        "row_norm_mean": float(np.mean(row_norms)),
        "fallback_row_count": int(fallback_row_count),
        "telm2_recon_last": recon_last,
        "telm2_recon_mean": recon_mean,
        "telm2_recon_std": recon_std,
        "telm2_n_iters": int(telm2_n_iters),
        "telm2_c_repr": float(telm2_c_repr),
        "telm2_activation": str(telm2_activation),
        "telm2_bias_update_mode": str(telm2_bias_update_mode),
    }
    return bank.astype(np.float32), meta
