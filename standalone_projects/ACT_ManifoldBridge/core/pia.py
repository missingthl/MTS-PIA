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


def build_pca_direction_bank(
    X_train_z: np.ndarray,
    k_dir: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Build template directions using PCA principal components."""
    from sklearn.decomposition import PCA
    X = np.asarray(X_train_z, dtype=np.float64)
    pca = PCA(n_components=min(k_dir, X.shape[0], X.shape[1]), random_state=seed)
    pca.fit(X)
    W = pca.components_
    if W.shape[0] < k_dir:
        # Pad with random if not enough components
        rng = np.random.default_rng(seed)
        extra = rng.standard_normal((k_dir - W.shape[0], X.shape[1]))
        W = np.concatenate([W, extra], axis=0)
    
    # Normalize
    W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
    # Canonicalize
    output = np.zeros_like(W)
    for i in range(len(W)):
        output[i] = _canonicalize_axis(W[i])
    
    return output, {"bank_source": "pca", "k_dir": k_dir}


def build_random_orthogonal_direction_bank(
    X_train_z: np.ndarray,
    k_dir: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Build template directions using a random orthogonal basis."""
    rng = np.random.default_rng(seed)
    d = X_train_z.shape[1]
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    W = Q[:k_dir]
    if W.shape[0] < k_dir:
        # If d < k_dir, we repeat or pad
        extra = rng.standard_normal((k_dir - W.shape[0], d))
        W = np.concatenate([W, extra], axis=0)
    
    # Normalize
    W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
    # Canonicalize
    output = np.zeros_like(W)
    for i in range(len(W)):
        output[i] = _canonicalize_axis(W[i])
        
    return output, {"bank_source": "random_orthogonal", "k_dir": k_dir}


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
        from .PIA.telm2 import TELM2Config, TELM2Transformer
    except (ImportError, ValueError):
        from core.PIA.telm2 import TELM2Config, TELM2Transformer

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


def _build_spectral_structure_basis_from_zpia_bank(
    zpia_bank: np.ndarray,
    energy_ratio: float = 0.90,
    rank_tol: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Build an orthonormal z-space structure basis from the zPIA bank.

    The returned basis has shape ``[d, k_eff]`` and satisfies ``B.T @ B ≈ I``.
    """
    D = np.asarray(zpia_bank, dtype=np.float64)
    if D.ndim != 2:
        raise ValueError("zPIA bank must be a 2D matrix.")
    if D.shape[0] <= 0 or D.shape[1] <= 0:
        raise ValueError("zPIA bank must have at least one row and one feature.")
    if not np.isfinite(D).all():
        raise ValueError("zPIA bank contains NaN or inf values.")
    if not (0.0 < float(energy_ratio) <= 1.0):
        raise ValueError("energy_ratio must satisfy 0 < value <= 1.")

    try:
        _, singular_values, vh = np.linalg.svd(D, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"SVD failed while building spectral basis: {exc}") from exc

    singular_values = np.asarray(singular_values, dtype=np.float64)
    vh = np.asarray(vh, dtype=np.float64)
    if singular_values.ndim != 1 or vh.ndim != 2:
        raise ValueError("Unexpected SVD output shape while building spectral basis.")
    if not np.isfinite(singular_values).all() or not np.isfinite(vh).all():
        raise ValueError("SVD produced NaN or inf values for the spectral basis.")

    valid = singular_values > float(rank_tol)
    rank_raw = int(np.sum(valid))
    if rank_raw < 1:
        raise ValueError("zPIA bank is rank-deficient with no valid spectral directions.")

    energies = np.square(singular_values[valid])
    total_energy = float(np.sum(energies))
    if not np.isfinite(total_energy) or total_energy <= 0.0:
        raise ValueError("Spectral basis total energy is not finite and positive.")

    cum_energy = np.cumsum(energies) / total_energy
    target = float(energy_ratio)
    k_eff = int(np.searchsorted(cum_energy, target, side="left") + 1)
    k_eff = max(1, min(k_eff, rank_raw))

    B = vh[valid][:k_eff].T.copy()
    gram = B.T @ B
    orth_error = float(np.linalg.norm(gram - np.eye(k_eff), ord="fro"))
    if not np.isfinite(orth_error):
        raise ValueError("Spectral basis orthogonality check produced NaN or inf.")

    meta: Dict[str, object] = {
        "spectral_k_eff": int(k_eff),
        "spectral_rank_raw": int(rank_raw),
        "spectral_energy_ratio_target": target,
        "spectral_energy_ratio_eff": float(cum_energy[k_eff - 1]),
        "spectral_singular_values": singular_values.tolist(),
        "spectral_rank_deficient": bool(rank_raw < min(D.shape)),
        "spectral_basis_orth_error": orth_error,
    }
    return B.astype(np.float32), meta


# ═══════════════════════════════════════════════════════════════════
# AO-PIA: Augmentation-Objective PIA direction estimators
# ═══════════════════════════════════════════════════════════════════

def _symmetrize(S: np.ndarray) -> np.ndarray:
    return 0.5 * (S + S.T)


def _normalize_trace(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    tr = float(np.trace(S))
    z_dim = int(S.shape[0])
    denom = max(float(tr / z_dim), float(eps))
    return S / denom


def _class_balanced_within_scatter(
    Z: np.ndarray, y: np.ndarray, eps: float = 1e-12
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Class-balanced within-class scatter S_W."""
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).ravel()
    classes = sorted(np.unique(y).tolist())
    d = int(Z.shape[1])
    S_W = np.zeros((d, d), dtype=np.float64)
    mu_by_class: Dict[int, np.ndarray] = {}
    for c in classes:
        Zc = Z[y == c]
        if Zc.shape[0] <= 0:
            continue
        mu_c = np.mean(Zc, axis=0).astype(np.float64)
        mu_by_class[int(c)] = mu_c
        diff = Zc - mu_c[None, :]
        S_W += (diff.T @ diff) / float(max(1, Zc.shape[0]))
    S_W /= float(max(1, len(classes)))
    return _symmetrize(S_W), mu_by_class


def _between_class_scatter(
    mu_by_class: Dict[int, np.ndarray], global_mean: np.ndarray
) -> np.ndarray:
    """Between-class scatter S_B."""
    classes = sorted(mu_by_class.keys())
    d = int(global_mean.shape[0])
    S_B = np.zeros((d, d), dtype=np.float64)
    for c in classes:
        dm = mu_by_class[c] - global_mean
        S_B += np.outer(dm, dm)
    S_B /= float(max(1, len(classes)))
    return _symmetrize(S_B)


def _same_class_knn_pair_scatter(
    Z: np.ndarray, y: np.ndarray, k_pos: int = 5
) -> np.ndarray:
    """Same-class kNN positive-pair scatter S_P."""
    y_arr = np.asarray(y, dtype=np.int64).ravel()
    classes = np.unique(y_arr)
    d = int(Z.shape[1])
    S_P = np.zeros((d, d), dtype=np.float64)
    pair_count = 0
    for c in classes:
        idx_c = np.where(y_arr == c)[0]
        Zc = Z[idx_c]
        nc = Zc.shape[0]
        k_eff = min(int(k_pos), nc - 1)
        if k_eff <= 0:
            continue
        nn = NearestNeighbors(n_neighbors=min(k_eff + 1, nc), metric="euclidean")
        nn.fit(Zc)
        nn_idx = nn.kneighbors(Zc, return_distance=False)
        for i_local in range(nc):
            for j_local in nn_idx[i_local, 1 : k_eff + 1]:
                j_local = int(j_local)
                diff = Zc[i_local] - Zc[j_local]
                S_P += np.outer(diff, diff)
                pair_count += 1
    if pair_count > 0:
        S_P /= float(pair_count)
    return _symmetrize(S_P)


def _diff_class_knn_pair_scatter(
    Z: np.ndarray, y: np.ndarray, k_neg: int = 5
) -> np.ndarray:
    """Different-class kNN negative-pair scatter S_N."""
    y_arr = np.asarray(y, dtype=np.int64).ravel()
    classes = np.unique(y_arr)
    d = int(Z.shape[1])
    S_N = np.zeros((d, d), dtype=np.float64)
    pair_count = 0
    # Build per-class trees
    class_trees = {
        int(c): NearestNeighbors(
            n_neighbors=min(int(k_neg), max(1, int(np.sum(y_arr == c)) - 1)),
            metric="euclidean",
        ).fit(Z[y_arr == c])
        for c in classes
        if np.sum(y_arr == c) > 0
    }
    for c in classes:
        idx_c = np.where(y_arr == c)[0]
        if len(idx_c) == 0:
            continue
        for idx_i in idx_c:
            zi = Z[idx_i]
            for other_c in classes:
                if int(other_c) == int(c):
                    continue
                tree = class_trees.get(int(other_c))
                if tree is None:
                    continue
                k_other = min(int(k_neg), int(np.sum(y_arr == other_c)))
                if k_other <= 0:
                    continue
                _, nn_local = tree.kneighbors(zi.reshape(1, -1), n_neighbors=k_other)
                Z_other = Z[y_arr == other_c]
                for j_local in nn_local[0]:
                    j_local = int(j_local)
                    if j_local >= Z_other.shape[0]:
                        continue
                    diff = zi - Z_other[j_local]
                    S_N += np.outer(diff, diff)
                    pair_count += 1
    if pair_count > 0:
        S_N /= float(pair_count)
    return _symmetrize(S_N)


def _solve_generalized_eigen(
    S_plus: np.ndarray,
    S_minus: np.ndarray,
    rho: float,
    k_dir: int,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    """Solve S_plus d = lambda * (S_minus + rho I) d."""
    d = S_plus.shape[0]
    M = S_minus + rho * np.eye(d)
    M = _symmetrize(M)
    # Add small ridge for numerical stability
    M += eps * np.eye(d)

    eig_fallback = False
    eig_fallback_reason = ""

    try:
        from scipy.linalg import eigh as scipy_eigh
        eigvals, eigvecs = scipy_eigh(S_plus, M)
    except (ImportError, np.linalg.LinAlgError):
        # Fallback: regular EVD of M^{-1} S_plus
        try:
            A = np.linalg.solve(M, S_plus)
            eigvals, eigvecs = np.linalg.eig(A)
            eigvals = np.real(eigvals)
            eigvecs = np.real(eigvecs)
            eig_fallback = True
            eig_fallback_reason = "generalized_eigh_failed_used_eig_of_invM_Splus"
        except np.linalg.LinAlgError:
            # Last resort: PCA of S_plus
            eigvals, eigvecs = np.linalg.eigh(S_plus)
            eig_fallback = True
            eig_fallback_reason = "all_eig_failed_used_pca_of_Splus"

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    k_actual = min(int(k_dir), d)
    return eigvals[:k_actual], eigvecs[:, :k_actual], eig_fallback, eig_fallback_reason


def build_ao_pia_direction_bank(
    Z: np.ndarray,
    y: np.ndarray,
    k_dir: int = 10,
    rho_scale: float = 1e-3,
    mode: str = "ao_fisher",
    lambda_pos: float = 0.5,
    lambda_neg: float = 0.5,
    k_pos: int = 5,
    k_neg: int = 5,
    seed: int = 1,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Build augmentation-objective-guided PIA direction bank.

    Modes:
      - ``ao_fisher``: within-class (S_W) vs between-class (S_B)
      - ``ao_contrastive``: within+positive (S_W+S_P) vs between+negative (S_B+S_N)

    Returns (direction_bank [k_dir, z_dim], meta).
    """  # noqa: D401
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).ravel()
    if Z.ndim != 2 or Z.shape[0] <= 0 or Z.shape[1] <= 0:
        raise ValueError("Z must be a non-empty 2D array.")
    if not np.isfinite(Z).all():
        raise ValueError("Z contains NaN or inf values.")

    if mode not in ("ao_fisher", "ao_contrastive"):
        raise ValueError(f"Unsupported AO-PIA mode: {mode}")

    z_dim = int(Z.shape[1])
    k_actual = min(int(k_dir), z_dim)
    global_mean = np.mean(Z, axis=0).astype(np.float64)

    # --- Compute scatter matrices ---
    S_W, mu_by_class = _class_balanced_within_scatter(Z, y, eps=eps)
    S_B = _between_class_scatter(mu_by_class, global_mean)

    S_P = np.zeros((z_dim, z_dim), dtype=np.float64)
    S_N = np.zeros((z_dim, z_dim), dtype=np.float64)

    if mode == "ao_contrastive":
        S_P = _same_class_knn_pair_scatter(Z, y, k_pos=int(k_pos))
        S_N = _diff_class_knn_pair_scatter(Z, y, k_neg=int(k_neg))

    # --- Build S_plus and S_minus ---
    S_plus_raw = _normalize_trace(S_W, eps=eps)
    S_minus_raw = _normalize_trace(S_B, eps=eps)

    sw_trace = float(np.trace(S_W))
    sb_trace = float(np.trace(S_B))
    sp_trace = float(np.trace(S_P))
    sn_trace = float(np.trace(S_N))

    if mode == "ao_contrastive":
        S_plus_raw = S_plus_raw + float(lambda_pos) * _normalize_trace(S_P, eps=eps)
        S_minus_raw = S_minus_raw + float(lambda_neg) * _normalize_trace(S_N, eps=eps)

    S_plus = _symmetrize(S_plus_raw)
    S_minus = _symmetrize(S_minus_raw)

    # --- Rho ---
    tr_sminus = float(np.trace(S_minus))
    if np.isfinite(tr_sminus) and tr_sminus > eps:
        rho = float(rho_scale) * tr_sminus / float(z_dim)
    else:
        rho = float(rho_scale)
    rho = max(rho, eps)

    # --- Solve generalized eigenvalue problem ---
    eigvals, eigvecs, eig_fallback, eig_fallback_reason = _solve_generalized_eigen(
        S_plus, S_minus, rho, k_actual, eps=eps
    )

    # --- Normalize directions ---
    # We must ensure we return exactly k_dir directions, even if z_dim < k_dir
    # by padding with random unit directions if needed.
    bank = np.zeros((k_dir, z_dim), dtype=np.float64)
    for i in range(k_actual):
        v = eigvecs[:, i].copy()
        nrm = float(np.linalg.norm(v))
        if nrm > eps:
            v /= nrm
        else:
            v = np.zeros(z_dim)
            v[i % z_dim] = 1.0
        bank[i] = _canonicalize_axis(v)

    # Pad if k_dir > k_actual
    pad_count = k_dir - k_actual
    if pad_count > 0:
        rng = np.random.default_rng(int(seed) + 999)
        for i in range(pad_count):
            v = rng.standard_normal(z_dim)
            v /= (np.linalg.norm(v) + eps)
            bank[k_actual + i] = _canonicalize_axis(v)

    row_norms = np.linalg.norm(bank, axis=1)

    # --- Meta ---
    eig_top_all = eigvals.tolist() if len(eigvals) > 0 else []
    eig_finite = eigvals[np.isfinite(eigvals)]
    meta: Dict[str, object] = {
        "bank_source": mode,
        "k_dir": k_dir,
        "z_dim": z_dim,
        "n_train": int(Z.shape[0]),
        "rho_scale": float(rho_scale),
        "rho_value": float(rho),
        "lambda_pos": float(lambda_pos),
        "lambda_neg": float(lambda_neg),
        "k_pos": int(k_pos),
        "k_neg": int(k_neg),
        "sw_trace": sw_trace,
        "sb_trace": sb_trace,
        "sp_trace": sp_trace,
        "sn_trace": sn_trace,
        "eig_top": eig_top_all[0] if len(eig_top_all) > 0 else np.nan,
        "eig_mean": float(np.mean(eig_finite)) if eig_finite.size > 0 else np.nan,
        "eig_min": float(np.min(eig_finite)) if eig_finite.size > 0 else np.nan,
        "eig_max": float(np.max(eig_finite)) if eig_finite.size > 0 else np.nan,
        "eig_fallback": bool(eig_fallback),
        "eig_fallback_reason": eig_fallback_reason,
        "ao_direction_pad_count": int(pad_count),
        "ao_direction_pad_mode": "seeded_random_unit",
        "response_centering": "class_residual",
        "direction_norm_mean": float(np.mean(row_norms)),
        "direction_norm_min": float(np.min(row_norms)),
        "direction_norm_max": float(np.max(row_norms)),
    }
    return bank.astype(np.float32), meta
