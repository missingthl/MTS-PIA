from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def symmetrize(mat: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mat + mat.transpose(-1, -2))


def _spd_eigh(mat: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    mat = symmetrize(mat)
    n = int(mat.shape[-1])
    eye = torch.eye(n, device=mat.device, dtype=mat.dtype)
    mat = mat + float(eps) * eye
    vals, vecs = torch.linalg.eigh(mat)
    vals = torch.clamp(vals, min=float(eps))
    return vals, vecs


def spd_sqrtm(mat: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    vals, vecs = _spd_eigh(mat, eps)
    out = vecs @ torch.diag_embed(torch.sqrt(vals)) @ vecs.transpose(-1, -2)
    return symmetrize(out)


def spd_invsqrtm(mat: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    vals, vecs = _spd_eigh(mat, eps)
    out = vecs @ torch.diag_embed(torch.rsqrt(vals)) @ vecs.transpose(-1, -2)
    return symmetrize(out)


def spd_expm(sym_mat: torch.Tensor) -> torch.Tensor:
    sym_mat = symmetrize(sym_mat)
    vals, vecs = torch.linalg.eigh(sym_mat)
    out = vecs @ torch.diag_embed(torch.exp(vals)) @ vecs.transpose(-1, -2)
    return symmetrize(out)


def spd_logm(mat: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    vals, vecs = _spd_eigh(mat, eps)
    out = vecs @ torch.diag_embed(torch.log(vals)) @ vecs.transpose(-1, -2)
    return symmetrize(out)


def unvec_utri_sym(vec: np.ndarray, dim: int) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).ravel()
    idx = np.triu_indices(int(dim))
    if v.size != idx[0].size:
        raise ValueError(f"upper-tri vector size mismatch: got {v.size}, expected {idx[0].size}")
    out = np.zeros((int(dim), int(dim)), dtype=np.float64)
    out[idx] = v
    out[(idx[1], idx[0])] = v
    return out


def logvec_to_spd(vec: np.ndarray, mean_log: np.ndarray) -> np.ndarray:
    mean_log = np.asarray(mean_log, dtype=np.float64)
    log_centered = unvec_utri_sym(vec, int(mean_log.shape[0]))
    log_cov = (log_centered + mean_log)
    return spd_expm(torch.from_numpy(log_cov).double()).cpu().numpy().astype(np.float32)


def covariance_from_signal(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    denom = max(1, int(x.shape[-1]) - 1)
    cov = (x @ x.transpose(-1, -2)) / float(denom)
    n = int(cov.shape[-1])
    cov = symmetrize(cov)
    cov = cov + float(eps) * torch.eye(n, device=cov.device, dtype=cov.dtype)
    return cov


def whitening_step(
    x_orig: torch.Tensor, 
    sigma_orig: torch.Tensor, 
    eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Step 1: Whitening.
    Center the signal and remove the original covariance structure. This is the
    first half of the whitening-coloring covariance realization bridge.
    """
    W_whiten = spd_invsqrtm(sigma_orig, eps=eps)
    x_centered = x_orig - x_orig.mean(dim=-1, keepdim=True)
    x_white = W_whiten @ x_centered
    return x_white, W_whiten


def coloring_step(
    x_white: torch.Tensor, 
    sigma_target: torch.Tensor, 
    x_mean_orig: torch.Tensor,
    eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Step 2: Coloring.
    Inject the target covariance into the whitened signal. This realizes a
    covariance target in raw time-series space; it is not claimed to be an
    exact Bures-Wasserstein optimal transport map.
    """
    W_color = spd_sqrtm(sigma_target, eps=eps)
    x_colored = W_color @ x_white + x_mean_orig
    return x_colored, W_color


def check_isometry(
    x_orig: torch.Tensor, 
    x_aug: torch.Tensor, 
    A: torch.Tensor, 
    eps: float = 1e-12
) -> Dict[str, float]:
    """
    Bridge deformation and conditioning diagnostics.
    ``metric_preservation_error`` is a legacy field name kept for CSV
    compatibility; interpret it as ||A^T A - I||_F, a deformation diagnostic,
    not as a strict isometry guarantee.
    """
    # Legacy metric_preservation_error field: ||A^T A - I||_F.
    n = A.shape[-1]
    eye = torch.eye(n, device=A.device, dtype=A.dtype)
    isometry_err = torch.linalg.norm(A.transpose(-1, -2) @ A - eye)
    
    # Operator condition number: measures numeric stability of the transport
    svals = torch.linalg.svdvals(A)
    cond_A = svals.max() / (svals.min() + eps)
    
    # Gain norm (how much we moved from identity mapping)
    gain_norm = torch.linalg.norm(A - eye)
    
    return {
        "metric_preservation_error": float(isometry_err.item()),
        "operator_cond_number": float(cond_A.item()),
        "operator_gain_norm": float(gain_norm.item())
    }


def bridge_single(
    x_orig: torch.Tensor,
    sigma_orig: torch.Tensor,
    sigma_aug: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Whitening-coloring covariance realization map.

    Given an original signal, its covariance, and a target covariance, this
    bridge whitens the original sample and colors it with the target covariance.
    The returned diagnostics measure covariance-target fidelity and bridge
    deformation/conditioning; the implementation does not claim exact Gaussian
    OT or Bures-Wasserstein optimality.
    """
    x_orig = x_orig.to(dtype=torch.float64)
    sigma_orig = sigma_orig.to(dtype=torch.float64)
    sigma_aug = sigma_aug.to(dtype=torch.float64)
    
    mu_orig = x_orig.mean(dim=-1, keepdim=True)

    # 1. Whitening
    x_white, W_whiten = whitening_step(x_orig, sigma_orig, eps=eps)
    
    # 2. Coloring
    x_aug, W_color = coloring_step(x_white, sigma_aug, mu_orig, eps=eps)
    
    # 3. Bridge operator diagnostics
    A = W_color @ W_whiten
    iso_metrics = check_isometry(x_orig, x_aug, A, eps=eps)

    # 4. Covariance-target fidelity diagnostics
    cov_aug_emp = covariance_from_signal(x_aug, eps=eps)
    
    # Covariance realization error.
    transport_err_fro = torch.linalg.norm(cov_aug_emp - sigma_aug)
    
    log_cov_aug_emp = spd_logm(cov_aug_emp, eps=eps)
    log_sigma_aug = spd_logm(sigma_aug, eps=eps)
    transport_err_logeuc = torch.linalg.norm(log_cov_aug_emp - log_sigma_aug)
    
    # Distance from origin
    dist_to_orig_fro = torch.linalg.norm(cov_aug_emp - sigma_orig)
    
    meta = {
        "transport_error_fro": float(transport_err_fro.item()),
        "transport_error_logeuc": float(transport_err_logeuc.item()),
        "transport_to_orig_fro": float(dist_to_orig_fro.item()),
        "bridge_cond_A": iso_metrics["operator_cond_number"],
        "bridge_gain_norm": iso_metrics["operator_gain_norm"],
        "metric_preservation_error": iso_metrics["metric_preservation_error"],
        "raw_mean_shift_abs": float(torch.mean(torch.abs(x_aug.mean(dim=-1) - mu_orig.squeeze(-1))).item())
    }
    
    return x_aug.to(dtype=torch.float32), meta
