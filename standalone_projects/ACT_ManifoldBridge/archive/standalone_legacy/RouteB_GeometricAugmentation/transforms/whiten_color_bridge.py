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
    log_cov = symmetrize(torch.from_numpy(log_centered + mean_log).double()).cpu().numpy()
    return spd_expm(torch.from_numpy(log_cov).double()).cpu().numpy().astype(np.float32)


def covariance_from_signal(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    denom = max(1, int(x.shape[-1]) - 1)
    cov = (x @ x.transpose(-1, -2)) / float(denom)
    n = int(cov.shape[-1])
    cov = symmetrize(cov)
    cov = cov + float(eps) * torch.eye(n, device=cov.device, dtype=cov.dtype)
    return cov


def bridge_single(
    x_orig: torch.Tensor,
    sigma_orig: torch.Tensor,
    sigma_aug: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Whitening-coloring bridge for one [C, T] sample."""

    x_orig = x_orig.to(dtype=torch.float64)
    sigma_orig = sigma_orig.to(dtype=torch.float64)
    sigma_aug = sigma_aug.to(dtype=torch.float64)

    sqrt_aug = spd_sqrtm(sigma_aug, eps=eps)
    invsqrt_orig = spd_invsqrtm(sigma_orig, eps=eps)
    A = sqrt_aug @ invsqrt_orig

    x_centered = x_orig - x_orig.mean(dim=1, keepdim=True)
    x_aug = A @ x_centered + x_orig.mean(dim=1, keepdim=True)

    cov_aug_emp = covariance_from_signal(x_aug, eps=eps)
    cov_delta_aug = cov_aug_emp - sigma_aug
    cov_delta_orig = cov_aug_emp - sigma_orig
    cov_gap = torch.linalg.norm(cov_delta_aug) / (torch.linalg.norm(sigma_aug) + 1e-12)
    cov_gap_fro = torch.linalg.norm(cov_delta_aug)
    cov_to_orig_fro = torch.linalg.norm(cov_delta_orig)
    log_cov_aug_emp = spd_logm(cov_aug_emp, eps=eps)
    log_sigma_aug = spd_logm(sigma_aug, eps=eps)
    log_sigma_orig = spd_logm(sigma_orig, eps=eps)
    cov_gap_logeuc = torch.linalg.norm(log_cov_aug_emp - log_sigma_aug)
    cov_to_orig_logeuc = torch.linalg.norm(log_cov_aug_emp - log_sigma_orig)
    gain_norm = torch.linalg.norm(A - torch.eye(A.shape[0], device=A.device, dtype=A.dtype))
    energy_ratio = torch.linalg.norm(x_aug) / (torch.linalg.norm(x_orig) + 1e-12)
    svals_A = torch.linalg.svdvals(A)
    cond_A = svals_A.max() / (svals_A.min() + 1e-12)
    vals_orig, _ = _spd_eigh(sigma_orig, eps)
    mu_orig = x_orig.mean(dim=1)
    mu_aug = x_aug.mean(dim=1)
    raw_mean_shift_abs = torch.mean(torch.abs(mu_aug - mu_orig))

    meta = {
        "bridge_cov_match_error": float(cov_gap.item()),
        "bridge_cov_match_error_fro": float(cov_gap_fro.item()),
        "bridge_cov_match_error_logeuc": float(cov_gap_logeuc.item()),
        "bridge_cov_to_orig_distance_fro": float(cov_to_orig_fro.item()),
        "bridge_cov_to_orig_distance_logeuc": float(cov_to_orig_logeuc.item()),
        "bridge_gain_norm": float(gain_norm.item()),
        "bridge_energy_ratio": float(energy_ratio.item()),
        "bridge_cond_A": float(cond_A.item()),
        "sigma_orig_min_eig": float(vals_orig.min().item()),
        "sigma_orig_max_eig": float(vals_orig.max().item()),
        "raw_mean_shift_abs": float(raw_mean_shift_abs.item()),
    }
    return x_aug.to(dtype=torch.float32), meta
