from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from core.bridge import covariance_from_signal, spd_invsqrtm, spd_logm, spd_sqrtm, symmetrize


def whiten_sample(
    x: torch.Tensor,
    sigma: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map a C x T trial into covariance-whitened coordinates."""
    x = x.to(dtype=torch.float64)
    sigma = sigma.to(dtype=torch.float64)
    mu = x.mean(dim=-1, keepdim=True)
    w_whiten = spd_invsqrtm(sigma, eps=eps)
    x_white = w_whiten @ (x - mu)
    return x_white, mu, w_whiten


def _canonical_rank1_basis(x_white: torch.Tensor, eps: float) -> Tuple[torch.Tensor, str]:
    if torch.linalg.norm(x_white) <= float(eps):
        return torch.zeros_like(x_white), "no_edit"

    try:
        u, s, vh = torch.linalg.svd(x_white, full_matrices=False)
    except RuntimeError:
        return torch.zeros_like(x_white), "svd_failed"

    if s.numel() == 0 or float(s[0].item()) <= float(eps):
        return torch.zeros_like(x_white), "no_edit"

    basis = torch.outer(u[:, 0], vh[0, :])
    basis_norm = torch.linalg.norm(basis)
    if float(basis_norm.item()) <= float(eps):
        return torch.zeros_like(x_white), "no_edit"

    basis = basis / basis_norm
    flat = basis.reshape(-1)
    pivot = int(torch.argmax(torch.abs(flat)).item())
    if float(flat[pivot].item()) < 0.0:
        basis = -basis
    return basis, "ok"


def line_edit(
    x_white: torch.Tensor,
    *,
    sign: float,
    edit_alpha_scale: float,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Apply a single rank-1 line edit in whitened coordinates."""
    x_white = x_white.to(dtype=torch.float64)
    t_len = int(x_white.shape[-1])
    denom = math.sqrt(float(max(t_len - 1, 1)))
    edit_alpha = float(edit_alpha_scale) * denom
    basis, status = _canonical_rank1_basis(x_white, eps=eps)
    if status != "ok" or abs(edit_alpha) <= float(eps):
        delta = torch.zeros_like(x_white)
        status = "ok_zero_alpha" if status == "ok" else status
    else:
        delta = float(sign) * edit_alpha * basis

    x_edit = x_white + delta
    edit_norm = float(torch.linalg.norm(delta).item())
    basis_norm = float(torch.linalg.norm(basis).item())
    meta = {
        "edit_alpha": float(edit_alpha),
        "edit_norm": edit_norm,
        "edit_energy": float(edit_norm / (denom + 1e-12)),
        "edit_basis_fro_norm": basis_norm,
        "edit_status_code": 1.0 if status.startswith("ok") else 0.0,
    }
    return x_edit, meta


def recolor_sample(
    x_white_edit: torch.Tensor,
    sigma_target: torch.Tensor,
    mu: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map edited whitened coordinates to the target covariance state."""
    sigma_target = sigma_target.to(dtype=torch.float64)
    mu = mu.to(dtype=torch.float64)
    w_color = spd_sqrtm(sigma_target, eps=eps)
    x_recolored = w_color @ x_white_edit.to(dtype=torch.float64) + mu
    return x_recolored, w_color


def white_edit_single(
    x_orig: torch.Tensor,
    sigma_orig: torch.Tensor,
    sigma_target: torch.Tensor,
    *,
    sign: float,
    edit_alpha_scale: float,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Whiten, rank-1 edit, recolor, and report target-covariance fidelity."""
    x_white, mu, w_whiten = whiten_sample(x_orig, sigma_orig, eps=eps)
    x_edit, edit_meta = line_edit(
        x_white,
        sign=sign,
        edit_alpha_scale=edit_alpha_scale,
        eps=eps,
    )
    x_aug, w_color = recolor_sample(x_edit, sigma_target, mu, eps=eps)

    cov_aug_emp = covariance_from_signal(x_aug, eps=eps)
    sigma_target = symmetrize(sigma_target.to(dtype=torch.float64))
    transport_err_fro = torch.linalg.norm(cov_aug_emp - sigma_target)
    log_cov_aug_emp = spd_logm(cov_aug_emp, eps=eps)
    log_sigma_target = spd_logm(sigma_target, eps=eps)
    transport_err_logeuc = torch.linalg.norm(log_cov_aug_emp - log_sigma_target)
    a_op = w_color @ w_whiten
    svals = torch.linalg.svdvals(a_op)
    cond_a = svals.max() / (svals.min() + eps)

    meta = {
        **edit_meta,
        "transport_error_fro": float(transport_err_fro.item()),
        "transport_error_logeuc": float(transport_err_logeuc.item()),
        "recolor_transport_error_fro": float(transport_err_fro.item()),
        "recolor_transport_error_logeuc": float(transport_err_logeuc.item()),
        "bridge_cond_A": float(cond_a.item()),
        "raw_mean_shift_abs": float(torch.mean(torch.abs(x_aug.mean(dim=-1) - mu.squeeze(-1))).item()),
    }
    return x_aug.to(dtype=torch.float32), meta


def white_identity_error(
    x_orig: torch.Tensor,
    sigma_orig: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> float:
    """Check that whiten + recolor with the original covariance reconstructs x."""
    x_orig64 = x_orig.to(dtype=torch.float64)
    x_white, mu, _ = whiten_sample(x_orig64, sigma_orig, eps=eps)
    x_rec, _ = recolor_sample(x_white, sigma_orig, mu, eps=eps)
    denom = math.sqrt(float(max(x_orig64.numel(), 1)))
    return float((torch.linalg.norm(x_rec - x_orig64) / (denom + 1e-12)).item())
