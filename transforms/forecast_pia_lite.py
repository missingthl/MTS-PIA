from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PIALiteConfig:
    epsilon: float = 0.01
    aug_ratio: float = 1.0
    direction_source: str = "batch_local"
    anchor_protection_enabled: bool = False
    anchor_protection_scope: str = "none"


def _deterministic_alt_signs(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.arange(n, device=device)
    signs = torch.where((idx % 2) == 0, 1.0, -1.0)
    return signs.to(dtype=dtype)


def _batch_local_top1_direction(x_flat: torch.Tensor) -> torch.Tensor:
    # x_flat: [B, D]
    if x_flat.ndim != 2:
        raise ValueError(f"x_flat must be 2D, got shape {tuple(x_flat.shape)}")
    if x_flat.shape[0] < 2:
        return torch.zeros(x_flat.shape[1], device=x_flat.device, dtype=x_flat.dtype)
    x_center = x_flat - x_flat.mean(dim=0, keepdim=True)
    if torch.linalg.norm(x_center) <= 1e-12:
        return torch.zeros(x_flat.shape[1], device=x_flat.device, dtype=x_flat.dtype)
    try:
        _, _, vh = torch.linalg.svd(x_center, full_matrices=False)
        w = vh[0]
    except RuntimeError:
        cov = x_center.t() @ x_center
        eigvals, eigvecs = torch.linalg.eigh(cov)
        w = eigvecs[:, -1]
    norm = torch.linalg.norm(w)
    if norm <= 1e-12:
        return torch.zeros_like(w)
    return w / norm


def apply_pia_lite_batch(
    x: torch.Tensor,
    *,
    cfg: PIALiteConfig,
) -> tuple[torch.Tensor, int, float]:
    """
    x: [B, L, C]
    returns:
      x_aug: [B_aug, L, C]
      n_aug: int
      runtime_sec: float
    """
    start = time.perf_counter()
    if cfg.aug_ratio <= 0.0 or cfg.epsilon <= 0.0:
        return x.new_empty((0, *x.shape[1:])), 0, 0.0

    batch_size, lookback, n_feat = x.shape
    n_aug = int(round(batch_size * cfg.aug_ratio))
    n_aug = max(0, min(batch_size, n_aug))
    if n_aug == 0:
        return x.new_empty((0, lookback, n_feat)), 0, 0.0

    if cfg.direction_source != "batch_local":
        raise ValueError(f"Unsupported direction_source: {cfg.direction_source}")

    x_flat = x.reshape(batch_size, lookback * n_feat)
    w = _batch_local_top1_direction(x_flat)
    aug_idx = torch.arange(n_aug, device=x.device)
    x_sel = x_flat[aug_idx]
    signs = _deterministic_alt_signs(n_aug, x.device, x.dtype).unsqueeze(1)
    x_aug_flat = x_sel + float(cfg.epsilon) * signs * w.unsqueeze(0).to(dtype=x.dtype)
    x_aug = x_aug_flat.reshape(n_aug, lookback, n_feat)

    if cfg.anchor_protection_enabled and cfg.anchor_protection_scope == "last_step_only":
        x_aug[:, -1, :] = x[aug_idx, -1, :]

    runtime = time.perf_counter() - start
    return x_aug, n_aug, float(runtime)
