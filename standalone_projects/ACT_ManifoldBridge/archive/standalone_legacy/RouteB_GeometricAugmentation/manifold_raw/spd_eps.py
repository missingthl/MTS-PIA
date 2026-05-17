from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_spd_eps(
    cov: np.ndarray,
    *,
    mode: str,
    absolute: float,
    alpha: float,
    floor_mult: float,
    ceil_mult: float,
) -> Tuple[float, float]:
    mode = (mode or "absolute").lower()
    if mode == "absolute":
        return float(absolute), float(np.trace(cov) / cov.shape[0]) if cov.size else 0.0

    if cov.size == 0:
        return float(absolute), 0.0

    diag = np.diag(cov)
    if mode == "relative_trace":
        base = float(np.trace(cov) / cov.shape[0])
    elif mode == "relative_diag":
        base = float(np.mean(diag)) if diag.size else 0.0
    else:
        raise ValueError(f"Unknown spd_eps mode: {mode}")

    if not np.isfinite(base) or base <= 0.0:
        base = float(np.mean(np.abs(diag))) if diag.size else 0.0

    if base <= 0.0:
        return float(absolute), 0.0

    eps0 = alpha * base
    eps_floor = floor_mult * base
    eps_ceil = ceil_mult * base
    eps = min(max(eps0, eps_floor), eps_ceil)
    return float(eps), float(base)
