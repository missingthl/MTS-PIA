from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from utils.external_baseline_methods.base import resample_ct


def dtw_path_tc(
    prototype_tc: np.ndarray,
    sample_tc: np.ndarray,
    *,
    slope_constraint: str = "symmetric",
    window: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Small multivariate DTW path helper for guided-warping baselines."""
    prototype_tc = np.asarray(prototype_tc, dtype=np.float64)
    sample_tc = np.asarray(sample_tc, dtype=np.float64)
    p = int(prototype_tc.shape[0])
    s = int(sample_tc.shape[0])
    if p <= 0 or s <= 0:
        raise ValueError("DTW inputs must be non-empty.")
    if slope_constraint not in {"symmetric", "asymmetric"}:
        raise ValueError(f"Unsupported slope_constraint={slope_constraint!r}")
    window_i = max(p, s) if window is None else max(1, int(window))

    cost = np.full((p, s), np.inf, dtype=np.float64)
    for i in range(p):
        start = max(0, i - window_i)
        stop = min(s, i + window_i + 1)
        if start < stop:
            cost[i, start:stop] = np.linalg.norm(sample_tc[start:stop] - prototype_tc[i], axis=1)

    dtw = np.full((p + 1, s + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0
    if slope_constraint == "symmetric":
        for i in range(1, p + 1):
            for j in range(max(1, i - window_i), min(s, i + window_i) + 1):
                dtw[i, j] = cost[i - 1, j - 1] + min(dtw[i - 1, j - 1], dtw[i - 1, j], dtw[i, j - 1])
    else:
        for i in range(1, p + 1):
            if i <= window_i + 1:
                dtw[i, 1] = cost[i - 1, 0] + min(dtw[i - 1, 0], dtw[i - 1, 1])
            for j in range(max(2, i - window_i), min(s, i + window_i) + 1):
                dtw[i, j] = cost[i - 1, j - 1] + min(dtw[i - 1, j - 2], dtw[i - 1, j - 1], dtw[i - 1, j])

    if not np.isfinite(dtw[p, s]):
        raise RuntimeError("DTW failed to find a finite path; try disabling the window constraint.")

    i, j = p, s
    path_p: List[int] = [p - 1]
    path_s: List[int] = [s - 1]
    while i > 1 or j > 1:
        if slope_constraint == "symmetric":
            options = (
                (dtw[i - 1, j - 1], i - 1, j - 1),
                (dtw[i - 1, j], i - 1, j),
                (dtw[i, j - 1], i, j - 1),
            )
        else:
            options = (
                (dtw[i - 1, j], i - 1, j),
                (dtw[i - 1, j - 1], i - 1, j - 1),
                (dtw[i - 1, j - 2] if j >= 2 else np.inf, i - 1, max(0, j - 2)),
            )
        _, i, j = min(options, key=lambda item: item[0])
        path_p.insert(0, max(0, i - 1))
        path_s.insert(0, max(0, j - 1))
        if i <= 1 and j <= 1:
            break

    return np.asarray(path_p, dtype=np.int64), np.asarray(path_s, dtype=np.int64), float(dtw[p, s])


def guided_warp_ct(
    anchor_ct: np.ndarray,
    prototype_ct: np.ndarray,
    *,
    slope_constraint: str,
    use_window: bool,
) -> Tuple[np.ndarray, float, float]:
    """Warp an anchor [C, T] by the DTW path from prototype to anchor."""
    anchor_ct = np.asarray(anchor_ct, dtype=np.float32)
    prototype_ct = np.asarray(prototype_ct, dtype=np.float32)
    t = int(anchor_ct.shape[1])
    window = int(np.ceil(t / 10.0)) if use_window else None
    _, path_s, dtw_value = dtw_path_tc(
        prototype_ct.T,
        anchor_ct.T,
        slope_constraint=slope_constraint,
        window=window,
    )
    warped_ct = anchor_ct[:, path_s]
    x_aug = resample_ct(warped_ct, t)
    orig_steps = np.arange(t, dtype=np.float64)
    warp_path_interp = np.interp(orig_steps, np.linspace(0.0, max(t - 1.0, 0.0), num=len(path_s)), path_s)
    warp_amount = float(np.sum(np.abs(orig_steps - warp_path_interp)))
    return x_aug.astype(np.float32), float(dtw_value), float(warp_amount)


def window_slice_ct(
    x_ct: np.ndarray,
    *,
    reduce_ratio: float,
    rng: np.random.Generator,
    min_window_len: int = 4,
) -> np.ndarray:
    x_ct = np.asarray(x_ct, dtype=np.float32)
    t = int(x_ct.shape[1])
    if t <= 1:
        return x_ct.astype(np.float32, copy=True)
    win_len = int(round(float(reduce_ratio) * t))
    win_len = max(int(min_window_len), win_len)
    win_len = min(max(1, win_len), t)
    if win_len >= t:
        return x_ct.astype(np.float32, copy=True)
    start = int(rng.integers(0, t - win_len + 1))
    return resample_ct(x_ct[:, start:start + win_len], t)
