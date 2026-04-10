from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class BandSpec:
    name: str
    lo: float
    hi: float


def parse_band_spec(spec: str) -> List[BandSpec]:
    bands: List[BandSpec] = []
    if not spec:
        return bands
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part or "-" not in part:
            raise ValueError(f"Invalid band spec: {part}")
        name, rng = part.split(":", 1)
        lo, hi = rng.split("-", 1)
        bands.append(BandSpec(name=name.strip(), lo=float(lo), hi=float(hi)))
    return bands


def window_slices(
    n_samples: int,
    fs: float,
    win_sec: float,
    hop_sec: float,
) -> List[Tuple[int, int]]:
    win_len = int(round(win_sec * fs))
    hop_len = int(round(hop_sec * fs))
    if win_len <= 0 or hop_len <= 0:
        raise ValueError("window_sec/hop_sec produce non-positive window length")
    if n_samples < win_len:
        return [(0, n_samples)]
    n_windows = 1 + (n_samples - win_len) // hop_len
    return [(i * hop_len, i * hop_len + win_len) for i in range(n_windows)]


def bandpass(data: np.ndarray, fs: float, band: BandSpec, *, chunk_size: int = 0) -> np.ndarray:
    import mne

    # Ultra-short symbolic sequences (for example PenDigits with T=8) are not
    # meaningful for FIR band-pass filtering and can trigger unstable / very
    # expensive filtering behaviour. Keep them as pass-through features.
    if int(data.shape[1]) < 16:
        return np.array(data, dtype=np.float32, copy=True)

    n_ch = data.shape[0]
    if chunk_size <= 0 or chunk_size >= n_ch:
        data64 = np.asarray(data, dtype=np.float64)
        return mne.filter.filter_data(
            data64,
            sfreq=fs,
            l_freq=band.lo,
            h_freq=band.hi,
            n_jobs=1,
            verbose="ERROR",
        ).astype(np.float32, copy=False)

    out = np.empty_like(data, dtype=np.float32)
    for start in range(0, n_ch, chunk_size):
        end = min(start + chunk_size, n_ch)
        chunk64 = np.asarray(data[start:end], dtype=np.float64)
        chunk = mne.filter.filter_data(
            chunk64,
            sfreq=fs,
            l_freq=band.lo,
            h_freq=band.hi,
            n_jobs=1,
            verbose="ERROR",
        )
        out[start:end] = chunk.astype(np.float32, copy=False)
    return out


def cov_shrink(data: np.ndarray, method: str) -> np.ndarray:
    from sklearn.covariance import LedoitWolf, OAS

    X = np.asarray(data).T  # (T, C)
    method = method.lower()
    if method == "scm":
        X = X - X.mean(axis=0, keepdims=True)
        n_samples = X.shape[0]
        denom = max(1, n_samples - 1)
        cov = (X.T @ X) / float(denom)
    elif method == "shrinkage_oas":
        cov = OAS().fit(X).covariance_
    elif method == "shrinkage_lw":
        cov = LedoitWolf().fit(X).covariance_
    else:
        raise ValueError(f"Unknown covariance method: {method}")
    cov = 0.5 * (cov + cov.T)
    return cov


def logmap_spd(cov: np.ndarray, eps: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, eps)
    log_vals = np.log(vals)
    return (vecs * log_vals) @ vecs.T


def vec_utri(mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(mat.shape[0])
    return mat[idx]


def trial_aggregate(
    band_windows: List[np.ndarray],
    n_windows: int,
) -> np.ndarray:
    if n_windows <= 0:
        raise ValueError("n_windows must be positive for trial aggregation")
    if not band_windows:
        raise ValueError("band_windows must be non-empty for trial aggregation")
    for band in band_windows:
        if band.shape[0] != n_windows:
            raise ValueError("band_windows entries must share the same n_windows")
    return np.concatenate(band_windows, axis=1)
