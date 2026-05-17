from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.covariance import LedoitWolf, OAS

from manifold_raw.features import bandpass, window_slices


def regularize_spd(cov: np.ndarray, eps: float) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    cov = cov + np.eye(cov.shape[0], dtype=cov.dtype) * float(eps)
    return cov


def logm_spd(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    vals = np.maximum(vals, float(eps))
    log_vals = np.log(vals)
    return (vecs * log_vals) @ vecs.T


def vec_utri(mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(mat.shape[0])
    return np.asarray(mat, dtype=np.float64)[idx]


def _cov_empirical(x_cat: np.ndarray, eps: float) -> np.ndarray:
    x_c = np.asarray(x_cat, dtype=np.float64)
    x_c = x_c - x_c.mean(axis=1, keepdims=True)
    denom = max(1, int(x_c.shape[1]) - 1)
    cov = (x_c @ x_c.T) / float(denom)
    return regularize_spd(cov, float(eps))


def _cov_oas(x_cat: np.ndarray, eps: float) -> np.ndarray:
    try:
        oa = OAS(assume_centered=False)
        oa.fit(np.asarray(x_cat, dtype=np.float64).T)
        return regularize_spd(np.asarray(oa.covariance_, dtype=np.float32), float(eps))
    except Exception:
        return _cov_empirical(x_cat, float(eps))


def _cov_lw(x_cat: np.ndarray, eps: float) -> np.ndarray:
    try:
        lw = LedoitWolf(assume_centered=False)
        lw.fit(np.asarray(x_cat, dtype=np.float64).T)
        return regularize_spd(np.asarray(lw.covariance_, dtype=np.float32), float(eps))
    except Exception:
        return _cov_empirical(x_cat, float(eps))


def extract_features_block(
    trials: List[Dict],
    win_sec: float,
    hop_sec: float,
    est_mode: str,
    spd_eps: float,
    bands_spec: List,
    *,
    progress_prefix: Optional[str] = None,
    progress_every: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    cov_list: List[np.ndarray] = []
    y_list: List[int] = []
    tid_list: List[str] = []

    bands_sorted = tuple(sorted(bands_spec, key=lambda b: str(b.name)))
    window_cache: Dict[Tuple[int, float, float, float], List[Tuple[int, int]]] = {}
    total_trials = len(trials)
    total_windows = 0
    t_start = time.time()

    if progress_prefix:
        print(
            f"{progress_prefix} start trials={total_trials} "
            f"win={float(win_sec):.2f}s hop={float(hop_sec):.2f}s est={est_mode}",
            flush=True,
        )

    for trial_idx, t in enumerate(trials, start=1):
        b_data = [(b.name, bandpass(t["x_trial"], t["sfreq"], b)) for b in bands_sorted]
        first_band = b_data[0][1]
        n_samples = int(first_band.shape[1])
        fs = float(t["sfreq"])
        w_key = (n_samples, fs, float(win_sec), float(hop_sec))
        w_list = window_cache.get(w_key)
        if w_list is None:
            w_list = window_slices(n_samples, fs, float(win_sec), float(hop_sec))
            window_cache[w_key] = w_list
        total_windows += len(w_list)

        for s, e in w_list:
            band_chunks: List[np.ndarray] = []
            for _, band_arr in b_data:
                chunk = np.asarray(band_arr[:, s:e], dtype=np.float32, copy=False)
                m = float(chunk.mean())
                sd = float(chunk.std()) + 1e-6
                chunk = (chunk - m) / sd
                band_chunks.append(chunk)

            x_cat = np.concatenate(band_chunks, axis=1)
            if est_mode == "sample":
                cov = _cov_empirical(x_cat, float(spd_eps))
            elif est_mode == "oas":
                cov = _cov_oas(x_cat, float(spd_eps))
            elif est_mode == "ledoitwolf":
                cov = _cov_lw(x_cat, float(spd_eps))
            else:
                raise ValueError(f"Unknown cov estimator: {est_mode}")

            cov_list.append(np.asarray(cov, dtype=np.float32))
            y_list.append(int(t["label"]))
            tid_list.append(str(t["trial_id_str"]))

        if progress_prefix and progress_every > 0 and (
            trial_idx % int(progress_every) == 0 or trial_idx == total_trials
        ):
            elapsed = time.time() - t_start
            print(
                f"{progress_prefix} progress "
                f"{trial_idx}/{total_trials} trials "
                f"windows={total_windows} elapsed={elapsed:.1f}s",
                flush=True,
            )

    if progress_prefix:
        print(
            f"{progress_prefix} done trials={total_trials} windows={total_windows} "
            f"elapsed={time.time() - t_start:.1f}s",
            flush=True,
        )

    return np.asarray(cov_list, dtype=np.float32), np.asarray(y_list), tid_list


def apply_logcenter(covs_train: np.ndarray, covs_test: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    log_train = np.asarray([logm_spd(c, float(eps)) for c in covs_train], dtype=np.float32)
    mean_log = np.mean(log_train, axis=0)
    start_train = log_train - mean_log
    log_test = np.asarray([logm_spd(c, float(eps)) for c in covs_test], dtype=np.float32)
    start_test = log_test - mean_log
    return start_train, start_test


def covs_to_features(covs: np.ndarray) -> np.ndarray:
    return np.asarray([vec_utri(c) for c in np.asarray(covs, dtype=np.float32)], dtype=np.float32)
