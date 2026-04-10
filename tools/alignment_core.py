from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

from datasets.seed_raw_cnt import build_eeg62_view, load_one_cnt
from datasets.seed_raw_trials import build_trial_index
from manifold_raw.features import bandpass, parse_band_spec, window_slices
from tools.official_de_reader import BAND_ORDER, _normalize_trial_array, _resolve_mat_by_session
from tools.smooth_timeseries import smooth_ema, smooth_kalman_1d, smooth_moving_average


class AlignmentError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    a = a[:n]
    b = b[:n]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def best_lag_corr(a: np.ndarray, b: np.ndarray, max_lag: int) -> Tuple[float, int]:
    best_r = float("-inf")
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            aa = a[lag:]
            bb = b[:-lag]
        elif lag < 0:
            aa = a[:lag]
            bb = b[-lag:]
        else:
            aa = a
            bb = b
        n = min(aa.size, bb.size)
        if n == 0:
            continue
        r = pearson_r(aa[:n], bb[:n])
        if np.isnan(r):
            continue
        if r > best_r:
            best_r = r
            best_lag = lag
    return best_r, best_lag


def apply_smooth(series: np.ndarray, mode: str, param: Optional[float | int]) -> np.ndarray:
    mode = (mode or "none").strip().lower()
    if mode == "none":
        return series.astype(np.float64, copy=False)
    if mode == "ma":
        if param is None:
            raise ValueError("ma smoothing requires window param")
        return smooth_moving_average(series, int(param))
    if mode == "ema":
        if param is None:
            raise ValueError("ema smoothing requires alpha param")
        return smooth_ema(series, float(param))
    if mode == "kalman":
        if param is None:
            raise ValueError("kalman smoothing requires q_over_r param")
        return smooth_kalman_1d(series, float(param))
    raise ValueError(f"Unknown smooth mode: {mode}")


@lru_cache(maxsize=128)
def _load_mat_cached(mat_path: str):
    import scipy.io

    return scipy.io.loadmat(mat_path)


def load_official_de_series(
    *,
    mat_root: str,
    subject: int,
    session: int,
    trial: int,
    band: str,
    target_key: str = "de_LDS",
) -> np.ndarray:
    band = band.strip().lower()
    if band not in BAND_ORDER:
        raise ValueError(f"Unknown band: {band}")
    mat_path = _resolve_mat_by_session(Path(mat_root), subject, session)
    mat = _load_mat_cached(str(mat_path))
    key_name = f"{target_key}{int(trial)}"
    if key_name not in mat:
        raise AlignmentError("missing_key", f"{key_name} not found in {mat_path}")
    arr = _normalize_trial_array(mat[key_name])
    band_idx = BAND_ORDER.index(band)
    curve = arr[:, :, band_idx].mean(axis=0).astype(np.float64)
    if not np.isfinite(curve).all():
        raise AlignmentError("non_finite_official", f"non-finite in official curve {key_name}")
    return curve


@lru_cache(maxsize=512)
def _trial_list_cached(cnt_path: str, time_unit: str):
    time_txt = os.path.join(os.path.dirname(cnt_path), "time.txt")
    if not os.path.isfile(time_txt):
        time_txt = os.path.join("data", "SEED", "SEED_EEG", "time.txt")
    stim_xlsx = os.path.join("data", "SEED", "SEED_EEG", "SEED_stimulation.xlsx")
    trials = build_trial_index(cnt_path, time_txt, stim_xlsx, time_unit=time_unit)
    return tuple(trials)


def get_trial_meta(
    *,
    cnt_path: str,
    trial_input: int,
    time_unit: str,
):
    trials = _trial_list_cached(cnt_path, time_unit)
    trial_zero = int(trial_input) - 1
    if trial_zero < 0 or trial_zero >= len(trials):
        raise AlignmentError("trial_out_of_range", f"trial {trial_input} out of range for {cnt_path}")
    return trials[trial_zero]


def compute_raw_de_proxy_series(
    *,
    cnt_path: str,
    trial_input: int,
    band: str,
    window_sec: float,
    hop_sec: float,
    offset_sec: float,
    smooth_mode: str,
    smooth_param: Optional[float | int],
    time_unit: str = "samples@1000",
    duration_sec: Optional[float] = None,
    eps_var: float = 1e-12,
    raw62=None,
    sfreq: Optional[float] = None,
    trial_meta=None,
    locs_path: str = "data/SEED/channel_62_pos.locs",
) -> np.ndarray:
    band = band.strip().lower()
    if raw62 is None or sfreq is None:
        raw = load_one_cnt(cnt_path, preload=False)
        raw62, _ = build_eeg62_view(raw, locs_path=locs_path)
        sfreq = float(raw62.info["sfreq"])
    else:
        sfreq = float(sfreq)

    if trial_meta is None:
        trial_meta = get_trial_meta(
            cnt_path=cnt_path,
            trial_input=trial_input,
            time_unit=time_unit,
        )

    start_sec = float(trial_meta.t_start_s) + float(offset_sec)
    if duration_sec is None:
        end_sec = float(trial_meta.t_end_s) + float(offset_sec)
    else:
        end_sec = start_sec + float(duration_sec)

    start_idx = int(round(start_sec * sfreq))
    end_idx = int(round(end_sec * sfreq))
    if start_idx < 0 or end_idx <= start_idx or end_idx > int(raw62.n_times):
        raise AlignmentError(
            "segment_oob",
            f"segment out of bounds start={start_idx} end={end_idx} n_times={raw62.n_times}",
        )

    seg = raw62.get_data(start=start_idx, stop=end_idx).astype(np.float32)
    band_ranges = {"delta": (1.0, 4.0), "gamma": (31.0, 50.0)}
    if band not in band_ranges:
        raise AlignmentError("band_range", f"unsupported band for raw proxy: {band}")
    lo, hi = band_ranges[band]
    band_spec = parse_band_spec(f"{band}:{lo}-{hi}")
    if not band_spec:
        raise AlignmentError("band_spec", f"failed to parse band spec for {band}")
    band_obj = band_spec[0]
    band_data = bandpass(seg, sfreq, band_obj)
    slices = window_slices(seg.shape[1], sfreq, window_sec, hop_sec)
    if not slices:
        raise AlignmentError("empty_windows", "no windows generated")
    vals = []
    for s, e in slices:
        win = band_data[:, s:e]
        var = np.var(win, axis=1)
        logvar = np.log(var + float(eps_var))
        vals.append(float(np.mean(logvar)))
    curve = np.asarray(vals, dtype=np.float64)
    curve = apply_smooth(curve, smooth_mode, smooth_param)
    if not np.isfinite(curve).all():
        raise AlignmentError("non_finite_raw", "non-finite raw curve")
    return curve


def scan_offsets(
    *,
    offsets: Iterable[float],
    cnt_path: str,
    trial_input: int,
    band: str,
    window_sec: float,
    hop_sec: float,
    time_unit: str,
    smooth_mode: str,
    smooth_param: Optional[float | int],
    duration_sec: float,
    eps_var: float,
    official_curve: np.ndarray,
    lag_max: int = 0,
    raw62=None,
    sfreq: Optional[float] = None,
    trial_meta=None,
) -> Tuple[float, float, int, list]:
    best_r = float("-inf")
    best_offset = None
    best_lag = 0
    errors = []
    for offset in offsets:
        try:
            raw_curve = compute_raw_de_proxy_series(
                cnt_path=cnt_path,
                trial_input=trial_input,
                band=band,
                window_sec=window_sec,
                hop_sec=hop_sec,
                offset_sec=float(offset),
                smooth_mode=smooth_mode,
                smooth_param=smooth_param,
                time_unit=time_unit,
                duration_sec=duration_sec,
                eps_var=eps_var,
                raw62=raw62,
                sfreq=sfreq,
                trial_meta=trial_meta,
            )
        except AlignmentError as exc:
            errors.append({"offset": float(offset), "code": exc.code})
            continue
        if lag_max > 0:
            r, lag = best_lag_corr(official_curve, raw_curve, int(lag_max))
        else:
            r = pearson_r(official_curve, raw_curve)
            lag = 0
        if np.isnan(r):
            errors.append({"offset": float(offset), "code": "nan_r"})
            continue
        if r > best_r:
            best_r = float(r)
            best_offset = float(offset)
            best_lag = int(lag)
    return best_r, best_offset, best_lag, errors
