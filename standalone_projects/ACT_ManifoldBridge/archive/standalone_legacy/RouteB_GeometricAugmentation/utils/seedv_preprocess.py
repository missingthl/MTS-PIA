#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
seedv_preprocess.py — SEED-V 预处理与划分工具。

数据来源：
- DE 特征：data/SEED_V/EEG_DE_features/*.npz
- RAW EEG：data/SEED_V/EEG_raw/*.cnt + trial_start_end_timestamp.txt
npz 内含 pickled dict：
  - data[k]  : trial 的 DE 特征 (T, 310)
  - label[k] : trial 标签（长度为 T 或标量）

提供：
- 试次迭代器（trial-level）
- 按 trial/session/subject 的 fold 划分（sample-level / trial-level）
"""

from __future__ import annotations

import os
import pickle
import re
import warnings
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np

# ===== 路径配置 =====
def _resolve_seedv_root() -> str:
    candidates = [
        "./data/SEED_V",
        "./data/SEED-V",
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


SEEDV_ROOT = _resolve_seedv_root()
SEEDV_DE_ROOT = os.path.join(SEEDV_ROOT, "EEG_DE_features")
SEEDV_RAW_ROOT = os.path.join(SEEDV_ROOT, "EEG_raw")
SEEDV_TRIAL_TS_PATH = os.path.join(SEEDV_ROOT, "trial_start_end_timestamp.txt")
SEEDV_CHANNELS_PATH = os.path.join(SEEDV_ROOT, "channel_62_pos.locs")
SEEDV_RAW_CACHE_ROOT = os.path.join(SEEDV_ROOT, "cache", "raw_cov")
RAW_COV_CACHE_VERSION = "v6"
RAW_FBCOV_CACHE_VERSION = "v1"
RAW_FB_DEFAULT_BANDS = [(4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]

# Silence CNT meas_date warnings (metadata only).
warnings.filterwarnings("ignore", message=".*Could not parse meas date.*", category=RuntimeWarning)

# Raw EEG -> Covariance defaults (RA pipeline)
RAW_ZSCORE = True
RAW_ZSCORE_EPS = 1e-6
RAW_SHRINKAGE = 0.0
RAW_TRACE_NORMALIZE = True
RAW_COV_EPS = 1e-6
RAW_COV_CACHE_PARAMS = {
    "zscore": RAW_ZSCORE,
    "zscore_eps": RAW_ZSCORE_EPS,
    "shrinkage": RAW_SHRINKAGE,
    "trace_normalize": RAW_TRACE_NORMALIZE,
    "cov_eps": RAW_COV_EPS,
    "repr": "cov",
}

# DE -> LDS defaults (random walk smoother)
DE_LDS_DEFAULT_Q = 1e-3
DE_LDS_DEFAULT_R = 1.0
DE_LDS_DEFAULT_METHOD = "fixed"
DE_LDS_DEFAULT_EM_ITERS = 10
DE_LDS_DEFAULT_EM_TOL = 1e-4
DE_LDS_DEFAULT_EM_MIN_VAR = 1e-6
DE_LDS_DEFAULT_EM_A_MIN = 0.0
DE_LDS_DEFAULT_EM_A_MAX = 0.999


def _raw_cov_params(
    shrinkage: float,
    *,
    repr_name: str = "cov",
    bands: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, object]:
    params: Dict[str, object] = dict(RAW_COV_CACHE_PARAMS)
    params["shrinkage"] = float(shrinkage)
    params["repr"] = repr_name
    if bands is not None:
        params["bands"] = [tuple(map(float, b)) for b in bands]
    return params


def _seedv_trial_id(subject: str, session_idx: int, trial_idx: int) -> str:
    return f"{subject}_s{session_idx}_t{trial_idx}"


def _parse_trial_id(trial_id: str) -> Tuple[str, int, int]:
    """
    Parse trial_id formatted as "{subject}_s{session}_t{trial}".
    Returns (subject, session_idx, trial_idx).
    """
    tid = str(trial_id)
    if "_s" not in tid or "_t" not in tid:
        raise ValueError(f"Invalid trial_id format: {trial_id}")
    subject, rest = tid.rsplit("_s", 1)
    session_str, trial_str = rest.split("_t", 1)
    return subject, int(session_str), int(trial_str)


def _load_seedv_subject_npz(npz_path: str):
    """
    读取 SEED-V 原始 DE 特征 .npz（data/label 为 pickled dict）。
    返回 (data_dict, label_dict)。
    """
    npz = np.load(npz_path, allow_pickle=True)
    data = pickle.loads(npz["data"].item())
    label = pickle.loads(npz["label"].item())
    return data, label


def _parse_seedv_raw_filename(fname: str) -> Tuple[str, int]:
    """
    Parse raw EEG filename like "1_1_20180804.cnt" or "7_1_20180411_repaired.cnt".
    Returns (subject, session_idx) where session_idx is 0-based.
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid raw EEG filename: {fname}")
    subject = parts[0]
    session_str = parts[1]
    if not session_str.isdigit():
        raise ValueError(f"Invalid session token in filename: {fname}")
    session_idx = int(session_str) - 1
    if session_idx not in {0, 1, 2}:
        raise ValueError(f"Unexpected session_idx={session_idx} from filename: {fname}")
    return subject, session_idx


def _load_seedv_trial_timestamps(ts_path: str = SEEDV_TRIAL_TS_PATH) -> Dict[int, List[Tuple[int, int]]]:
    """
    Parse trial start/end timestamps. Returns {session_idx: [(start_s, end_s), ...]}.
    """
    if not os.path.isfile(ts_path):
        raise FileNotFoundError(f"SEED-V trial timestamps not found: {ts_path}")
    sessions: Dict[int, Dict[str, List[int]]] = {}
    current_session: Optional[int] = None
    with open(ts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("session"):
                nums = re.findall(r"\d+", line)
                if not nums:
                    raise ValueError(f"Invalid session header: {line}")
                current_session = int(nums[0]) - 1
                sessions[current_session] = {}
                continue
            if current_session is None:
                continue
            if line.startswith("start_second"):
                sessions[current_session]["start"] = [int(v) for v in re.findall(r"\d+", line)]
            elif line.startswith("end_second"):
                sessions[current_session]["end"] = [int(v) for v in re.findall(r"\d+", line)]

    out: Dict[int, List[Tuple[int, int]]] = {}
    for session_idx, payload in sessions.items():
        start = payload.get("start", [])
        end = payload.get("end", [])
        if len(start) != len(end):
            raise ValueError(f"Timestamp length mismatch for session {session_idx}: {len(start)} vs {len(end)}")
        if len(start) != 15:
            raise ValueError(f"Unexpected trial count for session {session_idx}: {len(start)}")
        out[session_idx] = list(zip(start, end))
    if set(out.keys()) != {0, 1, 2}:
        raise ValueError(f"Missing sessions in trial timestamps: {sorted(out.keys())}")
    return out


def _load_seedv_label_map(subject: str, seedv_de_root: str = SEEDV_DE_ROOT) -> Dict[Tuple[int, int], int]:
    """
    Load trial labels from DE features and map to (session_idx, trial_idx) -> label.
    """
    npz_path = os.path.join(seedv_de_root, f"{subject}_123.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Missing SEED-V DE labels for subject {subject}: {npz_path}")
    _data, label = _load_seedv_subject_npz(npz_path)
    label_map: Dict[Tuple[int, int], int] = {}
    for k in sorted(label.keys()):
        session_idx = k // 15
        trial_idx = k % 15
        y_arr = np.asarray(label[k]).astype(int).ravel()
        if y_arr.size == 0:
            raise ValueError(f"Empty label for subject {subject}, key {k}")
        y_val = int(np.unique(y_arr)[0])
        label_map[(session_idx, trial_idx)] = y_val
    return label_map


def _lds_smooth_sequence(X: np.ndarray, q: float, r: float) -> np.ndarray:
    """
    Random-walk LDS smoother for DE sequences.
    Model: x_t = x_{t-1} + w_t, y_t = x_t + v_t, Q=qI, R=rI.
    Applies RTS smoothing per feature (independent dims).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"LDS expects 2D input (T,D), got {X.shape}")
    T, D = X.shape
    if T <= 1:
        return X.copy()
    if q <= 0.0 or r <= 0.0:
        raise ValueError(f"LDS requires q>0 and r>0, got q={q}, r={r}")

    x_filt = np.zeros((T, D), dtype=np.float64)
    P_filt = np.zeros((T, D), dtype=np.float64)

    x_prev = X[0].copy()
    P_prev = np.full((D,), r, dtype=np.float64)
    x_filt[0] = x_prev
    P_filt[0] = P_prev

    for t in range(1, T):
        P_pred = P_prev + q
        K = P_pred / (P_pred + r)
        x_pred = x_prev
        x_cur = x_pred + K * (X[t] - x_pred)
        P_cur = (1.0 - K) * P_pred
        x_filt[t] = x_cur
        P_filt[t] = P_cur
        x_prev = x_cur
        P_prev = P_cur

    x_smooth = np.zeros_like(x_filt)
    P_smooth = np.zeros_like(P_filt)
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]
    for t in range(T - 2, -1, -1):
        P_pred = P_filt[t] + q
        C = P_filt[t] / P_pred
        x_smooth[t] = x_filt[t] + C * (x_smooth[t + 1] - x_filt[t])
        P_smooth[t] = P_filt[t] + C * (P_smooth[t + 1] - P_pred) * C

    return x_smooth


def _lds_em_smooth_sequence(
    X: np.ndarray,
    *,
    q_init: float,
    r_init: float,
    n_iter: int = DE_LDS_DEFAULT_EM_ITERS,
    tol: float = DE_LDS_DEFAULT_EM_TOL,
    min_var: float = DE_LDS_DEFAULT_EM_MIN_VAR,
    a_min: float = DE_LDS_DEFAULT_EM_A_MIN,
    a_max: float = DE_LDS_DEFAULT_EM_A_MAX,
) -> np.ndarray:
    """
    EM-estimated LDS smoother (diagonal A/Q/R, C=I) for DE sequences.
    Treats each feature dimension independently and estimates A/Q/R via EM.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"LDS expects 2D input (T,D), got {X.shape}")
    T, D = X.shape
    if T <= 1:
        return X.copy()
    if q_init <= 0.0 or r_init <= 0.0:
        raise ValueError(f"LDS EM requires q>0 and r>0, got q={q_init}, r={r_init}")
    if n_iter < 1:
        return X.copy()

    A = np.full((D,), 1.0, dtype=np.float64)
    Q = np.full((D,), float(q_init), dtype=np.float64)
    R = np.full((D,), float(r_init), dtype=np.float64)

    for _ in range(n_iter):
        # --- Kalman filter ---
        x_filt = np.zeros((T, D), dtype=np.float64)
        P_filt = np.zeros((T, D), dtype=np.float64)
        x_pred = np.zeros((T, D), dtype=np.float64)
        P_pred = np.zeros((T, D), dtype=np.float64)

        x_filt[0] = X[0]
        P_filt[0] = R
        x_pred[0] = x_filt[0]
        P_pred[0] = P_filt[0]

        for t in range(1, T):
            x_pred[t] = A * x_filt[t - 1]
            P_pred[t] = A * A * P_filt[t - 1] + Q
            denom = P_pred[t] + R
            denom = np.where(denom < min_var, min_var, denom)
            K = P_pred[t] / denom
            x_filt[t] = x_pred[t] + K * (X[t] - x_pred[t])
            P_filt[t] = (1.0 - K) * P_pred[t]

        # --- RTS smoother ---
        x_smooth = np.zeros_like(x_filt)
        P_smooth = np.zeros_like(P_filt)
        x_smooth[-1] = x_filt[-1]
        P_smooth[-1] = P_filt[-1]

        J = np.zeros((T - 1, D), dtype=np.float64)
        for t in range(T - 2, -1, -1):
            denom = P_pred[t + 1]
            denom = np.where(denom < min_var, min_var, denom)
            J[t] = (P_filt[t] * A) / denom
            x_smooth[t] = x_filt[t] + J[t] * (x_smooth[t + 1] - x_pred[t + 1])
            P_smooth[t] = P_filt[t] + J[t] * J[t] * (P_smooth[t + 1] - P_pred[t + 1])

        # Lag-one covariance approximation (diagonal)
        P_cross = J * P_smooth[1:]

        Exx = np.sum(P_smooth + x_smooth * x_smooth, axis=0)
        Exx1 = np.sum(P_cross + x_smooth[1:] * x_smooth[:-1], axis=0)
        Ex1x1 = np.sum(P_smooth[:-1] + x_smooth[:-1] * x_smooth[:-1], axis=0)
        denom = np.where(Ex1x1 < min_var, min_var, Ex1x1)
        A_new = Exx1 / denom
        A_new = np.clip(A_new, a_min, a_max)

        diff = x_smooth[1:] - A_new * x_smooth[:-1]
        Q_new = np.mean(
            P_smooth[1:]
            + x_smooth[1:] * x_smooth[1:]
            - 2.0 * A_new * (P_cross + x_smooth[1:] * x_smooth[:-1])
            + (A_new * A_new) * (P_smooth[:-1] + x_smooth[:-1] * x_smooth[:-1]),
            axis=0,
        )
        Q_new = np.where(Q_new < min_var, min_var, Q_new)

        R_new = np.mean((X - x_smooth) ** 2 + P_smooth, axis=0)
        R_new = np.where(R_new < min_var, min_var, R_new)

        delta = max(
            float(np.max(np.abs(A_new - A) / (np.abs(A) + 1e-8))),
            float(np.max(np.abs(Q_new - Q) / (np.abs(Q) + 1e-8))),
            float(np.max(np.abs(R_new - R) / (np.abs(R) + 1e-8))),
        )
        A, Q, R = A_new, Q_new, R_new
        if delta < tol:
            break

    return x_smooth


def _apply_de_lds(
    data: Dict[int, np.ndarray],
    *,
    level: str,
    q: float,
    r: float,
    method: str = DE_LDS_DEFAULT_METHOD,
    em_iters: int = DE_LDS_DEFAULT_EM_ITERS,
    em_tol: float = DE_LDS_DEFAULT_EM_TOL,
    em_min_var: float = DE_LDS_DEFAULT_EM_MIN_VAR,
    em_a_min: float = DE_LDS_DEFAULT_EM_A_MIN,
    em_a_max: float = DE_LDS_DEFAULT_EM_A_MAX,
) -> Dict[int, np.ndarray]:
    level = (level or "session").lower()
    if level not in {"session", "trial"}:
        raise ValueError(f"Unknown LDS level: {level}")
    method = (method or "fixed").lower()
    if method not in {"fixed", "em"}:
        raise ValueError(f"Unknown LDS method: {method}")

    def _smooth_seq(seq: np.ndarray) -> np.ndarray:
        if method == "fixed":
            return _lds_smooth_sequence(seq, q, r)
        return _lds_em_smooth_sequence(
            seq,
            q_init=q,
            r_init=r,
            n_iter=em_iters,
            tol=em_tol,
            min_var=em_min_var,
            a_min=em_a_min,
            a_max=em_a_max,
        )

    if level == "trial":
        return {k: _smooth_seq(np.asarray(data[k])) for k in sorted(data.keys())}

    out: Dict[int, np.ndarray] = {}
    keys_sorted = sorted(data.keys())
    for session_idx in range(3):
        trial_keys = [k for k in keys_sorted if k // 15 == session_idx]
        if len(trial_keys) != 15:
            raise ValueError(f"Unexpected trial count for session {session_idx}: {len(trial_keys)}")
        trials = [np.asarray(data[k], dtype=np.float64) for k in trial_keys]
        lengths = [t.shape[0] for t in trials]
        if not lengths:
            continue
        X_cat = np.vstack(trials)
        X_smooth = _lds_smooth_sequence(X_cat, q, r)
        splits = np.split(X_smooth, np.cumsum(lengths)[:-1])
        for k, Xt in zip(trial_keys, splits):
            out[k] = Xt
    return out


def _raw_trial_cov(
    X: np.ndarray,
    *,
    zscore: bool = RAW_ZSCORE,
    zscore_eps: float = RAW_ZSCORE_EPS,
    shrinkage: float = RAW_SHRINKAGE,
    trace_normalize: bool = RAW_TRACE_NORMALIZE,
    cov_eps: float = RAW_COV_EPS,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"raw trial must be 2D (T,C), got {X.shape}")
    if zscore:
        mu = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std = np.where(std < zscore_eps, 1.0, std)
        X = (X - mu) / std
    T = X.shape[0]
    if T < 2:
        d = X.shape[1]
        C = np.eye(d, dtype=np.float64)
    else:
        C = (X.T @ X) / max(1, T - 1)
    if shrinkage > 0.0:
        d = C.shape[0]
        tr = float(np.trace(C))
        C = (1.0 - shrinkage) * C + shrinkage * (tr / d) * np.eye(d)
    if trace_normalize:
        tr = float(np.trace(C))
        if tr > 0:
            C = C / tr
    if cov_eps > 0:
        d = C.shape[0]
        C = C + cov_eps * np.eye(d)
    return (C + C.T) * 0.5


def _raw_cov_cache_path(
    subject: str,
    session_idx: int,
    cache_root: str = SEEDV_RAW_CACHE_ROOT,
    version: str = RAW_COV_CACHE_VERSION,
    repr_name: str = "cov",
) -> str:
    sess = session_idx + 1
    fname = f"{subject}_s{sess}_{repr_name}_{version}.npz"
    return os.path.join(cache_root, fname)


def _load_raw_cov_cache(path: str):
    npz = np.load(path, allow_pickle=True)
    covs = np.asarray(npz["covs"])
    labels = np.asarray(npz["labels"]).astype(int)
    meta = None
    if "meta" in npz.files:
        meta = npz["meta"].item()
    return covs, labels, meta


def _raw_cov_cache_valid(
    meta: Optional[dict],
    expected_params: Optional[dict] = None,
    expected_version: Optional[str] = None,
) -> bool:
    if not meta:
        return False
    if expected_version is None:
        expected_version = RAW_COV_CACHE_VERSION
    if meta.get("version") != expected_version:
        return False
    params = meta.get("params", {})
    if expected_params is None:
        expected_params = RAW_COV_CACHE_PARAMS
    return params == expected_params


def _load_seedv_channel_names(path: str = SEEDV_CHANNELS_PATH) -> List[str]:
    """
    Load SEED-V 62-channel list from channel_62_pos.locs.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"SEED-V channel list not found: {path}")
    names: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            names.append(parts[-1])
    if len(names) != 62:
        raise ValueError(f"Unexpected channel count in {path}: {len(names)}")
    return names


def _normalize_ch_name(name: str) -> str:
    n = str(name).strip().lower()
    if n.startswith("eeg"):
        n = n[3:]
    # strip non-alphanumeric
    n = "".join(ch for ch in n if ch.isalnum())
    if n.endswith("ref"):
        n = n[:-3]
    alias_map = {
        "m1": "cb1",
        "a1": "cb1",
        "tp9": "cb1",
        "lpa": "cb1",
        "m2": "cb2",
        "a2": "cb2",
        "tp10": "cb2",
        "rpa": "cb2",
    }
    n = alias_map.get(n, n)
    return n


def _normalize_bands(
    bands: Optional[List[Tuple[float, float]]],
) -> List[Tuple[float, float]]:
    if not bands:
        return RAW_FB_DEFAULT_BANDS
    out: List[Tuple[float, float]] = []
    for lo, hi in bands:
        lo_f = float(lo)
        hi_f = float(hi)
        if lo_f <= 0 or hi_f <= 0 or lo_f >= hi_f:
            raise ValueError(f"Invalid band: {(lo, hi)}")
        out.append((lo_f, hi_f))
    return out


def _bandpass_filter(X: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    try:
        import mne
    except ImportError as exc:
        raise ImportError(
            "mne is required to filter SEED-V raw CNT files. Install via `pip install mne`."
        ) from exc
    X = np.asarray(X, dtype=np.float64)
    Xf = mne.filter.filter_data(X.T, sfreq, l_freq=lo, h_freq=hi, verbose="ERROR")
    return Xf.T


def _match_seedv_channels(raw, channel_names: List[str]):
    """
    Match channel indices by name (normalized).
    Returns (picks, target_positions, missing).
    """
    name_to_idx = {}
    for i, name in enumerate(raw.ch_names):
        key = _normalize_ch_name(name)
        if key not in name_to_idx:
            name_to_idx[key] = i
    picks: List[int] = []
    positions: List[int] = []
    missing: List[Tuple[str, int, str]] = []
    used_keys = set()
    for pos, name in enumerate(channel_names):
        key = _normalize_ch_name(name)
        idx = name_to_idx.get(key)
        if idx is None:
            missing.append((name, pos, key))
        else:
            picks.append(idx)
            positions.append(pos)
            used_keys.add(key)

    if missing:
        unresolved: List[str] = []
        for name, pos, key in missing:
            candidates = [k for k in name_to_idx.keys() if k.startswith(key) and k not in used_keys]
            if len(candidates) == 1:
                k = candidates[0]
                picks.append(name_to_idx[k])
                positions.append(pos)
                used_keys.add(k)
            else:
                unresolved.append(name)
        return picks, positions, unresolved

    return picks, positions, []


def _build_raw_cov_cache(
    subject: str,
    session_idx: int,
    raw_path: str,
    label_map: Dict[Tuple[int, int], int],
    ts_by_session: Dict[int, List[Tuple[int, int]]],
    cache_root: str,
    cache_mode: str,
    channel_policy: str,
    raw_shrinkage: float,
    cov_params: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import mne
    except ImportError as exc:
        raise ImportError(
            "mne is required to load SEED-V raw CNT files. Install via `pip install mne`."
        ) from exc

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Could not parse meas date.*", category=RuntimeWarning)
        raw = mne.io.read_raw_cnt(raw_path, preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    channel_names = _load_seedv_channel_names()
    picks, positions, missing = _match_seedv_channels(raw, channel_names)
    if missing:
        if channel_policy == "fallback":
            picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, misc=False)
            positions = None
        elif channel_policy == "pad":
            print(
                f"[raw][pad] {subject} s{session_idx + 1} missing channels: {missing[:8]}"
            )
            positions = positions
        else:
            raise ValueError(
                f"Missing {len(missing)} SEED-V channels (first 8): {missing[:8]}. "
                f"raw has {len(raw.ch_names)} channels; first 8: {raw.ch_names[:8]}"
            )
    n_samples = raw.n_times

    bounds = ts_by_session.get(session_idx)
    if bounds is None:
        raise ValueError(f"Missing trial timestamps for session {session_idx}")

    covs: List[np.ndarray] = []
    labels: List[int] = []
    for trial_idx, (start_s, end_s) in enumerate(bounds):
        start_idx = int(round(start_s * sfreq))
        end_idx = int(round(end_s * sfreq))
        start_idx = max(0, min(start_idx, n_samples - 1))
        end_idx = max(start_idx + 1, min(end_idx, n_samples))
        seg = raw.get_data(picks=picks, start=start_idx, stop=end_idx)
        X = seg.T
        if positions is not None and len(positions) != X.shape[1]:
            X_full = np.zeros((X.shape[0], len(channel_names)), dtype=X.dtype)
            X_full[:, positions] = X
            X = X_full
        C = _raw_trial_cov(X, shrinkage=raw_shrinkage)
        covs.append(C)
        y_val = label_map.get((session_idx, trial_idx))
        if y_val is None:
            raise ValueError(
                f"Missing label for subject {subject}, session {session_idx}, trial {trial_idx}"
            )
        labels.append(int(y_val))

    covs_arr = np.stack(covs, axis=0)
    labels_arr = np.asarray(labels, dtype=int)
    if cache_mode != "off":
        os.makedirs(cache_root, exist_ok=True)
        if cov_params is None:
            cov_params = _raw_cov_params(raw_shrinkage, repr_name="cov")
        meta = {
            "version": RAW_COV_CACHE_VERSION,
            "params": cov_params,
            "subject": subject,
            "session_idx": session_idx,
            "n_trials": len(labels),
        }
        cache_path = _raw_cov_cache_path(
            subject,
            session_idx,
            cache_root=cache_root,
            version=RAW_COV_CACHE_VERSION,
            repr_name="cov",
        )
        np.savez_compressed(cache_path, covs=covs_arr, labels=labels_arr, meta=np.array([meta], dtype=object))
    return covs_arr, labels_arr


def _build_raw_fb_cov_cache(
    subject: str,
    session_idx: int,
    raw_path: str,
    label_map: Dict[Tuple[int, int], int],
    ts_by_session: Dict[int, List[Tuple[int, int]]],
    cache_root: str,
    cache_mode: str,
    channel_policy: str,
    raw_shrinkage: float,
    raw_bands: Optional[List[Tuple[float, float]]],
    cov_params: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import mne
    except ImportError as exc:
        raise ImportError(
            "mne is required to load SEED-V raw CNT files. Install via `pip install mne`."
        ) from exc

    bands = _normalize_bands(raw_bands)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Could not parse meas date.*", category=RuntimeWarning)
        raw = mne.io.read_raw_cnt(raw_path, preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    channel_names = _load_seedv_channel_names()
    picks, positions, missing = _match_seedv_channels(raw, channel_names)
    if missing:
        if channel_policy == "fallback":
            picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, misc=False)
            positions = None
        elif channel_policy == "pad":
            print(
                f"[raw][pad] {subject} s{session_idx + 1} missing channels: {missing[:8]}"
            )
            positions = positions
        else:
            raise ValueError(
                f"Missing {len(missing)} SEED-V channels (first 8): {missing[:8]}. "
                f"raw has {len(raw.ch_names)} channels; first 8: {raw.ch_names[:8]}"
            )
    n_samples = raw.n_times

    bounds = ts_by_session.get(session_idx)
    if bounds is None:
        raise ValueError(f"Missing trial timestamps for session {session_idx}")

    covs: List[np.ndarray] = []
    labels: List[int] = []
    for trial_idx, (start_s, end_s) in enumerate(bounds):
        start_idx = int(round(start_s * sfreq))
        end_idx = int(round(end_s * sfreq))
        start_idx = max(0, min(start_idx, n_samples - 1))
        end_idx = max(start_idx + 1, min(end_idx, n_samples))
        seg = raw.get_data(picks=picks, start=start_idx, stop=end_idx)
        X = seg.T
        if positions is not None and len(positions) != X.shape[1]:
            X_full = np.zeros((X.shape[0], len(channel_names)), dtype=X.dtype)
            X_full[:, positions] = X
            X = X_full
        band_covs: List[np.ndarray] = []
        for lo, hi in bands:
            Xb = _bandpass_filter(X, sfreq, lo, hi)
            C = _raw_trial_cov(Xb, shrinkage=raw_shrinkage)
            band_covs.append(C)
        covs.append(np.stack(band_covs, axis=0))
        y_val = label_map.get((session_idx, trial_idx))
        if y_val is None:
            raise ValueError(
                f"Missing label for subject {subject}, session {session_idx}, trial {trial_idx}"
            )
        labels.append(int(y_val))

    covs_arr = np.stack(covs, axis=0)
    labels_arr = np.asarray(labels, dtype=int)
    if cache_mode != "off":
        os.makedirs(cache_root, exist_ok=True)
        if cov_params is None:
            cov_params = _raw_cov_params(raw_shrinkage, repr_name="fb_cov", bands=bands)
        meta = {
            "version": RAW_FBCOV_CACHE_VERSION,
            "params": cov_params,
            "subject": subject,
            "session_idx": session_idx,
            "n_trials": len(labels),
        }
        cache_path = _raw_cov_cache_path(
            subject,
            session_idx,
            cache_root=cache_root,
            version=RAW_FBCOV_CACHE_VERSION,
            repr_name="fbcov",
        )
        np.savez_compressed(cache_path, covs=covs_arr, labels=labels_arr, meta=np.array([meta], dtype=object))
    return covs_arr, labels_arr


def _iter_seedv_raw_trials(
    seedv_raw_root: str = SEEDV_RAW_ROOT,
    seedv_de_root: str = SEEDV_DE_ROOT,
    ts_path: str = SEEDV_TRIAL_TS_PATH,
    raw_repr: str = "signal",
    raw_cache: str = "auto",
    raw_cache_root: str = SEEDV_RAW_CACHE_ROOT,
    raw_channel_policy: str = "strict",
    raw_shrinkage: Optional[float] = None,
    raw_bands: Optional[List[Tuple[float, float]]] = None,
) -> Iterable[Tuple[str, np.ndarray, int, int]]:
    """
    Iterate raw EEG trials. Returns (trial_id, X_trial[T,C], label, trial_idx).
    """
    if not os.path.isdir(seedv_raw_root):
        raise FileNotFoundError(f"SEED-V raw EEG root not found: {seedv_raw_root}")
    raw_repr = (raw_repr or "signal").lower()
    raw_cache = (raw_cache or "auto").lower()
    raw_channel_policy = (raw_channel_policy or "strict").lower()
    if raw_repr not in {"signal", "cov", "fb_cov"}:
        raise ValueError(f"Unknown raw_repr: {raw_repr}")
    if raw_cache not in {"auto", "off", "refresh"}:
        raise ValueError(f"Unknown raw_cache mode: {raw_cache}")
    if raw_channel_policy not in {"strict", "pad", "fallback"}:
        raise ValueError(f"Unknown raw_channel_policy: {raw_channel_policy}")
    if raw_shrinkage is None:
        raw_shrinkage = RAW_SHRINKAGE
    if not (0.0 <= float(raw_shrinkage) <= 1.0):
        raise ValueError("raw_shrinkage must be in [0, 1]")
    bands = None
    if raw_repr == "fb_cov":
        bands = _normalize_bands(raw_bands)
        cov_params = _raw_cov_params(float(raw_shrinkage), repr_name="fb_cov", bands=bands)
    else:
        cov_params = _raw_cov_params(float(raw_shrinkage), repr_name="cov")

    ts_by_session = _load_seedv_trial_timestamps(ts_path=ts_path)
    label_cache: Dict[str, Dict[Tuple[int, int], int]] = {}

    files = [f for f in os.listdir(seedv_raw_root) if f.endswith(".cnt")]
    files_by_key: Dict[Tuple[str, int], str] = {}
    for fname in files:
        subject, session_idx = _parse_seedv_raw_filename(fname)
        key = (subject, session_idx)
        if key not in files_by_key:
            files_by_key[key] = fname
            continue
        # Prefer non-repaired when both exist; repaired is a fallback.
        if "repaired" in files_by_key[key] and "repaired" not in fname:
            files_by_key[key] = fname

    def _sort_key(item):
        subject, session_idx = item
        return (_subject_sort_key(subject), session_idx)

    for subject, session_idx in sorted(files_by_key.keys(), key=_sort_key):
        if subject not in label_cache:
            label_cache[subject] = _load_seedv_label_map(subject, seedv_de_root=seedv_de_root)
        label_map = label_cache[subject]
        fname = files_by_key[(subject, session_idx)]
        raw_path = os.path.join(seedv_raw_root, fname)

        if raw_repr == "cov":
            covs = labels = None
            cache_path = _raw_cov_cache_path(
                subject,
                session_idx,
                cache_root=raw_cache_root,
                version=RAW_COV_CACHE_VERSION,
                repr_name="cov",
            )
            if raw_cache != "off" and os.path.isfile(cache_path) and raw_cache != "refresh":
                covs, labels, meta = _load_raw_cov_cache(cache_path)
                if not _raw_cov_cache_valid(
                    meta,
                    expected_params=cov_params,
                    expected_version=RAW_COV_CACHE_VERSION,
                ):
                    covs = labels = None
            if covs is None or labels is None:
                covs, labels = _build_raw_cov_cache(
                    subject,
                    session_idx,
                    raw_path,
                    label_map,
                    ts_by_session,
                    cache_root=raw_cache_root,
                    cache_mode=raw_cache,
                    channel_policy=raw_channel_policy,
                    raw_shrinkage=float(raw_shrinkage),
                    cov_params=cov_params,
                )
            for trial_idx in range(covs.shape[0]):
                trial_id = _seedv_trial_id(subject, session_idx, trial_idx)
                yield trial_id, covs[trial_idx], int(labels[trial_idx]), trial_idx
            continue
        if raw_repr == "fb_cov":
            covs = labels = None
            cache_path = _raw_cov_cache_path(
                subject,
                session_idx,
                cache_root=raw_cache_root,
                version=RAW_FBCOV_CACHE_VERSION,
                repr_name="fbcov",
            )
            if raw_cache != "off" and os.path.isfile(cache_path) and raw_cache != "refresh":
                covs, labels, meta = _load_raw_cov_cache(cache_path)
                if not _raw_cov_cache_valid(
                    meta,
                    expected_params=cov_params,
                    expected_version=RAW_FBCOV_CACHE_VERSION,
                ):
                    covs = labels = None
            if covs is None or labels is None:
                covs, labels = _build_raw_fb_cov_cache(
                    subject,
                    session_idx,
                    raw_path,
                    label_map,
                    ts_by_session,
                    cache_root=raw_cache_root,
                    cache_mode=raw_cache,
                    channel_policy=raw_channel_policy,
                    raw_shrinkage=float(raw_shrinkage),
                    raw_bands=bands,
                    cov_params=cov_params,
                )
            for trial_idx in range(covs.shape[0]):
                trial_id = _seedv_trial_id(subject, session_idx, trial_idx)
                yield trial_id, covs[trial_idx], int(labels[trial_idx]), trial_idx
            continue

        # signal path (no cache)
        try:
            import mne
        except ImportError as exc:
            raise ImportError(
                "mne is required to load SEED-V raw CNT files. Install via `pip install mne`."
            ) from exc
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Could not parse meas date.*", category=RuntimeWarning)
            raw = mne.io.read_raw_cnt(raw_path, preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        channel_names = _load_seedv_channel_names()
        picks, positions, missing = _match_seedv_channels(raw, channel_names)
        if missing:
            if raw_channel_policy == "fallback":
                picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, misc=False)
                positions = None
            elif raw_channel_policy == "pad":
                positions = positions
            else:
                raise ValueError(
                    f"Missing {len(missing)} SEED-V channels (first 8): {missing[:8]}. "
                    f"raw has {len(raw.ch_names)} channels; first 8: {raw.ch_names[:8]}"
                )
        n_samples = raw.n_times
        bounds = ts_by_session.get(session_idx)
        if bounds is None:
            raise ValueError(f"Missing trial timestamps for session {session_idx}")
        for trial_idx, (start_s, end_s) in enumerate(bounds):
            start_idx = int(round(start_s * sfreq))
            end_idx = int(round(end_s * sfreq))
            start_idx = max(0, min(start_idx, n_samples - 1))
            end_idx = max(start_idx + 1, min(end_idx, n_samples))
            seg = raw.get_data(picks=picks, start=start_idx, stop=end_idx)
            X = seg.T
            if positions is not None and len(positions) != X.shape[1]:
                X_full = np.zeros((X.shape[0], len(channel_names)), dtype=X.dtype)
                X_full[:, positions] = X
                X = X_full
            y_val = label_map.get((session_idx, trial_idx))
            if y_val is None:
                raise ValueError(
                    f"Missing label for subject {subject}, session {session_idx}, trial {trial_idx}"
                )
            trial_id = _seedv_trial_id(subject, session_idx, trial_idx)
            yield trial_id, np.asarray(X, dtype=np.float32), int(y_val), trial_idx


def iter_seedv_trials(
    source: str = "de",
    seedv_de_root: str = SEEDV_DE_ROOT,
    seedv_raw_root: str = SEEDV_RAW_ROOT,
    raw_repr: str = "signal",
    raw_cache: str = "auto",
    raw_channel_policy: str = "strict",
    raw_shrinkage: Optional[float] = None,
    raw_bands: Optional[List[Tuple[float, float]]] = None,
    de_lds: bool = False,
    de_lds_level: str = "session",
    de_lds_q: float = DE_LDS_DEFAULT_Q,
    de_lds_r: float = DE_LDS_DEFAULT_R,
    de_lds_method: str = DE_LDS_DEFAULT_METHOD,
    de_lds_em_iters: int = DE_LDS_DEFAULT_EM_ITERS,
    de_lds_em_tol: float = DE_LDS_DEFAULT_EM_TOL,
) -> Iterable[Tuple[str, np.ndarray, int, int]]:
    """
    迭代 SEED-V 的 trial，返回 (trial_id, X_trial, label, trial_idx)。
    source="de" -> X_trial 为 (T, 310) (optional LDS smoothing)
    source="raw" -> X_trial 为 (T, C) 或 (B, C, C) (fb_cov)
    """
    source = (source or "de").lower()
    if source == "raw":
        yield from _iter_seedv_raw_trials(
            seedv_raw_root=seedv_raw_root,
            seedv_de_root=seedv_de_root,
            raw_repr=raw_repr,
            raw_cache=raw_cache,
            raw_channel_policy=raw_channel_policy,
            raw_shrinkage=raw_shrinkage,
            raw_bands=raw_bands,
        )
        return
    if source != "de":
        raise ValueError("SEED-V source currently supports DE or raw.")
    if not os.path.isdir(seedv_de_root):
        raise FileNotFoundError(f"SEED-V DE root not found: {seedv_de_root}")

    files = [f for f in os.listdir(seedv_de_root) if f.endswith(".npz")]
    for fname in sorted(files, key=lambda x: int(x.split("_")[0])):
        subject = fname.split("_")[0]
        data, label = _load_seedv_subject_npz(os.path.join(seedv_de_root, fname))
        if de_lds:
            data = _apply_de_lds(
                data,
                level=de_lds_level,
                q=de_lds_q,
                r=de_lds_r,
                method=de_lds_method,
                em_iters=de_lds_em_iters,
                em_tol=de_lds_em_tol,
            )
        for k in sorted(data.keys()):
            session_idx = k // 15
            trial_idx = k % 15
            Xt = np.asarray(data[k], dtype=np.float64)
            y_arr = np.asarray(label[k]).astype(int).ravel()
            cur_label = int(np.unique(y_arr)[0])
            trial_id = _seedv_trial_id(subject, session_idx, trial_idx)
            yield trial_id, Xt, cur_label, trial_idx


def iter_seedv_trials_meta(
    source: str = "de",
    seedv_de_root: str = SEEDV_DE_ROOT,
    seedv_raw_root: str = SEEDV_RAW_ROOT,
    raw_repr: str = "signal",
    raw_cache: str = "auto",
    raw_channel_policy: str = "strict",
    raw_shrinkage: Optional[float] = None,
    raw_bands: Optional[List[Tuple[float, float]]] = None,
    de_lds: bool = False,
    de_lds_level: str = "session",
    de_lds_q: float = DE_LDS_DEFAULT_Q,
    de_lds_r: float = DE_LDS_DEFAULT_R,
    de_lds_method: str = DE_LDS_DEFAULT_METHOD,
    de_lds_em_iters: int = DE_LDS_DEFAULT_EM_ITERS,
    de_lds_em_tol: float = DE_LDS_DEFAULT_EM_TOL,
) -> Iterable[Tuple[str, np.ndarray, int, int, int, str]]:
    """
    迭代 SEED-V 的 trial，返回 (trial_id, X_trial, label, trial_idx, session_idx, subject)。
    source="raw" 时，X_trial 形状为 (T, C) 或 (B, C, C) (fb_cov)。
    """
    for trial_id, Xt, y, t_idx in iter_seedv_trials(
        source=source,
        seedv_de_root=seedv_de_root,
        seedv_raw_root=seedv_raw_root,
        raw_repr=raw_repr,
        raw_cache=raw_cache,
        raw_channel_policy=raw_channel_policy,
        raw_shrinkage=raw_shrinkage,
        raw_bands=raw_bands,
        de_lds=de_lds,
        de_lds_level=de_lds_level,
        de_lds_q=de_lds_q,
        de_lds_r=de_lds_r,
        de_lds_method=de_lds_method,
        de_lds_em_iters=de_lds_em_iters,
        de_lds_em_tol=de_lds_em_tol,
    ):
        subject, session_idx, trial_idx = _parse_trial_id(trial_id)
        if trial_idx != t_idx:
            raise ValueError(f"trial_id parse mismatch: {trial_id} vs t_idx={t_idx}")
        yield trial_id, Xt, y, trial_idx, session_idx, subject


def split_fold_with_trial_id(
    split_by: str = "trial",
    *,
    de_lds: bool = False,
    de_lds_level: str = "session",
    de_lds_q: float = DE_LDS_DEFAULT_Q,
    de_lds_r: float = DE_LDS_DEFAULT_R,
    de_lds_method: str = DE_LDS_DEFAULT_METHOD,
    de_lds_em_iters: int = DE_LDS_DEFAULT_EM_ITERS,
    de_lds_em_tol: float = DE_LDS_DEFAULT_EM_TOL,
):
    """
    按 trial / session 分三组，同时返回每个 sample 的 trial_id。

    split_by:
      - "trial": trials 0~4 / 5~9 / 10~14
      - "session": sessions 0 / 1 / 2

    返回：
        X1, y1, tid1, X2, y2, tid2, X3, y3, tid3
    """
    set_1_x, set_1_y, set_1_tid = [], [], []
    set_2_x, set_2_y, set_2_tid = [], [], []
    set_3_x, set_3_y, set_3_tid = [], [], []

    split_by = (split_by or "trial").lower()
    for trial_id, Xt, y, t_idx, session_idx, _ in iter_seedv_trials_meta(
        de_lds=de_lds,
        de_lds_level=de_lds_level,
        de_lds_q=de_lds_q,
        de_lds_r=de_lds_r,
        de_lds_method=de_lds_method,
        de_lds_em_iters=de_lds_em_iters,
        de_lds_em_tol=de_lds_em_tol,
    ):
        if split_by == "trial":
            key = t_idx
            if 0 <= key <= 4:
                set_1_x.append(Xt)
                set_1_y.append(np.full((Xt.shape[0],), y))
                set_1_tid.append(np.full((Xt.shape[0],), trial_id))
            elif 5 <= key <= 9:
                set_2_x.append(Xt)
                set_2_y.append(np.full((Xt.shape[0],), y))
                set_2_tid.append(np.full((Xt.shape[0],), trial_id))
            elif 10 <= key <= 14:
                set_3_x.append(Xt)
                set_3_y.append(np.full((Xt.shape[0],), y))
                set_3_tid.append(np.full((Xt.shape[0],), trial_id))
            else:
                raise ValueError(f"Unexpected trial_idx {key}")
        elif split_by == "session":
            key = session_idx
            if key == 0:
                set_1_x.append(Xt)
                set_1_y.append(np.full((Xt.shape[0],), y))
                set_1_tid.append(np.full((Xt.shape[0],), trial_id))
            elif key == 1:
                set_2_x.append(Xt)
                set_2_y.append(np.full((Xt.shape[0],), y))
                set_2_tid.append(np.full((Xt.shape[0],), trial_id))
            elif key == 2:
                set_3_x.append(Xt)
                set_3_y.append(np.full((Xt.shape[0],), y))
                set_3_tid.append(np.full((Xt.shape[0],), trial_id))
            else:
                raise ValueError(f"Unexpected session_idx {key}")
        else:
            raise ValueError(f"Unknown split_by: {split_by}")

    X1 = np.vstack(set_1_x)
    y1 = np.concatenate(set_1_y)
    tid1 = np.concatenate(set_1_tid)

    X2 = np.vstack(set_2_x)
    y2 = np.concatenate(set_2_y)
    tid2 = np.concatenate(set_2_tid)

    X3 = np.vstack(set_3_x)
    y3 = np.concatenate(set_3_y)
    tid3 = np.concatenate(set_3_tid)

    return X1, y1, tid1, X2, y2, tid2, X3, y3, tid3


def _subject_sort_key(subject: str):
    s = str(subject)
    if s.isdigit():
        return (0, int(s))
    return (1, s)


def _subject_splits(subjects, mode: str = "loso", n_splits: int = 5, seed: int = 0):
    subjects_sorted = sorted(list(subjects), key=_subject_sort_key)
    mode = (mode or "loso").lower()
    if mode == "loso":
        return subjects_sorted, [[s] for s in subjects_sorted]
    if mode == "kfold":
        if n_splits < 2:
            raise ValueError("subject kfold requires n_splits >= 2")
        if n_splits > len(subjects_sorted):
            raise ValueError("subject kfold n_splits exceeds number of subjects")
        rng = np.random.RandomState(seed)
        shuffled = subjects_sorted.copy()
        rng.shuffle(shuffled)
        splits = np.array_split(shuffled, n_splits)
        return subjects_sorted, [list(s) for s in splits]
    raise ValueError(f"Unknown subject split mode: {mode}")


def split_subject_folds_with_trial_id(
    source: str = "de",
    mode: str = "loso",
    n_splits: int = 5,
    seed: int = 0,
    de_lds: bool = False,
    de_lds_level: str = "session",
    de_lds_q: float = DE_LDS_DEFAULT_Q,
    de_lds_r: float = DE_LDS_DEFAULT_R,
    de_lds_method: str = DE_LDS_DEFAULT_METHOD,
    de_lds_em_iters: int = DE_LDS_DEFAULT_EM_ITERS,
    de_lds_em_tol: float = DE_LDS_DEFAULT_EM_TOL,
):
    """
    生成 subject-level 交叉验证 folds（sample-level），返回:
        {
          "fold1": (X_tr, y_tr, tid_tr, X_te, y_te, tid_te),
          ...
        }
    """
    buckets = {}
    for trial_id, Xt, y, _t_idx, _session_idx, subject in iter_seedv_trials_meta(
        source=source,
        de_lds=de_lds,
        de_lds_level=de_lds_level,
        de_lds_q=de_lds_q,
        de_lds_r=de_lds_r,
        de_lds_method=de_lds_method,
        de_lds_em_iters=de_lds_em_iters,
        de_lds_em_tol=de_lds_em_tol,
    ):
        b = buckets.setdefault(subject, {"X": [], "y": [], "tid": []})
        b["X"].append(Xt)
        b["y"].append(np.full((Xt.shape[0],), y))
        b["tid"].append(np.full((Xt.shape[0],), trial_id))

    subjects_sorted, splits = _subject_splits(buckets.keys(), mode=mode, n_splits=n_splits, seed=seed)

    def _concat_subjects(subj_list):
        X_parts = []
        y_parts = []
        tid_parts = []
        for subj in subj_list:
            b = buckets[subj]
            X_parts.extend(b["X"])
            y_parts.extend(b["y"])
            tid_parts.extend(b["tid"])
        return np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(tid_parts)

    folds = {}
    for idx, test_subjects in enumerate(splits):
        test_set = set(test_subjects)
        train_subjects = [s for s in subjects_sorted if s not in test_set]
        if not train_subjects:
            raise ValueError("subject split produced empty train set")
        X_tr, y_tr, tid_tr = _concat_subjects(train_subjects)
        X_te, y_te, tid_te = _concat_subjects(test_subjects)
        folds[f"fold{idx + 1}"] = (X_tr, y_tr, tid_tr, X_te, y_te, tid_te)
    return folds


def split_subject_trial_folds(
    source: str = "de",
    mode: str = "loso",
    n_splits: int = 5,
    seed: int = 0,
    raw_repr: str = "signal",
    raw_cache: str = "auto",
    raw_channel_policy: str = "strict",
    raw_shrinkage: Optional[float] = None,
    raw_bands: Optional[List[Tuple[float, float]]] = None,
    de_lds: bool = False,
    de_lds_level: str = "session",
    de_lds_q: float = DE_LDS_DEFAULT_Q,
    de_lds_r: float = DE_LDS_DEFAULT_R,
    de_lds_method: str = DE_LDS_DEFAULT_METHOD,
    de_lds_em_iters: int = DE_LDS_DEFAULT_EM_ITERS,
    de_lds_em_tol: float = DE_LDS_DEFAULT_EM_TOL,
):
    """
    生成 subject-level 交叉验证 folds（trial-level），返回:
        {
          "fold1": (trials_tr, y_tr, tid_tr, trials_te, y_te, tid_te),
          ...
        }
    """
    buckets = {}
    for trial_id, Xt, y, _t_idx, _session_idx, subject in iter_seedv_trials_meta(
        source=source,
        raw_repr=raw_repr,
        raw_cache=raw_cache,
        raw_channel_policy=raw_channel_policy,
        raw_shrinkage=raw_shrinkage,
        raw_bands=raw_bands,
        de_lds=de_lds,
        de_lds_level=de_lds_level,
        de_lds_q=de_lds_q,
        de_lds_r=de_lds_r,
        de_lds_method=de_lds_method,
        de_lds_em_iters=de_lds_em_iters,
        de_lds_em_tol=de_lds_em_tol,
    ):
        b = buckets.setdefault(subject, {"trials": [], "y": [], "tid": []})
        b["trials"].append(Xt)
        b["y"].append(int(y))
        b["tid"].append(trial_id)

    subjects_sorted, splits = _subject_splits(buckets.keys(), mode=mode, n_splits=n_splits, seed=seed)

    def _concat_trials(subj_list):
        trials = []
        ys = []
        tids = []
        for subj in subj_list:
            b = buckets[subj]
            trials.extend(b["trials"])
            ys.extend(b["y"])
            tids.extend(b["tid"])
        return trials, np.asarray(ys, dtype=int), np.asarray(tids)

    folds = {}
    for idx, test_subjects in enumerate(splits):
        test_set = set(test_subjects)
        train_subjects = [s for s in subjects_sorted if s not in test_set]
        if not train_subjects:
            raise ValueError("subject split produced empty train set")
        trials_tr, y_tr, tid_tr = _concat_trials(train_subjects)
        trials_te, y_te, tid_te = _concat_trials(test_subjects)
        folds[f"fold{idx + 1}"] = (trials_tr, y_tr, tid_tr, trials_te, y_te, tid_te)
    return folds


def get_seedv_trial_lengths_for_folds(
    split_by: str = "trial",
    subject_split: str = "loso",
    subject_k: int = 5,
    subject_seed: int = 0,
):
    """
    统计 SEED-V 中每个 trial 的帧数，并按 split 协议返回测试集 trial 长度列表。

    split_by:
      - "trial": 按 trial_idx 分组 (0~4 / 5~9 / 10~14)
      - "session": 按 session_idx 分组 (0 / 1 / 2)
      - "subject": 按 subject 分组（LOSO / K-fold）
    """
    group1_lengths = []
    group2_lengths = []
    group3_lengths = []

    split_by = (split_by or "trial").lower()
    if split_by in {"trial", "session"}:
        for _tid, Xt, _y, t_idx, session_idx, _ in iter_seedv_trials_meta():
            T = Xt.shape[0]
            if split_by == "trial":
                if 0 <= t_idx <= 4:
                    group1_lengths.append(T)
                elif 5 <= t_idx <= 9:
                    group2_lengths.append(T)
                elif 10 <= t_idx <= 14:
                    group3_lengths.append(T)
                else:
                    raise ValueError(f"Unexpected trial_idx {t_idx}")
            else:
                if session_idx == 0:
                    group1_lengths.append(T)
                elif session_idx == 1:
                    group2_lengths.append(T)
                elif session_idx == 2:
                    group3_lengths.append(T)
                else:
                    raise ValueError(f"Unexpected session_idx {session_idx}")

        return {
            "fold1": group3_lengths,
            "fold2": group2_lengths,
            "fold3": group1_lengths,
        }

    if split_by == "subject":
        lengths_by_subject = {}
        for _tid, Xt, _y, _t_idx, _session_idx, subject in iter_seedv_trials_meta():
            lengths_by_subject.setdefault(subject, []).append(Xt.shape[0])

        _subjects, splits = _subject_splits(
            lengths_by_subject.keys(), mode=subject_split, n_splits=subject_k, seed=subject_seed
        )
        folds = {}
        for idx, test_subjects in enumerate(splits):
            lengths = []
            for subj in test_subjects:
                lengths.extend(lengths_by_subject[subj])
            folds[f"fold{idx + 1}"] = lengths
        return folds

    raise ValueError(f"Unknown split_by: {split_by}")
