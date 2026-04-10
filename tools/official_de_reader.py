import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io

BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]


def _resolve_mat_by_session(root: Path, subject: int, session: int) -> Path:
    subject_str = str(subject)
    files = sorted(
        p
        for p in root.iterdir()
        if p.suffix.lower() == ".mat"
        and p.name.startswith(subject_str + "_")
        and p.name.lower() != "label.mat"
    )
    if not files:
        raise FileNotFoundError(f"No .mat files found for subject {subject} in {root}")
    by_date = []
    for p in files:
        parts = p.stem.split("_")
        if len(parts) != 2:
            continue
        try:
            by_date.append((int(parts[1]), p.name))
        except ValueError:
            continue
    if not by_date:
        raise FileNotFoundError(f"No date-coded .mat files for subject {subject} in {root}")
    by_date.sort(key=lambda x: x[0])
    if session < 1 or session > len(by_date):
        raise ValueError(f"Session {session} out of range for subject {subject}")
    return root / by_date[session - 1][1]


def _normalize_trial_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array; got {arr.shape}")
    shape = list(arr.shape)
    if 62 not in shape or 5 not in shape:
        raise ValueError(f"Unexpected trial shape {shape}; missing C=62 or B=5")
    ch_axis = shape.index(62)
    band_axis = shape.index(5)
    time_axis = [i for i in range(3) if i not in (ch_axis, band_axis)]
    if len(time_axis) != 1:
        raise ValueError(f"Cannot infer time axis from shape {shape}")
    return np.moveaxis(arr, [ch_axis, time_axis[0], band_axis], [0, 1, 2])


def load_official_de_curves(
    *,
    mat_root: str,
    subject: int,
    session: int,
    trial: int,
    feature_base: str,
    bands: List[str],
) -> Dict[str, object]:
    root = Path(mat_root)
    mat_path = _resolve_mat_by_session(root, subject, session)
    mat = scipy.io.loadmat(mat_path)
    key_name = f"{feature_base}{int(trial)}"
    if key_name not in mat:
        raise KeyError(f"{key_name} not found in {mat_path}")
    arr = _normalize_trial_array(mat[key_name])
    T_off = int(arr.shape[1])

    band_index_map = {}
    curves = {}
    for name in bands:
        idx = BAND_ORDER.index(name)
        band_index_map[name] = idx
        curve = arr[:, :, idx].mean(axis=0).astype(np.float64)
        curves[name] = curve

    return {
        "mat_root": str(root),
        "mat_path": str(mat_path),
        "key_name": key_name,
        "T_off": T_off,
        "band_index_map": band_index_map,
        "curves": curves,
    }

