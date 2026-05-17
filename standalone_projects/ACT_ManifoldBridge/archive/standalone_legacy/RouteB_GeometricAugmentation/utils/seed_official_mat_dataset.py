from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.io

from datasets.seed_raw_trials import load_seed_stimulation_labels


def _parse_subject_date(fname: str) -> Optional[Tuple[str, int]]:
    base = os.path.splitext(os.path.basename(fname))[0]
    match = re.match(r"^(\d+)_([0-9]{8})$", base)
    if not match:
        return None
    subject = match.group(1)
    date = int(match.group(2))
    return subject, date


def _build_session_map(root_dir: str) -> Tuple[Dict[Tuple[str, int], str], List[str]]:
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"SEED official feature root not found: {root}")
    files = sorted(p for p in root.iterdir() if p.suffix.lower() == ".mat")
    by_subject: Dict[str, List[Tuple[int, str]]] = {}
    skipped: List[str] = []
    for p in files:
        if p.name.lower() == "label.mat":
            continue
        parsed = _parse_subject_date(p.name)
        if not parsed:
            skipped.append(p.name)
            continue
        subject, date = parsed
        by_subject.setdefault(subject, []).append((date, str(p)))
    mapping: Dict[Tuple[str, int], str] = {}
    for subject, items in by_subject.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        for idx, (_, path) in enumerate(items_sorted, start=1):
            mapping[(subject, idx)] = path
    return mapping, skipped


def _normalize_feature_base(feature_base: str) -> str:
    base = re.sub(r"_?\d+$", "", feature_base)
    return base or feature_base


def _extract_trial_keys(keys: Iterable[str], feature_base: str) -> Dict[int, str]:
    pattern = re.compile(rf"^{re.escape(feature_base)}_?(\d+)$")
    trial_keys: Dict[int, str] = {}
    for key in keys:
        match = pattern.match(key)
        if not match:
            continue
        idx = int(match.group(1))
        trial_keys[idx] = key
    return trial_keys


def _normalize_trial_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (C,T,B); got {arr.shape}")
    shape = arr.shape
    if 62 not in shape or 5 not in shape:
        raise ValueError(f"Unexpected trial shape {shape}; missing C=62 or B=5")
    ch_axis = shape.index(62)
    band_axis = shape.index(5)
    time_axis = [i for i in range(3) if i not in (ch_axis, band_axis)]
    if len(time_axis) != 1:
        raise ValueError(f"Cannot infer time axis from shape {shape}")
    arr = np.moveaxis(arr, [ch_axis, time_axis[0], band_axis], [0, 1, 2])
    return arr


def _aggregate_trial(arr: np.ndarray, agg_mode: str) -> np.ndarray:
    arr = arr.astype(np.float64, copy=False)
    if agg_mode == "mean_time":
        out = arr.mean(axis=1)
    elif agg_mode == "mean_time_logvar":
        var = arr.var(axis=1)
        out = np.log(var + 1e-8)
    else:
        raise ValueError(f"Unsupported agg_mode: {agg_mode}")
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def build_official_trial_index(
    *,
    root_dir: str,
    feature_base: str = "de_LDS",
    label_path: Optional[str] = None,
) -> Tuple[List[dict], dict]:
    base = _normalize_feature_base(feature_base)
    labels = load_seed_stimulation_labels(
        label_path or "data/SEED/SEED_EEG/SEED_stimulation.xlsx"
    )
    session_map, skipped = _build_session_map(root_dir)
    trials: List[dict] = []
    missing_trials: Dict[str, List[int]] = {}

    sorted_keys = sorted(session_map.keys(), key=lambda x: (int(x[0]), x[1]))
    for subject, session in sorted_keys:
        mat_path = session_map[(subject, session)]
        mat = scipy.io.loadmat(mat_path)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        trial_keys = _extract_trial_keys(keys, base)
        if not trial_keys:
            raise KeyError(f"No keys found for base '{base}' in {mat_path}")
        expected = list(range(1, len(labels) + 1))
        missing = [idx for idx in expected if idx not in trial_keys]
        if missing:
            missing_trials[f"{subject}_s{session}"] = missing
        for idx in sorted(trial_keys.keys()):
            if idx < 1 or idx > len(labels):
                continue
            trials.append(
                {
                    "subject": subject,
                    "session": session,
                    "trial": idx - 1,
                    "label": int(labels[idx - 1]),
                    "mat_path": mat_path,
                    "key_name": trial_keys[idx],
                }
            )
    meta = {
        "feature_base": base,
        "skipped_files": skipped,
        "missing_trials": missing_trials,
    }
    return trials, meta


class OfficialMatTrialDataset:
    def __init__(
        self,
        root_dir: str,
        feature_base: str = "de_LDS",
        agg_mode: str = "mean_time",
        manifest_path: Optional[str] = None,
        label_path: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        self.root_dir = root_dir
        self.feature_base = _normalize_feature_base(feature_base)
        self.agg_mode = agg_mode
        self.manifest_path = manifest_path
        self.label_path = label_path
        self.verbose = verbose
        self._mat_cache: Dict[str, dict] = {}
        self.trials, self.meta = self._load_trials()
        if self.verbose:
            self._print_example()

    def _load_trials(self) -> Tuple[List[dict], dict]:
        if self.manifest_path:
            data = json.loads(Path(self.manifest_path).read_text())
            if isinstance(data, dict) and "trials" in data:
                return list(data["trials"]), data.get("meta", {})
            if isinstance(data, list):
                return list(data), {}
            raise ValueError("Unsupported manifest format for official dataset")
        return build_official_trial_index(
            root_dir=self.root_dir,
            feature_base=self.feature_base,
            label_path=self.label_path,
        )

    def __len__(self) -> int:
        return len(self.trials)

    def _load_mat(self, path: str) -> dict:
        if path not in self._mat_cache:
            self._mat_cache[path] = scipy.io.loadmat(path)
        return self._mat_cache[path]

    def _trial_to_feature(self, trial: dict) -> np.ndarray:
        mat = self._load_mat(trial["mat_path"])
        key_name = trial["key_name"]
        if key_name not in mat:
            raise KeyError(f"Key {key_name} missing in {trial['mat_path']}")
        arr = mat[key_name]
        arr = _normalize_trial_array(arr)
        agg = _aggregate_trial(arr, self.agg_mode)
        feat = agg.transpose(1, 0).reshape(-1)
        return feat.astype(np.float32)

    def __getitem__(self, idx: int):
        trial = self.trials[idx]
        feat = self._trial_to_feature(trial)
        label = int(trial["label"])
        meta = {
            "subject": trial["subject"],
            "session": trial["session"],
            "trial": trial["trial"],
            "mat_path": trial["mat_path"],
            "key_name": trial["key_name"],
        }
        return feat, label, meta

    def build_features(self) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        feats = []
        labels = []
        metas = []
        for idx in range(len(self)):
            feat, label, meta = self[idx]
            feats.append(feat)
            labels.append(label)
            metas.append(meta)
        return np.stack(feats, axis=0), np.asarray(labels, dtype=np.int64), metas

    def _print_example(self) -> None:
        if not self.trials:
            return
        trial = self.trials[0]
        mat = self._load_mat(trial["mat_path"])
        key_name = trial["key_name"]
        arr = mat[key_name]
        raw_shape = tuple(arr.shape)
        arr = _normalize_trial_array(arr)
        agg = _aggregate_trial(arr, self.agg_mode)
        feat = agg.transpose(1, 0).reshape(-1)
        print(
            "[official_mat] example_key="
            f"{key_name} raw_shape={list(raw_shape)} agg_shape={list(feat.shape)}",
            flush=True,
        )
