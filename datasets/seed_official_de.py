from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io


def _parse_subject_date(fname: str) -> Tuple[str, int] | None:
    base = os.path.splitext(os.path.basename(fname))[0]
    match = re.match(r"^(\d+)_([0-9]{8})$", base)
    if not match:
        return None
    subject = match.group(1)
    date = int(match.group(2))
    return subject, date


def _de_file_map(de_root: str) -> Tuple[Dict[Tuple[str, int], str], List[str]]:
    de_root_path = Path(de_root)
    if not de_root_path.is_dir():
        raise FileNotFoundError(f"SEED official DE root not found: {de_root_path}")
    files = sorted([p for p in de_root_path.iterdir() if p.suffix.lower() == ".mat"])
    by_subject: Dict[str, List[Tuple[int, str]]] = {}
    skipped = []
    for p in files:
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


def _base_var_name(name: str) -> str:
    return re.sub(r"\d+$", "", name)


def _load_manifest(manifest_path: str) -> List[dict]:
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Seed raw trial manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not manifest:
        raise ValueError(f"Seed raw trial manifest is empty: {manifest_path}")
    return manifest


def _trial_sort_key(row: dict) -> Tuple[int, int, int]:
    try:
        subject_int = int(row["subject"])
    except Exception:
        subject_int = -1
    return subject_int, int(row["session"]), int(row["trial"])


def _de_trial_to_samples(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"DE trial expects 3D array (C, T, B); got {arr.shape}")
    if arr.shape[0] != 62 or arr.shape[2] != 5:
        raise ValueError(f"Unexpected DE shape {arr.shape} (expected 62xT×5)")
    # arr: (C, T, B) -> (T, B*C) with band-major order
    arr = arr.transpose(2, 0, 1)  # (B, C, T)
    arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])  # (B*C, T)
    return arr.T.astype(np.float32)  # (T, 310)


def load_seed_official_de(
    *,
    seed_de_root: str,
    seed_de_var: str,
    manifest_path: str,
    freeze_align: bool = True,
    seed_de_window: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict], List[str]]:
    if seed_de_window:
        raise NotImplementedError("--seed-de-window is not implemented yet")

    manifest = _load_manifest(manifest_path)
    de_map, skipped = _de_file_map(seed_de_root)
    de_base = _base_var_name(seed_de_var)

    if freeze_align:
        ordered = sorted(manifest, key=_trial_sort_key)
        align_key = "subject/session/trial"
    else:
        ordered = list(manifest)
        align_key = "manifest_order"

    train_trials = set(range(0, 9))
    test_trials = set(range(9, 15))

    de_cache: Dict[str, Dict[str, np.ndarray]] = {}
    train_x, train_y = [], []
    test_x, test_y = [], []
    trial_index_log: List[dict] = []
    trial_shape_example = None

    for row in ordered:
        subject = str(row["subject"])
        session = int(row["session"])
        trial = int(row["trial"])
        label = int(row["label"])

        de_path = de_map.get((subject, session))
        if not de_path:
            raise FileNotFoundError(
                f"Missing DE file for subject={subject} session={session}. "
                f"Available keys: {sorted(de_map.keys())[:5]}..."
            )
        if de_path not in de_cache:
            de_cache[de_path] = scipy.io.loadmat(de_path)
        mat = de_cache[de_path]

        var_name = f"{de_base}{trial + 1}"
        if var_name not in mat:
            raise KeyError(f"DE variable {var_name} not found in {de_path}")

        arr = np.asarray(mat[var_name])
        if trial_shape_example is None:
            trial_shape_example = tuple(arr.shape)
        X_trial = _de_trial_to_samples(arr)
        split = "train" if trial in train_trials else "test"
        trial_index_log.append(
            {
                "subject": subject,
                "session": session,
                "trial": trial,
                "trial_id": f"{subject}_s{session}_t{trial}",
                "label": label,
                "split": split,
                "source_mat": de_path,
                "source_cnt_path": row.get("source_cnt_path") or row.get("cnt_path"),
                "de_var": var_name,
                "n_windows": int(X_trial.shape[0]),
                "align_key": align_key,
            }
        )

        if split == "train":
            train_x.append(X_trial)
            train_y.append(np.full((X_trial.shape[0],), label, dtype=np.int64))
        else:
            test_x.append(X_trial)
            test_y.append(np.full((X_trial.shape[0],), label, dtype=np.int64))

    X_train = np.concatenate(train_x, axis=0)
    y_train = np.concatenate(train_y, axis=0)
    X_test = np.concatenate(test_x, axis=0)
    y_test = np.concatenate(test_y, axis=0)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"[seed1][de] root={seed_de_root} var={seed_de_var} mode=official")
    print("[seed1][de] session_inference=per-subject date-sorted filenames -> session index")
    if trial_shape_example is not None:
        print(f"[seed1][de] trial_shape_example={trial_shape_example}")
    labels_unique = sorted(set(np.concatenate([y_train, y_test]).tolist()))
    print(f"[seed1][de] trials={len(trial_index_log)} samples_train={len(y_train)} samples_test={len(y_test)}")
    print(f"[seed1][de] sample_shape={X_train.shape[1]} labels_unique={labels_unique}")

    return X_train, y_train, X_test, y_test, trial_index_log, skipped


def write_seed_train_test_index(
    trial_index: List[dict],
    *,
    out_dir: str = "logs",
) -> str:
    Path(out_dir).mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(out_dir) / f"seed_train_test_index_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trial_index, f, ensure_ascii=False, indent=2)
    return str(out_path)
