from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .seed_raw_cnt import build_eeg62_view, load_one_raw
from .seed_raw_trials import build_trial_index


@dataclass
class RawTrialBatch:
    X: List[np.ndarray]
    y: np.ndarray
    meta: List[Dict[str, object]]
    channel_names: List[str]
    fs: float
    raw_root: str
    manifest: List[Dict[str, object]]
    labels_all: np.ndarray


def _parse_cnt_name(cnt_path: str) -> Tuple[int, int, str]:
    base = os.path.splitext(os.path.basename(cnt_path))[0]
    parts = base.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid CNT filename: {cnt_path}")
    subject_str = parts[0]
    session = int(parts[1])
    try:
        subject_int = int(subject_str)
    except ValueError:
        subject_int = -1
    return subject_int, session, subject_str


def _sorted_raw_files(raw_dir: str, ext: str) -> List[str]:
    raw_root = Path(raw_dir)
    files = [str(p) for p in raw_root.iterdir() if p.suffix.lower() == ext]
    return sorted(files, key=lambda p: _parse_cnt_name(p))


def _resolve_time_txt(seed_raw_root: str) -> str:
    cand = os.path.join(seed_raw_root, "time.txt")
    if os.path.isfile(cand):
        return cand
    fallback = os.path.join("data", "SEED", "SEED_EEG", "time.txt")
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError(
        "time.txt not found under seed_raw_root or default SEED_EEG. "
        f"Checked: {cand}, {fallback}"
    )


def _resolve_stim_xlsx() -> str:
    cand = os.path.join("data", "SEED", "SEED_EEG", "SEED_stimulation.xlsx")
    if not os.path.isfile(cand):
        raise FileNotFoundError(f"SEED_stimulation.xlsx not found: {cand}")
    return cand


def _hash_channels(channel_names: List[str]) -> str:
    joined = ",".join(channel_names).encode("utf-8")
    return hashlib.sha1(joined).hexdigest()


def load_seed1_raw_trials(
    seed_raw_root: str,
    *,
    fs: Optional[int] = None,
    time_unit: Optional[str] = None,
    channel_policy: str = "strict",
    locs_path: Optional[str] = "data/SEED/channel_62_pos.locs",
    debug_trials: int = 1,
    max_cnt_files: Optional[int] = None,
    raw_backend: str = "cnt",
) -> RawTrialBatch:
    seed_raw_root = os.path.abspath(seed_raw_root)
    if "ExtractedFeatures" in seed_raw_root:
        raise ValueError(f"seed_raw_root looks like features, not raw: {seed_raw_root}")
    if not os.path.isdir(seed_raw_root):
        raise FileNotFoundError(f"seed_raw_root not found: {seed_raw_root}")
    if channel_policy != "strict":
        raise ValueError(f"Unsupported channel policy: {channel_policy}")

    ext = ".fif" if raw_backend == "fif" else ".cnt"
    cnt_files = _sorted_raw_files(seed_raw_root, ext)
    if not cnt_files:
        raise FileNotFoundError(f"No {ext} files found under {seed_raw_root}")
    if max_cnt_files is not None and max_cnt_files > 0:
        cnt_files = cnt_files[: max_cnt_files]

    time_txt_path = _resolve_time_txt(seed_raw_root)
    stim_xlsx_path = _resolve_stim_xlsx()

    X_debug: List[np.ndarray] = []
    y_debug: List[int] = []
    meta_debug: List[Dict[str, object]] = []
    manifest: List[Dict[str, object]] = []
    labels_all: List[int] = []
    channel_names: List[str] = []
    fs_used: Optional[float] = None

    for cnt_path in cnt_files:
        raw = load_one_raw(cnt_path, backend=raw_backend, preload=False)
        raw62, meta = build_eeg62_view(raw, locs_path=locs_path or "")
        current_names = meta["selected_names"]
        if not channel_names:
            channel_names = list(current_names)
        elif channel_names != list(current_names):
            raise ValueError(
                "Channel order mismatch across CNT files. "
                f"First head={channel_names[:5]} current head={current_names[:5]}"
            )

        raw_fs = float(raw62.info.get("sfreq", 0.0))
        if raw_fs <= 0 and fs is None:
            raise ValueError("Sample rate missing in raw file; provide --seed-raw-fs")
        if raw_fs > 0:
            if fs is not None and abs(raw_fs - float(fs)) > 1e-3:
                print(f"[seed1][raw] Warning: raw sfreq={raw_fs} overrides seed-raw-fs={fs}")
            fs_used = raw_fs
        else:
            fs_used = float(fs)

        trials = build_trial_index(
            cnt_path,
            time_txt_path,
            stim_xlsx_path,
            time_unit=time_unit,
        )
        for t in trials:
            trial_id = f"{t.subject}_s{t.session}_t{t.trial}"
            start_idx = int(round(t.t_start_s * fs_used))
            end_idx = int(round(t.t_end_s * fs_used))
            len_T = max(0, end_idx - start_idx)
            manifest.append(
                {
                    "subject": t.subject,
                    "session": t.session,
                    "trial": t.trial,
                    "trial_id": trial_id,
                    "label": t.label,
                    "t_start_s": t.t_start_s,
                    "t_end_s": t.t_end_s,
                    "len_T": len_T,
                    "source_cnt_path": t.source_cnt_path,
                    "time_unit": getattr(t, "time_unit", None),
                }
            )
            labels_all.append(int(t.label))
            if len(X_debug) < max(1, int(debug_trials)):
                seg = raw62.get_data(start=start_idx, stop=end_idx).astype(np.float32)
                X_debug.append(seg)
                y_debug.append(int(t.label))
                meta_debug.append(
                    {
                        "subject": t.subject,
                        "session": t.session,
                        "trial": t.trial,
                        "trial_id": trial_id,
                        "label": t.label,
                        "t_start_s": t.t_start_s,
                        "t_end_s": t.t_end_s,
                        "len_T": len_T,
                        "source_cnt_path": t.source_cnt_path,
                    }
                )

    return RawTrialBatch(
        X=X_debug,
        y=np.asarray(y_debug, dtype=np.int64),
        meta=meta_debug,
        channel_names=channel_names,
        fs=float(fs_used) if fs_used is not None else 0.0,
        raw_root=seed_raw_root,
        manifest=manifest,
        labels_all=np.asarray(labels_all, dtype=np.int64),
    )
