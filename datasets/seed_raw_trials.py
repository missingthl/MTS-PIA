from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .seed_raw_cnt import build_eeg62_view, load_one_cnt


@dataclass
class TrialIndex:
    subject: str
    session: int  # 1-based session index from filename
    trial: int    # 0-based trial index (0..14)
    label: int
    t_start_s: float
    t_end_s: float
    start_1000hz: int
    end_1000hz: int
    source_cnt_path: str
    time_unit: str = "samples@1000"


_TIME_UNIT_WARNED = False
_SLICE_LOGGED = False


def _normalize_time_unit(time_unit: Optional[str]) -> str:
    global _TIME_UNIT_WARNED
    if time_unit is None or str(time_unit).strip() == "":
        if not _TIME_UNIT_WARNED:
            print(
                "[seed_raw][time] Warning: time_unit not set; defaulting to samples@1000 "
                "(legacy behavior)",
                flush=True,
            )
            _TIME_UNIT_WARNED = True
        return "samples@1000"
    key = str(time_unit).strip().lower()
    mapping = {
        "samples@1000": "samples@1000",
        "samples@1k": "samples@1000",
        "1000": "samples@1000",
        "samples@200": "samples@200",
        "200": "samples@200",
        "seconds": "seconds",
        "sec": "seconds",
        "s": "seconds",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported time_unit: {time_unit}")
    return mapping[key]


def _parse_time_list(line: str) -> List[int]:
    vals = [int(x) for x in re.findall(r"-?\d+", line)]
    # Strip trailing "1000" from the "#1000Hz" comment if present.
    if len(vals) > 15 and vals[-1] == 1000:
        vals = vals[:-1]
    return vals


def load_seed_time_points(time_txt_path: str) -> Tuple[List[int], List[int]]:
    if not os.path.isfile(time_txt_path):
        raise FileNotFoundError(f"time.txt not found: {time_txt_path}")
    start_pts: List[int] = []
    end_pts: List[int] = []
    with open(time_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("start_point_list"):
                start_pts = _parse_time_list(line)
            elif line.startswith("end_point_list"):
                end_pts = _parse_time_list(line)
    if not start_pts or not end_pts:
        raise ValueError(f"Failed to parse start/end points from {time_txt_path}")
    if len(start_pts) != len(end_pts):
        raise ValueError(
            f"start/end length mismatch in {time_txt_path}: "
            f"{len(start_pts)} vs {len(end_pts)}"
        )
    return start_pts, end_pts


def load_seed_stimulation_labels(xlsx_path: str) -> List[int]:
    if not os.path.isfile(xlsx_path):
        raise FileNotFoundError(f"SEED_stimulation.xlsx not found: {xlsx_path}")
    import pandas as pd

    df = pd.read_excel(xlsx_path)
    if "Label" not in df.columns:
        raise ValueError(f"Missing 'Label' column in {xlsx_path}")
    labels = [int(x) for x in df["Label"].dropna().tolist()]
    if len(labels) != 15:
        raise ValueError(f"Expected 15 labels in {xlsx_path}, got {len(labels)}")
    return labels


def _parse_cnt_name(cnt_path: str) -> Tuple[str, int]:
    base = os.path.splitext(os.path.basename(cnt_path))[0]
    parts = base.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid CNT filename: {cnt_path}")
    subject = parts[0]
    session = int(parts[1])
    return subject, session


def build_trial_index(
    cnt_path: str,
    time_txt_path: str,
    stim_xlsx_path: str,
    time_unit: Optional[str] = None,
) -> List[TrialIndex]:
    subject, session = _parse_cnt_name(cnt_path)
    start_pts, end_pts = load_seed_time_points(time_txt_path)
    labels = load_seed_stimulation_labels(stim_xlsx_path)
    if len(labels) != len(start_pts):
        raise ValueError(
            f"Label count {len(labels)} does not match time points {len(start_pts)}"
        )
    trials: List[TrialIndex] = []
    unit = _normalize_time_unit(time_unit)
    denom = None
    if unit == "samples@1000":
        denom = 1000.0
    elif unit == "samples@200":
        denom = 200.0
    elif unit == "seconds":
        denom = None
    else:
        raise ValueError(f"Unsupported time_unit: {unit}")

    for idx, (s, e, y) in enumerate(zip(start_pts, end_pts, labels)):
        if denom is None:
            t_start_s = float(s)
            t_end_s = float(e)
        else:
            t_start_s = float(s) / denom
            t_end_s = float(e) / denom
        start_1000hz = int(round(t_start_s * 1000.0))
        end_1000hz = int(round(t_end_s * 1000.0))
        trials.append(
            TrialIndex(
                subject=subject,
                session=session,
                trial=idx,
                label=int(y),
                t_start_s=t_start_s,
                t_end_s=t_end_s,
                start_1000hz=start_1000hz,
                end_1000hz=end_1000hz,
                source_cnt_path=cnt_path,
                time_unit=unit,
            )
        )
    return trials


def slice_raw_trials(
    raw_eeg62,
    trials: Iterable[TrialIndex],
    trial_offset_sec: float = 0.0,
) -> List[Tuple[np.ndarray, Dict[str, object]]]:
    global _SLICE_LOGGED
    sfreq = float(raw_eeg62.info["sfreq"])
    n_times = int(raw_eeg62.n_times)
    out: List[Tuple[np.ndarray, Dict[str, object]]] = []
    for t in trials:
        t_start_raw = float(t.t_start_s)
        t_end_raw = float(t.t_end_s)
        t_start = t_start_raw + float(trial_offset_sec)
        t_end = t_end_raw + float(trial_offset_sec)
        start_idx = int(round(t_start * sfreq))
        end_idx = int(round(t_end * sfreq))
        if end_idx <= start_idx:
            print(
                "[seed_raw][slice] skip trial="
                f"{t.subject}_s{t.session}_t{t.trial} "
                f"reason=non_positive_duration start_idx={start_idx} end_idx={end_idx}",
                flush=True,
            )
            continue
        start_idx = max(0, min(start_idx, n_times - 1))
        end_idx = max(start_idx + 1, min(end_idx, n_times))
        if end_idx <= start_idx:
            print(
                "[seed_raw][slice] skip trial="
                f"{t.subject}_s{t.session}_t{t.trial} "
                f"reason=clamped_empty start_idx={start_idx} end_idx={end_idx}",
                flush=True,
            )
            continue
        if not _SLICE_LOGGED:
            time_unit = getattr(t, "time_unit", "unknown")
            duration_sec = float(end_idx - start_idx) / sfreq
            print(
                "[seed_raw][slice] "
                f"raw_sfreq={sfreq:.2f} time_unit={time_unit} "
                f"trial_offset_sec={float(trial_offset_sec):.3f} "
                f"start_idx={start_idx} end_idx={end_idx} "
                f"trial_duration_sec={duration_sec:.3f}",
                flush=True,
            )
            _SLICE_LOGGED = True
        seg = raw_eeg62.get_data(start=start_idx, stop=end_idx)
        meta = {
            "subject": t.subject,
            "session": t.session,
            "trial": t.trial,
            "label": t.label,
            "t_start_s_raw": t_start_raw,
            "t_end_s_raw": t_end_raw,
            "t_start_s_adj": t_start,
            "t_end_s_adj": t_end,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "source_cnt_path": t.source_cnt_path,
            "time_unit": getattr(t, "time_unit", None),
            "trial_offset_sec": float(trial_offset_sec),
        }
        out.append((seg, meta))
    return out


def export_trial_manifest(
    trials: Iterable[TrialIndex],
    *,
    json_path: str,
    csv_path: str,
) -> None:
    rows = [
        {
            "subject": t.subject,
            "session": t.session,
            "trial": t.trial,
            "label": t.label,
            "t_start_s": t.t_start_s,
            "t_end_s": t.t_end_s,
            "start_1000hz": t.start_1000hz,
            "end_1000hz": t.end_1000hz,
            "source_cnt_path": t.source_cnt_path,
            "cnt_path": t.source_cnt_path,
            "time_unit": t.time_unit,
        }
        for t in trials
    ]
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(csv_path, "w", encoding="utf-8") as f:
        header = list(rows[0].keys()) if rows else []
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in header) + "\n")


def build_eeg62_trials_from_cnt(
    cnt_path: str,
    locs_path: str,
    time_txt_path: str,
    stim_xlsx_path: str,
    time_unit: Optional[str] = None,
    trial_offset_sec: float = 0.0,
) -> Tuple[List[Tuple[np.ndarray, Dict[str, object]]], Dict[str, object]]:
    raw = load_one_cnt(cnt_path, preload=False)
    raw62, mapping = build_eeg62_view(raw, locs_path=locs_path)
    trials = build_trial_index(cnt_path, time_txt_path, stim_xlsx_path, time_unit=time_unit)
    slices = slice_raw_trials(raw62, trials, trial_offset_sec=trial_offset_sec)
    return slices, mapping
