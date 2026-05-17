from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _normalize_name(name: str) -> str:
    return str(name).strip().upper()


def _resolve_locs_path(locs_path: str) -> str:
    if locs_path and os.path.isfile(locs_path):
        return locs_path
    root = Path(__file__).resolve().parents[1]
    matches = list(root.rglob("channel_62_pos.locs"))
    if not matches:
        raise FileNotFoundError(
            f"channel_62_pos.locs not found. Checked: {locs_path or '(none)'}"
        )
    chosen = str(matches[0])
    print(f"[seed_raw] using locs file: {chosen}")
    return chosen


def _load_locs_names(locs_path: str) -> List[str]:
    names: List[str] = []
    with open(locs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            names.append(parts[-1])
    if len(names) != 62:
        raise ValueError(f"Unexpected channel count in {locs_path}: {len(names)}")
    return names


def load_one_cnt(path: str, *, preload: bool = False, data_format: Optional[str] = None):
    try:
        import mne
    except ImportError as exc:
        raise ImportError("mne is required to read CNT files. Install via `pip install mne`.") from exc
    try:
        if data_format:
            return mne.io.read_raw_cnt(
                path, preload=preload, verbose="ERROR", data_format=data_format
            )
        return mne.io.read_raw_cnt(path, preload=preload, verbose="ERROR")
    except Exception:
        for fmt in ("int16", "int32"):
            try:
                print(f"[seed_raw] read_raw_cnt retry with data_format={fmt} for {path}")
                return mne.io.read_raw_cnt(
                    path, preload=preload, verbose="ERROR", data_format=fmt
                )
            except Exception:
                continue
    if hasattr(mne.io, "read_raw_ant"):
        try:
            print(f"[seed_raw] read_raw_cnt failed, trying read_raw_ant for {path}")
            return mne.io.read_raw_ant(path, preload=preload, verbose="ERROR")
        except Exception:
            pass
    raise RuntimeError(f"Failed to read CNT file: {path}")


def load_one_raw(
    path: str,
    *,
    backend: str = "cnt",
    preload: bool = False,
    data_format: Optional[str] = None,
):
    backend = backend.lower()
    if backend == "cnt":
        return load_one_cnt(path, preload=preload, data_format=data_format)
    if backend == "fif":
        try:
            import mne
        except ImportError as exc:
            raise ImportError("mne is required to read FIF files. Install via `pip install mne`.") from exc
        return mne.io.read_raw_fif(path, preload=preload, verbose="ERROR")
    raise ValueError(f"Unsupported raw backend: {backend}")


def _duplicate_indices(names: List[str]) -> Dict[str, List[int]]:
    dupes: Dict[str, List[int]] = {}
    seen: Dict[str, int] = {}
    for idx, name in enumerate(names):
        if name in seen:
            if name not in dupes:
                dupes[name] = [seen[name]]
            dupes[name].append(idx)
        else:
            seen[name] = idx
    return dupes


def _index_map(names: List[str]) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    for idx, name in enumerate(names):
        mapping.setdefault(name, []).append(idx)
    return mapping


def build_eeg62_view(
    raw,
    locs_path: str,
    *,
    debug_map: bool = False,
    debug_out_path: Optional[str] = None,
    debug_sec: Optional[float] = None,
) -> Tuple[object, Dict[str, object]]:
    locs_path = _resolve_locs_path(locs_path)
    locs_names = _load_locs_names(locs_path)
    locs_norm = [_normalize_name(n) for n in locs_names]

    raw_names = list(raw.ch_names)
    raw_norm = [_normalize_name(n) for n in raw_names]

    raw_norm_dupes = _duplicate_indices(raw_norm)
    locs_norm_dupes = _duplicate_indices(locs_norm)

    seen = {}
    for idx, name in enumerate(raw_norm):
        if name in seen:
            if not debug_map:
                raise ValueError(
                    f"Duplicate channel name after normalization: {name} "
                    f"at indices {seen[name]} and {idx}"
                )
        seen[name] = idx

    missing = [locs_names[i] for i, n in enumerate(locs_norm) if n not in seen]
    locs_norm_set = set(locs_norm)
    if missing:
        extra = [raw_names[i] for i, n in enumerate(raw_norm) if n not in locs_norm_set]
        if not debug_map:
            raise ValueError(
                "Failed to match 62 EEG channels.\n"
                f"Missing: {missing}\n"
                f"Extra: {extra}\n"
                f"raw.ch_names: {raw_names}\n"
                f"locs names: {locs_names}"
            )

    if missing:
        picks = [seen[n] for n in locs_norm if n in seen]
    else:
        picks = [seen[n] for n in locs_norm]
    raw_eeg62 = raw.copy().pick(picks)

    selected_names = [raw_names[i] for i in picks]
    dropped_names = [raw_names[i] for i, n in enumerate(raw_norm) if n not in locs_norm_set]
    name_to_index = {name: idx for idx, name in enumerate(selected_names)}
    index_to_name = {str(idx): name for idx, name in enumerate(selected_names)}

    source_path = None
    if hasattr(raw, "filenames") and raw.filenames:
        source_path = raw.filenames[0]
    meta = {
        "source_cnt_path": source_path,
        "locs_path": locs_path,
        "selected_names": selected_names,
        "dropped_names": dropped_names,
        "name_to_index": name_to_index,
        "index_to_name": index_to_name,
    }

    if debug_map:
        raw_exact_map = _index_map(raw_names)
        raw_norm_map = _index_map(raw_norm)

        mapping_table = []
        missing_list = []
        ambiguous_expected = []
        matched_count = 0
        for expected_name, expected_norm in zip(locs_names, locs_norm):
            match_rule = "missing"
            matched_idx = -1
            matched_name = None
            candidates: List[int] = []

            exact_candidates = raw_exact_map.get(expected_name, [])
            if len(exact_candidates) == 1:
                matched_idx = exact_candidates[0]
                matched_name = raw_names[matched_idx]
                match_rule = "exact"
            elif len(exact_candidates) > 1:
                candidates = list(exact_candidates)
                match_rule = "exact_ambiguous"
            else:
                norm_candidates = raw_norm_map.get(expected_norm, [])
                if len(norm_candidates) == 1:
                    matched_idx = norm_candidates[0]
                    matched_name = raw_names[matched_idx]
                    match_rule = "normalized"
                elif len(norm_candidates) > 1:
                    candidates = list(norm_candidates)
                    match_rule = "normalized_ambiguous"

            entry = {
                "expected_name": expected_name,
                "matched_raw_name": matched_name,
                "matched_raw_index": int(matched_idx),
                "match_rule": match_rule,
            }
            if candidates:
                entry["candidates"] = candidates
                ambiguous_expected.append(entry)
            if matched_idx >= 0:
                matched_count += 1
            else:
                missing_list.append(expected_name)
            mapping_table.append(entry)

        duplicate_raw_indices = []
        idx_to_expected: Dict[int, List[str]] = {}
        for entry in mapping_table:
            idx = int(entry["matched_raw_index"])
            if idx < 0:
                continue
            idx_to_expected.setdefault(idx, []).append(entry["expected_name"])
        for idx, names in idx_to_expected.items():
            if len(names) > 1:
                duplicate_raw_indices.append(
                    {
                        "raw_index": int(idx),
                        "raw_name": raw_names[idx],
                        "expected_names": names,
                    }
                )

        debug_sec_val = float(debug_sec) if debug_sec is not None else 10.0
        sfreq = float(raw.info.get("sfreq", 0.0))
        stop = int(round(debug_sec_val * sfreq)) if sfreq > 0 else 0

        per_chan_std: List[float] = []
        per_chan_ptp: List[float] = []
        near_zero_entries = []
        top10_std_channels: List[Dict[str, object]] = []

        if stop > 0:
            matched_indices = sorted({entry["matched_raw_index"] for entry in mapping_table if entry["matched_raw_index"] >= 0})
            if matched_indices:
                data = raw.get_data(picks=matched_indices, start=0, stop=stop).astype(np.float64, copy=False)
            else:
                data = np.empty((0, stop), dtype=np.float64)
            idx_map = {raw_idx: pos for pos, raw_idx in enumerate(matched_indices)}
            x62 = np.zeros((len(locs_names), data.shape[1]), dtype=np.float64)
            for row_idx, entry in enumerate(mapping_table):
                raw_idx = int(entry["matched_raw_index"])
                if raw_idx >= 0 and raw_idx in idx_map:
                    x62[row_idx] = data[idx_map[raw_idx]]

            std = x62.std(axis=1)
            ptp = np.ptp(x62, axis=1)
            per_chan_std = [float(v) for v in std]
            per_chan_ptp = [float(v) for v in ptp]

            for threshold in (1e-10, 1e-9):
                names = [locs_names[i] for i, v in enumerate(std) if v < threshold]
                near_zero_entries.append(
                    {
                        "threshold": float(threshold),
                        "count": int(len(names)),
                        "expected_names": names,
                    }
                )

            order = np.argsort(std)[::-1][:10]
            for idx in order:
                top10_std_channels.append(
                    {
                        "expected_name": locs_names[int(idx)],
                        "std": float(std[int(idx)]),
                        "std_uV": float(std[int(idx)] * 1e6),
                    }
                )

        audit = {
            "raw_ch_names": raw_names,
            "expected_62": locs_names,
            "mapping_table": mapping_table,
            "matched_count": int(matched_count),
            "missing_count": int(len(missing_list)),
            "missing_list": missing_list,
            "duplicate_raw_indices": duplicate_raw_indices,
            "ambiguous_expected": ambiguous_expected,
            "raw_norm_duplicates": raw_norm_dupes,
            "expected_norm_duplicates": locs_norm_dupes,
            "debug_sec": float(debug_sec_val),
            "debug_samples": int(stop),
            "picked_count": int(len(picks)),
            "per_chan_std": per_chan_std,
            "per_chan_ptp": per_chan_ptp,
            "near_zero": near_zero_entries,
            "top10_std_channels": top10_std_channels,
        }
        meta["mapping_audit"] = audit

        if matched_count < 58:
            print(
                f"[seed_raw][map] ERROR matched_count={matched_count} < 58; "
                "mapping likely invalid. Stop training.",
                flush=True,
            )
        if missing_list:
            print(f"[seed_raw][map] missing_count={len(missing_list)}", flush=True)
        if duplicate_raw_indices:
            print(
                f"[seed_raw][map] WARNING duplicate_raw_indices={len(duplicate_raw_indices)}",
                flush=True,
            )
        if ambiguous_expected:
            print(
                f"[seed_raw][map] WARNING ambiguous_expected={len(ambiguous_expected)}",
                flush=True,
            )

        if debug_out_path:
            out_path = Path(debug_out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(audit, f, ensure_ascii=True, indent=2)
            print(f"[seed_raw][map] wrote audit {out_path}", flush=True)

    return raw_eeg62, meta
