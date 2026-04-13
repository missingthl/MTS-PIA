#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inspect channel counts/names for SEED-V raw CNT files.
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple
import warnings


def normalize_ch_name(name: str) -> str:
    n = str(name).strip().lower()
    if n.startswith("eeg"):
        n = n[3:]
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
    return alias_map.get(n, n)


def load_seedv_channel_names(path: str) -> List[str]:
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
    return names


def match_channels(raw_names: List[str], target_names: List[str]) -> Tuple[List[int], List[int], List[str]]:
    name_to_idx = {}
    for i, name in enumerate(raw_names):
        key = normalize_ch_name(name)
        if key not in name_to_idx:
            name_to_idx[key] = i
    picks: List[int] = []
    positions: List[int] = []
    missing: List[Tuple[str, int, str]] = []
    used_keys = set()
    for pos, name in enumerate(target_names):
        key = normalize_ch_name(name)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=str, default="data/SEED_V/EEG_raw")
    ap.add_argument("--channel-list", type=str, default="data/SEED_V/channel_62_pos.locs")
    ap.add_argument("--limit", type=int, default=0, help="limit number of files to inspect")
    args = ap.parse_args()

    try:
        import mne
    except ImportError as exc:
        raise SystemExit("mne is required: pip install mne") from exc

    channel_names = load_seedv_channel_names(args.channel_list)
    target_norm = [normalize_ch_name(n) for n in channel_names]
    name_set = set(target_norm)

    files = [f for f in os.listdir(args.raw_root) if f.endswith(".cnt")]
    files = sorted(files, key=lambda x: (int(x.split("_")[0]), x))
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    warnings.filterwarnings("ignore", message=".*Could not parse meas date.*", category=RuntimeWarning)
    for fname in files:
        raw_path = os.path.join(args.raw_root, fname)
        raw = mne.io.read_raw_cnt(raw_path, preload=False, verbose="ERROR")
        ch_names = raw.ch_names
        norm_names = [normalize_ch_name(n) for n in ch_names]
        matched = [n for n in norm_names if n in name_set]
        _picks, _pos, missing = match_channels(ch_names, channel_names)
        extra = [n for n in norm_names if n not in name_set]
        print(
            f"{fname}: n_channels={len(ch_names)}, matched={len(matched)}, "
            f"missing={len(missing)}, extra={len(extra)}"
        )
        if missing:
            print(f"  missing (first 8): {missing[:8]}")
        if extra:
            print(f"  extra (first 8): {extra[:8]}")


if __name__ == "__main__":
    main()
