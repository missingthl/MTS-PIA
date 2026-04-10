#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check SEED-V dataset integrity for raw CNT and DE feature files.
Reports missing sessions and channel mismatches for selected raw files.
"""
from __future__ import annotations

import argparse
import glob
import os
import warnings
from typing import Dict, List, Tuple


def list_subjects_from_de(de_root: str) -> List[str]:
    files = glob.glob(os.path.join(de_root, "*_123.npz"))
    subjects = sorted({os.path.basename(f).split("_")[0] for f in files}, key=lambda x: int(x))
    return subjects


def pick_raw_file(files: List[str]) -> str:
    # Prefer non-repaired if both exist
    files_sorted = sorted(files)
    for f in files_sorted:
        if "repaired" not in f:
            return f
    return files_sorted[0]


def find_raw_files(raw_root: str, subject: str, session: int) -> List[str]:
    pattern = os.path.join(raw_root, f"{subject}_{session}_*.cnt")
    return sorted(glob.glob(pattern))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=str, default="data/SEED_V/EEG_raw")
    ap.add_argument("--de-root", type=str, default="data/SEED_V/EEG_DE_features")
    ap.add_argument("--channel-list", type=str, default="data/SEED_V/channel_62_pos.locs")
    ap.add_argument("--check-channels", action="store_true", help="check channel matching (requires mne)")
    args = ap.parse_args()

    subjects = list_subjects_from_de(args.de_root)
    if not subjects:
        raise SystemExit(f"No DE files found under {args.de_root}")

    missing_raw = []
    repaired_used = []
    channel_issues = []

    if args.check_channels:
        try:
            import mne
        except ImportError as exc:
            raise SystemExit("mne is required: pip install mne") from exc
        from datasets.seedv_preprocess import _load_seedv_channel_names, _match_seedv_channels
        channel_names = _load_seedv_channel_names(args.channel_list)
        warnings.filterwarnings("ignore", message=".*Could not parse meas date.*", category=RuntimeWarning)

    for subject in subjects:
        for session in (1, 2, 3):
            files = find_raw_files(args.raw_root, subject, session)
            if not files:
                missing_raw.append((subject, session))
                continue
            chosen = pick_raw_file(files)
            if "repaired" in os.path.basename(chosen):
                repaired_used.append((subject, session, os.path.basename(chosen)))
            if args.check_channels:
                raw = mne.io.read_raw_cnt(chosen, preload=False, verbose="ERROR")
                picks, _positions, missing = _match_seedv_channels(raw, channel_names)
                if missing:
                    channel_issues.append((os.path.basename(chosen), missing[:8]))

    print("SEED-V integrity summary")
    print(f"  subjects: {len(subjects)}")
    print(f"  missing raw sessions: {len(missing_raw)}")
    if missing_raw:
        print(f"    examples: {missing_raw[:5]}")
    print(f"  repaired files used: {len(repaired_used)}")
    if repaired_used:
        print(f"    examples: {repaired_used[:5]}")
    if args.check_channels:
        print(f"  channel mismatches: {len(channel_issues)}")
        if channel_issues:
            print(f"    examples: {channel_issues[:5]}")


if __name__ == "__main__":
    main()
