#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from datasets.seed_raw_trials import build_trial_index, export_trial_manifest


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


def _sorted_cnts(cnt_dir: str) -> List[str]:
    cnt_root = Path(cnt_dir)
    files = [str(p) for p in cnt_root.iterdir() if p.suffix.lower() == ".cnt"]
    return sorted(files, key=lambda p: _parse_cnt_name(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cnt-dir",
        default="data/SEED/SEED_EEG/SEED_RAW_EEG",
        help="directory with SEED_RAW_EEG/*.cnt",
    )
    ap.add_argument(
        "--time-txt",
        default="data/SEED/SEED_EEG/SEED_RAW_EEG/time.txt",
        help="time.txt with start/end point lists",
    )
    ap.add_argument(
        "--stim-xlsx",
        default="data/SEED/SEED_EEG/SEED_stimulation.xlsx",
        help="SEED_stimulation.xlsx",
    )
    ap.add_argument(
        "--out-json",
        default="logs/seed_raw_trial_manifest_full.json",
        help="output JSON manifest path",
    )
    ap.add_argument(
        "--out-csv",
        default="logs/seed_raw_trial_manifest_full.csv",
        help="output CSV manifest path",
    )
    ap.add_argument("--limit", type=int, default=0, help="limit number of CNT files")
    args = ap.parse_args()

    cnt_files = _sorted_cnts(args.cnt_dir)
    if args.limit > 0:
        cnt_files = cnt_files[: args.limit]

    all_trials = []
    for cnt_path in cnt_files:
        trials = build_trial_index(cnt_path, args.time_txt, args.stim_xlsx)
        all_trials.extend(trials)

    export_trial_manifest(all_trials, json_path=args.out_json, csv_path=args.out_csv)

    print(f"cnt_dir={Path(args.cnt_dir).resolve()}")
    print(f"cnt_files={len(cnt_files)}")
    print(f"total_trials={len(all_trials)}")
    print("session_inference=cnt_filename (subject_session.cnt)")
    print(f"manifest_json={Path(args.out_json).resolve()}")
    print(f"manifest_csv={Path(args.out_csv).resolve()}")


if __name__ == "__main__":
    main()
