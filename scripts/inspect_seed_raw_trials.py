#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from datasets.seed_raw_trials import (
    build_trial_index,
    export_trial_manifest,
    load_seed_time_points,
    load_seed_stimulation_labels,
    slice_raw_trials,
)
from datasets.seed_raw_cnt import build_eeg62_view, load_one_cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnt", required=True, help="path to a SEED CNT file")
    ap.add_argument(
        "--locs",
        default="data/SEED/channel_62_pos.locs",
        help="channel_62_pos.locs path",
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
    ap.add_argument("--out-json", default="logs/seed_raw_trial_manifest.json")
    ap.add_argument("--out-csv", default="logs/seed_raw_trial_manifest.csv")
    ap.add_argument("--show", type=int, default=3, help="show first N trial slices")
    args = ap.parse_args()

    raw = load_one_cnt(args.cnt, preload=False)
    raw62, mapping = build_eeg62_view(raw, locs_path=args.locs)
    trials = build_trial_index(args.cnt, args.time_txt, args.stim_xlsx)
    slices = slice_raw_trials(raw62, trials)

    export_trial_manifest(trials, json_path=args.out_json, csv_path=args.out_csv)

    print(f"cnt_path={args.cnt}")
    print(f"raw_n_ch={len(raw.ch_names)} eeg62_n_ch={len(raw62.ch_names)}")
    print(f"dropped_names={mapping['dropped_names']}")
    print(f"selected_head={mapping['selected_names'][:10]}")
    print(f"manifest_json={Path(args.out_json).resolve()}")
    print(f"manifest_csv={Path(args.out_csv).resolve()}")

    for i, (seg, meta) in enumerate(slices[: args.show]):
        print(
            f"trial[{i}] shape={seg.shape} label={meta['label']} "
            f"start_idx={meta['start_idx']} end_idx={meta['end_idx']}"
        )


if __name__ == "__main__":
    main()
