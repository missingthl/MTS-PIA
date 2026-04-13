from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.seed_raw_trials import build_trial_index


def _resolve_time_txt(seed_raw_root: str) -> str:
    cand = os.path.join(seed_raw_root, "time.txt")
    if os.path.isfile(cand):
        return cand
    fallback = os.path.join("data", "SEED", "SEED_EEG", "time.txt")
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError("time.txt not found under seed_raw_root or default SEED_EEG")


def _resolve_stim_xlsx() -> str:
    cand = os.path.join("data", "SEED", "SEED_EEG", "SEED_stimulation.xlsx")
    if not os.path.isfile(cand):
        raise FileNotFoundError(f"SEED_stimulation.xlsx not found: {cand}")
    return cand


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-raw-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--channel-policy", type=str, default="strict", choices=["strict"])
    parser.add_argument("--dtype", type=str, default="int16", choices=["int16", "int32"])
    parser.add_argument("--format", type=str, default="fif", choices=["fif", "npz"])
    parser.add_argument("--manifest-out", type=str, required=True)
    parser.add_argument("--report-out", type=str, required=True)
    parser.add_argument(
        "--locs-path",
        type=str,
        default="data/SEED/channel_62_pos.locs",
        help="channel_62_pos.locs path",
    )
    args = parser.parse_args()

    seed_raw_root = os.path.abspath(args.seed_raw_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    time_txt = _resolve_time_txt(seed_raw_root)
    stim_xlsx = _resolve_stim_xlsx()

    cnt_files = sorted([str(p) for p in Path(seed_raw_root).glob("*.cnt")])
    if not cnt_files:
        raise FileNotFoundError(f"No .cnt files found under {seed_raw_root}")

    manifest = []
    failures = []
    start_time = time.time()

    for cnt_path in cnt_files:
        base = Path(cnt_path).stem
        if args.format == "fif":
            out_path = os.path.join(out_dir, f"{base}_eeg62_raw.fif")
        else:
            out_path = os.path.join(out_dir, f"{base}_eeg62_raw.npz")
        try:
            trials = build_trial_index(cnt_path, time_txt, stim_xlsx)
            entry = {
                "cnt_path": cnt_path,
                "out_path": out_path,
                "subject": trials[0].subject if trials else None,
                "session": trials[0].session if trials else None,
                "channel_policy": args.channel_policy,
                "dtype": args.dtype,
                "n_trials": len(trials),
                "status": "pending",
            }
            manifest.append(entry)
            with open(args.manifest_out, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            print(f"[seed_raw_manifest] queued {cnt_path} -> {out_path}")
        except Exception as exc:
            failures.append({"cnt_path": cnt_path, "error": str(exc)})
            print(f"[seed_raw_manifest] fail {cnt_path}: {exc}")

    elapsed = time.time() - start_time
    report = {
        "seed_raw_root": seed_raw_root,
        "out_dir": out_dir,
        "format": args.format,
        "dtype": args.dtype,
        "total_cnt_found": len(cnt_files),
        "listed_cnt": len(manifest),
        "list_fail": len(failures),
        "failures": failures,
        "elapsed_sec": elapsed,
    }
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
