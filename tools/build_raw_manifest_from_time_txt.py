from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _load_time_txt(path: Path) -> Tuple[List[int], List[int]]:
    text = path.read_text()
    start_match = re.search(r"start_point_list\s*=\s*\[(.*?)\]", text, re.DOTALL)
    end_match = re.search(r"end_point_list\s*=\s*\[(.*?)\]", text, re.DOTALL)
    if not start_match or not end_match:
        raise ValueError(f"Unable to parse time.txt lists from {path}")
    start = [int(x) for x in start_match.group(1).split(",") if x.strip()]
    end = [int(x) for x in end_match.group(1).split(",") if x.strip()]
    if len(start) != len(end) or not start:
        raise ValueError("start/end list length mismatch in time.txt")
    return start, end


def _load_labels(path: Path) -> List[int]:
    df = pd.read_excel(path)
    if "Label" not in df.columns:
        raise ValueError("Label column not found in SEED_stimulation.xlsx")
    labels = df["Label"].dropna().astype(int).tolist()
    if not labels:
        raise ValueError("No labels loaded from SEED_stimulation.xlsx")
    return labels


def _parse_cnt_name(name: str) -> Tuple[str, int]:
    match = re.match(r"^(\\d+)_([123])\\.cnt$", name)
    if not match:
        raise ValueError(f"Unexpected CNT filename: {name}")
    subject = match.group(1)
    session = int(match.group(2))
    return subject, session


def main() -> None:
    parser = argparse.ArgumentParser(description="Build raw trial manifest from time.txt.")
    parser.add_argument(
        "--raw-root",
        default="data/SEED/SEED_Multimodal/Chinese/01-EEG-raw",
        help="directory containing CNT files",
    )
    parser.add_argument(
        "--time-txt",
        default="data/SEED/SEED_Multimodal/Chinese/01-EEG-raw/time.txt",
        help="path to time.txt with trial boundaries",
    )
    parser.add_argument(
        "--label-path",
        default="data/SEED/SEED_EEG/SEED_stimulation.xlsx",
        help="path to SEED_stimulation.xlsx",
    )
    parser.add_argument(
        "--out-manifest",
        default="logs/manifests/raw_trial_manifest.json",
        help="output manifest path",
    )
    parser.add_argument(
        "--out-audit",
        default="logs/manifests/raw_manifest_audit.json",
        help="output audit path",
    )
    parser.add_argument("--sampling-rate", type=float, default=1000.0)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    time_path = Path(args.time_txt)
    label_path = Path(args.label_path)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")
    if not time_path.exists():
        raise FileNotFoundError(f"time.txt not found: {time_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"label file not found: {label_path}")

    start_list, end_list = _load_time_txt(time_path)
    labels = _load_labels(label_path)
    if len(start_list) != len(labels):
        raise ValueError(
            f"time.txt trials ({len(start_list)}) != labels ({len(labels)})"
        )

    cnt_files = sorted(p for p in raw_root.iterdir() if p.suffix.lower() == ".cnt")
    if not cnt_files:
        raise FileNotFoundError(f"No CNT files found in {raw_root}")

    manifest = []
    trials_per_session: Dict[str, int] = {}
    duration_list = []
    label_hist: Dict[int, int] = {}
    missing = []

    for cnt in cnt_files:
        try:
            subject, session = _parse_cnt_name(cnt.name)
        except ValueError as err:
            missing.append({"file": cnt.name, "error": str(err)})
            continue
        key = f"{subject}_s{session}"
        trials_per_session[key] = 0
        for idx, (start, end) in enumerate(zip(start_list, end_list), start=1):
            if idx > len(labels):
                break
            label = int(labels[idx - 1])
            if label not in {0, 1, 2}:
                raise ValueError(f"Unexpected label {label} at trial {idx}")
            if end <= start:
                raise ValueError(f"Non-positive duration for {cnt} trial {idx}")
            duration_sec = (end - start) / float(args.sampling_rate)
            duration_list.append(duration_sec)
            label_hist[label] = label_hist.get(label, 0) + 1
            trials_per_session[key] += 1
            manifest.append(
                {
                    "subject_id": subject,
                    "session_id": session,
                    "trial_idx": idx,
                    "raw_file_path": str(cnt),
                    "start_sample": int(start),
                    "end_sample": int(end),
                    "start_sec": float(start) / float(args.sampling_rate),
                    "end_sec": float(end) / float(args.sampling_rate),
                    "duration_sec": float(duration_sec),
                    "label": label,
                    "sampling_rate": float(args.sampling_rate),
                    "label_source": str(label_path),
                    "time_txt_path": str(time_path),
                }
            )

    if missing:
        raise ValueError(f"Unparsed CNT filenames: {missing}")

    for key, count in trials_per_session.items():
        if count != len(labels):
            raise ValueError(f"Unexpected trial count for {key}: {count}")

    audit = {
        "subjects": sorted({m["subject_id"] for m in manifest}),
        "subject_counts": len({m["subject_id"] for m in manifest}),
        "sessions": sorted({m["session_id"] for m in manifest}),
        "trials_per_session": trials_per_session,
        "duration_sec": {
            "min": float(min(duration_list)) if duration_list else 0.0,
            "mean": float(sum(duration_list) / len(duration_list)) if duration_list else 0.0,
            "max": float(max(duration_list)) if duration_list else 0.0,
        },
        "label_hist": label_hist,
    }

    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps({"trials": manifest}, indent=2))

    out_audit = Path(args.out_audit)
    out_audit.parent.mkdir(parents=True, exist_ok=True)
    out_audit.write_text(json.dumps(audit, indent=2))

    print(
        f"[raw_manifest] trials={len(manifest)} subjects={audit['subject_counts']} "
        f"out={out_manifest}",
        flush=True,
    )
    print(
        f"[raw_manifest] duration_sec min/mean/max="
        f"{audit['duration_sec']['min']:.1f}/"
        f"{audit['duration_sec']['mean']:.1f}/"
        f"{audit['duration_sec']['max']:.1f}",
        flush=True,
    )
    print(f"[raw_manifest] label_hist={audit['label_hist']}", flush=True)


if __name__ == "__main__":
    main()
