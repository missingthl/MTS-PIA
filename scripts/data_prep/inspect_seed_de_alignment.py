#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import scipy.io

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


def _parse_subject_date(fname: str) -> Tuple[str, int] | None:
    base = os.path.splitext(os.path.basename(fname))[0]
    match = re.match(r"^(\d+)_([0-9]{8})$", base)
    if not match:
        return None
    subject = match.group(1)
    date = int(match.group(2))
    return subject, date


def _load_manifest(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_stim_labels(xlsx_path: str) -> List[int]:
    import pandas as pd

    df = pd.read_excel(xlsx_path)
    if "Label" not in df.columns:
        raise ValueError(f"Missing Label column in {xlsx_path}")
    labels = [int(x) for x in df["Label"].dropna().tolist()]
    if len(labels) != 15:
        raise ValueError(f"Expected 15 labels, got {len(labels)}")
    return labels


def _group_manifest(manifest: List[dict]) -> Dict[Tuple[str, int], List[dict]]:
    grouped: Dict[Tuple[str, int], List[dict]] = {}
    for row in manifest:
        key = (str(row["subject"]), int(row["session"]))
        grouped.setdefault(key, []).append(row)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda r: int(r["trial"]))
    return grouped


def _de_file_map(de_root: str) -> Tuple[Dict[Tuple[str, int], str], List[str]]:
    de_root_path = Path(de_root)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="raw trial manifest JSON")
    ap.add_argument("--de-root", required=True, help="ExtractedFeatures_1s or _4s directory")
    ap.add_argument("--de-var", required=True, help="de_LDS1 or de_movingAve1 (prefix+index)")
    ap.add_argument(
        "--stim-xlsx",
        default="data/SEED/SEED_EEG/SEED_stimulation.xlsx",
        help="SEED_stimulation.xlsx path",
    )
    ap.add_argument("--mismatch-limit", type=int, default=5)
    ap.add_argument("--out", default=None, help="output report JSON (default logs/...)")
    args = ap.parse_args()

    manifest = _load_manifest(args.manifest)
    stim_labels = _load_stim_labels(args.stim_xlsx)
    grouped = _group_manifest(manifest)
    de_map, skipped = _de_file_map(args.de_root)
    de_base = _base_var_name(args.de_var)

    report = {
        "manifest": os.path.abspath(args.manifest),
        "de_root": os.path.abspath(args.de_root),
        "de_var_requested": args.de_var,
        "de_var_base": de_base,
        "session_inference": "Per subject, sort DE files by date token and assign session index starting at 1.",
        "skipped_de_files": skipped,
        "subjects": [],
    }

    summary = {
        "total_subject_sessions": 0,
        "total_raw_trials": 0,
        "total_de_trials": 0,
        "label_match_rate_global": 0.0,
        "sessions_missing_de_file": 0,
        "sessions_trial_count_mismatch": 0,
        "sessions_label_mismatch": 0,
        "sessions_with_mismatch": 0,
    }
    total_label_matches = 0
    total_label_count = 0

    for key, trials in grouped.items():
        subject, session = key
        de_path = de_map.get((subject, session))
        entry = {
            "subject": subject,
            "session": session,
            "de_file": de_path,
            "trial_count_raw": len(trials),
            "trial_count_de": 0,
            "label_match_rate": 0.0,
            "label_mismatches": [],
            "missing_de_trials": [],
            "mismatch_reasons": [],
            "de_shape_example": None,
            "raw_trial_duration_s": None,
        }

        raw_labels = [int(t["label"]) for t in trials]
        matches_count = 0
        if len(raw_labels) == len(stim_labels):
            matches = [int(a == b) for a, b in zip(raw_labels, stim_labels)]
            matches_count = int(sum(matches))
            entry["label_match_rate"] = float(matches_count) / float(len(matches))
            if entry["label_match_rate"] < 1.0:
                for i, (a, b) in enumerate(zip(raw_labels, stim_labels)):
                    if a != b:
                        entry["label_mismatches"].append(
                            {
                                "trial": i,
                                "trial_1based": i + 1,
                                "trial_id": f"{subject}_s{session}_t{i}",
                                "raw_label": a,
                                "stim_label": b,
                            }
                        )
        else:
            entry["mismatch_reasons"].append(
                {"reason": "label_length_mismatch", "raw": len(raw_labels), "stim": len(stim_labels)}
            )

        if trials:
            entry["raw_trial_duration_s"] = float(trials[0]["t_end_s"] - trials[0]["t_start_s"])

        if not de_path:
            entry["mismatch_reasons"].append({"reason": "missing_de_file"})
            report["subjects"].append(entry)
            summary["sessions_missing_de_file"] += 1
            summary["total_subject_sessions"] += 1
            summary["total_raw_trials"] += len(trials)
            total_label_matches += matches_count
            total_label_count += len(raw_labels)
            continue

        whos = scipy.io.whosmat(de_path)
        var_shapes = {name: shape for name, shape, _ in whos}
        available_trials = []
        expected_trials = list(range(1, len(stim_labels) + 1))
        for i in expected_trials:
            var_name = f"{de_base}{i}"
            if var_name in var_shapes:
                available_trials.append((i, var_shapes[var_name]))
        entry["trial_count_de"] = len(available_trials)
        if available_trials:
            entry["de_shape_example"] = {
                "var": f"{de_base}{available_trials[0][0]}",
                "shape": available_trials[0][1],
                "interpretation": "shape=(channels, windows, bands)",
            }

        available_idx = {i for i, _ in available_trials}
        missing = [i for i in expected_trials if i not in available_idx]
        if missing:
            entry["missing_de_trials"] = [
                {
                    "trial_1based": i,
                    "trial": i - 1,
                    "trial_id": f"{subject}_s{session}_t{i - 1}",
                }
                for i in missing
            ]
        if entry["trial_count_de"] != len(trials):
            entry["mismatch_reasons"].append(
                {
                    "reason": "trial_count_mismatch",
                    "raw": len(trials),
                    "de": entry["trial_count_de"],
                }
            )
            summary["sessions_trial_count_mismatch"] += 1

        if entry["label_mismatches"]:
            summary["sessions_label_mismatch"] += 1

        report["subjects"].append(entry)
        summary["total_subject_sessions"] += 1
        summary["total_raw_trials"] += len(trials)
        summary["total_de_trials"] += entry["trial_count_de"]
        total_label_matches += matches_count
        total_label_count += len(raw_labels)

        if entry["mismatch_reasons"] or entry["label_mismatches"] or entry["missing_de_trials"]:
            summary["sessions_with_mismatch"] += 1

    out_path = args.out
    if not out_path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = str(Path("logs") / f"seed_de_alignment_report_{ts}.json")
    Path(os.path.dirname(out_path)).mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        if total_label_count:
            summary["label_match_rate_global"] = float(total_label_matches) / float(total_label_count)
        report["summary"] = summary
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[report] saved={out_path}")
    for entry in report["subjects"]:
        print(
            f"[subject={entry['subject']} session={entry['session']}] "
            f"raw={entry['trial_count_raw']} de={entry['trial_count_de']} "
            f"label_match={entry['label_match_rate']:.3f} de_file={entry['de_file']}"
        )
        if entry["de_shape_example"]:
            print(f"  de_shape_example={entry['de_shape_example']}")
        if entry["raw_trial_duration_s"] is not None:
            print(f"  raw_trial_duration_s={entry['raw_trial_duration_s']:.1f}")
        if entry["mismatch_reasons"]:
            print(f"  mismatch_reasons={entry['mismatch_reasons'][:args.mismatch_limit]}")
        if entry["label_mismatches"]:
            print(f"  label_mismatches={entry['label_mismatches'][:args.mismatch_limit]}")
        if entry["missing_de_trials"]:
            print(f"  missing_de_trials={entry['missing_de_trials'][:args.mismatch_limit]}")

    print("[summary]")
    print(f"  total_subject_sessions={summary['total_subject_sessions']}")
    print(f"  total_raw_trials={summary['total_raw_trials']}")
    print(f"  total_de_trials={summary['total_de_trials']}")
    print(f"  label_match_rate_global={summary['label_match_rate_global']:.3f}")
    print(f"  sessions_missing_de_file={summary['sessions_missing_de_file']}")
    print(f"  sessions_trial_count_mismatch={summary['sessions_trial_count_mismatch']}")
    print(f"  sessions_label_mismatch={summary['sessions_label_mismatch']}")
    print(f"  sessions_with_mismatch={summary['sessions_with_mismatch']}")


if __name__ == "__main__":
    main()
