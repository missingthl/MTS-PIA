from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


def _subject_sort_key(subject: str) -> Tuple[int, str]:
    try:
        return (0, f"{int(subject):04d}")
    except (ValueError, TypeError):
        return (1, str(subject))


def _load_manifest(path: str) -> Tuple[List[dict], str]:
    manifest_path = None
    if "*" in path or "?" in path:
        matches = sorted(glob.glob(path))
        if matches:
            manifest_path = matches[-1]
    elif os.path.isfile(path):
        manifest_path = path

    if not manifest_path:
        raise FileNotFoundError(f"manifest not found: {path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "trials" not in data:
            raise ValueError(f"manifest dict missing 'trials': {manifest_path}")
        return data["trials"], manifest_path
    if isinstance(data, list):
        return data, manifest_path
    raise ValueError(f"unsupported manifest format: {manifest_path}")


def _subject_list_from_manifest(
    rows: List[dict],
    subject_list: Optional[str],
    max_subjects: int,
) -> List[str]:
    subjects_all = sorted({str(r["subject"]) for r in rows}, key=_subject_sort_key)
    if subject_list:
        requested = [s.strip() for s in subject_list.split(",") if s.strip()]
        subjects = [s for s in requested if s in subjects_all]
        missing = [s for s in requested if s not in subjects_all]
        if missing:
            print(f"[raw_v1_subproc] Warning: subjects not found: {missing}")
    else:
        subjects = list(subjects_all)
    if max_subjects and max_subjects > 0:
        subjects = subjects[: int(max_subjects)]
    return subjects


def run_subprocess(
    *,
    manifest_path: str,
    seed_raw_root: str,
    raw_backend: str,
    out_prefix: str,
    raw_window_sec: float,
    raw_window_hop_sec: float,
    raw_resample_fs: float,
    raw_bands: str,
    raw_cov: str,
    raw_logmap_eps: float,
    raw_seq_save_format: str,
    spd_eps: float,
    spd_eps_mode: str,
    spd_eps_alpha: float,
    spd_eps_floor_mult: float,
    spd_eps_ceil_mult: float,
    clf: str,
    trial_protocol: str,
    raw_save_trial: str,
    raw_mem_debug: int,
    raw_mem_interval: int,
    raw_subject_list: Optional[str],
    raw_max_subjects: int,
    raw_stop_on_error: bool,
    raw_filter_chunk: int,
    raw_resample_chunk: int,
    raw_cnt_subprocess: int,
    raw_time_unit: Optional[str],
    raw_trial_offset_sec: float,
) -> None:
    if trial_protocol == "loso_subject":
        raise ValueError(
            "subprocess runner does not support loso_subject "
            "(requires full-subject training). Use inproc instead."
        )

    rows, used_manifest = _load_manifest(manifest_path)
    subjects = _subject_list_from_manifest(rows, raw_subject_list, raw_max_subjects)
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["MNE_USE_CUDA"] = "false"

    per_subject_reports: List[str] = []
    split_rows: List[dict] = []
    failed_subjects: List[str] = []
    subject_logs: Dict[str, str] = {}

    for subject in subjects:
        sub_prefix = f"{out_prefix}_sub{subject}"
        log_path = f"{out_prefix}_sub{subject}.log"
        subject_logs[str(subject)] = log_path
        cmd = [
            sys.executable,
            os.path.join("scripts", "run_manifold_raw_subject.py"),
            "--dataset",
            "seed1",
            "--raw-manifest",
            used_manifest,
            "--subject",
            str(subject),
            "--seed-raw-root",
            seed_raw_root,
            "--seed-raw-backend",
            raw_backend,
            "--raw-window-sec",
            str(raw_window_sec),
            "--raw-window-hop-sec",
            str(raw_window_hop_sec),
            "--raw-resample-fs",
            str(raw_resample_fs),
            "--raw-bands",
            raw_bands,
            "--raw-time-unit",
            str(raw_time_unit) if raw_time_unit is not None else "",
            "--raw-trial-offset-sec",
            str(raw_trial_offset_sec),
            "--raw-cov",
            raw_cov,
            "--raw-logmap-eps",
            str(raw_logmap_eps),
            "--raw-seq-save-format",
            raw_seq_save_format,
            "--spd-eps",
            str(spd_eps),
            "--spd-eps-mode",
            str(spd_eps_mode),
            "--spd-eps-alpha",
            str(spd_eps_alpha),
            "--spd-eps-floor-mult",
            str(spd_eps_floor_mult),
            "--spd-eps-ceil-mult",
            str(spd_eps_ceil_mult),
            "--clf",
            clf,
            "--trial-protocol",
            trial_protocol,
            "--raw-save-trial",
            raw_save_trial,
            "--raw-mem-debug",
            str(raw_mem_debug),
            "--raw-mem-interval",
            str(raw_mem_interval),
            "--raw-filter-chunk",
            str(raw_filter_chunk),
            "--raw-resample-chunk",
            str(raw_resample_chunk),
            "--raw-cnt-subprocess",
            str(raw_cnt_subprocess),
            "--out-prefix",
            out_prefix,
        ]

        print(f"[raw_v1_subproc] launch subject={subject} log={log_path}")
        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.run(cmd, env=env, stdout=log_file, stderr=log_file)
        if proc.returncode != 0:
            failed_subjects.append(str(subject))
            print(f"[raw_v1_subproc] subject={subject} failed rc={proc.returncode}")
            if raw_stop_on_error:
                break
            continue

        report_path = f"{sub_prefix}_report.json"
        if not os.path.isfile(report_path):
            failed_subjects.append(str(subject))
            print(f"[raw_v1_subproc] missing report for subject={subject}: {report_path}")
            if raw_stop_on_error:
                break
            continue

        per_subject_reports.append(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        for row in report.get("splits", []):
            split_rows.append(row)

    acc_list = [row["acc"] for row in split_rows]
    f1_list = [row["macro_f1"] for row in split_rows]
    acc_mean = float(np.mean(acc_list)) if acc_list else 0.0
    acc_std = float(np.std(acc_list)) if acc_list else 0.0
    f1_mean = float(np.mean(f1_list)) if f1_list else 0.0
    f1_std = float(np.std(f1_list)) if f1_list else 0.0

    report = {
        "protocol": trial_protocol,
        "runner": "subprocess",
        "raw_backend": raw_backend,
        "window_sec": raw_window_sec,
        "hop_sec": raw_window_hop_sec,
        "resample_fs": raw_resample_fs,
        "cov": raw_cov,
        "logmap_eps": raw_logmap_eps,
        "clf": clf,
        "subjects_total": len(subjects),
        "subjects_done": len(per_subject_reports),
        "subjects_failed": failed_subjects,
        "per_subject_reports": per_subject_reports,
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "macro_f1_mean": f1_mean,
        "macro_f1_std": f1_std,
        "splits": split_rows,
    }

    report_json = f"{out_prefix}_report.json"
    report_csv = f"{out_prefix}_report.csv"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(report_csv, "w", encoding="utf-8") as f:
        f.write("name,subject,n_train,n_test,acc,macro_f1\n")
        for row in split_rows:
            f.write(
                f"{row['name']},{row['subject']},{row['n_train']},"
                f"{row['n_test']},{row['acc']:.6f},{row['macro_f1']:.6f}\n"
            )

    print(
        f"[raw_v1_subproc] done subjects={len(per_subject_reports)} failed={len(failed_subjects)} "
        f"acc={acc_mean:.4f}±{acc_std:.4f} f1={f1_mean:.4f}±{f1_std:.4f}"
    )
