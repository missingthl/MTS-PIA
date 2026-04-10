from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.seed_raw_cnt import build_eeg62_view, load_one_raw
from datasets.seed_raw_trials import build_trial_index


def _hash_channels(names: list[str]) -> str:
    joined = ",".join(names).encode("utf-8")
    return hashlib.sha1(joined).hexdigest()


def _resolve_time_txt(cnt_path: str) -> str:
    cnt_dir = os.path.dirname(cnt_path)
    cand = os.path.join(cnt_dir, "time.txt")
    if os.path.isfile(cand):
        return cand
    fallback = os.path.join("data", "SEED", "SEED_EEG", "time.txt")
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError("time.txt not found near CNT or default SEED_EEG path")


def _resolve_stim_xlsx() -> str:
    cand = os.path.join("data", "SEED", "SEED_EEG", "SEED_stimulation.xlsx")
    if not os.path.isfile(cand):
        raise FileNotFoundError(f"SEED_stimulation.xlsx not found: {cand}")
    return cand


def convert_cnt(
    cnt_path: str,
    out_dir: str,
    *,
    channel_policy: str,
    dtype: str,
    out_format: str,
    overwrite: bool,
    locs_path: str,
    buffer_sec: float | None,
) -> dict:
    if channel_policy != "strict":
        raise ValueError(f"Unsupported channel policy: {channel_policy}")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    base = Path(cnt_path).stem
    if out_format == "fif":
        out_path = os.path.join(out_dir, f"{base}_eeg62_raw.fif")
    else:
        out_path = os.path.join(out_dir, f"{base}_eeg62_raw.npz")
    if os.path.isfile(out_path) and not overwrite:
        raise FileExistsError(f"Output exists: {out_path} (use --overwrite)")

    raw = load_one_raw(cnt_path, backend="cnt", preload=False, data_format=dtype)
    raw62, mapping = build_eeg62_view(raw, locs_path=locs_path)
    if out_format == "fif":
        if buffer_sec and buffer_sec > 0:
            raw62.save(out_path, overwrite=overwrite, buffer_size_sec=buffer_sec)
        else:
            raw62.save(out_path, overwrite=overwrite)
    else:
        data = raw62.get_data().astype("float32", copy=False)
        npz_payload = {
            "data": data,
            "sfreq": float(raw62.info.get("sfreq", 0.0)),
            "ch_names": raw62.ch_names,
        }
        import numpy as np

        np.savez_compressed(out_path, **npz_payload)

    time_txt = _resolve_time_txt(cnt_path)
    stim_xlsx = _resolve_stim_xlsx()
    trials = build_trial_index(cnt_path, time_txt, stim_xlsx)

    fs_raw = float(raw62.info.get("sfreq", 0.0))
    n_ch_raw = int(len(raw.ch_names))
    n_ch_used = int(len(raw62.ch_names))
    channel_hash = _hash_channels(mapping["selected_names"])

    meta = {
        "cnt_path": cnt_path,
        "out_path": out_path,
        "subject": trials[0].subject if trials else None,
        "session": trials[0].session if trials else None,
        "fs_raw": fs_raw,
        "n_ch_raw": n_ch_raw,
        "n_ch_used": n_ch_used,
        "channel_policy": channel_policy,
        "channel_hash": channel_hash,
        "dtype": dtype,
        "n_trials": len(trials),
    }
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnt", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--channel-policy", type=str, default="strict", choices=["strict"])
    parser.add_argument("--dtype", type=str, default="int16", choices=["int16", "int32"])
    parser.add_argument("--format", type=str, default="fif", choices=["fif", "npz"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--buffer-sec", type=float, default=0.0)
    parser.add_argument(
        "--locs-path",
        type=str,
        default="data/SEED/channel_62_pos.locs",
        help="channel_62_pos.locs path",
    )
    parser.add_argument("--meta-out", type=str, default=None)
    parser.add_argument("--manifest-out", type=str, default=None)
    parser.add_argument("--report-out", type=str, default=None)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=0)
    parser.add_argument("--spawn-subprocess", type=int, default=0)
    args = parser.parse_args()

    if args.manifest and args.cnt:
        raise ValueError("Use either --cnt or --manifest, not both")
    if not args.manifest and not args.cnt:
        raise ValueError("Either --cnt or --manifest must be provided")

    if args.cnt:
        meta = convert_cnt(
            args.cnt,
            args.out_dir,
            channel_policy=args.channel_policy,
            dtype=args.dtype,
            out_format=args.format,
            overwrite=args.overwrite,
            locs_path=args.locs_path,
            buffer_sec=args.buffer_sec,
        )
        print(f"[seed_raw_convert] cnt={args.cnt}")
        print(f"[seed_raw_convert] out={meta['out_path']}")
        print(f"[seed_raw_convert] fs_raw={meta['fs_raw']} n_trials={meta['n_trials']}")
        if args.meta_out:
            with open(args.meta_out, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        return

    manifest_path = os.path.abspath(args.manifest)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, list):
        raise ValueError("manifest must be a list of entries")

    entries = list(manifest)
    if args.end_idx and args.end_idx > 0:
        entries = entries[args.start_idx : args.end_idx]
    elif args.start_idx:
        entries = entries[args.start_idx :]
    if args.max_files and args.max_files > 0:
        entries = entries[: args.max_files]

    out_manifest_path = args.manifest_out or manifest_path
    report_path = args.report_out or "logs/seed_raw_convert_report.json"
    os.makedirs(os.path.dirname(out_manifest_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)

    converted = []
    failures = []
    skipped = []
    start_time = time.time()

    for entry in entries:
        cnt_path = entry.get("cnt_path")
        if not cnt_path:
            failures.append({"cnt_path": None, "error": "missing cnt_path"})
            continue
        base = Path(cnt_path).stem
        out_path = entry.get("out_path") or os.path.join(
            os.path.abspath(args.out_dir),
            f"{base}_eeg62_raw.{args.format}",
        )
        if os.path.isfile(out_path) and not args.overwrite:
            entry["status"] = "skipped"
            skipped.append(cnt_path)
            continue

        try:
            if args.spawn_subprocess:
                cmd = [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--cnt",
                    cnt_path,
                    "--out-dir",
                    args.out_dir,
                    "--channel-policy",
                    args.channel_policy,
                    "--dtype",
                    args.dtype,
                    "--format",
                    args.format,
                    "--locs-path",
                    args.locs_path,
                ]
                if args.buffer_sec and args.buffer_sec > 0:
                    cmd.extend(["--buffer-sec", str(args.buffer_sec)])
                if args.overwrite:
                    cmd.append("--overwrite")
                proc = subprocess.run(cmd)
                if proc.returncode != 0:
                    raise RuntimeError(f"subprocess failed rc={proc.returncode}")
                meta = {
                    "cnt_path": cnt_path,
                    "out_path": out_path,
                    "status": "converted",
                }
            else:
                meta = convert_cnt(
                    cnt_path,
                    args.out_dir,
                    channel_policy=args.channel_policy,
                    dtype=args.dtype,
                    out_format=args.format,
                    overwrite=args.overwrite,
                    locs_path=args.locs_path,
                    buffer_sec=args.buffer_sec,
                )
                meta["status"] = "converted"

            entry.update(meta)
            converted.append(cnt_path)
            print(f"[seed_raw_convert] ok {cnt_path} -> {entry.get('out_path')}")
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)
            failures.append({"cnt_path": cnt_path, "error": str(exc)})
            print(f"[seed_raw_convert] fail {cnt_path}: {exc}")

        with open(out_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    report = {
        "manifest_in": manifest_path,
        "manifest_out": out_manifest_path,
        "out_dir": os.path.abspath(args.out_dir),
        "format": args.format,
        "dtype": args.dtype,
        "total_entries": len(entries),
        "converted_ok": len(converted),
        "converted_fail": len(failures),
        "converted_skip": len(skipped),
        "failures": failures,
        "elapsed_sec": time.time() - start_time,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
