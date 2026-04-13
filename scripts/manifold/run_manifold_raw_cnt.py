from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from runners.manifold_raw_v1 import ManifoldRawV1Runner


def _load_manifest(path: str) -> List[dict]:
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
        return data["trials"]
    if isinstance(data, list):
        return data
    raise ValueError(f"unsupported manifest format: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="seed1")
    parser.add_argument("--raw-manifest", type=str, required=True)
    parser.add_argument("--cnt-path", type=str, required=True)
    parser.add_argument(
        "--seed-raw-root",
        type=str,
        default="data/SEED/SEED_EEG/SEED_RAW_EEG",
    )
    parser.add_argument(
        "--seed-raw-backend",
        type=str,
        default="fif",
        choices=["cnt", "fif"],
    )
    parser.add_argument("--raw-window-sec", type=float, default=4.0)
    parser.add_argument("--raw-window-hop-sec", type=float, default=4.0)
    parser.add_argument("--raw-resample-fs", type=float, default=0.0)
    parser.add_argument("--raw-resample-chunk", type=int, default=0)
    parser.add_argument(
        "--raw-bands",
        type=str,
        default="delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50",
    )
    parser.add_argument(
        "--raw-time-unit",
        type=str,
        default="",
        help="time unit for time.txt (samples@1000, samples@200, seconds)",
    )
    parser.add_argument(
        "--raw-trial-offset-sec",
        type=float,
        default=-3.0,
        help="global trial offset in seconds (applied to start/end)",
    )
    parser.add_argument("--raw-cov", type=str, default="shrinkage_oas")
    parser.add_argument("--raw-logmap-eps", type=float, default=1e-6)
    parser.add_argument(
        "--raw-seq-save-format",
        type=str,
        default="vec_utri",
        choices=["vec_utri", "cov_spd"],
    )
    parser.add_argument("--spd-eps", type=float, default=1e-5)
    parser.add_argument(
        "--spd-eps-mode",
        type=str,
        default="relative_trace",
        choices=["absolute", "relative_trace", "relative_diag"],
    )
    parser.add_argument("--spd-eps-alpha", type=float, default=1e-2)
    parser.add_argument("--spd-eps-floor-mult", type=float, default=1e-6)
    parser.add_argument("--spd-eps-ceil-mult", type=float, default=1e-1)
    parser.add_argument("--raw-filter-chunk", type=int, default=0)
    parser.add_argument("--clf", type=str, default="ridge")
    parser.add_argument(
        "--trial-protocol",
        type=str,
        default="session_holdout",
        choices=["session_holdout", "loso_subject"],
    )
    parser.add_argument(
        "--raw-save-trial",
        type=str,
        default="yes",
        choices=["auto", "yes", "no"],
    )
    parser.add_argument("--raw-mem-debug", type=int, default=0)
    parser.add_argument("--raw-mem-interval", type=int, default=0)
    parser.add_argument("--out-prefix", type=str, default=None)
    args = parser.parse_args()

    if args.dataset != "seed1":
        raise ValueError("run_manifold_raw_cnt supports seed1 only")

    rows = _load_manifest(args.raw_manifest)
    if args.seed_raw_backend != "fif":
        raise ValueError(
            "manifold_raw_v1_frozen: CNT backend is not allowed. "
            "Use offline conversion to FIF and set --seed-raw-backend fif."
        )
    if args.seed_raw_backend == "fif":
        rows = [r for r in rows if r.get("out_path") == args.cnt_path]
    else:
        rows = [r for r in rows if (r.get("source_cnt_path") or r.get("cnt_path")) == args.cnt_path]
    if not rows:
        raise ValueError(f"No trials found for cnt {args.cnt_path}")

    if args.out_prefix:
        out_prefix = args.out_prefix
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_prefix = os.path.join("logs", f"manifold_raw_v1_cnt_{ts}")

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    manifest_path = f"{out_prefix}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    runner = ManifoldRawV1Runner(
        raw_manifest=manifest_path,
        seed_raw_root=args.seed_raw_root,
        raw_window_sec=args.raw_window_sec,
        raw_window_hop_sec=args.raw_window_hop_sec,
        raw_resample_fs=args.raw_resample_fs,
        raw_bands=args.raw_bands,
        raw_time_unit=args.raw_time_unit or None,
        raw_trial_offset_sec=args.raw_trial_offset_sec,
        raw_cov=args.raw_cov,
        raw_logmap_eps=args.raw_logmap_eps,
        raw_seq_save_format=args.raw_seq_save_format,
        spd_eps=args.spd_eps,
        spd_eps_mode=args.spd_eps_mode,
        spd_eps_alpha=args.spd_eps_alpha,
        spd_eps_floor_mult=args.spd_eps_floor_mult,
        spd_eps_ceil_mult=args.spd_eps_ceil_mult,
        clf=args.clf,
        trial_protocol=args.trial_protocol,
        out_prefix=out_prefix,
        raw_chunk_by="none",
        raw_max_subjects=0,
        raw_subject_list=None,
        raw_mem_debug=args.raw_mem_debug,
        raw_mem_interval=args.raw_mem_interval,
        raw_save_trial=args.raw_save_trial,
        raw_filter_chunk=args.raw_filter_chunk,
        raw_resample_chunk=args.raw_resample_chunk,
        raw_backend=args.seed_raw_backend,
    )
    runner.run()


if __name__ == "__main__":
    main()
