from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.seed_raw_cnt import _load_locs_names, _normalize_name  # noqa: PLC2701


def _hash_channels(names: list[str]) -> str:
    joined = ",".join(names).encode("utf-8")
    return hashlib.sha1(joined).hexdigest()


def _load_manifest(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"manifest must be a list: {path}")
    return data


def _ensure_mne():
    try:
        import mne  # noqa: F401
    except ImportError as exc:
        raise ImportError("mne is required to read FIF files. Install via `pip install mne`.") from exc


def _check_strict_channel_order(raw_names: list[str], locs_names: list[str]) -> list[dict]:
    raw_norm = [_normalize_name(n) for n in raw_names]
    locs_norm = [_normalize_name(n) for n in locs_names]
    mismatches = []
    for idx, (r, l) in enumerate(zip(raw_norm, locs_norm)):
        if r != l:
            mismatches.append(
                {
                    "index": idx,
                    "raw_name": raw_names[idx],
                    "locs_name": locs_names[idx],
                }
            )
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fif-root", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--channel-policy", type=str, default="strict", choices=["strict"])
    parser.add_argument(
        "--locs-path",
        type=str,
        default="data/SEED/channel_62_pos.locs",
        help="channel_62_pos.locs path",
    )
    parser.add_argument("--report-out", type=str, default="logs/seed_raw_fif_validate_report.json")
    args = parser.parse_args()

    _ensure_mne()
    import mne  # noqa: WPS433

    manifest_path = os.path.abspath(args.manifest)
    fif_root = os.path.abspath(args.fif_root)
    report_out = os.path.abspath(args.report_out)
    os.makedirs(os.path.dirname(report_out) or ".", exist_ok=True)

    entries = _load_manifest(manifest_path)
    locs_names = _load_locs_names(args.locs_path)
    ok = 0
    missing = 0
    failed = 0
    failures: list[dict] = []
    channel_hashes: set[str] = set()

    for entry in entries:
        out_path = entry.get("out_path")
        if not out_path:
            missing += 1
            failures.append({"cnt_path": entry.get("cnt_path"), "error": "missing out_path"})
            continue
        out_path = os.path.abspath(out_path)
        if not out_path.startswith(fif_root):
            out_path = os.path.join(fif_root, Path(out_path).name)
        if not os.path.isfile(out_path):
            missing += 1
            failures.append({"cnt_path": entry.get("cnt_path"), "out_path": out_path, "error": "fif missing"})
            continue

        try:
            raw = mne.io.read_raw_fif(out_path, preload=False, verbose="ERROR")
            raw_names = list(raw.ch_names)
            if args.channel_policy == "strict":
                if len(raw_names) != len(locs_names):
                    raise ValueError(f"expected 62 channels, got {len(raw_names)}")
                mismatches = _check_strict_channel_order(raw_names, locs_names)
                if mismatches:
                    raise ValueError(f"channel order mismatch: first={mismatches[:3]}")
            channel_hashes.add(_hash_channels(raw_names))
            ok += 1
        except Exception as exc:
            failed += 1
            failures.append({"cnt_path": entry.get("cnt_path"), "out_path": out_path, "error": str(exc)})

    report = {
        "manifest": manifest_path,
        "fif_root": fif_root,
        "channel_policy": args.channel_policy,
        "total_entries": len(entries),
        "ok": ok,
        "missing": missing,
        "failed": failed,
        "channel_hashes": sorted(channel_hashes),
        "failures": failures,
    }
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(
        "[seed_raw_fif_validate] ok="
        f"{ok} missing={missing} failed={failed} channel_hashes={len(channel_hashes)}"
    )
    print(f"[seed_raw_fif_validate] report={report_out}")


if __name__ == "__main__":
    main()
