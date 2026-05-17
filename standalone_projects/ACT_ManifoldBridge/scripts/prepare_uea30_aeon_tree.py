#!/usr/bin/env python3
"""Prepare an isolated UEA30 data tree via aeon.

This script intentionally keeps the UEA30 archive separate from the project's
existing hand-maintained ``data/<Dataset>`` folders.  It uses aeon's downloader
without loading the datasets into memory, then writes a manifest that can be
used before extending ACT/CSTA loaders or launching UEA30 experiments.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


UEA30_DATASETS: tuple[str, ...] = (
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "CharacterTrajectories",
    "Cricket",
    "DuckDuckGeese",
    "EigenWorms",
    "Epilepsy",
    "ERing",
    "EthanolConcentration",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "InsectWingbeat",
    "JapaneseVowels",
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PEMS-SF",
    "PenDigits",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    "StandWalkJump",
    "UWaveGestureLibrary",
)


@dataclass
class DatasetTreeStatus:
    dataset: str
    key: str
    dataset_dir: str
    train_ts_exists: bool
    test_ts_exists: bool
    downloaded: bool
    status: str
    error: str = ""


def _parse_dataset_arg(text: str) -> list[str]:
    if text.strip().lower() in {"all", "uea30"}:
        return list(UEA30_DATASETS)
    wanted = [x.strip() for x in text.split(",") if x.strip()]
    known = {x.lower(): x for x in UEA30_DATASETS}
    out: list[str] = []
    for item in wanted:
        key = item.lower()
        if key not in known:
            raise ValueError(f"Unknown UEA30 dataset: {item}")
        out.append(known[key])
    return out


def _status_for_dataset(root: Path, dataset: str, downloaded: bool = False, error: str = "") -> DatasetTreeStatus:
    dataset_dir = root / dataset
    train_path = dataset_dir / f"{dataset}_TRAIN.ts"
    test_path = dataset_dir / f"{dataset}_TEST.ts"
    train_ok = train_path.is_file()
    test_ok = test_path.is_file()
    if error:
        status = "error"
    elif train_ok and test_ok:
        status = "ready"
    elif dataset_dir.exists():
        status = "partial"
    else:
        status = "missing"
    return DatasetTreeStatus(
        dataset=dataset,
        key=dataset.lower().replace("-", ""),
        dataset_dir=str(dataset_dir),
        train_ts_exists=train_ok,
        test_ts_exists=test_ok,
        downloaded=downloaded,
        status=status,
        error=error,
    )


def _write_manifest(out_dir: Path, rows: Iterable[DatasetTreeStatus]) -> None:
    rows = list(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "uea30_aeon_manifest.json"
    csv_path = out_dir / "uea30_aeon_manifest.csv"
    json_path.write_text(json.dumps([asdict(r) for r in rows], indent=2) + "\n", encoding="utf-8")
    header = [
        "dataset",
        "key",
        "dataset_dir",
        "train_ts_exists",
        "test_ts_exists",
        "downloaded",
        "status",
        "error",
    ]
    lines = [",".join(header)]
    for r in rows:
        vals = [str(getattr(r, h)).replace("\n", " ").replace(",", ";") for h in header]
        lines.append(",".join(vals))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "data" / "UEA30_aeon",
        help="Isolated aeon extraction root. Defaults to data/UEA30_aeon.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated UEA30 dataset names, or 'all'.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing datasets via aeon. Without this flag the script only audits the tree.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first download error.",
    )
    parser.add_argument(
        "--socket-timeout-sec",
        type=float,
        default=120.0,
        help="Global socket timeout for aeon downloads; avoids indefinite stalls.",
    )
    args = parser.parse_args()

    if args.socket_timeout_sec > 0:
        socket.setdefaulttimeout(float(args.socket_timeout_sec))

    datasets = _parse_dataset_arg(args.datasets)
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[DatasetTreeStatus] = []
    if args.download:
        from aeon.datasets._data_loaders import download_dataset

    for dataset in datasets:
        downloaded = False
        error = ""
        before = _status_for_dataset(out_root, dataset)
        if args.download and before.status != "ready":
            try:
                print(f"[download] {dataset} -> {out_root}", flush=True)
                download_dataset(dataset, save_path=str(out_root))
                downloaded = True
            except Exception as exc:  # keep audit going unless fail-fast is requested
                error = f"{type(exc).__name__}: {exc}"
                print(f"[error] {dataset}: {error}", flush=True)
                if args.fail_fast:
                    rows.append(_status_for_dataset(out_root, dataset, downloaded=downloaded, error=error))
                    _write_manifest(out_root, rows)
                    raise
        rows.append(_status_for_dataset(out_root, dataset, downloaded=downloaded, error=error))

    _write_manifest(out_root, rows)
    ready = sum(r.status == "ready" for r in rows)
    partial = sum(r.status == "partial" for r in rows)
    missing = sum(r.status == "missing" for r in rows)
    errors = sum(r.status == "error" for r in rows)
    print(
        f"Wrote manifest to {out_root}. ready={ready} partial={partial} "
        f"missing={missing} errors={errors} total={len(rows)}"
    )


if __name__ == "__main__":
    main()
