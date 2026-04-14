#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import os
import shutil
import sys

import numpy as np
from aeon.datasets import load_classification

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from datasets.aeon_fixedsplit_trials import AEON_FIXED_SPLIT_SPECS
from datasets.trial_dataset_factory import normalize_dataset_name


FIRST_BATCH = [
    "racketsports",
    "articularywordrecognition",
    "heartbeat",
    "selfregulationscp2",
    "libras",
    "japanesevowels",
]

SECOND_BATCH = [
    "cricket",
    "handwriting",
    "ering",
    "motorimagery",
    "ethanolconcentration",
]


def _resolve_dataset_keys(args: argparse.Namespace) -> List[str]:
    if args.datasets:
        return [normalize_dataset_name(x) for x in args.datasets.split(",") if x.strip()]
    if args.batch == "first":
        return list(FIRST_BATCH)
    if args.batch == "second":
        return list(SECOND_BATCH)
    if args.batch == "all":
        return list(FIRST_BATCH) + list(SECOND_BATCH)
    raise ValueError(f"unsupported batch: {args.batch}")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: dict) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_fixedsplit_npz(path: Path, *, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> None:
    _ensure_parent(path)
    np.savez_compressed(
        path,
        X_train=np.asarray(train_x, dtype=np.float32),
        y_train=np.asarray(train_y, dtype=np.int64),
        X_test=np.asarray(test_x, dtype=np.float32),
        y_test=np.asarray(test_y, dtype=np.int64),
    )


def _sync_raw_dataset_dir(*, src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.is_dir():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for child in src_dir.iterdir():
        target = dst_dir / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def _encode_labels(train_y: np.ndarray, test_y: np.ndarray, class_values: list[str] | None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_raw = [str(v) for v in np.asarray(train_y).tolist()]
    test_raw = [str(v) for v in np.asarray(test_y).tolist()]
    if class_values:
        ordered = [str(v) for v in class_values]
    else:
        ordered = sorted(set(train_raw + test_raw))
    label_map = {label: idx for idx, label in enumerate(ordered)}
    missing = sorted((set(train_raw) | set(test_raw)) - set(label_map.keys()))
    for label in missing:
        label_map[label] = len(label_map)
        ordered.append(label)
    train_enc = np.asarray([label_map[v] for v in train_raw], dtype=np.int64)
    test_enc = np.asarray([label_map[v] for v in test_raw], dtype=np.int64)
    return train_enc, test_enc, ordered


def _fetch_one(*, dataset_key: str, target_root: Path, extract_root: Path, overwrite: bool, sync_raw: bool) -> dict:
    spec = AEON_FIXED_SPLIT_SPECS[dataset_key]
    dataset_dir = target_root / spec.dataset_name
    npz_path = dataset_dir / f"{spec.dataset_name}_fixedsplit.npz"
    meta_path = dataset_dir / f"{spec.dataset_name}_fixedsplit_meta.json"
    raw_src_dir = extract_root / spec.dataset_name

    if npz_path.is_file() and meta_path.is_file() and not overwrite:
        if sync_raw:
            _sync_raw_dataset_dir(src_dir=raw_src_dir, dst_dir=dataset_dir)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["status"] = "cached"
        return meta

    train_x, train_y, train_meta = load_classification(
        spec.dataset_name,
        split="train",
        return_metadata=True,
        extract_path=str(extract_root),
        load_equal_length=True,
    )
    test_x, test_y, test_meta = load_classification(
        spec.dataset_name,
        split="test",
        return_metadata=True,
        extract_path=str(extract_root),
        load_equal_length=True,
    )

    train_x = np.asarray(train_x, dtype=np.float32)
    test_x = np.asarray(test_x, dtype=np.float32)
    class_values = train_meta.get("class_values") or test_meta.get("class_values") or []
    train_y, test_y, class_values = _encode_labels(train_y, test_y, class_values)

    if sync_raw:
        _sync_raw_dataset_dir(src_dir=raw_src_dir, dst_dir=dataset_dir)

    if train_x.ndim != 3 or test_x.ndim != 3:
        raise ValueError(
            f"{spec.dataset_name}: expected equal-length 3D arrays, "
            f"got train={train_x.shape}, test={test_x.shape}"
        )
    if train_x.shape[1] != test_x.shape[1] or train_x.shape[2] != test_x.shape[2]:
        raise ValueError(
            f"{spec.dataset_name}: train/test shape mismatch: train={train_x.shape}, test={test_x.shape}"
        )

    _save_fixedsplit_npz(
        npz_path,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
    )

    meta = {
        "status": "downloaded",
        "dataset_key": spec.dataset_key,
        "dataset_name": spec.dataset_name,
        "sfreq": float(spec.sfreq),
        "source": "aeon.load_classification",
        "load_equal_length": True,
        "extract_root": str(extract_root),
        "target_npz": str(npz_path),
        "train_shape": [int(v) for v in train_x.shape],
        "test_shape": [int(v) for v in test_x.shape],
        "num_classes": int(max(train_y.max(initial=0), test_y.max(initial=0)) + 1),
        "class_values": class_values,
        "train_size": int(train_x.shape[0]),
        "test_size": int(test_x.shape[0]),
        "metadata_train": train_meta,
        "metadata_test": test_meta,
    }
    _write_json(meta_path, meta)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch aeon classification datasets into fixedsplit.npz format.")
    parser.add_argument("--datasets", type=str, default="", help="Comma-separated dataset keys/names.")
    parser.add_argument("--batch", type=str, default="first", choices=["first", "second", "all"])
    parser.add_argument("--target-root", type=str, default="data")
    parser.add_argument("--extract-root", type=str, default="data/_cache/aeon")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--sync-raw", action="store_true", default=True)
    args = parser.parse_args()

    dataset_keys = _resolve_dataset_keys(args)
    target_root = Path(args.target_root).expanduser().resolve()
    extract_root = Path(args.extract_root).expanduser().resolve()
    extract_root.mkdir(parents=True, exist_ok=True)

    results: List[dict] = []
    for dataset_key in dataset_keys:
        if dataset_key not in AEON_FIXED_SPLIT_SPECS:
            raise ValueError(f"Unsupported aeon fixed-split dataset: {dataset_key}")
        meta = _fetch_one(
            dataset_key=dataset_key,
            target_root=target_root,
            extract_root=extract_root,
            overwrite=bool(args.overwrite),
            sync_raw=bool(args.sync_raw),
        )
        results.append(meta)
        print(
            f"[aeon-fetch] {meta['dataset_name']} status={meta['status']} "
            f"train={meta['train_shape']} test={meta['test_shape']} classes={meta['num_classes']}",
            flush=True,
        )

    summary = {
        "target_root": str(target_root),
        "extract_root": str(extract_root),
        "datasets": results,
    }
    summary_path = target_root / "_cache" / "aeon" / "fetch_summary.json"
    _write_json(summary_path, summary)
    print(f"[done] wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
