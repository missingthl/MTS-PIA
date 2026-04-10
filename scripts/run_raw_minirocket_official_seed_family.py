#!/usr/bin/env python
"""MiniROCKET baseline under native official SEED-family protocols.

Protocol locks:
- SEED: per session first 9 trials train, last 6 trials test
- SEED_IV: per session first 16 trials train, last 8 trials test
- SEED_V: per session first 9 trials train, last 6 trials test
- Raw EEG is segmented into official 4-second non-overlapping windows
- No extra random resplit, no overlap, no augmentation, no input z-score
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.trial_dataset_factory import (  # noqa: E402
    DEFAULT_SEEDIV_ROOT,
    DEFAULT_SEEDV_ROOT,
    load_trials_for_dataset,
    normalize_dataset_name,
)
from scripts.resource_probe_utils import ResourceProbeLogger  # noqa: E402


OFFICIAL_SEED_DATASETS = {"seed1", "seediv", "seedv"}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _write_json(path: str, obj) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(obj), f, ensure_ascii=False, indent=2)


def _build_model(n_kernels: int, random_state: int, n_jobs: int):
    try:
        from aeon.classification.convolution_based import MiniRocketClassifier
    except Exception as e:
        raise ImportError(
            "MiniROCKET official baseline requires aeon. Install in pia env: `pip install aeon`."
        ) from e
    return MiniRocketClassifier(
        n_kernels=int(n_kernels),
        random_state=int(random_state),
        n_jobs=int(n_jobs),
    )


def _ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        s = str(v)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _split_hash(train_ids: Sequence[str], test_ids: Sequence[str]) -> str:
    payload = json.dumps(
        {"train": [str(x) for x in train_ids], "test": [str(x) for x in test_ids]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _zero_based_trial_index(dataset: str, trial_value: int) -> int:
    ds = normalize_dataset_name(dataset)
    t = int(trial_value)
    if ds == "seed1":
        if 1 <= t <= 15:
            return t - 1
        if 0 <= t <= 14:
            return t
        raise ValueError(f"Unexpected SEED trial index: {trial_value}")
    if ds == "seediv":
        if 0 <= t <= 23:
            return t
        if 1 <= t <= 24:
            return t - 1
        raise ValueError(f"Unexpected SEED_IV trial index: {trial_value}")
    if ds == "seedv":
        if 0 <= t <= 14:
            return t
        if 1 <= t <= 15:
            return t - 1
        raise ValueError(f"Unexpected SEED_V trial index: {trial_value}")
    raise ValueError(f"Unsupported dataset: {dataset}")


def _official_protocol_config(dataset: str) -> Dict[str, object]:
    ds = normalize_dataset_name(dataset)
    if ds == "seed1":
        return {
            "train_trial_last": 8,
            "test_trial_first": 9,
            "n_trials_per_session": 15,
            "protocol": "official_seed_first9_last6_per_session_raw_4s_nonoverlap",
        }
    if ds == "seediv":
        return {
            "train_trial_last": 15,
            "test_trial_first": 16,
            "n_trials_per_session": 24,
            "protocol": "official_seediv_first16_last8_per_session_raw_4s_nonoverlap",
        }
    if ds == "seedv":
        return {
            "train_trial_last": 8,
            "test_trial_first": 9,
            "n_trials_per_session": 15,
            "protocol": "official_seedv_first9_last6_per_session_raw_4s_nonoverlap",
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def _load_trials(dataset: str, args: argparse.Namespace) -> List[Dict]:
    return load_trials_for_dataset(
        dataset=dataset,
        processed_root=args.processed_root,
        stim_xlsx=args.stim_xlsx,
        seediv_root=args.seediv_root,
        seedv_root=args.seedv_root,
    )


def _split_trials_official(dataset: str, all_trials: Sequence[Dict]) -> Tuple[List[Dict], List[Dict], Dict[str, object]]:
    cfg = _official_protocol_config(dataset)
    train_trials: List[Dict] = []
    test_trials: List[Dict] = []
    for tr in all_trials:
        t0 = _zero_based_trial_index(dataset, int(tr["trial"]))
        if t0 < 0 or t0 >= int(cfg["n_trials_per_session"]):
            raise ValueError(f"Trial index out of range for {dataset}: {tr['trial']}")
        if t0 <= int(cfg["train_trial_last"]):
            train_trials.append(tr)
        else:
            test_trials.append(tr)

    if not train_trials or not test_trials:
        raise RuntimeError(f"Official protocol produced empty split for dataset={dataset}.")

    train_ids = [str(t["trial_id_str"]) for t in train_trials]
    test_ids = [str(t["trial_id_str"]) for t in test_trials]
    overlap = set(train_ids).intersection(test_ids)
    if overlap:
        raise RuntimeError(f"Split leakage: {len(overlap)} overlapping trials")

    return train_trials, test_trials, {
        "protocol": str(cfg["protocol"]),
        "train_count_trials": int(len(train_trials)),
        "test_count_trials": int(len(test_trials)),
        "train_trial_ids": train_ids,
        "test_trial_ids": test_ids,
        "split_hash": _split_hash(train_ids, test_ids),
    }


def _window_starts_nonoverlap(n_samples: int, win_len: int) -> np.ndarray:
    n_windows = int(n_samples) // int(win_len)
    if n_windows <= 0:
        return np.asarray([], dtype=np.int32)
    return (np.arange(n_windows, dtype=np.int32) * int(win_len)).astype(np.int32, copy=False)


def _build_window_index(trials: Sequence[Dict], win_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], int]:
    trial_pos: List[int] = []
    start_pos: List[int] = []
    y_all: List[int] = []
    tid_all: List[str] = []
    per_trial_counts: Dict[str, int] = {}
    n_short_trials = 0

    for i, tr in enumerate(trials):
        x = np.asarray(tr["x_trial"], dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D trial for {tr['trial_id_str']}, got {x.shape}")
        if x.shape[0] > x.shape[1]:
            x = x.T
        if x.shape[1] < int(win_len):
            n_short_trials += 1
            per_trial_counts[str(tr["trial_id_str"])] = 0
            continue
        starts = _window_starts_nonoverlap(int(x.shape[1]), int(win_len))
        tid = str(tr["trial_id_str"])
        per_trial_counts[tid] = int(starts.size)
        for s in starts.tolist():
            trial_pos.append(int(i))
            start_pos.append(int(s))
            y_all.append(int(tr["label"]))
            tid_all.append(tid)

    return (
        np.asarray(trial_pos, dtype=np.int32),
        np.asarray(start_pos, dtype=np.int32),
        np.asarray(y_all, dtype=np.int64),
        np.asarray(tid_all, dtype=object),
        per_trial_counts,
        int(n_short_trials),
    )


def _materialize_train_windows(
    trials: Sequence[Dict],
    win_len: int,
    *,
    seed_dir: str,
    memmap_threshold_bytes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    trial_pos, start_pos, y_train, tid_train, per_trial_counts, n_short_trials = _build_window_index(
        trials=trials,
        win_len=win_len,
    )
    if y_train.size <= 0:
        raise RuntimeError("No train windows generated under official protocol.")

    c = int(np.asarray(trials[0]["x_trial"]).shape[0])
    n_items = int(y_train.shape[0])
    n_bytes = int(n_items * c * int(win_len) * np.dtype(np.float32).itemsize)
    mmap_path = None
    if n_bytes >= int(memmap_threshold_bytes):
        _ensure_dir(seed_dir)
        mmap_path = os.path.join(seed_dir, "train_windows_official.f32.mmap")
        x_train = np.memmap(
            mmap_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_items, c, int(win_len)),
        )
        storage = "memmap"
    else:
        x_train = np.empty((n_items, c, int(win_len)), dtype=np.float32)
        storage = "ram"

    for j, (trial_idx, start_idx) in enumerate(zip(trial_pos.tolist(), start_pos.tolist())):
        x = np.asarray(trials[int(trial_idx)]["x_trial"], dtype=np.float32)
        if x.shape[0] > x.shape[1]:
            x = x.T
        x_train[j] = x[:, int(start_idx) : int(start_idx) + int(win_len)].astype(np.float32, copy=False)

    if isinstance(x_train, np.memmap):
        x_train.flush()

    meta = {
        "train_window_storage": storage,
        "train_window_memmap_path": mmap_path,
        "train_window_mem_bytes": int(n_bytes),
        "total_train_windows": int(n_items),
        "per_trial_window_counts_train": per_trial_counts,
        "n_short_trials_train": int(n_short_trials),
    }
    return x_train, y_train, tid_train, meta


def _iter_test_window_chunks(trials: Sequence[Dict], win_len: int, chunk_size: int):
    x_parts: List[np.ndarray] = []
    y_parts: List[int] = []
    tid_parts: List[str] = []
    n_short_trials = 0

    for tr in trials:
        x = np.asarray(tr["x_trial"], dtype=np.float32)
        if x.shape[0] > x.shape[1]:
            x = x.T
        starts = _window_starts_nonoverlap(int(x.shape[1]), int(win_len))
        if starts.size <= 0:
            n_short_trials += 1
            continue
        label = int(tr["label"])
        tid = str(tr["trial_id_str"])
        for s in starts.tolist():
            x_parts.append(x[:, int(s) : int(s) + int(win_len)].astype(np.float32, copy=False))
            y_parts.append(label)
            tid_parts.append(tid)
            if len(x_parts) >= int(chunk_size):
                yield (
                    np.stack(x_parts, axis=0).astype(np.float32, copy=False),
                    np.asarray(y_parts, dtype=np.int64),
                    np.asarray(tid_parts, dtype=object),
                    int(n_short_trials),
                )
                x_parts.clear()
                y_parts.clear()
                tid_parts.clear()
                n_short_trials = 0

    if x_parts:
        yield (
            np.stack(x_parts, axis=0).astype(np.float32, copy=False),
            np.asarray(y_parts, dtype=np.int64),
            np.asarray(tid_parts, dtype=object),
            int(n_short_trials),
        )


def _count_by_label(trials: Sequence[Dict]) -> Dict[str, int]:
    c = Counter(int(t["label"]) for t in trials)
    return {str(int(k)): int(v) for k, v in sorted(c.items())}


def _aggregate_trial_majority(y_true_win: np.ndarray, y_pred_win: np.ndarray, tid_win: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true_win, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred_win, dtype=np.int64).ravel()
    tids = np.asarray(tid_win, dtype=object)

    out_true: List[int] = []
    out_pred: List[int] = []
    for tid in sorted(_ordered_unique(tids.tolist())):
        idx = np.where(tids == tid)[0]
        yy = y_true[idx]
        pp = y_pred[idx]
        out_true.append(int(yy[0]))
        out_pred.append(int(Counter(pp.tolist()).most_common(1)[0][0]))
    return np.asarray(out_true, dtype=np.int64), np.asarray(out_pred, dtype=np.int64)


def _count_stats(counts: Dict[str, int]) -> Dict[str, float]:
    if not counts:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    vals = np.asarray(list(counts.values()), dtype=np.float64)
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _run_dataset(dataset: str, args: argparse.Namespace) -> Dict[str, object]:
    ds = normalize_dataset_name(dataset)
    out_dir = os.path.join(args.out_root, ds)
    seed_dir = os.path.join(out_dir, f"seed{int(args.seed)}")
    _ensure_dir(seed_dir)
    probe = ResourceProbeLogger(seed_dir) if bool(args.resource_probe) else None
    current_stage = "init"

    try:
        current_stage = "load_trials"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        all_trials = _load_trials(ds, args)
        if probe is not None:
            probe.mark_stage_end(current_stage, note=f"trials={len(all_trials)}")

        current_stage = "split_trials"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        train_trials, test_trials, split_meta = _split_trials_official(ds, all_trials)
        train_class_counts = _count_by_label(train_trials)
        test_class_counts = _count_by_label(test_trials)
        if probe is not None:
            probe.mark_stage_end(
                current_stage,
                note=f"train_trials={len(train_trials)} test_trials={len(test_trials)}",
            )

        current_stage = "build_train_windows"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        sfreq = float(train_trials[0]["sfreq"])
        win_len = int(round(float(args.window_sec) * sfreq))
        x_train, y_train, tid_train, train_meta = _materialize_train_windows(
            trials=train_trials,
            win_len=win_len,
            seed_dir=seed_dir,
            memmap_threshold_bytes=int(float(args.memmap_threshold_gb) * (1024**3)),
        )
        if probe is not None:
            probe.mark_stage_end(
                current_stage,
                note=(
                    f"train_windows={int(train_meta['total_train_windows'])} "
                    f"storage={train_meta['train_window_storage']}"
                ),
            )

        del all_trials
        del train_trials
        gc.collect()

        current_stage = "model_fit"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        model = _build_model(
            n_kernels=int(args.n_kernels),
            random_state=int(args.seed),
            n_jobs=int(args.n_jobs),
        )
        t_fit0 = time.perf_counter()
        model.fit(x_train, y_train)
        fit_sec = float(time.perf_counter() - t_fit0)
        if probe is not None:
            probe.mark_stage_end(
                current_stage,
                note=f"fit_sec={fit_sec:.3f} train_windows={len(y_train)}",
            )

        current_stage = "predict_test"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        y_test_parts: List[np.ndarray] = []
        y_pred_parts: List[np.ndarray] = []
        tid_test_parts: List[np.ndarray] = []
        n_short_test = 0
        total_test_windows = 0
        chunk_count = 0
        for x_chunk, y_chunk, tid_chunk, n_short_chunk in _iter_test_window_chunks(
            trials=test_trials,
            win_len=win_len,
            chunk_size=int(args.test_chunk_size),
        ):
            y_pred_chunk = np.asarray(model.predict(x_chunk), dtype=np.int64)
            y_test_parts.append(y_chunk)
            y_pred_parts.append(y_pred_chunk)
            tid_test_parts.append(tid_chunk)
            n_short_test += int(n_short_chunk)
            total_test_windows += int(len(y_chunk))
            chunk_count += 1
            del x_chunk, y_chunk, tid_chunk, y_pred_chunk
            gc.collect()
        if not y_test_parts:
            raise RuntimeError(f"No test windows generated under official protocol for {ds}.")
        y_test = np.concatenate(y_test_parts, axis=0).astype(np.int64, copy=False)
        y_pred_win = np.concatenate(y_pred_parts, axis=0).astype(np.int64, copy=False)
        tid_test = np.concatenate(tid_test_parts, axis=0).astype(object, copy=False)
        if probe is not None:
            probe.mark_stage_end(
                current_stage,
                note=f"test_windows={total_test_windows} chunks={chunk_count}",
            )

        current_stage = "aggregate_metrics"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        window_acc = float(accuracy_score(y_test, y_pred_win))
        window_macro_f1 = float(f1_score(y_test, y_pred_win, average="macro"))
        y_true_trial, y_pred_trial = _aggregate_trial_majority(y_test, y_pred_win, tid_test)
        trial_acc = float(accuracy_score(y_true_trial, y_pred_trial))
        trial_macro_f1 = float(f1_score(y_true_trial, y_pred_trial, average="macro"))
        if probe is not None:
            probe.mark_stage_end(current_stage, note=f"trial_f1={trial_macro_f1:.4f}")

        metrics = {
            "window_acc": window_acc,
            "window_macro_f1": window_macro_f1,
            "trial_acc": trial_acc,
            "trial_macro_f1": trial_macro_f1,
        }
        _write_json(os.path.join(seed_dir, "metrics.json"), metrics)

        run_meta = {
            "pipeline_name": "raw_minirocket_official_seed_family",
            "dataset": ds,
            "seed": int(args.seed),
            "protocol": split_meta["protocol"],
            "split_hash": split_meta["split_hash"],
            "train_trial_count": int(split_meta["train_count_trials"]),
            "test_trial_count": int(split_meta["test_count_trials"]),
            "window_policy_name": "official_raw_nonoverlap_4s",
            "window_sec": float(args.window_sec),
            "hop_sec": float(args.window_sec),
            "window_len_samples": int(win_len),
            "hop_len_samples": int(win_len),
            "n_channels": int(np.asarray(x_train).shape[1]),
            "n_kernels": int(args.n_kernels),
            "n_jobs": int(args.n_jobs),
            "train_only_fit": True,
            "extra_windowing": False,
            "raw_segmented_to_official_windows": True,
            "input_normalization": "none",
            "class_counts_train": train_class_counts,
            "class_counts_test": test_class_counts,
            "total_train_windows": int(train_meta["total_train_windows"]),
            "total_test_windows": int(total_test_windows),
            "window_count_stats_train": _count_stats(train_meta["per_trial_window_counts_train"]),
            "n_short_trials_train": int(train_meta["n_short_trials_train"]),
            "n_short_trials_test": int(n_short_test),
            "train_window_storage": train_meta["train_window_storage"],
            "train_window_mem_bytes": int(train_meta["train_window_mem_bytes"]),
        }
        _write_json(os.path.join(seed_dir, "run_meta.json"), run_meta)

        row = {
            "dataset": ds,
            "seed": int(args.seed),
            "protocol": split_meta["protocol"],
            "train_trial_count": int(split_meta["train_count_trials"]),
            "test_trial_count": int(split_meta["test_count_trials"]),
            "train_window_count": int(train_meta["total_train_windows"]),
            "test_window_count": int(total_test_windows),
            "window_acc": window_acc,
            "window_macro_f1": window_macro_f1,
            "trial_acc": trial_acc,
            "trial_macro_f1": trial_macro_f1,
            "n_channels": int(np.asarray(x_train).shape[1]),
            "window_len_samples": int(win_len),
        }
        pd.DataFrame([row]).to_csv(os.path.join(out_dir, "summary_per_seed.csv"), index=False)
        pd.DataFrame(
            [
                {"metric": "window_acc", "mean": window_acc, "std": 0.0},
                {"metric": "window_macro_f1", "mean": window_macro_f1, "std": 0.0},
                {"metric": "trial_acc", "mean": trial_acc, "std": 0.0},
                {"metric": "trial_macro_f1", "mean": trial_macro_f1, "std": 0.0},
                {"metric": "n_seeds", "mean": 1, "std": 0.0},
            ]
        ).to_csv(os.path.join(out_dir, "summary_agg.csv"), index=False)

        mmap_path = train_meta.get("train_window_memmap_path")
        if isinstance(mmap_path, str) and mmap_path and os.path.isfile(mmap_path) and not bool(args.keep_memmap_files):
            try:
                if isinstance(x_train, np.memmap) and getattr(x_train, "_mmap", None) is not None:
                    x_train._mmap.close()
            except Exception:
                pass
            try:
                os.remove(mmap_path)
            except Exception:
                pass

        print(
            f"[{ds}][official][seed={args.seed}] "
            f"trial_f1={trial_macro_f1:.4f} trial_acc={trial_acc:.4f} "
            f"window_f1={window_macro_f1:.4f} window_acc={window_acc:.4f}"
        )
        if probe is not None:
            probe.mark_success()
        return row
    except Exception as exc:
        if probe is not None:
            probe.mark_failure(current_stage, exc)
        raise
    finally:
        gc.collect()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Official SEED-family MiniROCKET runner.")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["seed1"],
        choices=sorted(OFFICIAL_SEED_DATASETS),
    )
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out-root", type=str, default="out/raw_minirocket_official_seed_family")
    p.add_argument("--processed-root", type=str, default="data/SEED/SEED_EEG/Preprocessed_EEG")
    p.add_argument("--stim-xlsx", type=str, default="data/SEED/SEED_EEG/SEED_stimulation.xlsx")
    p.add_argument("--seediv-root", type=str, default=DEFAULT_SEEDIV_ROOT)
    p.add_argument("--seedv-root", type=str, default=DEFAULT_SEEDV_ROOT)
    p.add_argument("--window-sec", type=float, default=4.0)
    p.add_argument("--n-kernels", type=int, default=10000)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--test-chunk-size", type=int, default=2048)
    p.add_argument("--memmap-threshold-gb", type=float, default=2.0)
    p.add_argument("--resource-probe", action="store_true")
    p.add_argument("--keep-memmap-files", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    for ds in [normalize_dataset_name(d) for d in args.datasets]:
        if ds not in OFFICIAL_SEED_DATASETS:
            raise ValueError(f"Unsupported dataset: {ds}")
        _run_dataset(ds, args)


if __name__ == "__main__":
    main()
