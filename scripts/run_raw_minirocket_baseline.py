#!/usr/bin/env python
"""Raw + MiniROCKET external baseline line (separate from z-space mainline).

Design locks:
- No PIA / Gate / manifold modules
- Trial-level 80/20 repeated holdout split + split_hash
- Train-only fitting, test never augmented
- Trial-level majority aggregation
- Nominal per-trial train window cap K=120
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.trial_dataset_factory import (  # noqa: E402
    DEFAULT_ATRIALFIBRILLATION_ROOT,
    DEFAULT_BASICMOTIONS_ROOT,
    DEFAULT_EPILEPSY_ROOT,
    DEFAULT_FINGERMOVEMENTS_ROOT,
    DEFAULT_HAR_ROOT,
    DEFAULT_HANDMOVEMENTDIRECTION_ROOT,
    DEFAULT_MITBIH_NPZ,
    DEFAULT_NATOPS_ROOT,
    DEFAULT_PENDIGITS_ROOT,
    DEFAULT_SEEDIV_ROOT,
    DEFAULT_SEEDV_ROOT,
    DEFAULT_SELFREGULATIONSCP1_ROOT,
    DEFAULT_UWAVEGESTURELIBRARY_ROOT,
    load_trials_for_dataset,
    normalize_dataset_name,
)
from run_phase15_step0a_paired_lock import (  # noqa: E402
    _aggregate_trials,
    _apply_window_cap,
    _make_trial_split,
)
from scripts.resource_probe_utils import ResourceProbeLogger  # noqa: E402


LONG_SEQUENCE_DATASETS = {"seed1", "seediv", "seedv"}
SHORT_SEQUENCE_DATASETS = {"har", "natops", "mitbih", "fingermovements", "selfregulationscp1"}
SHORT_SEQUENCE_DATASETS |= {
    "basicmotions",
    "handmovementdirection",
    "uwavegesturelibrary",
    "epilepsy",
    "atrialfibrillation",
    "pendigits",
}


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


def _window_starts(n_samples: int, win_len: int, hop_len: int) -> Tuple[np.ndarray, bool]:
    if n_samples <= 0:
        return np.asarray([0], dtype=np.int64), True
    if n_samples < win_len:
        return np.asarray([0], dtype=np.int64), True
    n_windows = 1 + (n_samples - win_len) // hop_len
    starts = np.arange(n_windows, dtype=np.int64) * int(hop_len)
    return starts, False


def _normalize_window_per_channel_zscore(x_win: np.ndarray) -> np.ndarray:
    m = np.mean(x_win, axis=1, keepdims=True)
    s = np.std(x_win, axis=1, keepdims=True)
    return (x_win - m) / (s + 1e-6)


def _extract_window_by_start(x_trial: np.ndarray, start: int, win_len: int, is_short: bool) -> np.ndarray:
    x = np.asarray(x_trial, dtype=np.float32)
    c, t = x.shape
    if is_short:
        out = np.zeros((c, win_len), dtype=np.float32)
        keep = min(int(t), int(win_len))
        if keep > 0:
            out[:, :keep] = x[:, :keep]
        return out
    return x[:, start : start + win_len].astype(np.float32, copy=False)


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


def _to_class_count_dict(trials: Sequence[Dict]) -> Dict[str, int]:
    c = Counter(int(t["label"]) for t in trials)
    return {str(int(k)): int(v) for k, v in sorted(c.items())}


def _build_train_window_index(
    train_trials: Sequence[Dict],
    win_len: int,
    hop_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], int]:
    trial_pos: List[int] = []
    start_pos: List[int] = []
    short_flags: List[bool] = []
    y_all: List[int] = []
    tid_all: List[str] = []
    per_trial_before: Dict[str, int] = {}
    n_short_trials = 0

    for i, tr in enumerate(train_trials):
        x = np.asarray(tr["x_trial"])
        tid = str(tr["trial_id_str"])
        starts, is_short = _window_starts(int(x.shape[1]), int(win_len), int(hop_len))
        if is_short:
            n_short_trials += 1
        per_trial_before[tid] = int(starts.size)
        label = int(tr["label"])
        for s in starts.tolist():
            trial_pos.append(int(i))
            start_pos.append(int(s))
            short_flags.append(bool(is_short))
            y_all.append(label)
            tid_all.append(tid)

    return (
        np.asarray(trial_pos, dtype=np.int32),
        np.asarray(start_pos, dtype=np.int32),
        np.asarray(short_flags, dtype=bool),
        np.asarray(y_all, dtype=np.int64),
        np.asarray(tid_all, dtype=object),
        per_trial_before,
        int(n_short_trials),
    )


def _build_capped_train_windows(
    train_trials: Sequence[Dict],
    win_len: int,
    hop_len: int,
    cap_k: int,
    cap_seed: int,
    cap_policy: str,
    *,
    seed_out_dir: str,
    memmap_threshold_bytes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    (
        trial_pos,
        start_pos,
        short_flags,
        y_all,
        tid_all,
        per_trial_before,
        n_short_trials_padded,
    ) = _build_train_window_index(train_trials=train_trials, win_len=win_len, hop_len=hop_len)

    n_total = int(y_all.shape[0])
    if n_total <= 0:
        raise RuntimeError("No train windows generated.")

    max_before = max(per_trial_before.values()) if per_trial_before else 0
    effective_cap_k = int(min(int(cap_k), int(max_before))) if int(cap_k) > 0 else 0

    # Reuse Phase15 cap helper directly on lightweight dummy payload (window index).
    dummy_x = np.arange(n_total, dtype=np.int64).reshape(-1, 1)
    dummy_selected, y_cap, tid_cap, _is_aug_cap, per_trial_after, _aug_ratio = _apply_window_cap(
        X=dummy_x,
        y=y_all,
        tid=tid_all,
        cap_k=effective_cap_k,
        seed=int(cap_seed),
        is_aug=np.zeros((n_total,), dtype=bool),
        policy=cap_policy,
    )
    selected_idx = np.asarray(dummy_selected[:, 0], dtype=np.int64)

    if selected_idx.size <= 0:
        raise RuntimeError("No train windows selected after cap.")

    c = int(np.asarray(train_trials[0]["x_trial"]).shape[0])
    n_items = int(selected_idx.size)
    n_bytes = int(n_items * c * int(win_len) * np.dtype(np.float32).itemsize)
    mmap_path = None
    if n_bytes >= int(memmap_threshold_bytes):
        _ensure_dir(seed_out_dir)
        mmap_path = os.path.join(seed_out_dir, "train_windows_cap.f32.mmap")
        x_cap = np.memmap(
            mmap_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_items, c, int(win_len)),
        )
        storage = "memmap"
    else:
        x_cap = np.empty((n_items, c, int(win_len)), dtype=np.float32)
        storage = "ram"

    for j, idx in enumerate(selected_idx.tolist()):
        tr = train_trials[int(trial_pos[idx])]
        w = _extract_window_by_start(
            x_trial=np.asarray(tr["x_trial"], dtype=np.float32),
            start=int(start_pos[idx]),
            win_len=int(win_len),
            is_short=bool(short_flags[idx]),
        )
        x_cap[j] = _normalize_window_per_channel_zscore(w).astype(np.float32, copy=False)

    vals_after = np.asarray(list(per_trial_after.values()), dtype=np.float64) if per_trial_after else np.asarray([], dtype=np.float64)
    vals_before = np.asarray(list(per_trial_before.values()), dtype=np.float64) if per_trial_before else np.asarray([], dtype=np.float64)
    frac_hit = (
        float(np.mean(vals_before > float(effective_cap_k)))
        if vals_before.size > 0 and effective_cap_k > 0
        else 0.0
    )

    if isinstance(x_cap, np.memmap):
        x_cap.flush()

    meta = {
        "nominal_cap_K": int(cap_k),
        "effective_cap_K": int(effective_cap_k),
        "total_train_windows_before_cap": int(n_total),
        "total_train_windows": int(n_items),
        "per_trial_windows_mean_after_cap": float(np.mean(vals_after)) if vals_after.size else 0.0,
        "per_trial_windows_std_after_cap": float(np.std(vals_after)) if vals_after.size else 0.0,
        "fraction_trials_hit_cap": float(frac_hit),
        "per_trial_window_counts_before_cap": per_trial_before,
        "per_trial_window_counts_after_cap": per_trial_after,
        "n_short_trials_padded_train": int(n_short_trials_padded),
        "train_window_storage": storage,
        "train_window_memmap_path": mmap_path,
        "train_window_mem_bytes": int(n_bytes),
    }
    return x_cap, np.asarray(y_cap, dtype=np.int64), np.asarray(tid_cap, dtype=object), meta


def _iter_test_windows(
    test_trials: Sequence[Dict],
    win_len: int,
    hop_len: int,
) -> Tuple[List[np.ndarray], List[int], List[str], int]:
    x_parts: List[np.ndarray] = []
    y_parts: List[int] = []
    tid_parts: List[str] = []
    n_short_trials = 0

    for tr in test_trials:
        x = np.asarray(tr["x_trial"], dtype=np.float32)
        tid = str(tr["trial_id_str"])
        label = int(tr["label"])
        starts, is_short = _window_starts(int(x.shape[1]), int(win_len), int(hop_len))
        if is_short:
            n_short_trials += 1
        for s in starts.tolist():
            w = _extract_window_by_start(x_trial=x, start=int(s), win_len=int(win_len), is_short=bool(is_short))
            x_parts.append(_normalize_window_per_channel_zscore(w).astype(np.float32, copy=False))
            y_parts.append(label)
            tid_parts.append(tid)

    return x_parts, y_parts, tid_parts, int(n_short_trials)


def _iter_test_windows_chunked(
    test_trials: Sequence[Dict],
    win_len: int,
    hop_len: int,
    chunk_size: int,
):
    x_parts: List[np.ndarray] = []
    y_parts: List[int] = []
    tid_parts: List[str] = []
    n_short_trials = 0

    for tr in test_trials:
        x = np.asarray(tr["x_trial"], dtype=np.float32)
        tid = str(tr["trial_id_str"])
        label = int(tr["label"])
        starts, is_short = _window_starts(int(x.shape[1]), int(win_len), int(hop_len))
        if is_short:
            n_short_trials += 1
        for s in starts.tolist():
            w = _extract_window_by_start(x_trial=x, start=int(s), win_len=int(win_len), is_short=bool(is_short))
            x_parts.append(_normalize_window_per_channel_zscore(w).astype(np.float32, copy=False))
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


def _select_trials_by_ids(all_trials: Sequence[Dict], keep_ids: Sequence[str]) -> List[Dict]:
    keep = set(str(x) for x in keep_ids)
    return [tr for tr in all_trials if str(tr["trial_id_str"]) in keep]


def _resolve_window_policy(
    dataset: str,
    train_trials: Sequence[Dict],
    fixed_window_sec: float,
    fixed_hop_sec: float,
    prop_win_ratio: float,
    prop_hop_ratio: float,
    min_win_len: int,
    min_hop_len: int,
) -> Dict[str, object]:
    ds = normalize_dataset_name(dataset)
    if not train_trials:
        raise RuntimeError(f"No train trials for dataset={ds}.")

    if ds in LONG_SEQUENCE_DATASETS:
        sfreq = float(train_trials[0]["sfreq"])
        win_len = max(1, int(round(float(fixed_window_sec) * sfreq)))
        hop_len = max(1, int(round(float(fixed_hop_sec) * sfreq)))
        return {
            "window_policy_name": "fixed_seconds_v1",
            "window_sec": float(fixed_window_sec),
            "hop_sec": float(fixed_hop_sec),
            "window_len_samples": int(win_len),
            "hop_len_samples": int(hop_len),
            "adaptive_rule": "none",
            "train_median_trial_len_samples": int(np.median([int(np.asarray(t['x_trial']).shape[1]) for t in train_trials])),
        }

    if ds in SHORT_SEQUENCE_DATASETS:
        t_med = int(np.median([int(np.asarray(t["x_trial"]).shape[1]) for t in train_trials]))
        win_len = int(round(float(prop_win_ratio) * float(t_med)))
        hop_len = int(round(float(prop_hop_ratio) * float(t_med)))
        win_len = max(int(min_win_len), int(win_len))
        hop_len = max(int(min_hop_len), int(hop_len))
        hop_len = min(int(hop_len), int(win_len))
        return {
            "window_policy_name": "proportional_samples_v1",
            "window_sec": None,
            "hop_sec": None,
            "window_len_samples": int(win_len),
            "hop_len_samples": int(hop_len),
            "adaptive_rule": (
                f"win=round({prop_win_ratio}*T_med_train), hop=round({prop_hop_ratio}*T_med_train), "
                f"clamp(win>={int(min_win_len)},hop>={int(min_hop_len)}), hop<=win"
            ),
            "train_median_trial_len_samples": int(t_med),
        }

    raise ValueError(f"Unsupported dataset for window policy: {ds}")


def _build_model(n_kernels: int, random_state: int, n_jobs: int):
    try:
        from aeon.classification.convolution_based import MiniRocketClassifier
    except Exception as e:
        raise ImportError(
            "MiniROCKET baseline requires aeon. Install in pia env: `pip install aeon`."
        ) from e
    return MiniRocketClassifier(
        n_kernels=int(n_kernels),
        random_state=int(random_state),
        n_jobs=int(n_jobs),
    )


def _to_scores(y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    classes = np.asarray(classes).astype(int).ravel()
    cls_to_col = {int(c): i for i, c in enumerate(classes.tolist())}
    out = np.zeros((len(y_pred), len(classes)), dtype=np.float64)
    for i, yp in enumerate(np.asarray(y_pred).astype(int).ravel().tolist()):
        j = cls_to_col.get(int(yp))
        if j is not None:
            out[i, j] = 1.0
    return out


def _load_trials(dataset: str, args: argparse.Namespace) -> List[Dict]:
    ds = normalize_dataset_name(dataset)
    return load_trials_for_dataset(
        dataset=ds,
        processed_root=args.processed_root,
        stim_xlsx=args.stim_xlsx,
        har_root=args.har_root,
        mitbih_npz=args.mitbih_npz,
        seediv_root=args.seediv_root,
        seedv_root=args.seedv_root,
        natops_root=args.natops_root,
        fingermovements_root=args.fingermovements_root,
        selfregulationscp1_root=args.selfregulationscp1_root,
    )


def _run_seed(dataset: str, all_trials: Sequence[Dict], seed: int, args: argparse.Namespace, dataset_out_dir: str) -> Dict[str, object]:
    seed_dir = os.path.join(dataset_out_dir, f"seed{seed}")
    _ensure_dir(seed_dir)
    probe = ResourceProbeLogger(seed_dir) if bool(args.resource_probe) else None
    current_stage = "init"

    try:
        current_stage = "split_trials"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        train_trials, test_trials, split_meta = _make_trial_split(list(all_trials), int(seed))
        if probe is not None:
            probe.mark_stage_end(
                current_stage,
                note=f"train_trials={len(train_trials)} test_trials={len(test_trials)}",
            )

        train_class_counts = _to_class_count_dict(train_trials)
        test_class_counts = _to_class_count_dict(test_trials)

        current_stage = "resolve_window_policy"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        policy = _resolve_window_policy(
            dataset=dataset,
            train_trials=train_trials,
            fixed_window_sec=float(args.window_sec),
            fixed_hop_sec=float(args.hop_sec),
            prop_win_ratio=float(args.prop_win_ratio),
            prop_hop_ratio=float(args.prop_hop_ratio),
            min_win_len=int(args.min_window_len_samples),
            min_hop_len=int(args.min_hop_len_samples),
        )
        win_len = int(policy["window_len_samples"])
        hop_len = int(policy["hop_len_samples"])
        if probe is not None:
            probe.mark_stage_end(
                current_stage,
                note=f"window_len={win_len} hop_len={hop_len}",
            )

        current_stage = "build_train_windows"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        x_train, y_train, _tid_train, cap_meta = _build_capped_train_windows(
            train_trials=train_trials,
            win_len=win_len,
            hop_len=hop_len,
            cap_k=int(args.nominal_cap_k),
            cap_seed=int(seed),
            cap_policy=args.cap_sampling_policy,
            seed_out_dir=seed_dir,
            memmap_threshold_bytes=int(float(args.memmap_threshold_gb) * (1024**3)),
        )
        if probe is not None:
            probe.mark_stage_end(
                current_stage,
                note=(
                    f"train_windows={int(cap_meta['total_train_windows'])} "
                    f"storage={cap_meta['train_window_storage']}"
                ),
            )
        train_window_is_memmap = isinstance(x_train, np.memmap)

        if bool(args.memory_optimize):
            del train_trials
            del all_trials
            del test_trials
            gc.collect()

        current_stage = "model_fit"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        gc.collect()
        model = _build_model(
            n_kernels=int(args.n_kernels),
            random_state=int(seed),
            n_jobs=int(args.n_jobs),
        )
        t_fit0 = time.perf_counter()
        model.fit(x_train, y_train)
        fit_elapsed = float(time.perf_counter() - t_fit0)
        if probe is not None:
            probe.mark_stage_end(
                current_stage,
                note=f"fit_sec={fit_elapsed:.3f} train_windows={len(y_train)}",
            )

        mmap_path = cap_meta.get("train_window_memmap_path")
        if bool(args.memory_optimize):
            try:
                del x_train
            except Exception:
                pass
            gc.collect()

        if bool(args.memory_optimize):
            current_stage = "reload_test_trials"
            if probe is not None:
                probe.mark_stage_start(current_stage)
            reloaded_trials = _load_trials(dataset, args)
            test_trials = _select_trials_by_ids(reloaded_trials, split_meta["test_trial_ids"])
            del reloaded_trials
            gc.collect()
            if probe is not None:
                probe.mark_stage_end(
                    current_stage,
                    note=f"test_trials={len(test_trials)}",
                )

        current_stage = "predict_test"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        n_short_test = 0
        y_test_parts: List[np.ndarray] = []
        tid_test_parts: List[np.ndarray] = []
        y_pred_parts: List[np.ndarray] = []

        if bool(args.stream_test_windows):
            chunk_count = 0
            total_test_windows = 0
            for x_chunk, y_chunk, tid_chunk, n_short_chunk in _iter_test_windows_chunked(
                test_trials=test_trials,
                win_len=win_len,
                hop_len=hop_len,
                chunk_size=int(args.test_chunk_size),
            ):
                y_pred_chunk = np.asarray(model.predict(x_chunk), dtype=np.int64)
                y_pred_parts.append(y_pred_chunk)
                y_test_parts.append(y_chunk)
                tid_test_parts.append(tid_chunk)
                total_test_windows += int(len(y_chunk))
                n_short_test += int(n_short_chunk)
                chunk_count += 1
                del x_chunk, y_chunk, tid_chunk, y_pred_chunk
                gc.collect()
            if not y_test_parts:
                raise RuntimeError(f"No test windows generated for dataset={dataset}, seed={seed}.")
            y_test = np.concatenate(y_test_parts, axis=0).astype(np.int64, copy=False)
            tid_test = np.concatenate(tid_test_parts, axis=0).astype(object, copy=False)
            y_pred_win = np.concatenate(y_pred_parts, axis=0).astype(np.int64, copy=False)
            predict_note = f"test_windows={total_test_windows} chunks={chunk_count}"
        else:
            x_test_list, y_test_list, tid_test_list, n_short_test = _iter_test_windows(
                test_trials=test_trials,
                win_len=win_len,
                hop_len=hop_len,
            )
            if not x_test_list:
                raise RuntimeError(f"No test windows generated for dataset={dataset}, seed={seed}.")
            x_test = np.stack(x_test_list, axis=0).astype(np.float32, copy=False)
            y_test = np.asarray(y_test_list, dtype=np.int64)
            tid_test = np.asarray(tid_test_list, dtype=object)
            y_pred_win = np.asarray(model.predict(x_test), dtype=np.int64)
            predict_note = f"test_windows={len(y_test)} chunks=1"
            del x_test, x_test_list, y_test_list, tid_test_list
            gc.collect()

        if probe is not None:
            probe.mark_stage_end(current_stage, note=predict_note)

        try:
            del test_trials
        except Exception:
            pass
        gc.collect()

        current_stage = "aggregate_metrics"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        window_acc = float(accuracy_score(y_test, y_pred_win))
        window_macro_f1 = float(f1_score(y_test, y_pred_win, average="macro"))

        classes = np.unique(np.concatenate([y_train, y_test]))
        scores_win = _to_scores(y_pred_win, classes=classes)
        y_true_trial, y_pred_trial = _aggregate_trials(
            y_true_win=y_test,
            y_pred_win=y_pred_win,
            scores_win=scores_win,
            tid_win=tid_test,
            mode=args.aggregation_mode,
        )
        trial_acc = float(accuracy_score(y_true_trial, y_pred_trial))
        trial_macro_f1 = float(f1_score(y_true_trial, y_pred_trial, average="macro"))
        if probe is not None:
            probe.mark_stage_end(current_stage, note=f"trial_f1={trial_macro_f1:.4f}")

        metrics = {
            "trial_acc": trial_acc,
            "trial_macro_f1": trial_macro_f1,
            "window_acc": window_acc,
            "window_macro_f1": window_macro_f1,
            "aggregation_mode": args.aggregation_mode,
        }
        _write_json(os.path.join(seed_dir, "metrics.json"), metrics)

        run_meta = {
            "pipeline_name": "raw_minirocket_baseline",
            "dataset": dataset,
            "seed": int(seed),
            "split_hash": split_meta["split_hash"],
            "train_trial_count": int(split_meta["train_count_trials"]),
            "test_trial_count": int(split_meta["test_count_trials"]),
            "train_trial_ids_preview": split_meta["train_trial_ids"][: int(args.split_preview_n)],
            "test_trial_ids_preview": split_meta["test_trial_ids"][: int(args.split_preview_n)],
            "window_policy_name": policy["window_policy_name"],
            "window_sec": policy["window_sec"],
            "hop_sec": policy["hop_sec"],
            "window_len_samples": int(policy["window_len_samples"]),
            "hop_len_samples": int(policy["hop_len_samples"]),
            "adaptive_rule": policy["adaptive_rule"],
            "train_median_trial_len_samples": int(policy["train_median_trial_len_samples"]),
            "window_norm_mode": "per_window_per_channel_zscore",
            "model_type": "aeon_minirocket_wrapper",
            "n_kernels": int(args.n_kernels),
            "random_state": int(seed),
            "head_type": "wrapper_internal",
            "n_jobs": int(args.n_jobs),
            "aggregation_mode": args.aggregation_mode,
            "train_only_fit": True,
            "test_augmented": False,
            "nominal_cap_K": int(cap_meta["nominal_cap_K"]),
            "effective_cap_K": int(cap_meta["effective_cap_K"]),
            "total_train_windows_before_cap": int(cap_meta["total_train_windows_before_cap"]),
            "total_train_windows": int(cap_meta["total_train_windows"]),
            "per_trial_windows_mean_after_cap": float(cap_meta["per_trial_windows_mean_after_cap"]),
            "per_trial_windows_std_after_cap": float(cap_meta["per_trial_windows_std_after_cap"]),
            "fraction_trials_hit_cap": float(cap_meta["fraction_trials_hit_cap"]),
            "n_short_trials_padded_train": int(cap_meta["n_short_trials_padded_train"]),
            "n_short_trials_padded_test": int(n_short_test),
            "n_short_trials_padded": int(cap_meta["n_short_trials_padded_train"] + n_short_test),
            "class_counts_train": train_class_counts,
            "class_counts_test": test_class_counts,
            "window_count_stats_after_cap": _count_stats(cap_meta["per_trial_window_counts_after_cap"]),
            "train_window_storage": cap_meta["train_window_storage"],
            "train_window_mem_bytes": int(cap_meta["train_window_mem_bytes"]),
            "memory_optimize": bool(args.memory_optimize),
            "stream_test_windows": bool(args.stream_test_windows),
            "test_chunk_size": int(args.test_chunk_size),
            "resource_probe": bool(args.resource_probe),
        }
        _write_json(os.path.join(seed_dir, "run_meta.json"), run_meta)

        if (
            isinstance(mmap_path, str)
            and mmap_path
            and os.path.isfile(mmap_path)
            and not bool(args.keep_memmap_files)
        ):
            try:
                if train_window_is_memmap and "x_train" in locals() and getattr(x_train, "_mmap", None) is not None:
                    x_train._mmap.close()
            except Exception:
                pass
            try:
                os.remove(mmap_path)
            except Exception:
                pass

        print(
            f"[{dataset}][seed={seed}] "
            f"trial_f1={trial_macro_f1:.4f} trial_acc={trial_acc:.4f} "
            f"window_f1={window_macro_f1:.4f} window_acc={window_acc:.4f} "
            f"split_hash={split_meta['split_hash'][:12]}..."
        )

        if probe is not None:
            probe.mark_success()

        return {
            "dataset": dataset,
            "seed": int(seed),
            "split_hash": split_meta["split_hash"],
            "train_trial_count": int(split_meta["train_count_trials"]),
            "test_trial_count": int(split_meta["test_count_trials"]),
            "trial_acc": trial_acc,
            "trial_macro_f1": trial_macro_f1,
            "window_acc": window_acc,
            "window_macro_f1": window_macro_f1,
            "effective_cap_K": int(cap_meta["effective_cap_K"]),
            "total_train_windows": int(cap_meta["total_train_windows"]),
            "n_short_trials_padded": int(cap_meta["n_short_trials_padded_train"] + n_short_test),
            "window_policy_name": policy["window_policy_name"],
            "window_len_samples": int(policy["window_len_samples"]),
            "hop_len_samples": int(policy["hop_len_samples"]),
        }
    except Exception as exc:
        if probe is not None:
            probe.mark_failure(current_stage, exc)
        raise


def _write_dataset_summaries(dataset_out_dir: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    per_seed_csv = os.path.join(dataset_out_dir, "summary_per_seed.csv")
    _ensure_dir(dataset_out_dir)
    df.to_csv(per_seed_csv, index=False)

    agg_rows: List[Dict[str, object]] = []
    for metric in [
        "trial_acc",
        "trial_macro_f1",
        "window_acc",
        "window_macro_f1",
        "total_train_windows",
        "n_short_trials_padded",
    ]:
        vals = df[metric].astype(float).to_numpy()
        agg_rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
        )
    agg_rows.append({"metric": "n_seeds", "mean": int(len(df)), "std": 0.0})
    pd.DataFrame(agg_rows).to_csv(os.path.join(dataset_out_dir, "summary_agg.csv"), index=False)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Raw + MiniROCKET baseline runner (external line).")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["seed1"],
        help="Dataset list. Aliases accepted (e.g., seed_iv -> seediv).",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[4])
    p.add_argument("--out-root", type=str, default="out/raw_minirocket_baseline")

    # Dataset roots
    p.add_argument("--processed-root", type=str, default="data/SEED/SEED_EEG/Preprocessed_EEG")
    p.add_argument("--stim-xlsx", type=str, default="data/SEED/SEED_EEG/SEED_stimulation.xlsx")
    p.add_argument("--har-root", type=str, default=DEFAULT_HAR_ROOT)
    p.add_argument("--mitbih-npz", type=str, default=DEFAULT_MITBIH_NPZ)
    p.add_argument("--seediv-root", type=str, default=DEFAULT_SEEDIV_ROOT)
    p.add_argument("--seedv-root", type=str, default=DEFAULT_SEEDV_ROOT)
    p.add_argument("--natops-root", type=str, default=DEFAULT_NATOPS_ROOT)
    p.add_argument("--fingermovements-root", type=str, default=DEFAULT_FINGERMOVEMENTS_ROOT)
    p.add_argument("--selfregulationscp1-root", type=str, default=DEFAULT_SELFREGULATIONSCP1_ROOT)

    # Window policy
    p.add_argument("--window-sec", type=float, default=4.0)
    p.add_argument("--hop-sec", type=float, default=1.0)
    p.add_argument("--prop-win-ratio", type=float, default=0.5)
    p.add_argument("--prop-hop-ratio", type=float, default=0.25)
    p.add_argument("--min-window-len-samples", type=int, default=16)
    p.add_argument("--min-hop-len-samples", type=int, default=8)

    # Cap / eval
    p.add_argument("--nominal-cap-k", type=int, default=120)
    p.add_argument(
        "--cap-sampling-policy",
        type=str,
        default="random",
        choices=["random", "balanced_real_aug", "prefer_real", "prefer_aug"],
    )
    p.add_argument("--aggregation-mode", type=str, default="majority", choices=["majority"])
    p.add_argument("--split-preview-n", type=int, default=5)

    # MiniROCKET
    p.add_argument("--n-kernels", type=int, default=10000)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--memmap-threshold-gb", type=float, default=2.0)
    p.add_argument("--keep-memmap-files", action="store_true")
    p.add_argument("--memory-optimize", action="store_true")
    p.add_argument("--stream-test-windows", action="store_true")
    p.add_argument("--test-chunk-size", type=int, default=1024)
    p.add_argument("--resource-probe", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if int(args.nominal_cap_k) < 0:
        raise ValueError("--nominal-cap-k must be >= 0.")
    if int(args.min_window_len_samples) <= 0:
        raise ValueError("--min-window-len-samples must be > 0.")
    if int(args.min_hop_len_samples) <= 0:
        raise ValueError("--min-hop-len-samples must be > 0.")
    if float(args.prop_win_ratio) <= 0 or float(args.prop_hop_ratio) <= 0:
        raise ValueError("--prop-win-ratio and --prop-hop-ratio must be > 0.")
    if int(args.test_chunk_size) <= 0:
        raise ValueError("--test-chunk-size must be > 0.")

    datasets = [normalize_dataset_name(d) for d in args.datasets]
    # Keep order, remove duplicates.
    seen = set()
    datasets = [d for d in datasets if not (d in seen or seen.add(d))]

    for ds in datasets:
        dataset_out = os.path.join(args.out_root, ds)
        _ensure_dir(dataset_out)
        rows: List[Dict[str, object]] = []
        if bool(args.memory_optimize):
            for seed in args.seeds:
                print(f"[dataset={ds}] loading trials for seed={seed} ...")
                rows.append(_run_seed(ds, _load_trials(ds, args), int(seed), args, dataset_out))
                gc.collect()
        else:
            print(f"[dataset={ds}] loading trials ...")
            all_trials = _load_trials(ds, args)
            for seed in args.seeds:
                rows.append(_run_seed(ds, all_trials, int(seed), args, dataset_out))
        _write_dataset_summaries(dataset_out, rows)
        print(f"[dataset={ds}] done. summary: {os.path.join(dataset_out, 'summary_agg.csv')}")


if __name__ == "__main__":
    main()
