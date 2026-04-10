#!/usr/bin/env python
"""Strict official-split MiniROCKET baseline for fixed-length datasets.

Design locks:
- Respect dataset-provided TRAIN/TEST split exactly
- Use full sequence per sample (no extra windowing, no majority aggregation)
- Report direct test-set accuracy / macro-F1
- Intended for public-alignment baselines on fixed-length datasets
"""

from __future__ import annotations

import argparse
import gc
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
    DEFAULT_ATRIALFIBRILLATION_ROOT,
    DEFAULT_BASICMOTIONS_ROOT,
    DEFAULT_EPILEPSY_ROOT,
    DEFAULT_FINGERMOVEMENTS_ROOT,
    DEFAULT_HAR_ROOT,
    DEFAULT_HANDMOVEMENTDIRECTION_ROOT,
    DEFAULT_NATOPS_ROOT,
    DEFAULT_PENDIGITS_ROOT,
    DEFAULT_SELFREGULATIONSCP1_ROOT,
    DEFAULT_UWAVEGESTURELIBRARY_ROOT,
    load_trials_for_dataset,
    normalize_dataset_name,
)
from scripts.resource_probe_utils import ResourceProbeLogger  # noqa: E402


OFFICIAL_FIXED_SPLIT_DATASETS = {
    "har",
    "natops",
    "fingermovements",
    "selfregulationscp1",
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


def _count_by_label(trials: Sequence[Dict]) -> Dict[str, int]:
    c = Counter(int(t["label"]) for t in trials)
    return {str(int(k)): int(v) for k, v in sorted(c.items())}


def _select_by_split(trials: Sequence[Dict]) -> Tuple[List[Dict], List[Dict]]:
    train = [t for t in trials if str(t.get("split", "")).lower() == "train"]
    test = [t for t in trials if str(t.get("split", "")).lower() == "test"]
    if not train or not test:
        raise RuntimeError("Official fixed-split dataset must contain both split=train and split=test.")
    return train, test


def _stack_trials(trials: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray, int]:
    x_list = [np.asarray(t["x_trial"], dtype=np.float32) for t in trials]
    y = np.asarray([int(t["label"]) for t in trials], dtype=np.int64)
    shapes = {tuple(x.shape) for x in x_list}
    if len(shapes) != 1:
        raise ValueError(f"Official fixed-split runner requires equal-length samples, got shapes={sorted(shapes)}")
    x = np.stack(x_list, axis=0).astype(np.float32, copy=False)
    orig_seq_len = int(x.shape[2])
    if orig_seq_len < 9:
        pad_width = 9 - orig_seq_len
        x = np.pad(x, ((0, 0), (0, 0), (0, pad_width)), mode="constant")
    return x, y, orig_seq_len


def _load_trials(dataset: str, args: argparse.Namespace) -> List[Dict]:
    return load_trials_for_dataset(
        dataset=dataset,
        har_root=args.har_root,
        natops_root=args.natops_root,
        fingermovements_root=args.fingermovements_root,
        selfregulationscp1_root=args.selfregulationscp1_root,
        basicmotions_root=args.basicmotions_root,
        handmovementdirection_root=args.handmovementdirection_root,
        uwavegesturelibrary_root=args.uwavegesturelibrary_root,
        epilepsy_root=args.epilepsy_root,
        atrialfibrillation_root=args.atrialfibrillation_root,
        pendigits_root=args.pendigits_root,
    )


def _run_dataset(dataset: str, args: argparse.Namespace) -> Dict[str, object]:
    out_dir = os.path.join(args.out_root, dataset)
    seed_dir = os.path.join(out_dir, f"seed{int(args.seed)}")
    _ensure_dir(seed_dir)
    probe = ResourceProbeLogger(seed_dir) if bool(args.resource_probe) else None
    current_stage = "init"

    try:
        current_stage = "load_trials"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        all_trials = _load_trials(dataset, args)
        train_trials, test_trials = _select_by_split(all_trials)
        if probe is not None:
            probe.mark_stage_end(
                current_stage,
                note=f"train_trials={len(train_trials)} test_trials={len(test_trials)}",
            )

        current_stage = "stack_train_test"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        x_train, y_train, train_seq_len_orig = _stack_trials(train_trials)
        x_test, y_test, test_seq_len_orig = _stack_trials(test_trials)
        if train_seq_len_orig != test_seq_len_orig:
            raise ValueError(
                f"Train/test original sequence lengths differ: train={train_seq_len_orig} test={test_seq_len_orig}"
            )
        if probe is not None:
            probe.mark_stage_end(
                current_stage,
                note=f"train_shape={tuple(x_train.shape)} test_shape={tuple(x_test.shape)}",
            )

        current_stage = "model_fit"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        model = _build_model(
            n_kernels=int(args.n_kernels),
            random_state=int(args.seed),
            n_jobs=int(args.n_jobs),
        )
        t0 = time.perf_counter()
        model.fit(x_train, y_train)
        fit_sec = float(time.perf_counter() - t0)
        if probe is not None:
            probe.mark_stage_end(current_stage, note=f"fit_sec={fit_sec:.3f} train_cases={len(y_train)}")

        current_stage = "predict_test"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        y_pred = np.asarray(model.predict(x_test), dtype=np.int64)
        if probe is not None:
            probe.mark_stage_end(current_stage, note=f"test_cases={len(y_test)}")

        current_stage = "compute_metrics"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        test_acc = float(accuracy_score(y_test, y_pred))
        test_macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
        if probe is not None:
            probe.mark_stage_end(current_stage, note=f"test_acc={test_acc:.4f} test_f1={test_macro_f1:.4f}")

        metrics = {
            "test_acc": test_acc,
            "test_macro_f1": test_macro_f1,
        }
        _write_json(os.path.join(seed_dir, "metrics.json"), metrics)

        run_meta = {
            "pipeline_name": "raw_minirocket_official_fixedsplit",
            "dataset": dataset,
            "seed": int(args.seed),
            "protocol": "dataset_provided_train_test_full_sequence",
            "n_kernels": int(args.n_kernels),
            "n_jobs": int(args.n_jobs),
            "train_trial_count": int(len(train_trials)),
            "test_trial_count": int(len(test_trials)),
            "train_shape": tuple(int(v) for v in x_train.shape),
            "test_shape": tuple(int(v) for v in x_test.shape),
            "seq_len_original": int(train_seq_len_orig),
            "seq_len_model_input": int(x_train.shape[2]),
            "class_counts_train": _count_by_label(train_trials),
            "class_counts_test": _count_by_label(test_trials),
            "full_sequence_used": True,
            "extra_windowing": False,
            "extra_majority_aggregation": False,
            "extra_resplit": False,
            "zero_padded_to_min_9": bool(int(train_seq_len_orig) < 9),
        }
        _write_json(os.path.join(seed_dir, "run_meta.json"), run_meta)

        row = {
            "dataset": dataset,
            "seed": int(args.seed),
            "test_acc": test_acc,
            "test_macro_f1": test_macro_f1,
            "train_trial_count": int(len(train_trials)),
            "test_trial_count": int(len(test_trials)),
            "n_channels": int(x_train.shape[1]),
            "seq_len": int(train_seq_len_orig),
            "seq_len_model_input": int(x_train.shape[2]),
            "protocol": "dataset_provided_train_test_full_sequence",
        }
        pd.DataFrame([row]).to_csv(os.path.join(out_dir, "summary_per_seed.csv"), index=False)
        pd.DataFrame(
            [
                {"metric": "test_acc", "mean": test_acc, "std": 0.0},
                {"metric": "test_macro_f1", "mean": test_macro_f1, "std": 0.0},
                {"metric": "n_seeds", "mean": 1, "std": 0.0},
            ]
        ).to_csv(os.path.join(out_dir, "summary_agg.csv"), index=False)

        print(
            f"[{dataset}][official][seed={args.seed}] "
            f"test_f1={test_macro_f1:.4f} test_acc={test_acc:.4f}"
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
    p = argparse.ArgumentParser(description="Strict official-split MiniROCKET baseline for fixed-length datasets.")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["natops"],
        choices=sorted(OFFICIAL_FIXED_SPLIT_DATASETS),
    )
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out-root", type=str, default="out/raw_minirocket_official_fixedsplit")
    p.add_argument("--har-root", type=str, default=DEFAULT_HAR_ROOT)
    p.add_argument("--natops-root", type=str, default=DEFAULT_NATOPS_ROOT)
    p.add_argument("--fingermovements-root", type=str, default=DEFAULT_FINGERMOVEMENTS_ROOT)
    p.add_argument("--selfregulationscp1-root", type=str, default=DEFAULT_SELFREGULATIONSCP1_ROOT)
    p.add_argument("--basicmotions-root", type=str, default=DEFAULT_BASICMOTIONS_ROOT)
    p.add_argument("--handmovementdirection-root", type=str, default=DEFAULT_HANDMOVEMENTDIRECTION_ROOT)
    p.add_argument("--uwavegesturelibrary-root", type=str, default=DEFAULT_UWAVEGESTURELIBRARY_ROOT)
    p.add_argument("--epilepsy-root", type=str, default=DEFAULT_EPILEPSY_ROOT)
    p.add_argument("--atrialfibrillation-root", type=str, default=DEFAULT_ATRIALFIBRILLATION_ROOT)
    p.add_argument("--pendigits-root", type=str, default=DEFAULT_PENDIGITS_ROOT)
    p.add_argument("--n-kernels", type=int, default=10000)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--resource-probe", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    datasets = [normalize_dataset_name(d) for d in args.datasets]
    status_rows: List[Dict[str, object]] = []
    for ds in datasets:
        if ds not in OFFICIAL_FIXED_SPLIT_DATASETS:
            raise ValueError(f"Dataset {ds} is not supported by the strict fixed-split runner.")
        row = _run_dataset(ds, args)
        status_rows.append(
            {
                "dataset": ds,
                "status": "success",
                "seed": int(args.seed),
                "protocol": row["protocol"],
                "train_trial_count": int(row["train_trial_count"]),
                "test_trial_count": int(row["test_trial_count"]),
                "n_channels": int(row["n_channels"]),
                "seq_len": int(row["seq_len"]),
                "window_len_samples": np.nan,
                "test_acc": float(row["test_acc"]),
                "test_macro_f1": float(row["test_macro_f1"]),
                "window_acc": np.nan,
                "window_macro_f1": np.nan,
                "trial_acc": np.nan,
                "trial_macro_f1": np.nan,
                "peak_rss_gb": np.nan,
                "note": (
                    "strict official fixed-split MiniROCKET runner"
                    if int(row.get("seq_len_model_input", row["seq_len"])) == int(row["seq_len"])
                    else f"strict official fixed-split MiniROCKET runner; zero padded to {int(row['seq_len_model_input'])}"
                ),
            }
        )
    if status_rows:
        pd.DataFrame(status_rows).to_csv(os.path.join(args.out_root, "official_protocol_status.csv"), index=False)


if __name__ == "__main__":
    main()
