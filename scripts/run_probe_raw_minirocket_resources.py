#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import os
import sys
from typing import Dict, List

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from resource_probe_utils import ResourceProbeLogger, _write_json  # noqa: E402
from run_raw_minirocket_baseline import (  # noqa: E402
    DEFAULT_SEEDIV_ROOT,
    _aggregate_trials,
    _build_capped_train_windows,
    _build_model,
    _iter_test_windows,
    _load_trials,
    _resolve_window_policy,
    _to_scores,
)
from run_phase15_step0a_paired_lock import _make_trial_split  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Resource probe for raw MiniROCKET.")
    p.add_argument("--dataset", type=str, required=True, choices=["seed1", "seediv"])
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out-root", type=str, default="out/resource_probes/raw_minirocket")
    p.add_argument("--processed-root", type=str, default="data/SEED/SEED_EEG/Preprocessed_EEG")
    p.add_argument("--stim-xlsx", type=str, default="data/SEED/SEED_EEG/SEED_stimulation.xlsx")
    p.add_argument("--har-root", type=str, default="data/UCI HAR Dataset")
    p.add_argument("--mitbih-npz", type=str, default="data/MITBIH/mitbih_beats.npz")
    p.add_argument("--seediv-root", type=str, default=DEFAULT_SEEDIV_ROOT)
    p.add_argument("--natops-root", type=str, default="data/NATOPS")
    p.add_argument("--fingermovements-root", type=str, default="data/FingerMovements")
    p.add_argument("--window-sec", type=float, default=4.0)
    p.add_argument("--hop-sec", type=float, default=1.0)
    p.add_argument("--prop-win-ratio", type=float, default=0.5)
    p.add_argument("--prop-hop-ratio", type=float, default=0.25)
    p.add_argument("--min-window-len-samples", type=int, default=16)
    p.add_argument("--min-hop-len-samples", type=int, default=8)
    p.add_argument("--nominal-cap-k", type=int, default=120)
    p.add_argument("--cap-sampling-policy", type=str, default="random")
    p.add_argument("--aggregation-mode", type=str, default="majority")
    p.add_argument("--n-kernels", type=int, default=10000)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--memmap-threshold-gb", type=float, default=2.0)
    p.add_argument("--keep-memmap-files", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = os.path.join(args.out_root, str(args.dataset), f"seed{int(args.seed)}")
    logger = ResourceProbeLogger(out_dir=out_dir)
    current_stage = "init"
    try:
        current_stage = "load_trials"
        logger.mark_stage_start(current_stage)
        all_trials = _load_trials(args.dataset, args)
        logger.mark_stage_end(current_stage, note=f"n_trials={len(all_trials)}")

        current_stage = "split_trials"
        logger.mark_stage_start(current_stage)
        train_trials, test_trials, split_meta = _make_trial_split(list(all_trials), int(args.seed))
        logger.mark_stage_end(
            current_stage,
            note=f"train_trials={len(train_trials)} test_trials={len(test_trials)}",
        )

        current_stage = "resolve_window_policy"
        logger.mark_stage_start(current_stage)
        policy = _resolve_window_policy(
            dataset=args.dataset,
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
        logger.mark_stage_end(current_stage, note=f"win_len={win_len} hop_len={hop_len}")

        current_stage = "build_train_windows"
        logger.mark_stage_start(current_stage)
        x_train, y_train, tid_train, cap_meta = _build_capped_train_windows(
            train_trials=train_trials,
            win_len=win_len,
            hop_len=hop_len,
            cap_k=int(args.nominal_cap_k),
            cap_seed=int(args.seed),
            cap_policy=args.cap_sampling_policy,
            seed_out_dir=out_dir,
            memmap_threshold_bytes=int(float(args.memmap_threshold_gb) * (1024**3)),
        )
        logger.mark_stage_end(
            current_stage,
            note=(
                f"train_windows={len(y_train)} storage={cap_meta['train_window_storage']} "
                f"bytes={int(cap_meta['train_window_mem_bytes'])}"
            ),
        )

        current_stage = "build_test_windows"
        logger.mark_stage_start(current_stage)
        x_test_list, y_test_list, tid_test_list, n_short_test = _iter_test_windows(
            test_trials=test_trials,
            win_len=win_len,
            hop_len=hop_len,
        )
        x_test = np.stack(x_test_list, axis=0).astype(np.float32, copy=False)
        y_test = np.asarray(y_test_list, dtype=np.int64)
        tid_test = np.asarray(tid_test_list, dtype=object)
        logger.mark_stage_end(
            current_stage,
            note=f"test_windows={len(y_test)} n_short_test={n_short_test}",
        )

        gc.collect()

        current_stage = "model_fit"
        logger.mark_stage_start(current_stage)
        model = _build_model(
            n_kernels=int(args.n_kernels),
            random_state=int(args.seed),
            n_jobs=int(args.n_jobs),
        )
        model.fit(x_train, y_train)
        logger.mark_stage_end(current_stage, note=f"n_kernels={int(args.n_kernels)}")

        current_stage = "model_predict"
        logger.mark_stage_start(current_stage)
        y_pred_win = np.asarray(model.predict(x_test), dtype=np.int64)
        logger.mark_stage_end(current_stage, note=f"pred_windows={len(y_pred_win)}")

        current_stage = "aggregate"
        logger.mark_stage_start(current_stage)
        classes = np.unique(np.concatenate([y_train, y_test]))
        scores_win = _to_scores(y_pred_win, classes=classes)
        y_true_trial, y_pred_trial = _aggregate_trials(
            y_true_win=y_test,
            y_pred_win=y_pred_win,
            scores_win=scores_win,
            tid_win=tid_test,
            mode=args.aggregation_mode,
        )
        logger.mark_stage_end(current_stage, note=f"trial_count={len(y_true_trial)}")

        _write_json(
            os.path.join(out_dir, "probe_meta.json"),
            {
                "dataset": args.dataset,
                "seed": int(args.seed),
                "split_hash": split_meta["split_hash"],
                "policy": policy,
                "cap_meta": cap_meta,
                "n_short_trials_padded_test": int(n_short_test),
            },
        )
        logger.mark_success()
    except Exception as exc:
        logger.mark_failure(current_stage, exc)
        raise


if __name__ == "__main__":
    main()
