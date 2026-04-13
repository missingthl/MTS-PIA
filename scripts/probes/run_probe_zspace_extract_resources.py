#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from scripts.support.resource_probe_utils import ResourceProbeLogger, _write_json  # noqa: E402
from datasets.trial_dataset_factory import (  # noqa: E402
    DEFAULT_BANDS_EEG,
    load_trials_for_dataset,
    normalize_dataset_name,
    resolve_band_spec,
)
from manifold_raw.features import parse_band_spec  # noqa: E402
from scripts.legacy_phase.run_phase14r_step6b1_rev2 import apply_logcenter, covs_to_features, extract_features_block  # noqa: E402
from scripts.legacy_phase.run_phase15_step0a_paired_lock import _make_trial_split  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Resource probe for z-space feature extraction.")
    p.add_argument("--dataset", type=str, default="seed1", choices=["seed1"])
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out-root", type=str, default="out/resource_probes/zspace_extract")
    p.add_argument("--processed-root", type=str, default="data/SEED/SEED_EEG/Preprocessed_EEG")
    p.add_argument("--stim-xlsx", type=str, default="data/SEED/SEED_EEG/SEED_stimulation.xlsx")
    p.add_argument("--window-sec", type=float, default=4.0)
    p.add_argument("--hop-sec", type=float, default=1.0)
    p.add_argument("--cov-est", type=str, default="sample", choices=["sample", "oas", "ledoitwolf"])
    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--bands", type=str, default=DEFAULT_BANDS_EEG)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = os.path.join(args.out_root, str(args.dataset), f"seed{int(args.seed)}")
    logger = ResourceProbeLogger(out_dir=out_dir)
    current_stage = "init"
    try:
        current_stage = "load_trials"
        logger.mark_stage_start(current_stage)
        dataset = normalize_dataset_name(args.dataset)
        all_trials = load_trials_for_dataset(
            dataset=dataset,
            processed_root=args.processed_root,
            stim_xlsx=args.stim_xlsx,
        )
        bands_spec = resolve_band_spec(dataset, args.bands)
        bands = parse_band_spec(bands_spec)
        logger.mark_stage_end(current_stage, note=f"n_trials={len(all_trials)} bands={bands_spec}")

        current_stage = "split_trials"
        logger.mark_stage_start(current_stage)
        train_trials, test_trials, split_meta = _make_trial_split(list(all_trials), int(args.seed))
        logger.mark_stage_end(
            current_stage,
            note=f"train_trials={len(train_trials)} test_trials={len(test_trials)}",
        )

        current_stage = "extract_features_train"
        logger.mark_stage_start(current_stage)
        covs_train, y_train, tid_train = extract_features_block(
            train_trials,
            args.window_sec,
            args.hop_sec,
            args.cov_est,
            args.spd_eps,
            bands,
        )
        logger.mark_stage_end(current_stage, note=f"train_windows={len(y_train)}")

        current_stage = "extract_features_test"
        logger.mark_stage_start(current_stage)
        covs_test, y_test, tid_test = extract_features_block(
            test_trials,
            args.window_sec,
            args.hop_sec,
            args.cov_est,
            args.spd_eps,
            bands,
        )
        logger.mark_stage_end(current_stage, note=f"test_windows={len(y_test)}")

        current_stage = "apply_logcenter"
        logger.mark_stage_start(current_stage)
        covs_train_lc, covs_test_lc = apply_logcenter(covs_train, covs_test, args.spd_eps)
        logger.mark_stage_end(current_stage, note="done")

        current_stage = "covs_to_features"
        logger.mark_stage_start(current_stage)
        X_train = covs_to_features(covs_train_lc)
        X_test = covs_to_features(covs_test_lc)
        logger.mark_stage_end(
            current_stage,
            note=f"train_shape={tuple(X_train.shape)} test_shape={tuple(X_test.shape)}",
        )

        _write_json(
            os.path.join(out_dir, "probe_meta.json"),
            {
                "dataset": dataset,
                "seed": int(args.seed),
                "split_hash": split_meta["split_hash"],
                "train_windows": int(len(y_train)),
                "test_windows": int(len(y_test)),
                "feature_dim": int(X_train.shape[1]) if X_train.ndim == 2 else None,
                "bands": bands_spec,
            },
        )
        logger.mark_success()
    except Exception as exc:
        logger.mark_failure(current_stage, exc)
        raise


if __name__ == "__main__":
    main()
