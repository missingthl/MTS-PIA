#!/usr/bin/env python
"""Protocol-locked Phase15 mainline freeze runner.

Mainline conditions:
- baseline: z-space + LinearSVC
- step1b: z-space + Step1B multi-direction PIA (no gate)
- step1b_gate: z-space + Step1B + Gate1 + Gate2

Protocol lock:
- Fixed split datasets keep dataset-provided TRAIN/TEST:
  har, natops, fingermovements, selfregulationscp1,
  basicmotions, handmovementdirection, uwavegesturelibrary,
  epilepsy, atrialfibrillation, pendigits
- SEED family uses native trial protocol:
  seed1   -> first 9 / last 6 trials per session
  seediv  -> first 16 / last 8 trials per session
  seedv   -> first 9 / last 6 trials per session
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.trial_dataset_factory import (  # noqa: E402
    DEFAULT_ATRIALFIBRILLATION_ROOT,
    DEFAULT_BASICMOTIONS_ROOT,
    DEFAULT_BANDS_EEG,
    DEFAULT_EPILEPSY_ROOT,
    DEFAULT_FINGERMOVEMENTS_ROOT,
    DEFAULT_HAR_ROOT,
    DEFAULT_HANDMOVEMENTDIRECTION_ROOT,
    DEFAULT_NATOPS_ROOT,
    DEFAULT_PENDIGITS_ROOT,
    DEFAULT_SEEDIV_ROOT,
    DEFAULT_SEEDV_ROOT,
    DEFAULT_SELFREGULATIONSCP1_ROOT,
    DEFAULT_UWAVEGESTURELIBRARY_ROOT,
    load_trials_for_dataset,
    normalize_dataset_name,
    resolve_band_spec,
)
from manifold_raw.features import parse_band_spec  # noqa: E402
from run_phase14r_step6b1_rev2 import (  # noqa: E402
    apply_logcenter,
    covs_to_features,
    ensure_dir,
    extract_features_block,
    write_json,
)
from scripts.run_phase15_step1a_maxplane import (  # noqa: E402
    _apply_gates,
    _fit_eval_linearsvc,
    _fit_gate1_from_train,
)
from scripts.run_phase15_step1b_multidir_matrix import (  # noqa: E402
    _build_direction_bank_d1,
    _build_multidir_aug_candidates,
    _compute_mech_metrics,
    _write_condition,
)


FIXED_SPLIT_DATASETS = {
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

SEED_NATIVE_DATASETS = {
    "seed1",
    "seediv",
    "seedv",
}


def _parse_csv_list(text: str) -> List[str]:
    out: List[str] = []
    for tok in str(text).split(","):
        t = tok.strip()
        if t:
            out.append(t)
    if not out:
        raise ValueError("list cannot be empty")
    return out


def _parse_seed_list(text: str) -> List[int]:
    out: List[int] = []
    for tok in _parse_csv_list(text):
        out.append(int(tok))
    out = sorted(set(out))
    if not out:
        raise ValueError("seed list cannot be empty")
    return out


def _split_hash(train_ids: Sequence[str], test_ids: Sequence[str]) -> str:
    payload = json.dumps(
        {"train": [str(x) for x in train_ids], "test": [str(x) for x in test_ids]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _seed_native_split_rule(dataset: str, trial_value: int) -> str:
    ds = normalize_dataset_name(dataset)
    if ds == "seed1":
        return "train" if 1 <= int(trial_value) <= 9 else "test"
    if ds == "seediv":
        return "train" if 0 <= int(trial_value) <= 15 else "test"
    if ds == "seedv":
        return "train" if 0 <= int(trial_value) <= 8 else "test"
    raise ValueError(f"Unsupported SEED native protocol dataset: {dataset}")


def _protocol_type_and_note(dataset: str) -> Tuple[str, str]:
    ds = normalize_dataset_name(dataset)
    if ds in FIXED_SPLIT_DATASETS:
        return (
            "fixed_split",
            "dataset-provided TRAIN/TEST split; z-space mainline keeps standard windowed SPD/tangent pipeline",
        )
    if ds == "seed1":
        return (
            "seed_family_native",
            "SEED native protocol: per session first 9 trials train, last 6 trials test",
        )
    if ds == "seediv":
        return (
            "seed_family_native",
            "SEED_IV native protocol: per session first 16 trials train, last 8 trials test",
        )
    if ds == "seedv":
        return (
            "seed_family_native",
            "SEED_V native protocol: per session first 9 trials train, last 6 trials test",
        )
    raise ValueError(f"Unsupported dataset for protocol note: {dataset}")


def _make_protocol_split(dataset: str, all_trials: List[Dict]) -> Tuple[List[Dict], List[Dict], Dict[str, object]]:
    ds = normalize_dataset_name(dataset)
    train_trials: List[Dict] = []
    test_trials: List[Dict] = []

    for trial in all_trials:
        if ds in FIXED_SPLIT_DATASETS:
            split = str(trial.get("split", "")).strip().lower()
            if split == "train":
                train_trials.append(trial)
            elif split == "test":
                test_trials.append(trial)
            else:
                raise ValueError(f"{ds} trial missing valid split field: {trial.get('trial_id_str')}")
            continue

        if ds in SEED_NATIVE_DATASETS:
            side = _seed_native_split_rule(ds, int(trial["trial"]))
            if side == "train":
                train_trials.append(trial)
            else:
                test_trials.append(trial)
            continue

        raise ValueError(f"Unsupported dataset in protocol split: {dataset}")

    train_ids = [str(t["trial_id_str"]) for t in train_trials]
    test_ids = [str(t["trial_id_str"]) for t in test_trials]
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        preview = sorted(list(overlap))[:5]
        raise RuntimeError(f"Protocol split leakage for {ds}: {preview}")
    if not train_trials or not test_trials:
        raise RuntimeError(
            f"Protocol split produced empty side for {ds}: "
            f"train={len(train_trials)} test={len(test_trials)}"
        )

    protocol_type, protocol_note = _protocol_type_and_note(ds)
    return train_trials, test_trials, {
        "protocol_type": protocol_type,
        "protocol_note": protocol_note,
        "split_hash": _split_hash(train_ids, test_ids),
        "train_count_trials": len(train_ids),
        "test_count_trials": len(test_ids),
        "train_trial_ids": train_ids,
        "test_trial_ids": test_ids,
    }


def _summary_stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _summarize_dir_profile(dir_profile: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(dir_profile, dict) or not dir_profile:
        return {
            "worst_dir_id": None,
            "worst_dir_accept_rate": None,
            "worst_dir_flip_rate": None,
            "worst_dir_margin_drop": None,
            "dir_profile_summary": "n/a",
        }

    items: List[Tuple[int, Dict[str, object]]] = []
    for k, v in dir_profile.items():
        if isinstance(v, dict):
            items.append((int(k), v))
    if not items:
        return {
            "worst_dir_id": None,
            "worst_dir_accept_rate": None,
            "worst_dir_flip_rate": None,
            "worst_dir_margin_drop": None,
            "dir_profile_summary": "n/a",
        }

    worst_id, worst_row = max(
        items,
        key=lambda kv: (
            float(kv[1].get("flip_rate", 0.0)),
            -float(kv[1].get("accept_rate", 0.0)),
            float(kv[1].get("margin_drop_median", 0.0)),
        ),
    )
    return {
        "worst_dir_id": int(worst_id),
        "worst_dir_accept_rate": float(worst_row.get("accept_rate", 0.0)),
        "worst_dir_flip_rate": float(worst_row.get("flip_rate", 0.0)),
        "worst_dir_margin_drop": float(worst_row.get("margin_drop_median", 0.0)),
        "dir_profile_summary": (
            f"worst_dir={int(worst_id)};"
            f"accept={float(worst_row.get('accept_rate', 0.0)):.3f};"
            f"flip={float(worst_row.get('flip_rate', 0.0)):.3f};"
            f"margin={float(worst_row.get('margin_drop_median', 0.0)):.4f}"
        ),
    }


def _write_seed_status(
    status_path: str,
    *,
    dataset: str,
    seed: int,
    stage: str,
    seed_started_at: float,
    extra: Dict[str, object] | None = None,
) -> None:
    payload: Dict[str, object] = {
        "dataset": dataset,
        "seed": int(seed),
        "stage": str(stage),
        "seed_elapsed_sec": float(time.time() - seed_started_at),
        "updated_at_epoch_sec": float(time.time()),
    }
    if extra:
        payload.update(extra)
    write_json(status_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="seed1,seediv,seedv,har,natops,fingermovements,selfregulationscp1,basicmotions,handmovementdirection,uwavegesturelibrary,epilepsy,atrialfibrillation,pendigits",
    )
    parser.add_argument("--seeds", type=str, default="3")
    parser.add_argument("--out-root", type=str, default="out/phase15_mainline_freeze_20260319")
    parser.add_argument("--processed-root", type=str, default="data/SEED/SEED_EEG/Preprocessed_EEG")
    parser.add_argument("--stim-xlsx", type=str, default="data/SEED/SEED_EEG/SEED_stimulation.xlsx")
    parser.add_argument("--har-root", type=str, default=DEFAULT_HAR_ROOT)
    parser.add_argument("--seediv-root", type=str, default=DEFAULT_SEEDIV_ROOT)
    parser.add_argument("--seedv-root", type=str, default=DEFAULT_SEEDV_ROOT)
    parser.add_argument("--natops-root", type=str, default=DEFAULT_NATOPS_ROOT)
    parser.add_argument("--fingermovements-root", type=str, default=DEFAULT_FINGERMOVEMENTS_ROOT)
    parser.add_argument("--selfregulationscp1-root", type=str, default=DEFAULT_SELFREGULATIONSCP1_ROOT)
    parser.add_argument("--basicmotions-root", type=str, default=DEFAULT_BASICMOTIONS_ROOT)
    parser.add_argument("--handmovementdirection-root", type=str, default=DEFAULT_HANDMOVEMENTDIRECTION_ROOT)
    parser.add_argument("--uwavegesturelibrary-root", type=str, default=DEFAULT_UWAVEGESTURELIBRARY_ROOT)
    parser.add_argument("--epilepsy-root", type=str, default=DEFAULT_EPILEPSY_ROOT)
    parser.add_argument("--atrialfibrillation-root", type=str, default=DEFAULT_ATRIALFIBRILLATION_ROOT)
    parser.add_argument("--pendigits-root", type=str, default=DEFAULT_PENDIGITS_ROOT)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--hop-sec", type=float, default=1.0)
    parser.add_argument("--cov-est", type=str, default="sample", choices=["sample", "oas", "ledoitwolf"])
    parser.add_argument("--spd-eps", type=float, default=1e-4)
    parser.add_argument("--bands", type=str, default=DEFAULT_BANDS_EEG)
    parser.add_argument("--window-cap-k", type=int, default=120)
    parser.add_argument("--cap-sampling-policy", type=str, default="balanced_real_aug")
    parser.add_argument("--aggregation-mode", type=str, default="majority")
    parser.add_argument("--linear-c", type=float, default=1.0)
    parser.add_argument("--linear-class-weight", type=str, default="none")
    parser.add_argument("--linear-max-iter", type=int, default=1000)
    parser.add_argument("--k-dir", type=int, default=5)
    parser.add_argument("--subset-size", type=int, default=1)
    parser.add_argument("--pia-multiplier", type=int, default=1)
    parser.add_argument("--pia-gamma", type=float, default=0.10)
    parser.add_argument("--pia-n-iters", type=int, default=2)
    parser.add_argument("--pia-activation", type=str, default="sine")
    parser.add_argument("--pia-bias-update-mode", type=str, default="residual")
    parser.add_argument("--pia-c-repr", type=float, default=1.0)
    parser.add_argument("--gate1-q", type=float, default=95.0)
    parser.add_argument("--gate2-q-src", type=float, default=90.0)
    parser.add_argument("--mech-knn-k", type=int, default=20)
    parser.add_argument("--mech-max-aug-for-metrics", type=int, default=2000)
    parser.add_argument("--mech-max-real-knn-ref", type=int, default=10000)
    parser.add_argument("--mech-max-real-knn-query", type=int, default=1000)
    parser.add_argument("--split-preview-n", type=int, default=5)
    parser.add_argument("--extract-progress-every", type=int, default=25)
    args = parser.parse_args()

    datasets = [normalize_dataset_name(x) for x in _parse_csv_list(args.datasets)]
    seeds = _parse_seed_list(args.seeds)
    ensure_dir(args.out_root)

    all_perf_rows: List[Dict[str, object]] = []
    all_mech_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        ds_t0 = time.time()
        print(f"[freeze][{dataset}] start", flush=True)
        all_trials = load_trials_for_dataset(
            dataset=dataset,
            processed_root=args.processed_root,
            stim_xlsx=args.stim_xlsx,
            har_root=args.har_root,
            seediv_root=args.seediv_root,
            seedv_root=args.seedv_root,
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
        print(f"[freeze][{dataset}] loaded all_trials={len(all_trials)} in {time.time() - ds_t0:.1f}s", flush=True)
        split_t0 = time.time()
        train_trials, test_trials, split_meta = _make_protocol_split(dataset, all_trials)
        print(
            f"[freeze][{dataset}] split train_trials={len(train_trials)} "
            f"test_trials={len(test_trials)} in {time.time() - split_t0:.1f}s",
            flush=True,
        )
        bands_spec = resolve_band_spec(dataset, args.bands)
        if bands_spec != args.bands:
            print(f"[freeze][{dataset}] bands override -> {bands_spec}", flush=True)
        bands = parse_band_spec(bands_spec)
        dataset_dir = os.path.join(args.out_root, dataset)
        ensure_dir(dataset_dir)

        dataset_rows: List[Dict[str, object]] = []
        dataset_mech_rows: List[Dict[str, object]] = []

        for seed in seeds:
            seed_dir = os.path.join(dataset_dir, f"seed{seed}")
            ensure_dir(seed_dir)
            status_path = os.path.join(seed_dir, "freeze_status.json")
            seed_started_at = time.time()
            print(
                f"[freeze][{dataset}][seed={seed}] protocol={split_meta['protocol_type']} "
                f"train_trials={len(train_trials)} test_trials={len(test_trials)}",
                flush=True,
            )
            _write_seed_status(
                status_path,
                dataset=dataset,
                seed=int(seed),
                stage="extract_train",
                seed_started_at=seed_started_at,
                extra={
                    "protocol_type": split_meta["protocol_type"],
                    "train_count_trials": int(len(train_trials)),
                    "test_count_trials": int(len(test_trials)),
                },
            )

            stage_t0 = time.time()
            covs_train, y_train, tid_train = extract_features_block(
                train_trials,
                args.window_sec,
                args.hop_sec,
                args.cov_est,
                args.spd_eps,
                bands,
                progress_prefix=f"[freeze][{dataset}][seed={seed}][extract_train]",
                progress_every=max(0, int(args.extract_progress_every)),
            )
            print(
                f"[freeze][{dataset}][seed={seed}] extract_train done "
                f"windows={len(y_train)} in {time.time() - stage_t0:.1f}s",
                flush=True,
            )
            _write_seed_status(
                status_path,
                dataset=dataset,
                seed=int(seed),
                stage="extract_test",
                seed_started_at=seed_started_at,
                extra={"train_windows": int(len(y_train))},
            )
            stage_t0 = time.time()
            covs_test, y_test, tid_test = extract_features_block(
                test_trials,
                args.window_sec,
                args.hop_sec,
                args.cov_est,
                args.spd_eps,
                bands,
                progress_prefix=f"[freeze][{dataset}][seed={seed}][extract_test]",
                progress_every=max(0, int(args.extract_progress_every)),
            )
            print(
                f"[freeze][{dataset}][seed={seed}] extract_test done "
                f"windows={len(y_test)} in {time.time() - stage_t0:.1f}s",
                flush=True,
            )
            _write_seed_status(
                status_path,
                dataset=dataset,
                seed=int(seed),
                stage="logcenter_vectorize",
                seed_started_at=seed_started_at,
                extra={
                    "train_windows": int(len(y_train)),
                    "test_windows": int(len(y_test)),
                },
            )
            stage_t0 = time.time()
            covs_train_lc, covs_test_lc = apply_logcenter(covs_train, covs_test, args.spd_eps)
            X_train_base = covs_to_features(covs_train_lc).astype(np.float32)
            X_test = covs_to_features(covs_test_lc).astype(np.float32)
            y_train_base = np.asarray(y_train).astype(int).ravel()
            y_test = np.asarray(y_test).astype(int).ravel()
            tid_train = np.asarray(tid_train)
            tid_test = np.asarray(tid_test)
            print(
                f"[freeze][{dataset}][seed={seed}] logcenter_vectorize done "
                f"train_shape={tuple(X_train_base.shape)} test_shape={tuple(X_test.shape)} "
                f"in {time.time() - stage_t0:.1f}s",
                flush=True,
            )

            _write_seed_status(
                status_path,
                dataset=dataset,
                seed=int(seed),
                stage="fit_gate1",
                seed_started_at=seed_started_at,
                extra={
                    "train_feature_shape": [int(X_train_base.shape[0]), int(X_train_base.shape[1])],
                    "test_feature_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
                },
            )
            stage_t0 = time.time()
            mu_gate1, tau_gate1, gate1_fit_meta = _fit_gate1_from_train(
                X_train=X_train_base,
                y_train=y_train_base,
                q=float(args.gate1_q),
            )
            print(
                f"[freeze][{dataset}][seed={seed}] fit_gate1 done in {time.time() - stage_t0:.1f}s",
                flush=True,
            )

            cap_seed = int(seed) + 41
            _write_seed_status(
                status_path,
                dataset=dataset,
                seed=int(seed),
                stage="fit_baseline",
                seed_started_at=seed_started_at,
            )
            stage_t0 = time.time()
            metrics_a, train_meta_a = _fit_eval_linearsvc(
                X_train_base,
                y_train_base,
                tid_train,
                X_test,
                y_test,
                tid_test,
                seed=int(seed),
                cap_k=int(args.window_cap_k),
                cap_seed=cap_seed,
                cap_sampling_policy=args.cap_sampling_policy,
                linear_c=float(args.linear_c),
                class_weight=args.linear_class_weight,
                max_iter=int(args.linear_max_iter),
                agg_mode=args.aggregation_mode,
                is_aug_train=np.zeros((len(y_train_base),), dtype=bool),
                progress_prefix=f"[freeze][{dataset}][seed={seed}][fit_baseline]",
            )
            print(
                f"[freeze][{dataset}][seed={seed}] fit_baseline done "
                f"f1={metrics_a['trial_macro_f1']:.4f} in {time.time() - stage_t0:.1f}s",
                flush=True,
            )

            _write_seed_status(
                status_path,
                dataset=dataset,
                seed=int(seed),
                stage="build_step1b",
                seed_started_at=seed_started_at,
            )
            stage_t0 = time.time()
            direction_bank, bank_meta = _build_direction_bank_d1(
                X_train=X_train_base,
                k_dir=int(args.k_dir),
                seed=int(seed * 10000 + args.k_dir * 113 + 17),
                n_iters=int(args.pia_n_iters),
                activation=args.pia_activation,
                bias_update_mode=args.pia_bias_update_mode,
                c_repr=float(args.pia_c_repr),
            )
            X_ck, y_ck, tid_ck, src_ck, dir_ck, ck_aug_meta = _build_multidir_aug_candidates(
                X_train=X_train_base,
                y_train=y_train_base,
                tid_train=tid_train,
                direction_bank=direction_bank,
                subset_size=int(args.subset_size),
                gamma=float(args.pia_gamma),
                multiplier=int(args.pia_multiplier),
                seed=int(seed + 100000 + args.k_dir * 101 + args.subset_size * 7),
            )
            print(
                f"[freeze][{dataset}][seed={seed}] build_step1b done "
                f"aug={len(y_ck)} in {time.time() - stage_t0:.1f}s",
                flush=True,
            )

            _write_seed_status(
                status_path,
                dataset=dataset,
                seed=int(seed),
                stage="fit_step1b",
                seed_started_at=seed_started_at,
                extra={"generated_aug_windows": int(len(y_ck))},
            )
            stage_t0 = time.time()
            mech_step1b = _compute_mech_metrics(
                X_train_real=X_train_base,
                y_train_real=y_train_base,
                X_aug_generated=X_ck,
                y_aug_generated=y_ck,
                X_aug_accepted=X_ck,
                y_aug_accepted=y_ck,
                X_src_accepted=src_ck,
                dir_generated=dir_ck,
                dir_accepted=dir_ck,
                seed=int(seed),
                linear_c=float(args.linear_c),
                class_weight=args.linear_class_weight,
                linear_max_iter=int(args.linear_max_iter),
                knn_k=int(args.mech_knn_k),
                max_aug_for_mech=int(args.mech_max_aug_for_metrics),
                max_real_knn_ref=int(args.mech_max_real_knn_ref),
                max_real_knn_query=int(args.mech_max_real_knn_query),
                progress_prefix=f"[freeze][{dataset}][seed={seed}][mech_step1b]",
            )
            X_train_step1b = np.vstack([X_train_base, X_ck]) if len(y_ck) else X_train_base.copy()
            y_train_step1b = np.concatenate([y_train_base, y_ck]) if len(y_ck) else y_train_base.copy()
            tid_train_step1b = np.concatenate([tid_train, tid_ck]) if len(y_ck) else tid_train.copy()
            is_aug_step1b = (
                np.concatenate(
                    [
                        np.zeros((len(y_train_base),), dtype=bool),
                        np.ones((len(y_ck),), dtype=bool),
                    ]
                )
                if len(y_ck)
                else np.zeros((len(y_train_base),), dtype=bool)
            )
            metrics_step1b, train_meta_step1b = _fit_eval_linearsvc(
                X_train_step1b,
                y_train_step1b,
                tid_train_step1b,
                X_test,
                y_test,
                tid_test,
                seed=int(seed),
                cap_k=int(args.window_cap_k),
                cap_seed=cap_seed,
                cap_sampling_policy=args.cap_sampling_policy,
                linear_c=float(args.linear_c),
                class_weight=args.linear_class_weight,
                max_iter=int(args.linear_max_iter),
                agg_mode=args.aggregation_mode,
                is_aug_train=is_aug_step1b,
                progress_prefix=f"[freeze][{dataset}][seed={seed}][fit_step1b]",
            )
            print(
                f"[freeze][{dataset}][seed={seed}] fit_step1b done "
                f"f1={metrics_step1b['trial_macro_f1']:.4f} in {time.time() - stage_t0:.1f}s",
                flush=True,
            )

            _write_seed_status(
                status_path,
                dataset=dataset,
                seed=int(seed),
                stage="apply_gate",
                seed_started_at=seed_started_at,
                extra={"generated_aug_windows": int(len(y_ck))},
            )
            stage_t0 = time.time()
            X_ck_keep, y_ck_keep, tid_ck_keep, src_ck_keep, ck_gate_meta = _apply_gates(
                X_aug=X_ck,
                y_aug=y_ck,
                tid_aug=tid_ck,
                src_aug=src_ck,
                mu_y=mu_gate1,
                tau_y=tau_gate1,
                enable_gate2=True,
                gate2_q_src=float(args.gate2_q_src),
            )
            print(
                f"[freeze][{dataset}][seed={seed}] apply_gate done "
                f"kept={len(y_ck_keep)}/{len(y_ck)} "
                f"accept={ck_gate_meta['accept_rate_final']:.3f} "
                f"in {time.time() - stage_t0:.1f}s",
                flush=True,
            )
            keep_mask = np.zeros((len(y_ck),), dtype=bool)
            # Reconstruct keep mask using the deterministic Gate1 / Gate2 rules.
            # This avoids fragile matching against accepted arrays when duplicates exist.
            d_center = np.zeros((len(y_ck),), dtype=np.float64)
            keep1 = np.zeros((len(y_ck),), dtype=bool)
            y_ck_int = np.asarray(y_ck).astype(int).ravel()
            for i, cls in enumerate(y_ck_int.tolist()):
                muc = mu_gate1.get(int(cls))
                tauc = tau_gate1.get(int(cls))
                if muc is None or tauc is None:
                    d_center[i] = np.inf
                    keep1[i] = False
                    continue
                di = float(np.linalg.norm(X_ck[i] - muc))
                d_center[i] = di
                keep1[i] = di <= float(tauc)
            d_src = np.linalg.norm(np.asarray(X_ck, dtype=np.float32) - np.asarray(src_ck, dtype=np.float32), axis=1)
            tau_src_y: Dict[int, float] = {}
            for cls in sorted(np.unique(y_ck_int).tolist()):
                ds = d_src[y_ck_int == cls]
                if ds.size:
                    tau_src_y[int(cls)] = float(np.percentile(ds, float(args.gate2_q_src)))
            keep2 = np.asarray(
                [d_src[i] <= tau_src_y.get(int(y_ck_int[i]), -np.inf) for i in range(len(y_ck_int))],
                dtype=bool,
            )
            keep_mask = keep1 & keep2
            if int(np.sum(keep_mask)) != int(len(y_ck_keep)):
                raise RuntimeError(
                    f"Gate keep mismatch for {dataset}/seed{seed}: "
                    f"mask={int(np.sum(keep_mask))} apply={int(len(y_ck_keep))}"
                )

            _write_seed_status(
                status_path,
                dataset=dataset,
                seed=int(seed),
                stage="fit_step1b_gate",
                seed_started_at=seed_started_at,
                extra={
                    "generated_aug_windows": int(len(y_ck)),
                    "accepted_aug_windows": int(len(y_ck_keep)),
                },
            )
            stage_t0 = time.time()
            mech_step1b_gate = _compute_mech_metrics(
                X_train_real=X_train_base,
                y_train_real=y_train_base,
                X_aug_generated=X_ck,
                y_aug_generated=y_ck,
                X_aug_accepted=X_ck_keep,
                y_aug_accepted=y_ck_keep,
                X_src_accepted=src_ck_keep,
                dir_generated=dir_ck,
                dir_accepted=np.asarray(dir_ck, dtype=np.int64)[keep_mask],
                seed=int(seed),
                linear_c=float(args.linear_c),
                class_weight=args.linear_class_weight,
                linear_max_iter=int(args.linear_max_iter),
                knn_k=int(args.mech_knn_k),
                max_aug_for_mech=int(args.mech_max_aug_for_metrics),
                max_real_knn_ref=int(args.mech_max_real_knn_ref),
                max_real_knn_query=int(args.mech_max_real_knn_query),
                progress_prefix=f"[freeze][{dataset}][seed={seed}][mech_step1b_gate]",
            )
            X_train_gate = np.vstack([X_train_base, X_ck_keep]) if len(y_ck_keep) else X_train_base.copy()
            y_train_gate = np.concatenate([y_train_base, y_ck_keep]) if len(y_ck_keep) else y_train_base.copy()
            tid_train_gate = np.concatenate([tid_train, tid_ck_keep]) if len(y_ck_keep) else tid_train.copy()
            is_aug_gate = (
                np.concatenate(
                    [
                        np.zeros((len(y_train_base),), dtype=bool),
                        np.ones((len(y_ck_keep),), dtype=bool),
                    ]
                )
                if len(y_ck_keep)
                else np.zeros((len(y_train_base),), dtype=bool)
            )
            metrics_step1b_gate, train_meta_step1b_gate = _fit_eval_linearsvc(
                X_train_gate,
                y_train_gate,
                tid_train_gate,
                X_test,
                y_test,
                tid_test,
                seed=int(seed),
                cap_k=int(args.window_cap_k),
                cap_seed=cap_seed,
                cap_sampling_policy=args.cap_sampling_policy,
                linear_c=float(args.linear_c),
                class_weight=args.linear_class_weight,
                max_iter=int(args.linear_max_iter),
                agg_mode=args.aggregation_mode,
                is_aug_train=is_aug_gate,
                progress_prefix=f"[freeze][{dataset}][seed={seed}][fit_step1b_gate]",
            )
            print(
                f"[freeze][{dataset}][seed={seed}] fit_step1b_gate done "
                f"f1={metrics_step1b_gate['trial_macro_f1']:.4f} in {time.time() - stage_t0:.1f}s",
                flush=True,
            )
            common_meta = {
                "dataset": dataset,
                "protocol_type": split_meta["protocol_type"],
                "protocol_note": split_meta["protocol_note"],
                "seed": int(seed),
                "split_hash": split_meta["split_hash"],
                "train_count_trials": int(split_meta["train_count_trials"]),
                "test_count_trials": int(split_meta["test_count_trials"]),
                "train_trial_ids_preview": split_meta["train_trial_ids"][: max(0, int(args.split_preview_n))],
                "test_trial_ids_preview": split_meta["test_trial_ids"][: max(0, int(args.split_preview_n))],
                "window_cap_k": int(args.window_cap_k),
                "cap_sampling_policy": args.cap_sampling_policy,
                "aggregation_mode": args.aggregation_mode,
                "feature_pipeline": {
                    "window_sec": float(args.window_sec),
                    "hop_sec": float(args.hop_sec),
                    "cov_est": args.cov_est,
                    "spd_eps": float(args.spd_eps),
                    "center": "logcenter_train_only",
                    "vectorize": "upper_triangle",
                    "bands": bands_spec,
                },
                "step1b_config": {
                    "k_dir": int(args.k_dir),
                    "subset_size": int(args.subset_size),
                    "pia_multiplier": int(args.pia_multiplier),
                    "pia_gamma": float(args.pia_gamma),
                    "pia_n_iters": int(args.pia_n_iters),
                    "pia_activation": args.pia_activation,
                    "pia_bias_update_mode": args.pia_bias_update_mode,
                    "pia_c_repr": float(args.pia_c_repr),
                },
                "gate_config": {
                    "gate1_q": float(args.gate1_q),
                    "gate2_enabled": True,
                    "gate2_q_src": float(args.gate2_q_src),
                },
                "test_augmentation": "disabled",
            }

            _write_condition(
                os.path.join(seed_dir, "A_baseline"),
                metrics_a,
                {**common_meta, **train_meta_a, "condition": "A_baseline"},
            )
            _write_condition(
                os.path.join(seed_dir, "B_step1b"),
                metrics_step1b,
                {
                    **common_meta,
                    **train_meta_step1b,
                    "condition": "B_step1b",
                    "augmentation": ck_aug_meta,
                    "direction_bank": {**bank_meta, "subset_size": int(args.subset_size)},
                    "final_accept_rate": 1.0,
                    "mech": mech_step1b,
                },
            )
            _write_condition(
                os.path.join(seed_dir, "C_step1b_gate"),
                metrics_step1b_gate,
                {
                    **common_meta,
                    **train_meta_step1b_gate,
                    "condition": "C_step1b_gate",
                    "augmentation": ck_aug_meta,
                    "direction_bank": {**bank_meta, "subset_size": int(args.subset_size)},
                    "gate1_fit": gate1_fit_meta,
                    "gate_apply": ck_gate_meta,
                    "final_accept_rate": float(ck_gate_meta["accept_rate_final"]),
                    "mech": mech_step1b_gate,
                },
            )

            perf_row = {
                "dataset": dataset,
                "seed": int(seed),
                "protocol_type": split_meta["protocol_type"],
                "protocol_note": split_meta["protocol_note"],
                "split_hash": split_meta["split_hash"],
                "baseline_acc": float(metrics_a["trial_acc"]),
                "baseline_f1": float(metrics_a["trial_macro_f1"]),
                "step1b_acc": float(metrics_step1b["trial_acc"]),
                "step1b_f1": float(metrics_step1b["trial_macro_f1"]),
                "step1b_gate_acc": float(metrics_step1b_gate["trial_acc"]),
                "step1b_gate_f1": float(metrics_step1b_gate["trial_macro_f1"]),
                "step1b_gate_accept_rate": float(ck_gate_meta["accept_rate_final"]),
                "window_cap_k": int(args.window_cap_k),
                "bands": bands_spec,
            }
            dataset_rows.append(perf_row)
            all_perf_rows.append(perf_row)

            for step_variant, mech in [
                ("step1b", mech_step1b),
                ("step1b_gate", mech_step1b_gate),
            ]:
                dir_summary = _summarize_dir_profile(mech.get("dir_profile", {}))
                mech_row = {
                    "dataset": dataset,
                    "seed": int(seed),
                    "protocol_type": split_meta["protocol_type"],
                    "step_variant": step_variant,
                    "flip_rate": float(mech.get("flip_rate", 0.0)),
                    "margin_drop_median": float(mech.get("margin_drop_median", 0.0)),
                    "knn_intrusion_rate": float(mech.get("knn_intrusion_rate", 0.0)),
                    "real_knn_intrusion_rate": float(mech.get("real_knn_intrusion_rate", 0.0)),
                    "delta_intrusion": float(mech.get("delta_intrusion", 0.0)),
                    "n_aug_used_for_mech": int(mech.get("mech_sample_sizes", {}).get("n_aug_used_for_mech", 0)),
                    **dir_summary,
                }
                dataset_mech_rows.append(mech_row)
                all_mech_rows.append(mech_row)

            _write_seed_status(
                status_path,
                dataset=dataset,
                seed=int(seed),
                stage="done",
                seed_started_at=seed_started_at,
                extra={
                    "baseline_f1": float(metrics_a["trial_macro_f1"]),
                    "step1b_f1": float(metrics_step1b["trial_macro_f1"]),
                    "step1b_gate_f1": float(metrics_step1b_gate["trial_macro_f1"]),
                    "step1b_gate_accept_rate": float(ck_gate_meta["accept_rate_final"]),
                },
            )
            print(
                f"[freeze][{dataset}][seed={seed}] "
                f"A={metrics_a['trial_macro_f1']:.4f} "
                f"B={metrics_step1b['trial_macro_f1']:.4f} "
                f"C={metrics_step1b_gate['trial_macro_f1']:.4f} "
                f"accept={ck_gate_meta['accept_rate_final']:.3f} "
                f"total_elapsed={time.time() - seed_started_at:.1f}s",
                flush=True,
            )

        df_perf = pd.DataFrame(dataset_rows).sort_values(["dataset", "seed"]).reset_index(drop=True)
        df_perf.to_csv(os.path.join(dataset_dir, "summary_per_seed.csv"), index=False)

        agg_row = {
            "dataset": dataset,
            "protocol_type": split_meta["protocol_type"],
            "protocol_note": split_meta["protocol_note"],
            "n_seeds": int(len(df_perf)),
        }
        for col in [
            "baseline_acc",
            "baseline_f1",
            "step1b_acc",
            "step1b_f1",
            "step1b_gate_acc",
            "step1b_gate_f1",
            "step1b_gate_accept_rate",
        ]:
            stats = _summary_stats(df_perf[col].tolist())
            agg_row[f"{col}_mean"] = stats["mean"]
            agg_row[f"{col}_std"] = stats["std"]
            agg_row[f"{col}_min"] = stats["min"]
            agg_row[f"{col}_max"] = stats["max"]
        pd.DataFrame([agg_row]).to_csv(os.path.join(dataset_dir, "summary_agg.csv"), index=False)

        df_mech = pd.DataFrame(dataset_mech_rows).sort_values(["dataset", "step_variant", "seed"]).reset_index(drop=True)
        df_mech.to_csv(os.path.join(dataset_dir, "mechanism_per_seed.csv"), index=False)

    if all_perf_rows:
        perf_master = pd.DataFrame(all_perf_rows).sort_values(["dataset", "seed"]).reset_index(drop=True)
        perf_master.to_csv(os.path.join(args.out_root, "summary_per_seed.csv"), index=False)
    if all_mech_rows:
        mech_master = pd.DataFrame(all_mech_rows).sort_values(["dataset", "step_variant", "seed"]).reset_index(drop=True)
        mech_master.to_csv(os.path.join(args.out_root, "mechanism_per_seed.csv"), index=False)


if __name__ == "__main__":
    main()
