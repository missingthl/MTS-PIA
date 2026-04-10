#!/usr/bin/env python
"""Phase K1: Step1B + read-only Gate3 KNN local customs layer.

Compares paired conditions under the same split:
- A: baseline
- Ck_ref: current Step1B-style multi-direction PIA + Gate1 + Gate2
- Ck_gate3: Ck_ref + read-only Gate3 KNN local gate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.trial_dataset_factory import (  # noqa: E402
    DEFAULT_BANDS_EEG,
    DEFAULT_FINGERMOVEMENTS_ROOT,
    DEFAULT_HAR_ROOT,
    DEFAULT_MITBIH_NPZ,
    DEFAULT_NATOPS_ROOT,
    DEFAULT_SEEDIV_ROOT,
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
)
from scripts.local_knn_gate import LocalKNNGateConfig, ReadOnlyLocalKNNGate  # noqa: E402
from scripts.run_phase15_step1a_maxplane import (  # noqa: E402
    _apply_window_cap,
    _fit_eval_linearsvc,
    _fit_gate1_from_train,
    _make_trial_split,
)
from scripts.run_phase15_step1b_multidir_matrix import (  # noqa: E402
    _build_direction_bank_d1,
    _build_multidir_aug_candidates,
    _compute_mech_metrics,
    _ordered_unique,
    _summary_stats,
    _write_condition,
)


def _parse_seed_list(text: str) -> List[int]:
    out: List[int] = []
    for tok in str(text).split(","):
        t = tok.strip()
        if t:
            out.append(int(t))
    out = sorted(set(out))
    if not out:
        raise ValueError("seed list cannot be empty")
    return out


def _apply_gate12_with_diag(
    X_aug: np.ndarray,
    y_aug: np.ndarray,
    tid_aug: np.ndarray,
    src_aug: np.ndarray,
    *,
    mu_y: Dict[int, np.ndarray],
    tau_y: Dict[int, float],
    gate2_q_src: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    X = np.asarray(X_aug, dtype=np.float32)
    y = np.asarray(y_aug).astype(int).ravel()
    tid = np.asarray(tid_aug)
    src = np.asarray(src_aug, dtype=np.float32)
    n = int(len(y))
    if n == 0:
        empty = np.zeros((0,), dtype=bool)
        return X, y, tid, src, empty, empty, {
            "accept_rate_gate1": 0.0,
            "accept_rate_gate2": 1.0,
            "accept_rate_final": 0.0,
            "accepted_count": 0,
            "rejected_count": 0,
            "reject_by_gate1_count": 0,
            "reject_by_gate2_count": 0,
            "gate1_dist_accepted_summary": _summary_stats(np.asarray([], dtype=np.float64)),
            "gate1_dist_rejected_summary": _summary_stats(np.asarray([], dtype=np.float64)),
            "gate2": {
                "enabled": True,
                "q_src": float(gate2_q_src),
                "tau_src_y": {},
                "src_dist_accepted_summary": _summary_stats(np.asarray([], dtype=np.float64)),
                "src_dist_rejected_summary": _summary_stats(np.asarray([], dtype=np.float64)),
            },
        }

    d_center = np.zeros((n,), dtype=np.float64)
    keep1 = np.zeros((n,), dtype=bool)
    for i, cls in enumerate(y.tolist()):
        muc = mu_y.get(int(cls))
        tauc = tau_y.get(int(cls))
        if muc is None or tauc is None:
            d_center[i] = np.inf
            keep1[i] = False
            continue
        di = float(np.linalg.norm(X[i] - muc))
        d_center[i] = di
        keep1[i] = di <= float(tauc)

    d_src = np.linalg.norm(X - src, axis=1).astype(np.float64)
    tau_src_y: Dict[int, float] = {}
    for cls in sorted(np.unique(y).tolist()):
        ds = d_src[y == cls]
        if ds.size:
            tau_src_y[int(cls)] = float(np.percentile(ds, float(gate2_q_src)))
    keep2 = np.asarray([d_src[i] <= tau_src_y.get(int(y[i]), -np.inf) for i in range(n)], dtype=bool)
    keep = keep1 & keep2
    meta = {
        "accept_rate_gate1": float(np.mean(keep1)),
        "accept_rate_gate2": float(np.mean(keep2)),
        "accept_rate_final": float(np.mean(keep)),
        "accepted_count": int(np.sum(keep)),
        "rejected_count": int(np.sum(~keep)),
        "reject_by_gate1_count": int(np.sum(~keep1)),
        "reject_by_gate2_count": int(np.sum(keep1 & (~keep2))),
        "gate1_dist_accepted_summary": _summary_stats(d_center[keep]),
        "gate1_dist_rejected_summary": _summary_stats(d_center[~keep]),
        "gate2": {
            "enabled": True,
            "q_src": float(gate2_q_src),
            "tau_src_y": {str(k): float(v) for k, v in tau_src_y.items()},
            "src_dist_accepted_summary": _summary_stats(d_src[keep2]),
            "src_dist_rejected_summary": _summary_stats(d_src[~keep2]),
        },
    }
    return X[keep], y[keep], tid[keep], src[keep], keep1, keep2, meta


def _merge_gate3_into_dir_profile(
    mech: Dict[str, object],
    *,
    dir_ids_in: np.ndarray,
    keep3: np.ndarray,
    gate3_diag: Dict[str, object],
    gamma_value: float,
) -> Dict[str, object]:
    mech_out = json.loads(json.dumps(mech))
    profile = mech_out.get("dir_profile", {})
    if not isinstance(profile, dict):
        profile = {}

    diag = gate3_diag.get("candidate_diag_arrays", {})
    dir_arr = np.asarray(diag.get("direction_ids", []), dtype=np.int64)
    purity = np.asarray(diag.get("knn_same_class_purity", []), dtype=np.float64)
    intrusion = np.asarray(diag.get("knn_intrusion_ratio", []), dtype=np.float64)
    keep_arr = np.asarray(keep3, dtype=bool)
    if not (len(dir_arr) == len(purity) == len(intrusion) == len(keep_arr)):
        return mech_out

    max_dir = int(np.max(dir_arr)) if dir_arr.size else -1
    for i in range(max_dir + 1):
        mask_i = dir_arr == i
        n_in = int(np.sum(mask_i))
        n_rej = int(np.sum(mask_i & (~keep_arr)))
        entry = profile.get(str(i), {})
        if not isinstance(entry, dict):
            entry = {}
        entry["gate3_reject_rate_i"] = float(n_rej / n_in) if n_in > 0 else 0.0
        entry["gate3_first_reject_gamma_i"] = float(gamma_value) if n_rej > 0 else None
        entry["gate3_mean_purity_i"] = float(np.mean(purity[mask_i])) if n_in > 0 else 0.0
        entry["gate3_mean_intrusion_i"] = float(np.mean(intrusion[mask_i])) if n_in > 0 else 0.0
        profile[str(i)] = entry

    mech_out["dir_profile"] = profile
    return mech_out


def _write_gate3_rejects(cond_dir: str, reject_rows: List[Dict[str, object]]) -> None:
    if not reject_rows:
        return
    pd.DataFrame(reject_rows).to_csv(os.path.join(cond_dir, "gate3_rejects.csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="seed1", choices=["seed1", "seed", "har", "mitbih", "seediv", "natops", "fingermovements"])
    parser.add_argument("--seeds", type=str, default="3")
    parser.add_argument("--out-root", type=str, default="out/phase15_k1_knn_gate")
    parser.add_argument("--har-root", type=str, default=DEFAULT_HAR_ROOT)
    parser.add_argument("--mitbih-npz", type=str, default=DEFAULT_MITBIH_NPZ)
    parser.add_argument("--seediv-root", type=str, default=DEFAULT_SEEDIV_ROOT)
    parser.add_argument("--natops-root", type=str, default=DEFAULT_NATOPS_ROOT)
    parser.add_argument("--fingermovements-root", type=str, default=DEFAULT_FINGERMOVEMENTS_ROOT)
    parser.add_argument("--processed-root", type=str, default="data/SEED/SEED_EEG/Preprocessed_EEG")
    parser.add_argument("--stim-xlsx", type=str, default="data/SEED/SEED_EEG/SEED_stimulation.xlsx")
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
    parser.add_argument("--kdir", type=int, default=5)
    parser.add_argument("--subset-size", type=int, default=1)
    parser.add_argument("--pia-multiplier", type=int, default=1)
    parser.add_argument("--pia-gamma", type=float, default=0.10)
    parser.add_argument("--pia-n-iters", type=int, default=2)
    parser.add_argument("--pia-activation", type=str, default="sine")
    parser.add_argument("--pia-bias-update-mode", type=str, default="residual")
    parser.add_argument("--pia-c-repr", type=float, default=1.0)
    parser.add_argument("--gate1-q", type=float, default=95.0)
    parser.add_argument("--gate2-q-src", type=float, default=90.0)
    parser.add_argument("--gate3-k", type=int, default=20)
    parser.add_argument("--gate3-tau-purity", type=float, default=0.7)
    parser.add_argument("--gate3-anchor-cap-k", type=int, default=120)
    parser.add_argument("--gate3-knn-algorithm", type=str, default="auto", choices=["auto", "ball_tree", "kd_tree", "brute"])
    parser.add_argument("--gate3-query-batch-size", type=int, default=4096)
    parser.add_argument("--mech-knn-k", type=int, default=20)
    parser.add_argument("--mech-max-aug-for-metrics", type=int, default=500)
    parser.add_argument("--mech-max-real-knn-ref", type=int, default=3000)
    parser.add_argument("--mech-max-real-knn-query", type=int, default=300)
    parser.add_argument("--skip-mech", action="store_true")
    parser.add_argument("--split-preview-n", type=int, default=5)
    args = parser.parse_args()
    args.dataset = normalize_dataset_name(args.dataset)

    seeds = _parse_seed_list(args.seeds)
    if int(args.window_cap_k) <= 0:
        raise ValueError("--window-cap-k must be > 0.")
    if int(args.gate3_k) <= 0:
        raise ValueError("--gate3-k must be > 0.")
    if not (0.0 <= float(args.gate3_tau_purity) <= 1.0):
        raise ValueError("--gate3-tau-purity must be in [0,1].")
    if int(args.gate3_query_batch_size) <= 0:
        raise ValueError("--gate3-query-batch-size must be > 0.")

    all_trials = load_trials_for_dataset(
        dataset=args.dataset,
        processed_root=args.processed_root,
        stim_xlsx=args.stim_xlsx,
        har_root=args.har_root,
        mitbih_npz=args.mitbih_npz,
        seediv_root=args.seediv_root,
        natops_root=args.natops_root,
        fingermovements_root=args.fingermovements_root,
    )
    bands_spec = resolve_band_spec(args.dataset, args.bands)
    bands = parse_band_spec(bands_spec)
    out_root = os.path.join(args.out_root, args.dataset)
    ensure_dir(out_root)

    summary_rows: List[Dict[str, object]] = []

    for seed in seeds:
        seed_t0 = time.perf_counter()
        print(f"[K1][{args.dataset}][seed={seed}] start")
        train_trials, test_trials, split_meta = _make_trial_split(all_trials, seed=int(seed))
        feat_t0 = time.perf_counter()
        covs_train, y_train, tid_train = extract_features_block(
            train_trials, args.window_sec, args.hop_sec, args.cov_est, args.spd_eps, bands
        )
        covs_test, y_test, tid_test = extract_features_block(
            test_trials, args.window_sec, args.hop_sec, args.cov_est, args.spd_eps, bands
        )
        covs_train_lc, covs_test_lc = apply_logcenter(covs_train, covs_test, args.spd_eps)
        X_train_base = covs_to_features(covs_train_lc).astype(np.float32)
        X_test = covs_to_features(covs_test_lc).astype(np.float32)
        feature_extraction_time_sec = float(time.perf_counter() - feat_t0)
        y_train_base = np.asarray(y_train).astype(int).ravel()
        y_test = np.asarray(y_test).astype(int).ravel()
        tid_train = np.asarray(tid_train)
        tid_test = np.asarray(tid_test)
        print(f"[K1][{args.dataset}][seed={seed}] train_windows={len(y_train_base)} test_windows={len(y_test)} dim={X_train_base.shape[1]}")
        print(f"[K1][{args.dataset}][seed={seed}] feature_extraction_time={feature_extraction_time_sec:.3f}s")

        mu_gate1, tau_gate1, gate1_fit_meta = _fit_gate1_from_train(
            X_train=X_train_base,
            y_train=y_train_base,
            q=float(args.gate1_q),
        )
        cap_seed = int(seed) + 41
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
        )

        bank_seed = int(seed * 10000 + int(args.kdir) * 113 + 17)
        cand_t0 = time.perf_counter()
        direction_bank, bank_meta = _build_direction_bank_d1(
            X_train=X_train_base,
            k_dir=int(args.kdir),
            seed=bank_seed,
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
            seed=int(seed + 100000 + int(args.kdir) * 101 + int(args.subset_size) * 7),
        )

        X_ck_ref, y_ck_ref, tid_ck_ref, src_ck_ref, keep1, keep2, gate12_meta = _apply_gate12_with_diag(
            X_ck,
            y_ck,
            tid_ck,
            src_ck,
            mu_y=mu_gate1,
            tau_y=tau_gate1,
            gate2_q_src=float(args.gate2_q_src),
        )
        keep12 = keep1 & keep2
        dir_ck_ref = np.asarray(dir_ck, dtype=np.int64)[keep12]
        candidate_generation_time_sec = float(time.perf_counter() - cand_t0)
        print(f"[K1][{args.dataset}][seed={seed}] candidate_generation_time={candidate_generation_time_sec:.3f}s")

        if bool(args.skip_mech):
            mech_ref: Dict[str, object] = {}
        else:
            mech_ref = _compute_mech_metrics(
                X_train_real=X_train_base,
                y_train_real=y_train_base,
                X_aug_generated=X_ck,
                y_aug_generated=y_ck,
                X_aug_accepted=X_ck_ref,
                y_aug_accepted=y_ck_ref,
                X_src_accepted=src_ck_ref,
                dir_generated=dir_ck,
                dir_accepted=dir_ck_ref,
                seed=int(seed),
                linear_c=float(args.linear_c),
                class_weight=args.linear_class_weight,
                linear_max_iter=int(args.linear_max_iter),
                knn_k=int(args.mech_knn_k),
                max_aug_for_mech=int(args.mech_max_aug_for_metrics),
                max_real_knn_ref=int(args.mech_max_real_knn_ref),
                max_real_knn_query=int(args.mech_max_real_knn_query),
            )

        X_train_ck_ref = np.vstack([X_train_base, X_ck_ref]) if len(y_ck_ref) else X_train_base.copy()
        y_train_ck_ref = np.concatenate([y_train_base, y_ck_ref]) if len(y_ck_ref) else y_train_base.copy()
        tid_train_ck_ref = np.concatenate([tid_train, tid_ck_ref]) if len(y_ck_ref) else tid_train.copy()
        is_aug_ck_ref = (
            np.concatenate([np.zeros((len(y_train_base),), dtype=bool), np.ones((len(y_ck_ref),), dtype=bool)])
            if len(y_ck_ref)
            else np.zeros((len(y_train_base),), dtype=bool)
        )
        metrics_ck_ref, train_meta_ck_ref = _fit_eval_linearsvc(
            X_train_ck_ref,
            y_train_ck_ref,
            tid_train_ck_ref,
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
            is_aug_train=is_aug_ck_ref,
        )

        X_anchor, y_anchor, tid_anchor, _, anchor_counts, _ = _apply_window_cap(
            X_train_base,
            y_train_base,
            tid_train,
            cap_k=int(args.gate3_anchor_cap_k),
            seed=int(seed) + 7301,
            is_aug=np.zeros((len(y_train_base),), dtype=bool),
            policy="random",
        )
        gate3 = ReadOnlyLocalKNNGate(
            LocalKNNGateConfig(
                k=int(args.gate3_k),
                tau_purity=float(args.gate3_tau_purity),
                algorithm=str(args.gate3_knn_algorithm),
                query_batch_size=int(args.gate3_query_batch_size),
            )
        ).fit(X_anchor, y_anchor)
        gate3_stage_t0 = time.perf_counter()
        keep3, gate3_diag = gate3.evaluate_batch(
            X_ck_ref,
            y_ck_ref,
            direction_ids=dir_ck_ref,
            gamma_used=np.full((len(y_ck_ref),), float(args.pia_gamma), dtype=np.float64),
            source_tids=tid_ck_ref,
        )
        gate3_stage_time_sec = float(time.perf_counter() - gate3_stage_t0)
        print(f"[K1][{args.dataset}][seed={seed}] gate3_runtime_time={gate3_stage_time_sec:.3f}s")
        X_ck_g3 = X_ck_ref[keep3]
        y_ck_g3 = y_ck_ref[keep3]
        tid_ck_g3 = tid_ck_ref[keep3]
        src_ck_g3 = src_ck_ref[keep3]
        dir_ck_g3 = dir_ck_ref[keep3]
        if bool(args.skip_mech):
            mech_g3 = {}
        else:
            mech_g3 = _compute_mech_metrics(
                X_train_real=X_train_base,
                y_train_real=y_train_base,
                X_aug_generated=X_ck,
                y_aug_generated=y_ck,
                X_aug_accepted=X_ck_g3,
                y_aug_accepted=y_ck_g3,
                X_src_accepted=src_ck_g3,
                dir_generated=dir_ck,
                dir_accepted=dir_ck_g3,
                seed=int(seed),
                linear_c=float(args.linear_c),
                class_weight=args.linear_class_weight,
                linear_max_iter=int(args.linear_max_iter),
                knn_k=int(args.mech_knn_k),
                max_aug_for_mech=int(args.mech_max_aug_for_metrics),
                max_real_knn_ref=int(args.mech_max_real_knn_ref),
                max_real_knn_query=int(args.mech_max_real_knn_query),
            )
            mech_g3 = _merge_gate3_into_dir_profile(
                mech_g3,
                dir_ids_in=dir_ck_ref,
                keep3=keep3,
                gate3_diag=gate3_diag,
                gamma_value=float(args.pia_gamma),
            )

        X_train_ck_g3 = np.vstack([X_train_base, X_ck_g3]) if len(y_ck_g3) else X_train_base.copy()
        y_train_ck_g3 = np.concatenate([y_train_base, y_ck_g3]) if len(y_ck_g3) else y_train_base.copy()
        tid_train_ck_g3 = np.concatenate([tid_train, tid_ck_g3]) if len(y_ck_g3) else tid_train.copy()
        is_aug_ck_g3 = (
            np.concatenate([np.zeros((len(y_train_base),), dtype=bool), np.ones((len(y_ck_g3),), dtype=bool)])
            if len(y_ck_g3)
            else np.zeros((len(y_train_base),), dtype=bool)
        )
        metrics_ck_g3, train_meta_ck_g3 = _fit_eval_linearsvc(
            X_train_ck_g3,
            y_train_ck_g3,
            tid_train_ck_g3,
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
            is_aug_train=is_aug_ck_g3,
        )

        common_meta = {
            "seed": int(seed),
            "split_hash": split_meta["split_hash"],
            "train_count_trials": int(split_meta["train_count_trials"]),
            "test_count_trials": int(split_meta["test_count_trials"]),
            "train_trial_ids_preview": split_meta["train_trial_ids"][: max(0, int(args.split_preview_n))],
            "test_trial_ids_preview": split_meta["test_trial_ids"][: max(0, int(args.split_preview_n))],
            "window_cap_K": int(args.window_cap_k),
            "cap_sampling_policy": args.cap_sampling_policy,
            "dataset": args.dataset,
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
            "test_augmentation": "disabled",
            "runtime_breakdown_sec": {
                "feature_extraction_time": feature_extraction_time_sec,
                "candidate_generation_time": candidate_generation_time_sec,
                "gate3_runtime_time": gate3_stage_time_sec,
                "gate3_query_runtime_time": float(gate3_diag["gate3_runtime_sec"]),
                "total_runtime": float(time.perf_counter() - seed_t0),
            },
        }
        tau_tag = str(float(args.gate3_tau_purity)).replace(".", "p")
        setting_tag = f"kdir{int(args.kdir)}_s{int(args.subset_size)}__g3k{int(args.gate3_k)}_tau{tau_tag}"
        setting_dir = os.path.join(out_root, setting_tag, f"seed{seed}")
        cond_dirs = {
            "A_baseline": os.path.join(setting_dir, "A_baseline"),
            "Ck_ref": os.path.join(setting_dir, "Ck_ref"),
            "Ck_gate3": os.path.join(setting_dir, "Ck_gate3"),
        }

        _write_condition(
            cond_dirs["A_baseline"],
            metrics_a,
            {
                **common_meta,
                **train_meta_a,
                "condition": "A_baseline",
                "setting": setting_tag,
            },
        )
        _write_condition(
            cond_dirs["Ck_ref"],
            metrics_ck_ref,
            {
                **common_meta,
                **train_meta_ck_ref,
                "condition": "Ck_ref",
                "setting": setting_tag,
                "augmentation": ck_aug_meta,
                "direction_bank": {**bank_meta, "subset_size": int(args.subset_size)},
                "gate1_fit": gate1_fit_meta,
                "gate_apply": gate12_meta,
                "gate2_config": {"enabled": True, "q_src": float(args.gate2_q_src)},
                "gate3_enabled": False,
                "gate3_anchor_space": "clean_train_only",
                "gate3_k": int(args.gate3_k),
                "gate3_tau_purity": float(args.gate3_tau_purity),
                "gate3_query_mode": "batch",
                "gate3_knn_algorithm": str(args.gate3_knn_algorithm),
                "gate3_query_batch_size": int(args.gate3_query_batch_size),
                "gate3_runtime_sec": 0.0,
                "gate3_accept_count": 0,
                "gate3_reject_count": 0,
                "gate3_accept_rate": 0.0,
                "gate3_early_stop_enabled": False,
                "gate3_reject_reason_summary": {},
                "final_accept_rate": float(gate12_meta["accept_rate_final"]),
                "mech": mech_ref,
            },
        )
        _write_condition(
            cond_dirs["Ck_gate3"],
            metrics_ck_g3,
            {
                **common_meta,
                **train_meta_ck_g3,
                "condition": "Ck_gate3",
                "setting": setting_tag,
                "augmentation": ck_aug_meta,
                "direction_bank": {**bank_meta, "subset_size": int(args.subset_size)},
                "gate1_fit": gate1_fit_meta,
                "gate_apply": gate12_meta,
                "gate2_config": {"enabled": True, "q_src": float(args.gate2_q_src)},
                "gate3_enabled": True,
                "gate3_anchor_space": "clean_train_only",
                "gate3_anchor_n": int(len(y_anchor)),
                "gate3_anchor_cap_k": int(args.gate3_anchor_cap_k),
                "gate3_anchor_per_trial_stats": {
                    "mean": float(np.mean(list(anchor_counts.values()))) if anchor_counts else 0.0,
                    "std": float(np.std(list(anchor_counts.values()))) if anchor_counts else 0.0,
                    "min": float(np.min(list(anchor_counts.values()))) if anchor_counts else 0.0,
                    "max": float(np.max(list(anchor_counts.values()))) if anchor_counts else 0.0,
                },
                "gate3_k": int(args.gate3_k),
                "gate3_tau_purity": float(args.gate3_tau_purity),
                "gate3_query_mode": str(gate3_diag["gate3_query_mode"]),
                "gate3_knn_algorithm": str(gate3_diag["gate3_knn_algorithm"]),
                "gate3_query_batch_size": int(gate3_diag["gate3_query_batch_size"]),
                "gate3_runtime_sec": float(gate3_diag["gate3_runtime_sec"]),
                "gate3_accept_count": int(gate3_diag["gate3_accept_count"]),
                "gate3_reject_count": int(gate3_diag["gate3_reject_count"]),
                "gate3_accept_rate": float(gate3_diag["gate3_accept_rate"]),
                "gate3_early_stop_enabled": False,
                "gate3_reject_reason_summary": gate3_diag["gate3_reject_reason_summary"],
                "gate3_gamma_stats": gate3_diag.get("gamma_stats", {}),
                "final_accept_rate": float(gate12_meta["accept_rate_final"] * gate3_diag["gate3_accept_rate"]),
                "mech": mech_g3,
            },
        )
        _write_gate3_rejects(cond_dirs["Ck_gate3"], gate3_diag["candidate_rows"])

        summary_rows.extend(
            [
                {
                    "dataset": args.dataset,
                    "seed": int(seed),
                    "condition": "A_baseline",
                    "setting": setting_tag,
                    "split_hash": split_meta["split_hash"],
                    "trial_acc": metrics_a["trial_acc"],
                    "trial_macro_f1": metrics_a["trial_macro_f1"],
                    "accept_rate_total": 1.0,
                    "reject_by_gate1_count": 0,
                    "reject_by_gate2_count": 0,
                    "reject_by_gate3_count": 0,
                    "gate3_accept_rate": 0.0,
                    "gate3_runtime_sec": 0.0,
                    "gate3_knn_algorithm": str(args.gate3_knn_algorithm),
                    "skip_mech": bool(args.skip_mech),
                    "flip_rate": None,
                    "margin_drop_median": None,
                    "knn_intrusion_rate": None,
                    "feature_extraction_time": feature_extraction_time_sec,
                    "candidate_generation_time": candidate_generation_time_sec,
                    "gate3_runtime_time": 0.0,
                    "total_runtime": float(time.perf_counter() - seed_t0),
                },
                {
                    "dataset": args.dataset,
                    "seed": int(seed),
                    "condition": "Ck_ref",
                    "setting": setting_tag,
                    "split_hash": split_meta["split_hash"],
                    "trial_acc": metrics_ck_ref["trial_acc"],
                    "trial_macro_f1": metrics_ck_ref["trial_macro_f1"],
                    "accept_rate_total": gate12_meta["accept_rate_final"],
                    "reject_by_gate1_count": gate12_meta["reject_by_gate1_count"],
                    "reject_by_gate2_count": gate12_meta["reject_by_gate2_count"],
                    "reject_by_gate3_count": 0,
                    "gate3_accept_rate": 0.0,
                    "gate3_runtime_sec": 0.0,
                    "gate3_knn_algorithm": str(args.gate3_knn_algorithm),
                    "skip_mech": bool(args.skip_mech),
                    "flip_rate": mech_ref.get("flip_rate"),
                    "margin_drop_median": mech_ref.get("margin_drop_median"),
                    "knn_intrusion_rate": mech_ref.get("knn_intrusion_rate"),
                    "feature_extraction_time": feature_extraction_time_sec,
                    "candidate_generation_time": candidate_generation_time_sec,
                    "gate3_runtime_time": 0.0,
                    "total_runtime": float(time.perf_counter() - seed_t0),
                },
                {
                    "dataset": args.dataset,
                    "seed": int(seed),
                    "condition": "Ck_gate3",
                    "setting": setting_tag,
                    "split_hash": split_meta["split_hash"],
                    "trial_acc": metrics_ck_g3["trial_acc"],
                    "trial_macro_f1": metrics_ck_g3["trial_macro_f1"],
                    "accept_rate_total": gate12_meta["accept_rate_final"] * gate3_diag["gate3_accept_rate"],
                    "reject_by_gate1_count": gate12_meta["reject_by_gate1_count"],
                    "reject_by_gate2_count": gate12_meta["reject_by_gate2_count"],
                    "reject_by_gate3_count": gate3_diag["gate3_reject_count"],
                    "gate3_accept_rate": gate3_diag["gate3_accept_rate"],
                    "gate3_runtime_sec": gate3_diag["gate3_runtime_sec"],
                    "gate3_knn_algorithm": gate3_diag["gate3_knn_algorithm"],
                    "skip_mech": bool(args.skip_mech),
                    "flip_rate": mech_g3.get("flip_rate"),
                    "margin_drop_median": mech_g3.get("margin_drop_median"),
                    "knn_intrusion_rate": mech_g3.get("knn_intrusion_rate"),
                    "feature_extraction_time": feature_extraction_time_sec,
                    "candidate_generation_time": candidate_generation_time_sec,
                    "gate3_runtime_time": gate3_stage_time_sec,
                    "total_runtime": float(time.perf_counter() - seed_t0),
                },
            ]
        )
        print(
            f"[K1][{args.dataset}][seed={seed}] "
            f"A={metrics_a['trial_macro_f1']:.4f} "
            f"Ck_ref={metrics_ck_ref['trial_macro_f1']:.4f} "
            f"Ck_gate3={metrics_ck_g3['trial_macro_f1']:.4f} "
            f"gate3_accept={gate3_diag['gate3_accept_rate']:.3f}"
        )
        print(f"[K1][{args.dataset}][seed={seed}] total_runtime={float(time.perf_counter() - seed_t0):.3f}s")

    summary_df = pd.DataFrame(summary_rows).sort_values(["seed", "condition"]).reset_index(drop=True)
    summary_df.to_csv(os.path.join(out_root, "summary_per_seed.csv"), index=False)
    agg_df = (
        summary_df.groupby(["dataset", "condition"], as_index=False)
        .agg(
            trial_acc_mean=("trial_acc", "mean"),
            trial_acc_std=("trial_acc", "std"),
            trial_macro_f1_mean=("trial_macro_f1", "mean"),
            trial_macro_f1_std=("trial_macro_f1", "std"),
            accept_rate_total_mean=("accept_rate_total", "mean"),
            gate3_accept_rate_mean=("gate3_accept_rate", "mean"),
            gate3_runtime_sec_mean=("gate3_runtime_sec", "mean"),
            reject_by_gate3_count_mean=("reject_by_gate3_count", "mean"),
        )
    )
    agg_df.to_csv(os.path.join(out_root, "summary_agg.csv"), index=False)


if __name__ == "__main__":
    main()
