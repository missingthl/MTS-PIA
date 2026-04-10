#!/usr/bin/env python
"""Phase 15 Step 1C: Closed-loop gamma feedback on multi-direction PIA (k_dir=5, s=1).

Per seed under one locked split:
- A: baseline
- C0: fixed-gamma multi-direction reference
- C_adapt: gamma feedback adaptation (gamma_max with per-sample feedback)

Per run supports multiple gamma_min variants. Features/split are computed once per seed.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.trial_dataset_factory import (
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
from manifold_raw.features import parse_band_spec
from run_phase14r_step6b1_rev2 import (
    apply_logcenter,
    covs_to_features,
    ensure_dir,
    extract_features_block,
    write_json,
)
from scripts.run_phase15_step1a_maxplane import (
    _apply_gates,
    _fit_eval_linearsvc,
    _fit_gate1_from_train,
    _make_trial_split,
)
from scripts.run_phase15_step1b_multidir_matrix import (
    _build_direction_bank_d1,
    _build_multidir_aug_candidates,
)


def _parse_seed_list(text: str) -> List[int]:
    out: List[int] = []
    for tok in str(text).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(int(t))
    out = sorted(set(out))
    if not out:
        raise ValueError("seed list cannot be empty")
    return out


def _parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for tok in str(text).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(float(t))
    out = sorted(set(out))
    if not out:
        raise ValueError("float list cannot be empty")
    return out


def _quantile_stats(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "min": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
        }
    return {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _feedback_adapt_candidates(
    X_src: np.ndarray,
    y_aug: np.ndarray,
    w_mix: np.ndarray,
    *,
    mu_y: Dict[int, np.ndarray],
    tau_y: Dict[int, float],
    gate2_q_src: float,
    gamma_max: float,
    gamma_min: float,
    power_p: float,
    eps: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """One-step closed-loop gamma adaptation."""
    X0 = np.asarray(X_src, dtype=np.float32)
    y = np.asarray(y_aug).astype(int).ravel()
    w = np.asarray(w_mix, dtype=np.float32)
    n = int(len(y))

    Z0 = (X0 + float(gamma_max) * w).astype(np.float32)

    d1_pre = np.zeros((n,), dtype=np.float64)
    tau1 = np.zeros((n,), dtype=np.float64)
    for cls in sorted(np.unique(y).tolist()):
        idx = np.where(y == cls)[0]
        mu = mu_y.get(int(cls))
        tau = tau_y.get(int(cls))
        if mu is None or tau is None:
            d1_pre[idx] = np.inf
            tau1[idx] = 0.0
            continue
        d1_pre[idx] = np.linalg.norm(Z0[idx] - np.asarray(mu, dtype=np.float32)[None, :], axis=1)
        tau1[idx] = float(tau)

    d2_pre = np.linalg.norm(Z0 - X0, axis=1).astype(np.float64)
    tau_src_y_pre: Dict[int, float] = {}
    for cls in sorted(np.unique(y).tolist()):
        ds = d2_pre[y == cls]
        if ds.size == 0:
            continue
        tau_src_y_pre[int(cls)] = float(np.percentile(ds, float(gate2_q_src)))

    tau2 = np.asarray([tau_src_y_pre.get(int(c), 0.0) for c in y.tolist()], dtype=np.float64)
    s1 = np.minimum(1.0, tau1 / (d1_pre + float(eps)))
    s2 = np.minimum(1.0, tau2 / (d2_pre + float(eps)))
    s = np.clip(s1 * s2, 0.0, 1.0)
    gamma = float(gamma_max) * np.power(s, float(power_p))
    gamma = np.clip(gamma, float(gamma_min), float(gamma_max)).astype(np.float32)

    Z = (X0 + gamma[:, None] * w).astype(np.float32)

    d1_post = np.zeros((n,), dtype=np.float64)
    for cls in sorted(np.unique(y).tolist()):
        idx = np.where(y == cls)[0]
        mu = mu_y.get(int(cls))
        if mu is None:
            d1_post[idx] = np.inf
            continue
        d1_post[idx] = np.linalg.norm(Z[idx] - np.asarray(mu, dtype=np.float32)[None, :], axis=1)
    d2_post = np.linalg.norm(Z - X0, axis=1).astype(np.float64)

    meta = {
        "gamma_stats": {
            "gamma_min_cfg": float(gamma_min),
            "gamma_max_cfg": float(gamma_max),
            "mean": float(np.mean(gamma)),
            "std": float(np.std(gamma)),
            "p05": float(np.percentile(gamma, 5)),
            "p50": float(np.percentile(gamma, 50)),
            "p95": float(np.percentile(gamma, 95)),
            "max": float(np.max(gamma)),
        },
        "distance_stats_pre": {
            "d1": _quantile_stats(d1_pre),
            "d2": _quantile_stats(d2_pre),
        },
        "distance_stats_post": {
            "d1": _quantile_stats(d1_post),
            "d2": _quantile_stats(d2_post),
        },
        "feedback_rule": {
            "p": float(power_p),
            "eps": float(eps),
            "gate2_q_src": float(gate2_q_src),
            "tau_src_y_pre": {str(k): float(v) for k, v in tau_src_y_pre.items()},
        },
    }
    return Z, meta


def _write_condition(cond_dir: str, metrics: Dict, run_meta: Dict) -> None:
    ensure_dir(cond_dir)
    write_json(os.path.join(cond_dir, "metrics.json"), metrics)
    write_json(os.path.join(cond_dir, "run_meta.json"), run_meta)


def _gmin_tag(v: float) -> str:
    return f"gmin{int(round(v * 1000)):03d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,3,4")
    parser.add_argument("--gamma-min-list", type=str, default="0.03,0.05")
    parser.add_argument("--gamma-max", type=float, default=0.15)
    parser.add_argument("--gamma-fixed-c0", type=float, default=0.10)
    parser.add_argument("--gamma-power-p", type=float, default=1.0)
    parser.add_argument("--feedback-eps", type=float, default=1e-6)
    parser.add_argument("--out-root", type=str, default="out/phase15_step1c_feedback")
    parser.add_argument("--dataset", type=str, default="seed1", choices=["seed1", "seed", "har", "mitbih", "seediv", "natops", "fingermovements"])
    parser.add_argument("--har-root", type=str, default=DEFAULT_HAR_ROOT)
    parser.add_argument("--mitbih-npz", type=str, default=DEFAULT_MITBIH_NPZ)
    parser.add_argument("--seediv-root", type=str, default=DEFAULT_SEEDIV_ROOT)
    parser.add_argument("--natops-root", type=str, default=DEFAULT_NATOPS_ROOT)
    parser.add_argument("--fingermovements-root", type=str, default=DEFAULT_FINGERMOVEMENTS_ROOT)
    parser.add_argument("--processed-root", type=str, default="data/SEED/SEED_EEG/Preprocessed_EEG")
    parser.add_argument("--stim-xlsx", type=str, default="data/SEED/SEED_EEG/SEED_stimulation.xlsx")

    parser.add_argument("--k-dir", type=int, default=5)
    parser.add_argument("--subset-size", type=int, default=1)
    parser.add_argument("--pia-multiplier", type=int, default=1)
    parser.add_argument("--pia-n-iters", type=int, default=2)
    parser.add_argument("--pia-activation", type=str, default="sine")
    parser.add_argument("--pia-bias-update-mode", type=str, default="residual")
    parser.add_argument("--pia-c-repr", type=float, default=1.0)

    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--hop-sec", type=float, default=1.0)
    parser.add_argument("--cov-est", type=str, default="sample", choices=["sample", "oas", "ledoitwolf"])
    parser.add_argument("--spd-eps", type=float, default=1e-4)
    parser.add_argument(
        "--bands",
        type=str,
        default=DEFAULT_BANDS_EEG,
    )

    parser.add_argument("--window-cap-k", type=int, default=120)
    parser.add_argument("--cap-sampling-policy", type=str, default="balanced_real_aug")
    parser.add_argument("--aggregation-mode", type=str, default="majority")
    parser.add_argument("--linear-c", type=float, default=1.0)
    parser.add_argument("--linear-class-weight", type=str, default="none")
    parser.add_argument("--linear-max-iter", type=int, default=1000)

    parser.add_argument("--gate1-q", type=float, default=95.0)
    parser.add_argument("--gate2-q-src", type=float, default=90.0)
    parser.add_argument("--split-preview-n", type=int, default=5)
    args = parser.parse_args()
    args.dataset = normalize_dataset_name(args.dataset)

    seeds = _parse_seed_list(args.seeds)
    gamma_min_list = _parse_float_list(args.gamma_min_list)
    gtags = [_gmin_tag(v) for v in gamma_min_list]

    if args.window_cap_k <= 0:
        raise ValueError("--window-cap-k must be > 0.")
    if args.k_dir <= 0:
        raise ValueError("--k-dir must be > 0.")
    if args.subset_size <= 0:
        raise ValueError("--subset-size must be > 0.")
    if not (0 < args.gate1_q <= 100):
        raise ValueError("--gate1-q must be in (0,100].")
    if not (0 < args.gate2_q_src <= 100):
        raise ValueError("--gate2-q-src must be in (0,100].")
    if not (0 < args.gamma_fixed_c0 <= args.gamma_max):
        raise ValueError("--gamma-fixed-c0 must be in (0, gamma-max].")

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
    if bands_spec != args.bands:
        print(f"[bands] auto override for {args.dataset}: {bands_spec}")
    bands = parse_band_spec(bands_spec)

    print(
        f"[setup] seeds={seeds} gamma_min={gamma_min_list} "
        f"k_dir={args.k_dir} subset_size={args.subset_size}"
    )

    all_rows: List[Dict[str, object]] = []
    protocol_issues: List[str] = []

    for seed in seeds:
        print(f"[seed={seed}] start")
        train_trials, test_trials, split_meta = _make_trial_split(all_trials, seed=int(seed))
        print(
            f"[seed={seed}] split train={len(train_trials)} test={len(test_trials)} "
            f"hash={split_meta['split_hash']}"
        )

        covs_train, y_train, tid_train = extract_features_block(
            train_trials, args.window_sec, args.hop_sec, args.cov_est, args.spd_eps, bands
        )
        covs_test, y_test, tid_test = extract_features_block(
            test_trials, args.window_sec, args.hop_sec, args.cov_est, args.spd_eps, bands
        )
        covs_train_lc, covs_test_lc = apply_logcenter(covs_train, covs_test, args.spd_eps)
        X_train_base = covs_to_features(covs_train_lc).astype(np.float32)
        X_test = covs_to_features(covs_test_lc).astype(np.float32)
        y_train_base = np.asarray(y_train).astype(int).ravel()
        y_test = np.asarray(y_test).astype(int).ravel()
        tid_train = np.asarray(tid_train)
        tid_test = np.asarray(tid_test)
        print(
            f"[seed={seed}] feat train_windows={len(y_train_base)} "
            f"test_windows={len(y_test)} dim={X_train_base.shape[1]}"
        )

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

        bank_seed = int(seed * 10000 + int(args.k_dir) * 113 + 17)
        direction_bank, bank_meta = _build_direction_bank_d1(
            X_train=X_train_base,
            k_dir=int(args.k_dir),
            seed=bank_seed,
            n_iters=int(args.pia_n_iters),
            activation=args.pia_activation,
            bias_update_mode=args.pia_bias_update_mode,
            c_repr=float(args.pia_c_repr),
        )
        X_mix, y_mix, tid_mix, src_mix, _, mix_meta = _build_multidir_aug_candidates(
            X_train=X_train_base,
            y_train=y_train_base,
            tid_train=tid_train,
            direction_bank=direction_bank,
            subset_size=int(args.subset_size),
            gamma=1.0,
            multiplier=int(args.pia_multiplier),
            seed=int(seed + 100000 + args.k_dir * 101 + args.subset_size * 7),
        )
        w_mix = (X_mix - src_mix).astype(np.float32)

        X_c0 = (src_mix + float(args.gamma_fixed_c0) * w_mix).astype(np.float32)
        X_c0_keep, y_c0_keep, tid_c0_keep, _, c0_gate_meta = _apply_gates(
            X_aug=X_c0,
            y_aug=y_mix,
            tid_aug=tid_mix,
            src_aug=src_mix,
            mu_y=mu_gate1,
            tau_y=tau_gate1,
            enable_gate2=True,
            gate2_q_src=float(args.gate2_q_src),
        )
        X_train_c0 = np.vstack([X_train_base, X_c0_keep]) if len(y_c0_keep) else X_train_base.copy()
        y_train_c0 = np.concatenate([y_train_base, y_c0_keep]) if len(y_c0_keep) else y_train_base.copy()
        tid_train_c0 = np.concatenate([tid_train, tid_c0_keep]) if len(y_c0_keep) else tid_train.copy()
        is_aug_c0 = (
            np.concatenate(
                [
                    np.zeros((len(y_train_base),), dtype=bool),
                    np.ones((len(y_c0_keep),), dtype=bool),
                ]
            )
            if len(y_c0_keep)
            else np.zeros((len(y_train_base),), dtype=bool)
        )
        metrics_c0, train_meta_c0 = _fit_eval_linearsvc(
            X_train_c0,
            y_train_c0,
            tid_train_c0,
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
            is_aug_train=is_aug_c0,
        )
        print(f"[seed={seed}] C0 pre={len(y_mix)} post_gate={len(y_c0_keep)}")

        common_meta = {
            "seed": int(seed),
            "split_hash": split_meta["split_hash"],
            "train_count_trials": int(split_meta["train_count_trials"]),
            "test_count_trials": int(split_meta["test_count_trials"]),
            "train_trial_ids_preview": split_meta["train_trial_ids"][: max(0, int(args.split_preview_n))],
            "test_trial_ids_preview": split_meta["test_trial_ids"][: max(0, int(args.split_preview_n))],
            "window_cap_K": int(args.window_cap_k),
            "cap_sampling_policy": args.cap_sampling_policy,
            "feature_pipeline": {
                "window_sec": float(args.window_sec),
                "hop_sec": float(args.hop_sec),
                "cov_est": args.cov_est,
                "spd_eps": float(args.spd_eps),
                "center": "logcenter_train_only",
                "vectorize": "upper_triangle",
                "bands": bands_spec,
            },
            "dataset": args.dataset,
            "aggregation_mode": args.aggregation_mode,
            "test_augmentation": "disabled",
            "direction_bank": {
                **bank_meta,
                "subset_size": int(args.subset_size),
            },
        }

        for gmin, gtag in zip(gamma_min_list, gtags):
            X_adapt, adapt_meta = _feedback_adapt_candidates(
                X_src=src_mix,
                y_aug=y_mix,
                w_mix=w_mix,
                mu_y=mu_gate1,
                tau_y=tau_gate1,
                gate2_q_src=float(args.gate2_q_src),
                gamma_max=float(args.gamma_max),
                gamma_min=float(gmin),
                power_p=float(args.gamma_power_p),
                eps=float(args.feedback_eps),
            )
            X_adapt_keep, y_adapt_keep, tid_adapt_keep, _, adapt_gate_meta = _apply_gates(
                X_aug=X_adapt,
                y_aug=y_mix,
                tid_aug=tid_mix,
                src_aug=src_mix,
                mu_y=mu_gate1,
                tau_y=tau_gate1,
                enable_gate2=True,
                gate2_q_src=float(args.gate2_q_src),
            )
            X_train_adapt = np.vstack([X_train_base, X_adapt_keep]) if len(y_adapt_keep) else X_train_base.copy()
            y_train_adapt = np.concatenate([y_train_base, y_adapt_keep]) if len(y_adapt_keep) else y_train_base.copy()
            tid_train_adapt = np.concatenate([tid_train, tid_adapt_keep]) if len(y_adapt_keep) else tid_train.copy()
            is_aug_adapt = (
                np.concatenate(
                    [
                        np.zeros((len(y_train_base),), dtype=bool),
                        np.ones((len(y_adapt_keep),), dtype=bool),
                    ]
                )
                if len(y_adapt_keep)
                else np.zeros((len(y_train_base),), dtype=bool)
            )
            metrics_adapt, train_meta_adapt = _fit_eval_linearsvc(
                X_train_adapt,
                y_train_adapt,
                tid_train_adapt,
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
                is_aug_train=is_aug_adapt,
            )

            run_dir = os.path.join(args.out_root, gtag, f"seed{seed}")
            cond_dirs = {
                "A_baseline": os.path.join(run_dir, "A_baseline"),
                "C0_multidir_fixed": os.path.join(run_dir, "C0_multidir_fixed"),
                "C_adapt_feedback": os.path.join(run_dir, "C_adapt_feedback"),
            }
            _write_condition(
                cond_dirs["A_baseline"],
                metrics_a,
                {
                    **common_meta,
                    **train_meta_a,
                    "condition": "A_baseline",
                    "gamma_min_setting": float(gmin),
                },
            )
            _write_condition(
                cond_dirs["C0_multidir_fixed"],
                metrics_c0,
                {
                    **common_meta,
                    **train_meta_c0,
                    "condition": "C0_multidir_fixed",
                    "gamma_min_setting": float(gmin),
                    "augmentation": {
                        **mix_meta,
                        "gamma_fixed_c0": float(args.gamma_fixed_c0),
                    },
                    "gate1_fit": gate1_fit_meta,
                    "gate_apply": c0_gate_meta,
                    "gate2_config": {"enabled": True, "q_src": float(args.gate2_q_src)},
                    "final_accept_rate": float(c0_gate_meta["accept_rate_final"]),
                },
            )
            _write_condition(
                cond_dirs["C_adapt_feedback"],
                metrics_adapt,
                {
                    **common_meta,
                    **train_meta_adapt,
                    "condition": "C_adapt_feedback",
                    "gamma_min_setting": float(gmin),
                    "augmentation": mix_meta,
                    "gamma_feedback": adapt_meta,
                    "gamma_stats": adapt_meta["gamma_stats"],
                    "distance_stats_pre": adapt_meta["distance_stats_pre"],
                    "distance_stats_post": adapt_meta["distance_stats_post"],
                    "gate1_fit": gate1_fit_meta,
                    "gate_apply": adapt_gate_meta,
                    "gate2_config": {"enabled": True, "q_src": float(args.gate2_q_src)},
                    "final_accept_rate": float(adapt_gate_meta["accept_rate_final"]),
                },
            )

            paired_rows = [
                {
                    "condition": "A_baseline",
                    "acc": metrics_a["trial_acc"],
                    "macro_f1": metrics_a["trial_macro_f1"],
                    "accept_rate": 1.0,
                    "split_hash": split_meta["split_hash"],
                    "gamma_min": float(gmin),
                },
                {
                    "condition": "C0_multidir_fixed",
                    "acc": metrics_c0["trial_acc"],
                    "macro_f1": metrics_c0["trial_macro_f1"],
                    "accept_rate": c0_gate_meta["accept_rate_final"],
                    "split_hash": split_meta["split_hash"],
                    "gamma_min": float(gmin),
                },
                {
                    "condition": "C_adapt_feedback",
                    "acc": metrics_adapt["trial_acc"],
                    "macro_f1": metrics_adapt["trial_macro_f1"],
                    "accept_rate": adapt_gate_meta["accept_rate_final"],
                    "split_hash": split_meta["split_hash"],
                    "gamma_min": float(gmin),
                },
            ]
            paired_csv = os.path.join(args.out_root, gtag, f"phase15_step1c_seed{seed}_paired.csv")
            ensure_dir(os.path.dirname(paired_csv))
            pd.DataFrame(paired_rows).to_csv(paired_csv, index=False)

            all_rows.append(
                {
                    "seed": int(seed),
                    "gamma_min": float(gmin),
                    "gamma_tag": gtag,
                    "split_hash": split_meta["split_hash"],
                    "A_f1": metrics_a["trial_macro_f1"],
                    "C0_f1": metrics_c0["trial_macro_f1"],
                    "C_adapt_f1": metrics_adapt["trial_macro_f1"],
                    "A_acc": metrics_a["trial_acc"],
                    "C0_acc": metrics_c0["trial_acc"],
                    "C_adapt_acc": metrics_adapt["trial_acc"],
                    "delta_Cadapt_minus_C0_f1": metrics_adapt["trial_macro_f1"] - metrics_c0["trial_macro_f1"],
                    "delta_Cadapt_minus_A_f1": metrics_adapt["trial_macro_f1"] - metrics_a["trial_macro_f1"],
                    "delta_Cadapt_minus_C0_acc": metrics_adapt["trial_acc"] - metrics_c0["trial_acc"],
                    "delta_Cadapt_minus_A_acc": metrics_adapt["trial_acc"] - metrics_a["trial_acc"],
                    "accept_rate_final": adapt_gate_meta["accept_rate_final"],
                    "gamma_mean": adapt_meta["gamma_stats"]["mean"],
                    "gamma_std": adapt_meta["gamma_stats"]["std"],
                    "gamma_p95": adapt_meta["gamma_stats"]["p95"],
                }
            )

            print(
                f"[seed={seed}][{gtag}] A={metrics_a['trial_macro_f1']:.4f} "
                f"C0={metrics_c0['trial_macro_f1']:.4f} C_adapt={metrics_adapt['trial_macro_f1']:.4f} "
                f"C_adapt-C0={metrics_adapt['trial_macro_f1']-metrics_c0['trial_macro_f1']:+.4f} "
                f"accept={adapt_gate_meta['accept_rate_final']:.3f} "
                f"gamma_mean={adapt_meta['gamma_stats']['mean']:.3f}"
            )

    summary_df = pd.DataFrame(all_rows).sort_values(["gamma_min", "seed"]).reset_index(drop=True)
    summary_csv = os.path.join(args.out_root, "summary", "phase15_step1c_summary.csv")
    ensure_dir(os.path.dirname(summary_csv))
    summary_df.to_csv(summary_csv, index=False)

    agg_rows: List[Dict[str, object]] = []
    if not summary_df.empty:
        for gmin in sorted(summary_df["gamma_min"].unique().tolist()):
            d = summary_df[summary_df["gamma_min"] == gmin]
            for metric in [
                "A_f1",
                "C0_f1",
                "C_adapt_f1",
                "A_acc",
                "C0_acc",
                "C_adapt_acc",
                "delta_Cadapt_minus_C0_f1",
                "delta_Cadapt_minus_A_f1",
                "delta_Cadapt_minus_C0_acc",
                "delta_Cadapt_minus_A_acc",
                "accept_rate_final",
                "gamma_mean",
                "gamma_std",
                "gamma_p95",
            ]:
                arr = d[metric].to_numpy(dtype=float)
                agg_rows.append(
                    {
                        "gamma_min": float(gmin),
                        "metric": metric,
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr, ddof=1) if arr.size > 1 else 0.0),
                    }
                )
    agg_df = pd.DataFrame(agg_rows)
    agg_csv = os.path.join(args.out_root, "summary", "phase15_step1c_agg.csv")
    agg_df.to_csv(agg_csv, index=False)

    report = {
        "seeds": seeds,
        "gamma_min_list": gamma_min_list,
        "summary_csv": summary_csv,
        "agg_csv": agg_csv,
        "protocol_issues": protocol_issues,
    }
    report_path = os.path.join(args.out_root, "summary", "phase15_step1c_report.json")
    write_json(report_path, report)
    print(f"[done] summary_csv={summary_csv}")
    print(f"[done] agg_csv={agg_csv}")
    print(f"[done] report={report_path}")


if __name__ == "__main__":
    main()
