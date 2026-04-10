#!/usr/bin/env python
"""Phase 15 Step 1A-rev2 matrix runner.

Per seed:
- extract features once
- run multiple variants (V1..V4 by default)

Variants:
- V1: gate2 q95 + sigma S1
- V2: gate2 q90 + sigma S1
- V3: gate2 q95 + sigma S2 (target gate1 accept)
- V4: gate2 q90 + sigma S2 (target gate1 accept)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
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
    PiaAugConfig,
    _build_pia_aug_candidates,
    _build_plane_aug_candidates,
    _fit_eval_linearsvc,
    _fit_gate1_from_train,
    _fit_max_plane,
    _make_trial_split,
    _project_to_max_plane,
    _apply_gates,
)


@dataclass(frozen=True)
class Variant:
    name: str
    sigma_mode: str
    q_src: float


def _parse_seed_list(text: str) -> List[int]:
    out: List[int] = []
    for tok in str(text).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(int(t))
    out = sorted(set(out))
    if not out:
        raise ValueError("seed list is empty")
    return out


def _variant_catalog() -> Dict[str, Variant]:
    return {
        "V1": Variant(name="V1", sigma_mode="s1", q_src=95.0),
        "V2": Variant(name="V2", sigma_mode="s1", q_src=90.0),
        "V3": Variant(name="V3", sigma_mode="s2", q_src=95.0),
        "V4": Variant(name="V4", sigma_mode="s2", q_src=90.0),
    }


def _parse_variants(text: str) -> List[Variant]:
    cat = _variant_catalog()
    names = [t.strip() for t in str(text).split(",") if t.strip()]
    if not names:
        names = ["V1", "V2", "V3", "V4"]
    out: List[Variant] = []
    for n in names:
        if n not in cat:
            raise ValueError(f"Unknown variant: {n}")
        out.append(cat[n])
    return out


def _write_condition(
    cond_dir: str,
    metrics: Dict,
    run_meta: Dict,
) -> None:
    ensure_dir(cond_dir)
    write_json(os.path.join(cond_dir, "metrics.json"), metrics)
    write_json(os.path.join(cond_dir, "run_meta.json"), run_meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,4")
    parser.add_argument("--variants", type=str, default="V1,V2,V3,V4")
    parser.add_argument("--out-root", type=str, default="out/phase15_step1a_rev2")
    parser.add_argument("--dataset", type=str, default="seed1", choices=["seed1", "seed", "har", "mitbih", "seediv", "natops", "fingermovements"])
    parser.add_argument("--har-root", type=str, default=DEFAULT_HAR_ROOT)
    parser.add_argument("--mitbih-npz", type=str, default=DEFAULT_MITBIH_NPZ)
    parser.add_argument("--seediv-root", type=str, default=DEFAULT_SEEDIV_ROOT)
    parser.add_argument("--natops-root", type=str, default=DEFAULT_NATOPS_ROOT)
    parser.add_argument("--fingermovements-root", type=str, default=DEFAULT_FINGERMOVEMENTS_ROOT)
    parser.add_argument("--processed-root", type=str, default="data/SEED/SEED_EEG/Preprocessed_EEG")
    parser.add_argument("--stim-xlsx", type=str, default="data/SEED/SEED_EEG/SEED_stimulation.xlsx")

    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--sigma-target-accept", type=float, default=0.95)
    parser.add_argument("--plane-multiplier", type=int, default=1)

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

    parser.add_argument("--pia-multiplier", type=int, default=1)
    parser.add_argument("--pia-gamma", type=float, default=0.10)
    parser.add_argument("--pia-gamma-jitter", type=float, default=0.0)
    parser.add_argument("--pia-n-iters", type=int, default=2)
    parser.add_argument("--pia-activation", type=str, default="sine")
    parser.add_argument("--pia-bias-update-mode", type=str, default="residual")
    parser.add_argument("--pia-c-repr", type=float, default=1.0)
    parser.add_argument("--gate1-q", type=float, default=95.0)
    parser.add_argument("--split-preview-n", type=int, default=5)
    args = parser.parse_args()
    args.dataset = normalize_dataset_name(args.dataset)

    seeds = _parse_seed_list(args.seeds)
    variants = _parse_variants(args.variants)

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

    print(f"[setup] seeds={seeds} variants={[v.name for v in variants]} k={args.k}")
    print(f"[setup] processed_root={args.processed_root}")

    all_rows: List[Dict] = []
    protocol_issues: List[str] = []

    for seed in seeds:
        print(f"[seed={seed}] start")
        train_trials, test_trials, split_meta = _make_trial_split(all_trials, seed=seed)
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
            f"[seed={seed}] feat train_windows={len(y_train_base)} test_windows={len(y_test)} "
            f"dim={X_train_base.shape[1]}"
        )

        mu_gate1, tau_gate1, gate1_fit_meta = _fit_gate1_from_train(
            X_train=X_train_base,
            y_train=y_train_base,
            q=float(args.gate1_q),
        )
        pia_cfg = PiaAugConfig(
            multiplier=int(args.pia_multiplier),
            gamma=float(args.pia_gamma),
            gamma_jitter=float(args.pia_gamma_jitter),
            n_iters=int(args.pia_n_iters),
            activation=args.pia_activation,
            bias_update_mode=args.pia_bias_update_mode,
            C_repr=float(args.pia_c_repr),
            seed=int(seed),
        )
        X_pia, y_pia, tid_pia, src_pia, pia_meta = _build_pia_aug_candidates(
            X_train=X_train_base,
            y_train=y_train_base,
            tid_train=tid_train,
            cfg=pia_cfg,
        )
        print(f"[seed={seed}] pia_candidates={len(y_pia)}")

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
        }
        cap_seed = int(seed) + 41

        # Baseline A once per seed (reused by all variants)
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

        for v in variants:
            print(f"[seed={seed}] variant={v.name} sigma={v.sigma_mode} q_src={v.q_src}")
            # C0: PIA + Gate1 + Gate2
            X_c0_keep, y_c0_keep, tid_c0_keep, _, c0_gate_meta = _apply_gates(
                X_aug=X_pia,
                y_aug=y_pia,
                tid_aug=tid_pia,
                src_aug=src_pia,
                mu_y=mu_gate1,
                tau_y=tau_gate1,
                enable_gate2=True,
                gate2_q_src=float(v.q_src),
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

            # Build max-plane for this sigma mode
            model, plane_meta = _fit_max_plane(
                X_train=X_train_base,
                y_train=y_train_base,
                k=int(args.k),
                sigma_mode=v.sigma_mode,
                gate_tau_y=tau_gate1,
                sigma_target_accept=float(args.sigma_target_accept),
                seed=int(seed),
            )

            # C1
            X_c1, y_c1, tid_c1, src_c1, c1_aug_meta = _build_plane_aug_candidates(
                X_train=X_train_base,
                y_train=y_train_base,
                tid_train=tid_train,
                model=model,
                multiplier=int(args.plane_multiplier),
                seed=int(seed) + 1701,
            )
            X_c1_keep, y_c1_keep, tid_c1_keep, _, c1_gate_meta = _apply_gates(
                X_aug=X_c1,
                y_aug=y_c1,
                tid_aug=tid_c1,
                src_aug=src_c1,
                mu_y=mu_gate1,
                tau_y=tau_gate1,
                enable_gate2=True,
                gate2_q_src=float(v.q_src),
            )
            X_train_c1 = np.vstack([X_train_base, X_c1_keep]) if len(y_c1_keep) else X_train_base.copy()
            y_train_c1 = np.concatenate([y_train_base, y_c1_keep]) if len(y_c1_keep) else y_train_base.copy()
            tid_train_c1 = np.concatenate([tid_train, tid_c1_keep]) if len(y_c1_keep) else tid_train.copy()
            is_aug_c1 = (
                np.concatenate(
                    [
                        np.zeros((len(y_train_base),), dtype=bool),
                        np.ones((len(y_c1_keep),), dtype=bool),
                    ]
                )
                if len(y_c1_keep)
                else np.zeros((len(y_train_base),), dtype=bool)
            )
            metrics_c1, train_meta_c1 = _fit_eval_linearsvc(
                X_train_c1,
                y_train_c1,
                tid_train_c1,
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
                is_aug_train=is_aug_c1,
            )

            # C2
            X_pia_fb, c2_fb_meta = _project_to_max_plane(X_pia, y_pia, model)
            X_c2_keep, y_c2_keep, tid_c2_keep, _, c2_gate_meta = _apply_gates(
                X_aug=X_pia_fb,
                y_aug=y_pia,
                tid_aug=tid_pia,
                src_aug=src_pia,
                mu_y=mu_gate1,
                tau_y=tau_gate1,
                enable_gate2=True,
                gate2_q_src=float(v.q_src),
            )
            X_train_c2 = np.vstack([X_train_base, X_c2_keep]) if len(y_c2_keep) else X_train_base.copy()
            y_train_c2 = np.concatenate([y_train_base, y_c2_keep]) if len(y_c2_keep) else y_train_base.copy()
            tid_train_c2 = np.concatenate([tid_train, tid_c2_keep]) if len(y_c2_keep) else tid_train.copy()
            is_aug_c2 = (
                np.concatenate(
                    [
                        np.zeros((len(y_train_base),), dtype=bool),
                        np.ones((len(y_c2_keep),), dtype=bool),
                    ]
                )
                if len(y_c2_keep)
                else np.zeros((len(y_train_base),), dtype=bool)
            )
            metrics_c2, train_meta_c2 = _fit_eval_linearsvc(
                X_train_c2,
                y_train_c2,
                tid_train_c2,
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
                is_aug_train=is_aug_c2,
            )

            # Persist
            variant_dir = os.path.join(args.out_root, v.name, f"k{args.k}", f"seed{seed}")
            cond_dirs = {
                "A_baseline": os.path.join(variant_dir, "A_baseline"),
                "C0_pia_gate": os.path.join(variant_dir, "C0_pia_gate"),
                "C1_plane_gate": os.path.join(variant_dir, "C1_plane_gate"),
                "C2_pia_plane_gate": os.path.join(variant_dir, "C2_pia_plane_gate"),
            }
            _write_condition(
                cond_dirs["A_baseline"],
                metrics_a,
                {
                    **common_meta,
                    **train_meta_a,
                    "condition": "A_baseline",
                    "k": int(args.k),
                    "variant": v.name,
                },
            )
            _write_condition(
                cond_dirs["C0_pia_gate"],
                metrics_c0,
                {
                    **common_meta,
                    **train_meta_c0,
                    "condition": "C0_pia_gate",
                    "k": int(args.k),
                    "variant": v.name,
                    "augmentation": pia_meta,
                    "gate1_fit": gate1_fit_meta,
                    "gate_apply": c0_gate_meta,
                    "sigma_calibration": {
                        "mode": v.sigma_mode,
                        "target": float(args.sigma_target_accept),
                        "target_type": "gate1_accept",
                    },
                    "gate2_config": {
                        "enabled": True,
                        "q_src": float(v.q_src),
                    },
                },
            )
            _write_condition(
                cond_dirs["C1_plane_gate"],
                metrics_c1,
                {
                    **common_meta,
                    **train_meta_c1,
                    "condition": "C1_plane_gate",
                    "k": int(args.k),
                    "variant": v.name,
                    "max_plane": plane_meta,
                    "augmentation": c1_aug_meta,
                    "gate1_fit": gate1_fit_meta,
                    "gate_apply": c1_gate_meta,
                    "sigma_calibration": {
                        "mode": v.sigma_mode,
                        "target": float(args.sigma_target_accept),
                        "target_type": "gate1_accept",
                    },
                    "gate2_config": {
                        "enabled": True,
                        "q_src": float(v.q_src),
                    },
                },
            )
            _write_condition(
                cond_dirs["C2_pia_plane_gate"],
                metrics_c2,
                {
                    **common_meta,
                    **train_meta_c2,
                    "condition": "C2_pia_plane_gate",
                    "k": int(args.k),
                    "variant": v.name,
                    "max_plane": plane_meta,
                    "augmentation": {
                        **pia_meta,
                        "feedback": "pia_project_to_max_plane",
                        **c2_fb_meta,
                    },
                    "gate1_fit": gate1_fit_meta,
                    "gate_apply": c2_gate_meta,
                    "sigma_calibration": {
                        "mode": v.sigma_mode,
                        "target": float(args.sigma_target_accept),
                        "target_type": "gate1_accept",
                    },
                    "gate2_config": {
                        "enabled": True,
                        "q_src": float(v.q_src),
                    },
                },
            )

            paired_rows = [
                {
                    "condition": "A_baseline",
                    "acc": metrics_a["trial_acc"],
                    "macro_f1": metrics_a["trial_macro_f1"],
                    "accept_rate": 1.0,
                    "split_hash": split_meta["split_hash"],
                    "k": int(args.k),
                    "variant": v.name,
                },
                {
                    "condition": "C0_pia_gate",
                    "acc": metrics_c0["trial_acc"],
                    "macro_f1": metrics_c0["trial_macro_f1"],
                    "accept_rate": c0_gate_meta["accept_rate_final"],
                    "split_hash": split_meta["split_hash"],
                    "k": int(args.k),
                    "variant": v.name,
                },
                {
                    "condition": "C1_plane_gate",
                    "acc": metrics_c1["trial_acc"],
                    "macro_f1": metrics_c1["trial_macro_f1"],
                    "accept_rate": c1_gate_meta["accept_rate_final"],
                    "split_hash": split_meta["split_hash"],
                    "k": int(args.k),
                    "variant": v.name,
                },
                {
                    "condition": "C2_pia_plane_gate",
                    "acc": metrics_c2["trial_acc"],
                    "macro_f1": metrics_c2["trial_macro_f1"],
                    "accept_rate": c2_gate_meta["accept_rate_final"],
                    "split_hash": split_meta["split_hash"],
                    "k": int(args.k),
                    "variant": v.name,
                },
            ]
            paired_csv = os.path.join(args.out_root, v.name, f"k{args.k}", f"phase15_step1a_seed{seed}_paired.csv")
            ensure_dir(os.path.dirname(paired_csv))
            pd.DataFrame(paired_rows).to_csv(paired_csv, index=False)

            all_rows.append(
                {
                    "seed": int(seed),
                    "variant": v.name,
                    "split_hash": split_meta["split_hash"],
                    "A_f1": metrics_a["trial_macro_f1"],
                    "C0_f1": metrics_c0["trial_macro_f1"],
                    "C1_f1": metrics_c1["trial_macro_f1"],
                    "C2_f1": metrics_c2["trial_macro_f1"],
                    "C2_minus_C0_f1": metrics_c2["trial_macro_f1"] - metrics_c0["trial_macro_f1"],
                    "C2_minus_A_f1": metrics_c2["trial_macro_f1"] - metrics_a["trial_macro_f1"],
                    "C1_accept_gate1": c1_gate_meta["accept_rate_gate1"],
                    "C1_accept_final": c1_gate_meta["accept_rate_final"],
                    "C2_accept_gate1": c2_gate_meta["accept_rate_gate1"],
                    "C2_accept_final": c2_gate_meta["accept_rate_final"],
                    "q_src": float(v.q_src),
                    "sigma_mode": v.sigma_mode,
                }
            )
            print(
                f"[seed={seed}][{v.name}] C0={metrics_c0['trial_macro_f1']:.4f} "
                f"C1={metrics_c1['trial_macro_f1']:.4f} C2={metrics_c2['trial_macro_f1']:.4f} "
                f"C2-C0={metrics_c2['trial_macro_f1']-metrics_c0['trial_macro_f1']:+.4f} "
                f"acc_final(C1/C2)={c1_gate_meta['accept_rate_final']:.3f}/{c2_gate_meta['accept_rate_final']:.3f}"
            )

    summary_df = pd.DataFrame(all_rows).sort_values(["variant", "seed"]).reset_index(drop=True)
    summary_csv = os.path.join(args.out_root, f"k{args.k}", "rev2_stoploss_summary.csv")
    ensure_dir(os.path.dirname(summary_csv))
    summary_df.to_csv(summary_csv, index=False)

    agg_rows = []
    for v in sorted(summary_df["variant"].unique().tolist()):
        d = summary_df[summary_df["variant"] == v]
        for metric in [
            "A_f1",
            "C0_f1",
            "C1_f1",
            "C2_f1",
            "C2_minus_C0_f1",
            "C2_minus_A_f1",
            "C1_accept_final",
            "C2_accept_final",
        ]:
            arr = d[metric].to_numpy(dtype=float)
            agg_rows.append(
                {
                    "variant": v,
                    "metric": metric,
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1) if arr.size > 1 else 0.0),
                }
            )
    agg_df = pd.DataFrame(agg_rows)
    agg_csv = os.path.join(args.out_root, f"k{args.k}", "rev2_stoploss_agg.csv")
    agg_df.to_csv(agg_csv, index=False)

    report = {
        "seeds": seeds,
        "variants": [v.name for v in variants],
        "k": int(args.k),
        "summary_csv": summary_csv,
        "agg_csv": agg_csv,
        "protocol_issues": protocol_issues,
    }
    report_path = os.path.join(args.out_root, f"k{args.k}", "rev2_stoploss_report.json")
    write_json(report_path, report)
    print(f"[done] summary_csv={summary_csv}")
    print(f"[done] agg_csv={agg_csv}")
    print(f"[done] report={report_path}")


if __name__ == "__main__":
    main()
