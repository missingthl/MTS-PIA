#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "scripts"))

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
    DEFAULT_SELFREGULATIONSCP1_ROOT,
    DEFAULT_UWAVEGESTURELIBRARY_ROOT,
    load_trials_for_dataset,
)
from run_bridge_curriculum_pilot import (  # noqa: E402
    _bridge_aug_trials,
    _dataset_title,
    _ensure_dir,
    _fit_raw_minirocket,
    _records_to_trial_dicts,
    _risk_comment,
    _select_best_curriculum_round,
    _write_json,
    _format_mean_std,
)
from run_phase15_step0a_paired_lock import _make_trial_split  # noqa: E402
from run_phase15_step1a_maxplane import _fit_eval_linearsvc  # noqa: E402
from run_phase15_step1b_multidir_matrix import (  # noqa: E402
    _build_direction_bank_d1,
    _build_multidir_aug_candidates,
    _compute_mech_metrics,
)
from run_raw_bridge_probe import TrialRecord, _apply_mean_log, _build_trial_records  # noqa: E402
from scripts.fisher_pia_utils import FisherPIAConfig  # noqa: E402
from scripts.lraes_utils import LRAESConfig, build_lraes_direction_bank  # noqa: E402
from scripts.run_phase15_mainline_freeze import _summarize_dir_profile  # noqa: E402
from scripts.run_phase15_multiround_curriculum_probe import (  # noqa: E402
    _active_direction_probs,
    _build_curriculum_aug_candidates,
    _compute_direction_intrusion,
    _mech_dir_maps,
    _update_direction_budget,
)


def _parse_seed_list(text: str) -> List[int]:
    out = sorted(set(int(tok.strip()) for tok in str(text).split(",") if tok.strip()))
    if not out:
        raise ValueError("seed list cannot be empty")
    return out


def _output_file_map(dataset: str) -> Dict[str, str]:
    key = str(dataset).strip().lower()
    if key == "selfregulationscp1":
        stem = "bridge_lraes_scp1"
    elif key == "natops":
        stem = "bridge_lraes_natops"
    elif key == "fingermovements":
        stem = "bridge_lraes_fm"
    elif key == "har":
        stem = "bridge_lraes_har"
    elif key == "mitbih":
        stem = "bridge_lraes_mitbih"
    else:
        stem = f"bridge_lraes_{key}"
    return {
        "per_seed": f"{stem}_pilot_per_seed.csv",
        "summary": f"{stem}_pilot_summary.csv",
        "target_health": f"{stem}_target_health_summary.csv",
        "fidelity": f"{stem}_fidelity_summary.csv",
        "solver": f"{stem}_solver_summary.csv",
        "conclusion": f"{stem}_pilot_conclusion.md",
    }


def _direction_usage_entropy_from_aug_meta(aug_meta: Dict[str, object]) -> float:
    val = aug_meta.get("direction_usage_entropy", 0.0)
    return float(val) if val is not None else 0.0


def _target_health_comment(variant: str, dir_summary: Dict[str, object], entropy: float, ref_margin: float | None) -> str:
    worst_margin = dir_summary.get("worst_dir_margin_drop")
    if variant == "bridge_single_round":
        return "single_round_equal_weight_reference"
    if worst_margin is None:
        return f"{variant}_missing_dir_summary"
    if ref_margin is not None and float(worst_margin) >= float(ref_margin) - 1e-12:
        if entropy < 1.2:
            return f"{variant}_cleaner_with_direction_focus"
        return f"{variant}_cleaner_than_single_round"
    return f"{variant}_not_cleaner_than_single_round"


def _solver_comment(state_counts: Dict[str, int], low_quality_axis_count: int) -> str:
    if int(state_counts.get("fully_risk_dominated", 0)) > 0:
        return "contains_fully_risk_dominated_local_regions"
    if int(low_quality_axis_count) > 0:
        return "mixed_expandable_and_low_quality_axes"
    return "no_clear_fully_risk_dominated_signal"


def _result_label(delta_vs_multi: float, fidelity_not_worse: bool) -> str:
    if delta_vs_multi > 1e-6 and fidelity_not_worse:
        return "positive"
    if delta_vs_multi >= -1e-6 and fidelity_not_worse:
        return "flat"
    return "negative"


def main() -> None:
    p = argparse.ArgumentParser(description="LRAES + curriculum -> bridge -> raw MiniROCKET pilot.")
    p.add_argument(
        "--dataset",
        type=str,
        default="selfregulationscp1",
        choices=[
            "har",
            "mitbih",
            "natops",
            "selfregulationscp1",
            "fingermovements",
            "basicmotions",
            "handmovementdirection",
            "uwavegesturelibrary",
            "epilepsy",
            "atrialfibrillation",
            "pendigits",
        ],
    )
    p.add_argument("--har-root", type=str, default=DEFAULT_HAR_ROOT)
    p.add_argument("--mitbih-npz", type=str, default=DEFAULT_MITBIH_NPZ)
    p.add_argument("--selfregulationscp1-root", type=str, default=DEFAULT_SELFREGULATIONSCP1_ROOT)
    p.add_argument("--natops-root", type=str, default=DEFAULT_NATOPS_ROOT)
    p.add_argument("--fingermovements-root", type=str, default=DEFAULT_FINGERMOVEMENTS_ROOT)
    p.add_argument("--basicmotions-root", type=str, default=DEFAULT_BASICMOTIONS_ROOT)
    p.add_argument("--handmovementdirection-root", type=str, default=DEFAULT_HANDMOVEMENTDIRECTION_ROOT)
    p.add_argument("--uwavegesturelibrary-root", type=str, default=DEFAULT_UWAVEGESTURELIBRARY_ROOT)
    p.add_argument("--epilepsy-root", type=str, default=DEFAULT_EPILEPSY_ROOT)
    p.add_argument("--atrialfibrillation-root", type=str, default=DEFAULT_ATRIALFIBRILLATION_ROOT)
    p.add_argument("--pendigits-root", type=str, default=DEFAULT_PENDIGITS_ROOT)
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/bridge_lraes_scp1_pilot_20260322")
    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--bridge-eps", type=float, default=1e-4)
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
    p.add_argument("--memmap-threshold-gb", type=float, default=1.0)
    p.add_argument("--k-dir", type=int, default=5)
    p.add_argument("--subset-size", type=int, default=1)
    p.add_argument("--pia-multiplier", type=int, default=1)
    p.add_argument("--pia-gamma", type=float, default=0.10)
    p.add_argument("--pia-n-iters", type=int, default=2)
    p.add_argument("--pia-activation", type=str, default="sine")
    p.add_argument("--pia-bias-update-mode", type=str, default="residual")
    p.add_argument("--pia-c-repr", type=float, default=1.0)
    p.add_argument("--mech-knn-k", type=int, default=20)
    p.add_argument("--mech-max-aug-for-metrics", type=int, default=2000)
    p.add_argument("--mech-max-real-knn-ref", type=int, default=10000)
    p.add_argument("--mech-max-real-knn-query", type=int, default=1000)
    p.add_argument("--linear-c", type=float, default=1.0)
    p.add_argument("--linear-class-weight", type=str, default="none")
    p.add_argument("--linear-max-iter", type=int, default=1000)
    p.add_argument("--curriculum-rounds", type=int, default=3)
    p.add_argument("--curriculum-init-gamma", type=float, default=0.06)
    p.add_argument("--curriculum-expand-factor", type=float, default=1.25)
    p.add_argument("--curriculum-shrink-factor", type=float, default=0.70)
    p.add_argument("--curriculum-gamma-max", type=float, default=0.16)
    p.add_argument("--curriculum-freeze-eps", type=float, default=0.02)
    p.add_argument("--lraes-beta", type=float, default=0.5)
    p.add_argument("--lraes-reg-lambda", type=float, default=1e-4)
    p.add_argument("--lraes-top-k-per-class", type=int, default=3)
    p.add_argument("--lraes-rank-tol", type=float, default=1e-8)
    p.add_argument("--lraes-eig-pos-eps", type=float, default=1e-9)
    p.add_argument("--lraes-knn-k", type=int, default=20)
    p.add_argument("--lraes-boundary-quantile", type=float, default=0.30)
    p.add_argument("--lraes-interior-quantile", type=float, default=0.70)
    p.add_argument("--lraes-hetero-k", type=int, default=3)
    args = p.parse_args()

    dataset_name = str(args.dataset).strip().lower()
    dataset_title = _dataset_title(dataset_name)
    output_files = _output_file_map(dataset_name)
    seeds = _parse_seed_list(args.seeds)

    _ensure_dir(args.out_root)
    all_trials = load_trials_for_dataset(
        dataset=dataset_name,
        har_root=args.har_root,
        mitbih_npz=args.mitbih_npz,
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

    fisher_cfg = FisherPIAConfig(
        knn_k=int(args.lraes_knn_k),
        interior_quantile=float(args.lraes_interior_quantile),
        boundary_quantile=float(args.lraes_boundary_quantile),
        hetero_k=int(args.lraes_hetero_k),
    )
    lraes_cfg = LRAESConfig(
        beta=float(args.lraes_beta),
        reg_lambda=float(args.lraes_reg_lambda),
        top_k_per_class=int(args.lraes_top_k_per_class),
        rank_tol=float(args.lraes_rank_tol),
        eig_pos_eps=float(args.lraes_eig_pos_eps),
    )

    per_seed_rows: List[Dict[str, object]] = []
    target_health_rows: List[Dict[str, object]] = []
    fidelity_rows: List[Dict[str, object]] = []
    solver_rows_out: List[Dict[str, object]] = []

    for seed in seeds:
        print(f"[bridge-lraes-pilot][{dataset_name}][seed={seed}] split_start", flush=True)
        seed_dir = os.path.join(args.out_root, dataset_name, f"seed{seed}")
        _ensure_dir(seed_dir)

        train_trials, test_trials, split_meta = _make_trial_split(list(all_trials), int(seed))
        train_tmp, mean_log_train = _build_trial_records(train_trials, spd_eps=float(args.spd_eps))
        test_tmp, _ = _build_trial_records(test_trials, spd_eps=float(args.spd_eps))
        train_records = _apply_mean_log(train_tmp, mean_log_train)
        test_records = _apply_mean_log(test_tmp, mean_log_train)

        X_train_base = np.stack([r.z for r in train_records], axis=0).astype(np.float32)
        y_train_base = np.asarray([int(r.y) for r in train_records], dtype=np.int64)
        tid_train = np.asarray([r.tid for r in train_records], dtype=object)
        X_test = np.stack([r.z for r in test_records], axis=0).astype(np.float32)
        y_test = np.asarray([int(r.y) for r in test_records], dtype=np.int64)
        tid_test = np.asarray([r.tid for r in test_records], dtype=object)

        direction_bank, _bank_meta = _build_direction_bank_d1(
            X_train=X_train_base,
            k_dir=int(args.k_dir),
            seed=int(seed * 10000 + int(args.k_dir) * 113 + 17),
            n_iters=int(args.pia_n_iters),
            activation=str(args.pia_activation),
            bias_update_mode=str(args.pia_bias_update_mode),
            c_repr=float(args.pia_c_repr),
        )

        X_step1b, y_step1b, tid_step1b, src_step1b, dir_step1b, step1b_aug_meta = _build_multidir_aug_candidates(
            X_train=X_train_base,
            y_train=y_train_base,
            tid_train=tid_train,
            direction_bank=direction_bank,
            subset_size=int(args.subset_size),
            gamma=float(args.pia_gamma),
            multiplier=int(args.pia_multiplier),
            seed=int(seed + 100000 + int(args.k_dir) * 101 + int(args.subset_size) * 7),
        )
        mech_step1b = _compute_mech_metrics(
            X_train_real=X_train_base,
            y_train_real=y_train_base,
            X_aug_generated=X_step1b,
            y_aug_generated=y_step1b,
            X_aug_accepted=X_step1b,
            y_aug_accepted=y_step1b,
            X_src_accepted=src_step1b,
            dir_generated=dir_step1b,
            dir_accepted=dir_step1b,
            seed=int(seed),
            linear_c=float(args.linear_c),
            class_weight=str(args.linear_class_weight),
            linear_max_iter=int(args.linear_max_iter),
            knn_k=int(args.mech_knn_k),
            max_aug_for_mech=int(args.mech_max_aug_for_metrics),
            max_real_knn_ref=int(args.mech_max_real_knn_ref),
            max_real_knn_query=int(args.mech_max_real_knn_query),
            progress_prefix=f"[bridge-lraes-pilot][{dataset_name}][seed={seed}][z-step1b-mech]",
        )
        X_train_step1b = np.vstack([X_train_base, X_step1b]) if len(y_step1b) else X_train_base.copy()
        y_train_step1b = np.concatenate([y_train_base, y_step1b]) if len(y_step1b) else y_train_base.copy()
        tid_train_step1b = np.concatenate([tid_train, tid_step1b]) if len(y_step1b) else tid_train.copy()
        is_aug_step1b = (
            np.concatenate([np.zeros((len(y_train_base),), dtype=bool), np.ones((len(y_step1b),), dtype=bool)])
            if len(y_step1b)
            else np.zeros((len(y_train_base),), dtype=bool)
        )
        z_step1b_metrics, _ = _fit_eval_linearsvc(
            X_train_step1b,
            y_train_step1b,
            tid_train_step1b,
            X_test,
            y_test,
            tid_test,
            seed=int(seed),
            cap_k=int(args.nominal_cap_k),
            cap_seed=int(seed + 41),
            cap_sampling_policy="balanced_real_aug",
            linear_c=float(args.linear_c),
            class_weight=str(args.linear_class_weight),
            max_iter=int(args.linear_max_iter),
            agg_mode="majority",
            is_aug_train=is_aug_step1b,
            progress_prefix=f"[bridge-lraes-pilot][{dataset_name}][seed={seed}][z-step1b-fit]",
        )
        step1b_dir_summary = _summarize_dir_profile(mech_step1b.get("dir_profile", {}))

        round_rows: List[Dict[str, object]] = []
        gamma_by_dir = np.full((int(args.k_dir),), float(args.curriculum_init_gamma), dtype=np.float64)
        for round_id in range(1, int(args.curriculum_rounds) + 1):
            direction_probs = _active_direction_probs(gamma_by_dir, freeze_eps=float(args.curriculum_freeze_eps))
            gamma_before = gamma_by_dir.copy()
            X_curr, y_curr, tid_curr, src_curr, dir_curr, curr_aug_meta = _build_curriculum_aug_candidates(
                X_train=X_train_base,
                y_train=y_train_base,
                tid_train=tid_train,
                direction_bank=direction_bank,
                direction_probs=direction_probs,
                gamma_by_dir=gamma_before,
                multiplier=int(args.pia_multiplier),
                seed=int(seed + 400000 + round_id * 1009),
            )
            mech_curr = _compute_mech_metrics(
                X_train_real=X_train_base,
                y_train_real=y_train_base,
                X_aug_generated=X_curr,
                y_aug_generated=y_curr,
                X_aug_accepted=X_curr,
                y_aug_accepted=y_curr,
                X_src_accepted=src_curr,
                dir_generated=dir_curr,
                dir_accepted=dir_curr,
                seed=int(seed),
                linear_c=float(args.linear_c),
                class_weight=str(args.linear_class_weight),
                linear_max_iter=int(args.linear_max_iter),
                knn_k=int(args.mech_knn_k),
                max_aug_for_mech=int(args.mech_max_aug_for_metrics),
                max_real_knn_ref=int(args.mech_max_real_knn_ref),
                max_real_knn_query=int(args.mech_max_real_knn_query),
                progress_prefix=f"[bridge-lraes-pilot][{dataset_name}][seed={seed}][z-curr-round={round_id}-mech]",
            )
            intrusion_by_dir = _compute_direction_intrusion(
                X_anchor=X_train_base,
                y_anchor=y_train_base,
                X_aug_accepted=X_curr,
                y_aug_accepted=y_curr,
                dir_accepted=dir_curr,
                seed=int(seed),
                knn_k=int(args.mech_knn_k),
                max_eval=int(args.mech_max_aug_for_metrics),
            )
            maps = _mech_dir_maps(mech_curr, intrusion_by_dir=intrusion_by_dir)
            X_train_curr = np.vstack([X_train_base, X_curr]) if len(y_curr) else X_train_base.copy()
            y_train_curr = np.concatenate([y_train_base, y_curr]) if len(y_curr) else y_train_base.copy()
            tid_train_curr = np.concatenate([tid_train, tid_curr]) if len(y_curr) else tid_train.copy()
            is_aug_curr = (
                np.concatenate([np.zeros((len(y_train_base),), dtype=bool), np.ones((len(y_curr),), dtype=bool)])
                if len(y_curr)
                else np.zeros((len(y_train_base),), dtype=bool)
            )
            z_curr_metrics, _ = _fit_eval_linearsvc(
                X_train_curr,
                y_train_curr,
                tid_train_curr,
                X_test,
                y_test,
                tid_test,
                seed=int(seed),
                cap_k=int(args.nominal_cap_k),
                cap_seed=int(seed + 41),
                cap_sampling_policy="balanced_real_aug",
                linear_c=float(args.linear_c),
                class_weight=str(args.linear_class_weight),
                max_iter=int(args.linear_max_iter),
                agg_mode="majority",
                is_aug_train=is_aug_curr,
                progress_prefix=f"[bridge-lraes-pilot][{dataset_name}][seed={seed}][z-curr-round={round_id}-fit]",
            )
            gamma_after, state_by_dir, score_by_dir = _update_direction_budget(
                gamma_before=gamma_before,
                margin_by_dir=maps["margin_drop_median"],
                flip_by_dir=maps["flip_rate"],
                intrusion_by_dir=maps["intrusion"],
                expand_factor=float(args.curriculum_expand_factor),
                shrink_factor=float(args.curriculum_shrink_factor),
                gamma_max=float(args.curriculum_gamma_max),
                freeze_eps=float(args.curriculum_freeze_eps),
            )
            round_rows.append(
                {
                    "round_id": int(round_id),
                    "z_aug": X_curr,
                    "y_aug": y_curr,
                    "tid_aug": tid_curr,
                    "z_trial_macro_f1": float(z_curr_metrics["trial_macro_f1"]),
                    "z_window_macro_f1": float(z_curr_metrics["window_macro_f1"]),
                    "mech": mech_curr,
                    "dir_summary": _summarize_dir_profile(mech_curr.get("dir_profile", {})),
                    "direction_usage_entropy": float(curr_aug_meta.get("direction_usage_entropy", 0.0)),
                    "aug_meta": curr_aug_meta,
                    "direction_probs": curr_aug_meta.get("direction_probs", {}),
                    "gamma_before": {str(i): float(gamma_before[i]) for i in range(len(gamma_before))},
                    "gamma_after": {str(i): float(gamma_after[i]) for i in range(len(gamma_after))},
                    "direction_state": {str(k): str(v) for k, v in state_by_dir.items()},
                    "direction_score": {str(k): float(v) for k, v in score_by_dir.items()},
                }
            )
            gamma_by_dir = gamma_after.copy()
        best_round = _select_best_curriculum_round(round_rows)

        lraes_bank, lraes_prior_frozen_mask, lraes_bank_meta, lraes_solver_rows = build_lraes_direction_bank(
            X_train_base,
            y_train_base,
            k_dir=int(args.k_dir),
            fisher_cfg=fisher_cfg,
            lraes_cfg=lraes_cfg,
        )
        lraes_gamma_by_dir = np.full((int(args.k_dir),), float(args.curriculum_init_gamma), dtype=np.float64)
        lraes_gamma_by_dir[np.asarray(lraes_prior_frozen_mask, dtype=bool)] = 0.0
        lraes_round_rows: List[Dict[str, object]] = []
        for round_id in range(1, int(args.curriculum_rounds) + 1):
            direction_probs = _active_direction_probs(lraes_gamma_by_dir, freeze_eps=float(args.curriculum_freeze_eps))
            gamma_before = lraes_gamma_by_dir.copy()
            X_lraes, y_lraes, tid_lraes, src_lraes, dir_lraes, lraes_aug_meta = _build_curriculum_aug_candidates(
                X_train=X_train_base,
                y_train=y_train_base,
                tid_train=tid_train,
                direction_bank=lraes_bank,
                direction_probs=direction_probs,
                gamma_by_dir=gamma_before,
                multiplier=int(args.pia_multiplier),
                seed=int(seed + 900000 + round_id * 1009),
            )
            mech_lraes = _compute_mech_metrics(
                X_train_real=X_train_base,
                y_train_real=y_train_base,
                X_aug_generated=X_lraes,
                y_aug_generated=y_lraes,
                X_aug_accepted=X_lraes,
                y_aug_accepted=y_lraes,
                X_src_accepted=src_lraes,
                dir_generated=dir_lraes,
                dir_accepted=dir_lraes,
                seed=int(seed),
                linear_c=float(args.linear_c),
                class_weight=str(args.linear_class_weight),
                linear_max_iter=int(args.linear_max_iter),
                knn_k=int(args.mech_knn_k),
                max_aug_for_mech=int(args.mech_max_aug_for_metrics),
                max_real_knn_ref=int(args.mech_max_real_knn_ref),
                max_real_knn_query=int(args.mech_max_real_knn_query),
                progress_prefix=f"[bridge-lraes-pilot][{dataset_name}][seed={seed}][z-lraes-round={round_id}-mech]",
            )
            intrusion_by_dir = _compute_direction_intrusion(
                X_anchor=X_train_base,
                y_anchor=y_train_base,
                X_aug_accepted=X_lraes,
                y_aug_accepted=y_lraes,
                dir_accepted=dir_lraes,
                seed=int(seed),
                knn_k=int(args.mech_knn_k),
                max_eval=int(args.mech_max_aug_for_metrics),
            )
            maps = _mech_dir_maps(mech_lraes, intrusion_by_dir=intrusion_by_dir)
            X_train_lraes = np.vstack([X_train_base, X_lraes]) if len(y_lraes) else X_train_base.copy()
            y_train_lraes = np.concatenate([y_train_base, y_lraes]) if len(y_lraes) else y_train_base.copy()
            tid_train_lraes = np.concatenate([tid_train, tid_lraes]) if len(y_lraes) else tid_train.copy()
            is_aug_lraes = (
                np.concatenate([np.zeros((len(y_train_base),), dtype=bool), np.ones((len(y_lraes),), dtype=bool)])
                if len(y_lraes)
                else np.zeros((len(y_train_base),), dtype=bool)
            )
            z_lraes_metrics, _ = _fit_eval_linearsvc(
                X_train_lraes,
                y_train_lraes,
                tid_train_lraes,
                X_test,
                y_test,
                tid_test,
                seed=int(seed),
                cap_k=int(args.nominal_cap_k),
                cap_seed=int(seed + 41),
                cap_sampling_policy="balanced_real_aug",
                linear_c=float(args.linear_c),
                class_weight=str(args.linear_class_weight),
                max_iter=int(args.linear_max_iter),
                agg_mode="majority",
                is_aug_train=is_aug_lraes,
                progress_prefix=f"[bridge-lraes-pilot][{dataset_name}][seed={seed}][z-lraes-round={round_id}-fit]",
            )
            gamma_after, state_by_dir, score_by_dir = _update_direction_budget(
                gamma_before=gamma_before,
                margin_by_dir=maps["margin_drop_median"],
                flip_by_dir=maps["flip_rate"],
                intrusion_by_dir=maps["intrusion"],
                expand_factor=float(args.curriculum_expand_factor),
                shrink_factor=float(args.curriculum_shrink_factor),
                gamma_max=float(args.curriculum_gamma_max),
                freeze_eps=float(args.curriculum_freeze_eps),
            )
            lraes_round_rows.append(
                {
                    "round_id": int(round_id),
                    "z_aug": X_lraes,
                    "y_aug": y_lraes,
                    "tid_aug": tid_lraes,
                    "z_trial_macro_f1": float(z_lraes_metrics["trial_macro_f1"]),
                    "z_window_macro_f1": float(z_lraes_metrics["window_macro_f1"]),
                    "mech": mech_lraes,
                    "dir_summary": _summarize_dir_profile(mech_lraes.get("dir_profile", {})),
                    "direction_usage_entropy": float(lraes_aug_meta.get("direction_usage_entropy", 0.0)),
                    "aug_meta": lraes_aug_meta,
                    "direction_probs": lraes_aug_meta.get("direction_probs", {}),
                    "gamma_before": {str(i): float(gamma_before[i]) for i in range(len(gamma_before))},
                    "gamma_after": {str(i): float(gamma_after[i]) for i in range(len(gamma_after))},
                    "direction_state": {str(k): str(v) for k, v in state_by_dir.items()},
                    "direction_score": {str(k): float(v) for k, v in score_by_dir.items()},
                }
            )
            lraes_gamma_by_dir = gamma_after.copy()
        best_lraes_round = _select_best_curriculum_round(lraes_round_rows)

        print(f"[bridge-lraes-pilot][{dataset_name}][seed={seed}] raw_only_start", flush=True)
        raw_base_metrics, raw_base_meta = _fit_raw_minirocket(
            dataset=dataset_name,
            train_trials=_records_to_trial_dicts(train_records),
            test_trials=_records_to_trial_dicts(test_records),
            seed=int(seed),
            args=args,
        )
        print(f"[bridge-lraes-pilot][{dataset_name}][seed={seed}] raw_only_done", flush=True)

        single_aug_trials, single_bridge_meta = _bridge_aug_trials(
            train_records=train_records,
            mean_log_train=np.asarray(mean_log_train, dtype=np.float32),
            z_aug=np.asarray(X_step1b, dtype=np.float32),
            y_aug=np.asarray(y_step1b, dtype=np.int64),
            tid_aug=np.asarray(tid_step1b),
            variant_tag="single_round",
            bridge_eps=float(args.bridge_eps),
        )
        raw_single_metrics, raw_single_meta = _fit_raw_minirocket(
            dataset=dataset_name,
            train_trials=_records_to_trial_dicts(list(train_records) + single_aug_trials),
            test_trials=_records_to_trial_dicts(test_records),
            seed=int(seed),
            args=args,
        )

        multi_aug_trials, multi_bridge_meta = _bridge_aug_trials(
            train_records=train_records,
            mean_log_train=np.asarray(mean_log_train, dtype=np.float32),
            z_aug=np.asarray(best_round["z_aug"], dtype=np.float32),
            y_aug=np.asarray(best_round["y_aug"], dtype=np.int64),
            tid_aug=np.asarray(best_round["tid_aug"]),
            variant_tag=f"multiround_r{int(best_round['round_id'])}",
            bridge_eps=float(args.bridge_eps),
        )
        raw_multi_metrics, raw_multi_meta = _fit_raw_minirocket(
            dataset=dataset_name,
            train_trials=_records_to_trial_dicts(list(train_records) + multi_aug_trials),
            test_trials=_records_to_trial_dicts(test_records),
            seed=int(seed),
            args=args,
        )

        lraes_aug_trials, lraes_bridge_meta = _bridge_aug_trials(
            train_records=train_records,
            mean_log_train=np.asarray(mean_log_train, dtype=np.float32),
            z_aug=np.asarray(best_lraes_round["z_aug"], dtype=np.float32),
            y_aug=np.asarray(best_lraes_round["y_aug"], dtype=np.int64),
            tid_aug=np.asarray(best_lraes_round["tid_aug"]),
            variant_tag=f"lraes_r{int(best_lraes_round['round_id'])}",
            bridge_eps=float(args.bridge_eps),
        )
        raw_lraes_metrics, raw_lraes_meta = _fit_raw_minirocket(
            dataset=dataset_name,
            train_trials=_records_to_trial_dicts(list(train_records) + lraes_aug_trials),
            test_trials=_records_to_trial_dicts(test_records),
            seed=int(seed),
            args=args,
        )

        variants = [
            ("bridge_single_round", None, step1b_aug_meta, mech_step1b, step1b_dir_summary, {"trial_macro_f1": float(z_step1b_metrics["trial_macro_f1"]), "window_macro_f1": float(z_step1b_metrics["window_macro_f1"])}, single_bridge_meta),
            ("bridge_multiround_curriculum", int(best_round["round_id"]), best_round["aug_meta"], best_round["mech"], best_round["dir_summary"], {"trial_macro_f1": float(best_round["z_trial_macro_f1"]), "window_macro_f1": float(best_round["z_window_macro_f1"])}, multi_bridge_meta),
            ("bridge_lraes_curriculum", int(best_lraes_round["round_id"]), best_lraes_round["aug_meta"], best_lraes_round["mech"], best_lraes_round["dir_summary"], {"trial_macro_f1": float(best_lraes_round["z_trial_macro_f1"]), "window_macro_f1": float(best_lraes_round["z_window_macro_f1"])}, lraes_bridge_meta),
        ]
        for target_variant, best_round_id, aug_meta, mech, dir_summary, z_metrics, bridge_meta in variants:
            target_health_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": int(seed),
                    "target_variant": target_variant,
                    "best_round": "" if best_round_id is None else int(best_round_id),
                    "direction_usage_entropy": _direction_usage_entropy_from_aug_meta(aug_meta),
                    "worst_dir_summary": str(dir_summary.get("dir_profile_summary", "n/a")),
                    "direction_health_comment": _target_health_comment(
                        target_variant,
                        dir_summary,
                        _direction_usage_entropy_from_aug_meta(aug_meta),
                        ref_margin=step1b_dir_summary.get("worst_dir_margin_drop"),
                    ),
                    "z_target_trial_macro_f1": float(z_metrics["trial_macro_f1"]),
                    "z_target_window_macro_f1": float(z_metrics["window_macro_f1"]),
                }
            )
            fidelity_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": int(seed),
                    "target_variant": target_variant,
                    "best_round": "" if best_round_id is None else int(best_round_id),
                    "bridge_cov_match_error": float(bridge_meta["bridge_cov_match_error_mean"]),
                    "bridge_cov_to_orig_distance": float(bridge_meta["bridge_cov_to_orig_distance_logeuc_mean"]),
                    "energy_ratio": float(bridge_meta["energy_ratio_mean"]),
                    "cond_A": float(bridge_meta["cond_A_mean"]),
                    "raw_mean_shift_abs": float(bridge_meta["raw_mean_shift_abs_mean"]),
                    "risk_comment": _risk_comment(bridge_meta),
                }
            )

        state_counts = dict(lraes_bank_meta.get("solver_state_counts", {}))
        solver_rows_out.append(
            {
                "dataset": dataset_name,
                "seed": int(seed),
                "beta": float(args.lraes_beta),
                "top1_eigenvalue": max((float(r["top1_eigenvalue"]) for r in lraes_solver_rows), default=0.0),
                "topK_positive_count": int(sum(int(r["topk_positive_count"]) for r in lraes_solver_rows)),
                "topK_nonpositive_count": int(sum(int(r["topk_nonpositive_count"]) for r in lraes_solver_rows)),
                "solver_state": (
                    "fully_risk_dominated"
                    if int(state_counts.get("fully_risk_dominated", 0)) > 0
                    else "marginal"
                    if int(state_counts.get("marginal", 0)) > 0
                    else "safe_expandable"
                ),
                "fully_risk_dominated_count": int(state_counts.get("fully_risk_dominated", 0)),
                "solver_comment": _solver_comment(state_counts, int(lraes_bank_meta.get("low_quality_axis_count", 0))),
            }
        )

        per_seed_rows.append(
            {
                "dataset": dataset_name,
                "seed": int(seed),
                "raw_only_acc": float(raw_base_metrics["trial_acc"]),
                "raw_only_f1": float(raw_base_metrics["trial_macro_f1"]),
                "bridge_single_round_acc": float(raw_single_metrics["trial_acc"]),
                "bridge_single_round_f1": float(raw_single_metrics["trial_macro_f1"]),
                "bridge_multiround_acc": float(raw_multi_metrics["trial_acc"]),
                "bridge_multiround_f1": float(raw_multi_metrics["trial_macro_f1"]),
                "bridge_lraes_acc": float(raw_lraes_metrics["trial_acc"]),
                "bridge_lraes_f1": float(raw_lraes_metrics["trial_macro_f1"]),
                "delta_vs_raw_only": float(raw_lraes_metrics["trial_macro_f1"] - raw_base_metrics["trial_macro_f1"]),
                "delta_vs_bridge_single_round": float(raw_lraes_metrics["trial_macro_f1"] - raw_single_metrics["trial_macro_f1"]),
                "delta_vs_bridge_multiround": float(raw_lraes_metrics["trial_macro_f1"] - raw_multi_metrics["trial_macro_f1"]),
                "best_multiround_round": int(best_round["round_id"]),
                "best_lraes_round": int(best_lraes_round["round_id"]),
                "result_label": "pending",
            }
        )

        _write_json(
            os.path.join(seed_dir, "pilot_run_meta.json"),
            {
                "split_meta": split_meta,
                "raw_only_metrics": raw_base_metrics,
                "raw_only_run_meta": raw_base_meta,
                "bridge_single_round_metrics": raw_single_metrics,
                "bridge_single_round_run_meta": raw_single_meta,
                "bridge_single_round_fidelity": single_bridge_meta,
                "bridge_multiround_metrics": raw_multi_metrics,
                "bridge_multiround_run_meta": raw_multi_meta,
                "bridge_multiround_fidelity": multi_bridge_meta,
                "bridge_lraes_curriculum_metrics": raw_lraes_metrics,
                "bridge_lraes_curriculum_run_meta": raw_lraes_meta,
                "bridge_lraes_curriculum_fidelity": lraes_bridge_meta,
                "zspace_best_multiround_round": {
                    "round_id": int(best_round["round_id"]),
                    "trial_macro_f1": float(best_round["z_trial_macro_f1"]),
                },
                "zspace_best_lraes_round": {
                    "round_id": int(best_lraes_round["round_id"]),
                    "trial_macro_f1": float(best_lraes_round["z_trial_macro_f1"]),
                    "lraes_beta": float(args.lraes_beta),
                },
                "lraes_bank_meta": lraes_bank_meta,
                "lraes_solver_rows": lraes_solver_rows,
            },
        )
        print(
            f"[bridge-lraes-pilot][{dataset_name}][seed={seed}] "
            f"raw_only_f1={raw_base_metrics['trial_macro_f1']:.4f} "
            f"single_bridge_f1={raw_single_metrics['trial_macro_f1']:.4f} "
            f"multiround_bridge_f1={raw_multi_metrics['trial_macro_f1']:.4f} "
            f"lraes_bridge_f1={raw_lraes_metrics['trial_macro_f1']:.4f}",
            flush=True,
        )

    per_seed_df = pd.DataFrame(per_seed_rows).sort_values("seed").reset_index(drop=True)
    fid_df = pd.DataFrame(fidelity_rows).sort_values(["target_variant", "seed"]).reset_index(drop=True)
    lraes_fid = fid_df[fid_df["target_variant"] == "bridge_lraes_curriculum"]
    multi_fid = fid_df[fid_df["target_variant"] == "bridge_multiround_curriculum"]
    lraes_fidelity_not_worse = bool(
        float(lraes_fid["bridge_cov_match_error"].mean()) <= float(multi_fid["bridge_cov_match_error"].mean()) + 1e-6
        and float(lraes_fid["cond_A"].mean()) <= float(multi_fid["cond_A"].mean()) + 1e-6
        and float(lraes_fid["raw_mean_shift_abs"].mean()) <= float(multi_fid["raw_mean_shift_abs"].mean()) + 1e-6
    )
    per_seed_df["result_label"] = [
        _result_label(float(v), lraes_fidelity_not_worse) for v in per_seed_df["delta_vs_bridge_multiround"].astype(float)
    ]
    per_seed_df.to_csv(os.path.join(args.out_root, output_files["per_seed"]), index=False)

    summary_row = {
        "dataset": dataset_name,
        "raw_only_acc": float(per_seed_df["raw_only_acc"].mean()),
        "raw_only_f1": float(per_seed_df["raw_only_f1"].mean()),
        "raw_only_acc_mean_std": _format_mean_std(per_seed_df["raw_only_acc"]),
        "raw_only_f1_mean_std": _format_mean_std(per_seed_df["raw_only_f1"]),
        "bridge_single_round_acc": float(per_seed_df["bridge_single_round_acc"].mean()),
        "bridge_single_round_f1": float(per_seed_df["bridge_single_round_f1"].mean()),
        "bridge_single_round_acc_mean_std": _format_mean_std(per_seed_df["bridge_single_round_acc"]),
        "bridge_single_round_f1_mean_std": _format_mean_std(per_seed_df["bridge_single_round_f1"]),
        "bridge_multiround_acc": float(per_seed_df["bridge_multiround_acc"].mean()),
        "bridge_multiround_f1": float(per_seed_df["bridge_multiround_f1"].mean()),
        "bridge_multiround_acc_mean_std": _format_mean_std(per_seed_df["bridge_multiround_acc"]),
        "bridge_multiround_f1_mean_std": _format_mean_std(per_seed_df["bridge_multiround_f1"]),
        "bridge_lraes_acc": float(per_seed_df["bridge_lraes_acc"].mean()),
        "bridge_lraes_f1": float(per_seed_df["bridge_lraes_f1"].mean()),
        "bridge_lraes_acc_mean_std": _format_mean_std(per_seed_df["bridge_lraes_acc"]),
        "bridge_lraes_f1_mean_std": _format_mean_std(per_seed_df["bridge_lraes_f1"]),
        "delta_vs_raw_only": float(per_seed_df["delta_vs_raw_only"].mean()),
        "delta_vs_bridge_single_round": float(per_seed_df["delta_vs_bridge_single_round"].mean()),
        "delta_vs_bridge_multiround": float(per_seed_df["delta_vs_bridge_multiround"].mean()),
        "result_label": _result_label(float(per_seed_df["delta_vs_bridge_multiround"].mean()), lraes_fidelity_not_worse),
    }
    pd.DataFrame([summary_row]).to_csv(os.path.join(args.out_root, output_files["summary"]), index=False)
    pd.DataFrame(target_health_rows).sort_values(["target_variant", "seed"]).reset_index(drop=True).to_csv(
        os.path.join(args.out_root, output_files["target_health"]), index=False
    )
    fid_df.to_csv(os.path.join(args.out_root, output_files["fidelity"]), index=False)
    pd.DataFrame(solver_rows_out).sort_values(["seed"]).reset_index(drop=True).to_csv(
        os.path.join(args.out_root, output_files["solver"]), index=False
    )

    delta_multi = float(summary_row["delta_vs_bridge_multiround"])
    if delta_multi > 1e-6 and lraes_fidelity_not_worse:
        promotion = "yes"
        interpretation = "边界修复升级正在转成更强 raw 收益"
        next_dataset = "natops"
    elif delta_multi >= -1e-6 and lraes_fidelity_not_worse:
        promotion = "candidate"
        interpretation = "target-side 更强，但 raw-side 仍偏边界修复"
        next_dataset = "natops"
    else:
        promotion = "no"
        interpretation = "当前优势仍主要停留在 target 端或 raw 转化不足"
        next_dataset = "hold"

    fully_risk_count = int(pd.DataFrame(solver_rows_out)["fully_risk_dominated_count"].sum()) if solver_rows_out else 0
    md = [
        "# Bridge LRAES Pilot Conclusion",
        "",
        "This is the first bridge-coupled LRAES pilot under the new Route B body.",
        "It is not for the old Phase15 freeze formal table and not for the SEED battlefield.",
        "",
        f"## {dataset_title} Result",
        "",
        f"- `raw_only` F1: `{summary_row['raw_only_f1_mean_std']}`",
        f"- `bridge_single_round` F1: `{summary_row['bridge_single_round_f1_mean_std']}`",
        f"- `bridge_multiround` F1: `{summary_row['bridge_multiround_f1_mean_std']}`",
        f"- `bridge_lraes` F1: `{summary_row['bridge_lraes_f1_mean_std']}`",
        f"- `delta_vs_raw_only`: `{summary_row['delta_vs_raw_only']:.6f}`",
        f"- `delta_vs_bridge_single_round`: `{summary_row['delta_vs_bridge_single_round']:.6f}`",
        f"- `delta_vs_bridge_multiround`: `{summary_row['delta_vs_bridge_multiround']:.6f}`",
        "",
        "## Decision",
        "",
        f"- Should LRAES be promoted to Route B front-end now: `{promotion}`",
        f"- SCP1 readout: `{interpretation}`",
        f"- Next pilot dataset: `{next_dataset}`",
        "",
        "## Solver Readout",
        "",
        f"- `beta`: `{float(args.lraes_beta):.2f}`",
        f"- `fully_risk_dominated_count`: `{fully_risk_count}`",
        f"- `lraes_fidelity_not_worse_than_multiround`: `{str(lraes_fidelity_not_worse).lower()}`",
    ]
    Path(os.path.join(args.out_root, output_files["conclusion"])).write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
