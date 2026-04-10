#!/usr/bin/env python
"""Phase15 LRAES + curriculum target-generation probe.

Independent upgrade line.

Goal:
- replace the old TELM2-derived direction bank with a local risk-aware
  eigen-solver (LRAES) bank built from:
    M = (S_expand + lambda I) - beta * (S_risk + lambda I)
- keep the existing curriculum stepping logic
- compare:
    baseline
    single_round_step1b
    multiround_curriculum
    lraes_curriculum

Important constraints:
- not part of Phase15 mainline freeze formal tables
- fixed-split small datasets only
- no bridge in this round
- no new heavyweight training loop
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.trial_dataset_factory import (  # noqa: E402
    DEFAULT_BANDS_EEG,
    DEFAULT_FINGERMOVEMENTS_ROOT,
    DEFAULT_NATOPS_ROOT,
    DEFAULT_SELFREGULATIONSCP1_ROOT,
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
from scripts.fisher_pia_utils import FisherPIAConfig  # noqa: E402
from scripts.lraes_utils import LRAESConfig, build_lraes_direction_bank  # noqa: E402
from scripts.run_phase15_mainline_freeze import (  # noqa: E402
    _make_protocol_split,
    _summarize_dir_profile,
)
from scripts.run_phase15_multiround_curriculum_probe import (  # noqa: E402
    _build_curriculum_aug_candidates,
    _compute_direction_intrusion,
    _entropy_from_probs,
    _mech_dir_maps,
    _update_direction_budget,
)
from scripts.run_phase15_step1a_maxplane import _fit_eval_linearsvc  # noqa: E402
from scripts.run_phase15_step1b_multidir_matrix import (  # noqa: E402
    _build_direction_bank_d1,
    _build_multidir_aug_candidates,
    _compute_mech_metrics,
    _write_condition,
)


CORE_DATASETS = {
    "natops",
    "selfregulationscp1",
    "fingermovements",
}


def _parse_csv_list(text: str) -> List[str]:
    out = [t.strip() for t in str(text).split(",") if t.strip()]
    if not out:
        raise ValueError("list cannot be empty")
    return out


def _parse_seed_list(text: str) -> List[int]:
    out = sorted(set(int(tok.strip()) for tok in str(text).split(",") if tok.strip()))
    if not out:
        raise ValueError("seed list cannot be empty")
    return out


def _parse_float_list(text: str) -> List[float]:
    out = [float(tok.strip()) for tok in str(text).split(",") if tok.strip()]
    if not out:
        raise ValueError("float list cannot be empty")
    return out


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


def _stats_string(values: Iterable[float], *, fmt: str = ".4f") -> str:
    s = _summary_stats(values)
    return (
        f"min={format(float(s['min']), fmt)}|"
        f"mean={format(float(s['mean']), fmt)}|"
        f"std={format(float(s['std']), fmt)}|"
        f"max={format(float(s['max']), fmt)}"
    )


def _format_mean_std(mean: float, std: float) -> str:
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def _dict_summary_string(values: Dict[int, float], *, fmt: str = ".4f") -> str:
    if not values:
        return "n/a"
    return "|".join(f"{int(k)}:{format(float(v), fmt)}" for k, v in sorted(values.items()))


def _result_label(delta: float) -> str:
    if delta > 1e-6:
        return "positive"
    if delta < -1e-6:
        return "negative"
    return "neutral"


def _active_direction_probs(gamma_by_dir: np.ndarray, *, freeze_eps: float) -> np.ndarray:
    g = np.asarray(gamma_by_dir, dtype=np.float64).ravel()
    active = g > float(freeze_eps)
    if not np.any(active):
        return np.full(g.shape, 1.0 / float(max(1, len(g))), dtype=np.float64)
    out = np.zeros_like(g, dtype=np.float64)
    out[active] = 1.0 / float(np.sum(active))
    return out


def _variant_health_row(
    *,
    dataset: str,
    variant: str,
    beta: float | None,
    round_id: int,
    usage_probs: np.ndarray,
    mech: Dict[str, object],
    intrusion_by_dir: Dict[int, float],
    frozen_dir_count: int,
    expanded_dir_count: int,
    note: str,
) -> Dict[str, object]:
    maps = _mech_dir_maps(mech, intrusion_by_dir=intrusion_by_dir)
    dir_summary = _summarize_dir_profile(mech.get("dir_profile", {}))
    return {
        "dataset": dataset,
        "variant": variant,
        "beta": (np.nan if beta is None else float(beta)),
        "round_id": int(round_id),
        "direction_usage_entropy": float(_entropy_from_probs(usage_probs)),
        "per_dir_margin_summary": _dict_summary_string(maps["margin_drop_median"]),
        "per_dir_flip_summary": _dict_summary_string(maps["flip_rate"]),
        "per_dir_intrusion_summary": _dict_summary_string(maps["intrusion"]),
        "worst_dir_id": dir_summary["worst_dir_id"],
        "worst_dir_summary": dir_summary["dir_profile_summary"],
        "frozen_dir_count": int(frozen_dir_count),
        "expanded_dir_count": int(expanded_dir_count),
        "direction_health_comment": str(note),
    }


def _solver_comment(
    *,
    solver_state: str,
    low_quality_axis_count: int,
    max_eigenvalue_is_positive: bool,
) -> str:
    if not bool(max_eigenvalue_is_positive):
        return "fully_risk_dominated_no_expandable_axis"
    if solver_state == "marginal" and low_quality_axis_count > 0:
        return "mixed_axis_pool_with_nonpositive_tail"
    if solver_state == "safe_expandable":
        return "positive_safe_axes_available"
    return "solver_signal_mixed"


def _run_curriculum_variant(
    *,
    dataset: str,
    seed: int,
    seed_dir: str,
    common_meta: Dict[str, object],
    X_train_base: np.ndarray,
    y_train_base: np.ndarray,
    tid_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tid_test: np.ndarray,
    direction_bank: np.ndarray,
    bank_meta: Dict[str, object],
    cap_seed: int,
    args: argparse.Namespace,
    variant_name: str,
    variant_beta: float | None,
    gamma_init: np.ndarray,
    prior_frozen_mask: np.ndarray,
    condition_prefix: str | None = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    perf_rows: List[Dict[str, object]] = []
    health_rows: List[Dict[str, object]] = []
    budget_rows: List[Dict[str, object]] = []

    gamma_by_dir = np.asarray(gamma_init, dtype=np.float64).copy()
    gamma_by_dir[np.asarray(prior_frozen_mask, dtype=bool)] = 0.0

    for round_id in range(1, int(args.n_rounds) + 1):
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
            seed=int(seed + 700000 + round_id * 1009 + (31 if variant_name == "lraes_curriculum" else 0)),
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
            class_weight=args.linear_class_weight,
            linear_max_iter=int(args.linear_max_iter),
            knn_k=int(args.mech_knn_k),
            max_aug_for_mech=int(args.mech_max_aug_for_metrics),
            max_real_knn_ref=int(args.mech_max_real_knn_ref),
            max_real_knn_query=int(args.mech_max_real_knn_query),
            progress_prefix=f"[lraes-curriculum][{dataset}][seed={seed}][{variant_name}][round={round_id}][mech]",
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
        for dir_id, frozen in enumerate(np.asarray(prior_frozen_mask, dtype=bool).tolist()):
            if frozen:
                gamma_after[dir_id] = 0.0
                state_by_dir[dir_id] = "freeze"
        gamma_by_dir = gamma_after.copy()

        X_train_curr = np.vstack([X_train_base, X_curr]) if len(y_curr) else X_train_base.copy()
        y_train_curr = np.concatenate([y_train_base, y_curr]) if len(y_curr) else y_train_base.copy()
        tid_train_curr = np.concatenate([tid_train, tid_curr]) if len(y_curr) else tid_train.copy()
        is_aug_curr = (
            np.concatenate([np.zeros((len(y_train_base),), dtype=bool), np.ones((len(y_curr),), dtype=bool)])
            if len(y_curr)
            else np.zeros((len(y_train_base),), dtype=bool)
        )
        metrics_curr, train_meta_curr = _fit_eval_linearsvc(
            X_train_curr,
            y_train_curr,
            tid_train_curr,
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
            is_aug_train=is_aug_curr,
            progress_prefix=f"[lraes-curriculum][{dataset}][seed={seed}][{variant_name}][round={round_id}]",
        )

        round_dir = os.path.join(seed_dir, f"{condition_prefix or variant_name}_R{round_id}")
        _write_condition(
            round_dir,
            metrics_curr,
            {
                **common_meta,
                **train_meta_curr,
                "condition": f"{condition_prefix or variant_name}_R{round_id}",
                "variant": variant_name,
                "variant_beta": (None if variant_beta is None else float(variant_beta)),
                "round_id": int(round_id),
                "augmentation": curr_aug_meta,
                "direction_bank": {**bank_meta, "subset_size": 1},
                "final_accept_rate": 1.0,
                "gamma_before": {str(i): float(gamma_before[i]) for i in range(len(gamma_before))},
                "gamma_after": {str(i): float(gamma_after[i]) for i in range(len(gamma_after))},
                "direction_state": {str(i): str(state_by_dir.get(i, "hold")) for i in range(len(gamma_after))},
                "direction_score": {str(i): float(score_by_dir.get(i, 0.0)) for i in range(len(gamma_after))},
                "mech": mech_curr,
                "direction_probs": {str(i): float(direction_probs[i]) for i in range(len(direction_probs))},
                "prior_frozen_mask": {str(i): bool(prior_frozen_mask[i]) for i in range(len(prior_frozen_mask))},
            },
        )

        frozen_dir_count = int(sum(1 for s in state_by_dir.values() if s == "freeze"))
        expanded_dir_count = int(sum(1 for s in state_by_dir.values() if s == "expand"))
        note = "lraes_curriculum_round" if variant_name == "lraes_curriculum" else "curriculum_round"
        usage_probs = np.asarray(
            [
                float(curr_aug_meta.get("mixing_stats", {}).get("direction_pick_fraction", {}).get(str(i), 0.0))
                for i in range(len(gamma_after))
            ],
            dtype=np.float64,
        )
        perf_rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "variant": variant_name,
                "beta": (np.nan if variant_beta is None else float(variant_beta)),
                "round_id": int(round_id),
                "acc": float(metrics_curr["trial_acc"]),
                "f1": float(metrics_curr["trial_macro_f1"]),
            }
        )
        health_rows.append(
            _variant_health_row(
                dataset=dataset,
                variant=variant_name,
                beta=variant_beta,
                round_id=int(round_id),
                usage_probs=usage_probs,
                mech=mech_curr,
                intrusion_by_dir=intrusion_by_dir,
                frozen_dir_count=frozen_dir_count,
                expanded_dir_count=expanded_dir_count,
                note=note,
            )
        )
        for dir_id in range(len(gamma_after)):
            budget_rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "variant": variant_name,
                    "beta": (np.nan if variant_beta is None else float(variant_beta)),
                    "round_id": int(round_id),
                    "dir_id": int(dir_id),
                    "gamma_before": float(gamma_before[dir_id]),
                    "gamma_after": float(gamma_after[dir_id]),
                    "direction_state": str(state_by_dir.get(dir_id, "hold")),
                    "usage_fraction": float(
                        curr_aug_meta.get("mixing_stats", {}).get("direction_pick_fraction", {}).get(str(dir_id), 0.0)
                    ),
                    "margin_drop": float(maps["margin_drop_median"].get(dir_id, 0.0)),
                    "flip_rate": float(maps["flip_rate"].get(dir_id, 0.0)),
                    "intrusion": float(maps["intrusion"].get(dir_id, 0.0)),
                }
            )
        print(
            f"[lraes-curriculum][{dataset}][seed={seed}][{variant_name}]"
            f"[round={round_id}] f1={metrics_curr['trial_macro_f1']:.4f} "
            f"freeze={frozen_dir_count} expand={expanded_dir_count}",
            flush=True,
        )

    return perf_rows, health_rows, budget_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="natops,selfregulationscp1,fingermovements")
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--out-root", type=str, default="out/phase15_lraes_curriculum_core_20260321")
    parser.add_argument("--natops-root", type=str, default=DEFAULT_NATOPS_ROOT)
    parser.add_argument("--fingermovements-root", type=str, default=DEFAULT_FINGERMOVEMENTS_ROOT)
    parser.add_argument("--selfregulationscp1-root", type=str, default=DEFAULT_SELFREGULATIONSCP1_ROOT)
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
    parser.add_argument("--k-dir", type=int, default=5)
    parser.add_argument("--subset-size", type=int, default=1)
    parser.add_argument("--pia-multiplier", type=int, default=1)
    parser.add_argument("--pia-gamma", type=float, default=0.10)
    parser.add_argument("--pia-n-iters", type=int, default=2)
    parser.add_argument("--pia-activation", type=str, default="sine")
    parser.add_argument("--pia-bias-update-mode", type=str, default="residual")
    parser.add_argument("--pia-c-repr", type=float, default=1.0)
    parser.add_argument("--mech-knn-k", type=int, default=20)
    parser.add_argument("--mech-max-aug-for-metrics", type=int, default=2000)
    parser.add_argument("--mech-max-real-knn-ref", type=int, default=10000)
    parser.add_argument("--mech-max-real-knn-query", type=int, default=1000)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--curriculum-init-gamma", type=float, default=0.06)
    parser.add_argument("--curriculum-expand-factor", type=float, default=1.25)
    parser.add_argument("--curriculum-shrink-factor", type=float, default=0.70)
    parser.add_argument("--curriculum-gamma-max", type=float, default=0.16)
    parser.add_argument("--curriculum-freeze-eps", type=float, default=0.02)
    parser.add_argument("--split-preview-n", type=int, default=5)
    parser.add_argument("--lraes-betas", type=str, default="0.5,1.0")
    parser.add_argument("--lraes-reg-lambda", type=float, default=1e-4)
    parser.add_argument("--lraes-top-k-per-class", type=int, default=3)
    parser.add_argument("--lraes-rank-tol", type=float, default=1e-8)
    parser.add_argument("--lraes-eig-pos-eps", type=float, default=1e-9)
    parser.add_argument("--lraes-knn-k", type=int, default=20)
    parser.add_argument("--lraes-boundary-quantile", type=float, default=0.30)
    parser.add_argument("--lraes-interior-quantile", type=float, default=0.70)
    parser.add_argument("--lraes-hetero-k", type=int, default=3)
    args = parser.parse_args()

    datasets = [normalize_dataset_name(x) for x in _parse_csv_list(args.datasets)]
    for ds in datasets:
        if ds not in CORE_DATASETS:
            raise ValueError(f"Unsupported dataset for LRAES probe: {ds}")
    if int(args.subset_size) != 1:
        raise ValueError("LRAES probe currently locks subset_size=1.")
    if int(args.n_rounds) < 1:
        raise ValueError("--n-rounds must be >= 1")

    seeds = _parse_seed_list(args.seeds)
    betas = _parse_float_list(args.lraes_betas)
    ensure_dir(args.out_root)

    perf_rows: List[Dict[str, object]] = []
    health_rows: List[Dict[str, object]] = []
    budget_rows: List[Dict[str, object]] = []
    solver_seed_rows: List[Dict[str, object]] = []
    solver_class_rows: List[Dict[str, object]] = []

    fisher_cfg = FisherPIAConfig(
        knn_k=int(args.lraes_knn_k),
        interior_quantile=float(args.lraes_interor_quantile) if hasattr(args, "lraes_interor_quantile") else float(args.lraes_interior_quantile),
        boundary_quantile=float(args.lraes_boundary_quantile),
        hetero_k=int(args.lraes_hetero_k),
    )

    for dataset in datasets:
        print(f"[lraes-curriculum][{dataset}] load", flush=True)
        all_trials = load_trials_for_dataset(
            dataset=dataset,
            processed_root=args.processed_root,
            stim_xlsx=args.stim_xlsx,
            natops_root=args.natops_root,
            fingermovements_root=args.fingermovements_root,
            selfregulationscp1_root=args.selfregulationscp1_root,
        )
        train_trials, test_trials, split_meta = _make_protocol_split(dataset, all_trials)
        bands_spec = resolve_band_spec(dataset, args.bands)
        bands = parse_band_spec(bands_spec)

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

        dataset_dir = os.path.join(args.out_root, dataset)
        ensure_dir(dataset_dir)

        for seed in seeds:
            print(f"[lraes-curriculum][{dataset}][seed={seed}] start", flush=True)
            seed_dir = os.path.join(dataset_dir, f"seed{seed}")
            ensure_dir(seed_dir)
            cap_seed = int(seed) + 41

            metrics_base, train_meta_base = _fit_eval_linearsvc(
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
                progress_prefix=f"[lraes-curriculum][{dataset}][seed={seed}][baseline]",
            )

            old_bank, old_bank_meta = _build_direction_bank_d1(
                X_train=X_train_base,
                k_dir=int(args.k_dir),
                seed=int(seed * 10000 + args.k_dir * 113 + 17),
                n_iters=int(args.pia_n_iters),
                activation=args.pia_activation,
                bias_update_mode=args.pia_bias_update_mode,
                c_repr=float(args.pia_c_repr),
            )

            X_step1b, y_step1b, tid_step1b, src_step1b, dir_step1b, step1b_aug_meta = _build_multidir_aug_candidates(
                X_train=X_train_base,
                y_train=y_train_base,
                tid_train=tid_train,
                direction_bank=old_bank,
                subset_size=int(args.subset_size),
                gamma=float(args.pia_gamma),
                multiplier=int(args.pia_multiplier),
                seed=int(seed + 100000 + args.k_dir * 101 + args.subset_size * 7),
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
                class_weight=args.linear_class_weight,
                linear_max_iter=int(args.linear_max_iter),
                knn_k=int(args.mech_knn_k),
                max_aug_for_mech=int(args.mech_max_aug_for_metrics),
                max_real_knn_ref=int(args.mech_max_real_knn_ref),
                max_real_knn_query=int(args.mech_max_real_knn_query),
                progress_prefix=f"[lraes-curriculum][{dataset}][seed={seed}][mech_step1b]",
            )
            step1b_intrusion_by_dir = _compute_direction_intrusion(
                X_anchor=X_train_base,
                y_anchor=y_train_base,
                X_aug_accepted=X_step1b,
                y_aug_accepted=y_step1b,
                dir_accepted=dir_step1b,
                seed=int(seed),
                knn_k=int(args.mech_knn_k),
                max_eval=int(args.mech_max_aug_for_metrics),
            )
            X_train_step1b = np.vstack([X_train_base, X_step1b]) if len(y_step1b) else X_train_base.copy()
            y_train_step1b = np.concatenate([y_train_base, y_step1b]) if len(y_step1b) else y_train_base.copy()
            tid_train_step1b = np.concatenate([tid_train, tid_step1b]) if len(y_step1b) else tid_train.copy()
            is_aug_step1b = (
                np.concatenate([np.zeros((len(y_train_base),), dtype=bool), np.ones((len(y_step1b),), dtype=bool)])
                if len(y_step1b)
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
                progress_prefix=f"[lraes-curriculum][{dataset}][seed={seed}][step1b]",
            )
            _write_condition(
                os.path.join(seed_dir, "A_baseline"),
                metrics_base,
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    **split_meta,
                    **train_meta_base,
                    "condition": "A_baseline",
                },
            )
            _write_condition(
                os.path.join(seed_dir, "B_step1b"),
                metrics_step1b,
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    **split_meta,
                    **train_meta_step1b,
                    "condition": "B_step1b",
                    "augmentation": step1b_aug_meta,
                    "direction_bank": {**old_bank_meta, "subset_size": int(args.subset_size)},
                    "mech": mech_step1b,
                },
            )

            perf_rows.extend(
                [
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "variant": "baseline",
                        "beta": np.nan,
                        "round_id": 0,
                        "acc": float(metrics_base["trial_acc"]),
                        "f1": float(metrics_base["trial_macro_f1"]),
                    },
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "variant": "single_round_step1b",
                        "beta": np.nan,
                        "round_id": 0,
                        "acc": float(metrics_step1b["trial_acc"]),
                        "f1": float(metrics_step1b["trial_macro_f1"]),
                    },
                ]
            )
            usage_probs_step1b = np.asarray(
                [
                    float(step1b_aug_meta.get("mixing_stats", {}).get("direction_pick_fraction", {}).get(str(i), 0.0))
                    for i in range(int(args.k_dir))
                ],
                dtype=np.float64,
            )
            health_rows.append(
                _variant_health_row(
                    dataset=dataset,
                    variant="single_round_step1b",
                    beta=None,
                    round_id=0,
                    usage_probs=usage_probs_step1b,
                    mech=mech_step1b,
                    intrusion_by_dir=step1b_intrusion_by_dir,
                    frozen_dir_count=0,
                    expanded_dir_count=0,
                    note="equal_weight_single_round",
                )
            )

            common_meta = {
                "dataset": dataset,
                "seed": int(seed),
                "protocol_type": split_meta["protocol_type"],
                "protocol_note": split_meta["protocol_note"],
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
                "curriculum_config": {
                    "n_rounds": int(args.n_rounds),
                    "init_gamma": float(args.curriculum_init_gamma),
                    "expand_factor": float(args.curriculum_expand_factor),
                    "shrink_factor": float(args.curriculum_shrink_factor),
                    "gamma_max": float(args.curriculum_gamma_max),
                    "freeze_eps": float(args.curriculum_freeze_eps),
                },
            }

            curr_perf, curr_health, curr_budget = _run_curriculum_variant(
                dataset=dataset,
                seed=int(seed),
                seed_dir=seed_dir,
                common_meta=common_meta,
                X_train_base=X_train_base,
                y_train_base=y_train_base,
                tid_train=tid_train,
                X_test=X_test,
                y_test=y_test,
                tid_test=tid_test,
                direction_bank=old_bank,
                bank_meta=old_bank_meta,
                cap_seed=cap_seed,
                args=args,
                variant_name="multiround_curriculum",
                variant_beta=None,
                gamma_init=np.full((int(args.k_dir),), float(args.curriculum_init_gamma), dtype=np.float64),
                prior_frozen_mask=np.zeros((int(args.k_dir),), dtype=bool),
                condition_prefix="multiround_curriculum",
            )
            perf_rows.extend(curr_perf)
            health_rows.extend(curr_health)
            budget_rows.extend(curr_budget)

            for beta in betas:
                lcfg = LRAESConfig(
                    beta=float(beta),
                    reg_lambda=float(args.lraes_reg_lambda),
                    top_k_per_class=int(args.lraes_top_k_per_class),
                    rank_tol=float(args.lraes_rank_tol),
                    eig_pos_eps=float(args.lraes_eig_pos_eps),
                )
                lraes_bank, prior_frozen_mask, lraes_meta, lraes_class_solver_rows = build_lraes_direction_bank(
                    X_train_base,
                    y_train_base,
                    k_dir=int(args.k_dir),
                    fisher_cfg=fisher_cfg,
                    lraes_cfg=lcfg,
                )
                selected_eigs = np.asarray(lraes_meta.get("selected_eigenvalues", []), dtype=np.float64)
                max_pos = bool(np.any(selected_eigs > float(args.lraes_eig_pos_eps)))
                low_quality_axis_count = int(lraes_meta.get("low_quality_axis_count", 0))
                if not max_pos:
                    solver_state = "fully_risk_dominated"
                elif low_quality_axis_count > 0:
                    solver_state = "marginal"
                else:
                    solver_state = "safe_expandable"
                solver_seed_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "beta": float(beta),
                        "local_matrix_rank_summary": str(lraes_meta.get("local_matrix_rank_summary", "n/a")),
                        "top1_eigenvalue": float(np.max(selected_eigs)) if selected_eigs.size else 0.0,
                        "topK_eigenvalue_summary": _stats_string(selected_eigs, fmt=".6f"),
                        "topK_positive_count": int(np.sum(selected_eigs > float(args.lraes_eig_pos_eps))),
                        "topK_nonpositive_count": int(np.sum(selected_eigs <= float(args.lraes_eig_pos_eps))),
                        "max_eigenvalue_is_positive": bool(max_pos),
                        "solver_state": solver_state,
                        "selected_axis_variance_summary": str(lraes_meta.get("selected_axis_variance_summary", "n/a")),
                        "low_quality_axis_count": low_quality_axis_count,
                        "solver_comment": _solver_comment(
                            solver_state=solver_state,
                            low_quality_axis_count=low_quality_axis_count,
                            max_eigenvalue_is_positive=max_pos,
                        ),
                    }
                )
                for row in lraes_class_solver_rows:
                    solver_class_rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            **row,
                        }
                    )

                lraes_perf, lraes_health, lraes_budget = _run_curriculum_variant(
                    dataset=dataset,
                    seed=int(seed),
                    seed_dir=seed_dir,
                    common_meta={
                        **common_meta,
                        "lraes_config": {
                            "beta": float(beta),
                            "reg_lambda": float(args.lraes_reg_lambda),
                            "top_k_per_class": int(args.lraes_top_k_per_class),
                            "rank_tol": float(args.lraes_rank_tol),
                            "eig_pos_eps": float(args.lraes_eig_pos_eps),
                        },
                    },
                    X_train_base=X_train_base,
                    y_train_base=y_train_base,
                    tid_train=tid_train,
                    X_test=X_test,
                    y_test=y_test,
                    tid_test=tid_test,
                    direction_bank=lraes_bank,
                    bank_meta=lraes_meta,
                    cap_seed=cap_seed,
                    args=args,
                    variant_name="lraes_curriculum",
                    variant_beta=float(beta),
                    gamma_init=np.full((int(lraes_bank.shape[0]),), float(args.curriculum_init_gamma), dtype=np.float64),
                    prior_frozen_mask=prior_frozen_mask,
                    condition_prefix=f"lraes_beta{str(beta).replace('.', 'p')}",
                )
                perf_rows.extend(lraes_perf)
                health_rows.extend(lraes_health)
                budget_rows.extend(lraes_budget)

    perf_df = pd.DataFrame(perf_rows).sort_values(["dataset", "variant", "beta", "seed", "round_id"]).reset_index(drop=True)
    perf_df.to_csv(os.path.join(args.out_root, "summary_per_seed.csv"), index=False)
    pd.DataFrame(budget_rows).sort_values(["dataset", "variant", "beta", "seed", "round_id", "dir_id"]).to_csv(
        os.path.join(args.out_root, "lraes_direction_budget_summary.csv"),
        index=False,
    )

    health_df = pd.DataFrame(health_rows)
    solver_seed_df = pd.DataFrame(solver_seed_rows)
    solver_class_df = pd.DataFrame(solver_class_rows)
    if not solver_class_df.empty:
        solver_class_df.to_csv(os.path.join(args.out_root, "lraes_solver_per_class.csv"), index=False)

    perf_summary_rows: List[Dict[str, object]] = []
    health_summary_rows: List[Dict[str, object]] = []
    solver_summary_rows: List[Dict[str, object]] = []

    for dataset, df_ds in perf_df.groupby("dataset"):
        df_base = df_ds[df_ds["variant"] == "baseline"]
        df_s1 = df_ds[df_ds["variant"] == "single_round_step1b"]
        df_curr = df_ds[df_ds["variant"] == "multiround_curriculum"]
        best_curr = df_curr.groupby("round_id", as_index=False)["f1"].mean()
        best_curr_row = best_curr.loc[int(best_curr["f1"].astype(float).idxmax())]
        best_curr_round = int(best_curr_row["round_id"])

        base_acc = _summary_stats(df_base["acc"])
        base_f1 = _summary_stats(df_base["f1"])
        s1_acc = _summary_stats(df_s1["acc"])
        s1_f1 = _summary_stats(df_s1["f1"])
        curr_acc = _summary_stats(df_curr[df_curr["round_id"] == best_curr_round]["acc"])
        curr_f1 = _summary_stats(df_curr[df_curr["round_id"] == best_curr_round]["f1"])

        for beta in betas:
            df_l = df_ds[(df_ds["variant"] == "lraes_curriculum") & (df_ds["beta"].astype(float) == float(beta))]
            if df_l.empty:
                continue
            best_l = df_l.groupby("round_id", as_index=False)["f1"].mean()
            best_l_row = best_l.loc[int(best_l["f1"].astype(float).idxmax())]
            best_l_round = int(best_l_row["round_id"])
            l_acc = _summary_stats(df_l[df_l["round_id"] == best_l_round]["acc"])
            l_f1 = _summary_stats(df_l[df_l["round_id"] == best_l_round]["f1"])
            delta_vs_curr = float(l_f1["mean"] - curr_f1["mean"])
            perf_summary_rows.append(
                {
                    "dataset": dataset,
                    "beta": float(beta),
                    "beta_role": ("main" if abs(float(beta) - 0.5) <= 1e-9 else "control"),
                    "baseline_acc": _format_mean_std(base_acc["mean"], base_acc["std"]),
                    "baseline_f1": _format_mean_std(base_f1["mean"], base_f1["std"]),
                    "single_round_step1b_acc": _format_mean_std(s1_acc["mean"], s1_acc["std"]),
                    "single_round_step1b_f1": _format_mean_std(s1_f1["mean"], s1_f1["std"]),
                    "multiround_curriculum_acc": _format_mean_std(curr_acc["mean"], curr_acc["std"]),
                    "multiround_curriculum_f1": _format_mean_std(curr_f1["mean"], curr_f1["std"]),
                    "lraes_curriculum_acc": _format_mean_std(l_acc["mean"], l_acc["std"]),
                    "lraes_curriculum_f1": _format_mean_std(l_f1["mean"], l_f1["std"]),
                    "best_curriculum_round": int(best_curr_round),
                    "best_lraes_round": int(best_l_round),
                    "delta_vs_single_round_step1b": float(l_f1["mean"] - s1_f1["mean"]),
                    "delta_vs_multiround_curriculum": delta_vs_curr,
                    "result_label": _result_label(delta_vs_curr),
                }
            )

            df_h_l = health_df[
                (health_df["dataset"] == dataset)
                & (health_df["variant"] == "lraes_curriculum")
                & (health_df["round_id"] == int(best_l_round))
                & (health_df["beta"].astype(float) == float(beta))
            ]
            if not df_h_l.empty:
                note_counts = df_h_l["direction_health_comment"].astype(str).value_counts().to_dict()
                health_summary_rows.append(
                    {
                        "dataset": dataset,
                        "beta": float(beta),
                        "variant": "lraes_curriculum",
                        "direction_usage_entropy": _format_mean_std(
                            float(df_h_l["direction_usage_entropy"].mean()),
                            float(df_h_l["direction_usage_entropy"].std()),
                        ),
                        "worst_dir_summary": " / ".join(df_h_l["worst_dir_summary"].astype(str).tolist()),
                        "frozen_dir_count": _format_mean_std(
                            float(df_h_l["frozen_dir_count"].mean()),
                            float(df_h_l["frozen_dir_count"].std()),
                        ),
                        "expanded_dir_count": _format_mean_std(
                            float(df_h_l["expanded_dir_count"].mean()),
                            float(df_h_l["expanded_dir_count"].std()),
                        ),
                        "direction_health_comment": "; ".join([f"{k}:{v}" for k, v in sorted(note_counts.items())]),
                    }
                )

            df_solver = solver_seed_df[(solver_seed_df["dataset"] == dataset) & (solver_seed_df["beta"].astype(float) == float(beta))]
            if not df_solver.empty:
                state_counts = df_solver["solver_state"].astype(str).value_counts().to_dict()
                solver_summary_rows.append(
                    {
                        "dataset": dataset,
                        "beta": float(beta),
                        "local_matrix_rank_summary": _stats_string(df_solver["top1_eigenvalue"].astype(float) * 0 + [
                            float(str(v).split("|")[1].split("=")[1]) if isinstance(v, str) and "|mean=" in str(v) else 0.0
                            for v in df_solver["local_matrix_rank_summary"].tolist()
                        ], fmt=".4f") if False else " / ".join(df_solver["local_matrix_rank_summary"].astype(str).tolist()),
                        "top1_eigenvalue": _format_mean_std(
                            float(df_solver["top1_eigenvalue"].astype(float).mean()),
                            float(df_solver["top1_eigenvalue"].astype(float).std()),
                        ),
                        "topK_eigenvalue_summary": " / ".join(df_solver["topK_eigenvalue_summary"].astype(str).tolist()),
                        "topK_positive_count": _format_mean_std(
                            float(df_solver["topK_positive_count"].astype(float).mean()),
                            float(df_solver["topK_positive_count"].astype(float).std()),
                        ),
                        "topK_nonpositive_count": _format_mean_std(
                            float(df_solver["topK_nonpositive_count"].astype(float).mean()),
                            float(df_solver["topK_nonpositive_count"].astype(float).std()),
                        ),
                        "max_eigenvalue_is_positive": bool(df_solver["max_eigenvalue_is_positive"].astype(bool).any()),
                        "solver_state": "; ".join([f"{k}:{v}" for k, v in sorted(state_counts.items())]),
                        "selected_axis_variance_summary": " / ".join(df_solver["selected_axis_variance_summary"].astype(str).tolist()),
                        "low_quality_axis_count": _format_mean_std(
                            float(df_solver["low_quality_axis_count"].astype(float).mean()),
                            float(df_solver["low_quality_axis_count"].astype(float).std()),
                        ),
                        "solver_comment": "; ".join(df_solver["solver_comment"].astype(str).value_counts().index.tolist()),
                    }
                )

        df_h_s1 = health_df[(health_df["dataset"] == dataset) & (health_df["variant"] == "single_round_step1b")]
        if not df_h_s1.empty:
            note_counts = df_h_s1["direction_health_comment"].astype(str).value_counts().to_dict()
            health_summary_rows.append(
                {
                    "dataset": dataset,
                    "beta": np.nan,
                    "variant": "single_round_step1b",
                    "direction_usage_entropy": _format_mean_std(
                        float(df_h_s1["direction_usage_entropy"].mean()),
                        float(df_h_s1["direction_usage_entropy"].std()),
                    ),
                    "worst_dir_summary": " / ".join(df_h_s1["worst_dir_summary"].astype(str).tolist()),
                    "frozen_dir_count": _format_mean_std(
                        float(df_h_s1["frozen_dir_count"].mean()),
                        float(df_h_s1["frozen_dir_count"].std()),
                    ),
                    "expanded_dir_count": _format_mean_std(
                        float(df_h_s1["expanded_dir_count"].mean()),
                        float(df_h_s1["expanded_dir_count"].std()),
                    ),
                    "direction_health_comment": "; ".join([f"{k}:{v}" for k, v in sorted(note_counts.items())]),
                }
            )
        df_h_curr = health_df[
            (health_df["dataset"] == dataset)
            & (health_df["variant"] == "multiround_curriculum")
            & (health_df["round_id"] == int(best_curr_round))
        ]
        if not df_h_curr.empty:
            note_counts = df_h_curr["direction_health_comment"].astype(str).value_counts().to_dict()
            health_summary_rows.append(
                {
                    "dataset": dataset,
                    "beta": np.nan,
                    "variant": "multiround_curriculum",
                    "direction_usage_entropy": _format_mean_std(
                        float(df_h_curr["direction_usage_entropy"].mean()),
                        float(df_h_curr["direction_usage_entropy"].std()),
                    ),
                    "worst_dir_summary": " / ".join(df_h_curr["worst_dir_summary"].astype(str).tolist()),
                    "frozen_dir_count": _format_mean_std(
                        float(df_h_curr["frozen_dir_count"].mean()),
                        float(df_h_curr["frozen_dir_count"].std()),
                    ),
                    "expanded_dir_count": _format_mean_std(
                        float(df_h_curr["expanded_dir_count"].mean()),
                        float(df_h_curr["expanded_dir_count"].std()),
                    ),
                    "direction_health_comment": "; ".join([f"{k}:{v}" for k, v in sorted(note_counts.items())]),
                }
            )

    perf_summary_df = pd.DataFrame(perf_summary_rows).sort_values(["dataset", "beta"]).reset_index(drop=True)
    perf_summary_df.to_csv(os.path.join(args.out_root, "lraes_curriculum_performance_summary.csv"), index=False)
    pd.DataFrame(health_summary_rows).sort_values(["dataset", "variant", "beta"]).to_csv(
        os.path.join(args.out_root, "lraes_direction_health_summary.csv"),
        index=False,
    )
    pd.DataFrame(solver_summary_rows).sort_values(["dataset", "beta"]).to_csv(
        os.path.join(args.out_root, "lraes_solver_summary.csv"),
        index=False,
    )

    main_df = perf_summary_df[np.isclose(perf_summary_df["beta"].astype(float), 0.5)] if not perf_summary_df.empty else perf_summary_df
    improved_main = int(np.sum(main_df["delta_vs_multiround_curriculum"].astype(float) > 1e-6)) if not main_df.empty else 0
    if improved_main >= 2:
        readout = "值得进入下一阶段"
    elif improved_main >= 1:
        readout = "继续作为探索线"
    else:
        readout = "当前方案暂缓"

    fully_risk_sets = []
    if solver_seed_df is not None and not solver_seed_df.empty:
        fully_risk_sets = sorted(
            set(
                solver_seed_df.loc[
                    solver_seed_df["solver_state"].astype(str) == "fully_risk_dominated",
                    "dataset",
                ].astype(str).tolist()
            )
        )

    best_dataset = "current evidence insufficient"
    if not main_df.empty:
        best_idx = int(main_df["delta_vs_multiround_curriculum"].astype(float).idxmax())
        best_dataset = str(main_df.loc[best_idx, "dataset"])

    beta_cmp_lines: List[str] = []
    for dataset in sorted(set(perf_summary_df["dataset"].tolist())):
        ds_rows = perf_summary_df[perf_summary_df["dataset"] == dataset]
        if ds_rows.empty:
            continue
        by_beta = {
            float(row["beta"]): float(row["delta_vs_multiround_curriculum"])
            for _, row in ds_rows.iterrows()
        }
        if 0.5 in by_beta and 1.0 in by_beta:
            beta_cmp_lines.append(
                f"- `{dataset}`: beta=0.5 delta `{by_beta[0.5]:+.4f}`, beta=1.0 delta `{by_beta[1.0]:+.4f}`"
            )

    bridge_hint = best_dataset if best_dataset != "current evidence insufficient" else "natops"
    conclusion_lines = [
        "# LRAES + Curriculum Conclusion",
        "",
        "更新时间：2026-03-21",
        "",
        "身份：`independent target-generator upgrade line`",
        "",
        "- `not for Phase15 mainline freeze table`",
        "- `not connected back to bridge in this round`",
        "- `goal = test whether local risk-aware eigen-directions improve target quality over pure multiround curriculum`",
        "",
        "## Core Datasets",
        "",
        "- `natops`",
        "- `selfregulationscp1`",
        "- `fingermovements`",
        "",
        "## Main Snapshot (beta = 0.5)",
        "",
    ]
    for _, row in main_df.iterrows():
        conclusion_lines.append(
            f"- `{row['dataset']}`: "
            f"curriculum={row['multiround_curriculum_f1']}, "
            f"lraes={row['lraes_curriculum_f1']}, "
            f"delta_vs_curriculum={float(row['delta_vs_multiround_curriculum']):+.4f}, "
            f"label=`{row['result_label']}`"
        )
    conclusion_lines.extend(
        [
            "",
            "## Beta Compare (0.5 main, 1.0 control)",
            "",
            *(beta_cmp_lines if beta_cmp_lines else ["- current evidence insufficient"]),
            "",
            "## Readout",
            "",
            f"- 当前是否应继续推进 LRAES + curriculum：`{readout}`",
            f"- 它是否已经超过纯 curriculum：`beta=0.5 improved={improved_main}/3`",
            "- 它的收益更集中在哪类数据集：`见 NATOPS / SCP1 / FingerMovements 分层结果`",
            f"- 当前是否观察到 fully_risk_dominated：`{'yes: ' + ', '.join(fully_risk_sets) if fully_risk_sets else 'no clear case in current core batch'}`",
            f"- 若后续重回 bridge，优先 very small pilot 数据集：`{bridge_hint}`",
            "",
            "## Guardrails",
            "",
            "- 当前在看的是上游 target 生成器，不是 bridge 层本身。",
            "- 若 beta=0.5 与 beta=1.0 差异显著，应优先以 beta=0.5 作为主报告，beta=1.0 仅作保守对照。",
            "- 若 fully_risk_dominated 出现增多，说明局部风险项已经压过可安全扩张谱结构。",
        ]
    )
    with open(os.path.join(args.out_root, "lraes_curriculum_conclusion.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(conclusion_lines) + "\n")


if __name__ == "__main__":
    main()
