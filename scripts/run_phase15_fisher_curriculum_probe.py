#!/usr/bin/env python
"""Phase15 Fisher/C0 + curriculum target-cleaning probe.

Independent upgrade line.

Goal:
- test whether Fisher/C0 prior cleaning improves target quality on top of the
  existing multiround curriculum stepping
- keep the experiment strictly in z-space (no bridge here)
- keep it outside Phase15 mainline freeze formal tables

Variants:
- baseline
- single_round_step1b
- multiround_curriculum
- fisher_c0_plus_curriculum
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
    DEFAULT_HANDMOVEMENTDIRECTION_ROOT,
    DEFAULT_HAR_ROOT,
    DEFAULT_NATOPS_ROOT,
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
)
from scripts.fisher_pia_utils import (  # noqa: E402
    FisherPIAConfig,
    compute_fisher_pia_terms,
    compute_safe_axis_scores,
)
from scripts.run_phase15_feedback_upgrade_probe import _score_to_probs  # noqa: E402
from scripts.run_phase15_mainline_freeze import (  # noqa: E402
    _make_protocol_split,
    _summarize_dir_profile,
)
from scripts.run_phase15_multiround_curriculum_probe import (  # noqa: E402
    _build_curriculum_aug_candidates,
    _compute_direction_intrusion,
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


CORE_DECISION_SETS = {
    "natops": "core_decision_sets",
    "selfregulationscp1": "core_decision_sets",
    "fingermovements": "core_decision_sets",
}
EXTENSION_SETS = {
    "har": "extension_sets",
    "handmovementdirection": "extension_sets",
    "uwavegesturelibrary": "extension_sets",
}
ALLOWED_DATASETS = {**CORE_DECISION_SETS, **EXTENSION_SETS}


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


def _format_mean_std(mean: float, std: float) -> str:
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def _entropy_from_probs(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _dict_summary_string(values: Dict[int, float], *, fmt: str = ".4f") -> str:
    if not values:
        return "n/a"
    parts = [f"{int(k)}:{format(float(v), fmt)}" for k, v in sorted(values.items())]
    return "|".join(parts)


def _minmax_norm(x: np.ndarray, *, constant_fill: float = 0.5) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64).ravel()
    if xx.size == 0:
        return np.asarray([], dtype=np.float64)
    xmin = float(np.min(xx))
    xmax = float(np.max(xx))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or abs(xmax - xmin) <= 1e-12:
        return np.full(xx.shape, float(constant_fill), dtype=np.float64)
    return (xx - xmin) / (xmax - xmin)


def _dataset_group(dataset: str) -> str:
    ds = normalize_dataset_name(dataset)
    if ds in ALLOWED_DATASETS:
        return str(ALLOWED_DATASETS[ds])
    return "unknown"


def _result_label(delta: float) -> str:
    if delta > 1e-6:
        return "positive"
    if delta < -1e-6:
        return "negative"
    return "neutral"


def _build_direction_probs_with_prior(
    *,
    fisher_probs: np.ndarray,
    gamma_by_dir: np.ndarray,
    freeze_eps: float,
) -> np.ndarray:
    probs = np.asarray(fisher_probs, dtype=np.float64).ravel().copy()
    gamma = np.asarray(gamma_by_dir, dtype=np.float64).ravel()
    active = gamma > float(freeze_eps)
    probs = probs * active.astype(np.float64)
    total = float(np.sum(probs))
    if total <= 0:
        if np.any(active):
            probs = active.astype(np.float64) / float(np.sum(active))
        else:
            probs = np.full((len(gamma),), 1.0 / float(max(1, len(gamma))), dtype=np.float64)
    else:
        probs = probs / total
    return probs


def _best_round_row(df: pd.DataFrame, metric_col: str) -> pd.Series:
    idx = int(df[metric_col].astype(float).idxmax())
    return df.loc[idx]


def _variant_health_row(
    *,
    dataset: str,
    dataset_group: str,
    variant: str,
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
        "dataset_group": dataset_group,
        "variant": variant,
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


def _prior_cleaning_comment(
    *,
    low_quality_dir_count: int,
    downweighted_dir_count: int,
    frozen_dir_count: int,
    score_signal: bool,
) -> str:
    if low_quality_dir_count <= 0 and downweighted_dir_count <= 0:
        return "prior_nearly_uniform_no_clear_cleaning_signal"
    if score_signal and frozen_dir_count > 0:
        return "clear_low_quality_direction_suppression"
    if score_signal:
        return "prior_downweights_low_quality_without_hard_freeze"
    if frozen_dir_count > 0:
        return "cleaning_attempt_present_but_signal_mixed"
    return "mild_prior_reweighting_signal"


def _run_curriculum_variant(
    *,
    dataset: str,
    dataset_group: str,
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
    base_direction_probs: np.ndarray,
    gamma_init: np.ndarray,
    prior_frozen_mask: np.ndarray,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    perf_rows: List[Dict[str, object]] = []
    health_rows: List[Dict[str, object]] = []
    budget_rows: List[Dict[str, object]] = []
    prior_rows: List[Dict[str, object]] = []

    gamma_by_dir = np.asarray(gamma_init, dtype=np.float64).copy()
    gamma_by_dir[np.asarray(prior_frozen_mask, dtype=bool)] = 0.0

    for round_id in range(1, int(args.n_rounds) + 1):
        if variant_name == "fisher_c0_plus_curriculum":
            direction_probs = _build_direction_probs_with_prior(
                fisher_probs=base_direction_probs,
                gamma_by_dir=gamma_by_dir,
                freeze_eps=float(args.curriculum_freeze_eps),
            )
        else:
            active = gamma_by_dir > float(args.curriculum_freeze_eps)
            if np.any(active):
                direction_probs = active.astype(np.float64) / float(np.sum(active))
            else:
                direction_probs = np.full((len(gamma_by_dir),), 1.0 / float(max(1, len(gamma_by_dir))), dtype=np.float64)

        gamma_before = gamma_by_dir.copy()
        X_curr, y_curr, tid_curr, src_curr, dir_curr, curr_aug_meta = _build_curriculum_aug_candidates(
            X_train=X_train_base,
            y_train=y_train_base,
            tid_train=tid_train,
            direction_bank=direction_bank,
            direction_probs=direction_probs,
            gamma_by_dir=gamma_before,
            multiplier=int(args.pia_multiplier),
            seed=int(seed + 500000 + round_id * 1009 + (17 if variant_name == "fisher_c0_plus_curriculum" else 0)),
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
            progress_prefix=f"[fisher-curriculum][{dataset}][seed={seed}][{variant_name}][round={round_id}][mech]",
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
            progress_prefix=f"[fisher-curriculum][{dataset}][seed={seed}][{variant_name}][round={round_id}]",
        )

        round_dir = os.path.join(seed_dir, f"{variant_name}_R{round_id}")
        _write_condition(
            round_dir,
            metrics_curr,
            {
                **common_meta,
                **train_meta_curr,
                "condition": f"{variant_name}_R{round_id}",
                "variant": variant_name,
                "round_id": int(round_id),
                "augmentation": curr_aug_meta,
                "direction_bank": {**bank_meta, "subset_size": int(args.subset_size)},
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
        dir_summary = _summarize_dir_profile(mech_curr.get("dir_profile", {}))
        note = (
            "fisher_prior_guided_cleaner_round"
            if variant_name == "fisher_c0_plus_curriculum" and frozen_dir_count > 0
            else "curriculum_round"
        )

        perf_rows.append(
            {
                "dataset": dataset,
                "dataset_group": dataset_group,
                "seed": int(seed),
                "variant": variant_name,
                "round_id": int(round_id),
                "acc": float(metrics_curr["trial_acc"]),
                "f1": float(metrics_curr["trial_macro_f1"]),
                "worst_dir_id": dir_summary["worst_dir_id"],
            }
        )
        health_rows.append(
                _variant_health_row(
                    dataset=dataset,
                    dataset_group=dataset_group,
                    variant=variant_name,
                    round_id=int(round_id),
                    usage_probs=np.asarray(
                        [
                            float(curr_aug_meta.get("mixing_stats", {}).get("direction_pick_fraction", {}).get(str(i), 0.0))
                            for i in range(int(args.k_dir))
                        ],
                    dtype=np.float64,
                ),
                mech=mech_curr,
                intrusion_by_dir=intrusion_by_dir,
                frozen_dir_count=frozen_dir_count,
                expanded_dir_count=expanded_dir_count,
                note=note,
            )
        )

        low_quality_count = int(np.sum(np.asarray(prior_frozen_mask, dtype=np.int64)))
        downweighted_count = int(np.sum((gamma_before > 0.0) & (gamma_before < float(args.curriculum_init_gamma) - 1e-12)))
        prior_rows.append(
            {
                "dataset": dataset,
                "dataset_group": dataset_group,
                "seed": int(seed),
                "variant": variant_name,
                "round_id": int(round_id),
                "low_quality_dir_count": low_quality_count,
                "downweighted_dir_count": downweighted_count,
                "frozen_dir_count": frozen_dir_count,
                "prior_cleaning_comment": _prior_cleaning_comment(
                    low_quality_dir_count=low_quality_count,
                    downweighted_dir_count=downweighted_count,
                    frozen_dir_count=frozen_dir_count,
                    score_signal=(low_quality_count > 0 or downweighted_count > 0),
                ),
            }
        )

        for dir_id in range(int(args.k_dir)):
            budget_rows.append(
                {
                    "dataset": dataset,
                    "dataset_group": dataset_group,
                    "seed": int(seed),
                    "variant": variant_name,
                    "round_id": int(round_id),
                    "dir_id": int(dir_id),
                    "gamma_before": float(gamma_before[dir_id]),
                    "gamma_after": float(gamma_after[dir_id]),
                    "direction_state": str(state_by_dir.get(dir_id, "hold")),
                    "usage_fraction": float(curr_aug_meta.get("mixing_stats", {}).get("direction_pick_fraction", {}).get(str(dir_id), 0.0)),
                    "margin_drop": float(maps["margin_drop_median"].get(dir_id, 0.0)),
                    "flip_rate": float(maps["flip_rate"].get(dir_id, 0.0)),
                    "intrusion": float(maps["intrusion"].get(dir_id, 0.0)),
                }
            )

        print(
            f"[fisher-curriculum][{dataset}][seed={seed}][{variant_name}][round={round_id}] "
            f"f1={metrics_curr['trial_macro_f1']:.4f} freeze={frozen_dir_count} expand={expanded_dir_count}",
            flush=True,
        )

    return perf_rows, health_rows, budget_rows, prior_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="natops,selfregulationscp1,fingermovements,har,handmovementdirection,uwavegesturelibrary",
    )
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--out-root", type=str, default="out/phase15_fisher_curriculum_20260321")
    parser.add_argument("--har-root", type=str, default=DEFAULT_HAR_ROOT)
    parser.add_argument("--natops-root", type=str, default=DEFAULT_NATOPS_ROOT)
    parser.add_argument("--fingermovements-root", type=str, default=DEFAULT_FINGERMOVEMENTS_ROOT)
    parser.add_argument("--selfregulationscp1-root", type=str, default=DEFAULT_SELFREGULATIONSCP1_ROOT)
    parser.add_argument("--handmovementdirection-root", type=str, default=DEFAULT_HANDMOVEMENTDIRECTION_ROOT)
    parser.add_argument("--uwavegesturelibrary-root", type=str, default=DEFAULT_UWAVEGESTURELIBRARY_ROOT)
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
    parser.add_argument("--fisher-beta", type=float, default=1.0)
    parser.add_argument("--fisher-knn-k", type=int, default=20)
    parser.add_argument("--fisher-boundary-quantile", type=float, default=0.30)
    parser.add_argument("--fisher-interior-quantile", type=float, default=0.70)
    parser.add_argument("--fisher-hetero-k", type=int, default=3)
    parser.add_argument("--fisher-low-quality-quantile", type=float, default=0.35)
    parser.add_argument("--fisher-downweight-quantile", type=float, default=0.55)
    parser.add_argument("--fisher-downweight-gamma-mult", type=float, default=0.70)
    parser.add_argument("--split-preview-n", type=int, default=5)
    args = parser.parse_args()

    datasets = [normalize_dataset_name(x) for x in _parse_csv_list(args.datasets)]
    for ds in datasets:
        if ds not in ALLOWED_DATASETS:
            raise ValueError(f"Unsupported dataset for fisher+curriculum probe: {ds}")
    if int(args.subset_size) != 1:
        raise ValueError("This probe currently locks subset_size=1 to stay aligned with current Step1B default.")

    seeds = _parse_seed_list(args.seeds)
    ensure_dir(args.out_root)

    fisher_cfg = FisherPIAConfig(
        knn_k=int(args.fisher_knn_k),
        interior_quantile=float(args.fisher_interior_quantile),
        boundary_quantile=float(args.fisher_boundary_quantile),
        hetero_k=int(args.fisher_hetero_k),
    )

    perf_rows: List[Dict[str, object]] = []
    health_rows: List[Dict[str, object]] = []
    prior_rows: List[Dict[str, object]] = []
    budget_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        dataset_group = _dataset_group(dataset)
        print(f"[fisher-curriculum][{dataset}] load", flush=True)
        all_trials = load_trials_for_dataset(
            dataset=dataset,
            processed_root=args.processed_root,
            stim_xlsx=args.stim_xlsx,
            har_root=args.har_root,
            natops_root=args.natops_root,
            fingermovements_root=args.fingermovements_root,
            selfregulationscp1_root=args.selfregulationscp1_root,
            handmovementdirection_root=args.handmovementdirection_root,
            uwavegesturelibrary_root=args.uwavegesturelibrary_root,
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
            print(f"[fisher-curriculum][{dataset}][seed={seed}] start", flush=True)
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
                progress_prefix=f"[fisher-curriculum][{dataset}][seed={seed}][baseline]",
            )

            direction_bank, bank_meta = _build_direction_bank_d1(
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
                direction_bank=direction_bank,
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
                progress_prefix=f"[fisher-curriculum][{dataset}][seed={seed}][mech_step1b]",
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
                progress_prefix=f"[fisher-curriculum][{dataset}][seed={seed}][step1b]",
            )
            _write_condition(
                os.path.join(seed_dir, "A_baseline"),
                metrics_base,
                {
                    "dataset": dataset,
                    "dataset_group": dataset_group,
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
                    "dataset_group": dataset_group,
                    "seed": int(seed),
                    **split_meta,
                    **train_meta_step1b,
                    "condition": "B_step1b",
                    "augmentation": step1b_aug_meta,
                    "direction_bank": {**bank_meta, "subset_size": int(args.subset_size)},
                    "mech": mech_step1b,
                },
            )

            perf_rows.extend(
                [
                    {
                        "dataset": dataset,
                        "dataset_group": dataset_group,
                        "seed": int(seed),
                        "variant": "baseline",
                        "round_id": 0,
                        "acc": float(metrics_base["trial_acc"]),
                        "f1": float(metrics_base["trial_macro_f1"]),
                    },
                    {
                        "dataset": dataset,
                        "dataset_group": dataset_group,
                        "seed": int(seed),
                        "variant": "single_round_step1b",
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
                    dataset_group=dataset_group,
                    variant="single_round_step1b",
                    round_id=0,
                    usage_probs=usage_probs_step1b,
                    mech=mech_step1b,
                    intrusion_by_dir=step1b_intrusion_by_dir,
                    frozen_dir_count=0,
                    expanded_dir_count=0,
                    note="equal_weight_single_round",
                )
            )

            class_terms, _ = compute_fisher_pia_terms(X_train_base, y_train_base, cfg=fisher_cfg)
            _, fisher_global_df = compute_safe_axis_scores(
                direction_bank,
                class_terms,
                beta=float(args.fisher_beta),
                gamma=0.0,
                include_approach=False,
                direction_score_mode="axis_level",
            )
            fisher_scores = fisher_global_df["revised_score"].to_numpy(dtype=np.float64)
            fisher_norm = _minmax_norm(fisher_scores, constant_fill=0.5)
            q_low = float(np.quantile(fisher_norm, float(args.fisher_low_quality_quantile))) if fisher_norm.size else 0.0
            q_down = float(np.quantile(fisher_norm, float(args.fisher_downweight_quantile))) if fisher_norm.size else 0.5
            prior_frozen_mask = fisher_norm <= q_low
            if np.all(prior_frozen_mask) and prior_frozen_mask.size:
                prior_frozen_mask[int(np.argmax(fisher_norm))] = False
            downweighted_mask = (~prior_frozen_mask) & (fisher_norm <= q_down)
            fisher_probs = _score_to_probs(np.where(prior_frozen_mask, np.min(fisher_scores) - 1.0, fisher_scores))
            gamma_init_fisher = np.full((int(args.k_dir),), float(args.curriculum_init_gamma), dtype=np.float64)
            gamma_init_fisher[downweighted_mask] *= float(args.fisher_downweight_gamma_mult)
            gamma_init_fisher[prior_frozen_mask] = 0.0

            common_meta = {
                "dataset": dataset,
                "dataset_group": dataset_group,
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
                "fisher_prior_config": {
                    "beta": float(args.fisher_beta),
                    "low_quality_quantile": float(args.fisher_low_quality_quantile),
                    "downweight_quantile": float(args.fisher_downweight_quantile),
                    "downweight_gamma_mult": float(args.fisher_downweight_gamma_mult),
                },
            }

            curr_perf, curr_health, curr_budget, curr_prior = _run_curriculum_variant(
                dataset=dataset,
                dataset_group=dataset_group,
                seed=int(seed),
                seed_dir=seed_dir,
                common_meta=common_meta,
                X_train_base=X_train_base,
                y_train_base=y_train_base,
                tid_train=tid_train,
                X_test=X_test,
                y_test=y_test,
                tid_test=tid_test,
                direction_bank=direction_bank,
                bank_meta=bank_meta,
                cap_seed=cap_seed,
                args=args,
                variant_name="multiround_curriculum",
                base_direction_probs=np.full((int(args.k_dir),), 1.0 / float(int(args.k_dir)), dtype=np.float64),
                gamma_init=np.full((int(args.k_dir),), float(args.curriculum_init_gamma), dtype=np.float64),
                prior_frozen_mask=np.zeros((int(args.k_dir),), dtype=bool),
            )
            fisher_perf, fisher_health, fisher_budget, fisher_prior = _run_curriculum_variant(
                dataset=dataset,
                dataset_group=dataset_group,
                seed=int(seed),
                seed_dir=seed_dir,
                common_meta=common_meta,
                X_train_base=X_train_base,
                y_train_base=y_train_base,
                tid_train=tid_train,
                X_test=X_test,
                y_test=y_test,
                tid_test=tid_test,
                direction_bank=direction_bank,
                bank_meta=bank_meta,
                cap_seed=cap_seed,
                args=args,
                variant_name="fisher_c0_plus_curriculum",
                base_direction_probs=fisher_probs,
                gamma_init=gamma_init_fisher,
                prior_frozen_mask=prior_frozen_mask,
            )

            perf_rows.extend(curr_perf)
            perf_rows.extend(fisher_perf)
            health_rows.extend(curr_health)
            health_rows.extend(fisher_health)
            budget_rows.extend(curr_budget)
            budget_rows.extend(fisher_budget)
            prior_rows.extend(curr_prior)
            prior_rows.extend(fisher_prior)

            prior_rows.append(
                {
                    "dataset": dataset,
                    "dataset_group": dataset_group,
                    "seed": int(seed),
                    "variant": "fisher_prior_init",
                    "round_id": 0,
                    "low_quality_dir_count": int(np.sum(prior_frozen_mask)),
                    "downweighted_dir_count": int(np.sum(downweighted_mask)),
                    "frozen_dir_count": int(np.sum(prior_frozen_mask)),
                    "prior_cleaning_comment": _prior_cleaning_comment(
                        low_quality_dir_count=int(np.sum(prior_frozen_mask)),
                        downweighted_dir_count=int(np.sum(downweighted_mask)),
                        frozen_dir_count=int(np.sum(prior_frozen_mask)),
                        score_signal=bool(fisher_scores.size),
                    ),
                }
            )

    perf_df = pd.DataFrame(perf_rows).sort_values(["dataset_group", "dataset", "variant", "seed", "round_id"]).reset_index(drop=True)
    perf_df.to_csv(os.path.join(args.out_root, "summary_per_seed.csv"), index=False)
    pd.DataFrame(budget_rows).sort_values(["dataset_group", "dataset", "variant", "seed", "round_id", "dir_id"]).to_csv(
        os.path.join(args.out_root, "fisher_curriculum_direction_budget_summary.csv"),
        index=False,
    )

    summary_rows: List[Dict[str, object]] = []
    health_summary_rows: List[Dict[str, object]] = []
    prior_summary_rows: List[Dict[str, object]] = []

    health_df = pd.DataFrame(health_rows)
    prior_df = pd.DataFrame(prior_rows)

    for dataset, df_ds in perf_df.groupby("dataset"):
        dataset_group = _dataset_group(dataset)
        df_base = df_ds[df_ds["variant"] == "baseline"]
        df_s1 = df_ds[df_ds["variant"] == "single_round_step1b"]
        df_curr = df_ds[df_ds["variant"] == "multiround_curriculum"]
        df_fish = df_ds[df_ds["variant"] == "fisher_c0_plus_curriculum"]
        best_curr = _best_round_row(df_curr.groupby("round_id", as_index=False)["f1"].mean(), "f1")
        best_fish = _best_round_row(df_fish.groupby("round_id", as_index=False)["f1"].mean(), "f1")
        best_curr_round = int(best_curr["round_id"])
        best_fish_round = int(best_fish["round_id"])

        base_acc = _summary_stats(df_base["acc"])
        base_f1 = _summary_stats(df_base["f1"])
        s1_acc = _summary_stats(df_s1["acc"])
        s1_f1 = _summary_stats(df_s1["f1"])
        curr_acc = _summary_stats(df_curr[df_curr["round_id"] == best_curr_round]["acc"])
        curr_f1 = _summary_stats(df_curr[df_curr["round_id"] == best_curr_round]["f1"])
        fish_acc = _summary_stats(df_fish[df_fish["round_id"] == best_fish_round]["acc"])
        fish_f1 = _summary_stats(df_fish[df_fish["round_id"] == best_fish_round]["f1"])
        delta_vs_curr = float(fish_f1["mean"] - curr_f1["mean"])
        summary_rows.append(
            {
                "dataset": dataset,
                "dataset_group": dataset_group,
                "baseline_acc": _format_mean_std(base_acc["mean"], base_acc["std"]),
                "baseline_f1": _format_mean_std(base_f1["mean"], base_f1["std"]),
                "single_round_step1b_acc": _format_mean_std(s1_acc["mean"], s1_acc["std"]),
                "single_round_step1b_f1": _format_mean_std(s1_f1["mean"], s1_f1["std"]),
                "multiround_curriculum_acc": _format_mean_std(curr_acc["mean"], curr_acc["std"]),
                "multiround_curriculum_f1": _format_mean_std(curr_f1["mean"], curr_f1["std"]),
                "fisher_c0_plus_curriculum_acc": _format_mean_std(fish_acc["mean"], fish_acc["std"]),
                "fisher_c0_plus_curriculum_f1": _format_mean_std(fish_f1["mean"], fish_f1["std"]),
                "best_curriculum_round": int(best_curr_round),
                "best_fisher_round": int(best_fish_round),
                "delta_vs_single_round_step1b": float(fish_f1["mean"] - s1_f1["mean"]),
                "delta_vs_multiround_curriculum": delta_vs_curr,
                "result_label": _result_label(delta_vs_curr),
            }
        )

        for variant, round_id in [
            ("single_round_step1b", 0),
            ("multiround_curriculum", best_curr_round),
            ("fisher_c0_plus_curriculum", best_fish_round),
        ]:
            df_h = health_df[
                (health_df["dataset"] == dataset)
                & (health_df["variant"] == variant)
                & (health_df["round_id"] == int(round_id))
            ]
            if df_h.empty:
                continue
            note_counts = df_h["direction_health_comment"].astype(str).value_counts().to_dict()
            health_summary_rows.append(
                {
                    "dataset": dataset,
                    "dataset_group": dataset_group,
                    "variant": variant,
                    "direction_usage_entropy": _format_mean_std(
                        float(df_h["direction_usage_entropy"].mean()),
                        float(df_h["direction_usage_entropy"].std()),
                    ),
                    "per_dir_margin_summary": " / ".join(df_h["per_dir_margin_summary"].astype(str).tolist()),
                    "per_dir_flip_summary": " / ".join(df_h["per_dir_flip_summary"].astype(str).tolist()),
                    "per_dir_intrusion_summary": " / ".join(df_h["per_dir_intrusion_summary"].astype(str).tolist()),
                    "worst_dir_summary": " / ".join(df_h["worst_dir_summary"].astype(str).tolist()),
                    "frozen_dir_count": _format_mean_std(
                        float(df_h["frozen_dir_count"].mean()),
                        float(df_h["frozen_dir_count"].std()),
                    ),
                    "expanded_dir_count": _format_mean_std(
                        float(df_h["expanded_dir_count"].mean()),
                        float(df_h["expanded_dir_count"].std()),
                    ),
                    "direction_health_comment": "; ".join([f"{k}:{v}" for k, v in sorted(note_counts.items())]),
                }
            )

        df_p = prior_df[prior_df["dataset"] == dataset]
        if not df_p.empty:
            init_rows = df_p[df_p["variant"] == "fisher_prior_init"]
            if init_rows.empty:
                init_rows = df_p[df_p["variant"] == "fisher_c0_plus_curriculum"]
            note_counts = init_rows["prior_cleaning_comment"].astype(str).value_counts().to_dict()
            prior_summary_rows.append(
                {
                    "dataset": dataset,
                    "dataset_group": dataset_group,
                    "low_quality_dir_count": _format_mean_std(
                        float(init_rows["low_quality_dir_count"].mean()),
                        float(init_rows["low_quality_dir_count"].std()),
                    ),
                    "downweighted_dir_count": _format_mean_std(
                        float(init_rows["downweighted_dir_count"].mean()),
                        float(init_rows["downweighted_dir_count"].std()),
                    ),
                    "frozen_dir_count": _format_mean_std(
                        float(init_rows["frozen_dir_count"].mean()),
                        float(init_rows["frozen_dir_count"].std()),
                    ),
                    "prior_cleaning_comment": "; ".join([f"{k}:{v}" for k, v in sorted(note_counts.items())]),
                }
            )

    perf_summary_df = pd.DataFrame(summary_rows).sort_values(["dataset_group", "dataset"]).reset_index(drop=True)
    perf_summary_df.to_csv(os.path.join(args.out_root, "fisher_curriculum_performance_summary.csv"), index=False)
    pd.DataFrame(health_summary_rows).sort_values(["dataset_group", "dataset", "variant"]).to_csv(
        os.path.join(args.out_root, "fisher_curriculum_direction_health_summary.csv"),
        index=False,
    )
    pd.DataFrame(prior_summary_rows).sort_values(["dataset_group", "dataset"]).to_csv(
        os.path.join(args.out_root, "fisher_curriculum_prior_cleaning_summary.csv"),
        index=False,
    )

    core_df = perf_summary_df[perf_summary_df["dataset_group"] == "core_decision_sets"]
    improved_core = int(np.sum(core_df["delta_vs_multiround_curriculum"].astype(float) > 1e-6)) if not core_df.empty else 0
    if improved_core >= 2:
        readout = "值得进入下一阶段"
    elif improved_core >= 1:
        readout = "继续作为探索线"
    else:
        readout = "当前方案暂缓"

    best_dataset = "current evidence insufficient"
    if not perf_summary_df.empty:
        best_idx = int(perf_summary_df["delta_vs_multiround_curriculum"].astype(float).idxmax())
        best_dataset = str(perf_summary_df.loc[best_idx, "dataset"])

    bridge_hint = "NATOPS" if "natops" in perf_summary_df["dataset"].tolist() else best_dataset
    conclusion_lines = [
        "# Fisher/C0 + Curriculum Conclusion",
        "",
        "更新时间：2026-03-21",
        "",
        "身份：`independent target-cleaning upgrade line`",
        "",
        "- `not for Phase15 mainline freeze table`",
        "- `not connected back to bridge in this round`",
        "- `goal = test whether upstream direction-bank purification improves curriculum target quality`",
        "",
        "## Dataset Layers",
        "",
        "- `core_decision_sets`: natops, selfregulationscp1, fingermovements",
        "- `extension_sets`: har, handmovementdirection, uwavegesturelibrary",
        "",
        "## Snapshot",
        "",
    ]
    for _, row in perf_summary_df.iterrows():
        conclusion_lines.append(
            f"- `{row['dataset']}` [{row['dataset_group']}]: "
            f"curriculum={row['multiround_curriculum_f1']}, "
            f"fisher+curriculum={row['fisher_c0_plus_curriculum_f1']}, "
            f"delta_vs_curriculum={float(row['delta_vs_multiround_curriculum']):+.4f}, "
            f"label=`{row['result_label']}`"
        )
    conclusion_lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- 当前是否应继续推进 Fisher/C0 + curriculum：`{readout}`",
            f"- 它是否已经超过纯 curriculum：`核心集 improved={improved_core}/3`",
            "- 它的收益更集中在哪类数据集：`见 core_decision_sets vs extension_sets 分层结果`",
            f"- 若后续重回 bridge，优先 very small pilot 数据集：`{bridge_hint}`",
            "",
            "## Interpretation Guardrails",
            "",
            "- 当前在看的是上游 target 质量，不是 bridge 层本身。",
            "- 若核心集没有稳定改善，则更像 curriculum 已接近上限或 Fisher/C0 净化仍不足。",
            f"- 当前最高增量数据集：`{best_dataset}`",
        ]
    )
    with open(os.path.join(args.out_root, "fisher_curriculum_conclusion.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(conclusion_lines) + "\n")


if __name__ == "__main__":
    main()
