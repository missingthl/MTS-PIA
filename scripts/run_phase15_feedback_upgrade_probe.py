#!/usr/bin/env python
"""Phase15 PIA feedback-upgrade probe on small fixed-split datasets.

This script is diagnostic-only and does NOT modify Phase15 mainline freeze.

Variants:
- baseline: z-space + LinearSVC
- step1b: current equal-weight Step1B direct train
- fisher_c0: Step1B with Fisher/C0 safe-axis score as direction sampling prior
- feedback_weighting: Step1B with offline direction credit weighting derived from
  Step1B mechanism statistics
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.trial_dataset_factory import (  # noqa: E402
    DEFAULT_BANDS_EEG,
    DEFAULT_FINGERMOVEMENTS_ROOT,
    DEFAULT_HAR_ROOT,
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
    write_json,
)
from scripts.fisher_pia_utils import (  # noqa: E402
    FisherPIAConfig,
    compute_fisher_pia_terms,
    compute_safe_axis_scores,
)
from scripts.run_phase15_mainline_freeze import (  # noqa: E402
    _make_protocol_split,
    _summarize_dir_profile,
)
from scripts.run_phase15_step1a_maxplane import _fit_eval_linearsvc  # noqa: E402
from scripts.run_phase15_step1b_multidir_matrix import (  # noqa: E402
    _build_direction_bank_d1,
    _build_multidir_aug_candidates,
    _compute_mech_metrics,
    _write_condition,
)


FIXED_SMALL_DATASETS = {
    "har",
    "selfregulationscp1",
    "fingermovements",
    "natops",
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


def _stable_tid_hash(tid: object) -> int:
    return abs(hash(str(tid))) % 1_000_003


def _ordered_unique(values: Iterable[object]) -> List[object]:
    seen = set()
    out: List[object] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _summary_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64).ravel()
    if x.size == 0:
        return np.asarray([], dtype=np.float64)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or abs(xmax - xmin) <= 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - xmin) / (xmax - xmin)


def _entropy_from_probs(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _score_to_probs(scores: np.ndarray, *, eps: float = 1e-3) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size == 0:
        return np.asarray([], dtype=np.float64)
    shifted = s - float(np.min(s)) + float(eps)
    total = float(np.sum(shifted))
    if not np.isfinite(total) or total <= 0:
        return np.full((len(s),), 1.0 / float(len(s)), dtype=np.float64)
    return (shifted / total).astype(np.float64)


def _format_mean_std(mean: float, std: float) -> str:
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def _dict_summary_string(values: Dict[int, float], *, fmt: str = ".4f") -> str:
    if not values:
        return "n/a"
    parts = [f"{int(k)}:{format(float(v), fmt)}" for k, v in sorted(values.items())]
    return "|".join(parts)


def _build_weighted_aug_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tid_train: np.ndarray,
    *,
    direction_bank: np.ndarray,
    direction_probs: np.ndarray,
    gamma: float,
    multiplier: int,
    seed: int,
    weighting_mode: str,
    score_by_dir: Dict[int, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    tid_arr = np.asarray(tid_train)
    y_arr = np.asarray(y_train).astype(int).ravel()
    k_dir = int(direction_bank.shape[0])
    probs = np.asarray(direction_probs, dtype=np.float64).ravel()
    if probs.shape[0] != k_dir:
        raise ValueError("direction_probs size mismatch with direction_bank")
    probs = probs / np.sum(probs)

    aug_X_parts: List[np.ndarray] = []
    aug_y_parts: List[np.ndarray] = []
    aug_tid_parts: List[np.ndarray] = []
    aug_src_parts: List[np.ndarray] = []
    aug_dir_parts: List[np.ndarray] = []
    aug_count_per_trial: Dict[str, int] = {}
    dir_pick_count = np.zeros((k_dir,), dtype=np.int64)

    for tid in sorted(_ordered_unique(tid_arr.tolist())):
        idx = np.where(tid_arr == tid)[0]
        if idx.size == 0:
            aug_count_per_trial[str(tid)] = 0
            continue
        X_tid = np.asarray(X_train[idx], dtype=np.float32)
        y_tid = np.asarray(y_arr[idx], dtype=np.int64)
        added = 0
        for m in range(max(0, int(multiplier))):
            rs = np.random.RandomState(int(seed + m * 1009 + _stable_tid_hash(tid)))
            dir_ids = rs.choice(k_dir, size=int(X_tid.shape[0]), replace=True, p=probs).astype(np.int64)
            signs = rs.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=int(X_tid.shape[0])).astype(np.float32)
            X_aug = (X_tid + float(gamma) * signs[:, None] * direction_bank[dir_ids]).astype(np.float32)

            aug_X_parts.append(X_aug)
            aug_y_parts.append(y_tid.copy())
            aug_tid_parts.append(np.asarray([tid] * len(idx)))
            aug_src_parts.append(X_tid.copy())
            aug_dir_parts.append(dir_ids.copy())
            dir_pick_count += np.bincount(dir_ids, minlength=k_dir).astype(np.int64)
            added += len(idx)
        aug_count_per_trial[str(tid)] = int(added)

    if aug_X_parts:
        X_aug_all = np.vstack(aug_X_parts).astype(np.float32)
        y_aug_all = np.concatenate(aug_y_parts).astype(np.int64)
        tid_aug_all = np.concatenate(aug_tid_parts)
        src_aug_all = np.vstack(aug_src_parts).astype(np.float32)
        dir_aug_all = np.concatenate(aug_dir_parts).astype(np.int64)
    else:
        X_aug_all = np.empty((0, X_train.shape[1]), dtype=np.float32)
        y_aug_all = np.empty((0,), dtype=np.int64)
        tid_aug_all = np.empty((0,), dtype=object)
        src_aug_all = np.empty((0, X_train.shape[1]), dtype=np.float32)
        dir_aug_all = np.empty((0,), dtype=np.int64)

    total_picks = int(np.sum(dir_pick_count))
    pick_frac = (
        {str(i): float(dir_pick_count[i] / total_picks) for i in range(k_dir)}
        if total_picks > 0
        else {str(i): 0.0 for i in range(k_dir)}
    )
    aug_vals = (
        np.asarray(list(aug_count_per_trial.values()), dtype=np.float64)
        if aug_count_per_trial
        else np.asarray([], dtype=np.float64)
    )
    meta = {
        "aug_total_count": int(len(y_aug_all)),
        "aug_count_per_trial": aug_count_per_trial,
        "aug_per_trial_mean": float(np.mean(aug_vals)) if aug_vals.size else 0.0,
        "aug_per_trial_std": float(np.std(aug_vals)) if aug_vals.size else 0.0,
        "gamma": float(gamma),
        "k_dir": int(k_dir),
        "subset_size": 1,
        "weighting_mode": str(weighting_mode),
        "mixing_stats": {
            "mean_abs_ai": 1.0,
            "avg_subset_size": 1.0,
            "direction_pick_fraction": pick_frac,
        },
        "direction_probs": {str(i): float(probs[i]) for i in range(k_dir)},
        "direction_usage_entropy": float(_entropy_from_probs(probs)),
        "direction_score_by_dir": (
            {str(int(k)): float(v) for k, v in sorted(score_by_dir.items())}
            if score_by_dir
            else None
        ),
    }
    return X_aug_all, y_aug_all, tid_aug_all, src_aug_all, dir_aug_all, meta


def _compute_direction_intrusion(
    *,
    X_anchor: np.ndarray,
    y_anchor: np.ndarray,
    X_aug_accepted: np.ndarray,
    y_aug_accepted: np.ndarray,
    dir_accepted: np.ndarray,
    seed: int,
    knn_k: int,
    max_eval: int,
) -> Dict[int, float]:
    Xa = np.asarray(X_aug_accepted, dtype=np.float32)
    ya = np.asarray(y_aug_accepted).astype(int).ravel()
    da = np.asarray(dir_accepted).astype(int).ravel()
    Xr = np.asarray(X_anchor, dtype=np.float32)
    yr = np.asarray(y_anchor).astype(int).ravel()
    if Xa.size == 0 or Xr.size == 0:
        return {}
    rs = np.random.RandomState(int(seed) + 9103)
    if Xa.shape[0] > int(max_eval):
        idx = np.sort(rs.choice(Xa.shape[0], size=int(max_eval), replace=False))
        Xa = Xa[idx]
        ya = ya[idx]
        da = da[idx]
    k_eff = int(min(max(1, int(knn_k)), len(yr)))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(Xr)
    nn_idx = nn.kneighbors(Xa, return_distance=False)
    y_nb = yr[nn_idx]
    intrusion = 1.0 - np.mean(y_nb == ya[:, None], axis=1).astype(np.float64)
    out: Dict[int, float] = {}
    for did in sorted(np.unique(da).tolist()):
        mask = da == int(did)
        out[int(did)] = float(np.mean(intrusion[mask])) if np.any(mask) else 0.0
    return out


def _mech_dir_maps(mech: Dict[str, object], *, intrusion_by_dir: Dict[int, float] | None = None) -> Dict[str, Dict[int, float]]:
    profile = mech.get("dir_profile", {})
    if not isinstance(profile, dict):
        profile = {}
    usage: Dict[int, float] = {}
    acc: Dict[int, float] = {}
    flip: Dict[int, float] = {}
    margin: Dict[int, float] = {}
    intrusion: Dict[int, float] = {}
    for k, row in profile.items():
        if not isinstance(row, dict):
            continue
        did = int(k)
        usage[did] = float(row.get("usage", 0.0))
        acc[did] = float(row.get("accept_rate", 0.0))
        flip[did] = float(row.get("flip_rate", 0.0))
        margin[did] = float(row.get("margin_drop_median", 0.0))
        if intrusion_by_dir and did in intrusion_by_dir:
            intrusion[did] = float(intrusion_by_dir[did])
    return {
        "usage": usage,
        "accept_rate": acc,
        "flip_rate": flip,
        "margin_drop_median": margin,
        "intrusion": intrusion,
    }


def _feedback_credit_from_step1b(
    mech_step1b: Dict[str, object],
    *,
    intrusion_by_dir: Dict[int, float],
) -> Tuple[np.ndarray, Dict[int, float]]:
    profile = mech_step1b.get("dir_profile", {})
    if not isinstance(profile, dict) or not profile:
        return np.asarray([], dtype=np.float64), {}
    dir_ids = sorted(int(k) for k in profile.keys())
    margin = np.asarray([float(profile[str(i)].get("margin_drop_median", 0.0)) for i in dir_ids], dtype=np.float64)
    flip = np.asarray([float(profile[str(i)].get("flip_rate", 0.0)) for i in dir_ids], dtype=np.float64)
    intrusion = np.asarray([float(intrusion_by_dir.get(int(i), 0.0)) for i in dir_ids], dtype=np.float64)

    margin_n = _minmax_norm(margin)
    flip_good = 1.0 - _minmax_norm(flip)
    intr_good = 1.0 - _minmax_norm(intrusion)
    credit = (margin_n + flip_good + intr_good) / 3.0
    credit_map = {int(did): float(credit[idx]) for idx, did in enumerate(dir_ids)}
    return credit.astype(np.float64), credit_map


def _variant_health_row(
    *,
    dataset: str,
    variant: str,
    mech: Dict[str, object],
    aug_meta: Dict[str, object] | None,
    intrusion_by_dir: Dict[int, float],
    note: str,
) -> Dict[str, object]:
    maps = _mech_dir_maps(mech, intrusion_by_dir=intrusion_by_dir)
    dir_summary = _summarize_dir_profile(mech.get("dir_profile", {}))
    usage_fracs = np.asarray(list(maps["usage"].values()), dtype=np.float64)
    if usage_fracs.size == 0 and aug_meta is not None:
        frac_map = aug_meta.get("mixing_stats", {}).get("direction_pick_fraction", {})
        if isinstance(frac_map, dict):
            usage_fracs = np.asarray([float(v) for _, v in sorted(frac_map.items(), key=lambda kv: int(kv[0]))], dtype=np.float64)
    return {
        "dataset": dataset,
        "variant": variant,
        "direction_usage_entropy": float(_entropy_from_probs(usage_fracs)),
        "per_dir_margin_summary": _dict_summary_string(maps["margin_drop_median"]),
        "per_dir_flip_summary": _dict_summary_string(maps["flip_rate"]),
        "per_dir_intrusion_summary": _dict_summary_string(maps["intrusion"]),
        "worst_dir_id": dir_summary["worst_dir_id"],
        "worst_dir_metric_summary": dir_summary["dir_profile_summary"],
        "note": note,
    }


def _result_label(best_delta_vs_step1b: float) -> str:
    if best_delta_vs_step1b > 1e-6:
        return "direction_upgrade_positive"
    if best_delta_vs_step1b < -1e-6:
        return "direction_upgrade_negative"
    return "direction_upgrade_neutral"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="har,selfregulationscp1,fingermovements,natops")
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--out-root", type=str, default="out/phase15_feedback_upgrade_20260320")
    parser.add_argument("--har-root", type=str, default=DEFAULT_HAR_ROOT)
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
    parser.add_argument("--fisher-beta", type=float, default=1.0)
    parser.add_argument("--fisher-knn-k", type=int, default=20)
    parser.add_argument("--fisher-boundary-quantile", type=float, default=0.30)
    parser.add_argument("--fisher-interior-quantile", type=float, default=0.70)
    parser.add_argument("--fisher-hetero-k", type=int, default=3)
    parser.add_argument("--feedback-enabled", action="store_true")
    parser.add_argument("--split-preview-n", type=int, default=5)
    args = parser.parse_args()

    datasets = [normalize_dataset_name(x) for x in _parse_csv_list(args.datasets)]
    for ds in datasets:
        if ds not in FIXED_SMALL_DATASETS:
            raise ValueError(f"Unsupported dataset for feedback probe: {ds}")
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

    for dataset in datasets:
        print(f"[feedback-upgrade][{dataset}] load", flush=True)
        all_trials = load_trials_for_dataset(
            dataset=dataset,
            processed_root=args.processed_root,
            stim_xlsx=args.stim_xlsx,
            har_root=args.har_root,
            natops_root=args.natops_root,
            fingermovements_root=args.fingermovements_root,
            selfregulationscp1_root=args.selfregulationscp1_root,
        )
        bands_spec = resolve_band_spec(dataset, args.bands)
        bands = parse_band_spec(bands_spec)

        dataset_dir = os.path.join(args.out_root, dataset)
        ensure_dir(dataset_dir)
        dataset_perf_rows: List[Dict[str, object]] = []
        dataset_health_rows: List[Dict[str, object]] = []

        for seed in seeds:
            print(f"[feedback-upgrade][{dataset}][seed={seed}] start", flush=True)
            train_trials, test_trials, split_meta = _make_protocol_split(dataset, all_trials)
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
                f"[feedback-upgrade][{dataset}][seed={seed}] feat "
                f"train={tuple(X_train_base.shape)} test={tuple(X_test.shape)}",
                flush=True,
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
                progress_prefix=f"[feedback-upgrade][{dataset}][seed={seed}][baseline]",
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
                progress_prefix=f"[feedback-upgrade][{dataset}][seed={seed}][mech_step1b]",
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
                progress_prefix=f"[feedback-upgrade][{dataset}][seed={seed}][step1b]",
            )

            class_terms, fisher_terms_meta = compute_fisher_pia_terms(X_train_base, y_train_base, cfg=fisher_cfg)
            _, fisher_global_df = compute_safe_axis_scores(
                direction_bank,
                class_terms,
                beta=float(args.fisher_beta),
                gamma=0.0,
                include_approach=False,
                direction_score_mode="axis_level",
            )
            fisher_scores = {
                int(row["direction_id"]): float(row["revised_score"])
                for _, row in fisher_global_df.iterrows()
            }
            fisher_probs = _score_to_probs(fisher_global_df["revised_score"].to_numpy(dtype=np.float64))
            X_fisher, y_fisher, tid_fisher, src_fisher, dir_fisher, fisher_aug_meta = _build_weighted_aug_candidates(
                X_train=X_train_base,
                y_train=y_train_base,
                tid_train=tid_train,
                direction_bank=direction_bank,
                direction_probs=fisher_probs,
                gamma=float(args.pia_gamma),
                multiplier=int(args.pia_multiplier),
                seed=int(seed + 200000 + args.k_dir * 131),
                weighting_mode="fisher_c0_safe_axis",
                score_by_dir=fisher_scores,
            )
            mech_fisher = _compute_mech_metrics(
                X_train_real=X_train_base,
                y_train_real=y_train_base,
                X_aug_generated=X_fisher,
                y_aug_generated=y_fisher,
                X_aug_accepted=X_fisher,
                y_aug_accepted=y_fisher,
                X_src_accepted=src_fisher,
                dir_generated=dir_fisher,
                dir_accepted=dir_fisher,
                seed=int(seed),
                linear_c=float(args.linear_c),
                class_weight=args.linear_class_weight,
                linear_max_iter=int(args.linear_max_iter),
                knn_k=int(args.mech_knn_k),
                max_aug_for_mech=int(args.mech_max_aug_for_metrics),
                max_real_knn_ref=int(args.mech_max_real_knn_ref),
                max_real_knn_query=int(args.mech_max_real_knn_query),
                progress_prefix=f"[feedback-upgrade][{dataset}][seed={seed}][mech_fisher]",
            )
            fisher_intrusion_by_dir = _compute_direction_intrusion(
                X_anchor=X_train_base,
                y_anchor=y_train_base,
                X_aug_accepted=X_fisher,
                y_aug_accepted=y_fisher,
                dir_accepted=dir_fisher,
                seed=int(seed),
                knn_k=int(args.mech_knn_k),
                max_eval=int(args.mech_max_aug_for_metrics),
            )
            X_train_fisher = np.vstack([X_train_base, X_fisher]) if len(y_fisher) else X_train_base.copy()
            y_train_fisher = np.concatenate([y_train_base, y_fisher]) if len(y_fisher) else y_train_base.copy()
            tid_train_fisher = np.concatenate([tid_train, tid_fisher]) if len(y_fisher) else tid_train.copy()
            is_aug_fisher = (
                np.concatenate([np.zeros((len(y_train_base),), dtype=bool), np.ones((len(y_fisher),), dtype=bool)])
                if len(y_fisher)
                else np.zeros((len(y_train_base),), dtype=bool)
            )
            metrics_fisher, train_meta_fisher = _fit_eval_linearsvc(
                X_train_fisher,
                y_train_fisher,
                tid_train_fisher,
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
                is_aug_train=is_aug_fisher,
                progress_prefix=f"[feedback-upgrade][{dataset}][seed={seed}][fisher]",
            )

            metrics_feedback = None
            train_meta_feedback = None
            mech_feedback = None
            feedback_aug_meta = None
            feedback_intrusion_by_dir: Dict[int, float] = {}
            feedback_credit_map: Dict[int, float] = {}
            if bool(args.feedback_enabled):
                feedback_credit_vec, feedback_credit_map = _feedback_credit_from_step1b(
                    mech_step1b,
                    intrusion_by_dir=step1b_intrusion_by_dir,
                )
                if feedback_credit_vec.size:
                    feedback_probs = _score_to_probs(feedback_credit_vec)
                    X_feedback, y_feedback, tid_feedback, src_feedback, dir_feedback, feedback_aug_meta = _build_weighted_aug_candidates(
                        X_train=X_train_base,
                        y_train=y_train_base,
                        tid_train=tid_train,
                        direction_bank=direction_bank,
                        direction_probs=feedback_probs,
                        gamma=float(args.pia_gamma),
                        multiplier=int(args.pia_multiplier),
                        seed=int(seed + 300000 + args.k_dir * 149),
                        weighting_mode="offline_direction_credit",
                        score_by_dir=feedback_credit_map,
                    )
                    mech_feedback = _compute_mech_metrics(
                        X_train_real=X_train_base,
                        y_train_real=y_train_base,
                        X_aug_generated=X_feedback,
                        y_aug_generated=y_feedback,
                        X_aug_accepted=X_feedback,
                        y_aug_accepted=y_feedback,
                        X_src_accepted=src_feedback,
                        dir_generated=dir_feedback,
                        dir_accepted=dir_feedback,
                        seed=int(seed),
                        linear_c=float(args.linear_c),
                        class_weight=args.linear_class_weight,
                        linear_max_iter=int(args.linear_max_iter),
                        knn_k=int(args.mech_knn_k),
                        max_aug_for_mech=int(args.mech_max_aug_for_metrics),
                        max_real_knn_ref=int(args.mech_max_real_knn_ref),
                        max_real_knn_query=int(args.mech_max_real_knn_query),
                        progress_prefix=f"[feedback-upgrade][{dataset}][seed={seed}][mech_feedback]",
                    )
                    feedback_intrusion_by_dir = _compute_direction_intrusion(
                        X_anchor=X_train_base,
                        y_anchor=y_train_base,
                        X_aug_accepted=X_feedback,
                        y_aug_accepted=y_feedback,
                        dir_accepted=dir_feedback,
                        seed=int(seed),
                        knn_k=int(args.mech_knn_k),
                        max_eval=int(args.mech_max_aug_for_metrics),
                    )
                    X_train_feedback = np.vstack([X_train_base, X_feedback]) if len(y_feedback) else X_train_base.copy()
                    y_train_feedback = np.concatenate([y_train_base, y_feedback]) if len(y_feedback) else y_train_base.copy()
                    tid_train_feedback = np.concatenate([tid_train, tid_feedback]) if len(y_feedback) else tid_train.copy()
                    is_aug_feedback = (
                        np.concatenate([np.zeros((len(y_train_base),), dtype=bool), np.ones((len(y_feedback),), dtype=bool)])
                        if len(y_feedback)
                        else np.zeros((len(y_train_base),), dtype=bool)
                    )
                    metrics_feedback, train_meta_feedback = _fit_eval_linearsvc(
                        X_train_feedback,
                        y_train_feedback,
                        tid_train_feedback,
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
                        is_aug_train=is_aug_feedback,
                        progress_prefix=f"[feedback-upgrade][{dataset}][seed={seed}][feedback]",
                    )

            seed_dir = os.path.join(dataset_dir, f"seed{seed}")
            ensure_dir(seed_dir)
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
                    "augmentation": step1b_aug_meta,
                    "direction_bank": {**bank_meta, "subset_size": int(args.subset_size)},
                    "final_accept_rate": 1.0,
                    "mech": mech_step1b,
                },
            )
            fisher_meta = {
                **common_meta,
                **train_meta_fisher,
                "condition": "C_fisher_c0_weighted",
                "augmentation": fisher_aug_meta,
                "direction_bank": {**bank_meta, "subset_size": int(args.subset_size)},
                "final_accept_rate": 1.0,
                "fisher_terms_meta": fisher_terms_meta,
                "fisher_beta": float(args.fisher_beta),
                "fisher_direction_scores": fisher_scores,
                "mech": mech_fisher,
            }
            _write_condition(
                os.path.join(seed_dir, "C_fisher_c0_weighted"),
                metrics_fisher,
                fisher_meta,
            )
            pd.DataFrame(
                {
                    "direction_id": fisher_global_df["direction_id"].astype(int),
                    "revised_score": fisher_global_df["revised_score"].astype(float),
                    "direction_prob": fisher_probs.astype(float),
                }
            ).to_csv(os.path.join(seed_dir, "C_fisher_c0_weighted", "direction_weight_table.csv"), index=False)

            if metrics_feedback is not None and train_meta_feedback is not None and mech_feedback is not None and feedback_aug_meta is not None:
                _write_condition(
                    os.path.join(seed_dir, "D_feedback_weighted"),
                    metrics_feedback,
                    {
                        **common_meta,
                        **train_meta_feedback,
                        "condition": "D_feedback_weighted",
                        "augmentation": feedback_aug_meta,
                        "direction_bank": {**bank_meta, "subset_size": int(args.subset_size)},
                        "feedback_credit_by_dir": feedback_credit_map,
                        "final_accept_rate": 1.0,
                        "mech": mech_feedback,
                    },
                )
                probs_feedback = feedback_aug_meta.get("direction_probs", {})
                pd.DataFrame(
                    {
                        "direction_id": [int(k) for k in sorted(probs_feedback.keys(), key=int)],
                        "credit_score": [float(feedback_credit_map.get(int(k), 0.0)) for k in sorted(probs_feedback.keys(), key=int)],
                        "direction_prob": [float(probs_feedback[k]) for k in sorted(probs_feedback.keys(), key=int)],
                    }
                ).to_csv(os.path.join(seed_dir, "D_feedback_weighted", "direction_weight_table.csv"), index=False)

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
                "fisher_c0_acc": float(metrics_fisher["trial_acc"]),
                "fisher_c0_f1": float(metrics_fisher["trial_macro_f1"]),
                "feedback_weighting_acc": float(metrics_feedback["trial_acc"]) if metrics_feedback else None,
                "feedback_weighting_f1": float(metrics_feedback["trial_macro_f1"]) if metrics_feedback else None,
            }
            dataset_perf_rows.append(perf_row)
            perf_rows.append(perf_row)

            dataset_health_rows.extend(
                [
                    _variant_health_row(
                        dataset=dataset,
                        variant="step1b",
                        mech=mech_step1b,
                        aug_meta=step1b_aug_meta,
                        intrusion_by_dir=step1b_intrusion_by_dir,
                        note="equal_weight_step1b",
                    ),
                    _variant_health_row(
                        dataset=dataset,
                        variant="fisher_c0",
                        mech=mech_fisher,
                        aug_meta=fisher_aug_meta,
                        intrusion_by_dir=fisher_intrusion_by_dir,
                        note="offline_safe_axis_weighted_sampling",
                    ),
                ]
            )
            if metrics_feedback is not None and mech_feedback is not None and feedback_aug_meta is not None:
                dataset_health_rows.append(
                    _variant_health_row(
                        dataset=dataset,
                        variant="feedback_weighting",
                        mech=mech_feedback,
                        aug_meta=feedback_aug_meta,
                        intrusion_by_dir=feedback_intrusion_by_dir,
                        note="offline_credit_from_step1b_mech",
                    )
                )

            print(
                f"[feedback-upgrade][{dataset}][seed={seed}] "
                f"A={metrics_a['trial_macro_f1']:.4f} "
                f"B={metrics_step1b['trial_macro_f1']:.4f} "
                f"C={metrics_fisher['trial_macro_f1']:.4f} "
                f"D={metrics_feedback['trial_macro_f1']:.4f}" if metrics_feedback else
                f"[feedback-upgrade][{dataset}][seed={seed}] "
                f"A={metrics_a['trial_macro_f1']:.4f} "
                f"B={metrics_step1b['trial_macro_f1']:.4f} "
                f"C={metrics_fisher['trial_macro_f1']:.4f}",
                flush=True,
            )

        df_seed = pd.DataFrame(dataset_perf_rows).sort_values(["dataset", "seed"]).reset_index(drop=True)
        df_seed.to_csv(os.path.join(dataset_dir, "summary_per_seed.csv"), index=False)
        df_health_seed = pd.DataFrame(dataset_health_rows).reset_index(drop=True)
        df_health_seed.to_csv(os.path.join(dataset_dir, "direction_health_per_seed.csv"), index=False)

    perf_df = pd.DataFrame(perf_rows).sort_values(["dataset", "seed"]).reset_index(drop=True)
    perf_df.to_csv(os.path.join(args.out_root, "summary_per_seed.csv"), index=False)

    health_df = pd.DataFrame(health_rows) if health_rows else pd.DataFrame()
    if health_df.empty:
        # Gather from per-dataset buffers if not already appended.
        all_health: List[pd.DataFrame] = []
        for ds in datasets:
            p = os.path.join(args.out_root, ds, "direction_health_per_seed.csv")
            if os.path.exists(p):
                all_health.append(pd.read_csv(p))
        health_df = pd.concat(all_health, axis=0, ignore_index=True) if all_health else pd.DataFrame()

    perf_summary_rows: List[Dict[str, object]] = []
    for dataset, df_ds in perf_df.groupby("dataset"):
        base_acc = _summary_stats(df_ds["baseline_acc"].tolist())
        base_f1 = _summary_stats(df_ds["baseline_f1"].tolist())
        s1_acc = _summary_stats(df_ds["step1b_acc"].tolist())
        s1_f1 = _summary_stats(df_ds["step1b_f1"].tolist())
        fc_acc = _summary_stats(df_ds["fisher_c0_acc"].tolist())
        fc_f1 = _summary_stats(df_ds["fisher_c0_f1"].tolist())
        row = {
            "dataset": dataset,
            "baseline_acc": _format_mean_std(base_acc["mean"], base_acc["std"]),
            "baseline_f1": _format_mean_std(base_f1["mean"], base_f1["std"]),
            "step1b_acc": _format_mean_std(s1_acc["mean"], s1_acc["std"]),
            "step1b_f1": _format_mean_std(s1_f1["mean"], s1_f1["std"]),
            "fisher_c0_acc": _format_mean_std(fc_acc["mean"], fc_acc["std"]),
            "fisher_c0_f1": _format_mean_std(fc_f1["mean"], fc_f1["std"]),
            "fisher_c0_delta_vs_step1b_acc": float(fc_acc["mean"] - s1_acc["mean"]),
            "fisher_c0_delta_vs_step1b_f1": float(fc_f1["mean"] - s1_f1["mean"]),
            "feedback_weighting_acc": None,
            "feedback_weighting_f1": None,
            "feedback_weighting_delta_vs_step1b_acc": None,
            "feedback_weighting_delta_vs_step1b_f1": None,
        }
        deltas = [float(fc_f1["mean"] - s1_f1["mean"])]
        if "feedback_weighting_f1" in df_ds.columns and df_ds["feedback_weighting_f1"].notna().any():
            fb_acc = _summary_stats(df_ds["feedback_weighting_acc"].dropna().tolist())
            fb_f1 = _summary_stats(df_ds["feedback_weighting_f1"].dropna().tolist())
            row["feedback_weighting_acc"] = _format_mean_std(fb_acc["mean"], fb_acc["std"])
            row["feedback_weighting_f1"] = _format_mean_std(fb_f1["mean"], fb_f1["std"])
            row["feedback_weighting_delta_vs_step1b_acc"] = float(fb_acc["mean"] - s1_acc["mean"])
            row["feedback_weighting_delta_vs_step1b_f1"] = float(fb_f1["mean"] - s1_f1["mean"])
            deltas.append(float(fb_f1["mean"] - s1_f1["mean"]))
        best_delta = max(deltas) if deltas else 0.0
        row["delta_vs_step1b"] = float(best_delta)
        row["result_label"] = _result_label(best_delta)
        perf_summary_rows.append(row)

    perf_summary_df = pd.DataFrame(perf_summary_rows).sort_values("dataset").reset_index(drop=True)
    perf_summary_df.to_csv(os.path.join(args.out_root, "feedback_upgrade_performance_summary.csv"), index=False)

    health_summary_rows: List[Dict[str, object]] = []
    if not health_df.empty:
        for (dataset, variant), df_g in health_df.groupby(["dataset", "variant"]):
            row = {
                "dataset": dataset,
                "variant": variant,
                "direction_usage_entropy": _format_mean_std(
                    float(df_g["direction_usage_entropy"].mean()),
                    float(df_g["direction_usage_entropy"].std()),
                ),
                "per_dir_margin_summary": str(df_g["per_dir_margin_summary"].iloc[0]) if len(df_g) == 1 else " / ".join(df_g["per_dir_margin_summary"].astype(str).tolist()),
                "per_dir_flip_summary": str(df_g["per_dir_flip_summary"].iloc[0]) if len(df_g) == 1 else " / ".join(df_g["per_dir_flip_summary"].astype(str).tolist()),
                "per_dir_intrusion_summary": str(df_g["per_dir_intrusion_summary"].iloc[0]) if len(df_g) == 1 else " / ".join(df_g["per_dir_intrusion_summary"].astype(str).tolist()),
                "worst_dir_id": str(df_g["worst_dir_id"].iloc[0]) if len(df_g) == 1 else ",".join(df_g["worst_dir_id"].astype(str).tolist()),
                "worst_dir_metric_summary": str(df_g["worst_dir_metric_summary"].iloc[0]) if len(df_g) == 1 else " / ".join(df_g["worst_dir_metric_summary"].astype(str).tolist()),
            }
            health_summary_rows.append(row)
    health_summary_df = pd.DataFrame(health_summary_rows).sort_values(["dataset", "variant"]).reset_index(drop=True)
    health_summary_df.to_csv(os.path.join(args.out_root, "feedback_direction_health_summary.csv"), index=False)

    conclusion_lines: List[str] = [
        "# Feedback Upgrade Conclusion",
        "",
        "更新时间：2026-03-20",
        "",
        "身份：`diagnostic-only`",
        "",
        "- `not for Phase15 mainline freeze table`",
        "- `small fixed-split datasets only`",
        "- `priority = Fisher/C0 direction upgrade first`",
        "",
        "## Result Snapshot",
        "",
    ]
    positive_datasets: List[str] = []
    priority_a = {"har", "selfregulationscp1", "fingermovements"}
    priority_a_positive: List[str] = []
    fisher_total_delta = 0.0
    feedback_total_delta = 0.0
    for _, row in perf_summary_df.iterrows():
        ds = str(row["dataset"])
        label = str(row["result_label"])
        d_fc = float(row["fisher_c0_delta_vs_step1b_f1"])
        d_fb = row["feedback_weighting_delta_vs_step1b_f1"]
        if label == "direction_upgrade_positive":
            positive_datasets.append(ds)
            if ds in priority_a:
                priority_a_positive.append(ds)
        fisher_total_delta += float(d_fc)
        if d_fb is not None:
            feedback_total_delta += float(d_fb)
        conclusion_lines.append(
            f"- `{ds}`: Step1B={row['step1b_f1']}, Fisher/C0={row['fisher_c0_f1']}, "
            f"Feedback={row['feedback_weighting_f1'] if row['feedback_weighting_f1'] is not None else 'not_run'}, "
            f"label=`{label}`"
        )

    if len(priority_a_positive) >= 2:
        trainability_readout = "是"
        raw_bridge_readout = "可以考虑，但仍只建议先回到 simple-set"
    elif len(positive_datasets) >= 1:
        trainability_readout = "局部是，整体仍不足"
        raw_bridge_readout = "暂不建议广泛回接；若只做极小规模 pilot，优先 HAR"
    else:
        trainability_readout = "否"
        raw_bridge_readout = "暂不建议"

    best_variant_name = "feedback_weighting" if feedback_total_delta > fisher_total_delta + 1e-6 else "fisher_c0"

    conclusion_lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- 当前是否把 PIA 从“稳定中性”推进到“更可训练”：`{trainability_readout}`",
            f"- 当前更有效的是方向库升级还是方向信用分：`{best_variant_name}`",
            f"- 最先出现清晰正信号的数据集：`{positive_datasets[0] if positive_datasets else 'current evidence insufficient'}`",
            f"- 当前是否足以考虑再接回 raw-bridge：`{raw_bridge_readout}`",
            "",
            "## Recommendation",
            "",
            "- 本轮输出只用于判断 PIA feedback upgrade 是否值得继续，不进入 Phase15 freeze 主表。",
            "- 若后续要接回 raw-bridge，优先只回到当前 upgrade 信号最清晰的 simple-set 数据集。",
        ]
    )
    with open(os.path.join(args.out_root, "feedback_upgrade_conclusion.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(conclusion_lines) + "\n")


if __name__ == "__main__":
    main()
