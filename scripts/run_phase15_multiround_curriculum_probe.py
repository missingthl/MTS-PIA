#!/usr/bin/env python
"""Phase15 multiround curriculum stepping probe on small fixed-split datasets.

Diagnostic-only upgrade line.

This probe keeps the current Step1B direction bank, but replaces the single
equal gamma with a per-direction curriculum budget updated across rounds.
It also reserves a minimal sample-pool interface for future evolutionary
resampling by tagging each augmented sample as one of:
- train_keep
- geometry_keep
- drop

Important constraints:
- not part of Phase15 mainline freeze table
- fixed-split small datasets only
- augmentation source is always the original train set of the current split
- no recursive child-of-child augmentation generation
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


def _ordered_unique(values: Iterable[object]) -> List[object]:
    seen = set()
    out: List[object] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _stable_tid_hash(tid: object) -> int:
    return abs(hash(str(tid))) % 1_000_003


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


def _format_mean_std(mean: float, std: float) -> str:
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def _entropy_from_probs(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _minmax_norm(x: np.ndarray, *, constant_fill: float = 0.5) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64).ravel()
    if xx.size == 0:
        return np.asarray([], dtype=np.float64)
    xmin = float(np.min(xx))
    xmax = float(np.max(xx))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or abs(xmax - xmin) <= 1e-12:
        return np.full(xx.shape, float(constant_fill), dtype=np.float64)
    return (xx - xmin) / (xmax - xmin)


def _dict_summary_string(values: Dict[int, float], *, fmt: str = ".4f") -> str:
    if not values:
        return "n/a"
    parts = [f"{int(k)}:{format(float(v), fmt)}" for k, v in sorted(values.items())]
    return "|".join(parts)


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
    probs = np.zeros_like(g, dtype=np.float64)
    probs[active] = 1.0 / float(np.sum(active))
    return probs


def _build_curriculum_aug_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tid_train: np.ndarray,
    *,
    direction_bank: np.ndarray,
    direction_probs: np.ndarray,
    gamma_by_dir: np.ndarray,
    multiplier: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    tid_arr = np.asarray(tid_train)
    y_arr = np.asarray(y_train).astype(int).ravel()
    k_dir = int(direction_bank.shape[0])
    probs = np.asarray(direction_probs, dtype=np.float64).ravel()
    gammas = np.asarray(gamma_by_dir, dtype=np.float64).ravel()
    if probs.shape[0] != k_dir or gammas.shape[0] != k_dir:
        raise ValueError("direction probs / gamma size mismatch")
    if float(np.sum(probs)) <= 0:
        probs = np.full((k_dir,), 1.0 / float(k_dir), dtype=np.float64)
    else:
        probs = probs / float(np.sum(probs))

    aug_X_parts: List[np.ndarray] = []
    aug_y_parts: List[np.ndarray] = []
    aug_tid_parts: List[np.ndarray] = []
    aug_src_parts: List[np.ndarray] = []
    aug_dir_parts: List[np.ndarray] = []
    aug_gamma_parts: List[np.ndarray] = []
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
            gamma_vec = gammas[dir_ids].astype(np.float32)
            X_aug = (X_tid + gamma_vec[:, None] * signs[:, None] * direction_bank[dir_ids]).astype(np.float32)

            aug_X_parts.append(X_aug)
            aug_y_parts.append(y_tid.copy())
            aug_tid_parts.append(np.asarray([tid] * len(idx)))
            aug_src_parts.append(X_tid.copy())
            aug_dir_parts.append(dir_ids.copy())
            aug_gamma_parts.append(gamma_vec.copy())
            dir_pick_count += np.bincount(dir_ids, minlength=k_dir).astype(np.int64)
            added += len(idx)
        aug_count_per_trial[str(tid)] = int(added)

    if aug_X_parts:
        X_aug_all = np.vstack(aug_X_parts).astype(np.float32)
        y_aug_all = np.concatenate(aug_y_parts).astype(np.int64)
        tid_aug_all = np.concatenate(aug_tid_parts)
        src_aug_all = np.vstack(aug_src_parts).astype(np.float32)
        dir_aug_all = np.concatenate(aug_dir_parts).astype(np.int64)
        gamma_aug_all = np.concatenate(aug_gamma_parts).astype(np.float32)
    else:
        X_aug_all = np.empty((0, X_train.shape[1]), dtype=np.float32)
        y_aug_all = np.empty((0,), dtype=np.int64)
        tid_aug_all = np.empty((0,), dtype=object)
        src_aug_all = np.empty((0, X_train.shape[1]), dtype=np.float32)
        dir_aug_all = np.empty((0,), dtype=np.int64)
        gamma_aug_all = np.empty((0,), dtype=np.float32)

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
        "k_dir": int(k_dir),
        "subset_size": 1,
        "weighting_mode": "multiround_curriculum_gamma_budget",
        "mixing_stats": {
            "mean_abs_ai": 1.0,
            "avg_subset_size": 1.0,
            "direction_pick_fraction": pick_frac,
        },
        "direction_probs": {str(i): float(probs[i]) for i in range(k_dir)},
        "direction_usage_entropy": float(_entropy_from_probs(np.asarray(list(pick_frac.values()), dtype=np.float64))),
        "gamma_by_dir": {str(i): float(gammas[i]) for i in range(k_dir)},
        "gamma_used_summary": _summary_stats(gamma_aug_all.tolist()),
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
    flip: Dict[int, float] = {}
    margin: Dict[int, float] = {}
    intrusion: Dict[int, float] = {}
    for k, row in profile.items():
        if not isinstance(row, dict):
            continue
        did = int(k)
        usage[did] = float(row.get("usage", 0.0))
        flip[did] = float(row.get("flip_rate", 0.0))
        margin[did] = float(row.get("margin_drop_median", 0.0))
        if intrusion_by_dir and did in intrusion_by_dir:
            intrusion[did] = float(intrusion_by_dir[did])
    return {
        "usage": usage,
        "flip_rate": flip,
        "margin_drop_median": margin,
        "intrusion": intrusion,
    }


def _safe_quantile(arr: np.ndarray, q: float, default: float) -> float:
    x = np.asarray(arr, dtype=np.float64).ravel()
    if x.size == 0:
        return float(default)
    return float(np.quantile(x, float(q)))


def _update_direction_budget(
    *,
    gamma_before: np.ndarray,
    margin_by_dir: Dict[int, float],
    flip_by_dir: Dict[int, float],
    intrusion_by_dir: Dict[int, float],
    expand_factor: float,
    shrink_factor: float,
    gamma_max: float,
    freeze_eps: float,
) -> Tuple[np.ndarray, Dict[int, str], Dict[int, float]]:
    k_dir = int(len(gamma_before))
    dir_ids = list(range(k_dir))
    margin = np.asarray([float(margin_by_dir.get(i, 0.0)) for i in dir_ids], dtype=np.float64)
    flip = np.asarray([float(flip_by_dir.get(i, 0.0)) for i in dir_ids], dtype=np.float64)
    intrusion = np.asarray([float(intrusion_by_dir.get(i, 0.0)) for i in dir_ids], dtype=np.float64)

    if np.allclose(margin, margin[0]) and np.allclose(flip, flip[0]) and np.allclose(intrusion, intrusion[0]):
        return gamma_before.copy(), {i: "hold" for i in dir_ids}, {i: 0.5 for i in dir_ids}

    margin_good = _minmax_norm(margin, constant_fill=0.5)
    flip_good = 1.0 - _minmax_norm(flip, constant_fill=0.5)
    intr_good = 1.0 - _minmax_norm(intrusion, constant_fill=0.5)
    safety = (margin_good + flip_good + intr_good) / 3.0

    q_expand = _safe_quantile(safety, 0.75, 0.5)
    q_shrink = _safe_quantile(safety, 0.35, 0.5)
    q_freeze = _safe_quantile(safety, 0.15, 0.25)
    med_margin = _safe_quantile(margin, 0.50, 0.0)
    med_flip = _safe_quantile(flip, 0.50, 0.0)
    med_intr = _safe_quantile(intrusion, 0.50, 0.0)

    gamma_after = np.asarray(gamma_before, dtype=np.float64).copy()
    state_by_dir: Dict[int, str] = {}
    score_by_dir: Dict[int, float] = {}

    for i in dir_ids:
        score = float(safety[i])
        score_by_dir[i] = score
        g0 = float(gamma_before[i])
        if g0 <= float(freeze_eps):
            gamma_after[i] = 0.0
            state_by_dir[i] = "freeze"
            continue

        risky = bool(
            margin[i] < min(0.0, med_margin)
            and (flip[i] >= med_flip or intrusion[i] >= med_intr)
        )

        if score <= q_freeze and risky:
            gamma_after[i] = 0.0
            state_by_dir[i] = "freeze"
        elif score >= q_expand and margin[i] >= med_margin and flip[i] <= med_flip and intrusion[i] <= med_intr:
            gamma_after[i] = min(float(gamma_max), g0 * float(expand_factor))
            state_by_dir[i] = "expand"
        elif score <= q_shrink or risky:
            g1 = g0 * float(shrink_factor)
            if g1 <= float(freeze_eps):
                gamma_after[i] = 0.0
                state_by_dir[i] = "freeze"
            else:
                gamma_after[i] = g1
                state_by_dir[i] = "shrink"
        else:
            gamma_after[i] = g0
            state_by_dir[i] = "hold"

    return gamma_after.astype(np.float64), state_by_dir, score_by_dir


def _sample_pool_label_counts(dir_aug: np.ndarray, state_by_dir: Dict[int, str]) -> Dict[str, int]:
    da = np.asarray(dir_aug).astype(int).ravel()
    train_keep = 0
    geometry_keep = 0
    drop = 0
    for did in da.tolist():
        state = state_by_dir.get(int(did), "hold")
        if state in {"expand", "hold"}:
            train_keep += 1
        elif state == "shrink":
            geometry_keep += 1
        else:
            drop += 1
    return {
        "train_keep_count": int(train_keep),
        "geometry_keep_count": int(geometry_keep),
        "drop_count": int(drop),
    }


def _variant_note(
    *,
    mean_intrusion: float,
    single_intrusion: float,
    worst_margin: float,
    single_worst_margin: float,
    frozen_dir_count: int,
) -> str:
    cleaner = mean_intrusion <= single_intrusion + 1e-9 and worst_margin >= single_worst_margin - 1e-9
    if cleaner and frozen_dir_count > 0:
        return "cleaner_than_single_round_with_targeted_freeze"
    if cleaner:
        return "cleaner_than_single_round"
    if frozen_dir_count > 0:
        return "freeze_signal_present_but_not_cleaner"
    return "mixed_signal"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="har,selfregulationscp1,fingermovements,natops")
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--out-root", type=str, default="out/phase15_multiround_curriculum_20260320")
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
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--curriculum-init-gamma", type=float, default=0.06)
    parser.add_argument("--curriculum-expand-factor", type=float, default=1.25)
    parser.add_argument("--curriculum-shrink-factor", type=float, default=0.70)
    parser.add_argument("--curriculum-gamma-max", type=float, default=0.16)
    parser.add_argument("--curriculum-freeze-eps", type=float, default=0.02)
    parser.add_argument("--split-preview-n", type=int, default=5)
    args = parser.parse_args()

    datasets = [normalize_dataset_name(x) for x in _parse_csv_list(args.datasets)]
    for ds in datasets:
        if ds not in FIXED_SMALL_DATASETS:
            raise ValueError(f"Unsupported dataset for multiround curriculum probe: {ds}")
    if int(args.subset_size) != 1:
        raise ValueError("This probe currently locks subset_size=1 to stay aligned with current Step1B default.")
    if int(args.n_rounds) < 1:
        raise ValueError("--n-rounds must be >= 1")

    seeds = _parse_seed_list(args.seeds)
    ensure_dir(args.out_root)

    perf_rows: List[Dict[str, object]] = []
    budget_rows: List[Dict[str, object]] = []
    health_rows: List[Dict[str, object]] = []
    pool_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        print(f"[multiround-curriculum][{dataset}] load", flush=True)
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
            f"[multiround-curriculum][{dataset}] feat "
            f"train={tuple(X_train_base.shape)} test={tuple(X_test.shape)}",
            flush=True,
        )

        for seed in seeds:
            print(f"[multiround-curriculum][{dataset}][seed={seed}] start", flush=True)
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
                progress_prefix=f"[multiround-curriculum][{dataset}][seed={seed}][baseline]",
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
                progress_prefix=f"[multiround-curriculum][{dataset}][seed={seed}][mech_step1b]",
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
                progress_prefix=f"[multiround-curriculum][{dataset}][seed={seed}][step1b]",
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
                "darwin_pool_interface": ["train_keep", "geometry_keep", "drop"],
            }
            _write_condition(
                os.path.join(seed_dir, "A_baseline"),
                metrics_base,
                {**common_meta, **train_meta_base, "condition": "A_baseline"},
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

            step1b_dir_summary = _summarize_dir_profile(mech_step1b.get("dir_profile", {}))
            step1b_avg_intr = float(np.mean(list(step1b_intrusion_by_dir.values()))) if step1b_intrusion_by_dir else 0.0

            gamma_by_dir = np.full((int(args.k_dir),), float(args.curriculum_init_gamma), dtype=np.float64)
            for round_id in range(1, int(args.n_rounds) + 1):
                direction_probs = _active_direction_probs(
                    gamma_by_dir,
                    freeze_eps=float(args.curriculum_freeze_eps),
                )
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
                    class_weight=args.linear_class_weight,
                    linear_max_iter=int(args.linear_max_iter),
                    knn_k=int(args.mech_knn_k),
                    max_aug_for_mech=int(args.mech_max_aug_for_metrics),
                    max_real_knn_ref=int(args.mech_max_real_knn_ref),
                    max_real_knn_query=int(args.mech_max_real_knn_query),
                    progress_prefix=f"[multiround-curriculum][{dataset}][seed={seed}][round={round_id}][mech]",
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
                    progress_prefix=f"[multiround-curriculum][{dataset}][seed={seed}][round={round_id}]",
                )

                round_dir = os.path.join(seed_dir, f"R{round_id}_curriculum")
                _write_condition(
                    round_dir,
                    metrics_curr,
                    {
                        **common_meta,
                        **train_meta_curr,
                        "condition": f"R{round_id}_curriculum",
                        "round_id": int(round_id),
                        "augmentation": curr_aug_meta,
                        "direction_bank": {**bank_meta, "subset_size": int(args.subset_size)},
                        "final_accept_rate": 1.0,
                        "gamma_before": {str(i): float(gamma_before[i]) for i in range(len(gamma_before))},
                        "gamma_after": {str(i): float(gamma_after[i]) for i in range(len(gamma_after))},
                        "direction_state": {str(i): str(state_by_dir.get(i, "hold")) for i in range(len(gamma_after))},
                        "direction_score": {str(i): float(score_by_dir.get(i, 0.0)) for i in range(len(gamma_after))},
                        "mech": mech_curr,
                    },
                )

                perf_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "round_id": int(round_id),
                        "baseline_acc": float(metrics_base["trial_acc"]),
                        "baseline_f1": float(metrics_base["trial_macro_f1"]),
                        "single_round_step1b_acc": float(metrics_step1b["trial_acc"]),
                        "single_round_step1b_f1": float(metrics_step1b["trial_macro_f1"]),
                        "multiround_curriculum_acc": float(metrics_curr["trial_acc"]),
                        "multiround_curriculum_f1": float(metrics_curr["trial_macro_f1"]),
                        "delta_vs_step1b_acc": float(metrics_curr["trial_acc"] - metrics_step1b["trial_acc"]),
                        "delta_vs_step1b": float(metrics_curr["trial_macro_f1"] - metrics_step1b["trial_macro_f1"]),
                        "result_label": _result_label(float(metrics_curr["trial_macro_f1"] - metrics_step1b["trial_macro_f1"])),
                    }
                )

                sample_pool_counts = _sample_pool_label_counts(dir_curr, state_by_dir)
                n_aug_total = int(len(y_curr))
                pool_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "round_id": int(round_id),
                        "n_aug_total": n_aug_total,
                        **sample_pool_counts,
                        "keep_ratio": float(sample_pool_counts["train_keep_count"] / max(1, n_aug_total)),
                        "geometry_ratio": float(sample_pool_counts["geometry_keep_count"] / max(1, n_aug_total)),
                        "drop_ratio": float(sample_pool_counts["drop_count"] / max(1, n_aug_total)),
                    }
                )

                dir_summary = _summarize_dir_profile(mech_curr.get("dir_profile", {}))
                mean_intrusion = float(np.mean(list(intrusion_by_dir.values()))) if intrusion_by_dir else 0.0
                frozen_dir_count = int(sum(1 for s in state_by_dir.values() if s == "freeze"))
                expanded_dir_count = int(sum(1 for s in state_by_dir.values() if s == "expand"))
                usage_fracs = curr_aug_meta.get("mixing_stats", {}).get("direction_pick_fraction", {})
                usage_probs = np.asarray(
                    [float(usage_fracs.get(str(i), 0.0)) for i in range(int(args.k_dir))],
                    dtype=np.float64,
                )
                health_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "round_id": int(round_id),
                        "direction_usage_entropy": float(_entropy_from_probs(usage_probs)),
                        "worst_dir_id": dir_summary["worst_dir_id"],
                        "worst_dir_metric_summary": dir_summary["dir_profile_summary"],
                        "frozen_dir_count": int(frozen_dir_count),
                        "expanded_dir_count": int(expanded_dir_count),
                        "note": _variant_note(
                            mean_intrusion=mean_intrusion,
                            single_intrusion=step1b_avg_intr,
                            worst_margin=float(dir_summary["worst_dir_margin_drop"] or 0.0),
                            single_worst_margin=float(step1b_dir_summary["worst_dir_margin_drop"] or 0.0),
                            frozen_dir_count=frozen_dir_count,
                        ),
                    }
                )

                for dir_id in range(int(args.k_dir)):
                    budget_rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
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
                    f"[multiround-curriculum][{dataset}][seed={seed}][round={round_id}] "
                    f"f1={metrics_curr['trial_macro_f1']:.4f} "
                    f"delta_vs_step1b={metrics_curr['trial_macro_f1'] - metrics_step1b['trial_macro_f1']:+.4f} "
                    f"freeze={frozen_dir_count} expand={expanded_dir_count}",
                    flush=True,
                )

    perf_df = pd.DataFrame(perf_rows).sort_values(["dataset", "round_id", "seed"]).reset_index(drop=True)
    perf_df.to_csv(os.path.join(args.out_root, "summary_per_seed.csv"), index=False)
    pd.DataFrame(budget_rows).sort_values(["dataset", "round_id", "seed", "dir_id"]).to_csv(
        os.path.join(args.out_root, "multiround_curriculum_direction_budget_summary.csv"),
        index=False,
    )
    pd.DataFrame(pool_rows).sort_values(["dataset", "round_id", "seed"]).to_csv(
        os.path.join(args.out_root, "multiround_curriculum_sample_pool_summary.csv"),
        index=False,
    )

    perf_summary_rows: List[Dict[str, object]] = []
    for (dataset, round_id), df_g in perf_df.groupby(["dataset", "round_id"]):
        base_acc = _summary_stats(df_g["baseline_acc"].tolist())
        base_f1 = _summary_stats(df_g["baseline_f1"].tolist())
        s1_acc = _summary_stats(df_g["single_round_step1b_acc"].tolist())
        s1_f1 = _summary_stats(df_g["single_round_step1b_f1"].tolist())
        mr_acc = _summary_stats(df_g["multiround_curriculum_acc"].tolist())
        mr_f1 = _summary_stats(df_g["multiround_curriculum_f1"].tolist())
        delta = float(mr_f1["mean"] - s1_f1["mean"])
        perf_summary_rows.append(
            {
                "dataset": dataset,
                "round_id": int(round_id),
                "baseline_acc": _format_mean_std(base_acc["mean"], base_acc["std"]),
                "baseline_f1": _format_mean_std(base_f1["mean"], base_f1["std"]),
                "single_round_step1b_acc": _format_mean_std(s1_acc["mean"], s1_acc["std"]),
                "single_round_step1b_f1": _format_mean_std(s1_f1["mean"], s1_f1["std"]),
                "multiround_curriculum_acc": _format_mean_std(mr_acc["mean"], mr_acc["std"]),
                "multiround_curriculum_f1": _format_mean_std(mr_f1["mean"], mr_f1["std"]),
                "delta_vs_step1b": delta,
                "result_label": _result_label(delta),
            }
        )
    perf_summary_df = pd.DataFrame(perf_summary_rows).sort_values(["dataset", "round_id"]).reset_index(drop=True)
    perf_summary_df.to_csv(os.path.join(args.out_root, "multiround_curriculum_performance_summary.csv"), index=False)

    health_df = pd.DataFrame(health_rows)
    health_summary_rows: List[Dict[str, object]] = []
    if not health_df.empty:
        for (dataset, round_id), df_g in health_df.groupby(["dataset", "round_id"]):
            note_counts = df_g["note"].astype(str).value_counts().to_dict()
            health_summary_rows.append(
                {
                    "dataset": dataset,
                    "round_id": int(round_id),
                    "direction_usage_entropy": _format_mean_std(
                        float(df_g["direction_usage_entropy"].mean()),
                        float(df_g["direction_usage_entropy"].std()),
                    ),
                    "worst_dir_id": ",".join(df_g["worst_dir_id"].astype(str).tolist()),
                    "worst_dir_metric_summary": " / ".join(df_g["worst_dir_metric_summary"].astype(str).tolist()),
                    "frozen_dir_count": _format_mean_std(
                        float(df_g["frozen_dir_count"].mean()),
                        float(df_g["frozen_dir_count"].std()),
                    ),
                    "expanded_dir_count": _format_mean_std(
                        float(df_g["expanded_dir_count"].mean()),
                        float(df_g["expanded_dir_count"].std()),
                    ),
                    "note": "; ".join([f"{k}:{v}" for k, v in sorted(note_counts.items())]),
                }
            )
    health_summary_df = pd.DataFrame(health_summary_rows).sort_values(["dataset", "round_id"]).reset_index(drop=True)
    health_summary_df.to_csv(
        os.path.join(args.out_root, "multiround_curriculum_direction_health_summary.csv"),
        index=False,
    )

    conclusion_lines: List[str] = [
        "# Multiround Curriculum Conclusion",
        "",
        "更新时间：2026-03-20",
        "",
        "身份：`diagnostic-only parallel upgrade line`",
        "",
        "- `not for Phase15 mainline freeze table`",
        "- `fixed-split small datasets only`",
        "- `paradigm_2 = curriculum geodesic stepping`",
        "- `paradigm_1 interface reserved via sample-pool labels`",
        "",
        "## Result Snapshot",
        "",
    ]

    priority_a = {"har", "selfregulationscp1", "fingermovements"}
    positive_priority_a: List[str] = []
    positive_any: List[str] = []
    best_round_by_dataset: Dict[str, Tuple[int, float]] = {}
    for dataset, df_ds in perf_summary_df.groupby("dataset"):
        best_idx = int(df_ds["delta_vs_step1b"].astype(float).idxmax())
        best_row = perf_summary_df.loc[best_idx]
        best_round = int(best_row["round_id"])
        best_delta = float(best_row["delta_vs_step1b"])
        best_round_by_dataset[dataset] = (best_round, best_delta)
        if best_delta > 1e-6:
            positive_any.append(str(dataset))
            if str(dataset) in priority_a:
                positive_priority_a.append(str(dataset))
        conclusion_lines.append(
            f"- `{dataset}`: best_round=`{best_round}`, "
            f"step1b={best_row['single_round_step1b_f1']}, "
            f"multiround={best_row['multiround_curriculum_f1']}, "
            f"delta_vs_step1b={best_delta:+.4f}, label=`{best_row['result_label']}`"
        )

    if len(positive_priority_a) >= 2:
        phase_readout = "值得进入下一阶段"
    elif len(positive_any) >= 1:
        phase_readout = "继续作为探索线"
    else:
        phase_readout = "当前方案暂缓"

    if positive_priority_a:
        raw_bridge_hint = "比单轮 Step1B 更值得作为 raw-bridge 前端 target quality upgrade 候选；very small pilot 仍优先 HAR"
    elif positive_any:
        raw_bridge_hint = "只有局部信号，暂不建议广泛接回；若必须 pilot，仍优先 HAR"
    else:
        raw_bridge_hint = "当前不值得接回 raw-bridge，主要因为方向预算更新仍未形成稳定收益"

    conclusion_lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- 多轮课程策略是否比单轮 Step1B 更有效：`{phase_readout}`",
            "- 当前跨轮反馈的主要收益来源：`步长课程为主；样本池标签当前仅作范式一接口预留`",
            f"- 最先出现清晰跨轮收益的数据集：`{positive_any[0] if positive_any else 'current evidence insufficient'}`",
            f"- 当前是否值得作为后续 bridge 前端 target quality upgrade 候选：`{raw_bridge_hint}`",
            "",
            "## Notes",
            "",
            "- 当前每轮增强源固定为原始训练样本，没有递归生成下一代增强样本。",
            "- `train_keep / geometry_keep / drop` 目前只用于落盘标记，不进入后续训练或构平面。",
        ]
    )
    with open(os.path.join(args.out_root, "multiround_curriculum_conclusion.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(conclusion_lines) + "\n")


if __name__ == "__main__":
    main()
