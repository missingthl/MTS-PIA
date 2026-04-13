#!/usr/bin/env python
"""Phase C0-directed: offline directed safe-axis scoring with plus/minus split."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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
from scripts.legacy_phase.run_phase14r_step6b1_rev2 import covs_to_features, ensure_dir, extract_features_block, logm_spd  # noqa: E402
from scripts.support.fisher_pia_utils import (  # noqa: E402
    FisherPIAConfig,
    compute_fisher_pia_terms,
    compute_generic_score_correlations,
    compute_safe_axis_scores,
    compute_safe_directed_scores,
    summarize_generic_score_signal,
)
from scripts.legacy_phase.run_phase15_k1_knn_gate import _apply_gate12_with_diag, _merge_gate3_into_dir_profile  # noqa: E402
from scripts.support.local_knn_gate import LocalKNNGateConfig, ReadOnlyLocalKNNGate  # noqa: E402
from scripts.legacy_phase.run_phase15_step1a_maxplane import _fit_gate1_from_train, _make_trial_split  # noqa: E402
from scripts.legacy_phase.run_phase15_step1b_multidir_matrix import (  # noqa: E402
    _build_direction_bank_d1,
    _true_class_margin,
    _write_condition,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402


def _parse_csv_list(text: str) -> List[str]:
    out = [t.strip() for t in str(text).split(",") if t.strip()]
    if not out:
        raise ValueError("list cannot be empty")
    return out


def _parse_int_list(text: str) -> List[int]:
    return sorted(set(int(t.strip()) for t in str(text).split(",") if t.strip()))


def _parse_float_list(text: str) -> List[float]:
    return [float(t.strip()) for t in str(text).split(",") if t.strip()]


def _parse_dataset_float_map(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in _parse_csv_list(text):
        if ":" not in part:
            raise ValueError(f"expected dataset:value entry, got {part!r}")
        key, value = part.split(":", 1)
        out[normalize_dataset_name(key.strip())] = float(value.strip())
    return out


def _stable_tid_hash(tid: object) -> int:
    return abs(hash(str(tid))) % 1_000_003


def _ordered_unique(values: List[object]) -> List[object]:
    seen = set()
    out: List[object] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _sample_subset_indices(rs: np.random.RandomState, n: int, *, k_dir: int, subset_size: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 0), dtype=np.int64)
    s = int(min(max(1, subset_size), k_dir))
    if s >= k_dir:
        base = np.arange(k_dir, dtype=np.int64)[None, :]
        return np.repeat(base, n, axis=0)
    out = np.empty((n, s), dtype=np.int64)
    for i in range(n):
        out[i] = rs.choice(k_dir, size=s, replace=False)
    return out


def _build_multidir_aug_candidates_with_sign(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tid_train: np.ndarray,
    *,
    direction_bank: np.ndarray,
    subset_size: int,
    gamma: float,
    multiplier: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    tid_arr = np.asarray(tid_train)
    y_arr = np.asarray(y_train).astype(int).ravel()
    k_dir = int(direction_bank.shape[0])
    trial_ids = sorted(_ordered_unique(tid_arr.tolist()))

    aug_X_parts: List[np.ndarray] = []
    aug_y_parts: List[np.ndarray] = []
    aug_tid_parts: List[np.ndarray] = []
    aug_src_parts: List[np.ndarray] = []
    aug_dir_parts: List[np.ndarray] = []
    aug_sign_parts: List[np.ndarray] = []

    for tid in trial_ids:
        idx = np.where(tid_arr == tid)[0]
        if idx.size == 0:
            continue
        X_tid = np.asarray(X_train[idx], dtype=np.float32)
        y_tid = np.asarray(y_arr[idx], dtype=np.int64)
        for m in range(max(0, int(multiplier))):
            rs = np.random.RandomState(int(seed + m * 1009 + _stable_tid_hash(tid)))
            sel = _sample_subset_indices(
                rs,
                int(X_tid.shape[0]),
                k_dir=k_dir,
                subset_size=int(subset_size),
            )
            s_eff = int(sel.shape[1])
            a = rs.normal(loc=0.0, scale=1.0, size=(X_tid.shape[0], s_eff)).astype(np.float32)
            a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)

            W_sel = direction_bank[sel]
            w_mix = np.sum(a[:, :, None] * W_sel, axis=1).astype(np.float32)
            X_aug = (X_tid + float(gamma) * w_mix).astype(np.float32)

            primary_sign = np.sign(a[:, 0]).astype(np.int8)
            primary_sign[primary_sign == 0] = 1

            aug_X_parts.append(X_aug)
            aug_y_parts.append(y_tid.copy())
            aug_tid_parts.append(np.asarray([tid] * len(idx)))
            aug_src_parts.append(X_tid.copy())
            aug_dir_parts.append(sel[:, 0].astype(np.int64).copy())
            aug_sign_parts.append(primary_sign.copy())

    if aug_X_parts:
        X_aug_all = np.vstack(aug_X_parts).astype(np.float32)
        y_aug_all = np.concatenate(aug_y_parts).astype(np.int64)
        tid_aug_all = np.concatenate(aug_tid_parts)
        src_aug_all = np.vstack(aug_src_parts).astype(np.float32)
        dir_aug_all = np.concatenate(aug_dir_parts).astype(np.int64)
        sign_aug_all = np.concatenate(aug_sign_parts).astype(np.int8)
    else:
        X_aug_all = np.empty((0, X_train.shape[1]), dtype=np.float32)
        y_aug_all = np.empty((0,), dtype=np.int64)
        tid_aug_all = np.empty((0,), dtype=object)
        src_aug_all = np.empty((0, X_train.shape[1]), dtype=np.float32)
        dir_aug_all = np.empty((0,), dtype=np.int64)
        sign_aug_all = np.empty((0,), dtype=np.int8)

    meta = {
        "aug_total_count": int(len(y_aug_all)),
        "gamma": float(gamma),
        "k_dir": int(k_dir),
        "subset_size": int(subset_size),
        "candidate_sign_definition": "primary_direction_coefficient_sign",
    }
    return X_aug_all, y_aug_all, tid_aug_all, src_aug_all, dir_aug_all, sign_aug_all, meta


def _logcenter_train_only(covs_train: np.ndarray, eps: float) -> np.ndarray:
    covs = np.asarray(covs_train, dtype=np.float32)
    if covs.size == 0:
        return np.empty((0,), dtype=np.float32)
    log_train = np.array([logm_spd(c, eps) for c in covs], dtype=np.float32)
    mean_log = np.mean(log_train, axis=0)
    return (log_train - mean_log).astype(np.float32)


def _compute_directed_profile(
    *,
    X_train_real: np.ndarray,
    y_train_real: np.ndarray,
    X_aug_generated: np.ndarray,
    y_aug_generated: np.ndarray,
    dir_generated: np.ndarray,
    sign_generated: np.ndarray,
    X_aug_accepted: np.ndarray,
    y_aug_accepted: np.ndarray,
    X_src_accepted: np.ndarray,
    dir_accepted: np.ndarray,
    sign_accepted: np.ndarray,
    seed: int,
    linear_c: float,
    linear_max_iter: int,
    knn_k: int,
    max_aug_for_mech: int,
    gate3_input_dir: np.ndarray | None = None,
    gate3_input_sign: np.ndarray | None = None,
    gate3_keep: np.ndarray | None = None,
) -> pd.DataFrame:
    Xr = np.asarray(X_train_real, dtype=np.float32)
    yr = np.asarray(y_train_real).astype(int).ravel()
    Xg = np.asarray(X_aug_generated, dtype=np.float32)
    yg = np.asarray(y_aug_generated).astype(int).ravel()
    dir_g = np.asarray(dir_generated).astype(int).ravel()
    sign_g = np.asarray(sign_generated).astype(int).ravel()
    Xa = np.asarray(X_aug_accepted, dtype=np.float32)
    ya = np.asarray(y_aug_accepted).astype(int).ravel()
    Xs = np.asarray(X_src_accepted, dtype=np.float32)
    dir_a = np.asarray(dir_accepted).astype(int).ravel()
    sign_a = np.asarray(sign_accepted).astype(int).ravel()

    scaler = StandardScaler()
    Xr_s = scaler.fit_transform(Xr)
    clf = LinearSVC(C=float(linear_c), max_iter=int(linear_max_iter), random_state=int(seed), dual="auto")
    clf.fit(Xr_s, yr)

    rs = np.random.RandomState(int(seed) + 8093)
    n_acc = int(len(ya))
    if n_acc <= 0:
        eval_idx = np.asarray([], dtype=np.int64)
    elif n_acc > int(max_aug_for_mech):
        eval_idx = np.sort(rs.choice(n_acc, size=int(max_aug_for_mech), replace=False))
    else:
        eval_idx = np.arange(n_acc, dtype=np.int64)

    if eval_idx.size:
        Xa_eval = Xa[eval_idx]
        Xs_eval = Xs[eval_idx]
        ya_eval = ya[eval_idx]
        dir_eval = dir_a[eval_idx]
        sign_eval = sign_a[eval_idx]

        src_scores = clf.decision_function(scaler.transform(Xs_eval))
        aug_scores = clf.decision_function(scaler.transform(Xa_eval))
        src_margin = _true_class_margin(src_scores, ya_eval, clf.classes_)
        aug_margin = _true_class_margin(aug_scores, ya_eval, clf.classes_)
        flip = (src_margin >= 0.0) != (aug_margin >= 0.0)
        margin_delta = aug_margin - src_margin

        k_eff = int(min(max(1, int(knn_k)), len(yr)))
        nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
        nn.fit(Xr)
        nn_idx = nn.kneighbors(Xa_eval, return_distance=False)
        y_nb = yr[nn_idx]
        intrusion_each = 1.0 - np.mean(y_nb == ya_eval[:, None], axis=1).astype(np.float64)
    else:
        dir_eval = np.empty((0,), dtype=np.int64)
        sign_eval = np.empty((0,), dtype=np.int64)
        flip = np.asarray([], dtype=bool)
        margin_delta = np.asarray([], dtype=np.float64)
        intrusion_each = np.asarray([], dtype=np.float64)

    gate3_reject_lookup: Dict[Tuple[int, int], float] = {}
    if gate3_input_dir is not None and gate3_input_sign is not None and gate3_keep is not None:
        d_in = np.asarray(gate3_input_dir).astype(int).ravel()
        s_in = np.asarray(gate3_input_sign).astype(int).ravel()
        k_in = np.asarray(gate3_keep).astype(bool).ravel()
        for did in sorted(np.unique(d_in).tolist()):
            for sgn in (-1, 1):
                mask = (d_in == did) & (s_in == sgn)
                if np.any(mask):
                    gate3_reject_lookup[(int(did), int(sgn))] = float(np.mean(~k_in[mask]))

    rows: List[Dict[str, object]] = []
    total_gen = int(len(dir_g))
    dir_candidates = sorted(set(dir_g.tolist()) | set(dir_a.tolist()))
    for did in dir_candidates:
        for sgn in (-1, 1):
            sign_name = "plus" if int(sgn) > 0 else "minus"
            gen_mask = (dir_g == int(did)) & (sign_g == int(sgn))
            acc_mask = (dir_a == int(did)) & (sign_a == int(sgn))
            eval_mask = (dir_eval == int(did)) & (sign_eval == int(sgn))
            n_gen = int(np.sum(gen_mask))
            n_acc = int(np.sum(acc_mask))
            rows.append(
                {
                    "direction_id": int(did),
                    "sign": sign_name,
                    "usage": float(n_gen / total_gen) if total_gen > 0 else 0.0,
                    "accept_rate": float(n_acc / n_gen) if n_gen > 0 else 0.0,
                    "flip_rate": float(np.mean(flip[eval_mask])) if np.any(eval_mask) else 0.0,
                    "margin_drop_median": float(np.median(margin_delta[eval_mask])) if np.any(eval_mask) else 0.0,
                    "intrusion": float(np.mean(intrusion_each[eval_mask])) if np.any(eval_mask) else 0.0,
                    "n_gen": n_gen,
                    "n_acc": n_acc,
                    "gate3_reject_rate": gate3_reject_lookup.get((int(did), int(sgn))),
                }
            )
    return pd.DataFrame(rows).sort_values(["direction_id", "sign"]).reset_index(drop=True)


def _score_comparison(axis_corr_df: pd.DataFrame, directed_corr_df: pd.DataFrame) -> pd.DataFrame:
    if axis_corr_df.empty and directed_corr_df.empty:
        return pd.DataFrame()
    axis_df = axis_corr_df[["metric_name", "pearson_r", "spearman_rho"]].rename(
        columns={"pearson_r": "axis_pearson_r", "spearman_rho": "axis_spearman_rho"}
    )
    dir_df = directed_corr_df[["metric_name", "pearson_r", "spearman_rho"]].rename(
        columns={"pearson_r": "directed_pearson_r", "spearman_rho": "directed_spearman_rho"}
    )
    out = axis_df.merge(dir_df, on="metric_name", how="outer")
    out["delta_abs_pearson"] = out["directed_pearson_r"].abs() - out["axis_pearson_r"].abs()
    out["delta_abs_spearman"] = out["directed_spearman_rho"].abs() - out["axis_spearman_rho"].abs()
    return out


def _plus_minus_gap_stats(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        return {"mean_abs_plus_minus_gap": 0.0, "max_abs_plus_minus_gap": 0.0, "n_plus_better": 0, "n_minus_better": 0}
    pv = df.pivot(index="direction_id", columns="sign", values="revised_score")
    plus = pv.get("plus", pd.Series(dtype=float)).astype(float)
    minus = pv.get("minus", pd.Series(dtype=float)).astype(float)
    common = pd.concat([plus, minus], axis=1, join="inner").dropna()
    if common.empty:
        return {"mean_abs_plus_minus_gap": 0.0, "max_abs_plus_minus_gap": 0.0, "n_plus_better": 0, "n_minus_better": 0}
    gap = (common.iloc[:, 0] - common.iloc[:, 1]).to_numpy(dtype=np.float64)
    return {
        "mean_abs_plus_minus_gap": float(np.mean(np.abs(gap))),
        "max_abs_plus_minus_gap": float(np.max(np.abs(gap))),
        "n_plus_better": int(np.sum(gap > 0.0)),
        "n_minus_better": int(np.sum(gap < 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="natops,har")
    parser.add_argument("--dataset-betas", type=str, default="natops:0.5,har:2.0")
    parser.add_argument("--seeds", type=str, default="3")
    parser.add_argument("--include-approach", action="store_true")
    parser.add_argument("--gammas", type=str, default="0.5,1.0")
    parser.add_argument("--out-root", type=str, default="out/phase15_c0_directed_safe_axis")
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
    parser.add_argument("--enable-gate3", action="store_true")
    parser.add_argument("--gate3-k", type=int, default=3)
    parser.add_argument("--gate3-tau-purity", type=float, default=0.66)
    parser.add_argument("--gate3-anchor-cap-k", type=int, default=120)
    parser.add_argument("--gate3-knn-algorithm", type=str, default="auto", choices=["auto", "ball_tree", "kd_tree", "brute"])
    parser.add_argument("--gate3-query-batch-size", type=int, default=4096)
    parser.add_argument("--mech-knn-k", type=int, default=20)
    parser.add_argument("--mech-max-aug-for-metrics", type=int, default=500)
    parser.add_argument("--linear-c", type=float, default=1.0)
    parser.add_argument("--linear-max-iter", type=int, default=1000)
    parser.add_argument("--fisher-knn-k", type=int, default=20)
    parser.add_argument("--fisher-boundary-quantile", type=float, default=0.30)
    parser.add_argument("--fisher-interior-quantile", type=float, default=0.70)
    parser.add_argument("--fisher-hetero-k", type=int, default=3)
    parser.add_argument("--split-preview-n", type=int, default=5)
    args = parser.parse_args()

    datasets = [normalize_dataset_name(d) for d in _parse_csv_list(args.datasets)]
    dataset_betas = _parse_dataset_float_map(args.dataset_betas)
    seeds = _parse_int_list(args.seeds)
    gammas = _parse_float_list(args.gammas)
    ensure_dir(args.out_root)
    summary_rows: List[Dict[str, object]] = []

    fisher_cfg = FisherPIAConfig(
        knn_k=int(args.fisher_knn_k),
        interior_quantile=float(args.fisher_interior_quantile),
        boundary_quantile=float(args.fisher_boundary_quantile),
        hetero_k=int(args.fisher_hetero_k),
    )

    for dataset in datasets:
        beta = float(dataset_betas[dataset])
        print(f"[C0-directed][{dataset}] load")
        all_trials = load_trials_for_dataset(
            dataset=dataset,
            processed_root=args.processed_root,
            stim_xlsx=args.stim_xlsx,
            har_root=args.har_root,
            mitbih_npz=args.mitbih_npz,
            seediv_root=args.seediv_root,
            natops_root=args.natops_root,
            fingermovements_root=args.fingermovements_root,
        )
        bands_spec = resolve_band_spec(dataset, args.bands)
        bands = parse_band_spec(bands_spec)

        for seed in seeds:
            print(f"[C0-directed][{dataset}][seed={seed}] start")
            train_trials, _, split_meta = _make_trial_split(all_trials, seed=int(seed))
            covs_train, y_train, tid_train = extract_features_block(
                train_trials, args.window_sec, args.hop_sec, args.cov_est, args.spd_eps, bands
            )
            covs_train_lc = _logcenter_train_only(covs_train, args.spd_eps)
            X_train = covs_to_features(covs_train_lc).astype(np.float32)
            y_train = np.asarray(y_train).astype(int).ravel()
            tid_train = np.asarray(tid_train)

            mu_gate1, tau_gate1, gate1_fit_meta = _fit_gate1_from_train(X_train=X_train, y_train=y_train, q=float(args.gate1_q))
            bank_seed = int(seed * 10000 + int(args.kdir) * 113 + 17)
            direction_bank, bank_meta = _build_direction_bank_d1(
                X_train=X_train,
                k_dir=int(args.kdir),
                seed=bank_seed,
                n_iters=int(args.pia_n_iters),
                activation=args.pia_activation,
                bias_update_mode=args.pia_bias_update_mode,
                c_repr=float(args.pia_c_repr),
            )
            X_aug, y_aug, tid_aug, src_aug, dir_aug, sign_aug, aug_meta = _build_multidir_aug_candidates_with_sign(
                X_train=X_train,
                y_train=y_train,
                tid_train=tid_train,
                direction_bank=direction_bank,
                subset_size=int(args.subset_size),
                gamma=float(args.pia_gamma),
                multiplier=int(args.pia_multiplier),
                seed=int(seed + 100000 + int(args.kdir) * 101 + int(args.subset_size) * 7),
            )
            X_keep12, y_keep12, tid_keep12, src_keep12, keep1, keep2, gate12_meta = _apply_gate12_with_diag(
                X_aug,
                y_aug,
                tid_aug,
                src_aug,
                mu_y=mu_gate1,
                tau_y=tau_gate1,
                gate2_q_src=float(args.gate2_q_src),
            )
            keep12 = np.asarray(keep1, dtype=bool) & np.asarray(keep2, dtype=bool)
            dir_keep12 = np.asarray(dir_aug, dtype=np.int64)[keep12]
            sign_keep12 = np.asarray(sign_aug, dtype=np.int8)[keep12]

            gate3_diag = None
            if bool(args.enable_gate3):
                gate3 = ReadOnlyLocalKNNGate(
                    LocalKNNGateConfig(
                        k=int(args.gate3_k),
                        tau_purity=float(args.gate3_tau_purity),
                        anchor_cap_k=int(args.gate3_anchor_cap_k),
                        algorithm=args.gate3_knn_algorithm,
                        query_batch_size=int(args.gate3_query_batch_size),
                    )
                )
                gate3.fit(clean_Z_train=X_train, y_train=y_train, tid_train=tid_train)
                keep3, gate3_diag = gate3.evaluate_batch(
                    candidate_Z=X_keep12,
                    source_label=y_keep12,
                    source_tid=tid_keep12,
                    direction_id=dir_keep12,
                    gamma_used=np.full((len(y_keep12),), float(args.pia_gamma), dtype=np.float32),
                )
            else:
                keep3 = np.ones((len(y_keep12),), dtype=bool)

            X_keep = X_keep12[keep3]
            y_keep = y_keep12[keep3]
            src_keep = src_keep12[keep3]
            dir_keep = dir_keep12[keep3]
            sign_keep = sign_keep12[keep3]

            class_terms, terms_meta = compute_fisher_pia_terms(X_train, y_train, cfg=fisher_cfg)
            directed_profile = _compute_directed_profile(
                X_train_real=X_train,
                y_train_real=y_train,
                X_aug_generated=X_aug,
                y_aug_generated=y_aug,
                dir_generated=dir_aug,
                sign_generated=sign_aug,
                X_aug_accepted=X_keep,
                y_aug_accepted=y_keep,
                X_src_accepted=src_keep,
                dir_accepted=dir_keep,
                sign_accepted=sign_keep,
                seed=int(seed),
                linear_c=float(args.linear_c),
                linear_max_iter=int(args.linear_max_iter),
                knn_k=int(args.mech_knn_k),
                max_aug_for_mech=int(args.mech_max_aug_for_metrics),
                gate3_input_dir=dir_keep12 if bool(args.enable_gate3) else None,
                gate3_input_sign=sign_keep12 if bool(args.enable_gate3) else None,
                gate3_keep=keep3 if bool(args.enable_gate3) else None,
            )

            axis_class_df, axis_global_df = compute_safe_axis_scores(
                direction_bank,
                class_terms,
                beta=float(beta),
                gamma=0.0,
                include_approach=False,
                direction_score_mode="axis_level",
            )
            axis_merge_df = axis_global_df.merge(
                directed_profile.groupby("direction_id", as_index=False)
                .agg(
                    usage=("usage", "sum"),
                    accept_rate=("accept_rate", "sum"),
                    flip_rate=("flip_rate", "mean"),
                    margin_drop_median=("margin_drop_median", "mean"),
                    intrusion=("intrusion", "mean"),
                    gate3_reject_rate_i=("gate3_reject_rate", "mean"),
                ),
                on="direction_id",
                how="left",
            )
            axis_corr_df = compute_generic_score_correlations(axis_merge_df, score_name="revised_score")

            gamma_list = gammas if bool(args.include_approach) else [0.0]
            for gamma in gamma_list:
                class_df, global_df = compute_safe_directed_scores(
                    direction_bank,
                    class_terms,
                    beta=float(beta),
                    gamma=float(gamma),
                    include_approach=bool(args.include_approach),
                    direction_score_mode="directed_plus_minus",
                )
                global_df = global_df.merge(directed_profile, on=["direction_id", "sign"], how="left")
                corr_df = compute_generic_score_correlations(global_df, score_name="revised_score")
                signal_meta = summarize_generic_score_signal(
                    global_df,
                    score_name="revised_score",
                    lower_is_better_metrics=["intrusion", "flip_rate", "gate3_reject_rate"],
                )
                compare_df = _score_comparison(axis_corr_df, corr_df)
                gap_meta = _plus_minus_gap_stats(global_df)

                gamma_tag = str(float(gamma)).replace(".", "p")
                setting_tag = f"kdir{int(args.kdir)}_s{int(args.subset_size)}__beta{str(beta).replace('.', 'p')}"
                if bool(args.include_approach):
                    setting_tag += f"__gamma{gamma_tag}"
                run_dir = os.path.join(args.out_root, dataset, setting_tag, f"seed{seed}")
                ensure_dir(run_dir)

                metrics = {
                    "dataset": dataset,
                    "representation_space": "z",
                    "stage_id": "C0-directed-B2" if bool(args.include_approach) else "C0-directed-B1",
                    "lookback": 1,
                    "horizon": 1,
                    "direction_source": "step1b_d1_bank",
                    "transition_model": "none_offline_directed_safe_axis_score",
                    "center_update_mode": "none",
                    "loss_mode": "offline_directed_safe_axis_diagnosis",
                    "n_dirs": int(direction_bank.shape[0]),
                    "direction_score_mode": "directed_plus_minus",
                    **signal_meta,
                    **gap_meta,
                }
                run_meta = {
                    "dataset": dataset,
                    "seed": int(seed),
                    "split_hash": split_meta["split_hash"],
                    "train_count_trials": int(split_meta["train_count_trials"]),
                    "test_count_trials": int(split_meta["test_count_trials"]),
                    "train_trial_ids_preview": split_meta["train_trial_ids"][: max(0, int(args.split_preview_n))],
                    "representation_space": "z",
                    "stage_id": "C0-directed-B2" if bool(args.include_approach) else "C0-directed-B1",
                    "lookback": 1,
                    "horizon": 1,
                    "direction_source": "step1b_d1_bank",
                    "transition_model": "none_offline_directed_safe_axis_score",
                    "center_update_mode": "none",
                    "loss_mode": "offline_directed_safe_axis_diagnosis",
                    "n_dirs": int(direction_bank.shape[0]),
                    "feature_pipeline": {
                        "window_sec": float(args.window_sec),
                        "hop_sec": float(args.hop_sec),
                        "cov_est": args.cov_est,
                        "spd_eps": float(args.spd_eps),
                        "center": "logcenter_train_only",
                        "vectorize": "upper_triangle",
                        "bands": bands_spec,
                    },
                    "direction_bank": {**bank_meta, "subset_size": int(args.subset_size)},
                    "augmentation": aug_meta,
                    "gate1_fit": gate1_fit_meta,
                    "gate_apply": gate12_meta,
                    "gate2_config": {"enabled": True, "q_src": float(args.gate2_q_src)},
                    "gate3_enabled": bool(args.enable_gate3),
                    "c0_directed_enabled": True,
                    "c0_direction_score_mode": "directed_plus_minus",
                    "c0_beta": float(beta),
                    "c0_gamma": float(gamma),
                    "c0_normalization": "minmax_per_bank",
                    "c0_expand_definition": "axis_level_projected_interior_spread_shared_for_plus_minus",
                    "c0_risk_definition": "directional_positive_alignment_to_boundary_hetero_vectors",
                    "c0_approach_definition": "directional_positive_dot_to_other_class_means" if bool(args.include_approach) else None,
                    "fisher_terms_meta": terms_meta,
                    "axis_level_score_signal": summarize_generic_score_signal(
                        axis_merge_df,
                        score_name="revised_score",
                        lower_is_better_metrics=["intrusion", "flip_rate", "gate3_reject_rate_i"],
                    ),
                    "directed_score_signal": signal_meta,
                    "plus_minus_gap": gap_meta,
                }
                _write_condition(run_dir, metrics, run_meta)
                global_df.to_csv(os.path.join(run_dir, "c0_directed_direction_table.csv"), index=False)
                class_df.to_csv(os.path.join(run_dir, "c0_directed_direction_table_full.csv"), index=False)
                corr_df.to_csv(os.path.join(run_dir, "c0_directed_score_correlation.csv"), index=False)
                compare_df.to_csv(os.path.join(run_dir, "c0_directed_vs_axis_comparison.csv"), index=False)

                summary_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "beta": float(beta),
                        "gamma": float(gamma),
                        "include_approach": bool(args.include_approach),
                        "setting": setting_tag,
                        "split_hash": split_meta["split_hash"],
                        "score_signal_pass": bool(signal_meta["score_signal_pass"]),
                        "mean_abs_plus_minus_gap": gap_meta["mean_abs_plus_minus_gap"],
                        "max_abs_plus_minus_gap": gap_meta["max_abs_plus_minus_gap"],
                        "n_plus_better": gap_meta["n_plus_better"],
                        "n_minus_better": gap_meta["n_minus_better"],
                        "mean_delta_abs_spearman": float(compare_df["delta_abs_spearman"].mean()) if not compare_df.empty else None,
                        "mean_delta_abs_pearson": float(compare_df["delta_abs_pearson"].mean()) if not compare_df.empty else None,
                        "top_direction_id": signal_meta.get("top_direction_id"),
                        "bottom_direction_id": signal_meta.get("bottom_direction_id"),
                        "delta_accept_top_vs_bottom": signal_meta.get("delta_accept_rate_top_vs_bottom"),
                        "delta_intrusion_top_vs_bottom": signal_meta.get("delta_intrusion_top_vs_bottom"),
                        "delta_flip_top_vs_bottom": signal_meta.get("delta_flip_rate_top_vs_bottom"),
                        "delta_margin_top_vs_bottom": signal_meta.get("delta_margin_drop_median_top_vs_bottom"),
                    }
                )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(args.out_root, "summary_per_run.csv"), index=False)
        agg_df = (
            summary_df.groupby(["dataset", "beta", "gamma", "include_approach"], dropna=False)
            .agg(
                score_signal_pass_rate=("score_signal_pass", "mean"),
                mean_abs_plus_minus_gap=("mean_abs_plus_minus_gap", "mean"),
                mean_delta_abs_spearman=("mean_delta_abs_spearman", "mean"),
                mean_delta_abs_pearson=("mean_delta_abs_pearson", "mean"),
            )
            .reset_index()
        )
        agg_df.to_csv(os.path.join(args.out_root, "summary_agg.csv"), index=False)


if __name__ == "__main__":
    main()
