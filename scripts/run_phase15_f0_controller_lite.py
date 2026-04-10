#!/usr/bin/env python
"""Phase F0-A/B: PIA-Controller Lite on top of Step1B."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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
from run_phase14r_step6b1_rev2 import apply_logcenter, covs_to_features, ensure_dir, extract_features_block  # noqa: E402
from scripts.local_knn_gate import LocalKNNGateConfig, ReadOnlyLocalKNNGate  # noqa: E402
from scripts.pia_controller import PiaControllerConfig, PiaControllerLite  # noqa: E402
from scripts.run_phase15_k1_knn_gate import _apply_gate12_with_diag  # noqa: E402
from scripts.run_phase15_step1a_maxplane import _fit_eval_linearsvc, _fit_gate1_from_train, _make_trial_split  # noqa: E402
from scripts.run_phase15_step1b_multidir_matrix import (  # noqa: E402
    _build_direction_bank_d1,
    _build_multidir_aug_candidates,
    _ordered_unique,
    _true_class_margin,
    _write_condition,
)


def _parse_seed_list(text: str) -> List[int]:
    out = sorted(set(int(tok.strip()) for tok in str(text).split(",") if tok.strip()))
    if not out:
        raise ValueError("seed list cannot be empty")
    return out


def _stable_tid_hash(tid: object) -> int:
    return abs(hash(str(tid))) % 1_000_003


def _build_round_assignments(y_train: np.ndarray, *, n_rounds: int, seed: int) -> np.ndarray:
    y = np.asarray(y_train).astype(int).ravel()
    out = np.zeros((len(y),), dtype=np.int64)
    rs = np.random.RandomState(int(seed))
    for cls in sorted(np.unique(y).tolist()):
        idx = np.where(y == int(cls))[0]
        perm = idx[rs.permutation(len(idx))]
        splits = np.array_split(perm, int(n_rounds))
        for r, split in enumerate(splits):
            out[np.asarray(split, dtype=np.int64)] = int(r)
    return out


def _fit_reference_tools(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    linear_c: float,
    linear_max_iter: int,
    knn_k: int,
    max_real_knn_ref: int,
) -> Tuple[StandardScaler, LinearSVC, NearestNeighbors, np.ndarray]:
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train).astype(int).ravel()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LinearSVC(C=float(linear_c), max_iter=int(linear_max_iter), random_state=int(seed), dual="auto")
    clf.fit(Xs, y)

    rs = np.random.RandomState(int(seed) + 8093)
    n_real = int(len(y))
    if int(max_real_knn_ref) > 0 and n_real > int(max_real_knn_ref):
        ref_idx = np.sort(rs.choice(n_real, size=int(max_real_knn_ref), replace=False))
    else:
        ref_idx = np.arange(n_real, dtype=np.int64)
    Xr = X[ref_idx]
    yr = y[ref_idx]
    k_eff = int(min(max(1, int(knn_k)), max(1, len(yr))))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(Xr)
    return scaler, clf, nn, yr


def _sample_dirs_for_round(
    *,
    X_round: np.ndarray,
    y_round: np.ndarray,
    tid_round: np.ndarray,
    direction_bank: np.ndarray,
    controller: PiaControllerLite,
    base_gamma: float,
    seed: int,
    round_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rs = np.random.RandomState(int(seed + round_id * 10007))
    dir_ids = controller.sample_direction_ids(y_round, rs)
    gamma_scale = controller.lookup_gamma_scale(y_round, dir_ids)
    gamma_eff = float(base_gamma) * gamma_scale
    X_aug = np.asarray(X_round, dtype=np.float32) + gamma_eff[:, None].astype(np.float32) * direction_bank[dir_ids]
    return X_aug.astype(np.float32), dir_ids.astype(np.int64), gamma_eff.astype(np.float64)


def _round_direction_stats(
    *,
    y_round: np.ndarray,
    dir_ids: np.ndarray,
    accept_mask: np.ndarray,
    intrusion_each: np.ndarray,
    flip_each: np.ndarray,
    margin_delta_each: np.ndarray,
) -> pd.DataFrame:
    y = np.asarray(y_round).astype(int).ravel()
    d = np.asarray(dir_ids).astype(int).ravel()
    keep = np.asarray(accept_mask).astype(bool).ravel()
    intrusion = np.asarray(intrusion_each, dtype=np.float64).ravel()
    flip = np.asarray(flip_each, dtype=np.float64).ravel()
    margin = np.asarray(margin_delta_each, dtype=np.float64).ravel()

    rows: List[Dict[str, object]] = []
    for cls in sorted(np.unique(y).tolist()):
        mask_cls = y == int(cls)
        for did in sorted(np.unique(d[mask_cls]).tolist()):
            mask = mask_cls & (d == int(did))
            rows.append(
                {
                    "class_id": int(cls),
                    "direction_id": int(did),
                    "usage_count": int(np.sum(mask)),
                    "accept_count": int(np.sum(keep[mask])),
                    "accept_rate": float(np.mean(keep[mask])) if np.any(mask) else 0.0,
                    "intrusion": float(np.mean(intrusion[mask])) if np.any(mask) else 0.0,
                    "flip_rate": float(np.mean(flip[mask])) if np.any(mask) else 0.0,
                    "margin_drop_median": float(np.median(margin[mask])) if np.any(mask) else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["class_id", "direction_id"]).reset_index(drop=True)


def _controller_summary(state_df: pd.DataFrame) -> Dict[str, object]:
    if state_df.empty:
        return {
            "n_rows": 0,
            "mean_reward": 0.0,
            "max_weight": 0.0,
            "min_weight": 0.0,
            "n_frozen": 0,
        }
    return {
        "n_rows": int(len(state_df)),
        "mean_reward": float(state_df["reward_i"].mean()),
        "max_weight": float(state_df["sampling_weight_i"].max()),
        "min_weight": float(state_df["sampling_weight_i"].min()),
        "mean_gamma_scale": float(state_df["gamma_scale_i"].mean()) if "gamma_scale_i" in state_df.columns else 1.0,
        "max_gamma_scale": float(state_df["gamma_scale_i"].max()) if "gamma_scale_i" in state_df.columns else 1.0,
        "min_gamma_scale": float(state_df["gamma_scale_i"].min()) if "gamma_scale_i" in state_df.columns else 1.0,
        "n_frozen": int(state_df["frozen_flag_i"].sum()),
    }


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    xx = np.asarray(x, dtype=np.float64).ravel()
    yy = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 2:
        return None
    if np.allclose(xx, xx[0]) or np.allclose(yy, yy[0]):
        return None
    return float(np.corrcoef(xx, yy)[0, 1])


def _round_controller_summary(
    *,
    round_id: int,
    round_df: pd.DataFrame,
    controller_state_df: pd.DataFrame,
) -> Dict[str, object]:
    if round_df.empty or controller_state_df.empty:
        return {
            "round_id": int(round_id),
            "mean_gamma_scale": 1.0,
            "max_gamma_scale": 1.0,
            "min_gamma_scale": 1.0,
            "corr_gamma_scale_reward": None,
            "top_used_direction_ids_per_round": [],
        }
    merged = round_df.merge(
        controller_state_df[["class_id", "direction_id", "reward_i", "gamma_scale_i"]],
        on=["class_id", "direction_id"],
        how="left",
    )
    top_dirs = (
        round_df.groupby("direction_id", as_index=False)["usage_count"]
        .sum()
        .sort_values("usage_count", ascending=False)["direction_id"]
        .head(3)
        .astype(int)
        .tolist()
    )
    return {
        "round_id": int(round_id),
        "mean_gamma_scale": float(merged["gamma_scale_i"].mean()),
        "max_gamma_scale": float(merged["gamma_scale_i"].max()),
        "min_gamma_scale": float(merged["gamma_scale_i"].min()),
        "corr_gamma_scale_reward": _safe_corr(
            merged["gamma_scale_i"].to_numpy(dtype=np.float64),
            merged["reward_i"].to_numpy(dtype=np.float64),
        ),
        "top_used_direction_ids_per_round": top_dirs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="natops", choices=["seed1", "seed", "har", "mitbih", "seediv", "natops", "fingermovements"])
    parser.add_argument("--seeds", type=str, default="3")
    parser.add_argument("--out-root", type=str, default="out/phase15_f0_controller_lite")
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
    parser.add_argument("--enable-gate3", action="store_true")
    parser.add_argument("--gate3-k", type=int, default=3)
    parser.add_argument("--gate3-tau-purity", type=float, default=0.66)
    parser.add_argument("--gate3-anchor-cap-k", type=int, default=120)
    parser.add_argument("--gate3-knn-algorithm", type=str, default="auto", choices=["auto", "ball_tree", "kd_tree", "brute"])
    parser.add_argument("--gate3-query-batch-size", type=int, default=4096)
    parser.add_argument("--controller-rounds", type=int, default=2)
    parser.add_argument("--controller-alpha", type=float, default=1.0)
    parser.add_argument("--controller-beta", type=float, default=1.0)
    parser.add_argument("--controller-gamma", type=float, default=1.0)
    parser.add_argument("--controller-eta", type=float, default=1.0)
    parser.add_argument("--controller-tau", type=float, default=1.0)
    parser.add_argument("--controller-kappa", type=float, default=0.25)
    parser.add_argument("--controller-lambda-ema", type=float, default=0.2)
    parser.add_argument("--controller-freeze-m", type=int, default=3)
    parser.add_argument(
        "--controller-phase",
        type=str,
        default="F0-A",
        choices=["F0-A", "F0-B", "F0-B-lite-repair"],
    )
    parser.add_argument("--controller-update-gamma-scale", action="store_true")
    parser.add_argument("--controller-gamma-scale-min", type=float, default=0.5)
    parser.add_argument("--controller-gamma-scale-max", type=float, default=1.5)
    parser.add_argument("--mech-knn-k", type=int, default=20)
    parser.add_argument("--mech-max-real-knn-ref", type=int, default=3000)
    parser.add_argument("--split-preview-n", type=int, default=5)
    args = parser.parse_args()

    if int(args.subset_size) != 1:
        raise ValueError("F0-A controller-lite currently supports subset_size=1 only.")
    if int(args.pia_multiplier) != 1:
        raise ValueError("F0-A controller-lite currently assumes pia_multiplier=1 to keep total candidate count comparable.")

    dataset = normalize_dataset_name(args.dataset)
    seeds = _parse_seed_list(args.seeds)
    out_root = os.path.join(args.out_root, dataset)
    ensure_dir(out_root)

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

    summary_rows: List[Dict[str, object]] = []

    for seed in seeds:
        print(f"[{args.controller_phase}][{dataset}][seed={seed}] start")
        seed_t0 = time.perf_counter()
        train_trials, test_trials, split_meta = _make_trial_split(all_trials, seed=int(seed))

        covs_train, y_train, tid_train = extract_features_block(
            train_trials, args.window_sec, args.hop_sec, args.cov_est, args.spd_eps, bands
        )
        covs_test, y_test, tid_test = extract_features_block(
            test_trials, args.window_sec, args.hop_sec, args.cov_est, args.spd_eps, bands
        )
        covs_train_lc, covs_test_lc = apply_logcenter(covs_train, covs_test, args.spd_eps)
        X_train = covs_to_features(covs_train_lc).astype(np.float32)
        X_test = covs_to_features(covs_test_lc).astype(np.float32)
        y_train = np.asarray(y_train).astype(int).ravel()
        y_test = np.asarray(y_test).astype(int).ravel()
        tid_train = np.asarray(tid_train)
        tid_test = np.asarray(tid_test)

        mu_gate1, tau_gate1, gate1_fit_meta = _fit_gate1_from_train(X_train=X_train, y_train=y_train, q=float(args.gate1_q))
        cap_seed = int(seed) + 41

        metrics_a, train_meta_a = _fit_eval_linearsvc(
            X_train,
            y_train,
            tid_train,
            X_test,
            y_test,
            tid_test,
            seed=int(seed),
            cap_k=int(args.window_cap_k),
            cap_seed=cap_seed,
            cap_sampling_policy=args.cap_sampling_policy,
            linear_c=float(args.linear_c),
            class_weight="none",
            max_iter=int(args.linear_max_iter),
            agg_mode=args.aggregation_mode,
            is_aug_train=np.zeros((len(y_train),), dtype=bool),
        )

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

        X_ref_aug, y_ref_aug, tid_ref_aug, src_ref_aug, dir_ref_aug, aug_meta_ref = _build_multidir_aug_candidates(
            X_train=X_train,
            y_train=y_train,
            tid_train=tid_train,
            direction_bank=direction_bank,
            subset_size=int(args.subset_size),
            gamma=float(args.pia_gamma),
            multiplier=int(args.pia_multiplier),
            seed=int(seed + 100000 + int(args.kdir) * 101 + int(args.subset_size) * 7),
        )
        X_ref_keep, y_ref_keep, tid_ref_keep, src_ref_keep, keep1_ref, keep2_ref, gate12_meta_ref = _apply_gate12_with_diag(
            X_ref_aug,
            y_ref_aug,
            tid_ref_aug,
            src_ref_aug,
            mu_y=mu_gate1,
            tau_y=tau_gate1,
            gate2_q_src=float(args.gate2_q_src),
        )
        if bool(args.enable_gate3):
            gate3 = ReadOnlyLocalKNNGate(
                LocalKNNGateConfig(
                    k=int(args.gate3_k),
                    tau_purity=float(args.gate3_tau_purity),
                    algorithm=str(args.gate3_knn_algorithm),
                    query_batch_size=int(args.gate3_query_batch_size),
                )
            ).fit(X_train, y_train)
            keep3_ref, gate3_diag_ref = gate3.evaluate_batch(
                X_ref_keep,
                y_ref_keep,
                direction_ids=np.asarray(dir_ref_aug, dtype=np.int64)[keep1_ref & keep2_ref],
                gamma_used=np.full((len(y_ref_keep),), float(args.pia_gamma), dtype=np.float64),
                source_tids=tid_ref_keep,
            )
            X_ref_keep = X_ref_keep[keep3_ref]
            y_ref_keep = y_ref_keep[keep3_ref]
            tid_ref_keep = tid_ref_keep[keep3_ref]
        else:
            gate3_diag_ref = None

        X_train_ref = np.vstack([X_train, X_ref_keep]) if len(y_ref_keep) else X_train.copy()
        y_train_ref = np.concatenate([y_train, y_ref_keep]) if len(y_ref_keep) else y_train.copy()
        tid_train_ref = np.concatenate([tid_train, tid_ref_keep]) if len(y_ref_keep) else tid_train.copy()
        is_aug_ref = np.concatenate([np.zeros((len(y_train),), dtype=bool), np.ones((len(y_ref_keep),), dtype=bool)]) if len(y_ref_keep) else np.zeros((len(y_train),), dtype=bool)
        metrics_ref, train_meta_ref = _fit_eval_linearsvc(
            X_train_ref,
            y_train_ref,
            tid_train_ref,
            X_test,
            y_test,
            tid_test,
            seed=int(seed),
            cap_k=int(args.window_cap_k),
            cap_seed=cap_seed,
            cap_sampling_policy=args.cap_sampling_policy,
            linear_c=float(args.linear_c),
            class_weight="none",
            max_iter=int(args.linear_max_iter),
            agg_mode=args.aggregation_mode,
            is_aug_train=is_aug_ref,
        )

        scaler_ref, clf_ref, nn_ref, y_knn = _fit_reference_tools(
            X_train,
            y_train,
            seed=int(seed),
            linear_c=float(args.linear_c),
            linear_max_iter=int(args.linear_max_iter),
            knn_k=int(args.mech_knn_k),
            max_real_knn_ref=int(args.mech_max_real_knn_ref),
        )

        controller = PiaControllerLite(
            classes=sorted(np.unique(y_train).tolist()),
            n_dirs=int(direction_bank.shape[0]),
            cfg=PiaControllerConfig(
                alpha=float(args.controller_alpha),
                beta=float(args.controller_beta),
                gamma=float(args.controller_gamma),
                eta=float(args.controller_eta),
                tau=float(args.controller_tau),
                kappa=float(args.controller_kappa),
                lambda_ema=float(args.controller_lambda_ema),
                freeze_M=int(args.controller_freeze_m),
                enable_weight_update=True,
                gamma_scale_min=float(args.controller_gamma_scale_min),
                gamma_scale_max=float(args.controller_gamma_scale_max),
                enable_gamma_update=bool(args.controller_update_gamma_scale),
                enable_freeze=False,
            ),
        )

        round_ids = _build_round_assignments(y_train, n_rounds=int(args.controller_rounds), seed=int(seed) + 991)
        round_history: List[pd.DataFrame] = []
        round_summary_rows: List[Dict[str, object]] = []
        acc_X_parts: List[np.ndarray] = []
        acc_y_parts: List[np.ndarray] = []
        acc_tid_parts: List[np.ndarray] = []
        controller_gate3 = None
        if bool(args.enable_gate3):
            controller_gate3 = ReadOnlyLocalKNNGate(
                LocalKNNGateConfig(
                    k=int(args.gate3_k),
                    tau_purity=float(args.gate3_tau_purity),
                    algorithm=str(args.gate3_knn_algorithm),
                    query_batch_size=int(args.gate3_query_batch_size),
                )
            ).fit(X_train, y_train)

        for round_id in range(int(args.controller_rounds)):
            idx = np.where(round_ids == int(round_id))[0]
            if idx.size == 0:
                continue
            X_round = X_train[idx]
            y_round = y_train[idx]
            tid_round = tid_train[idx]
            X_aug, dir_ids, gamma_eff = _sample_dirs_for_round(
                X_round=X_round,
                y_round=y_round,
                tid_round=tid_round,
                direction_bank=direction_bank,
                controller=controller,
                base_gamma=float(args.pia_gamma),
                seed=int(seed) + 5003,
                round_id=int(round_id),
            )

            src_scores = clf_ref.decision_function(scaler_ref.transform(X_round))
            aug_scores = clf_ref.decision_function(scaler_ref.transform(X_aug))
            src_margin = _true_class_margin(src_scores, y_round, clf_ref.classes_)
            aug_margin = _true_class_margin(aug_scores, y_round, clf_ref.classes_)
            flip_each = ((src_margin >= 0.0) != (aug_margin >= 0.0)).astype(np.float64)
            margin_delta_each = (aug_margin - src_margin).astype(np.float64)
            nn_idx = nn_ref.kneighbors(X_aug, return_distance=False)
            purity_each = np.mean(y_knn[nn_idx] == y_round[:, None], axis=1).astype(np.float64)
            intrusion_each = 1.0 - purity_each

            X_keep12, y_keep12, tid_keep12, src_keep12, keep1, keep2, _ = _apply_gate12_with_diag(
                X_aug,
                y_round,
                tid_round,
                X_round,
                mu_y=mu_gate1,
                tau_y=tau_gate1,
                gate2_q_src=float(args.gate2_q_src),
            )
            accept_mask = np.asarray(keep1, dtype=bool) & np.asarray(keep2, dtype=bool)
            if controller_gate3 is not None:
                dir_keep12 = dir_ids[accept_mask]
                gamma_keep12 = gamma_eff[accept_mask]
                keep3, _ = controller_gate3.evaluate_batch(
                    X_keep12,
                    y_keep12,
                    direction_ids=dir_keep12,
                    gamma_used=gamma_keep12,
                    source_tids=tid_keep12,
                )
                accept_mask_full = accept_mask.copy()
                accept_mask_full[accept_mask] = keep3
                accept_mask = accept_mask_full

            round_df = _round_direction_stats(
                y_round=y_round,
                dir_ids=dir_ids,
                accept_mask=accept_mask,
                intrusion_each=intrusion_each,
                flip_each=flip_each,
                margin_delta_each=margin_delta_each,
            )
            round_df.insert(0, "round_id", int(round_id))
            round_history.append(round_df)
            controller.update_from_round(round_df)
            round_summary_rows.append(
                _round_controller_summary(
                    round_id=int(round_id),
                    round_df=round_df,
                    controller_state_df=controller.state_dataframe(base_gamma=float(args.pia_gamma)),
                )
            )

            if np.any(accept_mask):
                acc_X_parts.append(X_aug[accept_mask])
                acc_y_parts.append(y_round[accept_mask])
                acc_tid_parts.append(tid_round[accept_mask])

        if acc_X_parts:
            X_ctrl_keep = np.vstack(acc_X_parts).astype(np.float32)
            y_ctrl_keep = np.concatenate(acc_y_parts).astype(np.int64)
            tid_ctrl_keep = np.concatenate(acc_tid_parts)
        else:
            X_ctrl_keep = np.empty((0, X_train.shape[1]), dtype=np.float32)
            y_ctrl_keep = np.empty((0,), dtype=np.int64)
            tid_ctrl_keep = np.empty((0,), dtype=tid_train.dtype)

        X_train_ctrl = np.vstack([X_train, X_ctrl_keep]) if len(y_ctrl_keep) else X_train.copy()
        y_train_ctrl = np.concatenate([y_train, y_ctrl_keep]) if len(y_ctrl_keep) else y_train.copy()
        tid_train_ctrl = np.concatenate([tid_train, tid_ctrl_keep]) if len(y_ctrl_keep) else tid_train.copy()
        is_aug_ctrl = np.concatenate([np.zeros((len(y_train),), dtype=bool), np.ones((len(y_ctrl_keep),), dtype=bool)]) if len(y_ctrl_keep) else np.zeros((len(y_train),), dtype=bool)
        metrics_ctrl, train_meta_ctrl = _fit_eval_linearsvc(
            X_train_ctrl,
            y_train_ctrl,
            tid_train_ctrl,
            X_test,
            y_test,
            tid_test,
            seed=int(seed),
            cap_k=int(args.window_cap_k),
            cap_seed=cap_seed,
            cap_sampling_policy=args.cap_sampling_policy,
            linear_c=float(args.linear_c),
            class_weight="none",
            max_iter=int(args.linear_max_iter),
            agg_mode=args.aggregation_mode,
            is_aug_train=is_aug_ctrl,
        )

        controller_state_df = controller.state_dataframe(base_gamma=float(args.pia_gamma))
        round_history_df = pd.concat(round_history, axis=0, ignore_index=True) if round_history else pd.DataFrame()
        round_summary_df = pd.DataFrame(round_summary_rows)
        ctrl_summary = _controller_summary(controller_state_df)

        setting_tag = f"kdir{int(args.kdir)}_s{int(args.subset_size)}_ctrlR{int(args.controller_rounds)}"
        setting_dir = os.path.join(out_root, setting_tag, f"seed{seed}")
        for cond in ["A_baseline", "Ck_ref", "Ck_controller"]:
            ensure_dir(os.path.join(setting_dir, cond))

        common_meta = {
            "dataset": dataset,
            "seed": int(seed),
            "split_hash": split_meta["split_hash"],
            "train_count_trials": int(split_meta["train_count_trials"]),
            "test_count_trials": int(split_meta["test_count_trials"]),
            "train_trial_ids_preview": split_meta["train_trial_ids"][: max(0, int(args.split_preview_n))],
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
            "controller_enabled": True,
            "controller_phase": args.controller_phase,
            "controller_mode": "weight_plus_gamma_update" if bool(args.controller_update_gamma_scale) else "weight_update_only",
            "controller_rounds": int(args.controller_rounds),
            "controller_tau": float(args.controller_tau),
            "controller_kappa": float(args.controller_kappa),
            "controller_lambda_ema": float(args.controller_lambda_ema),
            "controller_freeze_M": int(args.controller_freeze_m),
            "controller_update_sampling_weight": True,
            "controller_update_gamma_scale": bool(args.controller_update_gamma_scale),
            "controller_enable_freeze": False,
            "controller_gamma_scale_min": float(args.controller_gamma_scale_min),
            "controller_gamma_scale_max": float(args.controller_gamma_scale_max),
            "controller_reward_weights": {
                "alpha": float(args.controller_alpha),
                "beta": float(args.controller_beta),
                "gamma": float(args.controller_gamma),
                "eta": float(args.controller_eta),
            },
            "controller_feedback_scope": "generated_candidates_with_gate_accept_flag_and_generated_level_risk_metrics",
            "controller_gamma_update_enabled": bool(args.controller_update_gamma_scale),
            "controller_freeze_enabled": False,
            "gate1_fit": gate1_fit_meta,
            "gate2_config": {"enabled": True, "q_src": float(args.gate2_q_src)},
            "gate3_enabled": bool(args.enable_gate3),
            "pia_controller_summary": ctrl_summary,
            "runtime_sec": {"total_runtime": float(time.perf_counter() - seed_t0)},
        }

        _write_condition(
            os.path.join(setting_dir, "A_baseline"),
            {**metrics_a, "condition": "A_baseline"},
            {**common_meta, "condition": "A_baseline", "train_meta": train_meta_a},
        )
        _write_condition(
            os.path.join(setting_dir, "Ck_ref"),
            {**metrics_ref, "condition": "Ck_ref"},
            {**common_meta, "condition": "Ck_ref", "train_meta": train_meta_ref, "augmentation": aug_meta_ref, "gate_apply": gate12_meta_ref},
        )
        _write_condition(
            os.path.join(setting_dir, "Ck_controller"),
            {**metrics_ctrl, "condition": "Ck_controller"},
            {**common_meta, "condition": "Ck_controller", "train_meta": train_meta_ctrl},
        )
        controller_state_df.to_csv(os.path.join(setting_dir, "Ck_controller", "pia_controller_state.csv"), index=False)
        round_history_df.to_csv(os.path.join(setting_dir, "Ck_controller", "pia_controller_round_stats.csv"), index=False)
        round_summary_df.to_csv(os.path.join(setting_dir, "Ck_controller", "pia_controller_round_summary.csv"), index=False)
        pd.DataFrame([ctrl_summary]).to_json(
            os.path.join(setting_dir, "Ck_controller", "pia_controller_summary.json"),
            orient="records",
            indent=2,
        )

        summary_rows.extend(
            [
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "condition": "A_baseline",
                    "setting": setting_tag,
                    "split_hash": split_meta["split_hash"],
                    "trial_acc": metrics_a["trial_acc"],
                    "trial_macro_f1": metrics_a["trial_macro_f1"],
                    "accept_rate_total": 1.0,
                    "controller_enabled": False,
                    "controller_rounds": 0,
                    "controller_mean_reward": None,
                    "controller_max_weight": None,
                    "controller_n_frozen": None,
                },
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "condition": "Ck_ref",
                    "setting": setting_tag,
                    "split_hash": split_meta["split_hash"],
                    "trial_acc": metrics_ref["trial_acc"],
                    "trial_macro_f1": metrics_ref["trial_macro_f1"],
                    "accept_rate_total": gate12_meta_ref["accept_rate_final"] * (float(gate3_diag_ref["gate3_accept_rate"]) if gate3_diag_ref else 1.0),
                    "controller_enabled": False,
                    "controller_rounds": 0,
                    "controller_mean_reward": None,
                    "controller_max_weight": None,
                    "controller_n_frozen": None,
                },
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "condition": "Ck_controller",
                    "setting": setting_tag,
                    "split_hash": split_meta["split_hash"],
                    "trial_acc": metrics_ctrl["trial_acc"],
                    "trial_macro_f1": metrics_ctrl["trial_macro_f1"],
                    "accept_rate_total": float(len(y_ctrl_keep) / max(1, len(y_train))),
                    "controller_enabled": True,
                    "controller_rounds": int(args.controller_rounds),
                    "controller_mean_reward": ctrl_summary["mean_reward"],
                    "controller_max_weight": ctrl_summary["max_weight"],
                    "controller_mean_gamma_scale": ctrl_summary["mean_gamma_scale"],
                    "controller_max_gamma_scale": ctrl_summary["max_gamma_scale"],
                    "controller_min_gamma_scale": ctrl_summary["min_gamma_scale"],
                    "controller_n_frozen": ctrl_summary["n_frozen"],
                },
            ]
        )

        print(
            f"[{args.controller_phase}][{dataset}][seed={seed}] "
            f"A={metrics_a['trial_macro_f1']:.4f} "
            f"Ck_ref={metrics_ref['trial_macro_f1']:.4f} "
            f"Ck_controller={metrics_ctrl['trial_macro_f1']:.4f} "
            f"controller_max_weight={ctrl_summary['max_weight']:.3f} "
            f"controller_max_gamma={ctrl_summary['max_gamma_scale']:.3f}"
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["seed", "condition"]).reset_index(drop=True)
    summary_df.to_csv(os.path.join(out_root, "summary_per_seed.csv"), index=False)
    agg_df = (
        summary_df.groupby(["dataset", "condition"], dropna=False)
        .agg(
            trial_acc_mean=("trial_acc", "mean"),
            trial_macro_f1_mean=("trial_macro_f1", "mean"),
            accept_rate_total_mean=("accept_rate_total", "mean"),
            controller_mean_reward_mean=("controller_mean_reward", "mean"),
            controller_max_weight_mean=("controller_max_weight", "mean"),
            controller_mean_gamma_scale_mean=("controller_mean_gamma_scale", "mean"),
            controller_max_gamma_scale_mean=("controller_max_gamma_scale", "mean"),
        )
        .reset_index()
    )
    agg_df.to_csv(os.path.join(out_root, "summary_agg.csv"), index=False)


if __name__ == "__main__":
    main()
