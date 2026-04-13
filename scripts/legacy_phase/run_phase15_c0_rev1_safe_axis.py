#!/usr/bin/env python
"""Phase C0-rev1-v2-A: safe augmentation axis scoring with normalized Expand/Risk."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

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
from scripts.legacy_phase.run_phase14r_step6b1_rev2 import (  # noqa: E402
    covs_to_features,
    ensure_dir,
    extract_features_block,
    logm_spd,
)
from scripts.support.fisher_pia_utils import (  # noqa: E402
    FisherPIAConfig,
    compute_fisher_pia_terms,
    compute_generic_score_correlations,
    compute_safe_axis_scores,
    summarize_generic_score_signal,
)
from scripts.legacy_phase.run_phase15_k1_knn_gate import _apply_gate12_with_diag, _merge_gate3_into_dir_profile  # noqa: E402
from scripts.support.local_knn_gate import LocalKNNGateConfig, ReadOnlyLocalKNNGate  # noqa: E402
from scripts.legacy_phase.run_phase15_step1a_maxplane import _apply_window_cap, _fit_gate1_from_train, _make_trial_split  # noqa: E402
from scripts.legacy_phase.run_phase15_step1b_multidir_matrix import (  # noqa: E402
    _build_direction_bank_d1,
    _build_multidir_aug_candidates,
    _compute_mech_metrics,
    _write_condition,
)


def _parse_csv_list(text: str) -> List[str]:
    out = [t.strip() for t in str(text).split(",") if t.strip()]
    if not out:
        raise ValueError("list cannot be empty")
    return out


def _parse_int_list(text: str) -> List[int]:
    return sorted(set(int(t.strip()) for t in str(text).split(",") if t.strip()))


def _parse_float_list(text: str) -> List[float]:
    return [float(t.strip()) for t in str(text).split(",") if t.strip()]


def _logcenter_train_only(covs_train: np.ndarray, eps: float) -> np.ndarray:
    covs = np.asarray(covs_train, dtype=np.float32)
    if covs.size == 0:
        return np.empty((0,), dtype=np.float32)
    log_train = np.array([logm_spd(c, eps) for c in covs], dtype=np.float32)
    mean_log = np.mean(log_train, axis=0)
    return (log_train - mean_log).astype(np.float32)


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


def _merge_global_scores_with_mech(
    global_df: pd.DataFrame,
    *,
    mech: Dict[str, object],
    intrusion_by_dir: Dict[int, float],
) -> pd.DataFrame:
    df = global_df.copy()
    profile = mech.get("dir_profile", {})
    if not isinstance(profile, dict):
        profile = {}
    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        did = int(row["direction_id"])
        mech_row = profile.get(str(did), {})
        if not isinstance(mech_row, dict):
            mech_row = {}
        merged = dict(row.to_dict())
        merged.update(
            {
                "usage": mech_row.get("usage"),
                "accept_rate": mech_row.get("accept_rate"),
                "flip_rate": mech_row.get("flip_rate"),
                "margin_drop_median": mech_row.get("margin_drop_median"),
                "n_gen": mech_row.get("n_gen"),
                "n_acc": mech_row.get("n_acc"),
                "gate3_reject_rate_i": mech_row.get("gate3_reject_rate_i"),
                "gate3_first_reject_gamma_i": mech_row.get("gate3_first_reject_gamma_i"),
                "gate3_mean_purity_i": mech_row.get("gate3_mean_purity_i"),
                "gate3_mean_intrusion_i": mech_row.get("gate3_mean_intrusion_i"),
                "intrusion": intrusion_by_dir.get(did),
            }
        )
        rows.append(merged)
    return pd.DataFrame(rows).sort_values("direction_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="natops,har")
    parser.add_argument("--seeds", type=str, default="3")
    parser.add_argument("--betas", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--include-approach", action="store_true")
    parser.add_argument("--gammas", type=str, default="0.5,1.0")
    parser.add_argument("--out-root", type=str, default="out/phase15_c0_rev1_safe_axis")
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
    parser.add_argument("--mech-max-real-knn-ref", type=int, default=3000)
    parser.add_argument("--mech-max-real-knn-query", type=int, default=300)
    parser.add_argument("--fisher-knn-k", type=int, default=20)
    parser.add_argument("--fisher-boundary-quantile", type=float, default=0.30)
    parser.add_argument("--fisher-interior-quantile", type=float, default=0.70)
    parser.add_argument("--fisher-hetero-k", type=int, default=3)
    parser.add_argument("--split-preview-n", type=int, default=5)
    args = parser.parse_args()

    datasets = [normalize_dataset_name(d) for d in _parse_csv_list(args.datasets)]
    seeds = _parse_int_list(args.seeds)
    betas = _parse_float_list(args.betas)
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
        print(f"[C0-rev1-v2][{dataset}] load")
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
            print(f"[C0-rev1-v2][{dataset}][seed={seed}] start")
            train_trials, _, split_meta = _make_trial_split(all_trials, seed=int(seed))
            covs_train, y_train, tid_train = extract_features_block(
                train_trials, args.window_sec, args.hop_sec, args.cov_est, args.spd_eps, bands
            )
            covs_train_lc = _logcenter_train_only(covs_train, args.spd_eps)
            X_train = covs_to_features(covs_train_lc).astype(np.float32)
            y_train = np.asarray(y_train).astype(int).ravel()
            tid_train = np.asarray(tid_train)
            print(f"[C0-rev1-v2][{dataset}][seed={seed}] train_windows={len(y_train)} dim={X_train.shape[1]}")

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
            X_aug, y_aug, tid_aug, src_aug, dir_aug, aug_meta = _build_multidir_aug_candidates(
                X_train=X_train,
                y_train=y_train,
                tid_train=tid_train,
                direction_bank=direction_bank,
                subset_size=int(args.subset_size),
                gamma=float(args.pia_gamma),
                multiplier=int(args.pia_multiplier),
                seed=int(seed + 100000 + int(args.kdir) * 101 + int(args.subset_size) * 7),
            )
            X_keep, y_keep, tid_keep, src_keep, keep1, keep2, gate12_meta = _apply_gate12_with_diag(
                X_aug,
                y_aug,
                tid_aug,
                src_aug,
                mu_y=mu_gate1,
                tau_y=tau_gate1,
                gate2_q_src=float(args.gate2_q_src),
            )
            keep12 = keep1 & keep2
            dir_keep = np.asarray(dir_aug, dtype=np.int64)[keep12]

            gate3_diag: Dict[str, object] | None = None
            if bool(args.enable_gate3):
                X_anchor, y_anchor, tid_anchor, _, _, _ = _apply_window_cap(
                    X_train,
                    y_train,
                    tid_train,
                    cap_k=int(args.gate3_anchor_cap_k),
                    seed=int(seed) + 7301,
                    is_aug=np.zeros((len(y_train),), dtype=bool),
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
                keep3, gate3_diag = gate3.evaluate_batch(
                    X_keep,
                    y_keep,
                    direction_ids=dir_keep,
                    gamma_used=np.full((len(y_keep),), float(args.pia_gamma), dtype=np.float64),
                    source_tids=tid_keep,
                )
                X_keep = X_keep[keep3]
                y_keep = y_keep[keep3]
                tid_keep = tid_keep[keep3]
                src_keep = src_keep[keep3]
                dir_keep = dir_keep[keep3]
            mech = _compute_mech_metrics(
                X_train_real=X_train,
                y_train_real=y_train,
                X_aug_generated=X_aug,
                y_aug_generated=y_aug,
                X_aug_accepted=X_keep,
                y_aug_accepted=y_keep,
                X_src_accepted=src_keep,
                dir_generated=dir_aug,
                dir_accepted=dir_keep,
                seed=int(seed),
                linear_c=1.0,
                class_weight="none",
                linear_max_iter=1000,
                knn_k=int(args.mech_knn_k),
                max_aug_for_mech=int(args.mech_max_aug_for_metrics),
                max_real_knn_ref=int(args.mech_max_real_knn_ref),
                max_real_knn_query=int(args.mech_max_real_knn_query),
            )
            if gate3_diag is not None:
                mech = _merge_gate3_into_dir_profile(
                    mech,
                    dir_ids_in=np.asarray(dir_aug, dtype=np.int64)[keep12],
                    keep3=np.asarray(keep3, dtype=bool),
                    gate3_diag=gate3_diag,
                    gamma_value=float(args.pia_gamma),
                )

            class_terms, terms_meta = compute_fisher_pia_terms(X_train, y_train, cfg=fisher_cfg)
            intrusion_by_dir = _compute_direction_intrusion(
                X_anchor=X_train,
                y_anchor=y_train,
                X_aug_accepted=X_keep,
                y_aug_accepted=y_keep,
                dir_accepted=dir_keep,
                seed=int(seed),
                knn_k=int(args.mech_knn_k),
                max_eval=int(args.mech_max_aug_for_metrics),
            )

            gamma_list = gammas if bool(args.include_approach) else [0.0]
            for beta in betas:
                for gamma in gamma_list:
                    class_df, global_df = compute_safe_axis_scores(
                        direction_bank,
                        class_terms,
                        beta=float(beta),
                        gamma=float(gamma),
                        include_approach=bool(args.include_approach),
                        direction_score_mode="axis_level",
                    )
                    global_df = _merge_global_scores_with_mech(global_df, mech=mech, intrusion_by_dir=intrusion_by_dir)
                    corr_df = compute_generic_score_correlations(global_df, score_name="revised_score")
                    signal_meta = summarize_generic_score_signal(
                        global_df,
                        score_name="revised_score",
                        lower_is_better_metrics=["intrusion", "flip_rate", "gate3_reject_rate_i"],
                    )
                    combined_df = pd.concat([class_df, global_df], axis=0, ignore_index=True, sort=False)

                    beta_tag = str(float(beta)).replace(".", "p")
                    setting_tag = f"kdir{int(args.kdir)}_s{int(args.subset_size)}__beta{beta_tag}"
                    if bool(args.include_approach):
                        gamma_tag = str(float(gamma)).replace(".", "p")
                        setting_tag += f"__gamma{gamma_tag}"
                    run_dir = os.path.join(args.out_root, dataset, setting_tag, f"seed{seed}")
                    ensure_dir(run_dir)

                    metrics = {
                        "dataset": dataset,
                        "representation_space": "z",
                        "stage_id": "C0-rev1-v2-B" if bool(args.include_approach) else "C0-rev1-v2-A",
                        "lookback": 1,
                        "horizon": 1,
                        "direction_source": "step1b_d1_bank",
                        "transition_model": "none_offline_safe_axis_score",
                        "center_update_mode": "none",
                        "loss_mode": "offline_safe_axis_diagnosis",
                        "n_dirs": int(direction_bank.shape[0]),
                        "direction_score_mode": "axis_level",
                        "eval_metrics": ["revised_score", "accept_rate", "flip_rate", "margin_drop_median", "intrusion"],
                        **signal_meta,
                    }
                    run_meta = {
                        "dataset": dataset,
                        "seed": int(seed),
                        "split_hash": split_meta["split_hash"],
                        "train_count_trials": int(split_meta["train_count_trials"]),
                        "test_count_trials": int(split_meta["test_count_trials"]),
                        "train_trial_ids_preview": split_meta["train_trial_ids"][: max(0, int(args.split_preview_n))],
                        "representation_space": "z",
                        "stage_id": "C0-rev1-v2-B" if bool(args.include_approach) else "C0-rev1-v2-A",
                        "lookback": 1,
                        "horizon": 1,
                        "direction_source": "step1b_d1_bank",
                        "transition_model": "none_offline_safe_axis_score",
                        "center_update_mode": "none",
                        "loss_mode": "offline_safe_axis_diagnosis",
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
                        "c0_rev1_enabled": True,
                        "c0_rev1_version": "v2",
                        "c0_rev1_score_mode": "expand_minus_risk_minus_approach" if bool(args.include_approach) else "expand_minus_risk",
                        "c0_rev1_direction_score_mode": "axis_level",
                        "c0_rev1_normalization": "minmax_per_bank",
                        "c0_rev1_beta": float(beta),
                        "c0_rev1_gamma": float(gamma),
                        "c0_rev1_expand_definition": "class_weighted_projected_interior_spread",
                        "c0_rev1_risk_definition": "class_weighted_boundary_hetero_neighbor_risk",
                        "c0_rev1_approach_definition": "class_weighted_positive_dot_to_other_class_means" if bool(args.include_approach) else None,
                        "c0_rev1_datasets": datasets,
                        "fisher_terms_meta": terms_meta,
                        "score_signal": signal_meta,
                        "mech": mech,
                    }
                    _write_condition(run_dir, metrics, run_meta)
                    global_df.to_csv(os.path.join(run_dir, "c0_rev1_direction_table.csv"), index=False)
                    combined_df.to_csv(os.path.join(run_dir, "c0_rev1_direction_table_full.csv"), index=False)
                    corr_df.to_csv(os.path.join(run_dir, "c0_rev1_score_correlation.csv"), index=False)

                    row = {
                        "dataset": dataset,
                        "seed": int(seed),
                        "beta": float(beta),
                        "gamma": float(gamma),
                        "setting": setting_tag,
                        "split_hash": split_meta["split_hash"],
                        "score_signal_pass": bool(signal_meta["score_signal_pass"]),
                        "top_direction_id": signal_meta.get("top_direction_id"),
                        "bottom_direction_id": signal_meta.get("bottom_direction_id"),
                        "delta_accept_top_vs_bottom": signal_meta.get("delta_accept_rate_top_vs_bottom"),
                        "delta_intrusion_top_vs_bottom": signal_meta.get("delta_intrusion_top_vs_bottom"),
                        "delta_flip_top_vs_bottom": signal_meta.get("delta_flip_rate_top_vs_bottom"),
                        "delta_margin_top_vs_bottom": signal_meta.get("delta_margin_drop_median_top_vs_bottom"),
                    }
                    for _, corr_row in corr_df.iterrows():
                        row[f"pearson__{corr_row['metric_name']}"] = corr_row["pearson_r"]
                        row[f"spearman__{corr_row['metric_name']}"] = corr_row["spearman_rho"]
                    summary_rows.append(row)

                    print(
                        f"[C0-rev1-v2][{dataset}][seed={seed}][beta={beta}][gamma={gamma}] "
                        f"score_pass={signal_meta['score_signal_pass']} "
                        f"top={signal_meta.get('top_direction_id')} "
                        f"bottom={signal_meta.get('bottom_direction_id')}"
                    )

    summary_df = pd.DataFrame(summary_rows).sort_values(["dataset", "seed", "beta"]).reset_index(drop=True)
    summary_df.to_csv(os.path.join(args.out_root, "summary_per_run.csv"), index=False)
    agg_df = summary_df.groupby(["dataset", "beta"], as_index=False).mean(numeric_only=True)
    agg_df.to_csv(os.path.join(args.out_root, "summary_agg.csv"), index=False)


if __name__ == "__main__":
    main()
