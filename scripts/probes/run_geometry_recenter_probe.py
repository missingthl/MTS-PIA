#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from datasets.trial_dataset_factory import (  # noqa: E402
    DEFAULT_BANDS_EEG,
    DEFAULT_FINGERMOVEMENTS_ROOT,
    DEFAULT_HAR_ROOT,
    DEFAULT_SELFREGULATIONSCP1_ROOT,
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
    vec_utri,
    write_json,
)
from scripts.legacy_phase.run_phase15_mainline_freeze import _make_protocol_split  # noqa: E402
from scripts.legacy_phase.run_phase15_step1a_maxplane import _fit_eval_linearsvc  # noqa: E402
from scripts.legacy_phase.run_phase15_step1b_multidir_matrix import (  # noqa: E402
    _build_direction_bank_d1,
    _build_multidir_aug_candidates,
)
from transforms.whiten_color_bridge import unvec_utri_sym  # noqa: E402


DATASETS_DEFAULT = "har,selfregulationscp1,fingermovements"
SEEDS_DEFAULT = "1,2,3"
GEOMETRY_DELTA_THRESHOLD = 0.005


def _parse_csv_list(text: str) -> List[str]:
    out = []
    for tok in str(text).split(","):
        t = tok.strip()
        if t:
            out.append(t)
    if not out:
        raise ValueError("list cannot be empty")
    return out


def _parse_seed_list(text: str) -> List[int]:
    return [int(x) for x in _parse_csv_list(text)]


def _fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def _stable_cap_seed(dataset: str) -> int:
    return 20260320 + sum(ord(ch) for ch in str(dataset))


def _centered_features_from_mean(
    log_train: np.ndarray,
    log_test: np.ndarray,
    mean_log: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    centered_train = np.asarray(log_train - mean_log, dtype=np.float32)
    centered_test = np.asarray(log_test - mean_log, dtype=np.float32)
    return covs_to_features(centered_train).astype(np.float32), covs_to_features(centered_test).astype(np.float32)


def _center_shift_stats(mean_log_orig: np.ndarray, mean_log_recenter: np.ndarray) -> Dict[str, float]:
    delta = np.asarray(mean_log_recenter - mean_log_orig, dtype=np.float64)
    fro = float(np.linalg.norm(delta))
    rel = float(fro / (np.linalg.norm(mean_log_orig) + 1e-12))
    vec_l2 = float(np.linalg.norm(vec_utri(delta)))
    return {
        "center_shift_distance": fro,
        "center_shift_relative": rel,
        "reference_vec_shift_l2": vec_l2,
    }


def _shift_size_label(center_shift_relative: float) -> str:
    if center_shift_relative < 0.01:
        return "small"
    if center_shift_relative < 0.05:
        return "moderate"
    return "large"


def _geometry_value_label(delta_f1: float) -> str:
    if delta_f1 > GEOMETRY_DELTA_THRESHOLD:
        return "geometry_helpful"
    if delta_f1 < -GEOMETRY_DELTA_THRESHOLD:
        return "geometry_harmful"
    return "geometry_neutral"


def _role_label(delta_recenter_vs_baseline: float, delta_direct_vs_baseline: float) -> str:
    helpful_recenter = delta_recenter_vs_baseline > GEOMETRY_DELTA_THRESHOLD
    helpful_direct = delta_direct_vs_baseline > GEOMETRY_DELTA_THRESHOLD
    if helpful_recenter and not helpful_direct:
        return "geometry_auxiliary_sample"
    if helpful_direct and not helpful_recenter:
        return "training_sample"
    if helpful_recenter and helpful_direct:
        return "both_have_value"
    return "both_not_enough"


def _aggregate_condition(df: pd.DataFrame, prefix: str, condition: str) -> Dict[str, object]:
    row = df[df["condition"] == condition].copy()
    if row.empty:
        return {
            f"{prefix}_acc_mean": np.nan,
            f"{prefix}_acc_std": np.nan,
            f"{prefix}_f1_mean": np.nan,
            f"{prefix}_f1_std": np.nan,
            f"{prefix}_acc_mean_std": "n/a",
            f"{prefix}_f1_mean_std": "n/a",
        }
    acc_mean = float(row["trial_acc"].mean())
    acc_std = float(row["trial_acc"].std(ddof=0))
    f1_mean = float(row["trial_macro_f1"].mean())
    f1_std = float(row["trial_macro_f1"].std(ddof=0))
    return {
        f"{prefix}_acc_mean": acc_mean,
        f"{prefix}_acc_std": acc_std,
        f"{prefix}_f1_mean": f1_mean,
        f"{prefix}_f1_std": f1_std,
        f"{prefix}_acc_mean_std": _fmt_mean_std(acc_mean, acc_std),
        f"{prefix}_f1_mean_std": _fmt_mean_std(f1_mean, f1_std),
    }


def _run_condition(
    *,
    dataset: str,
    condition: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    tid_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tid_test: np.ndarray,
    is_aug_train: np.ndarray,
    seed: int,
    cap_seed: int,
    args,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    metrics, train_meta = _fit_eval_linearsvc(
        X_train,
        y_train,
        tid_train,
        X_test,
        y_test,
        tid_test,
        seed=int(seed),
        cap_k=int(args.window_cap_k),
        cap_seed=int(cap_seed),
        cap_sampling_policy=args.cap_sampling_policy,
        linear_c=float(args.linear_c),
        class_weight=args.linear_class_weight,
        max_iter=int(args.linear_max_iter),
        agg_mode=args.aggregation_mode,
        is_aug_train=np.asarray(is_aug_train, dtype=bool),
        progress_prefix=f"[geom-recenter][{dataset}][seed={seed}][{condition}]",
    )
    return metrics, train_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Geometry recenter diagnostic probe.")
    parser.add_argument("--datasets", type=str, default=DATASETS_DEFAULT)
    parser.add_argument("--seeds", type=str, default=SEEDS_DEFAULT)
    parser.add_argument("--out-root", type=str, default="out/geometry_recenter_probe_20260320")
    parser.add_argument("--har-root", type=str, default=DEFAULT_HAR_ROOT)
    parser.add_argument("--fingermovements-root", type=str, default=DEFAULT_FINGERMOVEMENTS_ROOT)
    parser.add_argument("--selfregulationscp1-root", type=str, default=DEFAULT_SELFREGULATIONSCP1_ROOT)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--hop-sec", type=float, default=1.0)
    parser.add_argument("--cov-est", type=str, default="sample", choices=["sample", "oas", "ledoitwolf"])
    parser.add_argument("--spd-eps", type=float, default=1e-4)
    parser.add_argument("--bands", type=str, default=DEFAULT_BANDS_EEG)
    parser.add_argument("--aggregation-mode", type=str, default="majority")
    parser.add_argument("--window-cap-k", type=int, default=120)
    parser.add_argument(
        "--cap-sampling-policy",
        type=str,
        default="balanced_real_aug",
        choices=["random", "balanced_real_aug", "prefer_real", "prefer_aug"],
    )
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
    parser.add_argument("--split-preview-n", type=int, default=5)
    args = parser.parse_args()

    datasets = [normalize_dataset_name(x) for x in _parse_csv_list(args.datasets)]
    seeds = _parse_seed_list(args.seeds)
    ensure_dir(args.out_root)

    summary_rows: List[Dict[str, object]] = []
    ref_shift_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        if dataset not in {"har", "selfregulationscp1", "fingermovements"}:
            raise ValueError(f"geometry recenter probe only supports simple-set datasets, got: {dataset}")

        dataset_dir = os.path.join(args.out_root, dataset)
        ensure_dir(dataset_dir)

        trials = load_trials_for_dataset(
            dataset=dataset,
            har_root=args.har_root,
            fingermovements_root=args.fingermovements_root,
            selfregulationscp1_root=args.selfregulationscp1_root,
        )
        train_trials, test_trials, split_meta = _make_protocol_split(dataset, trials)
        bands_spec = resolve_band_spec(dataset, args.bands)
        bands = parse_band_spec(bands_spec)
        cap_seed = _stable_cap_seed(dataset)

        per_seed_rows: List[Dict[str, object]] = []
        ref_seed_rows: List[Dict[str, object]] = []

        for seed in seeds:
            seed_dir = os.path.join(dataset_dir, f"seed{seed}")
            ensure_dir(seed_dir)

            covs_train, y_train, tid_train = extract_features_block(
                train_trials,
                args.window_sec,
                args.hop_sec,
                args.cov_est,
                args.spd_eps,
                bands,
                progress_prefix=f"[geom-recenter][{dataset}][seed={seed}][extract_train]",
                progress_every=0,
            )
            covs_test, y_test, tid_test = extract_features_block(
                test_trials,
                args.window_sec,
                args.hop_sec,
                args.cov_est,
                args.spd_eps,
                bands,
                progress_prefix=f"[geom-recenter][{dataset}][seed={seed}][extract_test]",
                progress_every=0,
            )
            y_train = np.asarray(y_train).astype(int).ravel()
            y_test = np.asarray(y_test).astype(int).ravel()
            tid_train = np.asarray(tid_train)
            tid_test = np.asarray(tid_test)

            log_train = np.asarray([logm_spd(c, args.spd_eps) for c in covs_train], dtype=np.float32)
            log_test = np.asarray([logm_spd(c, args.spd_eps) for c in covs_test], dtype=np.float32)
            mean_log_orig = np.mean(log_train, axis=0).astype(np.float32)
            X_train_base, X_test_base = _centered_features_from_mean(log_train, log_test, mean_log_orig)

            metrics_base, meta_base = _run_condition(
                dataset=dataset,
                condition="baseline_orig_plane",
                X_train=X_train_base,
                y_train=y_train,
                tid_train=tid_train,
                X_test=X_test_base,
                y_test=y_test,
                tid_test=tid_test,
                is_aug_train=np.zeros((len(y_train),), dtype=bool),
                seed=int(seed),
                cap_seed=int(cap_seed),
                args=args,
            )

            direction_bank, bank_meta = _build_direction_bank_d1(
                X_train=X_train_base,
                k_dir=int(args.k_dir),
                seed=int(seed * 10000 + args.k_dir * 113 + 17),
                n_iters=int(args.pia_n_iters),
                activation=str(args.pia_activation),
                bias_update_mode=str(args.pia_bias_update_mode),
                c_repr=float(args.pia_c_repr),
            )
            X_aug, y_aug, tid_aug, src_aug, dir_aug, aug_meta = _build_multidir_aug_candidates(
                X_train=X_train_base,
                y_train=y_train,
                tid_train=tid_train,
                direction_bank=direction_bank,
                subset_size=int(args.subset_size),
                gamma=float(args.pia_gamma),
                multiplier=int(args.pia_multiplier),
                seed=int(seed + 100000 + args.k_dir * 101 + args.subset_size * 7),
            )

            X_train_direct = np.vstack([X_train_base, X_aug]) if len(y_aug) else X_train_base.copy()
            y_train_direct = np.concatenate([y_train, y_aug]) if len(y_aug) else y_train.copy()
            tid_train_direct = np.concatenate([tid_train, tid_aug]) if len(y_aug) else tid_train.copy()
            is_aug_direct = (
                np.concatenate(
                    [np.zeros((len(y_train),), dtype=bool), np.ones((len(y_aug),), dtype=bool)]
                )
                if len(y_aug)
                else np.zeros((len(y_train),), dtype=bool)
            )

            metrics_direct, meta_direct = _run_condition(
                dataset=dataset,
                condition="step1b_direct_train",
                X_train=X_train_direct,
                y_train=y_train_direct,
                tid_train=tid_train_direct,
                X_test=X_test_base,
                y_test=y_test,
                tid_test=tid_test,
                is_aug_train=is_aug_direct,
                seed=int(seed),
                cap_seed=int(cap_seed),
                args=args,
            )

            dim_cov = int(covs_train.shape[1])
            if len(X_aug):
                aug_centered_mats = np.asarray([unvec_utri_sym(v, dim_cov) for v in X_aug], dtype=np.float32)
                aug_log_mats = aug_centered_mats + mean_log_orig[None, :, :]
                mean_log_recenter = np.mean(np.concatenate([log_train, aug_log_mats], axis=0), axis=0).astype(np.float32)
            else:
                mean_log_recenter = mean_log_orig.copy()

            X_train_recenter, X_test_recenter = _centered_features_from_mean(log_train, log_test, mean_log_recenter)
            metrics_recenter, meta_recenter = _run_condition(
                dataset=dataset,
                condition="aug_recenter_orig_train",
                X_train=X_train_recenter,
                y_train=y_train,
                tid_train=tid_train,
                X_test=X_test_recenter,
                y_test=y_test,
                tid_test=tid_test,
                is_aug_train=np.zeros((len(y_train),), dtype=bool),
                seed=int(seed),
                cap_seed=int(cap_seed),
                args=args,
            )

            shift_stats = _center_shift_stats(mean_log_orig, mean_log_recenter)
            shift_size = _shift_size_label(shift_stats["center_shift_relative"])
            ref_shift_row = {
                "dataset": dataset,
                "seed": int(seed),
                "center_shift_distance": float(shift_stats["center_shift_distance"]),
                "center_shift_relative": float(shift_stats["center_shift_relative"]),
                "reference_vec_shift_l2": float(shift_stats["reference_vec_shift_l2"]),
                "reference_plane_shift_summary": (
                    f"reestimate=global_log_center_only;"
                    f"fro={shift_stats['center_shift_distance']:.6f};"
                    f"rel={shift_stats['center_shift_relative']:.6f};"
                    f"vec_l2={shift_stats['reference_vec_shift_l2']:.6f}"
                ),
                "shift_size_label": shift_size,
            }
            ref_seed_rows.append(ref_shift_row)

            common_run_meta = {
                "dataset": dataset,
                "seed": int(seed),
                "protocol_type": split_meta["protocol_type"],
                "protocol_note": split_meta["protocol_note"],
                "split_hash": split_meta["split_hash"],
                "bands": bands_spec,
                "window_sec": float(args.window_sec),
                "hop_sec": float(args.hop_sec),
                "cov_est": args.cov_est,
                "spd_eps": float(args.spd_eps),
                "geometry_recenter_impl": "global_log_center_reestimate_only",
                "aug_used_for_reference": "step1b_raw_aug_candidates",
                "aug_used_in_supervised_train": False,
                "train_count_trials": int(split_meta["train_count_trials"]),
                "test_count_trials": int(split_meta["test_count_trials"]),
                "train_trial_ids_preview": split_meta["train_trial_ids"][: max(0, int(args.split_preview_n))],
                "test_trial_ids_preview": split_meta["test_trial_ids"][: max(0, int(args.split_preview_n))],
                "cap_seed_fixed_per_dataset": int(cap_seed),
                "step1b_config": {
                    "k_dir": int(args.k_dir),
                    "subset_size": int(args.subset_size),
                    "pia_multiplier": int(args.pia_multiplier),
                    "pia_gamma": float(args.pia_gamma),
                    "pia_n_iters": int(args.pia_n_iters),
                    "pia_activation": str(args.pia_activation),
                    "pia_bias_update_mode": str(args.pia_bias_update_mode),
                    "pia_c_repr": float(args.pia_c_repr),
                },
                "direction_bank_meta": bank_meta,
                "aug_candidate_meta": aug_meta,
                "reference_shift": ref_shift_row,
            }

            condition_payloads = [
                ("baseline_orig_plane", metrics_base, {**common_run_meta, **meta_base, "condition": "baseline_orig_plane"}),
                ("step1b_direct_train", metrics_direct, {**common_run_meta, **meta_direct, "condition": "step1b_direct_train", "aug_used_in_supervised_train": True}),
                ("aug_recenter_orig_train", metrics_recenter, {**common_run_meta, **meta_recenter, "condition": "aug_recenter_orig_train"}),
            ]
            for cond_name, metrics, run_meta in condition_payloads:
                cond_dir = os.path.join(seed_dir, cond_name)
                ensure_dir(cond_dir)
                write_json(os.path.join(cond_dir, "metrics.json"), metrics)
                write_json(os.path.join(cond_dir, "run_meta.json"), run_meta)
                per_seed_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "condition": cond_name,
                        "trial_acc": float(metrics["trial_acc"]),
                        "trial_macro_f1": float(metrics["trial_macro_f1"]),
                        "window_acc": float(metrics["window_acc"]),
                        "window_macro_f1": float(metrics["window_macro_f1"]),
                        "feature_dim": int(run_meta["feature_dim"]),
                        "train_selected_aug_ratio": float(run_meta["train_selected_aug_ratio"]),
                        "source_dir": cond_dir,
                    }
                )

        per_seed_df = pd.DataFrame(per_seed_rows)
        ref_seed_df = pd.DataFrame(ref_seed_rows)
        per_seed_df.to_csv(os.path.join(dataset_dir, "summary_per_seed.csv"), index=False)
        ref_seed_df.to_csv(os.path.join(dataset_dir, "reference_shift_per_seed.csv"), index=False)

        base_stats = _aggregate_condition(per_seed_df, "baseline", "baseline_orig_plane")
        direct_stats = _aggregate_condition(per_seed_df, "step1b_direct_train", "step1b_direct_train")
        recenter_stats = _aggregate_condition(per_seed_df, "aug_recenter_orig_train", "aug_recenter_orig_train")
        delta_vs_baseline = float(recenter_stats["aug_recenter_orig_train_f1_mean"] - base_stats["baseline_f1_mean"])
        delta_vs_step1b_direct = float(
            recenter_stats["aug_recenter_orig_train_f1_mean"] - direct_stats["step1b_direct_train_f1_mean"]
        )
        geometry_label = _geometry_value_label(delta_vs_baseline)
        role_label = _role_label(
            delta_recenter_vs_baseline=delta_vs_baseline,
            delta_direct_vs_baseline=float(direct_stats["step1b_direct_train_f1_mean"] - base_stats["baseline_f1_mean"]),
        )

        summary_rows.append(
            {
                "dataset": dataset,
                **base_stats,
                **direct_stats,
                **recenter_stats,
                "delta_vs_baseline_f1": delta_vs_baseline,
                "delta_vs_baseline_acc": float(
                    recenter_stats["aug_recenter_orig_train_acc_mean"] - base_stats["baseline_acc_mean"]
                ),
                "delta_vs_step1b_direct_f1": delta_vs_step1b_direct,
                "delta_vs_step1b_direct_acc": float(
                    recenter_stats["aug_recenter_orig_train_acc_mean"] - direct_stats["step1b_direct_train_acc_mean"]
                ),
                "geometry_value_label": geometry_label,
                "enhancement_role_label": role_label,
            }
        )

        shift_mean = float(ref_seed_df["center_shift_distance"].mean())
        shift_rel_mean = float(ref_seed_df["center_shift_relative"].mean())
        shift_size_major = pd.Series(ref_seed_df["shift_size_label"]).mode().iat[0]
        ref_shift_rows.append(
            {
                "dataset": dataset,
                "center_shift_distance": shift_mean,
                "center_shift_distance_std": float(ref_seed_df["center_shift_distance"].std(ddof=0)),
                "center_shift_relative_mean": shift_rel_mean,
                "reference_plane_shift_summary": (
                    f"reestimate=global_log_center_only;"
                    f"fro_mean={shift_mean:.6f};"
                    f"rel_mean={shift_rel_mean:.6f};"
                    f"vec_l2_mean={float(ref_seed_df['reference_vec_shift_l2'].mean()):.6f}"
                ),
                "shift_size_label": shift_size_major,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    shift_df = pd.DataFrame(ref_shift_rows)
    summary_df.to_csv(os.path.join(args.out_root, "geometry_recenter_probe_summary.csv"), index=False)
    shift_df.to_csv(os.path.join(args.out_root, "geometry_recenter_reference_shift_summary.csv"), index=False)

    helpful = int((summary_df["geometry_value_label"] == "geometry_helpful").sum())
    neutral = int((summary_df["geometry_value_label"] == "geometry_neutral").sum())
    harmful = int((summary_df["geometry_value_label"] == "geometry_harmful").sum())

    if helpful >= 2:
        overall_judgement = "geometry_reference_value_present"
    elif helpful >= 1 and harmful == 0:
        overall_judgement = "geometry_reference_value_local"
    else:
        overall_judgement = "geometry_reference_value_not_yet_clear"

    conclusion_lines = [
        "# Geometry Recenter Probe Conclusion",
        "",
        "更新时间：2026-03-20",
        "",
        "身份：`diagnostic-only`",
        "",
        "- `not a new mainline`",
        "- `not for Phase15 mainline freeze table`",
        "- `simple-set only`",
        "",
        "## Implementation Lock",
        "",
        "- 当前“重估平面”只实现为：重新估计全局 `log-center` 参考点。",
        "- 增强样本使用的是：`Step1B` 原始增强样本。",
        "- 增强样本不参与任何最终监督训练，只参与参考结构估计。",
        "",
        "## Overall",
        "",
        f"- 当前增强样本更像：`{overall_judgement}`",
        "- 当前是否应继续停留在 simple-set：`是`",
        "- 当前不应扩到 seed 家族：`是`",
        "",
        "## Dataset Readout",
        "",
    ]
    for _, row in summary_df.iterrows():
        conclusion_lines.append(
            f"- `{row['dataset']}`: baseline_f1={row['baseline_f1_mean']:.4f}, "
            f"step1b_direct_f1={row['step1b_direct_train_f1_mean']:.4f}, "
            f"aug_recenter_f1={row['aug_recenter_orig_train_f1_mean']:.4f}, "
            f"geometry_value_label=`{row['geometry_value_label']}`, "
            f"role=`{row['enhancement_role_label']}`"
        )

    conclusion_lines.extend(
        [
            "",
            "## Takeaway",
            "",
            f"- `geometry_helpful` datasets: {helpful}",
            f"- `geometry_neutral` datasets: {neutral}",
            f"- `geometry_harmful` datasets: {harmful}",
            "- 这个 probe 只用于判断增强样本是否具有几何参考价值，不改变当前主线语义。",
        ]
    )
    Path(args.out_root, "geometry_recenter_probe_conclusion.md").write_text(
        "\n".join(conclusion_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
