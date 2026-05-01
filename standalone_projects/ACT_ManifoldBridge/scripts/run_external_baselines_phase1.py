#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.datasets import load_trials_for_dataset, make_trial_split
from utils.backbone_trainers import (
    SUPPORTED_BACKBONES,
    fit_jobda_backbone,
    fit_hard_backbone,
    fit_manifold_mixup_backbone,
    fit_soft_backbone,
)
from utils.external_baselines import (
    ExternalAugResult,
    dba_sameclass,
    dgw_sameclass,
    pca_cov_state,
    random_cov_state,
    rgw_sameclass,
    raw_aug_jitter,
    raw_aug_magnitude_warping,
    raw_aug_scaling,
    raw_aug_timewarp,
    raw_aug_window_slicing,
    raw_aug_window_warping,
    jobda_cleanroom_augmented_set,
    raw_mixup,
    raw_smote_flatten_balanced,
    spawner_sameclass_style,
    timevae_classwise_optional,
    wdba_sameclass,
)
from core.pia_operator import pia_operator_metadata


DEFAULT_DATASETS = [
    "atrialfibrillation",
    "ering",
    "handmovementdirection",
    "handwriting",
    "japanesevowels",
    "natops",
    "racketsports",
]

DEFAULT_ARMS = [
    "no_aug",
    "raw_aug_jitter",
    "raw_aug_scaling",
    "raw_aug_timewarp",
    "raw_mixup",
    "dba_sameclass",
    "raw_smote_flatten_balanced",
    "random_cov_state",
    "pca_cov_state",
    "csta_top1_current",
    "csta_group_template_top",
]

PHASE2_ARMS = [
    "raw_aug_magnitude_warping",
    "raw_aug_window_warping",
    "raw_aug_window_slicing",
    "wdba_sameclass",
    "spawner_sameclass_style",
    "jobda_cleanroom",
    "rgw_sameclass",
    "dgw_sameclass",
]

PHASE3_ARMS = [
    "manifold_mixup",
    "timevae_classwise_optional",
]

CSTA_RESULT_PASSTHROUGH_FIELDS = [
    "transport_error_fro_mean",
    "transport_error_logeuc_mean",
    "bridge_cond_A_mean",
    "metric_preservation_error_mean",
    "safe_radius_ratio_mean",
    "manifold_margin_mean",
    "gamma_requested_mean",
    "gamma_used_mean",
    "gamma_zero_rate",
    "host_geom_cosine_mean",
    "host_conflict_rate",
    "candidate_total_count",
    "aug_total_count",
    "requested_k_dir",
    "effective_k_dir",
    "safe_clip_rate",
    "template_usage_entropy",
    "selected_template_entropy",
    "top_template_concentration",
    "aug_valid_rate",
    "candidate_audit_rows",
    "candidate_audit_available",
    "candidate_accept_rate",
    "candidate_physics_ok",
    "candidate_audit_path",
    "z_displacement_norm_mean",
    "template_response_abs_mean",
    "gamma_requested_mean_audit",
    "gamma_used_mean_audit",
    "safe_radius_ratio_mean_audit",
    "safe_clip_rate_audit",
    "gamma_zero_rate_audit",
    "manifold_margin_mean_audit",
    "transport_error_logeuc_mean_audit",
    "template_usage_entropy_audit",
    "top_template_concentration_audit",
    "gamma_used_gt_requested_count",
    "safe_radius_ratio_out_of_bounds_count",
    "direction_bank_source",
    "utilization_mode",
    "core_training_mode",
    "aug_train_ratio",
    "multi_template_pairs",
    "template_selection",
    "eta_safe",
    "zpia_z_dim",
    "zpia_n_train",
    "zpia_n_train_lt_z_dim",
    "zpia_row_norm_min",
    "zpia_row_norm_max",
    "zpia_row_norm_mean",
    "zpia_fallback_row_count",
    "telm2_recon_last",
    "telm2_recon_mean",
    "telm2_recon_std",
    "telm2_n_iters",
    "telm2_c_repr",
    "telm2_activation",
    "telm2_bias_update_mode",
]

RAW_AUG_METHODS = {
    "raw_aug_jitter",
    "raw_aug_scaling",
    "raw_aug_timewarp",
    "raw_aug_magnitude_warping",
    "raw_aug_window_warping",
    "raw_aug_window_slicing",
}
REF_METHODS = [
    "no_aug",
    "best_rawaug",
    "raw_mixup",
    "dba_sameclass",
    "raw_smote_flatten_balanced",
    "random_cov_state",
    "pca_cov_state",
    "csta_top1_current",
    "csta_group_template_top",
]


@dataclass(frozen=True)
class MethodInfo:
    source_space: str
    label_mode: str
    uses_external_library: bool
    library_name: str
    budget_matched: bool
    selection_rule: str


METHOD_INFO: Dict[str, MethodInfo] = {
    "no_aug": MethodInfo("none", "hard", False, "", True, "none"),
    "raw_aug_jitter": MethodInfo("raw_time", "hard", True, "tsaug", True, "repeat_train_anchors_addnoise"),
    "raw_aug_scaling": MethodInfo("raw_time", "hard", False, "", True, "repeat_train_anchors_amplitude_uniform"),
    "raw_aug_timewarp": MethodInfo("raw_time", "hard", True, "tsaug", True, "repeat_train_anchors_timewarp"),
    "raw_aug_magnitude_warping": MethodInfo(
        "raw_time",
        "hard",
        False,
        "",
        True,
        "repeat_train_anchors_magnitude_warping",
    ),
    "raw_aug_window_warping": MethodInfo(
        "raw_time",
        "hard",
        False,
        "",
        True,
        "repeat_train_anchors_window_warping",
    ),
    "raw_aug_window_slicing": MethodInfo(
        "raw_time",
        "hard",
        False,
        "",
        True,
        "repeat_train_anchors_window_slicing",
    ),
    "raw_mixup": MethodInfo("raw_mixup", "soft", False, "", True, "train_split_random_pair_beta"),
    "dba_sameclass": MethodInfo("dtw_barycenter", "hard", True, "tslearn", True, "same_class_dba"),
    "wdba_sameclass": MethodInfo(
        "dtw_barycenter",
        "hard",
        True,
        "tslearn",
        True,
        "same_class_weighted_dba_anchor_dtw_softmax",
    ),
    "spawner_sameclass_style": MethodInfo(
        "dtw_pattern_mix",
        "hard",
        True,
        "tslearn",
        True,
        "spawner_style_same_class_dtw_aligned_average",
    ),
    "jobda_cleanroom": MethodInfo(
        "raw_time_tsw",
        "joint_hard",
        False,
        "",
        False,
        "jobda_cleanroom_tsw_joint_label",
    ),
    "rgw_sameclass": MethodInfo(
        "dtw_guided_warp",
        "hard",
        False,
        "",
        True,
        "random_guided_warp_same_class_dtw_cleanroom",
    ),
    "dgw_sameclass": MethodInfo(
        "dtw_guided_warp",
        "hard",
        False,
        "",
        True,
        "discriminative_guided_warp_same_class_dtw_cleanroom",
    ),
    "raw_smote_flatten_balanced": MethodInfo(
        "flattened_raw",
        "hard",
        True,
        "imbalanced-learn",
        False,
        "class_balancing_smote_auto",
    ),
    "random_cov_state": MethodInfo("covariance_state", "hard", False, "", True, "random_unit_z_direction"),
    "pca_cov_state": MethodInfo("covariance_state", "hard", False, "", True, "pca_top_z_direction"),
    "csta_top1_current": MethodInfo("covariance_template", "hard", False, "", True, "anchor_top_response"),
    "csta_group_template_top": MethodInfo("covariance_template", "hard", False, "", True, "group_center_top_response"),
    "csta_topk_softmax_tau_0.05": MethodInfo("covariance_template", "hard", False, "", True, "softmax_tau_0.05_response"),
    "csta_topk_softmax_tau_0.10": MethodInfo("covariance_template", "hard", False, "", True, "softmax_tau_0.10_response"),
    "csta_topk_softmax_tau_0.20": MethodInfo("covariance_template", "hard", False, "", True, "softmax_tau_0.20_response"),
    "csta_topk_uniform_top5": MethodInfo("covariance_template", "hard", False, "", True, "uniform_top5_response"),
    "manifold_mixup": MethodInfo("hidden_state", "soft", False, "", False, "resnet1d_hidden_state_beta_mixup"),
    "timevae_classwise_optional": MethodInfo(
        "generative_model",
        "hard",
        False,
        "",
        False,
        "classwise_timevae_style_pytorch_cleanroom",
    ),
}


def _parse_csv(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _stack_trials(trials) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([t.x for t in trials], axis=0).astype(np.float32)
    y = np.asarray([t.y for t in trials], dtype=np.int64)
    return X, y


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), int(n_classes)), dtype=np.float32)
    out[np.arange(len(y)), y.astype(np.int64)] = 1.0
    return out


def _csta_policy_for_method(method: str) -> str:
    if method == "csta_top1_current":
        return "top1"
    if method == "csta_group_template_top":
        return "group_top"
    if method.startswith("csta_topk_"):
        return method.replace("csta_", "", 1)
    return method


def _clean_metric_value(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _extract_csta_extra_metrics(result_row: Dict[str, object], method: str) -> Dict[str, object]:
    extra = pia_operator_metadata(_csta_policy_for_method(method))
    for field in CSTA_RESULT_PASSTHROUGH_FIELDS:
        if field in result_row:
            extra[field] = _clean_metric_value(result_row[field])
    if "selected_template_entropy" not in extra and "template_usage_entropy" in extra:
        extra["selected_template_entropy"] = extra["template_usage_entropy"]
    return extra


def _fit_hard(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    args,
    seed: int,
):
    return fit_hard_backbone(
        args.backbone,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
        seed=int(seed),
        n_kernels=args.n_kernels,
    )


def _fit_soft(
    X_train,
    y_train_soft,
    X_val,
    y_val,
    X_test,
    y_test,
    args,
    seed: int,
):
    return fit_soft_backbone(
        args.backbone,
        X_train,
        y_train_soft,
        X_val,
        y_val,
        X_test,
        y_test,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
        seed=int(seed),
    )


def _build_external_aug(method: str, X_train, y_train, args, seed: int, n_classes: int) -> ExternalAugResult:
    builders: Dict[str, Callable[[], ExternalAugResult]] = {
        "raw_aug_jitter": lambda: raw_aug_jitter(X_train, y_train, multiplier=args.multiplier, seed=seed),
        "raw_aug_scaling": lambda: raw_aug_scaling(X_train, y_train, multiplier=args.multiplier, seed=seed),
        "raw_aug_timewarp": lambda: raw_aug_timewarp(X_train, y_train, multiplier=args.multiplier, seed=seed),
        "raw_aug_magnitude_warping": lambda: raw_aug_magnitude_warping(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            sigma=args.magnitude_warp_sigma,
            knots=args.magnitude_warp_knots,
            per_channel_curve=not args.magnitude_warp_shared_curve,
        ),
        "raw_aug_window_warping": lambda: raw_aug_window_warping(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            window_ratio=args.window_warp_ratio,
            speed_factors=tuple(float(x) for x in _parse_csv(args.window_warp_speeds)),
            min_window_len=args.window_min_len,
        ),
        "raw_aug_window_slicing": lambda: raw_aug_window_slicing(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            slice_ratio=args.window_slice_ratio,
            min_window_len=args.window_min_len,
        ),
        "raw_mixup": lambda: raw_mixup(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            alpha=args.mixup_alpha,
            n_classes=n_classes,
        ),
        "dba_sameclass": lambda: dba_sameclass(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            k=args.dba_k,
            max_iter=args.dba_max_iter,
        ),
        "wdba_sameclass": lambda: wdba_sameclass(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            k=args.wdba_k,
            max_iter=args.wdba_max_iter,
        ),
        "spawner_sameclass_style": lambda: spawner_sameclass_style(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            noise_scale=args.spawner_noise_scale,
        ),
        "jobda_cleanroom": lambda: jobda_cleanroom_augmented_set(
            X_train,
            y_train,
            transform_subseqs=tuple(int(x) for x in _parse_csv(args.jobda_transform_subseqs)),
        ),
        "rgw_sameclass": lambda: rgw_sameclass(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            slope_constraint=args.guided_warp_slope_constraint,
            use_window=not args.guided_warp_no_window,
        ),
        "dgw_sameclass": lambda: dgw_sameclass(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            batch_size=args.guided_warp_batch_size,
            slope_constraint=args.guided_warp_slope_constraint,
            use_window=not args.guided_warp_no_window,
            use_variable_slice=not args.dgw_no_variable_slice,
            min_window_len=args.window_min_len,
        ),
        "timevae_classwise_optional": lambda: timevae_classwise_optional(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            epochs=args.timevae_epochs,
            batch_size=args.timevae_batch_size,
            lr=args.timevae_lr,
            latent_dim=args.timevae_latent_dim,
            hidden_dim=args.timevae_hidden_dim,
            beta=args.timevae_beta,
            min_class_size=args.timevae_min_class_size,
            device=args.device,
        ),
        "raw_smote_flatten_balanced": lambda: raw_smote_flatten_balanced(X_train, y_train, seed=seed),
        "random_cov_state": lambda: random_cov_state(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            gamma=args.pia_gamma,
        ),
        "pca_cov_state": lambda: pca_cov_state(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            gamma=args.pia_gamma,
            k_dir=args.k_dir,
        ),
    }
    if method not in builders:
        raise ValueError(f"No external augmenter registered for method={method}")
    return builders[method]()


def _run_csta_method(dataset: str, seed: int, method: str, args, out_root: Path) -> Dict[str, object]:
    csta_root = out_root / "_csta_runs" / method / dataset / f"s{seed}"
    csta_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_act_pilot.py"),
        "--dataset",
        dataset,
        "--pipeline",
        "act",
        "--algo",
        "zpia_top1_pool",
        "--model",
        args.backbone,
        "--seeds",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--patience",
        str(args.patience),
        "--val-ratio",
        str(args.val_ratio),
        "--k-dir",
        str(args.k_dir),
        "--n-kernels",
        str(args.n_kernels),
        "--pia-gamma",
        str(args.pia_gamma),
        "--eta-safe",
        str(args.eta_safe),
        "--multiplier",
        str(args.multiplier),
        "--device",
        args.device,
        "--out-root",
        str(csta_root),
        "--audit-method-label",
        method,
    ]
    if method == "csta_group_template_top":
        cmd.extend(["--template-selection", "group_top", "--group-size", str(args.group_size)])
    elif method.startswith("csta_topk_"):
        cmd.extend(["--template-selection", method.replace("csta_", "")])

    (csta_root / "command.json").write_text(
        json.dumps({"method": method, "command": cmd}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    log_path = csta_root / "run.log"
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log_f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{method} failed for {dataset} seed {seed}; see {log_path}")

    result_csv = csta_root / f"{dataset}_results.csv"
    if not result_csv.is_file():
        raise FileNotFoundError(f"Missing CSTA result file: {result_csv}")
    df = pd.read_csv(result_csv)
    if df.empty or str(df.iloc[0].get("status", "success")) != "success":
        raise RuntimeError(f"{method} did not produce a success row in {result_csv}")
    row = df.iloc[0].to_dict()
    extra_metrics = _extract_csta_extra_metrics(row, method)
    if (
        float(extra_metrics.get("gamma_requested_mean", 0.0) or 0.0) == 0.0
        and float(row.get("gamma_zero_rate", 0.0) or 0.0) == 0.0
    ):
        # Older zPIA result rows did not always surface the requested gamma
        # from the ACT command even though the safe-step audit shows non-zero
        # displacements.  Preserve the locked performance while carrying the
        # run configuration into the external summary.
        extra_metrics["gamma_requested_mean"] = float(args.pia_gamma)
        if float(extra_metrics.get("safe_clip_rate", 0.0) or 0.0) == 0.0:
            extra_metrics["gamma_used_mean"] = float(args.pia_gamma)
    return {
        "base_f1": float(row.get("base_f1", np.nan)),
        "aug_f1": float(row.get("act_f1", np.nan)),
        "best_val_f1": float(row.get("act_best_val_f1", row.get("act_val_f1", row.get("base_best_val_f1", 0.0)))),
        "stop_epoch": int(row.get("act_stop_epoch", 0)) if not pd.isna(row.get("act_stop_epoch", np.nan)) else 0,
        "aug_count": int(row.get("aug_total_count", 0)) if "aug_total_count" in row else int(args.multiplier),
        "warning_count": 0,
        "extra_metrics": extra_metrics,
    }


def _base_result_row(
    *,
    dataset: str,
    seed: int,
    method: str,
    backbone: str,
    base_f1: float,
    aug_f1: float,
    aug_count: int,
    n_train: int,
    info: MethodInfo,
    best_val_f1: float,
    stop_epoch: int,
    warning_count: int = 0,
    fallback_count: int = 0,
    method_elapsed_sec: float = 0.0,
    status: str = "success",
    fail_reason: str = "",
    extra_metrics: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    actual_aug_ratio = float(aug_count) / max(float(n_train), 1.0)
    row = {
        "dataset": dataset,
        "seed": int(seed),
        "method": method,
        "status": status,
        "fail_reason": fail_reason,
        "base_f1": float(base_f1),
        "aug_f1": float(aug_f1),
        "gain": float(aug_f1) - float(base_f1),
        "aug_count": int(aug_count),
        "aug_ratio": float(actual_aug_ratio),
        "actual_aug_ratio": float(actual_aug_ratio),
        "source_space": info.source_space,
        "label_mode": info.label_mode,
        "backbone": str(backbone),
        "train_split_only": True,
        "uses_external_library": bool(info.uses_external_library),
        "library_name": info.library_name or "none",
        "budget_matched": bool(info.budget_matched),
        "selection_rule": info.selection_rule,
        "best_val_f1": float(best_val_f1),
        "stop_epoch": int(stop_epoch),
        "warning_count": int(warning_count),
        "fallback_count": int(fallback_count),
        "method_elapsed_sec": float(method_elapsed_sec),
    }
    if extra_metrics:
        for key, value in extra_metrics.items():
            if key not in row:
                row[key] = value
    return row


def _write_summaries(rows: List[Dict[str, object]], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    per_seed_path = out_root / "per_seed_external.csv"
    df.to_csv(per_seed_path, index=False)

    ok = df[df["status"] == "success"].copy()
    if ok.empty:
        return

    dataset_summary = (
        ok.groupby(["dataset", "method"], as_index=False)
        .agg(
            aug_f1_mean=("aug_f1", "mean"),
            aug_f1_std=("aug_f1", "std"),
            gain_mean=("gain", "mean"),
            gain_std=("gain", "std"),
            best_val_f1_mean=("best_val_f1", "mean"),
            aug_count_mean=("aug_count", "mean"),
            actual_aug_ratio_mean=("actual_aug_ratio", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["dataset", "aug_f1_mean"], ascending=[True, False])
    )
    dataset_summary.to_csv(out_root / "dataset_summary_external.csv", index=False)

    pivot = dataset_summary.pivot(index="dataset", columns="method", values="aug_f1_mean")
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    overall_rows = []
    for method in sorted(ok["method"].unique()):
        vals = pivot[method].dropna() if method in pivot else pd.Series(dtype=float)
        rank_vals = ranks[method].dropna() if method in ranks else pd.Series(dtype=float)
        overall_rows.append(
            {
                "method": method,
                "n_datasets": int(vals.shape[0]),
                "mean_f1": float(vals.mean()) if not vals.empty else np.nan,
                "mean_rank": float(rank_vals.mean()) if not rank_vals.empty else np.nan,
                "win_count": int((rank_vals == 1.0).sum()) if not rank_vals.empty else 0,
            }
        )
    pd.DataFrame(overall_rows).sort_values(["mean_rank", "mean_f1"], ascending=[True, False]).to_csv(
        out_root / "overall_rank_external.csv",
        index=False,
    )

    raw = ok[ok["method"].isin(RAW_AUG_METHODS)].copy()
    best_rows = []
    if not raw.empty:
        for (dataset, seed), sub in raw.groupby(["dataset", "seed"]):
            sub = sub.sort_values(["best_val_f1", "method"], ascending=[False, True])
            best = sub.iloc[0].to_dict()
            best["selected_method"] = best["method"]
            best["method"] = "best_rawaug"
            best_rows.append(best)
    best_raw_df = pd.DataFrame(best_rows)
    best_raw_df.to_csv(out_root / "best_rawaug_val_selected.csv", index=False)

    combined = ok.copy()
    if not best_raw_df.empty:
        combined = pd.concat([combined, best_raw_df], ignore_index=True)
    combined_summary = combined.groupby(["dataset", "method"], as_index=False).agg(aug_f1_mean=("aug_f1", "mean"))
    comp = combined_summary.pivot(index="dataset", columns="method", values="aug_f1_mean").reset_index()
    ref_rows = []
    for _, row in comp.iterrows():
        out = {"dataset": row["dataset"]}
        csta = row.get("csta_top1_current", np.nan)
        group = row.get("csta_group_template_top", np.nan)
        for method in REF_METHODS:
            out[method] = row.get(method, np.nan)
        for ref in [
            "no_aug",
            "best_rawaug",
            "raw_mixup",
            "dba_sameclass",
            "raw_smote_flatten_balanced",
            "random_cov_state",
            "pca_cov_state",
        ]:
            out[f"csta_top1_minus_{ref}"] = csta - row.get(ref, np.nan)
            out[f"csta_group_minus_{ref}"] = group - row.get(ref, np.nan)
        out["csta_group_minus_csta_top1"] = group - csta
        ref_rows.append(out)
    pd.DataFrame(ref_rows).to_csv(out_root / "csta_vs_external_refs.csv", index=False)


def _write_phase1_phase2_combined_summaries(out_root: Path, locked_phase1_root: Optional[Path]) -> None:
    if locked_phase1_root is None:
        return
    phase1_path = Path(locked_phase1_root) / "per_seed_external.csv"
    phase2_path = Path(out_root) / "per_seed_external.csv"
    if not phase1_path.is_file() or not phase2_path.is_file():
        return
    phase1 = pd.read_csv(phase1_path)
    phase2 = pd.read_csv(phase2_path)
    if phase1.empty or phase2.empty:
        return
    phase1 = phase1.copy()
    phase2 = phase2.copy()
    phase1["phase"] = "phase1_locked"
    phase2["phase"] = "phase2_new"
    combined = pd.concat([phase1, phase2], ignore_index=True, sort=False)
    ok = combined[combined["status"].fillna("success") == "success"].copy()
    if ok.empty:
        return

    summary = (
        ok.groupby(["phase", "dataset", "method"], as_index=False)
        .agg(
            aug_f1_mean=("aug_f1", "mean"),
            aug_f1_std=("aug_f1", "std"),
            gain_mean=("gain", "mean"),
            actual_aug_ratio_mean=("actual_aug_ratio", "mean"),
            method_elapsed_sec_mean=("method_elapsed_sec", "mean"),
            fallback_count_mean=("fallback_count", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["dataset", "aug_f1_mean"], ascending=[True, False])
    )
    summary.to_csv(out_root / "phase1_phase2_external_summary.csv", index=False)

    dataset_summary = ok.groupby(["dataset", "method"], as_index=False).agg(aug_f1_mean=("aug_f1", "mean"))
    pivot = dataset_summary.pivot(index="dataset", columns="method", values="aug_f1_mean").reset_index()
    csta_rows = []
    for _, row in pivot.iterrows():
        out = {"dataset": row["dataset"]}
        for col in pivot.columns:
            if col != "dataset":
                out[col] = row.get(col, np.nan)
        csta = row.get("csta_top1_current", np.nan)
        group = row.get("csta_group_template_top", np.nan)
        for method in sorted(set(ok["method"])):
            if method in {"csta_top1_current", "csta_group_template_top"}:
                continue
            out[f"csta_top1_minus_{method}"] = csta - row.get(method, np.nan)
            out[f"csta_group_minus_{method}"] = group - row.get(method, np.nan)
        out["csta_group_minus_csta_top1"] = group - csta
        csta_rows.append(out)
    pd.DataFrame(csta_rows).to_csv(out_root / "csta_vs_all_external_phase1_phase2.csv", index=False)

    phase2_ok = ok[ok["method"].isin(PHASE2_ARMS)].copy()
    if phase2_ok.empty:
        return
    method_means = phase2_ok.groupby("method", as_index=False).agg(mean_f1=("aug_f1", "mean"))
    method_means = method_means.sort_values(["mean_f1", "method"], ascending=[False, True])
    best_method = str(method_means.iloc[0]["method"])
    best_mean = float(method_means.iloc[0]["mean_f1"])
    best_rows = []
    for dataset, sub in dataset_summary.groupby("dataset"):
        vals = sub.set_index("method")["aug_f1_mean"]
        best_rows.append(
            {
                "dataset": dataset,
                "best_phase2_baseline": best_method,
                "best_phase2_overall_mean_f1": best_mean,
                "best_phase2_f1": float(vals.get(best_method, np.nan)),
                "csta_top1_current": float(vals.get("csta_top1_current", np.nan)),
                "csta_group_template_top": float(vals.get("csta_group_template_top", np.nan)),
                "best_phase2_minus_csta_top1": float(vals.get(best_method, np.nan) - vals.get("csta_top1_current", np.nan)),
                "best_phase2_minus_csta_group": float(vals.get(best_method, np.nan) - vals.get("csta_group_template_top", np.nan)),
            }
        )
    pd.DataFrame(best_rows).to_csv(out_root / "best_phase2_vs_csta.csv", index=False)


def run(args) -> List[Dict[str, object]]:
    datasets = _parse_csv(args.datasets)
    methods = _parse_csv(args.arms)
    unknown = sorted(set(methods) - set(METHOD_INFO))
    if unknown:
        raise ValueError(f"Unknown external baseline arms: {unknown}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    if args.dry_run:
        for dataset in datasets:
            for seed in _parse_csv(args.seeds):
                for method in methods:
                    print(f"DRY-RUN dataset={dataset} seed={seed} method={method}")
        return []

    rows: List[Dict[str, object]] = []
    seeds = [int(s) for s in _parse_csv(args.seeds)]
    for dataset in datasets:
        trials = load_trials_for_dataset(dataset)
        for seed in seeds:
            train_trials, test_trials, val_trials = make_trial_split(trials, seed=seed, val_ratio=args.val_ratio)
            X_train, y_train = _stack_trials(train_trials)
            X_test, y_test = _stack_trials(test_trials)
            X_val, y_val = _stack_trials(val_trials) if val_trials else (None, None)
            n_train = int(len(X_train))
            n_classes = int(max(y_train.max(), y_test.max(), y_val.max() if y_val is not None else 0) + 1)

            print(f"[{dataset} s{seed}] fitting no_aug baseline")
            base_res = _fit_hard(X_train, y_train, X_val, y_val, X_test, y_test, args, seed)
            base_f1 = float(base_res["macro_f1"])
            base_val = float(base_res.get("best_val_f1", np.nan))
            base_stop = int(base_res.get("stop_epoch", 0))

            for method in methods:
                info = METHOD_INFO[method]
                try:
                    print(f"[{dataset} s{seed}] method={method}")
                    t0 = time.perf_counter()
                    if method == "no_aug":
                        row = _base_result_row(
                            dataset=dataset,
                            seed=seed,
                            method=method,
                            backbone=args.backbone,
                            base_f1=base_f1,
                            aug_f1=base_f1,
                            aug_count=0,
                            n_train=n_train,
                            info=info,
                            best_val_f1=base_val,
                            stop_epoch=base_stop,
                            method_elapsed_sec=0.0,
                        )
                    elif method.startswith("csta_"):
                        csta_res = _run_csta_method(dataset, seed, method, args, out_root)
                        elapsed = time.perf_counter() - t0
                        row = _base_result_row(
                            dataset=dataset,
                            seed=seed,
                            method=method,
                            backbone=args.backbone,
                            base_f1=float(csta_res["base_f1"]),
                            aug_f1=float(csta_res["aug_f1"]),
                            aug_count=int(csta_res["aug_count"]),
                            n_train=n_train,
                            info=info,
                            best_val_f1=float(csta_res.get("best_val_f1", np.nan)),
                            stop_epoch=int(csta_res.get("stop_epoch", 0)),
                            warning_count=int(csta_res.get("warning_count", 0)),
                            fallback_count=int(csta_res.get("warning_count", 0)),
                            method_elapsed_sec=elapsed,
                            extra_metrics=dict(csta_res.get("extra_metrics", {})),
                        )
                    elif method == "jobda_cleanroom":
                        aug = _build_external_aug(method, X_train, y_train, args, seed, n_classes)
                        res = fit_jobda_backbone(
                            args.backbone,
                            aug.X_aug,
                            aug.y_aug,
                            X_val,
                            y_val,
                            X_test,
                            y_test,
                            num_classes=int(n_classes),
                            num_transforms=int(aug.meta.get("jobda_num_transforms", 4)),
                            epochs=args.epochs,
                            lr=args.lr,
                            batch_size=args.batch_size,
                            patience=args.patience,
                            device=args.device,
                            seed=int(seed),
                        )
                        elapsed = time.perf_counter() - t0
                        row = _base_result_row(
                            dataset=dataset,
                            seed=seed,
                            method=method,
                            backbone=args.backbone,
                            base_f1=base_f1,
                            aug_f1=float(res["macro_f1"]),
                            aug_count=int(max(0, aug.X_aug.shape[0] - n_train)),
                            n_train=n_train,
                            info=info,
                            best_val_f1=float(res.get("best_val_f1", np.nan)),
                            stop_epoch=int(res.get("stop_epoch", 0)),
                            warning_count=int(aug.warning_count),
                            fallback_count=int(aug.fallback_count),
                            method_elapsed_sec=elapsed,
                        )
                        for key, value in aug.meta.items():
                            row[key] = value
                    elif method == "manifold_mixup":
                        res = fit_manifold_mixup_backbone(
                            args.backbone,
                            X_train,
                            y_train,
                            X_val,
                            y_val,
                            X_test,
                            y_test,
                            epochs=args.epochs,
                            lr=args.lr,
                            batch_size=args.batch_size,
                            patience=args.patience,
                            device=args.device,
                            mixup_alpha=args.mixup_alpha,
                            seed=int(seed),
                        )
                        elapsed = time.perf_counter() - t0
                        row = _base_result_row(
                            dataset=dataset,
                            seed=seed,
                            method=method,
                            backbone=args.backbone,
                            base_f1=base_f1,
                            aug_f1=float(res["macro_f1"]),
                            aug_count=0,
                            n_train=n_train,
                            info=info,
                            best_val_f1=float(res.get("best_val_f1", np.nan)),
                            stop_epoch=int(res.get("stop_epoch", 0)),
                            method_elapsed_sec=elapsed,
                        )
                        row["manifold_mixup_alpha"] = float(args.mixup_alpha)
                    else:
                        aug = _build_external_aug(method, X_train, y_train, args, seed, n_classes)
                        aug_count = int(aug.X_aug.shape[0])
                        if aug.label_mode == "soft":
                            y_orig_soft = _one_hot(y_train, n_classes)
                            X_mix = np.concatenate([X_train, aug.X_aug], axis=0).astype(np.float32)
                            y_mix_soft = np.concatenate([y_orig_soft, aug.y_aug_soft], axis=0).astype(np.float32)
                            res = _fit_soft(X_mix, y_mix_soft, X_val, y_val, X_test, y_test, args, seed)
                        else:
                            if aug_count > 0:
                                X_mix = np.concatenate([X_train, aug.X_aug], axis=0).astype(np.float32)
                                y_mix = np.concatenate([y_train, aug.y_aug], axis=0).astype(np.int64)
                            else:
                                X_mix = X_train
                                y_mix = y_train
                            res = _fit_hard(X_mix, y_mix, X_val, y_val, X_test, y_test, args, seed)
                        elapsed = time.perf_counter() - t0
                        row = _base_result_row(
                            dataset=dataset,
                            seed=seed,
                            method=method,
                            backbone=args.backbone,
                            base_f1=base_f1,
                            aug_f1=float(res["macro_f1"]),
                            aug_count=aug_count,
                            n_train=n_train,
                            info=info,
                            best_val_f1=float(res.get("best_val_f1", np.nan)),
                            stop_epoch=int(res.get("stop_epoch", 0)),
                            warning_count=int(aug.warning_count),
                            fallback_count=int(getattr(aug, "fallback_count", 0) or aug.warning_count),
                            method_elapsed_sec=elapsed,
                        )
                        for key, value in aug.meta.items():
                            row[key] = value
                    rows.append(row)
                    _write_summaries(rows, out_root)
                    _write_phase1_phase2_combined_summaries(out_root, Path(args.locked_phase1_root) if args.locked_phase1_root else None)
                except Exception as exc:
                    fail = _base_result_row(
                        dataset=dataset,
                        seed=seed,
                        method=method,
                        backbone=args.backbone,
                        base_f1=base_f1,
                        aug_f1=np.nan,
                        aug_count=0,
                        n_train=n_train,
                        info=info,
                        best_val_f1=np.nan,
                        stop_epoch=0,
                        method_elapsed_sec=time.perf_counter() - t0 if "t0" in locals() else 0.0,
                        status="failed",
                        fail_reason=str(exc),
                    )
                    rows.append(fail)
                    _write_summaries(rows, out_root)
                    _write_phase1_phase2_combined_summaries(out_root, Path(args.locked_phase1_root) if args.locked_phase1_root else None)
                    if args.fail_fast:
                        raise
    _write_summaries(rows, out_root)
    _write_phase1_phase2_combined_summaries(out_root, Path(args.locked_phase1_root) if args.locked_phase1_root else None)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="CSTA external baseline runner for Phase 1/2/3 arms")
    parser.add_argument("--out-root", type=str, default=str(PROJECT_ROOT / "results" / "csta_external_baselines_phase1" / "resnet1d_s123"))
    parser.add_argument("--backbone", type=str, choices=list(SUPPORTED_BACKBONES), default="resnet1d")
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--arms", type=str, default=",".join(DEFAULT_ARMS))
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--multiplier", type=int, default=10)
    parser.add_argument("--k-dir", type=int, default=10)
    parser.add_argument("--n-kernels", type=int, default=10000)
    parser.add_argument("--pia-gamma", type=float, default=0.1)
    parser.add_argument("--eta-safe", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--dba-k", type=int, default=5)
    parser.add_argument("--dba-max-iter", type=int, default=5)
    parser.add_argument("--magnitude-warp-sigma", type=float, default=0.2)
    parser.add_argument("--magnitude-warp-knots", type=int, default=4)
    parser.add_argument("--magnitude-warp-shared-curve", action="store_true")
    parser.add_argument("--window-warp-ratio", type=float, default=0.10)
    parser.add_argument("--window-warp-speeds", type=str, default="0.5,2.0")
    parser.add_argument("--window-slice-ratio", type=float, default=0.90)
    parser.add_argument("--window-min-len", type=int, default=4)
    parser.add_argument("--wdba-k", type=int, default=5)
    parser.add_argument("--wdba-max-iter", type=int, default=5)
    parser.add_argument("--spawner-noise-scale", type=float, default=0.05)
    parser.add_argument("--jobda-transform-subseqs", type=str, default="0,2,4,8")
    parser.add_argument("--guided-warp-batch-size", type=int, default=6)
    parser.add_argument("--guided-warp-slope-constraint", type=str, choices=["symmetric", "asymmetric"], default="symmetric")
    parser.add_argument("--guided-warp-no-window", action="store_true")
    parser.add_argument("--dgw-no-variable-slice", action="store_true")
    parser.add_argument("--timevae-epochs", type=int, default=30)
    parser.add_argument("--timevae-batch-size", type=int, default=32)
    parser.add_argument("--timevae-lr", type=float, default=1e-3)
    parser.add_argument("--timevae-latent-dim", type=int, default=8)
    parser.add_argument("--timevae-hidden-dim", type=int, default=128)
    parser.add_argument("--timevae-beta", type=float, default=1.0)
    parser.add_argument("--timevae-min-class-size", type=int, default=4)
    parser.add_argument(
        "--locked-phase1-root",
        type=str,
        default=str(PROJECT_ROOT / "results" / "csta_external_baselines_phase1" / "resnet1d_s123"),
        help="Optional locked Phase 1 root to merge with new Phase 2 rows.",
    )
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
