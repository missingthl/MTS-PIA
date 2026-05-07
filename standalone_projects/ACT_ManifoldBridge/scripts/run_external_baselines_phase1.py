#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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
from utils.external_aug_dispatch import build_external_aug
from utils.external_runner_registry import (
    DEFAULT_ARMS,
    DEFAULT_DATASETS,
    MethodInfo,
    METHOD_INFO,
    PHASE2_ARMS,
    PHASE3_ARMS,
    RAW_AUG_METHODS,
    REF_METHODS,
    csta_policy_for_method as _csta_policy_for_method,
    extract_csta_extra_metrics as _extract_csta_extra_metrics,
    guard_locked_out_root as _guard_locked_out_root,
    parse_csv as _parse_csv,
)


def _stack_trials(trials) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([t.x for t in trials], axis=0).astype(np.float32)
    y = np.asarray([t.y for t in trials], dtype=np.int64)
    return X, y


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), int(n_classes)), dtype=np.float32)
    out[np.arange(len(y)), y.astype(np.int64)] = 1.0
    return out


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


def _run_csta_method(dataset: str, seed: int, method: str, args, out_root: Path) -> Dict[str, object]:
    out_root = out_root.resolve()
    csta_root = out_root / "_csta_runs" / method / dataset / f"s{seed}"
    csta_root.mkdir(parents=True, exist_ok=True)

    # Determine algo name from method
    if "ao_fisher" in method:
        algo_name = "ao_fisher"
        selection = "topk_uniform_top5"
    elif "ao_contrastive" in method:
        algo_name = "ao_contrastive"
        selection = "topk_uniform_top5"
    else:
        algo_name = "zpia_top1_pool"
        selection = None

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_act_pilot.py"),
        "--dataset",
        dataset,
        "--pipeline",
        "act",
        "--algo",
        algo_name,
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
    if selection is not None:
        cmd.extend(["--template-selection", selection])
    elif method == "csta_group_template_top":
        cmd.extend(["--template-selection", "group_top", "--group-size", str(args.group_size)])
    elif method.startswith("csta_topk_"):
        cmd.extend(["--template-selection", method.replace("csta_", "")])
    elif method in {
        "csta_template_random_within_bank",
        "csta_fv_filter_top5",
        "csta_fv_score_top5",
        "csta_random_feasible_selector",
    }:
        cmd.extend(["--template-selection", _csta_policy_for_method(method)])

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

    _guard_locked_out_root(args)
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
                        aug = build_external_aug(method, X_train, y_train, args, seed, n_classes)
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
                        aug = build_external_aug(method, X_train, y_train, args, seed, n_classes)
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
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(PROJECT_ROOT / "results" / "csta_external_baselines_local" / "resnet1d_s123"),
        help="Output root. Defaults to a local non-locked root; pass an explicit root for formal matrices.",
    )
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
    parser.add_argument("--eta-safe", type=float, default=0.75)
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
    parser.add_argument("--timevae-min-class-size", type=int, default=4)
    parser.add_argument("--diffusionts-epochs", type=int, default=500)
    parser.add_argument("--diffusionts-batch-size", type=int, default=128)
    parser.add_argument("--timevqvae-vqvae-epochs", type=int, default=100)
    parser.add_argument("--timevqvae-maskgit-epochs", type=int, default=100)
    parser.add_argument("--timevqvae-batch-size", type=int, default=64)
    parser.add_argument(
        "--locked-phase1-root",
        type=str,
        default=str(PROJECT_ROOT / "results" / "csta_external_baselines_phase1" / "resnet1d_s123"),
        help="Optional locked Phase 1 root to merge with new Phase 2 rows.",
    )
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--allow-locked-root-overwrite",
        action="store_true",
        help="Allow writing to locked Phase 1/2 reference roots. Use only for intentional reference regeneration.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
