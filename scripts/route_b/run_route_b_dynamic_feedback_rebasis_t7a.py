#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from route_b_unified.trajectory_classifier import TrajectoryModelConfig  # noqa: E402
from route_b_unified.trajectory_feedback_pool import (  # noqa: E402
    TrajectoryFeedbackPoolConfig,
    build_trajectory_feedback_pool,
)
from route_b_unified.trajectory_feedback_pool_windows import (  # noqa: E402
    TrajectoryWindowFeedbackPoolConfig,
    build_window_feedback_pool,
    build_window_feedback_reference_stats,
)
from route_b_unified.trajectory_feedback_rebasis import fit_trajectory_feedback_rebasis  # noqa: E402
from route_b_unified.trajectory_feedback_rebasis_t7 import (  # noqa: E402
    TrajectoryClassConditionedOVRRebasisResult,
    fit_trajectory_class_conditioned_rebasis_ovr,
)
from route_b_unified.trajectory_pia_evaluator import (  # noqa: E402
    TrajectoryPIAEvalConfig,
    TrajectoryPIAEvalResult,
    compute_trajectory_diagnostics,
    evaluate_trajectory_pia_t2a,
    evaluate_trajectory_train_final,
)
from route_b_unified.trajectory_pia_operator import TrajectoryPIAOperator, TrajectoryPIAOperatorConfig  # noqa: E402
from route_b_unified.trajectory_representation import (  # noqa: E402
    TrajectoryRepresentationConfig,
    build_trajectory_representation,
)
from route_b_unified.trajectory_unified_window_policy import (  # noqa: E402
    TrajectoryUnifiedWindowPolicyResult,
    build_unified_window_augmented_trajectories,
)


def _parse_csv_list(text: str) -> List[str]:
    out = [tok.strip().lower() for tok in str(text).split(",") if tok.strip()]
    if not out:
        raise ValueError("csv list cannot be empty")
    return out


def _parse_seed_list(text: str) -> List[int]:
    out = sorted(set(int(tok.strip()) for tok in str(text).split(",") if tok.strip()))
    if not out:
        raise ValueError("seed list cannot be empty")
    return out


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def _format_mean_std(values: Sequence[float]) -> str:
    mean, std = _mean_std(values)
    return f"{mean:.4f} +/- {std:.4f}"


def _save_result_json(path: str, result: TrajectoryPIAEvalResult) -> None:
    _write_json(
        path,
        {
            "dataset": result.dataset,
            "seed": result.seed,
            "operator_mode": result.operator_mode,
            "train_metrics": result.train_metrics,
            "val_metrics": result.val_metrics,
            "test_metrics": result.test_metrics,
            "best_epoch": result.best_epoch,
            "diagnostics": result.diagnostics,
            "operator_meta": result.operator_meta,
            "meta": result.meta,
            "history_rows": result.history_rows,
        },
    )


def _load_reference_t6b_map(path: str) -> Dict[tuple[str, int], float]:
    csv_path = str(path).strip()
    if not csv_path or not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    if "dataset" not in df.columns or "seed" not in df.columns or "test_macro_f1" not in df.columns:
        return {}
    if "mode" in df.columns:
        df = df[df["mode"].astype(str).str.contains("t6b", case=False, na=False)].copy()
    out: Dict[tuple[str, int], float] = {}
    for _, row in df.iterrows():
        out[(str(row["dataset"]).strip().lower(), int(row["seed"]))] = float(row["test_macro_f1"])
    return out


def _baseline_train_bundle(state) -> tuple[List[str], List[int], List[np.ndarray], Dict[str, object], Dict[str, float]]:
    train_tids = [str(v) for v in state.train.tids.tolist()]
    train_labels = [int(v) for v in state.train.y.tolist()]
    train_seqs = [np.asarray(v, dtype=np.float32) for v in state.train.z_seq_list]
    diagnostics = compute_trajectory_diagnostics(train_seqs, train_labels, continuity_ratio=1.0)
    return train_tids, train_labels, train_seqs, {"mode": "baseline"}, diagnostics


def _augment_with_shared_operator(
    state,
    *,
    operator: TrajectoryPIAOperator,
    gamma_main: float,
    smooth_lambda: float,
    mode: str,
    extra_meta: Dict[str, object] | None = None,
) -> tuple[List[str], List[int], List[np.ndarray], Dict[str, object], Dict[str, float]]:
    train_tids = [str(v) for v in state.train.tids.tolist()]
    train_labels = [int(v) for v in state.train.y.tolist()]
    train_seqs = [np.asarray(v, dtype=np.float32) for v in state.train.z_seq_list]
    aug_seqs, _delta_list, op_meta = operator.transform_many(
        train_seqs,
        gamma_main=float(gamma_main),
        smooth_lambda=float(smooth_lambda),
    )
    aug_tids = [f"{tid}__{mode}_aug" for tid in train_tids]
    final_tids = list(train_tids) + list(aug_tids)
    final_labels = list(train_labels) + list(train_labels)
    final_seqs = list(train_seqs) + list(aug_seqs)
    diagnostics = compute_trajectory_diagnostics(
        final_seqs,
        final_labels,
        continuity_ratio=float(op_meta["mean_continuity_distortion_ratio"]),
    )
    meta = dict(op_meta)
    if extra_meta:
        meta.update(dict(extra_meta))
    return final_tids, final_labels, final_seqs, meta, diagnostics


def _rebuild_selected_rows_with_rebased_windows(
    *,
    train_tids: Sequence[str],
    train_z_seq_list: Sequence[np.ndarray],
    operator_new: TrajectoryPIAOperator,
    selected_rows: Sequence[Dict[str, object]],
    gamma_main: float,
    smooth_lambda: float,
) -> List[Dict[str, object]]:
    seqs = [np.asarray(v, dtype=np.float32) for v in train_z_seq_list]
    rebased_aug_seqs, _delta_list, _op_meta = operator_new.transform_many(
        seqs,
        gamma_main=float(gamma_main),
        smooth_lambda=float(smooth_lambda),
    )
    tid_to_index = {str(tid): idx for idx, tid in enumerate(train_tids)}
    out: List[Dict[str, object]] = []
    for row in selected_rows:
        tid = str(row["trial_id"])
        trial_index = int(tid_to_index[tid])
        win = int(row["window_index"])
        rebased_window = np.asarray(rebased_aug_seqs[trial_index][win], dtype=np.float32)
        row_new = dict(row)
        row_new["z_window_aug"] = rebased_window
        out.append(row_new)
    return out


def _augment_with_unified_policy(
    state,
    *,
    unified_result: TrajectoryUnifiedWindowPolicyResult,
    pool_mode: str,
    rebasis_summary: Dict[str, object],
) -> tuple[List[str], List[int], List[np.ndarray], Dict[str, object], Dict[str, float]]:
    train_tids = [str(v) for v in state.train.tids.tolist()]
    train_labels = [int(v) for v in state.train.y.tolist()]
    train_seqs = [np.asarray(v, dtype=np.float32) for v in state.train.z_seq_list]
    final_tids = list(train_tids) + list(unified_result.aug_tids)
    final_labels = list(train_labels) + list(unified_result.aug_labels)
    final_seqs = list(train_seqs) + list(unified_result.aug_z_seq_list)
    diagnostics = compute_trajectory_diagnostics(
        final_seqs,
        final_labels,
        continuity_ratio=float(unified_result.stitching_summary["stitched_continuity_distortion_ratio"]),
    )
    meta = {
        "generator_mode": "t2a_default_frozen",
        "window_policy_mode": str(pool_mode),
        "pool_summary": dict(unified_result.pool_summary),
        "class_coverage_rows": [dict(v) for v in unified_result.class_coverage_rows],
        "stitching_summary": dict(unified_result.stitching_summary),
        "rebasis_summary": dict(rebasis_summary),
        "routing_scope": "train_augmentation_only",
        "test_time_routing": False,
    }
    return final_tids, final_labels, final_seqs, meta, diagnostics


def _augment_with_family(
    state,
    *,
    rebasis_result: TrajectoryClassConditionedOVRRebasisResult,
    gamma_main: float,
    smooth_lambda: float,
) -> tuple[List[str], List[int], List[np.ndarray], Dict[str, object], Dict[str, float]]:
    train_tids = [str(v) for v in state.train.tids.tolist()]
    train_labels = [int(v) for v in state.train.y.tolist()]
    train_seqs = [np.asarray(v, dtype=np.float32) for v in state.train.z_seq_list]
    aug_seqs, _delta_list, op_meta = rebasis_result.family.transform_many(
        train_seqs,
        labels=train_labels,
        gamma_main=float(gamma_main),
        smooth_lambda=float(smooth_lambda),
    )
    aug_tids = [f"{tid}__t7a_class_conditioned_ovr_aug" for tid in train_tids]
    final_tids = list(train_tids) + list(aug_tids)
    final_labels = list(train_labels) + list(train_labels)
    final_seqs = list(train_seqs) + list(aug_seqs)
    diagnostics = compute_trajectory_diagnostics(
        final_seqs,
        final_labels,
        continuity_ratio=float(op_meta["mean_continuity_distortion_ratio"]),
    )
    meta = dict(op_meta)
    meta["rebasis_summary"] = dict(rebasis_result.summary)
    meta["class_conditioned_basis_family"] = [
        {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in row.items()}
        for row in rebasis_result.class_rows
    ]
    meta["routing_note"] = (
        "True-label container selection is used only for training-time augmentation generation; "
        "val/test always consume original trajectories with no class-conditioned test-time routing."
    )
    return final_tids, final_labels, final_seqs, meta, diagnostics


def main() -> None:
    p = argparse.ArgumentParser(description="T7a class-conditioned OvR rebasis probe with frozen window-level constructive pool.")
    p.add_argument("--main-datasets", type=str, default="selfregulationscp1")
    p.add_argument("--anchor-datasets", type=str, default="natops")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/route_b_dynamic_feedback_rebasis_t7a_20260330_formal")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--prop-win-ratio", type=float, default=0.20)
    p.add_argument("--prop-hop-ratio", type=float, default=0.10)
    p.add_argument("--min-window-extra-channels", type=int, default=4)
    p.add_argument("--min-hop-len", type=int, default=4)
    p.add_argument("--gru-hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--gamma-main", type=float, default=0.05)
    p.add_argument("--smooth-lambda", type=float, default=0.50)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--max-purity-drop", type=float, default=0.10)
    p.add_argument("--continuity-quantile", type=float, default=75.0)
    p.add_argument("--t6b-reference-per-seed-csv", type=str, default="")
    args = p.parse_args()

    main_datasets = _parse_csv_list(args.main_datasets)
    anchor_datasets = _parse_csv_list(args.anchor_datasets)
    datasets = []
    for name in list(main_datasets) + list(anchor_datasets):
        if name not in datasets:
            datasets.append(name)
    seeds = _parse_seed_list(args.seeds)
    out_root = str(args.out_root)
    _ensure_dir(out_root)
    t6b_reference = _load_reference_t6b_map(str(args.t6b_reference_per_seed_csv))

    config_rows: List[Dict[str, object]] = []
    per_seed_rows: List[Dict[str, object]] = []
    pool_rows: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []
    basis_family_rows: List[Dict[str, object]] = []
    diagnostics_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        for seed in seeds:
            state = build_trajectory_representation(
                TrajectoryRepresentationConfig(
                    dataset=str(dataset),
                    seed=int(seed),
                    val_fraction=float(args.val_fraction),
                    spd_eps=float(args.spd_eps),
                    prop_win_ratio=float(args.prop_win_ratio),
                    prop_hop_ratio=float(args.prop_hop_ratio),
                    min_window_extra_channels=int(args.min_window_extra_channels),
                    min_hop_len=int(args.min_hop_len),
                )
            )
            seed_dir = os.path.join(out_root, f"{dataset}_seed{seed}")
            _ensure_dir(seed_dir)

            model_cfg = TrajectoryModelConfig(
                z_dim=int(state.z_dim),
                num_classes=int(state.num_classes),
                gru_hidden_dim=int(args.gru_hidden_dim),
                dropout=float(args.dropout),
            )
            eval_common = {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "patience": int(args.patience),
                "device": str(args.device),
            }
            operator_cfg = TrajectoryPIAOperatorConfig(seed=int(seed))
            frozen_operator = TrajectoryPIAOperator(operator_cfg).fit(state.train.z_seq_list)

            _write_json(
                os.path.join(seed_dir, "dynamic_feedback_rebasis_t7a_split_meta.json"),
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "split_meta": dict(state.split_meta),
                    "meta": dict(state.meta),
                    "window_len": int(state.window_len),
                    "hop_len": int(state.hop_len),
                    "frozen_generator": {
                        "operator": "t2a_default",
                        "gamma_main": float(args.gamma_main),
                        "smooth_lambda": float(args.smooth_lambda),
                    },
                    "frozen_container_scope": "shared_global_center + class_conditioned_single_axis_ovr_family",
                },
            )

            baseline_tids, baseline_labels, baseline_seqs, baseline_meta, baseline_diag = _baseline_train_bundle(state)
            baseline_result = evaluate_trajectory_train_final(
                state,
                seed=int(seed),
                model_cfg=model_cfg,
                eval_cfg=TrajectoryPIAEvalConfig(
                    operator_mode="baseline",
                    gamma_main=float(args.gamma_main),
                    smooth_lambda=float(args.smooth_lambda),
                    **eval_common,
                ),
                train_tids=baseline_tids,
                train_labels=baseline_labels,
                train_z_seq_list=baseline_seqs,
                diagnostics=baseline_diag,
                operator_meta=baseline_meta,
            )
            _save_result_json(os.path.join(seed_dir, "baseline_result.json"), baseline_result)

            t2a_result = evaluate_trajectory_pia_t2a(
                state,
                seed=int(seed),
                model_cfg=model_cfg,
                eval_cfg=TrajectoryPIAEvalConfig(
                    operator_mode="t2a_default",
                    gamma_main=float(args.gamma_main),
                    smooth_lambda=float(args.smooth_lambda),
                    **eval_common,
                ),
                operator_cfg=operator_cfg,
                prefit_operator=frozen_operator,
            )
            _save_result_json(os.path.join(seed_dir, "t2a_default_result.json"), t2a_result)

            t3_pool = build_trajectory_feedback_pool(
                train_tids=[str(v) for v in state.train.tids.tolist()],
                train_labels=[int(v) for v in state.train.y.tolist()],
                train_z_seq_list=state.train.z_seq_list,
                operator=frozen_operator,
                cfg=TrajectoryFeedbackPoolConfig(
                    gamma_main=float(args.gamma_main),
                    smooth_lambda=float(args.smooth_lambda),
                    knn_k=int(args.knn_k),
                    max_purity_drop=float(args.max_purity_drop),
                    continuity_quantile=float(args.continuity_quantile),
                ),
            )
            shared_rebasis = fit_trajectory_feedback_rebasis(
                orig_train_z_seq_list=state.train.z_seq_list,
                feedback_z_seq_list=t3_pool.accepted_z_seq_list,
                old_operator=frozen_operator,
                operator_cfg=operator_cfg,
            )
            t3_tids, t3_labels, t3_seqs, t3_meta, t3_diag = _augment_with_shared_operator(
                state,
                operator=shared_rebasis.operator_new,
                gamma_main=float(args.gamma_main),
                smooth_lambda=float(args.smooth_lambda),
                mode="t3_rebasis",
                extra_meta={
                    "rebasis_summary": dict(shared_rebasis.summary),
                    "feedback_summary": dict(t3_pool.summary),
                },
            )
            t3_result = evaluate_trajectory_train_final(
                state,
                seed=int(seed),
                model_cfg=model_cfg,
                eval_cfg=TrajectoryPIAEvalConfig(
                    operator_mode="t3_shared_rebasis",
                    gamma_main=float(args.gamma_main),
                    smooth_lambda=float(args.smooth_lambda),
                    **eval_common,
                ),
                train_tids=t3_tids,
                train_labels=t3_labels,
                train_z_seq_list=t3_seqs,
                diagnostics=t3_diag,
                operator_meta=t3_meta,
            )
            _save_result_json(os.path.join(seed_dir, "t3_shared_rebasis_result.json"), t3_result)

            window_ref = build_window_feedback_reference_stats(
                train_labels=[int(v) for v in state.train.y.tolist()],
                train_z_seq_list=state.train.z_seq_list,
            )

            radial_pool = build_window_feedback_pool(
                train_tids=[str(v) for v in state.train.tids.tolist()],
                train_labels=[int(v) for v in state.train.y.tolist()],
                train_z_seq_list=state.train.z_seq_list,
                operator=frozen_operator,
                reference_stats=window_ref,
                cfg=TrajectoryWindowFeedbackPoolConfig(
                    gamma_main=float(args.gamma_main),
                    smooth_lambda=float(args.smooth_lambda),
                    knn_k=int(args.knn_k),
                    max_purity_drop=float(args.max_purity_drop),
                    continuity_quantile=float(args.continuity_quantile),
                    informative_gate="radial",
                ),
            )
            radial_rebasis = fit_trajectory_feedback_rebasis(
                orig_train_z_seq_list=state.train.z_seq_list,
                feedback_z_seq_list=radial_pool.accepted_window_seq_list,
                old_operator=frozen_operator,
                operator_cfg=operator_cfg,
            )
            radial_rows_rebased = _rebuild_selected_rows_with_rebased_windows(
                train_tids=[str(v) for v in state.train.tids.tolist()],
                train_z_seq_list=state.train.z_seq_list,
                operator_new=radial_rebasis.operator_new,
                selected_rows=radial_pool.accepted_window_rows,
                gamma_main=float(args.gamma_main),
                smooth_lambda=float(args.smooth_lambda),
            )
            radial_unified = build_unified_window_augmented_trajectories(
                train_tids=[str(v) for v in state.train.tids.tolist()],
                train_labels=[int(v) for v in state.train.y.tolist()],
                train_z_seq_list=state.train.z_seq_list,
                selected_rows=radial_rows_rebased,
                pool_summary=radial_pool.summary,
                class_coverage_rows=radial_pool.class_coverage_rows,
            )
            t4b_tids, t4b_labels, t4b_seqs, t4b_meta, t4b_diag = _augment_with_unified_policy(
                state,
                unified_result=radial_unified,
                pool_mode="t4b_window_radial_gate",
                rebasis_summary=dict(radial_rebasis.summary),
            )
            t4b_result = evaluate_trajectory_train_final(
                state,
                seed=int(seed),
                model_cfg=model_cfg,
                eval_cfg=TrajectoryPIAEvalConfig(
                    operator_mode="t4b_window_radial_gate",
                    gamma_main=float(args.gamma_main),
                    smooth_lambda=float(args.smooth_lambda),
                    **eval_common,
                ),
                train_tids=t4b_tids,
                train_labels=t4b_labels,
                train_z_seq_list=t4b_seqs,
                diagnostics=t4b_diag,
                operator_meta=t4b_meta,
            )
            _save_result_json(os.path.join(seed_dir, "t4b_window_radial_gate_result.json"), t4b_result)

            constructive_pool = build_window_feedback_pool(
                train_tids=[str(v) for v in state.train.tids.tolist()],
                train_labels=[int(v) for v in state.train.y.tolist()],
                train_z_seq_list=state.train.z_seq_list,
                operator=frozen_operator,
                reference_stats=window_ref,
                cfg=TrajectoryWindowFeedbackPoolConfig(
                    gamma_main=float(args.gamma_main),
                    smooth_lambda=float(args.smooth_lambda),
                    knn_k=int(args.knn_k),
                    max_purity_drop=float(args.max_purity_drop),
                    continuity_quantile=float(args.continuity_quantile),
                    informative_gate="safety_only",
                ),
            )
            t7a_rebasis = fit_trajectory_class_conditioned_rebasis_ovr(
                orig_train_labels=[int(v) for v in state.train.y.tolist()],
                orig_train_z_seq_list=state.train.z_seq_list,
                feedback_labels=constructive_pool.accepted_labels,
                feedback_z_seq_list=constructive_pool.accepted_window_seq_list,
                old_operator=frozen_operator,
                operator_cfg=operator_cfg,
            )
            _write_json(
                os.path.join(seed_dir, "dynamic_feedback_rebasis_t7a_basis_family.json"),
                {
                    "summary": dict(t7a_rebasis.summary),
                    "class_rows": [
                        {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in row.items()}
                        for row in t7a_rebasis.class_rows
                    ],
                },
            )
            t7a_tids, t7a_labels, t7a_seqs, t7a_meta, t7a_diag = _augment_with_family(
                state,
                rebasis_result=t7a_rebasis,
                gamma_main=float(args.gamma_main),
                smooth_lambda=float(args.smooth_lambda),
            )
            t7a_meta["feedback_summary"] = dict(constructive_pool.summary)
            t7a_meta["class_coverage_rows"] = [dict(v) for v in constructive_pool.class_coverage_rows]
            t7a_result = evaluate_trajectory_train_final(
                state,
                seed=int(seed),
                model_cfg=model_cfg,
                eval_cfg=TrajectoryPIAEvalConfig(
                    operator_mode="t7a_class_conditioned_rebasis",
                    gamma_main=float(args.gamma_main),
                    smooth_lambda=float(args.smooth_lambda),
                    **eval_common,
                ),
                train_tids=t7a_tids,
                train_labels=t7a_labels,
                train_z_seq_list=t7a_seqs,
                diagnostics=t7a_diag,
                operator_meta=t7a_meta,
            )
            _save_result_json(os.path.join(seed_dir, "t7a_class_conditioned_rebasis_result.json"), t7a_result)

            mode_rows = [
                ("baseline", baseline_result),
                ("t2a_default", t2a_result),
                ("t3_shared_rebasis", t3_result),
                ("t4b_window_radial_gate", t4b_result),
                ("t7a_class_conditioned_rebasis", t7a_result),
            ]
            for mode, result in mode_rows:
                config_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "mode": str(mode),
                        "gamma_main": float(args.gamma_main),
                        "smooth_lambda": float(args.smooth_lambda),
                        "axis_count": 1,
                        "window_len": int(state.window_len),
                        "hop_len": int(state.hop_len),
                        "epochs": int(args.epochs),
                        "batch_size": int(args.batch_size),
                    }
                )
                per_seed_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "mode": str(mode),
                        "test_macro_f1": float(result.test_metrics["macro_f1"]),
                        "delta_vs_baseline": float(result.test_metrics["macro_f1"]) - float(baseline_result.test_metrics["macro_f1"]),
                        "delta_vs_t2a_default": float(result.test_metrics["macro_f1"]) - float(t2a_result.test_metrics["macro_f1"]),
                        "delta_vs_t3_shared_rebasis": float(result.test_metrics["macro_f1"]) - float(t3_result.test_metrics["macro_f1"]),
                        "delta_vs_t4b_window_radial_gate": float(result.test_metrics["macro_f1"]) - float(t4b_result.test_metrics["macro_f1"]),
                        "delta_vs_t6b_reference": (
                            float(result.test_metrics["macro_f1"]) - float(t6b_reference[(str(dataset).lower(), int(seed))])
                            if (str(dataset).lower(), int(seed)) in t6b_reference
                            else float("nan")
                        ),
                    }
                )
                diagnostics_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "mode": str(mode),
                        **{k: float(v) for k, v in result.diagnostics.items()},
                    }
                )

            pool_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "candidate_window_count": int(constructive_pool.summary["candidate_window_count"]),
                    "safe_window_count": int(constructive_pool.summary["safe_window_count"]),
                    "accepted_window_count": int(constructive_pool.summary["accepted_window_count"]),
                    "accept_rate": float(constructive_pool.summary["accept_rate"]),
                    "source_trial_coverage": float(constructive_pool.summary["source_trial_coverage"]),
                    "class_balance_proxy": float(constructive_pool.summary["class_balance_proxy"]),
                    "pool_type": str(constructive_pool.summary["pool_type"]),
                    "informative_gate": str(constructive_pool.summary["informative_gate"]),
                }
            )
            for row in constructive_pool.class_coverage_rows:
                coverage_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in row.items()},
                    }
                )
            for row in t7a_rebasis.class_rows:
                basis_family_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in row.items()},
                    }
                )

    config_df = pd.DataFrame(config_rows)
    per_seed_df = pd.DataFrame(per_seed_rows)
    pool_df = pd.DataFrame(pool_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    basis_family_df = pd.DataFrame(basis_family_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows)

    summary_rows: List[Dict[str, object]] = []
    for dataset in datasets:
        ds = per_seed_df[per_seed_df["dataset"] == dataset].copy()
        if ds.empty:
            continue

        def vals(mode: str) -> List[float]:
            return ds[ds["mode"] == mode]["test_macro_f1"].astype(float).tolist()

        baseline_vals = vals("baseline")
        t2a_vals = vals("t2a_default")
        t3_vals = vals("t3_shared_rebasis")
        t4b_vals = vals("t4b_window_radial_gate")
        t7a_vals = vals("t7a_class_conditioned_rebasis")
        mode_scores = {
            "baseline": _mean_std(baseline_vals)[0],
            "t2a_default": _mean_std(t2a_vals)[0],
            "t3_shared_rebasis": _mean_std(t3_vals)[0],
            "t4b_window_radial_gate": _mean_std(t4b_vals)[0],
            "t7a_class_conditioned_rebasis": _mean_std(t7a_vals)[0],
        }
        best_mode = max(mode_scores.items(), key=lambda kv: kv[1])[0]
        summary_rows.append(
            {
                "dataset": str(dataset),
                "baseline_macro_f1_mean": _mean_std(baseline_vals)[0],
                "baseline_macro_f1_std": _mean_std(baseline_vals)[1],
                "t2a_default_macro_f1_mean": _mean_std(t2a_vals)[0],
                "t2a_default_macro_f1_std": _mean_std(t2a_vals)[1],
                "t3_shared_rebasis_macro_f1_mean": _mean_std(t3_vals)[0],
                "t3_shared_rebasis_macro_f1_std": _mean_std(t3_vals)[1],
                "t4b_window_radial_gate_macro_f1_mean": _mean_std(t4b_vals)[0],
                "t4b_window_radial_gate_macro_f1_std": _mean_std(t4b_vals)[1],
                "t7a_class_conditioned_rebasis_macro_f1_mean": _mean_std(t7a_vals)[0],
                "t7a_class_conditioned_rebasis_macro_f1_std": _mean_std(t7a_vals)[1],
                "best_mode": str(best_mode),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    config_csv = os.path.join(out_root, "dynamic_feedback_rebasis_t7a_config_table.csv")
    per_seed_csv = os.path.join(out_root, "dynamic_feedback_rebasis_t7a_per_seed.csv")
    summary_csv = os.path.join(out_root, "dynamic_feedback_rebasis_t7a_dataset_summary.csv")
    pool_csv = os.path.join(out_root, "dynamic_feedback_rebasis_t7a_constructive_pool_summary.csv")
    coverage_csv = os.path.join(out_root, "dynamic_feedback_rebasis_t7a_class_coverage_summary.csv")
    basis_family_csv = os.path.join(out_root, "dynamic_feedback_rebasis_t7a_basis_family_summary.csv")
    diagnostics_csv = os.path.join(out_root, "dynamic_feedback_rebasis_t7a_diagnostics_summary.csv")
    config_df.to_csv(config_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    pool_df.to_csv(pool_csv, index=False)
    coverage_df.to_csv(coverage_csv, index=False)
    basis_family_df.to_csv(basis_family_csv, index=False)
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    lines: List[str] = [
        "# T7a Window-level Constructive Pool + Class-Conditioned Rebasis Conclusion",
        "",
        "## Key Findings",
    ]
    for dataset in datasets:
        ds = summary_df[summary_df["dataset"] == dataset]
        if ds.empty:
            continue
        row = ds.iloc[0]
        lines.extend(
            [
                f"- `{dataset}`",
                f"  - baseline: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 'baseline')]['test_macro_f1'].tolist())}",
                f"  - t2a_default: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't2a_default')]['test_macro_f1'].tolist())}",
                f"  - t3_shared_rebasis: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't3_shared_rebasis')]['test_macro_f1'].tolist())}",
                f"  - t4b_window_radial_gate: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't4b_window_radial_gate')]['test_macro_f1'].tolist())}",
                f"  - t7a_class_conditioned_rebasis: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't7a_class_conditioned_rebasis')]['test_macro_f1'].tolist())}",
                f"  - best_mode: `{row['best_mode']}`",
            ]
        )
        ds_family = basis_family_df[basis_family_df["dataset"] == dataset]
        if not ds_family.empty:
            lines.extend(
                [
                    f"  - inter_basis_cosine_mean: {float(ds_family['inter_basis_cosine_mean'].mean()):.4f}",
                    f"  - inter_basis_cosine_min: {float(ds_family['inter_basis_cosine_min'].mean()):.4f}",
                    f"  - inter_basis_cosine_max: {float(ds_family['inter_basis_cosine_max'].mean()):.4f}",
                ]
            )

    lines.extend(
        [
            "",
            "## Reading Notes",
            "- T7a routes class-conditioned containers only during training-time augmentation generation.",
            "- Validation/test always consume original trajectories only; there is no class-conditioned test-time routing.",
            "- Each class-conditioned axis uses train-only one-vs-rest fitting so the container remains discriminative rather than collapsing into class-only unsupervised PCA-like axes.",
        ]
    )

    conclusion_path = os.path.join(out_root, "dynamic_feedback_rebasis_t7a_conclusion.md")
    with open(conclusion_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
