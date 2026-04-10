#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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


def _parse_csv_list(text: str) -> List[str]:
    out = [tok.strip().lower() for tok in str(text).split(",") if tok.strip()]
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
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def _format_mean_std(values: Sequence[float]) -> str:
    mean, std = _mean_std(values)
    if not np.isfinite(mean):
        return "n/a"
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


def main() -> None:
    p = argparse.ArgumentParser(
        description="T4b window-conditioned rebasis-informative feedback pool probe."
    )
    p.add_argument("--main-datasets", type=str, default="selfregulationscp1")
    p.add_argument("--anchor-datasets", type=str, default="natops")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/route_b_dynamic_feedback_rebasis_t4b_20260329_formal")
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
    args = p.parse_args()

    main_datasets = _parse_csv_list(args.main_datasets)
    anchor_datasets = _parse_csv_list(args.anchor_datasets)
    datasets = list(dict.fromkeys(main_datasets + anchor_datasets))
    dataset_role = {
        ds: ("main" if ds in main_datasets else "anchor")
        for ds in datasets
    }
    seeds = _parse_seed_list(args.seeds)
    out_root = str(args.out_root)
    _ensure_dir(out_root)

    config_rows: List[Dict[str, object]] = []
    per_seed_rows: List[Dict[str, object]] = []
    basis_rows: List[Dict[str, object]] = []
    diagnostics_rows: List[Dict[str, object]] = []
    window_pool_rows: List[Dict[str, object]] = []

    gate_modes = [
        ("t4b_window_safety_only", "safety_only"),
        ("t4b_window_radial_gate", "radial"),
        ("t4b_window_margin_gate", "margin"),
    ]

    for dataset in datasets:
        role = dataset_role[str(dataset)]
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
            train_tids = [str(v) for v in state.train.tids.tolist()]
            train_labels = [int(v) for v in state.train.y.tolist()]

            ref_stats = build_window_feedback_reference_stats(
                train_labels=train_labels,
                train_z_seq_list=state.train.z_seq_list,
            )
            _write_json(
                os.path.join(seed_dir, "dynamic_feedback_rebasis_t4b_split_meta.json"),
                {
                    "dataset": str(dataset),
                    "dataset_role": str(role),
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
                    "window_reference_scope": "orig_train_only_windows",
                    "margin_center_scope": "orig_train_only_window_class_centers",
                },
            )

            mode_results: Dict[str, TrajectoryPIAEvalResult] = {}

            if role == "main":
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
                mode_results["baseline"] = baseline_result
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
                mode_results["t2a_default"] = t2a_result
                _save_result_json(os.path.join(seed_dir, "t2a_default_result.json"), t2a_result)

            t3_pool_result = build_trajectory_feedback_pool(
                train_tids=train_tids,
                train_labels=train_labels,
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
            t3_rebasis = fit_trajectory_feedback_rebasis(
                orig_train_z_seq_list=state.train.z_seq_list,
                feedback_z_seq_list=t3_pool_result.accepted_z_seq_list,
                old_operator=frozen_operator,
                operator_cfg=operator_cfg,
            )
            _write_json(
                os.path.join(seed_dir, "dynamic_feedback_rebasis_t4b_t3_feedback_pool.json"),
                {"summary": dict(t3_pool_result.summary)},
            )
            _write_json(
                os.path.join(seed_dir, "dynamic_feedback_rebasis_t4b_t3_basis_shift.json"),
                {"summary": dict(t3_rebasis.summary)},
            )
            t3_tids, t3_labels, t3_seqs, t3_meta, t3_diag = _augment_with_shared_operator(
                state,
                operator=t3_rebasis.operator_new,
                gamma_main=float(args.gamma_main),
                smooth_lambda=float(args.smooth_lambda),
                mode="t3_rebasis",
                extra_meta={
                    "feedback_summary": dict(t3_pool_result.summary),
                    "rebasis_summary": dict(t3_rebasis.summary),
                    "generator_mode": "frozen_t2a_default",
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
            mode_results["t3_shared_rebasis"] = t3_result
            _save_result_json(os.path.join(seed_dir, "t3_shared_rebasis_result.json"), t3_result)
            basis_rows.append(
                {
                    "dataset": str(dataset),
                    "dataset_role": str(role),
                    "seed": int(seed),
                    "mode": "t3_shared_rebasis",
                    **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in t3_rebasis.summary.items()},
                }
            )

            for mode_name, gate_name in gate_modes:
                pool_result = build_window_feedback_pool(
                    train_tids=train_tids,
                    train_labels=train_labels,
                    train_z_seq_list=state.train.z_seq_list,
                    operator=frozen_operator,
                    reference_stats=ref_stats,
                    cfg=TrajectoryWindowFeedbackPoolConfig(
                        gamma_main=float(args.gamma_main),
                        smooth_lambda=float(args.smooth_lambda),
                        knn_k=int(args.knn_k),
                        max_purity_drop=float(args.max_purity_drop),
                        continuity_quantile=float(args.continuity_quantile),
                        informative_gate=str(gate_name),
                    ),
                )
                _write_json(
                    os.path.join(seed_dir, f"{mode_name}_window_pool.json"),
                    {
                        "summary": dict(pool_result.summary),
                        "candidate_count": int(len(pool_result.candidate_rows)),
                    },
                )
                rebasis_result = fit_trajectory_feedback_rebasis(
                    orig_train_z_seq_list=state.train.z_seq_list,
                    feedback_z_seq_list=pool_result.accepted_window_seq_list,
                    old_operator=frozen_operator,
                    operator_cfg=operator_cfg,
                )
                _write_json(
                    os.path.join(seed_dir, f"{mode_name}_basis_shift.json"),
                    {"summary": dict(rebasis_result.summary)},
                )
                aug_tids, aug_labels, aug_seqs, op_meta, diag = _augment_with_shared_operator(
                    state,
                    operator=rebasis_result.operator_new,
                    gamma_main=float(args.gamma_main),
                    smooth_lambda=float(args.smooth_lambda),
                    mode=mode_name,
                    extra_meta={
                        "feedback_summary": dict(pool_result.summary),
                        "rebasis_summary": dict(rebasis_result.summary),
                        "generator_mode": "frozen_t2a_default",
                    },
                )
                result = evaluate_trajectory_train_final(
                    state,
                    seed=int(seed),
                    model_cfg=model_cfg,
                    eval_cfg=TrajectoryPIAEvalConfig(
                        operator_mode=str(mode_name),
                        gamma_main=float(args.gamma_main),
                        smooth_lambda=float(args.smooth_lambda),
                        **eval_common,
                    ),
                    train_tids=aug_tids,
                    train_labels=aug_labels,
                    train_z_seq_list=aug_seqs,
                    diagnostics=diag,
                    operator_meta=op_meta,
                )
                mode_results[str(mode_name)] = result
                _save_result_json(os.path.join(seed_dir, f"{mode_name}_result.json"), result)

                window_pool_rows.append(
                    {
                        "dataset": str(dataset),
                        "dataset_role": str(role),
                        "seed": int(seed),
                        "mode": str(mode_name),
                        **{
                            k: float(v) if isinstance(v, (int, float, np.floating)) else v
                            for k, v in pool_result.summary.items()
                        },
                    }
                )
                basis_rows.append(
                    {
                        "dataset": str(dataset),
                        "dataset_role": str(role),
                        "seed": int(seed),
                        "mode": str(mode_name),
                        **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in rebasis_result.summary.items()},
                    }
                )

            mode_order = (
                ["baseline", "t2a_default", "t3_shared_rebasis"] + [m for m, _ in gate_modes]
                if role == "main"
                else ["t3_shared_rebasis"] + [m for m, _ in gate_modes]
            )
            baseline_f1 = float(mode_results["baseline"].test_metrics["macro_f1"]) if "baseline" in mode_results else float("nan")
            t2a_f1 = float(mode_results["t2a_default"].test_metrics["macro_f1"]) if "t2a_default" in mode_results else float("nan")
            t3_f1 = float(mode_results["t3_shared_rebasis"].test_metrics["macro_f1"])
            safety_f1 = float(mode_results["t4b_window_safety_only"].test_metrics["macro_f1"])
            radial_f1 = float(mode_results["t4b_window_radial_gate"].test_metrics["macro_f1"])
            margin_f1 = float(mode_results["t4b_window_margin_gate"].test_metrics["macro_f1"])

            for mode in mode_order:
                result = mode_results[str(mode)]
                config_rows.append(
                    {
                        "dataset": str(dataset),
                        "dataset_role": str(role),
                        "seed": int(seed),
                        "mode": str(mode),
                        "gamma_main": float(args.gamma_main),
                        "smooth_lambda": float(args.smooth_lambda),
                        "window_len": int(state.window_len),
                        "hop_len": int(state.hop_len),
                        "epochs": int(args.epochs),
                        "batch_size": int(args.batch_size),
                    }
                )
                per_seed_rows.append(
                    {
                        "dataset": str(dataset),
                        "dataset_role": str(role),
                        "seed": int(seed),
                        "mode": str(mode),
                        "test_macro_f1": float(result.test_metrics["macro_f1"]),
                        "delta_vs_baseline": float(result.test_metrics["macro_f1"]) - baseline_f1 if np.isfinite(baseline_f1) else float("nan"),
                        "delta_vs_t2a_default": float(result.test_metrics["macro_f1"]) - t2a_f1 if np.isfinite(t2a_f1) else float("nan"),
                        "delta_vs_t3_shared_rebasis": float(result.test_metrics["macro_f1"]) - t3_f1,
                        "delta_vs_window_safety_only": float(result.test_metrics["macro_f1"]) - safety_f1,
                        "delta_vs_window_radial_gate": float(result.test_metrics["macro_f1"]) - radial_f1,
                        "delta_vs_window_margin_gate": float(result.test_metrics["macro_f1"]) - margin_f1,
                    }
                )
                diagnostics_rows.append(
                    {
                        "dataset": str(dataset),
                        "dataset_role": str(role),
                        "seed": int(seed),
                        "mode": str(mode),
                        **{k: float(v) for k, v in result.diagnostics.items()},
                    }
                )

    config_df = pd.DataFrame(config_rows)
    per_seed_df = pd.DataFrame(per_seed_rows)
    basis_df = pd.DataFrame(basis_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows)
    window_pool_df = pd.DataFrame(window_pool_rows)

    summary_rows: List[Dict[str, object]] = []
    for dataset in datasets:
        ds = per_seed_df[per_seed_df["dataset"] == dataset].copy()
        if ds.empty:
            continue
        row = {"dataset": str(dataset), "dataset_role": str(dataset_role[str(dataset)])}
        for mode in [
            "baseline",
            "t2a_default",
            "t3_shared_rebasis",
            "t4b_window_safety_only",
            "t4b_window_radial_gate",
            "t4b_window_margin_gate",
        ]:
            vals = ds[ds["mode"] == mode]["test_macro_f1"].astype(float).tolist()
            mean, std = _mean_std(vals)
            row[f"{mode}_macro_f1_mean"] = mean
            row[f"{mode}_macro_f1_std"] = std
        candidates = {
            mode: row.get(f"{mode}_macro_f1_mean", float("nan"))
            for mode in [
                "t3_shared_rebasis",
                "t4b_window_safety_only",
                "t4b_window_radial_gate",
                "t4b_window_margin_gate",
            ]
            if np.isfinite(row.get(f"{mode}_macro_f1_mean", float("nan")))
        }
        row["best_mode"] = max(candidates.items(), key=lambda kv: kv[1])[0] if candidates else "n/a"
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    config_csv = os.path.join(out_root, "dynamic_feedback_rebasis_t4b_config_table.csv")
    per_seed_csv = os.path.join(out_root, "dynamic_feedback_rebasis_t4b_per_seed.csv")
    summary_csv = os.path.join(out_root, "dynamic_feedback_rebasis_t4b_dataset_summary.csv")
    window_pool_csv = os.path.join(out_root, "dynamic_feedback_rebasis_t4b_window_pool_summary.csv")
    basis_csv = os.path.join(out_root, "dynamic_feedback_rebasis_t4b_basis_shift_summary.csv")
    diagnostics_csv = os.path.join(out_root, "dynamic_feedback_rebasis_t4b_diagnostics_summary.csv")
    config_df.to_csv(config_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    window_pool_df.to_csv(window_pool_csv, index=False)
    basis_df.to_csv(basis_csv, index=False)
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    lines: List[str] = [
        "# T4b Window-Conditioned Rebasis-Informative Feedback Pool Conclusion",
        "",
        "This first-pass T4b probe freezes generator, shared rebasis, window policy, and classifier, and upgrades only the feedback-pool object from whole trajectories to window-level objects.",
        "",
        "Interpretation guardrails:",
        "",
        "- `radial_gain` is treated as an outward-expansion proxy, not as a proven rebasis-optimal signal.",
        "- `margin_gain` is treated as a discriminative-gain proxy, not as a proven final answer.",
        "- admitted windows enter rebasis as length-1 pseudo sequences; T4b therefore measures window-conditioned rebasis signal, not full segment-aware rebasis geometry.",
        "",
    ]
    for dataset in datasets:
        ds_summary = summary_df[summary_df["dataset"] == dataset]
        if ds_summary.empty:
            continue
        role = str(ds_summary["dataset_role"].iloc[0])
        lines.append(f"## {dataset} ({role})")
        lines.append("")
        if role == "main":
            lines.append(
                f"- baseline: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 'baseline')]['test_macro_f1'].tolist())}"
            )
            lines.append(
                f"- t2a_default: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't2a_default')]['test_macro_f1'].tolist())}"
            )
        lines.append(
            f"- t3_shared_rebasis: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't3_shared_rebasis')]['test_macro_f1'].tolist())}"
        )
        lines.append(
            f"- t4b_window_safety_only: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't4b_window_safety_only')]['test_macro_f1'].tolist())}"
        )
        lines.append(
            f"- t4b_window_radial_gate: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't4b_window_radial_gate')]['test_macro_f1'].tolist())}"
        )
        lines.append(
            f"- t4b_window_margin_gate: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't4b_window_margin_gate')]['test_macro_f1'].tolist())}"
        )
        lines.append("")

    lines.extend(
        [
            "## Success Layers",
            "",
            "- Weak success: `window_safety_only > t3_shared_rebasis`.",
            "- Medium success: `window_radial_gate` or `window_margin_gate > window_safety_only`.",
            "- Strong success: SCP1 performance improves and basis shift changes from near-static to non-trivial and interpretable.",
        ]
    )
    conclusion_md = os.path.join(out_root, "dynamic_feedback_rebasis_t4b_conclusion.md")
    with open(conclusion_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[T4b] wrote config to: {config_csv}")
    print(f"[T4b] wrote per-seed to: {per_seed_csv}")
    print(f"[T4b] wrote dataset summary to: {summary_csv}")
    print(f"[T4b] wrote window-pool summary to: {window_pool_csv}")
    print(f"[T4b] wrote basis-shift summary to: {basis_csv}")
    print(f"[T4b] wrote diagnostics to: {diagnostics_csv}")
    print(f"[T4b] wrote conclusion to: {conclusion_md}")


if __name__ == "__main__":
    main()
