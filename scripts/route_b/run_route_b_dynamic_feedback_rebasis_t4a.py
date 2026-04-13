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
from route_b_unified.trajectory_feedback_rebasis import fit_trajectory_feedback_rebasis  # noqa: E402
from route_b_unified.trajectory_feedback_rebasis_t4 import (  # noqa: E402
    TrajectoryClassConditionedRebasisResult,
    fit_trajectory_class_conditioned_rebasis,
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
    return final_tids, final_labels, final_seqs, dict(op_meta), diagnostics


def _augment_with_family(
    state,
    *,
    rebasis_result: TrajectoryClassConditionedRebasisResult,
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
    aug_tids = [f"{tid}__t4a_class_family_aug" for tid in train_tids]
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
    return final_tids, final_labels, final_seqs, meta, diagnostics


def main() -> None:
    p = argparse.ArgumentParser(
        description="T4a class-conditioned basis-family rebasis probe: baseline vs T2a default vs T3 shared rebasis vs T4a class-conditioned rebasis."
    )
    p.add_argument("--datasets", type=str, default="natops,selfregulationscp1")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/route_b_dynamic_feedback_rebasis_t4a_20260329_formal")
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

    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_seed_list(args.seeds)
    out_root = str(args.out_root)
    _ensure_dir(out_root)

    config_rows: List[Dict[str, object]] = []
    per_seed_rows: List[Dict[str, object]] = []
    pool_rows: List[Dict[str, object]] = []
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
                os.path.join(seed_dir, "dynamic_feedback_rebasis_t4a_split_meta.json"),
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

            pool_result = build_trajectory_feedback_pool(
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
            candidate_dump = [
                {
                    k: (
                        np.asarray(v, dtype=np.float32).tolist()
                        if isinstance(v, np.ndarray)
                        else v
                    )
                    for k, v in row.items()
                    if k != "z_seq_aug"
                }
                for row in pool_result.candidate_rows
            ]
            _write_json(
                os.path.join(seed_dir, "dynamic_feedback_rebasis_t4a_feedback_pool.json"),
                {
                    "summary": dict(pool_result.summary),
                    "candidates": candidate_dump,
                },
            )

            shared_rebasis = fit_trajectory_feedback_rebasis(
                orig_train_z_seq_list=state.train.z_seq_list,
                feedback_z_seq_list=pool_result.accepted_z_seq_list,
                old_operator=frozen_operator,
                operator_cfg=operator_cfg,
            )
            _write_json(
                os.path.join(seed_dir, "dynamic_feedback_rebasis_t4a_shared_basis_shift.json"),
                {"summary": dict(shared_rebasis.summary)},
            )
            t3_tids, t3_labels, t3_seqs, t3_meta, t3_diag = _augment_with_shared_operator(
                state,
                operator=shared_rebasis.operator_new,
                gamma_main=float(args.gamma_main),
                smooth_lambda=float(args.smooth_lambda),
                mode="t3_rebasis",
            )
            t3_meta = dict(t3_meta)
            t3_meta["rebasis_summary"] = dict(shared_rebasis.summary)
            t3_meta["feedback_summary"] = dict(pool_result.summary)
            t3_result = evaluate_trajectory_train_final(
                state,
                seed=int(seed),
                model_cfg=model_cfg,
                eval_cfg=TrajectoryPIAEvalConfig(
                    operator_mode="t3_rebasis",
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
            _save_result_json(os.path.join(seed_dir, "t3_rebasis_result.json"), t3_result)

            t4a_rebasis = fit_trajectory_class_conditioned_rebasis(
                orig_train_labels=[int(v) for v in state.train.y.tolist()],
                orig_train_z_seq_list=state.train.z_seq_list,
                feedback_labels=pool_result.accepted_labels,
                feedback_z_seq_list=pool_result.accepted_z_seq_list,
                old_operator=frozen_operator,
                operator_cfg=operator_cfg,
            )
            _write_json(
                os.path.join(seed_dir, "dynamic_feedback_rebasis_t4a_basis_family.json"),
                {
                    "summary": dict(t4a_rebasis.summary),
                    "class_rows": [
                        {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in row.items()}
                        for row in t4a_rebasis.class_rows
                    ],
                },
            )
            t4a_tids, t4a_labels, t4a_seqs, t4a_meta, t4a_diag = _augment_with_family(
                state,
                rebasis_result=t4a_rebasis,
                gamma_main=float(args.gamma_main),
                smooth_lambda=float(args.smooth_lambda),
            )
            t4a_meta["feedback_summary"] = dict(pool_result.summary)
            t4a_result = evaluate_trajectory_train_final(
                state,
                seed=int(seed),
                model_cfg=model_cfg,
                eval_cfg=TrajectoryPIAEvalConfig(
                    operator_mode="t4a_class_conditioned_rebasis",
                    gamma_main=float(args.gamma_main),
                    smooth_lambda=float(args.smooth_lambda),
                    **eval_common,
                ),
                train_tids=t4a_tids,
                train_labels=t4a_labels,
                train_z_seq_list=t4a_seqs,
                diagnostics=t4a_diag,
                operator_meta=t4a_meta,
            )
            _save_result_json(os.path.join(seed_dir, "t4a_class_conditioned_rebasis_result.json"), t4a_result)

            mode_rows = [
                ("baseline", baseline_result),
                ("t2a_default", t2a_result),
                ("t3_shared_rebasis", t3_result),
                ("t4a_class_conditioned_rebasis", t4a_result),
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
                    "candidate_count": int(pool_result.summary["candidate_count"]),
                    "accepted_count": int(pool_result.summary["accepted_count"]),
                    "accept_rate": float(pool_result.summary["accept_rate"]),
                    "class_balance_proxy": float(pool_result.summary["class_balance_proxy"]),
                    "mean_purity_drop_accepted": float(pool_result.summary["mean_purity_drop_accepted"]),
                    "mean_continuity_ratio_accepted": float(pool_result.summary["mean_continuity_ratio_accepted"]),
                    "pool_type": str(pool_result.summary["pool_type"]),
                }
            )
            for row in t4a_rebasis.class_rows:
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
        t4a_vals = vals("t4a_class_conditioned_rebasis")
        mode_scores = {
            "baseline": _mean_std(baseline_vals)[0],
            "t2a_default": _mean_std(t2a_vals)[0],
            "t3_shared_rebasis": _mean_std(t3_vals)[0],
            "t4a_class_conditioned_rebasis": _mean_std(t4a_vals)[0],
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
                "t4a_class_conditioned_rebasis_macro_f1_mean": _mean_std(t4a_vals)[0],
                "t4a_class_conditioned_rebasis_macro_f1_std": _mean_std(t4a_vals)[1],
                "best_mode": str(best_mode),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    config_csv = os.path.join(out_root, "dynamic_feedback_rebasis_t4a_config_table.csv")
    per_seed_csv = os.path.join(out_root, "dynamic_feedback_rebasis_t4a_per_seed.csv")
    summary_csv = os.path.join(out_root, "dynamic_feedback_rebasis_t4a_dataset_summary.csv")
    pool_csv = os.path.join(out_root, "dynamic_feedback_rebasis_t4a_pool_summary.csv")
    basis_family_csv = os.path.join(out_root, "dynamic_feedback_rebasis_t4a_basis_family_summary.csv")
    diagnostics_csv = os.path.join(out_root, "dynamic_feedback_rebasis_t4a_diagnostics_summary.csv")
    config_df.to_csv(config_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    pool_df.to_csv(pool_csv, index=False)
    basis_family_df.to_csv(basis_family_csv, index=False)
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    lines: List[str] = [
        "# T4a Class-Conditioned Basis Family Re-basis Conclusion",
        "",
        "This first-pass T4a probe freezes the T3 feedback-pool protocol and upgrades only the rebasis organization from a shared single axis to a class-conditioned single-axis family.",
        "",
        "Important scope note:",
        "",
        "- T4a measures the overall effect of a class-conditioned basis-family generator during training-time augmentation.",
        "- It does not isolate basis organization as the only causal source, because train-time basis selection also uses true class labels.",
        "",
    ]
    for dataset in datasets:
        ds_summary = summary_df[summary_df["dataset"] == dataset]
        ds_family = basis_family_df[basis_family_df["dataset"] == dataset]
        if ds_summary.empty:
            continue
        lines.append(f"## {dataset}")
        lines.append("")
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
            f"- t4a_class_conditioned_rebasis: {_format_mean_std(per_seed_df[(per_seed_df['dataset'] == dataset) & (per_seed_df['mode'] == 't4a_class_conditioned_rebasis')]['test_macro_f1'].tolist())}"
        )
        if not ds_family.empty:
            lines.append(f"- inter-basis cosine mean: {float(np.mean(ds_family['inter_basis_cosine_mean'])):.4f}")
            lines.append(f"- inter-basis cosine min: {float(np.min(ds_family['inter_basis_cosine_min'])):.4f}")
            lines.append(f"- inter-basis cosine max: {float(np.max(ds_family['inter_basis_cosine_max'])):.4f}")
        lines.append("")

    lines.extend(
        [
            "## Judgment",
            "",
            "- T4a only counts as successful if class-conditioned basis-family rebasis improves end performance or yields clearer SCP1 gains than T3 shared rebasis.",
            "- Basis family differences alone do not count as success; they must be non-degenerate, non-runaway, and aligned with better downstream behavior.",
            "- If T4a fails, the next likely bottleneck is feedback-pool object quality rather than another round of operator parameter tuning.",
        ]
    )
    conclusion_md = os.path.join(out_root, "dynamic_feedback_rebasis_t4a_conclusion.md")
    with open(conclusion_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[T4a] wrote config to: {config_csv}")
    print(f"[T4a] wrote per-seed to: {per_seed_csv}")
    print(f"[T4a] wrote dataset summary to: {summary_csv}")
    print(f"[T4a] wrote pool summary to: {pool_csv}")
    print(f"[T4a] wrote basis family summary to: {basis_family_csv}")
    print(f"[T4a] wrote diagnostics to: {diagnostics_csv}")
    print(f"[T4a] wrote conclusion to: {conclusion_md}")


if __name__ == "__main__":
    main()
