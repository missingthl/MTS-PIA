#!/usr/bin/env python
"""Rebuild LRAES + curriculum summary tables from existing run outputs.

This is a lightweight post-processing helper for cases where datasets were run
sequentially into the same out_root and the top-level summary files were
overwritten by the last invocation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_phase15_lraes_curriculum_probe import (
    _entropy_from_probs,
    _format_mean_std,
    _result_label,
    _solver_comment,
    _stats_string,
)
from scripts.run_phase15_mainline_freeze import _summarize_dir_profile


def _summary_stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _variant_from_cond_dir(name: str) -> tuple[str, int]:
    if name == "A_baseline":
        return "baseline", 0
    if name == "B_step1b":
        return "single_round_step1b", 0
    if name.startswith("multiround_curriculum_R"):
        return "multiround_curriculum", int(name.rsplit("R", 1)[1])
    if name.startswith("lraes_curriculum_R"):
        return "lraes_curriculum", int(name.rsplit("R", 1)[1])
    if name.startswith("lraes_beta") and "_R" in name:
        return "lraes_curriculum", int(name.rsplit("R", 1)[1])
    raise ValueError(f"Unknown condition directory: {name}")


def _health_row_from_condition(dataset: str, seed: int, cond_dir: Path) -> Dict[str, object] | None:
    metrics_path = cond_dir / "metrics.json"
    meta_path = cond_dir / "run_meta.json"
    mech_path = cond_dir / "mech_table.csv"
    if not metrics_path.exists() or not meta_path.exists():
        return None

    meta = _read_json(meta_path)
    variant, round_id = _variant_from_cond_dir(cond_dir.name)
    if variant == "baseline":
        return None

    mech = meta.get("mech", {})
    dir_profile = mech.get("dir_profile", {})
    dir_summary = _summarize_dir_profile(dir_profile)
    beta = np.nan
    frozen_dir_count = 0
    expanded_dir_count = 0
    comment = "equal_weight_single_round"
    usage_probs = np.zeros((int(meta.get("step1b_config", {}).get("k_dir", 5)),), dtype=np.float64)

    if variant == "single_round_step1b":
        frac_map = meta.get("augmentation", {}).get("mixing_stats", {}).get("direction_pick_fraction", {})
        if frac_map:
            max_dir = max(int(k) for k in frac_map.keys())
            usage_probs = np.asarray(
                [float(frac_map.get(str(i), 0.0)) for i in range(max_dir + 1)],
                dtype=np.float64,
            )
    else:
        beta = _float_or_nan(meta.get("lraes_config", {}).get("beta")) if variant == "lraes_curriculum" else np.nan
        prob_map = meta.get("direction_probs", {})
        if prob_map:
            max_dir = max(int(k) for k in prob_map.keys())
            usage_probs = np.asarray(
                [float(prob_map.get(str(i), 0.0)) for i in range(max_dir + 1)],
                dtype=np.float64,
            )
        state_map = meta.get("direction_state", {})
        frozen_dir_count = int(sum(1 for s in state_map.values() if str(s) == "freeze"))
        expanded_dir_count = int(sum(1 for s in state_map.values() if str(s) == "expand"))
        comment = "lraes_curriculum_round" if variant == "lraes_curriculum" else "curriculum_round"

    if usage_probs.size == 0:
        usage_probs = np.asarray([1.0], dtype=np.float64)

    row = {
        "dataset": dataset,
        "seed": int(seed),
        "variant": variant,
        "beta": beta,
        "round_id": int(round_id),
        "direction_usage_entropy": float(_entropy_from_probs(usage_probs)),
        "worst_dir_summary": dir_summary["dir_profile_summary"],
        "frozen_dir_count": int(frozen_dir_count),
        "expanded_dir_count": int(expanded_dir_count),
        "direction_health_comment": str(comment),
    }
    if mech_path.exists():
        mech_df = pd.read_csv(mech_path)
        row["per_dir_margin_summary"] = "|".join(
            f"{int(r.direction_id)}:{float(r.margin_drop_median):.4f}" for _, r in mech_df.iterrows()
        )
        row["per_dir_flip_summary"] = "|".join(
            f"{int(r.direction_id)}:{float(r.flip_rate):.4f}" for _, r in mech_df.iterrows()
        )
    else:
        row["per_dir_margin_summary"] = "n/a"
        row["per_dir_flip_summary"] = "n/a"
    return row


def rebuild(out_root: Path) -> None:
    perf_rows: List[Dict[str, object]] = []
    health_rows: List[Dict[str, object]] = []
    solver_seed_rows: List[Dict[str, object]] = []

    datasets = sorted([p.name for p in out_root.iterdir() if p.is_dir() and (p / "seed1").exists()])
    for dataset in datasets:
        dataset_dir = out_root / dataset
        for seed_dir in sorted(dataset_dir.glob("seed*")):
            seed = int(seed_dir.name.replace("seed", ""))
            for cond_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
                metrics_path = cond_dir / "metrics.json"
                meta_path = cond_dir / "run_meta.json"
                if not metrics_path.exists() or not meta_path.exists():
                    continue
                metrics = _read_json(metrics_path)
                meta = _read_json(meta_path)
                variant, round_id = _variant_from_cond_dir(cond_dir.name)
                beta = np.nan
                if variant == "lraes_curriculum":
                    beta = _float_or_nan(meta.get("lraes_config", {}).get("beta"))
                perf_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "variant": variant,
                        "beta": beta,
                        "round_id": int(round_id),
                        "acc": float(metrics["trial_acc"]),
                        "f1": float(metrics["trial_macro_f1"]),
                    }
                )
                hrow = _health_row_from_condition(dataset, seed, cond_dir)
                if hrow is not None:
                    health_rows.append(hrow)

                if variant == "lraes_curriculum" and int(round_id) == 1:
                    db = meta.get("direction_bank", {})
                    selected_eigs = np.asarray(db.get("selected_eigenvalues", []), dtype=np.float64)
                    eig_pos_eps = float(db.get("eig_pos_eps", meta.get("lraes_config", {}).get("eig_pos_eps", 1e-9)))
                    low_quality_axis_count = int(db.get("low_quality_axis_count", 0))
                    max_pos = bool(np.any(selected_eigs > eig_pos_eps))
                    if not max_pos:
                        solver_state = "fully_risk_dominated"
                    elif low_quality_axis_count > 0:
                        solver_state = "marginal"
                    else:
                        solver_state = "safe_expandable"
                    solver_seed_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "beta": _float_or_nan(db.get("beta")),
                            "local_matrix_rank_summary": str(db.get("local_matrix_rank_summary", "n/a")),
                            "top1_eigenvalue": float(np.max(selected_eigs)) if selected_eigs.size else 0.0,
                            "topK_eigenvalue_summary": _stats_string(selected_eigs, fmt=".6f"),
                            "topK_positive_count": int(np.sum(selected_eigs > eig_pos_eps)),
                            "topK_nonpositive_count": int(np.sum(selected_eigs <= eig_pos_eps)),
                            "max_eigenvalue_is_positive": bool(max_pos),
                            "solver_state": solver_state,
                            "selected_axis_variance_summary": str(db.get("selected_axis_variance_summary", "n/a")),
                            "low_quality_axis_count": int(low_quality_axis_count),
                            "solver_comment": _solver_comment(
                                solver_state=solver_state,
                                low_quality_axis_count=low_quality_axis_count,
                                max_eigenvalue_is_positive=max_pos,
                            ),
                        }
                    )

    perf_df = pd.DataFrame(perf_rows).sort_values(["dataset", "variant", "beta", "seed", "round_id"]).reset_index(drop=True)
    perf_df.to_csv(out_root / "summary_per_seed.csv", index=False)

    health_df = pd.DataFrame(health_rows)
    solver_seed_df = pd.DataFrame(solver_seed_rows)

    perf_summary_rows: List[Dict[str, object]] = []
    health_summary_rows: List[Dict[str, object]] = []
    solver_summary_rows: List[Dict[str, object]] = []

    betas = sorted(solver_seed_df["beta"].dropna().astype(float).unique().tolist()) if not solver_seed_df.empty else []

    for dataset, df_ds in perf_df.groupby("dataset"):
        df_base = df_ds[df_ds["variant"] == "baseline"]
        df_s1 = df_ds[df_ds["variant"] == "single_round_step1b"]
        df_curr = df_ds[df_ds["variant"] == "multiround_curriculum"]
        best_curr = df_curr.groupby("round_id", as_index=False)["f1"].mean()
        best_curr_row = best_curr.loc[int(best_curr["f1"].astype(float).idxmax())]
        best_curr_round = int(best_curr_row["round_id"])

        base_acc = _summary_stats(df_base["acc"])
        base_f1 = _summary_stats(df_base["f1"])
        s1_acc = _summary_stats(df_s1["acc"])
        s1_f1 = _summary_stats(df_s1["f1"])
        curr_acc = _summary_stats(df_curr[df_curr["round_id"] == best_curr_round]["acc"])
        curr_f1 = _summary_stats(df_curr[df_curr["round_id"] == best_curr_round]["f1"])

        for beta in betas:
            df_l = df_ds[(df_ds["variant"] == "lraes_curriculum") & np.isclose(df_ds["beta"].astype(float), float(beta))]
            if df_l.empty:
                continue
            best_l = df_l.groupby("round_id", as_index=False)["f1"].mean()
            best_l_row = best_l.loc[int(best_l["f1"].astype(float).idxmax())]
            best_l_round = int(best_l_row["round_id"])
            l_acc = _summary_stats(df_l[df_l["round_id"] == best_l_round]["acc"])
            l_f1 = _summary_stats(df_l[df_l["round_id"] == best_l_round]["f1"])
            delta_vs_curr = float(l_f1["mean"] - curr_f1["mean"])
            perf_summary_rows.append(
                {
                    "dataset": dataset,
                    "beta": float(beta),
                    "beta_role": ("main" if abs(float(beta) - 0.5) <= 1e-9 else "control"),
                    "baseline_acc": _format_mean_std(base_acc["mean"], base_acc["std"]),
                    "baseline_f1": _format_mean_std(base_f1["mean"], base_f1["std"]),
                    "single_round_step1b_acc": _format_mean_std(s1_acc["mean"], s1_acc["std"]),
                    "single_round_step1b_f1": _format_mean_std(s1_f1["mean"], s1_f1["std"]),
                    "multiround_curriculum_acc": _format_mean_std(curr_acc["mean"], curr_acc["std"]),
                    "multiround_curriculum_f1": _format_mean_std(curr_f1["mean"], curr_f1["std"]),
                    "lraes_curriculum_acc": _format_mean_std(l_acc["mean"], l_acc["std"]),
                    "lraes_curriculum_f1": _format_mean_std(l_f1["mean"], l_f1["std"]),
                    "best_curriculum_round": int(best_curr_round),
                    "best_lraes_round": int(best_l_round),
                    "delta_vs_single_round_step1b": float(l_f1["mean"] - s1_f1["mean"]),
                    "delta_vs_multiround_curriculum": delta_vs_curr,
                    "result_label": _result_label(delta_vs_curr),
                }
            )

            df_h_l = health_df[
                (health_df["dataset"] == dataset)
                & (health_df["variant"] == "lraes_curriculum")
                & (health_df["round_id"] == int(best_l_round))
                & np.isclose(health_df["beta"].astype(float), float(beta), equal_nan=False)
            ]
            if not df_h_l.empty:
                note_counts = df_h_l["direction_health_comment"].astype(str).value_counts().to_dict()
                health_summary_rows.append(
                    {
                        "dataset": dataset,
                        "beta": float(beta),
                        "variant": "lraes_curriculum",
                        "direction_usage_entropy": _format_mean_std(
                            float(df_h_l["direction_usage_entropy"].mean()),
                            float(df_h_l["direction_usage_entropy"].std()),
                        ),
                        "worst_dir_summary": " / ".join(df_h_l["worst_dir_summary"].astype(str).tolist()),
                        "frozen_dir_count": _format_mean_std(
                            float(df_h_l["frozen_dir_count"].mean()),
                            float(df_h_l["frozen_dir_count"].std()),
                        ),
                        "expanded_dir_count": _format_mean_std(
                            float(df_h_l["expanded_dir_count"].mean()),
                            float(df_h_l["expanded_dir_count"].std()),
                        ),
                        "direction_health_comment": "; ".join([f"{k}:{v}" for k, v in sorted(note_counts.items())]),
                    }
                )

            df_solver = solver_seed_df[
                (solver_seed_df["dataset"] == dataset)
                & np.isclose(solver_seed_df["beta"].astype(float), float(beta))
            ]
            if not df_solver.empty:
                state_counts = df_solver["solver_state"].astype(str).value_counts().to_dict()
                solver_summary_rows.append(
                    {
                        "dataset": dataset,
                        "beta": float(beta),
                        "local_matrix_rank_summary": " / ".join(df_solver["local_matrix_rank_summary"].astype(str).tolist()),
                        "top1_eigenvalue": _format_mean_std(
                            float(df_solver["top1_eigenvalue"].astype(float).mean()),
                            float(df_solver["top1_eigenvalue"].astype(float).std()),
                        ),
                        "topK_eigenvalue_summary": " / ".join(df_solver["topK_eigenvalue_summary"].astype(str).tolist()),
                        "topK_positive_count": _format_mean_std(
                            float(df_solver["topK_positive_count"].astype(float).mean()),
                            float(df_solver["topK_positive_count"].astype(float).std()),
                        ),
                        "topK_nonpositive_count": _format_mean_std(
                            float(df_solver["topK_nonpositive_count"].astype(float).mean()),
                            float(df_solver["topK_nonpositive_count"].astype(float).std()),
                        ),
                        "max_eigenvalue_is_positive": bool(df_solver["max_eigenvalue_is_positive"].astype(bool).any()),
                        "solver_state": "; ".join([f"{k}:{v}" for k, v in sorted(state_counts.items())]),
                        "selected_axis_variance_summary": " / ".join(df_solver["selected_axis_variance_summary"].astype(str).tolist()),
                        "low_quality_axis_count": _format_mean_std(
                            float(df_solver["low_quality_axis_count"].astype(float).mean()),
                            float(df_solver["low_quality_axis_count"].astype(float).std()),
                        ),
                        "solver_comment": "; ".join(df_solver["solver_comment"].astype(str).value_counts().index.tolist()),
                    }
                )

        df_h_s1 = health_df[(health_df["dataset"] == dataset) & (health_df["variant"] == "single_round_step1b")]
        if not df_h_s1.empty:
            note_counts = df_h_s1["direction_health_comment"].astype(str).value_counts().to_dict()
            health_summary_rows.append(
                {
                    "dataset": dataset,
                    "beta": np.nan,
                    "variant": "single_round_step1b",
                    "direction_usage_entropy": _format_mean_std(
                        float(df_h_s1["direction_usage_entropy"].mean()),
                        float(df_h_s1["direction_usage_entropy"].std()),
                    ),
                    "worst_dir_summary": " / ".join(df_h_s1["worst_dir_summary"].astype(str).tolist()),
                    "frozen_dir_count": _format_mean_std(
                        float(df_h_s1["frozen_dir_count"].mean()),
                        float(df_h_s1["frozen_dir_count"].std()),
                    ),
                    "expanded_dir_count": _format_mean_std(
                        float(df_h_s1["expanded_dir_count"].mean()),
                        float(df_h_s1["expanded_dir_count"].std()),
                    ),
                    "direction_health_comment": "; ".join([f"{k}:{v}" for k, v in sorted(note_counts.items())]),
                }
            )

        df_h_curr = health_df[
            (health_df["dataset"] == dataset)
            & (health_df["variant"] == "multiround_curriculum")
            & (health_df["round_id"] == int(best_curr_round))
        ]
        if not df_h_curr.empty:
            note_counts = df_h_curr["direction_health_comment"].astype(str).value_counts().to_dict()
            health_summary_rows.append(
                {
                    "dataset": dataset,
                    "beta": np.nan,
                    "variant": "multiround_curriculum",
                    "direction_usage_entropy": _format_mean_std(
                        float(df_h_curr["direction_usage_entropy"].mean()),
                        float(df_h_curr["direction_usage_entropy"].std()),
                    ),
                    "worst_dir_summary": " / ".join(df_h_curr["worst_dir_summary"].astype(str).tolist()),
                    "frozen_dir_count": _format_mean_std(
                        float(df_h_curr["frozen_dir_count"].mean()),
                        float(df_h_curr["frozen_dir_count"].std()),
                    ),
                    "expanded_dir_count": _format_mean_std(
                        float(df_h_curr["expanded_dir_count"].mean()),
                        float(df_h_curr["expanded_dir_count"].std()),
                    ),
                    "direction_health_comment": "; ".join([f"{k}:{v}" for k, v in sorted(note_counts.items())]),
                }
            )

    perf_summary_df = pd.DataFrame(perf_summary_rows).sort_values(["dataset", "beta"]).reset_index(drop=True)
    perf_summary_df.to_csv(out_root / "lraes_curriculum_performance_summary.csv", index=False)
    pd.DataFrame(health_summary_rows).sort_values(["dataset", "variant", "beta"]).to_csv(
        out_root / "lraes_direction_health_summary.csv",
        index=False,
    )
    pd.DataFrame(solver_summary_rows).sort_values(["dataset", "beta"]).to_csv(
        out_root / "lraes_solver_summary.csv",
        index=False,
    )

    available_betas = sorted(perf_summary_df["beta"].dropna().astype(float).unique().tolist()) if not perf_summary_df.empty else []
    main_beta = 0.5 if 0.5 in available_betas else (available_betas[0] if available_betas else None)
    if main_beta is None:
        main_df = perf_summary_df
    else:
        main_df = perf_summary_df[np.isclose(perf_summary_df["beta"].astype(float), float(main_beta))]
    improved_main = int(np.sum(main_df["delta_vs_multiround_curriculum"].astype(float) > 1e-6)) if not main_df.empty else 0
    if improved_main >= 2:
        readout = "值得进入下一阶段"
    elif improved_main >= 1:
        readout = "继续作为探索线"
    else:
        readout = "当前方案暂缓"

    fully_risk_sets: List[str] = []
    if not solver_seed_df.empty:
        fully_risk_sets = sorted(
            set(
                solver_seed_df.loc[
                    solver_seed_df["solver_state"].astype(str) == "fully_risk_dominated",
                    "dataset",
                ].astype(str).tolist()
            )
        )

    best_dataset = "current evidence insufficient"
    if not main_df.empty:
        best_idx = int(main_df["delta_vs_multiround_curriculum"].astype(float).idxmax())
        best_dataset = str(main_df.loc[best_idx, "dataset"])

    beta_cmp_lines: List[str] = []
    for dataset in sorted(set(perf_summary_df["dataset"].tolist())):
        ds_rows = perf_summary_df[perf_summary_df["dataset"] == dataset]
        by_beta = {float(row["beta"]): float(row["delta_vs_multiround_curriculum"]) for _, row in ds_rows.iterrows()}
        if 0.5 in by_beta and 1.0 in by_beta:
            beta_cmp_lines.append(
            f"- `{dataset}`: beta=0.5 delta `{by_beta[0.5]:+.4f}`, beta=1.0 delta `{by_beta[1.0]:+.4f}`"
            )

    bridge_hint = best_dataset if best_dataset != "current evidence insufficient" else "natops"
    snapshot_title = (
        f"## Main Snapshot (beta = {main_beta:.1f})"
        if main_beta is not None
        else "## Main Snapshot"
    )
    missing_main_beta_note = (
        "- 当前输出目录未保留完整的 `beta=0.5` 分数据集原始落盘；本次重建以当前可恢复 beta 为准。"
        if main_beta is not None and abs(float(main_beta) - 0.5) > 1e-9
        else None
    )
    conclusion_lines = [
        "# LRAES + Curriculum Conclusion",
        "",
        "更新时间：2026-03-22",
        "",
        "身份：`independent target-generator upgrade line`",
        "",
        "- `not for Phase15 mainline freeze table`",
        "- `not connected back to bridge in this round`",
        "- `goal = test whether local risk-aware eigen-directions improve target quality over pure multiround curriculum`",
        "",
        "## Core Datasets",
        "",
        "- `natops`",
        "- `selfregulationscp1`",
        "- `fingermovements`",
        "",
        snapshot_title,
        "",
    ]
    if missing_main_beta_note:
        conclusion_lines.extend([missing_main_beta_note, ""])
    for _, row in main_df.iterrows():
        conclusion_lines.append(
            f"- `{row['dataset']}`: curriculum={row['multiround_curriculum_f1']}, "
            f"lraes={row['lraes_curriculum_f1']}, "
            f"delta_vs_curriculum={float(row['delta_vs_multiround_curriculum']):+.4f}, "
            f"label=`{row['result_label']}`"
        )
    conclusion_lines.extend(
        [
            "",
            "## Beta Compare (0.5 main, 1.0 control)",
            "",
            *(beta_cmp_lines if beta_cmp_lines else ["- current evidence insufficient"]),
            "",
            "## Readout",
            "",
            f"- 当前是否应继续推进 LRAES + curriculum：`{readout}`",
            f"- 它是否已经超过纯 curriculum：`beta={main_beta:.1f} improved={improved_main}/3`" if main_beta is not None else f"- 它是否已经超过纯 curriculum：`current evidence insufficient`",
            "- 它的收益更集中在哪类数据集：`见 NATOPS / SCP1 / FingerMovements 分层结果`",
            f"- 当前是否观察到 fully_risk_dominated：`{'yes: ' + ', '.join(fully_risk_sets) if fully_risk_sets else 'no clear case in current core batch'}`",
            f"- 若后续重回 bridge，优先 very small pilot 数据集：`{bridge_hint}`",
        ]
    )
    (out_root / "lraes_curriculum_conclusion.md").write_text("\n".join(conclusion_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, required=True)
    args = parser.parse_args()
    rebuild(Path(args.out_root))


if __name__ == "__main__":
    main()
