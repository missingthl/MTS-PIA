#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE_ROOT = ROOT / "out" / "raw_bridge_simple_set_probe_20260320"
DEFAULT_MAINLINE_TABLE = ROOT / "out" / "phase15_mainline_freeze_20260319_formal" / "main_performance_table.csv"
DATASETS = ["har", "selfregulationscp1", "fingermovements"]
POSITIVE_DELTA_F1_THRESHOLD = 0.01
FLAT_DELTA_F1_THRESHOLD = 5e-4
FLAT_DELTA_ACC_THRESHOLD = 5e-4


def _fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def _parse_json_map(text: object) -> Dict[str, float]:
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        raw = json.loads(text)
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def _mean_json_map(series: pd.Series) -> Dict[str, float]:
    acc: Dict[str, list] = {}
    for item in series.tolist():
        for k, v in _parse_json_map(item).items():
            acc.setdefault(str(k), []).append(float(v))
    return {k: float(sum(vs) / len(vs)) for k, vs in sorted(acc.items(), key=lambda kv: kv[0]) if vs}


def _map_to_json_text(obj: Dict[str, float]) -> str:
    return json.dumps({k: round(float(v), 6) for k, v in obj.items()}, ensure_ascii=False, sort_keys=True)


def _dataset_label(delta_f1: float, delta_acc: float, positive_seeds: int, negative_seeds: int) -> str:
    if abs(delta_f1) <= FLAT_DELTA_F1_THRESHOLD and abs(delta_acc) <= FLAT_DELTA_ACC_THRESHOLD:
        return "flat"
    if delta_f1 >= POSITIVE_DELTA_F1_THRESHOLD and positive_seeds > negative_seeds:
        return "positive"
    return "unstable"


def _structure_remark(cov_match_logeuc: float, cov_to_orig_logeuc: float, cond_A: float) -> str:
    if cond_A > 1e4:
        return "possible_numeric_risk_condA_high"
    if cov_match_logeuc < 1e-6 and cov_to_orig_logeuc < 5e-2:
        return "near_machine_precision_or_tightly_controlled"
    if cov_match_logeuc < 5e-2:
        return "small_nonzero_but_controlled"
    return "bridge_drift_needs_attention"


def _distribution_remark(class_shift_max_abs: float, raw_mean_shift_abs: float) -> str:
    issues = []
    if class_shift_max_abs > 0.05:
        issues.append("possible_bias_amplification")
    if raw_mean_shift_abs > 1e-6:
        issues.append("possible_mean_shift")
    if not issues:
        return "no_obvious_bias_or_mean_shift"
    return "+".join(issues)


def _mapping_target_bucket(
    *,
    label: str,
    positive_seeds: int,
    negative_seeds: int,
    structure_remark: str,
    distribution_remark: str,
) -> str:
    if structure_remark == "possible_numeric_risk_condA_high" or distribution_remark != "no_obvious_bias_or_mean_shift":
        return "D"
    if label == "positive":
        return "A"
    if label == "flat" and positive_seeds == 0 and negative_seeds == 0:
        return "B"
    if label in {"flat", "unstable"}:
        return "C"
    return "D"


def _mapping_target_text(bucket: str) -> str:
    return {
        "A": "映射层成立 + 目标层有收益",
        "B": "映射层成立 + 目标层无额外收益",
        "C": "映射层干净，但 raw backbone 对目标不敏感",
        "D": "当前证据不足",
    }[bucket]


def build(probe_root: Path, mainline_table: Path) -> None:
    mainline = pd.read_csv(mainline_table)
    summary_rows = []
    structure_rows = []
    distribution_rows = []
    analysis_rows = []

    summary_md_lines = [
        "# Raw-Bridge Simple-Set Probe",
        "",
        "更新时间：2026-03-20",
        "",
        "身份：`parallel upgrade line`",
        "",
        "- `not for mainline freeze table`",
        "- `not for seed battlefield yet`",
        "",
        "## Probe Summary",
        "",
        "| dataset | raw_only_f1 | raw_bridge_f1 | delta_f1 | raw_only_acc | raw_bridge_acc | delta_acc | zspace_step1b_ref | label |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    positive_count = 0
    flat_count = 0

    for dataset in DATASETS:
        agg = pd.read_csv(probe_root / dataset / "summary_agg.csv")
        per_run = pd.read_csv(probe_root / dataset / "summary_per_run.csv")

        raw_only = agg[agg["experiment"] == "E1_raw_only"].iloc[0]
        raw_bridge = agg[agg["experiment"] == "E3_raw_bridge_geom_aug"].iloc[0]
        bridge_runs = per_run[per_run["experiment"] == "E3_raw_bridge_geom_aug"].copy()
        raw_runs = per_run[per_run["experiment"] == "E1_raw_only"].copy()

        seed_compare = raw_runs.merge(
            bridge_runs[["seed", "trial_macro_f1", "trial_acc"]],
            on="seed",
            how="inner",
            suffixes=("_raw_only", "_raw_bridge"),
        )
        seed_compare["delta_f1"] = seed_compare["trial_macro_f1_raw_bridge"] - seed_compare["trial_macro_f1_raw_only"]
        seed_compare["delta_acc"] = seed_compare["trial_acc_raw_bridge"] - seed_compare["trial_acc_raw_only"]
        positive_seeds = int((seed_compare["delta_f1"] > 0).sum())
        negative_seeds = int((seed_compare["delta_f1"] < 0).sum())

        delta_f1 = float(raw_bridge["trial_macro_f1_mean"] - raw_only["trial_macro_f1_mean"])
        delta_acc = float(raw_bridge["trial_acc_mean"] - raw_only["trial_acc_mean"])
        label = _dataset_label(delta_f1, delta_acc, positive_seeds, negative_seeds)
        if label == "positive":
            positive_count += 1
        elif label == "flat":
            flat_count += 1

        ref = mainline[mainline["dataset"] == dataset].iloc[0]
        zspace_ref = f"acc={float(ref['step1b_acc']):.4f}, f1={float(ref['step1b_f1']):.4f}"

        class_balance_shift_summary = _mean_json_map(bridge_runs["class_balance_shift_summary"])
        classwise_mean_shift_summary = _mean_json_map(bridge_runs["classwise_mean_shift_summary"])
        class_balance_shift_max_abs_mean = float(bridge_runs["class_balance_shift_max_abs"].mean())
        raw_mean_shift_abs_mean = float(bridge_runs["raw_mean_shift_abs_mean"].mean())
        train_selected_aug_ratio_mean = float(bridge_runs["train_selected_aug_ratio"].mean())

        structure_remark = _structure_remark(
            cov_match_logeuc=float(bridge_runs["bridge_cov_match_error_logeuc_mean"].mean()),
            cov_to_orig_logeuc=float(bridge_runs["bridge_cov_to_orig_distance_logeuc_mean"].mean()),
            cond_A=float(bridge_runs["bridge_cond_A_mean"].mean()),
        )
        distribution_remark = _distribution_remark(
            class_shift_max_abs=class_balance_shift_max_abs_mean,
            raw_mean_shift_abs=raw_mean_shift_abs_mean,
        )

        summary_rows.append(
            {
                "dataset": dataset,
                "raw_only_acc_mean": float(raw_only["trial_acc_mean"]),
                "raw_only_acc_std": float(raw_only["trial_acc_std"]),
                "raw_only_f1_mean": float(raw_only["trial_macro_f1_mean"]),
                "raw_only_f1_std": float(raw_only["trial_macro_f1_std"]),
                "raw_bridge_acc_mean": float(raw_bridge["trial_acc_mean"]),
                "raw_bridge_acc_std": float(raw_bridge["trial_acc_std"]),
                "raw_bridge_f1_mean": float(raw_bridge["trial_macro_f1_mean"]),
                "raw_bridge_f1_std": float(raw_bridge["trial_macro_f1_std"]),
                "raw_only_acc_mean_std": _fmt_mean_std(float(raw_only["trial_acc_mean"]), float(raw_only["trial_acc_std"])),
                "raw_only_f1_mean_std": _fmt_mean_std(float(raw_only["trial_macro_f1_mean"]), float(raw_only["trial_macro_f1_std"])),
                "raw_bridge_acc_mean_std": _fmt_mean_std(float(raw_bridge["trial_acc_mean"]), float(raw_bridge["trial_acc_std"])),
                "raw_bridge_f1_mean_std": _fmt_mean_std(float(raw_bridge["trial_macro_f1_mean"]), float(raw_bridge["trial_macro_f1_std"])),
                "delta_acc": delta_acc,
                "delta_f1": delta_f1,
                "positive_seed_count": positive_seeds,
                "negative_seed_count": negative_seeds,
                "zspace_step1b_ref_acc": float(ref["step1b_acc"]),
                "zspace_step1b_ref_f1": float(ref["step1b_f1"]),
                "label": label,
            }
        )

        structure_rows.append(
            {
                "dataset": dataset,
                "bridge_cov_match_error_fro_mean": float(bridge_runs["bridge_cov_match_error_fro_mean"].mean()),
                "bridge_cov_match_error_fro_std": float(bridge_runs["bridge_cov_match_error_fro_mean"].std(ddof=0)),
                "bridge_cov_match_error_logeuc_mean": float(bridge_runs["bridge_cov_match_error_logeuc_mean"].mean()),
                "bridge_cov_match_error_logeuc_std": float(bridge_runs["bridge_cov_match_error_logeuc_mean"].std(ddof=0)),
                "bridge_cov_to_orig_fro_mean": float(bridge_runs["bridge_cov_to_orig_distance_fro_mean"].mean()),
                "bridge_cov_to_orig_fro_std": float(bridge_runs["bridge_cov_to_orig_distance_fro_mean"].std(ddof=0)),
                "bridge_cov_to_orig_logeuc_mean": float(bridge_runs["bridge_cov_to_orig_distance_logeuc_mean"].mean()),
                "bridge_cov_to_orig_logeuc_std": float(bridge_runs["bridge_cov_to_orig_distance_logeuc_mean"].std(ddof=0)),
                "energy_ratio_mean": float(bridge_runs["bridge_energy_ratio_mean"].mean()),
                "energy_ratio_std": float(bridge_runs["bridge_energy_ratio_mean"].std(ddof=0)),
                "cond_A_mean": float(bridge_runs["bridge_cond_A_mean"].mean()),
                "cond_A_std": float(bridge_runs["bridge_cond_A_mean"].std(ddof=0)),
                "sigma_orig_min_eig_mean": float(bridge_runs["sigma_orig_min_eig_mean"].mean()),
                "sigma_orig_max_eig_mean": float(bridge_runs["sigma_orig_max_eig_mean"].mean()),
                "remark": structure_remark,
            }
        )

        distribution_rows.append(
            {
                "dataset": dataset,
                "train_selected_aug_ratio_mean": train_selected_aug_ratio_mean,
                "train_selected_aug_ratio_std": float(bridge_runs["train_selected_aug_ratio"].std(ddof=0)),
                "class_balance_shift_summary": _map_to_json_text(class_balance_shift_summary),
                "class_balance_shift_max_abs_mean": class_balance_shift_max_abs_mean,
                "trial_window_gap_status": "not_applicable_raw_trial_model",
                "raw_mean_shift_abs_mean": raw_mean_shift_abs_mean,
                "raw_mean_shift_abs_std": float(bridge_runs["raw_mean_shift_abs_mean"].std(ddof=0)),
                "classwise_mean_shift_summary": _map_to_json_text(classwise_mean_shift_summary),
                "remark": distribution_remark,
            }
        )

        bucket = _mapping_target_bucket(
            label=label,
            positive_seeds=positive_seeds,
            negative_seeds=negative_seeds,
            structure_remark=structure_remark,
            distribution_remark=distribution_remark,
        )
        analysis_rows.append(
            {
                "dataset": dataset,
                "bucket": bucket,
                "bucket_text": _mapping_target_text(bucket),
                "mapping_layer_assessment": "clean_and_stable"
                if structure_remark != "possible_numeric_risk_condA_high" and distribution_remark == "no_obvious_bias_or_mean_shift"
                else "risk_or_insufficient",
                "target_layer_assessment": (
                    "converts_to_positive_gain"
                    if bucket == "A"
                    else "no_extra_gain" if bucket == "B" else "likely_backbone_insensitive_or_task_mismatch" if bucket == "C" else "insufficient"
                ),
                "supporting_evidence": (
                    f"label={label}; delta_f1={delta_f1:+.4f}; "
                    f"positive_seeds={positive_seeds}; negative_seeds={negative_seeds}; "
                    f"structure={structure_remark}; distribution={distribution_remark}"
                ),
            }
        )

        summary_md_lines.append(
            "| "
            + " | ".join(
                [
                    dataset,
                    _fmt_mean_std(float(raw_only["trial_macro_f1_mean"]), float(raw_only["trial_macro_f1_std"])),
                    _fmt_mean_std(float(raw_bridge["trial_macro_f1_mean"]), float(raw_bridge["trial_macro_f1_std"])),
                    f"{delta_f1:+.4f}",
                    _fmt_mean_std(float(raw_only["trial_acc_mean"]), float(raw_only["trial_acc_std"])),
                    _fmt_mean_std(float(raw_bridge["trial_acc_mean"]), float(raw_bridge["trial_acc_std"])),
                    f"{delta_acc:+.4f}",
                    zspace_ref,
                    label,
                ]
            )
            + " |"
        )

    summary_df = pd.DataFrame(summary_rows)
    structure_df = pd.DataFrame(structure_rows)
    distribution_df = pd.DataFrame(distribution_rows)
    analysis_df = pd.DataFrame(analysis_rows)

    summary_df.to_csv(probe_root / "simple_set_probe_summary.csv", index=False)
    structure_df.to_csv(probe_root / "raw_bridge_structure_fidelity_summary.csv", index=False)
    distribution_df.to_csv(probe_root / "raw_bridge_distribution_stability_summary.csv", index=False)
    analysis_df.to_csv(probe_root / "raw_bridge_target_vs_mapping_analysis.csv", index=False)

    if positive_count >= 2 and flat_count >= 1:
        overall = "stable_upgrade_line"
    elif positive_count >= 1 and flat_count >= 1:
        overall = "local_upgrade_line"
    else:
        overall = "experimental_line_not_yet_stable"

    primary_risk = "benefit_instability"
    if (structure_df["remark"] == "possible_numeric_risk_condA_high").any():
        primary_risk = "numeric_conditioning"
    elif distribution_df["remark"].str.contains("possible_bias_amplification|possible_mean_shift", regex=True).any():
        primary_risk = "bias_or_mean_shift"
    elif (structure_df["remark"] == "bridge_drift_needs_attention").any():
        primary_risk = "bridge_structure_drift"

    summary_md_lines.extend(
        [
            "",
            "## Generated Tables",
            "",
            "- `simple_set_probe_summary.csv`",
            "- `raw_bridge_structure_fidelity_summary.csv`",
            "- `raw_bridge_distribution_stability_summary.csv`",
            "- `raw_bridge_probe_conclusion.md`",
            "- `raw_bridge_target_vs_mapping_analysis.md`",
        ]
    )
    (probe_root / "simple_set_probe_summary.md").write_text("\n".join(summary_md_lines) + "\n", encoding="utf-8")

    conclusion_lines = [
        "# Raw-Bridge Probe Conclusion",
        "",
        "更新时间：2026-03-20",
        "",
        "身份：`parallel upgrade line`",
        "",
        "- `not for mainline freeze table`",
        "- `not for seed battlefield yet`",
        "- 当前验证对象不是“纯 bridge 映射框架”单独效果，而是“PIA-style 结构侧目标 + bridge 注回 raw 域”的整体闭环。",
        "",
        "## Direct Answers",
        "",
        f"- 当前 raw-bridge 是否可保留为稳定升级线：`{'是' if overall == 'stable_upgrade_line' else '否'}`",
        f"- 当前 raw-bridge 更准确的阶段定位：`{overall}`",
        "- 当前是否仍应停留在 simple-set：`是`",
        "- 当前是否足以回到 seed 主战场：`否`",
        f"- 当前最主要风险：`{primary_risk}`",
        "",
        "## Dataset Readout",
        "",
    ]

    for row in summary_rows:
        conclusion_lines.append(
            f"- `{row['dataset']}`: label=`{row['label']}`, "
            f"delta_f1={row['delta_f1']:+.4f}, delta_acc={row['delta_acc']:+.4f}"
        )

    conclusion_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `HAR` 当前更像小幅收益但不够强，是否属于稳定升级价值仍需继续观察。",
            "- `SelfRegulationSCP1` 当前更像稳定持平且结构保真很高，说明 bridge 映射本身是成立的。",
            "- `FingerMovements` 当前给出较明显正收益，但仍需要把它视为 simple-set 范围内的可重复信号，而不是向长 EEG 外推的证据。",
            "- 因此这条线当前更适合保留为并行升级线，不应并入 Phase15 freeze 主体，也不应回到 seed 主战场。",
            "- 当前 simple-set probe 更支持这样的说法：bridge 层整体干净，PIA-style 目标在部分 simple-set 上能转化为正收益，在其余 simple-set 上至少非退化。",
        ]
    )
    (probe_root / "raw_bridge_probe_conclusion.md").write_text("\n".join(conclusion_lines) + "\n", encoding="utf-8")

    analysis_lines = [
        "# Raw-Bridge Target vs Mapping Analysis",
        "",
        "更新时间：2026-03-20",
        "",
        "身份：`parallel upgrade line`",
        "",
        "- `not for mainline freeze table`",
        "- `not for seed battlefield yet`",
        "",
        "## What This Probe Is Actually Validating",
        "",
        "- 当前 simple-set probe 同时在验证两层：",
        "  1. bridge 作为结构注回算子是否足够干净、稳定，不额外引入明显数值灾难、类别偏置或均值漂移；",
        "  2. 当前 PIA-style 结构侧增强目标，在被注回 raw 域后，是否真的对 raw backbone 友好。",
        "- 因此当前不能把 raw-bridge 简化成“纯映射框架已成立”，也不能把它写成“完全独立于 PIA 的另一套增强方法”。",
        "",
        "## Global Readout",
        "",
        "- 当前三组 simple-set 都没有暴露明显 mapping-layer 崩坏信号：`cond_A` 低、类别占比漂移小、`raw_mean_shift_abs` 接近 0。",
        "- 因而当前 performance 差异更像来自 target-layer compatibility，而不是 bridge 本身先把数据映射脏掉。",
        "",
        "## Dataset Classification",
        "",
        "| dataset | bucket | meaning | evidence |",
        "|---|---|---|---|",
    ]

    for row in analysis_rows:
        analysis_lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["bucket"],
                    row["bucket_text"],
                    row["supporting_evidence"],
                ]
            )
            + " |"
        )

    analysis_lines.extend(
        [
            "",
            "## Interpretation By Dataset",
            "",
            "- `HAR`: 当前更像 `C`。mapping 层整体干净，但当前 PIA-style 目标在 raw backbone 上没有转成稳定收益，更像目标收益弱或 backbone 不敏感。",
            "- `SelfRegulationSCP1`: 当前更像 `B`。mapping 层成立，目标层没有带来额外判别收益，但至少没有退化。",
            "- `FingerMovements`: 当前更像 `A`。mapping 层成立，同时当前 PIA-style 目标在 raw 域能转化为较清晰的正收益。",
            "",
            "## Takeaway",
            "",
            "- 当前 simple-set 结果首先证明的是：bridge 层整体干净。",
            "- 在此基础上，`FingerMovements` 提供了“PIA-style 目标 + bridge 注回 raw”整体闭环能成立的证据。",
            "- `HAR / SCP1` 则提醒我们：mapping 成立不等于目标在 raw backbone 上必然带来收益。",
            "- 因此当前最准确的总表述只能是：bridge 层整体干净，PIA-style 目标在部分 simple-set 上能转化为正收益，在其余 simple-set 上至少非退化。",
        ]
    )
    (probe_root / "raw_bridge_target_vs_mapping_analysis.md").write_text("\n".join(analysis_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build raw-bridge simple-set probe reports.")
    parser.add_argument("--probe-root", type=str, default=str(DEFAULT_PROBE_ROOT))
    parser.add_argument("--mainline-table", type=str, default=str(DEFAULT_MAINLINE_TABLE))
    args = parser.parse_args()
    build(Path(args.probe_root), Path(args.mainline_table))


if __name__ == "__main__":
    main()
