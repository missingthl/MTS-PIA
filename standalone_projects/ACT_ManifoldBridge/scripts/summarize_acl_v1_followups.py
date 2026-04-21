#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from summarize_acl_small_matrix import (
    _as_bool,
    _as_float,
    _as_int,
    _mean,
    _sorted_fieldnames,
    _spearman,
    _std,
    _write_csv,
)


FAILURE_REVIEW_DATASETS = ["handwriting", "atrialfibrillation", "motorimagery"]
STABILITY_CONFIRM_DATASETS = ["natops", "japanesevowels", "heartbeat"]

MBA_ARM = "mba"
ACL_N4_ARM = "gcg_acl_n4"

FROZEN_MAIN_VARIANT = "gcg_acl_frozen_main"
FAILURE_REFERENCE_VARIANTS = ["mba_ref", "continue_ce_ref", FROZEN_MAIN_VARIANT]
FAILURE_ACTUAL_VARIANTS = [
    "gcg_acl_align1p0",
    "gcg_acl_loss0p1",
    "gcg_acl_temp0p05",
    "gcg_acl_temp0p10",
]
FAILURE_ALL_VARIANTS = FAILURE_REFERENCE_VARIANTS + FAILURE_ACTUAL_VARIANTS
FAILURE_REVIEW_TABLE_VARIANTS = [FROZEN_MAIN_VARIANT] + FAILURE_ACTUAL_VARIANTS

STABILITY_LOGICAL_ARMS = ["mba", "continue_ce", "gcg_acl"]
DOMINANCE_COMPONENTS = ["alignment_norm", "entropy_norm", "safe_radius_ratio", "fidelity_score"]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _collect_actual_rows(root: Path, arm_dirs: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for arm_dir in arm_dirs:
        arm_root = root / arm_dir
        if not arm_root.is_dir():
            continue
        for dataset_dir in sorted(p for p in arm_root.iterdir() if p.is_dir()):
            final_csv = dataset_dir / "final_results.csv"
            if not final_csv.is_file():
                continue
            for row in _read_csv(final_csv):
                entry: Dict[str, object] = dict(row)
                entry["arm_dir"] = arm_dir
                entry["dataset_dir"] = str(dataset_dir)
                rows.append(entry)
    return rows


def _candidate_audit_paths(root: Path, arm_dir: str, dataset: str, seed: int) -> Tuple[Path, Path]:
    audit_dir = root / arm_dir / dataset / "audit"
    return (
        audit_dir / f"{dataset}_s{seed}_candidate_scores.csv",
        audit_dir / f"{dataset}_s{seed}_selected_positives.csv",
    )


def _summarize_acl_audit(root: Path, arm_dir: str, dataset: str, seed: int) -> Dict[str, object]:
    candidate_path, selected_path = _candidate_audit_paths(root, arm_dir, dataset, seed)
    summary: Dict[str, object] = {
        "candidate_csv": str(candidate_path),
        "selected_csv": str(selected_path),
        "audit_files_present": candidate_path.is_file() and selected_path.is_file(),
        "valid_candidate_count": 0,
        "selected_candidate_count": 0,
        "selected_class_coverage_frac": None,
        "selected_class_missing_frac": None,
        "coverage_risk": None,
        "hard_positive_score": None,
        "alignment_cosine": None,
        "entropy_shift": None,
        "fidelity_score": None,
        "safe_radius_ratio": None,
        "alignment_norm_mean": None,
        "entropy_norm_mean": None,
        "safe_radius_ratio_mean": None,
        "fidelity_score_mean_selected": None,
        "dominant_metric": "",
        "dominant_corr": None,
        "dominance_gap": None,
        "dominance_risk": None,
    }
    if not summary["audit_files_present"]:
        return summary

    candidate_rows = _read_csv(candidate_path)
    selected_rows = _read_csv(selected_path)
    valid_rows = [row for row in candidate_rows if _as_bool(row.get("is_valid_candidate", ""))]
    summary["valid_candidate_count"] = len(valid_rows)
    summary["selected_candidate_count"] = len(selected_rows)

    all_classes = sorted({_as_int(row.get("y")) for row in candidate_rows if _as_int(row.get("y")) is not None})
    selected_classes = sorted({_as_int(row.get("y")) for row in selected_rows if _as_int(row.get("y")) is not None})
    if all_classes:
        coverage_frac = len(selected_classes) / float(len(all_classes))
        missing_frac = max(0.0, 1.0 - coverage_frac)
        summary["selected_class_coverage_frac"] = coverage_frac
        summary["selected_class_missing_frac"] = missing_frac
        summary["coverage_risk"] = missing_frac > 0.25

    summary["hard_positive_score"] = _mean(_as_float(row.get("hard_positive_score")) for row in selected_rows)
    summary["alignment_cosine"] = _mean(_as_float(row.get("alignment_cosine")) for row in selected_rows)
    summary["entropy_shift"] = _mean(_as_float(row.get("entropy_shift")) for row in selected_rows)
    summary["fidelity_score"] = _mean(_as_float(row.get("fidelity_score")) for row in selected_rows)
    summary["safe_radius_ratio"] = _mean(_as_float(row.get("safe_radius_ratio")) for row in selected_rows)

    summary["alignment_norm_mean"] = _mean(_as_float(row.get("alignment_norm")) for row in valid_rows)
    summary["entropy_norm_mean"] = _mean(_as_float(row.get("entropy_norm")) for row in valid_rows)
    summary["safe_radius_ratio_mean"] = _mean(_as_float(row.get("safe_radius_ratio")) for row in valid_rows)
    summary["fidelity_score_mean_selected"] = _mean(_as_float(row.get("fidelity_score")) for row in selected_rows)

    hard_scores = [_as_float(row.get("hard_positive_score")) for row in valid_rows]
    hard_scores = [v for v in hard_scores if v is not None]
    if len(valid_rows) >= 3 and len(hard_scores) == len(valid_rows):
        corrs: List[Tuple[str, float]] = []
        for key in DOMINANCE_COMPONENTS:
            vals = [_as_float(row.get(key)) for row in valid_rows]
            if any(v is None for v in vals):
                continue
            corrs.append((key, abs(_spearman(hard_scores, [float(v) for v in vals if v is not None]))))
        corrs.sort(key=lambda item: item[1], reverse=True)
        if corrs:
            summary["dominant_metric"] = corrs[0][0]
            summary["dominant_corr"] = corrs[0][1]
            gap = corrs[0][1] - corrs[1][1] if len(corrs) > 1 else corrs[0][1]
            summary["dominance_gap"] = gap
            summary["dominance_risk"] = corrs[0][1] >= 0.95 and gap >= 0.10
    return summary


def _build_reference_row(
    *,
    actual_row: Dict[str, object],
    variant: str,
    f1: Optional[float],
) -> Dict[str, object]:
    return {
        "dataset": str(actual_row["dataset"]),
        "seed": _as_int(actual_row["seed"]),
        "variant": variant,
        "variant_type": "reference",
        "status": str(actual_row.get("status", "")),
        "pipeline": actual_row.get("pipeline", ""),
        "arm_dir": actual_row.get("arm_dir", ""),
        "source_dataset_dir": actual_row.get("dataset_dir", ""),
        "is_reference": True,
        "warmup_f1": _as_float(actual_row.get("warmup_f1")),
        "base_f1": _as_float(actual_row.get("base_f1")),
        "act_f1": _as_float(actual_row.get("act_f1")),
        "f1": f1,
        "selected_anchor_count": None,
        "selected_positive_count": None,
        "candidate_total_count": None,
        "hard_positive_score_mean": None,
        "fidelity_score_mean": None,
        "acl_last_ce_loss": None,
        "acl_last_supcon_loss": None,
        "audit_files_present": None,
        "candidate_csv": "",
        "selected_csv": "",
        "valid_candidate_count": None,
        "selected_candidate_count": None,
        "selected_class_coverage_frac": None,
        "selected_class_missing_frac": None,
        "coverage_risk": None,
        "hard_positive_score": None,
        "alignment_cosine": None,
        "entropy_shift": None,
        "fidelity_score": None,
        "safe_radius_ratio": None,
        "alignment_norm_mean": None,
        "entropy_norm_mean": None,
        "safe_radius_ratio_mean": None,
        "fidelity_score_mean_selected": None,
        "dominant_metric": "",
        "dominant_corr": None,
        "dominance_gap": None,
        "dominance_risk": None,
    }


def _build_acl_row(
    *,
    actual_row: Dict[str, object],
    variant: str,
    root: Path,
    arm_dir: str,
) -> Dict[str, object]:
    dataset = str(actual_row["dataset"])
    seed = int(_as_int(actual_row["seed"]) or 0)
    audit = _summarize_acl_audit(root, arm_dir, dataset, seed)
    return {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "variant_type": "acl",
        "status": str(actual_row.get("status", "")),
        "pipeline": actual_row.get("pipeline", ""),
        "arm_dir": arm_dir,
        "source_dataset_dir": actual_row.get("dataset_dir", ""),
        "is_reference": False,
        "warmup_f1": _as_float(actual_row.get("warmup_f1")),
        "base_f1": _as_float(actual_row.get("base_f1")),
        "act_f1": _as_float(actual_row.get("act_f1")),
        "f1": _as_float(actual_row.get("act_f1")),
        "selected_anchor_count": _as_int(actual_row.get("selected_anchor_count")),
        "selected_positive_count": _as_int(actual_row.get("selected_positive_count")),
        "candidate_total_count": _as_int(actual_row.get("candidate_total_count")),
        "hard_positive_score_mean": _as_float(actual_row.get("hard_positive_score_mean")),
        "fidelity_score_mean": _as_float(actual_row.get("fidelity_score_mean")),
        "acl_last_ce_loss": _as_float(actual_row.get("acl_last_ce_loss")),
        "acl_last_supcon_loss": _as_float(actual_row.get("acl_last_supcon_loss")),
        **audit,
    }


def _failure_variant_rows(frozen_root: Path, review_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    frozen_actual = _collect_actual_rows(frozen_root, [MBA_ARM, ACL_N4_ARM])
    for row in frozen_actual:
        dataset = str(row.get("dataset", ""))
        if dataset not in FAILURE_REVIEW_DATASETS:
            continue
        arm_dir = str(row.get("arm_dir", ""))
        if arm_dir == MBA_ARM:
            rows.append(_build_reference_row(actual_row=row, variant="mba_ref", f1=_as_float(row.get("act_f1"))))
        elif arm_dir == ACL_N4_ARM:
            rows.append(_build_reference_row(actual_row=row, variant="continue_ce_ref", f1=_as_float(row.get("base_f1"))))
            rows.append(_build_acl_row(actual_row=row, variant=FROZEN_MAIN_VARIANT, root=frozen_root, arm_dir=ACL_N4_ARM))

    review_actual = _collect_actual_rows(review_root, FAILURE_ACTUAL_VARIANTS)
    for row in review_actual:
        dataset = str(row.get("dataset", ""))
        if dataset not in FAILURE_REVIEW_DATASETS:
            continue
        arm_dir = str(row.get("arm_dir", ""))
        rows.append(_build_acl_row(actual_row=row, variant=arm_dir, root=review_root, arm_dir=arm_dir))

    rows.sort(key=lambda r: (str(r["dataset"]), int(r["seed"]), FAILURE_ALL_VARIANTS.index(str(r["variant"]))))
    _attach_failure_deltas(rows)
    return rows


def _attach_failure_deltas(rows: List[Dict[str, object]]) -> None:
    baselines: Dict[Tuple[str, int], Dict[str, Optional[float]]] = {}
    for row in rows:
        key = (str(row["dataset"]), int(row["seed"]))
        slot = baselines.setdefault(key, {})
        if row["variant"] == "mba_ref":
            slot["mba_ref"] = _as_float(row.get("f1"))
        elif row["variant"] == "continue_ce_ref":
            slot["continue_ce_ref"] = _as_float(row.get("f1"))
        elif row["variant"] == FROZEN_MAIN_VARIANT:
            slot[FROZEN_MAIN_VARIANT] = _as_float(row.get("f1"))

    for row in rows:
        key = (str(row["dataset"]), int(row["seed"]))
        slot = baselines.get(key, {})
        f1 = _as_float(row.get("f1"))
        frozen_main = slot.get(FROZEN_MAIN_VARIANT)
        continue_ce = slot.get("continue_ce_ref")
        mba = slot.get("mba_ref")
        row["delta_vs_frozen_main"] = None if f1 is None or frozen_main is None else f1 - frozen_main
        row["delta_vs_continue_ce"] = None if f1 is None or continue_ce is None else f1 - continue_ce
        row["delta_vs_mba"] = None if f1 is None or mba is None else f1 - mba


def _failure_dataset_variant_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["variant"]))].append(row)

    summary_rows: List[Dict[str, object]] = []
    for (dataset, variant), group_rows in sorted(grouped.items()):
        dominance_flags = [bool(row["dominance_risk"]) for row in group_rows if row.get("dominance_risk") is not None]
        dominant_metrics = [str(row["dominant_metric"]) for row in group_rows if row.get("dominant_metric")]
        summary_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "variant_type": group_rows[0].get("variant_type", ""),
                "n": len(group_rows),
                "mean_f1": _mean(_as_float(row.get("f1")) for row in group_rows),
                "std_f1": _std(_as_float(row.get("f1")) for row in group_rows),
                "mean_delta_vs_frozen_main": _mean(_as_float(row.get("delta_vs_frozen_main")) for row in group_rows),
                "mean_delta_vs_continue_ce": _mean(_as_float(row.get("delta_vs_continue_ce")) for row in group_rows),
                "mean_delta_vs_mba": _mean(_as_float(row.get("delta_vs_mba")) for row in group_rows),
                "mean_selected_anchor_count": _mean(_as_float(row.get("selected_anchor_count")) for row in group_rows),
                "mean_selected_positive_count": _mean(_as_float(row.get("selected_positive_count")) for row in group_rows),
                "mean_candidate_total_count": _mean(_as_float(row.get("candidate_total_count")) for row in group_rows),
                "mean_valid_candidate_count": _mean(_as_float(row.get("valid_candidate_count")) for row in group_rows),
                "mean_hard_positive_score_mean": _mean(_as_float(row.get("hard_positive_score_mean")) for row in group_rows),
                "mean_fidelity_score_mean": _mean(_as_float(row.get("fidelity_score_mean")) for row in group_rows),
                "mean_acl_last_ce_loss": _mean(_as_float(row.get("acl_last_ce_loss")) for row in group_rows),
                "mean_acl_last_supcon_loss": _mean(_as_float(row.get("acl_last_supcon_loss")) for row in group_rows),
                "mean_selected_class_coverage_frac": _mean(_as_float(row.get("selected_class_coverage_frac")) for row in group_rows),
                "mean_hard_positive_score": _mean(_as_float(row.get("hard_positive_score")) for row in group_rows),
                "mean_alignment_cosine": _mean(_as_float(row.get("alignment_cosine")) for row in group_rows),
                "mean_entropy_shift": _mean(_as_float(row.get("entropy_shift")) for row in group_rows),
                "mean_fidelity_score": _mean(_as_float(row.get("fidelity_score")) for row in group_rows),
                "mean_safe_radius_ratio": _mean(_as_float(row.get("safe_radius_ratio")) for row in group_rows),
                "mean_alignment_norm_mean": _mean(_as_float(row.get("alignment_norm_mean")) for row in group_rows),
                "mean_entropy_norm_mean": _mean(_as_float(row.get("entropy_norm_mean")) for row in group_rows),
                "mean_safe_radius_ratio_mean": _mean(_as_float(row.get("safe_radius_ratio_mean")) for row in group_rows),
                "mean_fidelity_score_mean_selected": _mean(_as_float(row.get("fidelity_score_mean_selected")) for row in group_rows),
                "dominant_metric_majority": Counter(dominant_metrics).most_common(1)[0][0] if dominant_metrics else "",
                "dominance_risk_majority": len(dominance_flags) > 0 and sum(dominance_flags) > len(dominance_flags) / 2.0,
            }
        )
    return summary_rows


def _failure_review_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    review_rows = [row for row in rows if row["variant"] in FAILURE_REVIEW_TABLE_VARIANTS]
    review_rows.sort(key=lambda r: (str(r["dataset"]), int(r["seed"]), FAILURE_REVIEW_TABLE_VARIANTS.index(str(r["variant"]))))
    return review_rows


def _classify_failure_dataset(dataset: str, summary_rows: List[Dict[str, object]]) -> Tuple[str, str]:
    variant_rows = [row for row in summary_rows if row["dataset"] == dataset and row["variant"] in FAILURE_REVIEW_TABLE_VARIANTS]
    if not variant_rows:
        return "acl_no_clear_advantage_yet", "no follow-up rows"

    candidate_pool_flags = []
    scorer_bias_flags = []
    all_below_continue = True
    best_variant = None
    best_delta = None
    for row in variant_rows:
        mean_selected = _as_float(row.get("mean_selected_positive_count")) or 0.0
        mean_valid = _as_float(row.get("mean_valid_candidate_count")) or 0.0
        mean_total = _as_float(row.get("mean_candidate_total_count")) or 0.0
        mean_coverage = _as_float(row.get("mean_selected_class_coverage_frac"))
        valid_fraction = 0.0 if mean_total <= 0 else mean_valid / mean_total
        candidate_pool_flags.append(
            mean_selected <= 0.0
            or mean_valid <= 0.0
            or valid_fraction < 0.10
            or (mean_coverage is not None and mean_coverage < 0.75)
        )
        scorer_bias_flags.append(bool(row.get("dominance_risk_majority")))
        delta_continue = _as_float(row.get("mean_delta_vs_continue_ce"))
        if delta_continue is not None and delta_continue >= 0.0:
            all_below_continue = False
        if delta_continue is not None and (best_delta is None or delta_continue > best_delta):
            best_delta = delta_continue
            best_variant = str(row["variant"])

    candidate_pool_issue = sum(candidate_pool_flags) > len(candidate_pool_flags) / 2.0
    scorer_bias_issue = sum(scorer_bias_flags) > len(scorer_bias_flags) / 2.0
    if candidate_pool_issue:
        return "candidate_pool_issue", f"majority variants show low selected/valid coverage; best_variant={best_variant or 'n/a'}"
    if scorer_bias_issue:
        return "scorer_bias_issue", f"majority variants show dominance-risk scoring; best_variant={best_variant or 'n/a'}"
    if all_below_continue:
        return "objective_mismatch_issue", f"all ACL variants remain below continue_ce; best_variant={best_variant or 'n/a'}"
    return "acl_no_clear_advantage_yet", f"at least one follow-up variant recovers against continue_ce; best_variant={best_variant or 'n/a'}"


def _write_failure_review_note(
    summary_dir: Path,
    frozen_root: Path,
    review_root: Path,
    summary_rows: List[Dict[str, object]],
) -> None:
    lines = [
        "# ACL v1 Failure Review",
        "",
        f"- frozen_root: `{frozen_root}`",
        f"- review_root: `{review_root}`",
        "",
    ]
    for dataset in FAILURE_REVIEW_DATASETS:
        label, rationale = _classify_failure_dataset(dataset, summary_rows)
        dataset_rows = [row for row in summary_rows if row["dataset"] == dataset]
        best_row = None
        for row in dataset_rows:
            delta = _as_float(row.get("mean_delta_vs_continue_ce"))
            if delta is None:
                continue
            if best_row is None or delta > (_as_float(best_row.get("mean_delta_vs_continue_ce")) or float("-inf")):
                best_row = row
        lines.extend(
            [
                f"## {dataset}",
                f"- label: `{label}`",
                f"- rationale: {rationale}",
            ]
        )
        if best_row is not None:
            lines.extend(
                [
                    f"- best_variant: `{best_row['variant']}`",
                    f"- best_variant_mean_f1: `{best_row.get('mean_f1')}`",
                    f"- best_variant_delta_vs_continue_ce: `{best_row.get('mean_delta_vs_continue_ce')}`",
                ]
            )
        lines.append("")
    (summary_dir / "review_note.md").write_text("\n".join(lines), encoding="utf-8")


def _failure_review_summary(root: Path, frozen_root: Path) -> int:
    variant_rows = _failure_variant_rows(frozen_root, root)
    dataset_variant_rows = _failure_dataset_variant_summary(variant_rows)
    review_rows = _failure_review_rows(variant_rows)

    summary_dir = root / "_summary"
    variant_pref = [
        "dataset",
        "seed",
        "variant",
        "variant_type",
        "f1",
        "delta_vs_frozen_main",
        "delta_vs_continue_ce",
        "delta_vs_mba",
        "status",
        "warmup_f1",
        "base_f1",
        "act_f1",
        "selected_anchor_count",
        "selected_positive_count",
        "candidate_total_count",
        "valid_candidate_count",
        "hard_positive_score_mean",
        "fidelity_score_mean",
        "acl_last_ce_loss",
        "acl_last_supcon_loss",
        "selected_class_coverage_frac",
        "hard_positive_score",
        "alignment_cosine",
        "entropy_shift",
        "fidelity_score",
        "safe_radius_ratio",
        "alignment_norm_mean",
        "entropy_norm_mean",
        "safe_radius_ratio_mean",
        "fidelity_score_mean_selected",
        "dominant_metric",
        "dominant_corr",
        "dominance_gap",
        "dominance_risk",
        "source_dataset_dir",
        "candidate_csv",
        "selected_csv",
    ]
    summary_pref = [
        "dataset",
        "variant",
        "variant_type",
        "n",
        "mean_f1",
        "std_f1",
        "mean_delta_vs_frozen_main",
        "mean_delta_vs_continue_ce",
        "mean_delta_vs_mba",
        "mean_selected_anchor_count",
        "mean_selected_positive_count",
        "mean_candidate_total_count",
        "mean_valid_candidate_count",
        "mean_selected_class_coverage_frac",
        "mean_hard_positive_score",
        "mean_alignment_cosine",
        "mean_entropy_shift",
        "mean_fidelity_score",
        "mean_safe_radius_ratio",
        "mean_alignment_norm_mean",
        "mean_entropy_norm_mean",
        "mean_safe_radius_ratio_mean",
        "mean_fidelity_score_mean_selected",
        "dominant_metric_majority",
        "dominance_risk_majority",
    ]
    _write_csv(summary_dir / "variant_long.csv", variant_rows, _sorted_fieldnames(variant_rows, variant_pref))
    _write_csv(summary_dir / "dataset_variant_summary.csv", dataset_variant_rows, _sorted_fieldnames(dataset_variant_rows, summary_pref))
    _write_csv(summary_dir / "review_table.csv", review_rows, _sorted_fieldnames(review_rows, variant_pref))
    _write_failure_review_note(summary_dir, frozen_root, root, dataset_variant_rows)
    print(f"Failure review summary written to {summary_dir}")
    return 0


def _stability_logical_rows(frozen_root: Path, confirm_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    frozen_actual = _collect_actual_rows(frozen_root, [MBA_ARM, ACL_N4_ARM])
    confirm_actual = _collect_actual_rows(confirm_root, [MBA_ARM, ACL_N4_ARM])
    actual_rows = frozen_actual + confirm_actual
    actual_rows.sort(key=lambda row: (str(row.get("dataset", "")), _as_int(row.get("seed")) or 0, str(row.get("arm_dir", ""))))

    for row in actual_rows:
        dataset = str(row.get("dataset", ""))
        if dataset not in STABILITY_CONFIRM_DATASETS:
            continue
        seed = _as_int(row.get("seed"))
        arm_dir = str(row.get("arm_dir", ""))
        source_root = frozen_root if str(row.get("dataset_dir", "")).startswith(str(frozen_root)) else confirm_root
        common = {
            "dataset": dataset,
            "seed": seed,
            "status": str(row.get("status", "")),
            "source_arm_dir": arm_dir,
            "source_dataset_dir": row.get("dataset_dir", ""),
            "pipeline": row.get("pipeline", ""),
            "warmup_f1": _as_float(row.get("warmup_f1")),
            "base_f1": _as_float(row.get("base_f1")),
            "act_f1": _as_float(row.get("act_f1")),
            "selected_anchor_count": _as_int(row.get("selected_anchor_count")),
            "selected_positive_count": _as_int(row.get("selected_positive_count")),
            "candidate_total_count": _as_int(row.get("candidate_total_count")),
            "hard_positive_score_mean": _as_float(row.get("hard_positive_score_mean")),
            "fidelity_score_mean": _as_float(row.get("fidelity_score_mean")),
            "acl_last_ce_loss": _as_float(row.get("acl_last_ce_loss")),
            "acl_last_supcon_loss": _as_float(row.get("acl_last_supcon_loss")),
        }
        if arm_dir == MBA_ARM:
            rows.append({**common, "arm": "mba", "f1": _as_float(row.get("act_f1")), "supcon_active": None})
            continue

        audit = _summarize_acl_audit(source_root, arm_dir, dataset, int(seed or 0))
        rows.append(
            {
                **common,
                **audit,
                "arm": "gcg_acl",
                "f1": _as_float(row.get("act_f1")),
                "supcon_active": bool(
                    (_as_int(row.get("selected_anchor_count")) or 0) > 0
                    and (_as_int(row.get("selected_positive_count")) or 0) > 0
                    and (_as_float(row.get("acl_last_supcon_loss")) or 0.0) > 0.0
                ),
            }
        )
        rows.append(
            {
                **common,
                **audit,
                "arm": "continue_ce",
                "f1": _as_float(row.get("base_f1")),
                "acl_last_ce_loss": None,
                "acl_last_supcon_loss": None,
                "supcon_active": None,
            }
        )
    rows.sort(key=lambda r: (str(r["dataset"]), int(r["seed"]), STABILITY_LOGICAL_ARMS.index(str(r["arm"]))))
    return rows


def _stability_dataset_rows(logical_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in logical_rows:
        grouped[str(row["dataset"])][str(row["arm"])].append(row)

    summary_rows: List[Dict[str, object]] = []
    for dataset in sorted(grouped):
        per_arm = grouped[dataset]
        gcg_mean = _mean(_as_float(row.get("f1")) for row in per_arm.get("gcg_acl", []))
        gcg_std = _std(_as_float(row.get("f1")) for row in per_arm.get("gcg_acl", []))
        ce_mean = _mean(_as_float(row.get("f1")) for row in per_arm.get("continue_ce", []))
        mba_mean = _mean(_as_float(row.get("f1")) for row in per_arm.get("mba", []))
        mba_std = _std(_as_float(row.get("f1")) for row in per_arm.get("mba", []))
        gcg_mean_ge_continue = gcg_mean is not None and ce_mean is not None and gcg_mean >= ce_mean
        gcg_std_le_mba = gcg_std is not None and mba_std is not None and gcg_std <= mba_std + 1e-12
        n_complete = (
            len(per_arm.get("mba", [])) == 5
            and len(per_arm.get("continue_ce", [])) == 5
            and len(per_arm.get("gcg_acl", [])) == 5
        )
        stability_pass = bool(n_complete and gcg_mean_ge_continue and gcg_std_le_mba)
        for arm in STABILITY_LOGICAL_ARMS:
            arm_rows = per_arm.get(arm, [])
            summary_rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "n": len(arm_rows),
                    "mean_f1": _mean(_as_float(row.get("f1")) for row in arm_rows),
                    "std_f1": _std(_as_float(row.get("f1")) for row in arm_rows),
                    "gcg_acl_vs_continue_ce_mean_delta": None if gcg_mean is None or ce_mean is None else gcg_mean - ce_mean,
                    "gcg_acl_vs_mba_mean_delta": None if gcg_mean is None or mba_mean is None else gcg_mean - mba_mean,
                    "gcg_acl_std_le_mba_std": gcg_std_le_mba,
                    "gcg_acl_mean_ge_continue_ce_mean": gcg_mean_ge_continue,
                    "stability_confirm_pass": stability_pass,
                }
            )
    return summary_rows


def _write_stability_note(summary_dir: Path, dataset_rows: List[Dict[str, object]]) -> None:
    lines = ["# ACL v1 Stability Confirm", ""]
    for dataset in STABILITY_CONFIRM_DATASETS:
        rows = [row for row in dataset_rows if row["dataset"] == dataset]
        gcg_row = next((row for row in rows if row["arm"] == "gcg_acl"), None)
        if gcg_row is None:
            continue
        mean_ok = bool(gcg_row.get("gcg_acl_mean_ge_continue_ce_mean"))
        std_ok = bool(gcg_row.get("gcg_acl_std_le_mba_std"))
        n_complete = int(gcg_row.get("n") or 0) == 5
        if n_complete and mean_ok and std_ok:
            label = "confirmed_stable_positive"
        elif n_complete and mean_ok and not std_ok:
            label = "mean_positive_but_variance_open"
        else:
            label = "not_yet_confirmed"
        lines.extend(
            [
                f"## {dataset}",
                f"- label: `{label}`",
                f"- gcg_acl_mean_f1: `{gcg_row.get('mean_f1')}`",
                f"- gcg_acl_std_f1: `{gcg_row.get('std_f1')}`",
                f"- gcg_acl_vs_continue_ce_mean_delta: `{gcg_row.get('gcg_acl_vs_continue_ce_mean_delta')}`",
                f"- gcg_acl_vs_mba_mean_delta: `{gcg_row.get('gcg_acl_vs_mba_mean_delta')}`",
                "",
            ]
        )
    (summary_dir / "stability_note.md").write_text("\n".join(lines), encoding="utf-8")


def _stability_confirm_summary(root: Path, frozen_root: Path) -> int:
    logical_rows = _stability_logical_rows(frozen_root, root)
    dataset_rows = _stability_dataset_rows(logical_rows)
    summary_dir = root / "_summary"
    arm_pref = [
        "dataset",
        "seed",
        "arm",
        "f1",
        "status",
        "warmup_f1",
        "base_f1",
        "act_f1",
        "selected_anchor_count",
        "selected_positive_count",
        "candidate_total_count",
        "hard_positive_score_mean",
        "fidelity_score_mean",
        "acl_last_ce_loss",
        "acl_last_supcon_loss",
        "selected_class_coverage_frac",
        "selected_class_missing_frac",
        "coverage_risk",
        "dominant_metric",
        "dominant_corr",
        "dominance_gap",
        "dominance_risk",
        "supcon_active",
        "source_arm_dir",
        "source_dataset_dir",
    ]
    dataset_pref = [
        "dataset",
        "arm",
        "n",
        "mean_f1",
        "std_f1",
        "gcg_acl_vs_continue_ce_mean_delta",
        "gcg_acl_vs_mba_mean_delta",
        "gcg_acl_std_le_mba_std",
        "gcg_acl_mean_ge_continue_ce_mean",
        "stability_confirm_pass",
    ]
    _write_csv(summary_dir / "arm_long_5seed.csv", logical_rows, _sorted_fieldnames(logical_rows, arm_pref))
    _write_csv(summary_dir / "dataset_summary_5seed.csv", dataset_rows, _sorted_fieldnames(dataset_rows, dataset_pref))
    _write_stability_note(summary_dir, dataset_rows)
    print(f"Stability summary written to {summary_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize ACL v1 follow-up runs.")
    parser.add_argument("--mode", choices=["failure_review", "stability_confirm"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--frozen-root", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    frozen_root = Path(args.frozen_root).resolve()
    if args.mode == "failure_review":
        return _failure_review_summary(root, frozen_root)
    return _stability_confirm_summary(root, frozen_root)


if __name__ == "__main__":
    raise SystemExit(main())
