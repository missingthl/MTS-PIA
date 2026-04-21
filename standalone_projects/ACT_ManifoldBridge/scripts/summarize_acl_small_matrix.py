#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


STRUCTURE_DATASETS = ["natops", "japanesevowels", "handwriting", "libras"]
VOLATILITY_DATASETS = ["atrialfibrillation", "heartbeat", "motorimagery"]
SANITY_DATASETS = ["basicmotions"]
FULL_DATASETS = STRUCTURE_DATASETS + VOLATILITY_DATASETS + SANITY_DATASETS

MBA_ARM = "mba"
ACL_N4_ARM = "gcg_acl_n4"
ACL_N8_ARM = "gcg_acl_n8"
ARM_DIRS = [MBA_ARM, ACL_N4_ARM, ACL_N8_ARM]
LOGICAL_ARMS = ["mba", "continue_ce", "gcg_acl", "gcg_acl_n8"]
DOMINANCE_COMPONENTS = ["alignment_norm", "entropy_norm", "safe_radius_ratio", "fidelity_score"]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _as_float(value: object) -> Optional[float]:
    if value in (None, "", "nan", "NaN"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _as_int(value: object) -> Optional[int]:
    fval = _as_float(value)
    if fval is None:
        return None
    return int(round(fval))


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return None
    return statistics.mean(finite)


def _std(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if len(finite) < 2:
        return 0.0 if finite else None
    return statistics.stdev(finite)


def _rankdata(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx = statistics.mean(x)
    my = statistics.mean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    sx = math.sqrt(sum(v * v for v in dx))
    sy = math.sqrt(sum(v * v for v in dy))
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _candidate_audit_paths(root: Path, arm_dir: str, dataset: str, seed: int) -> Tuple[Path, Path]:
    audit_dir = root / arm_dir / dataset / "audit"
    return (
        audit_dir / f"{dataset}_s{seed}_candidate_scores.csv",
        audit_dir / f"{dataset}_s{seed}_selected_positives.csv",
    )


def _summarize_audit(root: Path, arm_dir: str, dataset: str, seed: int) -> Dict[str, object]:
    candidate_path, selected_path = _candidate_audit_paths(root, arm_dir, dataset, seed)
    summary: Dict[str, object] = {
        "candidate_csv": str(candidate_path),
        "selected_csv": str(selected_path),
        "audit_files_present": candidate_path.is_file() and selected_path.is_file(),
        "selected_class_coverage_frac": None,
        "selected_class_missing_frac": None,
        "coverage_risk": None,
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

    all_classes = sorted({_as_int(row.get("y")) for row in candidate_rows if _as_int(row.get("y")) is not None})
    selected_classes = sorted({_as_int(row.get("y")) for row in selected_rows if _as_int(row.get("y")) is not None})
    if all_classes:
        coverage_frac = len(selected_classes) / float(len(all_classes))
        missing_frac = max(0.0, 1.0 - coverage_frac)
        summary["selected_class_coverage_frac"] = coverage_frac
        summary["selected_class_missing_frac"] = missing_frac
        summary["coverage_risk"] = missing_frac > 0.25

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


def _collect_actual_runs(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for arm_dir in ARM_DIRS:
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


def _actual_to_logical_rows(root: Path, actual_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    logical_rows: List[Dict[str, object]] = []
    for row in actual_rows:
        dataset = str(row["dataset"])
        seed = _as_int(row["seed"])
        arm_dir = str(row["arm_dir"])
        status = str(row.get("status", ""))
        source_common = {
            "dataset": dataset,
            "seed": seed,
            "status": status,
            "source_arm_dir": arm_dir,
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
            logical_rows.append(
                {
                    **source_common,
                    "arm": "mba",
                    "f1": _as_float(row.get("act_f1")),
                }
            )
            continue

        audit = _summarize_audit(root, arm_dir, dataset, int(seed))
        logical_acl_arm = "gcg_acl" if arm_dir == ACL_N4_ARM else "gcg_acl_n8"

        logical_rows.append(
            {
                **source_common,
                **audit,
                "arm": logical_acl_arm,
                "f1": _as_float(row.get("act_f1")),
                "supcon_active": bool(
                    (_as_int(row.get("selected_anchor_count")) or 0) > 0
                    and (_as_int(row.get("selected_positive_count")) or 0) > 0
                    and (_as_float(row.get("acl_last_supcon_loss")) or 0.0) > 0.0
                ),
            }
        )
        if arm_dir == ACL_N4_ARM:
            logical_rows.append(
                {
                    **source_common,
                    **audit,
                    "arm": "continue_ce",
                    "f1": _as_float(row.get("base_f1")),
                    "acl_last_ce_loss": None,
                    "acl_last_supcon_loss": None,
                    "supcon_active": None,
                }
            )

    logical_rows.sort(key=lambda r: (str(r["dataset"]), int(r["seed"]), LOGICAL_ARMS.index(str(r["arm"]))))
    return logical_rows


def _dataset_summary(logical_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in logical_rows:
        grouped[str(row["dataset"])][str(row["arm"])].append(row)

    summaries: List[Dict[str, object]] = []
    for dataset in sorted(grouped):
        per_arm = grouped[dataset]
        summary: Dict[str, object] = {
            "dataset": dataset,
            "group": (
                "structure"
                if dataset in STRUCTURE_DATASETS
                else "volatility"
                if dataset in VOLATILITY_DATASETS
                else "sanity"
            ),
        }

        for arm in LOGICAL_ARMS:
            rows = per_arm.get(arm, [])
            f1s = [_as_float(row.get("f1")) for row in rows]
            summary[f"{arm}_n"] = len(rows)
            summary[f"{arm}_mean_f1"] = _mean(f1s)
            summary[f"{arm}_std_f1"] = _std(f1s)

        acl_rows = per_arm.get("gcg_acl", [])
        summary["gcg_acl_vs_continue_ce_mean_delta"] = None
        if summary.get("gcg_acl_mean_f1") is not None and summary.get("continue_ce_mean_f1") is not None:
            summary["gcg_acl_vs_continue_ce_mean_delta"] = summary["gcg_acl_mean_f1"] - summary["continue_ce_mean_f1"]
        summary["gcg_acl_vs_mba_mean_delta"] = None
        if summary.get("gcg_acl_mean_f1") is not None and summary.get("mba_mean_f1") is not None:
            summary["gcg_acl_vs_mba_mean_delta"] = summary["gcg_acl_mean_f1"] - summary["mba_mean_f1"]
        summary["gcg_acl_vs_n8_mean_delta"] = None
        if summary.get("gcg_acl_mean_f1") is not None and summary.get("gcg_acl_n8_mean_f1") is not None:
            summary["gcg_acl_vs_n8_mean_delta"] = summary["gcg_acl_mean_f1"] - summary["gcg_acl_n8_mean_f1"]

        acl_std = summary.get("gcg_acl_std_f1")
        mba_std = summary.get("mba_std_f1")
        summary["gcg_acl_not_worse_than_continue_ce"] = (
            summary.get("gcg_acl_mean_f1") is not None
            and summary.get("continue_ce_mean_f1") is not None
            and summary["gcg_acl_mean_f1"] >= summary["continue_ce_mean_f1"]
        )
        summary["gcg_acl_stabler_than_mba"] = (
            acl_std is not None and mba_std is not None and acl_std <= mba_std + 1e-12
        )
        coverage_flags = [bool(row["coverage_risk"]) for row in acl_rows if row.get("coverage_risk") is not None]
        dominance_flags = [bool(row["dominance_risk"]) for row in acl_rows if row.get("dominance_risk") is not None]
        supcon_flags = [bool(row["supcon_active"]) for row in acl_rows if row.get("supcon_active") is not None]
        summary["coverage_risk_majority"] = (
            len(coverage_flags) > 0 and sum(coverage_flags) > len(coverage_flags) / 2.0
        )
        summary["dominance_risk_majority"] = (
            len(dominance_flags) > 0 and sum(dominance_flags) > len(dominance_flags) / 2.0
        )
        summary["supcon_active_all_seeds"] = len(supcon_flags) > 0 and all(supcon_flags)
        summary["consistency_conclusion"] = (
            "stable_gain"
            if summary["gcg_acl_not_worse_than_continue_ce"] and summary["gcg_acl_stabler_than_mba"]
            else "mean_gain_only"
            if summary["gcg_acl_not_worse_than_continue_ce"]
            else "needs_review"
        )
        summary["dataset_gate_pass"] = (
            bool(summary["gcg_acl_not_worse_than_continue_ce"])
            and bool(summary["supcon_active_all_seeds"])
            and not bool(summary["coverage_risk_majority"])
        )
        summaries.append(summary)
    return summaries


def _gate_report(root: Path, phase: str, actual_rows: List[Dict[str, object]], dataset_rows: List[Dict[str, object]]) -> Dict[str, object]:
    if phase == "preflight":
        expected_actual = 3
        n4_rows = [row for row in actual_rows if row.get("arm_dir") == ACL_N4_ARM]
        n8_rows = [row for row in actual_rows if row.get("arm_dir") == ACL_N8_ARM]
        mba_rows = [row for row in actual_rows if row.get("arm_dir") == MBA_ARM]
        n4 = n4_rows[0] if n4_rows else {}
        n8 = n8_rows[0] if n8_rows else {}
        n4_audit = _summarize_audit(root, ACL_N4_ARM, "basicmotions", 1) if n4_rows else {}
        n8_audit = _summarize_audit(root, ACL_N8_ARM, "basicmotions", 1) if n8_rows else {}
        preflight_ok = (
            len(actual_rows) == expected_actual
            and all(str(row.get("status", "")) == "success" for row in actual_rows)
            and len(mba_rows) == 1
            and len(n4_rows) == 1
            and len(n8_rows) == 1
            and _as_float(n4.get("warmup_f1")) is not None
            and _as_float(n4.get("base_f1")) is not None
            and _as_float(n4.get("act_f1")) is not None
            and abs((_as_float(n4.get("base_f1")) or 0.0) - (_as_float(n4.get("warmup_f1")) or 0.0)) > 1e-12
            and bool(n4_audit.get("audit_files_present"))
            and bool(n8_audit.get("audit_files_present"))
            and _as_int(n4.get("selected_anchor_count")) is not None
            and (_as_int(n4.get("selected_anchor_count")) or 0) > 0
            and (_as_int(n4.get("selected_positive_count")) or 0) > 0
            and (_as_float(n4.get("candidate_total_count")) or 0.0) > 0
            and (_as_float(n4.get("hard_positive_score_mean")) or 0.0) > 0.0
            and (_as_float(n4.get("fidelity_score_mean")) or 0.0) > 0.0
            and (_as_float(n4.get("acl_last_supcon_loss")) or 0.0) > 0.0
        )
        return {
            "phase": phase,
            "expected_actual_runs": expected_actual,
            "observed_actual_runs": len(actual_rows),
            "all_status_success": all(str(row.get("status", "")) == "success" for row in actual_rows),
            "n4_audit_present": bool(n4_audit.get("audit_files_present")),
            "n8_audit_present": bool(n8_audit.get("audit_files_present")),
            "preflight_gate_pass": preflight_ok,
        }

    expected_actual = len(FULL_DATASETS) * len(ARM_DIRS)
    expected_seed_rows = expected_actual * 3
    gcg_rows = [row for row in actual_rows if row.get("arm_dir") in {ACL_N4_ARM, ACL_N8_ARM}]
    audit_complete = True
    for row in gcg_rows:
        dataset = str(row.get("dataset", ""))
        seed = _as_int(row.get("seed"))
        arm_dir = str(row.get("arm_dir", ""))
        if seed is None:
            audit_complete = False
            break
        audit = _summarize_audit(root, arm_dir, dataset, int(seed))
        if not bool(audit.get("audit_files_present")):
            audit_complete = False
            break
    all_success = len(actual_rows) == expected_seed_rows and all(str(row.get("status", "")) == "success" for row in actual_rows)
    structure_pass_count = sum(
        1 for row in dataset_rows if row["dataset"] in STRUCTURE_DATASETS and bool(row["gcg_acl_not_worse_than_continue_ce"])
    )
    volatility_pass_count = sum(
        1 for row in dataset_rows if row["dataset"] in VOLATILITY_DATASETS and bool(row["gcg_acl_stabler_than_mba"])
    )
    return {
        "phase": phase,
        "expected_actual_runs": expected_actual,
        "expected_seed_rows": expected_seed_rows,
        "observed_seed_rows": len(actual_rows),
        "all_status_success": all_success,
        "audit_files_complete": audit_complete,
        "structure_pass_count": structure_pass_count,
        "volatility_stability_pass_count": volatility_pass_count,
        "full_matrix_gate_pass": bool(all_success and audit_complete and structure_pass_count >= 3 and volatility_pass_count >= 2),
    }


def _sorted_fieldnames(rows: List[Dict[str, object]], preferred: Sequence[str]) -> List[str]:
    keys = set()
    for row in rows:
        keys.update(row.keys())
    ordered = [key for key in preferred if key in keys]
    ordered.extend(sorted(keys - set(ordered)))
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize ACL v1 small-matrix runs.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--phase", choices=["preflight", "full"], required=True)
    parser.add_argument("--fail-on-preflight", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    actual_rows = _collect_actual_runs(root)
    logical_rows = _actual_to_logical_rows(root, actual_rows)
    dataset_rows = _dataset_summary(logical_rows)
    gate_report = _gate_report(root, args.phase, actual_rows, dataset_rows)

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
    ]
    dataset_pref = [
        "dataset",
        "group",
        "mba_mean_f1",
        "mba_std_f1",
        "continue_ce_mean_f1",
        "continue_ce_std_f1",
        "gcg_acl_mean_f1",
        "gcg_acl_std_f1",
        "gcg_acl_n8_mean_f1",
        "gcg_acl_n8_std_f1",
        "gcg_acl_vs_continue_ce_mean_delta",
        "gcg_acl_vs_mba_mean_delta",
        "gcg_acl_vs_n8_mean_delta",
        "gcg_acl_not_worse_than_continue_ce",
        "gcg_acl_stabler_than_mba",
        "coverage_risk_majority",
        "dominance_risk_majority",
        "supcon_active_all_seeds",
        "consistency_conclusion",
        "dataset_gate_pass",
    ]
    _write_csv(summary_dir / "arm_long.csv", logical_rows, _sorted_fieldnames(logical_rows, arm_pref))
    _write_csv(summary_dir / "per_seed_summary.csv", logical_rows, _sorted_fieldnames(logical_rows, arm_pref))
    _write_csv(summary_dir / "dataset_summary.csv", dataset_rows, _sorted_fieldnames(dataset_rows, dataset_pref))
    (summary_dir / "gate_report.json").write_text(json.dumps(gate_report, indent=2), encoding="utf-8")

    print(f"Summary written to {summary_dir}")
    print(json.dumps(gate_report, indent=2))

    if args.phase == "preflight" and args.fail_on_preflight and not gate_report.get("preflight_gate_pass", False):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
