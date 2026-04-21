#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


MAIN_ARMS = [
    "mba",
    "mba_feedback_easy",
    "mba_wide",
    "mba_wide_feedback_easy",
    "mba_wide_feedback_hard",
]
REFILL_ARMS = [
    "mba_wide_feedback_easy_tau2",
    "mba_wide_feedback_hard_tau2",
]
ALL_ARMS = MAIN_ARMS + REFILL_ARMS
DATASETS = ["natops", "heartbeat", "atrialfibrillation"]


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
    return out if math.isfinite(out) else None


def _as_int(value: object) -> Optional[int]:
    fval = _as_float(value)
    return None if fval is None else int(round(fval))


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.mean(finite) if finite else None


def _std(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return None
    if len(finite) < 2:
        return 0.0
    return statistics.stdev(finite)


def _final_results_path(root: Path, arm: str, dataset: str) -> Path:
    return root / arm / dataset / "final_results.csv"


def _tier_summary_path(root: Path, arm: str, dataset: str, seed: int) -> Path:
    return root / arm / dataset / "audit" / f"{dataset}_s{seed}_tier_summary.csv"


def _candidate_audit_path(root: Path, arm: str, dataset: str, seed: int) -> Path:
    return root / arm / dataset / "audit" / f"{dataset}_s{seed}_widened_candidates.csv"


def _collect_actual_rows(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for arm in ALL_ARMS:
        for dataset in DATASETS:
            path = _final_results_path(root, arm, dataset)
            if not path.is_file():
                continue
            for row in _read_csv(path):
                rows.append({**row, "arm": arm})
    return rows


def _load_tier_summary(root: Path, arm: str, dataset: str, seed: int) -> Dict[str, object]:
    path = _tier_summary_path(root, arm, dataset, seed)
    if not path.is_file():
        return {}
    rows = _read_csv(path)
    return rows[0] if rows else {}


def _load_candidate_alignment(root: Path, arm: str, dataset: str, seed: int) -> Dict[str, Optional[float]]:
    path = _candidate_audit_path(root, arm, dataset, seed)
    if not path.is_file():
        return {}
    rows = _read_csv(path)
    per_tier: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        tier = str(row.get("tier_label", ""))
        align = _as_float(row.get("alignment_cosine"))
        if tier and align is not None:
            per_tier[tier].append(align)
    return {f"alignment_{tier}_mean": _mean(vals) for tier, vals in per_tier.items()}


def _build_arm_long(root: Path, actual_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    long_rows: List[Dict[str, object]] = []
    for row in actual_rows:
        dataset = str(row["dataset"])
        seed = _as_int(row["seed"])
        arm = str(row["arm"])
        tier_summary = _load_tier_summary(root, arm, dataset, int(seed or 0))
        align_summary = _load_candidate_alignment(root, arm, dataset, int(seed or 0))
        long_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "arm": arm,
                "status": row.get("status", ""),
                "pipeline": row.get("pipeline", ""),
                "mba_candidate_mode": row.get("mba_candidate_mode", ""),
                "feedback_margin_polarity": row.get("feedback_margin_polarity", ""),
                "base_f1": _as_float(row.get("base_f1")),
                "act_f1": _as_float(row.get("act_f1")),
                "gain": _as_float(row.get("gain")),
                "candidate_total_count": _as_int(row.get("candidate_total_count")),
                "aug_total_count": _as_int(row.get("aug_total_count")),
                "step_tier_count": _as_int(row.get("step_tier_count")),
                "feedback_weight_mean": _as_float(row.get("feedback_weight_mean")),
                "feedback_reject_frac": _as_float(row.get("feedback_reject_frac")),
                "last_aug_margin_mean": _as_float(row.get("last_aug_margin_mean")),
                "transport_error_logeuc_mean": _as_float(row.get("transport_error_logeuc_mean")),
                "candidate_audit_csv": row.get("candidate_audit_csv", ""),
                "tier_summary_csv": row.get("tier_summary_csv", ""),
                "ray_summary_csv": row.get("ray_summary_csv", ""),
                "E_w_small": _as_float(tier_summary.get("E_w_small")),
                "E_w_mid": _as_float(tier_summary.get("E_w_mid")),
                "E_w_edge": _as_float(tier_summary.get("E_w_edge")),
                "admission_small": _as_float(tier_summary.get("admission_small")),
                "admission_mid": _as_float(tier_summary.get("admission_mid")),
                "admission_edge": _as_float(tier_summary.get("admission_edge")),
                "tier_margin_mean_small": _as_float(tier_summary.get("tier_margin_mean_small")),
                "tier_margin_mean_mid": _as_float(tier_summary.get("tier_margin_mean_mid")),
                "tier_margin_mean_edge": _as_float(tier_summary.get("tier_margin_mean_edge")),
                "tier_transport_error_logeuc_mean_small": _as_float(tier_summary.get("tier_transport_error_logeuc_mean_small")),
                "tier_transport_error_logeuc_mean_mid": _as_float(tier_summary.get("tier_transport_error_logeuc_mean_mid")),
                "tier_transport_error_logeuc_mean_edge": _as_float(tier_summary.get("tier_transport_error_logeuc_mean_edge")),
                "tier_alignment_cosine_mean_small": _as_float(tier_summary.get("tier_alignment_cosine_mean_small")),
                "tier_alignment_cosine_mean_mid": _as_float(tier_summary.get("tier_alignment_cosine_mean_mid")),
                "tier_alignment_cosine_mean_edge": _as_float(tier_summary.get("tier_alignment_cosine_mean_edge")),
                "alignment_small_mean": _as_float(align_summary.get("alignment_small_mean")),
                "alignment_mid_mean": _as_float(align_summary.get("alignment_mid_mean")),
                "alignment_edge_mean": _as_float(align_summary.get("alignment_edge_mean")),
            }
        )
    long_rows.sort(key=lambda r: (str(r["dataset"]), int(r["seed"]), ALL_ARMS.index(str(r["arm"]))))
    return long_rows


def _build_collapse_candidates(arm_long: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in arm_long:
        if row["arm"] != "mba_wide_feedback_easy":
            continue
        ew_small = _as_float(row.get("E_w_small"))
        ew_edge = _as_float(row.get("E_w_edge"))
        admit_small = _as_float(row.get("admission_small"))
        admit_edge = _as_float(row.get("admission_edge"))
        trigger_weight = ew_small is not None and ew_edge is not None and ew_small >= 0.90 and ew_edge <= 0.10
        trigger_admission = admit_small is not None and admit_edge is not None and admit_small >= 0.90 and admit_edge <= 0.10
        if not trigger_weight and not trigger_admission:
            continue
        rows.append(
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "trigger_by_weight": trigger_weight,
                "trigger_by_admission": trigger_admission,
                "E_w_small": ew_small,
                "E_w_edge": ew_edge,
                "admission_small": admit_small,
                "admission_edge": admit_edge,
                "tier_alignment_cosine_mean_edge": _as_float(row.get("tier_alignment_cosine_mean_edge")),
                "tier_transport_error_logeuc_mean_edge": _as_float(row.get("tier_transport_error_logeuc_mean_edge")),
            }
        )
    rows.sort(key=lambda r: (str(r["dataset"]), int(r["seed"])))
    return rows


def _dataset_summary(arm_long: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in arm_long:
        grouped[str(row["dataset"])][str(row["arm"])].append(row)

    summaries: List[Dict[str, object]] = []
    for dataset in DATASETS:
        per_arm = grouped.get(dataset, {})
        summary: Dict[str, object] = {"dataset": dataset}
        for arm in MAIN_ARMS:
            rows = per_arm.get(arm, [])
            summary[f"{arm}_n"] = len(rows)
            summary[f"{arm}_mean_f1"] = _mean(_as_float(r.get("act_f1")) for r in rows)
            summary[f"{arm}_std_f1"] = _std(_as_float(r.get("act_f1")) for r in rows)

        refill_easy = per_arm.get("mba_wide_feedback_easy_tau2", [])
        refill_hard = per_arm.get("mba_wide_feedback_hard_tau2", [])
        summary["mba_wide_feedback_easy_tau2_mean_f1"] = _mean(_as_float(r.get("act_f1")) for r in refill_easy)
        summary["mba_wide_feedback_hard_tau2_mean_f1"] = _mean(_as_float(r.get("act_f1")) for r in refill_hard)
        core_feedback_rows = per_arm.get("mba_feedback_easy", [])
        summary["matched_ce_mean_f1"] = _mean(_as_float(r.get("base_f1")) for r in core_feedback_rows)

        summary["mba_wide_vs_mba_delta"] = None
        if summary.get("mba_wide_mean_f1") is not None and summary.get("mba_mean_f1") is not None:
            summary["mba_wide_vs_mba_delta"] = summary["mba_wide_mean_f1"] - summary["mba_mean_f1"]
        summary["wide_easy_vs_core_feedback_delta"] = None
        if summary.get("mba_wide_feedback_easy_mean_f1") is not None and summary.get("mba_feedback_easy_mean_f1") is not None:
            summary["wide_easy_vs_core_feedback_delta"] = summary["mba_wide_feedback_easy_mean_f1"] - summary["mba_feedback_easy_mean_f1"]
        summary["wide_hard_vs_wide_easy_delta"] = None
        if summary.get("mba_wide_feedback_hard_mean_f1") is not None and summary.get("mba_wide_feedback_easy_mean_f1") is not None:
            summary["wide_hard_vs_wide_easy_delta"] = summary["mba_wide_feedback_hard_mean_f1"] - summary["mba_wide_feedback_easy_mean_f1"]

        easy_rows = per_arm.get("mba_wide_feedback_easy", [])
        hard_rows = per_arm.get("mba_wide_feedback_hard", [])
        summary["collapse_seed_count"] = sum(
            1
            for row in easy_rows
            if (
                (_as_float(row.get("E_w_small")) or 0.0) >= 0.90 and (_as_float(row.get("E_w_edge")) or 1.0) <= 0.10
            )
            or (
                (_as_float(row.get("admission_small")) or 0.0) >= 0.90 and (_as_float(row.get("admission_edge")) or 1.0) <= 0.10
            )
        )
        summary["edge_alignment_mean_easy"] = _mean(_as_float(r.get("tier_alignment_cosine_mean_edge")) for r in easy_rows)
        summary["edge_transport_mean_easy"] = _mean(_as_float(r.get("tier_transport_error_logeuc_mean_edge")) for r in easy_rows)
        summary["edge_transport_mean_hard"] = _mean(_as_float(r.get("tier_transport_error_logeuc_mean_edge")) for r in hard_rows)
        summary["edge_weight_mean_easy"] = _mean(_as_float(r.get("E_w_edge")) for r in easy_rows)
        summary["edge_weight_mean_hard"] = _mean(_as_float(r.get("E_w_edge")) for r in hard_rows)

        conclusion = "insufficient_data"
        wide_mba_delta = summary.get("mba_wide_vs_mba_delta")
        wide_easy_delta = summary.get("wide_easy_vs_core_feedback_delta")
        hard_easy_delta = summary.get("wide_hard_vs_wide_easy_delta")
        edge_transport = summary.get("edge_transport_mean_easy")
        small_transport = _mean(_as_float(r.get("tier_transport_error_logeuc_mean_small")) for r in easy_rows)
        mid_transport = _mean(_as_float(r.get("tier_transport_error_logeuc_mean_mid")) for r in easy_rows)
        edge_align = summary.get("edge_alignment_mean_easy")
        edge_weight_easy = summary.get("edge_weight_mean_easy")
        edge_weight_hard = summary.get("edge_weight_mean_hard")

        if (
            edge_transport is not None
            and small_transport is not None
            and mid_transport is not None
            and edge_transport > max(small_transport, mid_transport) * 1.5
            and (edge_weight_easy or 0.0) <= 0.10
            and (edge_weight_hard or 0.0) <= 0.10
        ):
            conclusion = "bridge_nonlinearity_bottleneck"
        elif (
            summary["collapse_seed_count"] > 0
            and edge_align is not None
            and edge_align >= 0.25
            and (edge_weight_easy or 0.0) <= 0.10
        ):
            conclusion = "selective_collapse_under_confidence_bias"
        elif (
            wide_mba_delta is not None
            and wide_easy_delta is not None
            and hard_easy_delta is not None
            and wide_mba_delta > 0.0
            and wide_easy_delta <= 0.0
            and hard_easy_delta > 0.0
        ):
            conclusion = "scorer_value_misalignment"
        elif (
            wide_mba_delta is not None
            and wide_easy_delta is not None
            and hard_easy_delta is not None
            and wide_mba_delta > 0.0
            and wide_easy_delta <= 0.0
            and hard_easy_delta <= 0.0
        ):
            conclusion = "feedback_objective_not_helping_yet"
        elif wide_easy_delta is not None and wide_easy_delta > 0.0 and summary["collapse_seed_count"] == 0:
            conclusion = "candidate_space_was_too_narrow"
        summary["conclusion"] = conclusion
        summaries.append(summary)
    return summaries


def _tau_refill_summary(arm_long: List[Dict[str, object]]) -> List[Dict[str, object]]:
    keyed = {(str(r["dataset"]), int(r["seed"]), str(r["arm"])): r for r in arm_long}
    rows: List[Dict[str, object]] = []
    for dataset in DATASETS:
        for seed in [1, 2, 3]:
            easy = keyed.get((dataset, seed, "mba_wide_feedback_easy"))
            hard = keyed.get((dataset, seed, "mba_wide_feedback_hard"))
            easy_tau2 = keyed.get((dataset, seed, "mba_wide_feedback_easy_tau2"))
            hard_tau2 = keyed.get((dataset, seed, "mba_wide_feedback_hard_tau2"))
            if easy_tau2 is None and hard_tau2 is None:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "easy_f1_tau1": _as_float(easy.get("act_f1")) if easy else None,
                    "easy_f1_tau2": _as_float(easy_tau2.get("act_f1")) if easy_tau2 else None,
                    "easy_delta_tau2_minus_tau1": (
                        (_as_float(easy_tau2.get("act_f1")) if easy_tau2 else None) - (_as_float(easy.get("act_f1")) if easy else None)
                        if easy and easy_tau2 and _as_float(easy_tau2.get("act_f1")) is not None and _as_float(easy.get("act_f1")) is not None
                        else None
                    ),
                    "hard_f1_tau1": _as_float(hard.get("act_f1")) if hard else None,
                    "hard_f1_tau2": _as_float(hard_tau2.get("act_f1")) if hard_tau2 else None,
                    "hard_delta_tau2_minus_tau1": (
                        (_as_float(hard_tau2.get("act_f1")) if hard_tau2 else None) - (_as_float(hard.get("act_f1")) if hard else None)
                        if hard and hard_tau2 and _as_float(hard_tau2.get("act_f1")) is not None and _as_float(hard.get("act_f1")) is not None
                        else None
                    ),
                }
            )
    return rows


def _write_note(path: Path, dataset_rows: List[Dict[str, object]], collapse_rows: List[Dict[str, object]]) -> None:
    lines = ["# MBA Step-Tier Widening v1 Readout", ""]
    if collapse_rows:
        lines.append(f"- selective collapse candidates: {len(collapse_rows)}")
    else:
        lines.append("- selective collapse candidates: 0")
    lines.append("")
    for row in dataset_rows:
        lines.append(f"## {row['dataset']}")
        lines.append(f"- conclusion: `{row['conclusion']}`")
        lines.append(f"- mba mean: `{row.get('mba_mean_f1')}`")
        lines.append(f"- mba_wide mean: `{row.get('mba_wide_mean_f1')}`")
        lines.append(f"- mba_feedback_easy mean: `{row.get('mba_feedback_easy_mean_f1')}`")
        lines.append(f"- mba_wide_feedback_easy mean: `{row.get('mba_wide_feedback_easy_mean_f1')}`")
        lines.append(f"- mba_wide_feedback_hard mean: `{row.get('mba_wide_feedback_hard_mean_f1')}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize MBA step-tier widening matrix.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    summary_root = root / "_summary"
    actual_rows = _collect_actual_rows(root)
    arm_long = _build_arm_long(root, actual_rows)
    collapse_rows = _build_collapse_candidates(arm_long)
    dataset_rows = _dataset_summary(arm_long)
    tau_rows = _tau_refill_summary(arm_long)

    if arm_long:
        _write_csv(summary_root / "arm_long.csv", arm_long, fieldnames=list(arm_long[0].keys()))
    if collapse_rows:
        _write_csv(summary_root / "collapse_candidates.csv", collapse_rows, fieldnames=list(collapse_rows[0].keys()))
    else:
        _write_csv(
            summary_root / "collapse_candidates.csv",
            [],
            fieldnames=[
                "dataset",
                "seed",
                "trigger_by_weight",
                "trigger_by_admission",
                "E_w_small",
                "E_w_edge",
                "admission_small",
                "admission_edge",
                "tier_alignment_cosine_mean_edge",
                "tier_transport_error_logeuc_mean_edge",
            ],
        )
    if dataset_rows:
        _write_csv(summary_root / "dataset_summary.csv", dataset_rows, fieldnames=list(dataset_rows[0].keys()))
    if tau_rows:
        _write_csv(summary_root / "tau_refill_summary.csv", tau_rows, fieldnames=list(tau_rows[0].keys()))
    _write_note(summary_root / "readout_note.md", dataset_rows, collapse_rows)
    print(f"[OK] wrote summaries under {summary_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
