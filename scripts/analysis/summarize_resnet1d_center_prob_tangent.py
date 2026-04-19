#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(ROOT_BOOTSTRAP))


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float]) -> float | None:
    values = [float(v) for v in values]
    if not values:
        return None
    return float(sum(values) / len(values))


def _load_e0_reference(path: Path | None) -> Dict[str, float]:
    if path is None or not path.is_file():
        return {}
    rows = _read_csv(path)
    out: Dict[str, float] = {}
    for row in rows:
        dataset = str(row.get("dataset", "")).strip().lower()
        e0_acc = _as_float(row.get("E0_acc"))
        if dataset and e0_acc is not None:
            out[dataset] = e0_acc
    return out


def _load_reference_table(path: Path | None, *, label: str) -> Dict[str, Dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    rows = _read_csv(path)
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        dataset = str(row.get("dataset", "")).strip().lower()
        if dataset and str(row.get("label")) == label:
            out[dataset] = row
    return out


def _row_metric(row: Dict[str, Any], key: str) -> float | None:
    value = _as_float(row.get(key))
    if value is not None:
        return value
    if key == "final_acc":
        return _as_float(row.get("test_acc"))
    return None


def _group_mean(rows: List[Dict[str, Any]], *, label: str, key: str) -> float | None:
    values = [_as_float(row.get(key)) for row in rows if str(row.get("label")) == label]
    values = [float(v) for v in values if v is not None]
    return _mean(values)


def _average_ranks(values: List[float]) -> List[float]:
    order = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0 for _ in values]
    idx = 0
    while idx < len(order):
        j = idx
        while j + 1 < len(order) and order[j + 1][1] == order[idx][1]:
            j += 1
        avg_rank = 0.5 * (idx + j) + 1.0
        for k in range(idx, j + 1):
            ranks[order[k][0]] = avg_rank
        idx = j + 1
    return ranks


def _pearson(x: List[float], y: List[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    x_centered = [v - x_mean for v in x]
    y_centered = [v - y_mean for v in y]
    x_var = sum(v * v for v in x_centered)
    y_var = sum(v * v for v in y_centered)
    if x_var <= 0.0 or y_var <= 0.0:
        return None
    cov = sum(a * b for a, b in zip(x_centered, y_centered))
    return float(cov / math.sqrt(x_var * y_var))


def _spearman(x: List[float], y: List[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    return _pearson(_average_ranks(x), _average_ranks(y))


def _candidate_sort_key(candidate: Dict[str, Any]) -> tuple[float, float, float]:
    fm_gain = float(candidate.get("fm_gain_vs_current_v3") or float("-inf"))
    scp1_gain = float(candidate.get("scp1_gain_vs_current_v3") or float("-inf"))
    selectivity = float(candidate.get("k0_selectivity_gap") or float("-inf"))
    rank_center = -abs(float(candidate.get("avg_mean_selected_rank") or 0.0) - 2.5)
    return (fm_gain + scp1_gain, selectivity, rank_center)


def _build_phase_a_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"phase": "phaseA_probe", "n_rows": int(len(rows))}
    by_label: Dict[str, Dict[str, Any]] = {}
    for label in sorted({str(row["label"]) for row in rows}):
        subset = [row for row in rows if str(row["label"]) == label]
        by_label[label] = {
            "avg_test_acc": _group_mean(subset, label=label, key="test_acc"),
            "posterior_confidence_mean": _group_mean(subset, label=label, key="posterior_confidence_mean"),
            "posterior_confidence_qgap": _group_mean(subset, label=label, key="posterior_confidence_qgap"),
            "posterior_log_confidence_std": _group_mean(subset, label=label, key="posterior_log_confidence_std"),
            "posterior_sigma2_eff_mean": _group_mean(subset, label=label, key="posterior_sigma2_eff_mean"),
        }
    summary["by_label"] = by_label
    return summary


def _build_phase_b_refined_summary(
    rows: List[Dict[str, Any]],
    *,
    reference_phase05_table: Path | None,
) -> Dict[str, Any]:
    current_v3_reference = _load_reference_table(reference_phase05_table, label="center_prob_tangent_v3")
    smooth = {"natops", "epilepsy"}
    noisy = {"fingermovements", "selfregulationscp1"}
    summary: Dict[str, Any] = {
        "phase": "phaseB_short_formal_refined",
        "n_rows": int(len(rows)),
        "current_v3_reference_table": None if reference_phase05_table is None else str(reference_phase05_table),
        "current_v3_reference_is_static": True,
        "baseline_avg_k0": 0.4111,
        "baseline_avg_rank": 2.3558,
        "overconservative_k0_threshold": 0.56,
        "overconservative_rank_threshold": 1.85,
        "overconservative_threshold_note": "First-round empirical filters; may be uniformly re-estimated after Phase A if refined scales shift materially.",
    }
    candidates: List[Dict[str, Any]] = []
    candidate_rows_by_label: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        candidate_rows_by_label.setdefault(str(row["label"]), []).append(row)

    for label, candidate_rows in sorted(candidate_rows_by_label.items()):
        posterior_mode = str(candidate_rows[0].get("posterior_mode"))
        beta = _as_float(candidate_rows[0].get("mdl_penalty_beta"))
        row_by_dataset = {str(row["dataset"]).lower(): row for row in candidate_rows}
        smooth_conf = [_as_float(row_by_dataset[d].get("posterior_confidence_mean")) for d in smooth if d in row_by_dataset]
        noisy_conf = [_as_float(row_by_dataset[d].get("posterior_confidence_mean")) for d in noisy if d in row_by_dataset]
        smooth_conf = [float(v) for v in smooth_conf if v is not None]
        noisy_conf = [float(v) for v in noisy_conf if v is not None]
        q50_pass_count = sum(1 for row in candidate_rows if (_as_float(row.get("posterior_confidence_q50")) or 0.0) > 1e-6)
        qgap_pass_count = sum(1 for row in candidate_rows if (_as_float(row.get("posterior_confidence_qgap")) or 0.0) > 1e-4)
        log_std_pass_count = sum(1 for row in candidate_rows if (_as_float(row.get("posterior_log_confidence_std")) or 0.0) > 0.5)
        qratio_pass_count = sum(1 for row in candidate_rows if (_as_float(row.get("posterior_confidence_qratio")) or 0.0) >= 10.0)
        posterior_noncollapse_ok = bool(q50_pass_count >= 3)
        posterior_qgap_ok = bool(qgap_pass_count >= 3)
        posterior_log_std_ok = bool(log_std_pass_count >= 3)
        posterior_qratio_ok = bool(qratio_pass_count >= 3)
        posterior_flat_warning = bool(posterior_qgap_ok and not posterior_qratio_ok)
        posterior_direction_ok = bool(
            smooth_conf and noisy_conf and (_mean(smooth_conf) or 0.0) > (_mean(noisy_conf) or 0.0)
        )

        fm_ref = current_v3_reference.get("fingermovements", {})
        scp1_ref = current_v3_reference.get("selfregulationscp1", {})
        natops_ref = current_v3_reference.get("natops", {})
        epilepsy_ref = current_v3_reference.get("epilepsy", {})
        fm_gain = (_row_metric(row_by_dataset.get("fingermovements", {}), "final_acc") or 0.0) - (_row_metric(fm_ref, "final_acc") or 0.0)
        scp1_gain = (_row_metric(row_by_dataset.get("selfregulationscp1", {}), "final_acc") or 0.0) - (_row_metric(scp1_ref, "final_acc") or 0.0)
        natops_drop = (_row_metric(natops_ref, "final_acc") or 0.0) - (_row_metric(row_by_dataset.get("natops", {}), "final_acc") or 0.0)
        epilepsy_drop = (_row_metric(epilepsy_ref, "final_acc") or 0.0) - (_row_metric(row_by_dataset.get("epilepsy", {}), "final_acc") or 0.0)
        behavior_gate_ok = bool(fm_gain >= 0.03 and scp1_gain >= 0.01 and natops_drop <= 0.01 and epilepsy_drop <= 0.01)

        avg_k0 = _mean([_as_float(row.get("k0_fallback_rate")) or 0.0 for row in candidate_rows]) or 0.0
        avg_rank = _mean([_as_float(row.get("mean_selected_rank")) or 0.0 for row in candidate_rows]) or 0.0
        noisy_k0 = _mean([_as_float(row_by_dataset[d].get("k0_fallback_rate")) or 0.0 for d in noisy if d in row_by_dataset]) or 0.0
        smooth_k0 = _mean([_as_float(row_by_dataset[d].get("k0_fallback_rate")) or 0.0 for d in smooth if d in row_by_dataset]) or 0.0
        k0_selectivity_gap = float(noisy_k0 - smooth_k0)
        mdl_selective_ok = bool(
            (_as_float(row_by_dataset.get("selfregulationscp1", {}).get("k0_fallback_rate")) or 0.0) < 0.95
            and (
                (_as_float(row_by_dataset.get("fingermovements", {}).get("mean_selected_rank")) or 4.0) < 4.0
                or (_as_float(row_by_dataset.get("fingermovements", {}).get("k0_fallback_rate")) or 0.0) > 0.05
            )
            and k0_selectivity_gap > 0.0
        )

        filtered_by_damage = bool(natops_drop > 0.01 or epilepsy_drop > 0.01)
        filtered_by_posterior = not (
            posterior_noncollapse_ok
            and posterior_qgap_ok
            and posterior_log_std_ok
            and posterior_qratio_ok
            and posterior_direction_ok
        )
        overconservative = bool(avg_k0 >= 0.56 and avg_rank <= 1.85)
        candidates.append(
            {
                "label": label,
                "posterior_mode": posterior_mode,
                "mdl_penalty_beta": beta,
                "posterior_noncollapse_ok": posterior_noncollapse_ok,
                "posterior_qgap_ok": posterior_qgap_ok,
                "posterior_log_std_ok": posterior_log_std_ok,
                "posterior_qratio_ok": posterior_qratio_ok,
                "posterior_direction_ok": posterior_direction_ok,
                "posterior_flat_warning": posterior_flat_warning,
                "fm_gain_vs_current_v3": float(fm_gain),
                "scp1_gain_vs_current_v3": float(scp1_gain),
                "natops_drop_vs_current_v3": float(natops_drop),
                "epilepsy_drop_vs_current_v3": float(epilepsy_drop),
                "behavior_gate_ok": behavior_gate_ok,
                "avg_k0_fallback_rate": float(avg_k0),
                "avg_mean_selected_rank": float(avg_rank),
                "k0_selectivity_gap": float(k0_selectivity_gap),
                "mdl_selective_ok": mdl_selective_ok,
                "filtered_by_damage": filtered_by_damage,
                "filtered_by_posterior": filtered_by_posterior,
                "overconservative": overconservative,
            }
        )

    non_overconservative_exists = any(
        not bool(candidate["overconservative"]) for candidate in candidates if not bool(candidate["filtered_by_damage"]) and not bool(candidate["filtered_by_posterior"])
    )
    for candidate in candidates:
        candidate["filtered_by_overconservative"] = bool(non_overconservative_exists and candidate["overconservative"])
        candidate["eligible_after_filters"] = not (
            bool(candidate["filtered_by_damage"])
            or bool(candidate["filtered_by_posterior"])
            or bool(candidate["filtered_by_overconservative"])
        )
        candidate["phase_c_gate_passed"] = bool(
            candidate["eligible_after_filters"]
            and candidate["behavior_gate_ok"]
            and candidate["mdl_selective_ok"]
        )

    eligible_candidates = [candidate for candidate in candidates if bool(candidate["eligible_after_filters"])]
    selected_candidate = max(eligible_candidates, key=_candidate_sort_key) if eligible_candidates else None
    summary["candidates"] = candidates
    summary["selected_candidate"] = selected_candidate
    summary["phase_c_gate_passed"] = bool(selected_candidate and selected_candidate.get("phase_c_gate_passed"))
    if selected_candidate is not None:
        summary["selected_posterior_mode"] = selected_candidate.get("posterior_mode")
        summary["selected_mdl_penalty_beta"] = selected_candidate.get("mdl_penalty_beta")
    return summary


def _build_phase1_or_c_summary(
    rows: List[Dict[str, Any]],
    *,
    phase: str,
    reference_fullscale_table: Path | None,
) -> Dict[str, Any]:
    label_to_accs: Dict[str, List[float]] = {}
    for row in rows:
        label = str(row["label"])
        label_to_accs.setdefault(label, [])
        test_acc = _as_float(row.get("test_acc"))
        if test_acc is not None:
            label_to_accs[label].append(float(test_acc))
    avg_test_acc_by_label = {
        label: float(sum(vals) / len(vals))
        for label, vals in label_to_accs.items()
        if vals
    }
    summary: Dict[str, Any] = {
        "phase": phase,
        "n_rows": int(len(rows)),
        "avg_test_acc_by_label": avg_test_acc_by_label,
    }

    winners: Dict[str, str] = {}
    winner_counts: Dict[str, int] = {}
    datasets = sorted({str(row["dataset"]) for row in rows})
    for dataset in datasets:
        candidate_rows = [row for row in rows if str(row["dataset"]) == dataset]
        best_label = None
        best_acc = None
        for row in candidate_rows:
            test_acc = _as_float(row.get("test_acc"))
            if test_acc is None:
                continue
            if best_acc is None or float(test_acc) > float(best_acc):
                best_acc = float(test_acc)
                best_label = str(row["label"])
        if best_label is not None:
            winners[dataset] = best_label
            winner_counts[best_label] = winner_counts.get(best_label, 0) + 1
    summary["winner_counts"] = winner_counts
    summary["winners_by_dataset"] = winners

    if phase == "phaseC_fullscale_refined":
        refined_label = "center_prob_tangent_v3_refined"
        refined_rows = [row for row in rows if str(row.get("label")) == refined_label]
        current_v3_reference = _load_reference_table(reference_fullscale_table, label="center_prob_tangent_v3")
        if refined_rows:
            by_dataset = {str(row["dataset"]).lower(): row for row in refined_rows}
            helpful_x: List[float] = []
            harmful_x: List[float] = []
            delta_x: List[float] = []
            posterior_y: List[float] = []
            for dataset, row in by_dataset.items():
                posterior = _as_float(row.get("posterior_confidence_mean"))
                helpful = _as_float(row.get("helpful_override_rate"))
                harmful = _as_float(row.get("harmful_override_rate"))
                final_acc = _as_float(row.get("final_acc"))
                local_acc = _as_float(row.get("local_acc"))
                if posterior is not None and helpful is not None:
                    posterior_y.append(posterior)
                    helpful_x.append(helpful)
                if posterior is not None and harmful is not None:
                    harmful_x.append(harmful)
                if posterior is not None and final_acc is not None and local_acc is not None:
                    delta_x.append(final_acc - local_acc)
            posterior_vals = [_as_float(row.get("posterior_confidence_mean")) for row in refined_rows]
            helpful_vals = [_as_float(row.get("helpful_override_rate")) for row in refined_rows]
            harmful_vals = [_as_float(row.get("harmful_override_rate")) for row in refined_rows]
            delta_vals = [
                ((_as_float(row.get("final_acc")) or 0.0) - (_as_float(row.get("local_acc")) or 0.0))
                if _as_float(row.get("final_acc")) is not None and _as_float(row.get("local_acc")) is not None
                else None
                for row in refined_rows
            ]
            paired = [
                (
                    _as_float(row.get("posterior_confidence_mean")),
                    _as_float(row.get("helpful_override_rate")),
                    _as_float(row.get("harmful_override_rate")),
                    ((_as_float(row.get("final_acc")) or 0.0) - (_as_float(row.get("local_acc")) or 0.0))
                    if _as_float(row.get("final_acc")) is not None and _as_float(row.get("local_acc")) is not None
                    else None,
                )
                for row in refined_rows
            ]
            helpful_pairs = [(p, h) for p, h, _, _ in paired if p is not None and h is not None]
            harmful_pairs = [(p, h) for p, _, h, _ in paired if p is not None and h is not None]
            delta_pairs = [(p, d) for p, _, _, d in paired if p is not None and d is not None]
            helpful_rho = _spearman([p for p, _ in helpful_pairs], [h for _, h in helpful_pairs]) if helpful_pairs else None
            harmful_rho = _spearman([p for p, _ in harmful_pairs], [h for _, h in harmful_pairs]) if harmful_pairs else None
            delta_rho = _spearman([p for p, _ in delta_pairs], [d for _, d in delta_pairs]) if delta_pairs else None
            corr_checks = [
                {"metric": "helpful_override_rate", "rho": helpful_rho, "sign_ok": helpful_rho is not None and helpful_rho > 0.0},
                {"metric": "harmful_override_rate", "rho": harmful_rho, "sign_ok": harmful_rho is not None and harmful_rho < 0.0},
                {"metric": "final_minus_local_acc", "rho": delta_rho, "sign_ok": delta_rho is not None and delta_rho > 0.0},
            ]
            strong_behavioral = sum(1 for item in corr_checks if item["sign_ok"] and item["rho"] is not None and abs(float(item["rho"])) >= 0.20)
            weak_behavioral = sum(1 for item in corr_checks if item["sign_ok"])
            summary["posterior_behavior_spearman"] = corr_checks
            summary["posterior_structural_veto_triggered"] = bool(strong_behavioral < 2)
            summary["posterior_behavioral_trend_emerged_but_not_yet_strong"] = bool(strong_behavioral < 2 and weak_behavioral >= 2)

            smooth = {"natops", "epilepsy"}
            noisy = {"fingermovements", "selfregulationscp1"}
            smooth_conf = [_as_float(by_dataset[d].get("posterior_confidence_mean")) for d in smooth if d in by_dataset]
            noisy_conf = [_as_float(by_dataset[d].get("posterior_confidence_mean")) for d in noisy if d in by_dataset]
            smooth_conf = [float(v) for v in smooth_conf if v is not None]
            noisy_conf = [float(v) for v in noisy_conf if v is not None]
            summary["posterior_direction_ok"] = bool(smooth_conf and noisy_conf and (_mean(smooth_conf) or 0.0) > (_mean(noisy_conf) or 0.0))

            if current_v3_reference:
                summary["fm_gain_vs_current_v3"] = (
                    (_row_metric(by_dataset.get("fingermovements", {}), "final_acc") or 0.0)
                    - (_row_metric(current_v3_reference.get("fingermovements", {}), "final_acc") or 0.0)
                )
                summary["scp1_gain_vs_current_v3"] = (
                    (_row_metric(by_dataset.get("selfregulationscp1", {}), "final_acc") or 0.0)
                    - (_row_metric(current_v3_reference.get("selfregulationscp1", {}), "final_acc") or 0.0)
                )
    return summary


def _build_phase_summary(
    rows: List[Dict[str, Any]],
    *,
    phase: str,
    reference_phase05_table: Path | None,
    reference_fullscale_table: Path | None,
) -> Dict[str, Any]:
    if phase == "phase05_short_formal":
        smooth = {"natops", "epilepsy"}
        noisy = {"fingermovements", "selfregulationscp1"}
        v3_rows = [row for row in rows if str(row["label"]) == "center_prob_tangent_v3"]
        tangent_rows = {str(row["dataset"]): row for row in rows if str(row["label"]) == "center_tangent_v1"}
        smooth_conf = [_as_float(row.get("posterior_confidence_mean")) for row in v3_rows if str(row["dataset"]) in smooth]
        noisy_conf = [_as_float(row.get("posterior_confidence_mean")) for row in v3_rows if str(row["dataset"]) in noisy]
        smooth_conf = [float(v) for v in smooth_conf if v is not None]
        noisy_conf = [float(v) for v in noisy_conf if v is not None]
        posterior_direction_ok = bool(smooth_conf and noisy_conf and (sum(smooth_conf) / len(smooth_conf) > sum(noisy_conf) / len(noisy_conf)))
        rescue_ok = False
        for dataset in noisy:
            v3 = next((row for row in v3_rows if str(row["dataset"]) == dataset), None)
            tangent = tangent_rows.get(dataset)
            if v3 is None or tangent is None:
                continue
            v3_final = _as_float(v3.get("final_acc"))
            tangent_final = _as_float(tangent.get("final_acc"))
            v3_harm = _as_float(v3.get("harmful_override_rate"))
            tangent_harm = _as_float(tangent.get("harmful_override_rate"))
            if (
                (v3_final is not None and tangent_final is not None and v3_final > tangent_final)
                or (v3_harm is not None and tangent_harm is not None and v3_harm < tangent_harm)
            ):
                rescue_ok = True
                break
        mean_selected_rank_v3 = _group_mean(rows, label="center_prob_tangent_v3", key="mean_selected_rank")
        k0_fallback_v3 = _group_mean(rows, label="center_prob_tangent_v3", key="k0_fallback_rate")
        not_all_k0 = bool(
            mean_selected_rank_v3 is not None
            and k0_fallback_v3 is not None
            and mean_selected_rank_v3 > 0.5
            and k0_fallback_v3 < 0.95
        )
        return {
            "phase": phase,
            "n_rows": int(len(rows)),
            "avg_test_acc_by_label": {
                label: _group_mean(rows, label=label, key="test_acc")
                for label in sorted({str(row["label"]) for row in rows})
            },
            "posterior_direction_ok": bool(posterior_direction_ok),
            "rescue_ok": bool(rescue_ok),
            "mean_selected_rank_v3": mean_selected_rank_v3,
            "k0_fallback_rate_v3": k0_fallback_v3,
            "phase1_gate_passed": bool(posterior_direction_ok and rescue_ok and not_all_k0),
        }
    if phase == "phase1_fullscale":
        return _build_phase1_or_c_summary(rows, phase=phase, reference_fullscale_table=reference_fullscale_table)
    if phase == "phaseA_probe":
        return _build_phase_a_summary(rows)
    if phase.startswith("phaseB_short_formal_refined"):
        return _build_phase_b_refined_summary(rows, reference_phase05_table=reference_phase05_table)
    if phase == "phaseC_fullscale_refined":
        return _build_phase1_or_c_summary(rows, phase=phase, reference_fullscale_table=reference_fullscale_table)
    return {
        "phase": phase,
        "n_rows": int(len(rows)),
        "avg_test_acc_by_label": {
            label: _group_mean(rows, label=label, key="test_acc")
            for label in sorted({str(row["label"]) for row in rows})
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize center_prob_tangent phase results.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--e0-reference", type=str, default="")
    parser.add_argument("--reference-phase05-table", type=str, default="")
    parser.add_argument("--reference-fullscale-table", type=str, default="")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = _read_json(manifest_path)
    phase = str(manifest["phase"])
    out_root = Path(manifest["out_root"]).resolve()
    e0_reference = _load_e0_reference(Path(args.e0_reference).resolve() if args.e0_reference else None)
    reference_phase05_table = Path(args.reference_phase05_table).resolve() if args.reference_phase05_table else None
    reference_fullscale_table = Path(args.reference_fullscale_table).resolve() if args.reference_fullscale_table else None

    rows: List[Dict[str, Any]] = []
    for item in manifest.get("conditions", []):
        run_dir_raw = item.get("resolved_run_dir")
        if not run_dir_raw:
            continue
        run_dir = Path(run_dir_raw).resolve()
        summary_path = run_dir / "summary.json"
        meta_path = run_dir / "run_meta.json"
        dataflow_path = run_dir / "dataflow_agreement_summary.json"
        if not summary_path.is_file() or not meta_path.is_file():
            continue
        summary = _read_json(summary_path)
        meta = _read_json(meta_path)
        dataflow = _read_json(dataflow_path) if dataflow_path.is_file() else {}
        dataset = str(item["dataset"]).lower()
        row: Dict[str, Any] = {
            "dataset": dataset,
            "label": str(item["label"]),
            "prototype_geometry_mode": meta.get("prototype_geometry_mode"),
            "prob_tangent_version": meta.get("prob_tangent_version"),
            "rank_selection_mode": meta.get("rank_selection_mode"),
            "posterior_mode": meta.get("posterior_mode"),
            "posterior_student_dof": meta.get("posterior_student_dof"),
            "mdl_penalty_beta": meta.get("mdl_penalty_beta"),
            "subproto_temperature": meta.get("subproto_temperature"),
            "test_acc": summary.get("test_acc"),
            "test_macro_f1": summary.get("test_macro_f1"),
            "final_acc": dataflow.get("final_acc"),
            "local_acc": dataflow.get("local_acc"),
            "final_override_rate": dataflow.get("final_override_rate"),
            "helpful_override_rate": dataflow.get("helpful_override_rate"),
            "harmful_override_rate": dataflow.get("harmful_override_rate"),
            "selected_rank_distribution": json.dumps(dataflow.get("selected_rank_distribution", {}), ensure_ascii=False),
            "mean_selected_rank": dataflow.get("mean_selected_rank"),
            "k0_fallback_rate": dataflow.get("k0_fallback_rate"),
            "lw_shrinkage_alpha_mean": dataflow.get("lw_shrinkage_alpha_mean"),
            "ppca_sigma2_mean": dataflow.get("ppca_sigma2_mean"),
            "posterior_confidence_mean": dataflow.get("posterior_confidence_mean"),
            "posterior_confidence_std": dataflow.get("posterior_confidence_std"),
            "posterior_confidence_q10": dataflow.get("posterior_confidence_q10"),
            "posterior_confidence_q50": dataflow.get("posterior_confidence_q50"),
            "posterior_confidence_q90": dataflow.get("posterior_confidence_q90"),
            "posterior_confidence_qgap": dataflow.get("posterior_confidence_qgap"),
            "posterior_confidence_qratio": dataflow.get("posterior_confidence_qratio"),
            "posterior_confidence_far_decay_mean": dataflow.get("posterior_confidence_far_decay_mean"),
            "posterior_log_confidence_mean": dataflow.get("posterior_log_confidence_mean"),
            "posterior_log_confidence_std": dataflow.get("posterior_log_confidence_std"),
            "posterior_residual_energy_mean": dataflow.get("posterior_residual_energy_mean"),
            "posterior_residual_energy_per_dim_mean": dataflow.get("posterior_residual_energy_per_dim_mean"),
            "posterior_sigma2_eff_mean": dataflow.get("posterior_sigma2_eff_mean"),
            "run_dir": str(run_dir),
            "E0_acc": e0_reference.get(dataset),
        }
        rows.append(row)

    if phase == "phase05_short_formal":
        csv_path = out_root / "center_prob_tangent_phase05_table.csv"
        summary_path = out_root / "center_prob_tangent_phase05_summary.json"
    elif phase == "phase1_fullscale":
        csv_path = out_root / "center_prob_tangent_fullscale_table.csv"
        summary_path = out_root / "center_prob_tangent_fullscale_summary.json"
    else:
        csv_path = out_root / f"center_prob_tangent_{phase}_table.csv"
        summary_path = out_root / f"center_prob_tangent_{phase}_summary.json"

    _write_csv(csv_path, rows)
    summary_payload = {
        "manifest": str(manifest_path),
        "phase": phase,
        "comparison_csv": str(csv_path),
        **_build_phase_summary(
            rows,
            phase=phase,
            reference_phase05_table=reference_phase05_table,
            reference_fullscale_table=reference_fullscale_table,
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print(f"[done] comparison -> {csv_path}")
    print(f"[done] summary -> {summary_path}")


if __name__ == "__main__":
    main()
