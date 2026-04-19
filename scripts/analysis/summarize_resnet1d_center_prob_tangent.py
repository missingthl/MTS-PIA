#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

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


def _group_mean(rows: List[Dict[str, Any]], *, label: str, key: str) -> float | None:
    values = [_as_float(row.get(key)) for row in rows if str(row.get("label")) == label]
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _build_phase_summary(rows: List[Dict[str, Any]], *, phase: str) -> Dict[str, Any]:
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
        "phase": str(phase),
        "n_rows": int(len(rows)),
        "avg_test_acc_by_label": avg_test_acc_by_label,
    }

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
        summary.update(
            {
                "posterior_direction_ok": bool(posterior_direction_ok),
                "rescue_ok": bool(rescue_ok),
                "mean_selected_rank_v3": mean_selected_rank_v3,
                "k0_fallback_rate_v3": k0_fallback_v3,
                "phase1_gate_passed": bool(posterior_direction_ok and rescue_ok and not_all_k0),
            }
        )

    if phase == "phase1_fullscale":
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

    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize center_prob_tangent phase results.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--e0-reference", type=str, default="")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = _read_json(manifest_path)
    phase = str(manifest["phase"])
    out_root = Path(manifest["out_root"]).resolve()
    e0_reference = _load_e0_reference(Path(args.e0_reference).resolve() if args.e0_reference else None)

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
            "posterior_confidence_far_decay_mean": dataflow.get("posterior_confidence_far_decay_mean"),
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
        **_build_phase_summary(rows, phase=phase),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print(f"[done] comparison -> {csv_path}")
    print(f"[done] summary -> {summary_path}")


if __name__ == "__main__":
    main()
