#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def _iter_manifest_conditions(manifest: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for item in manifest.get("conditions", []):
        resolved = item.get("resolved_run_dir")
        if not resolved:
            continue
        yield item


def _load_run_payload(run_dir: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_meta": _read_json(run_dir / "run_meta.json"),
        "summary": _read_json(run_dir / "summary.json"),
    }
    agreement_path = run_dir / "dataflow_agreement_summary.json"
    if agreement_path.is_file():
        payload["agreement"] = _read_json(agreement_path)
    else:
        payload["agreement"] = {}
    return payload


def _condition_key(entry: Dict[str, Any]) -> str:
    label = str(entry["label"])
    if label == "E0":
        return "E0"
    if label == "center_only":
        return "center_only"
    if label.startswith("center_subproto_tau"):
        return label
    if label.startswith("center_tangent_rank"):
        return label
    raise ValueError(f"unknown condition label: {label}")


def _extract_acc(payload: Dict[str, Any]) -> float | None:
    summary = payload.get("summary", {})
    if "test_acc" not in summary:
        return None
    return float(summary["test_acc"])


def _extract_mechanism_row(entry: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    meta = payload["run_meta"]
    agreement = payload.get("agreement", {})
    row: Dict[str, Any] = {
        "dataset": str(entry["dataset"]),
        "label": str(entry["label"]),
        "arm": str(entry["arm"]),
        "prototype_geometry_mode": meta.get("prototype_geometry_mode"),
        "subproto_temperature": meta.get("subproto_temperature"),
        "tangent_rank": meta.get("tangent_rank"),
        "tangent_source": meta.get("tangent_source"),
        "run_dir": payload["run_dir"],
        "local_acc": agreement.get("local_acc"),
        "final_acc": agreement.get("final_acc", payload["summary"].get("test_acc")),
        "final_override_rate": agreement.get("final_override_rate"),
        "helpful_override_rate": agreement.get("helpful_override_rate"),
        "harmful_override_rate": agreement.get("harmful_override_rate"),
        "subproto_weight_entropy_mean": agreement.get("subproto_weight_entropy_mean"),
        "same_weight_max_mean": agreement.get("same_weight_max_mean"),
        "subproto_cos_top1_top2_gap_mean": agreement.get("subproto_cos_top1_top2_gap_mean"),
        "subproto_top1_occupancy_entropy": agreement.get("subproto_top1_occupancy_entropy"),
        "subproto_usage_effective_count": agreement.get("subproto_usage_effective_count"),
        "subproto_pairwise_cos_mean": agreement.get("learned_subproto_pairwise_cos_mean"),
    }
    return row


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize ResNet1D + DLCR behavioral matrix outputs.")
    parser.add_argument("--manifest", type=str, required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = _read_json(manifest_path)
    out_root = Path(manifest["out_root"]).resolve()

    dataset_result_rows: Dict[str, Dict[str, Any]] = {}
    mechanism_rows: List[Dict[str, Any]] = []

    for entry in _iter_manifest_conditions(manifest):
        run_dir = Path(entry["resolved_run_dir"]).resolve()
        payload = _load_run_payload(run_dir)
        dataset = str(entry["dataset"])
        condition_key = _condition_key(entry)
        if dataset not in dataset_result_rows:
            dataset_result_rows[dataset] = {"dataset": dataset}
        dataset_result_rows[dataset][f"{condition_key}_acc"] = _extract_acc(payload)
        mechanism_rows.append(_extract_mechanism_row(entry, payload))

    result_rows: List[Dict[str, Any]] = []
    for dataset in sorted(dataset_result_rows.keys()):
        row = dataset_result_rows[dataset]
        e0_acc = row.get("E0_acc")
        center_only_acc = row.get("center_only_acc")
        tau_candidates = {
            "1.0": row.get("center_subproto_tau1.0_acc"),
            "0.5": row.get("center_subproto_tau0.5_acc"),
            "0.2": row.get("center_subproto_tau0.2_acc"),
            "0.1": row.get("center_subproto_tau0.1_acc"),
        }
        best_tau = None
        best_acc = None
        for tau, acc in tau_candidates.items():
            if acc is None:
                continue
            if best_acc is None or float(acc) > float(best_acc):
                best_acc = float(acc)
                best_tau = tau
        row["best_subproto_tau"] = best_tau
        row["best_subproto_acc"] = best_acc
        row["best_subproto_delta_vs_E0"] = None if best_acc is None or e0_acc is None else float(best_acc) - float(e0_acc)
        row["best_subproto_delta_vs_center_only"] = (
            None if best_acc is None or center_only_acc is None else float(best_acc) - float(center_only_acc)
        )
        result_rows.append(row)

    results_csv = out_root / "behavioral_results_table.csv"
    mechanism_csv = out_root / "behavioral_mechanism_table.csv"
    summary_json = out_root / "behavioral_summary.json"

    _write_csv(results_csv, result_rows)
    _write_csv(mechanism_csv, mechanism_rows)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "manifest": str(manifest_path),
                "results_csv": str(results_csv),
                "mechanism_csv": str(mechanism_csv),
                "n_datasets": int(len(result_rows)),
                "n_conditions": int(len(list(_iter_manifest_conditions(manifest)))),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[done] results -> {results_csv}")
    print(f"[done] mechanism -> {mechanism_csv}")


if __name__ == "__main__":
    main()
