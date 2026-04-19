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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize ResNet1D + DLCR tangent probe outputs.")
    parser.add_argument("--manifest", type=str, required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = _read_json(manifest_path)
    out_root = Path(manifest["out_root"]).resolve()

    dataset_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []

    for entry in _iter_manifest_conditions(manifest):
        run_dir = Path(entry["resolved_run_dir"]).resolve()
        run_meta = _read_json(run_dir / "run_meta.json")
        probe_summary = _read_json(run_dir / "tangent_probe_summary.json")
        probe_full = _read_json(run_dir / "tangent_probe_full.json")
        dataset_rows.append(
            {
                "dataset": run_meta.get("dataset"),
                "prototype_geometry_mode": run_meta.get("prototype_geometry_mode"),
                "subproto_temperature": run_meta.get("subproto_temperature"),
                "tangent_rank": run_meta.get("tangent_rank"),
                "tangent_source": run_meta.get("tangent_source"),
                **probe_summary,
                "run_dir": str(run_dir),
            }
        )
        for row in probe_full.get("class_rows", []):
            class_rows.append(
                {
                    "dataset": run_meta.get("dataset"),
                    "prototype_geometry_mode": run_meta.get("prototype_geometry_mode"),
                    **row,
                    "run_dir": str(run_dir),
                }
            )

    recommended_union = sorted(
        {
            int(rank)
            for row in dataset_rows
            for rank in row.get("recommended_candidate_ranks", [])
        }
    )

    dataset_csv = out_root / "tangent_probe_dataset_table.csv"
    class_csv = out_root / "tangent_probe_class_table.csv"
    summary_json = out_root / "tangent_probe_summary_aggregate.json"

    _write_csv(dataset_csv, dataset_rows)
    _write_csv(class_csv, class_rows)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "manifest": str(manifest_path),
                "dataset_csv": str(dataset_csv),
                "class_csv": str(class_csv),
                "n_datasets": int(len(dataset_rows)),
                "n_class_rows": int(len(class_rows)),
                "recommended_candidate_ranks_union": recommended_union,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[done] dataset table -> {dataset_csv}")
    print(f"[done] class table -> {class_csv}")


if __name__ == "__main__":
    main()
