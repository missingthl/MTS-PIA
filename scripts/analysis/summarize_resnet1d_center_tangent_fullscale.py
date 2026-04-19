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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize center_tangent fullscale supplement against existing DLCR matrix.")
    parser.add_argument("--baseline-results", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    baseline_path = Path(args.baseline_results).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = _read_json(manifest_path)
    baseline_rows = {row["dataset"]: row for row in _read_csv(baseline_path)}

    rows: List[Dict[str, Any]] = []
    for item in manifest.get("conditions", []):
        run_dir = item.get("resolved_run_dir")
        if not run_dir:
            continue
        dataset = str(item["dataset"])
        summary_path = Path(run_dir) / "summary.json"
        if not summary_path.is_file():
            continue
        tangent_acc = float(_read_json(summary_path)["test_acc"])
        base = baseline_rows.get(dataset, {})
        e0_acc = float(base["E0_acc"]) if base.get("E0_acc") else None
        center_only_acc = float(base["center_only_acc"]) if base.get("center_only_acc") else None
        best_subproto_acc = float(base["best_subproto_acc"]) if base.get("best_subproto_acc") else None
        best_subproto_tau = base.get("best_subproto_tau")

        candidates = {
            "center_only": center_only_acc,
            "best_center_subproto": best_subproto_acc,
            "center_tangent_rank4": tangent_acc,
        }
        winner = None
        winner_acc = None
        for label, acc in candidates.items():
            if acc is None:
                continue
            if winner_acc is None or float(acc) > float(winner_acc):
                winner = label
                winner_acc = float(acc)

        rows.append(
            {
                "dataset": dataset,
                "E0_acc": e0_acc,
                "center_only_acc": center_only_acc,
                "best_subproto_tau": best_subproto_tau,
                "best_center_subproto_acc": best_subproto_acc,
                "center_tangent_rank4_acc": tangent_acc,
                "delta_tangent_vs_E0": None if e0_acc is None else tangent_acc - e0_acc,
                "delta_tangent_vs_center_only": None if center_only_acc is None else tangent_acc - center_only_acc,
                "delta_tangent_vs_best_subproto": None if best_subproto_acc is None else tangent_acc - best_subproto_acc,
                "winner_among_three": winner,
                "winner_acc": winner_acc,
            }
        )

    out_root = Path(manifest["out_root"]).resolve()
    csv_path = out_root / "center_tangent_comparison_table.csv"
    summary_path = out_root / "center_tangent_comparison_summary.json"
    _write_csv(csv_path, rows)

    counts: Dict[str, int] = {}
    for row in rows:
        winner = str(row["winner_among_three"])
        counts[winner] = counts.get(winner, 0) + 1

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline_results": str(baseline_path),
                "manifest": str(manifest_path),
                "comparison_csv": str(csv_path),
                "n_datasets": len(rows),
                "winner_counts": counts,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[done] comparison -> {csv_path}")
    print(f"[done] summary -> {summary_path}")


if __name__ == "__main__":
    main()
