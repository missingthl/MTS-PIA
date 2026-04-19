#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(ROOT_BOOTSTRAP))

from scripts.hosts.run_resnet1d_dlcr_behavioral_matrix import (
    DEFAULT_DATASETS,
    DEFAULT_PIA_PYTHON,
    REQUIRED_MATCH_FIELDS,
    MatrixCondition,
    _build_e2_condition,
    _ensure_dir,
    _execute_condition,
    _find_reusable_run,
    _parse_dataset_list,
    _write_json,
    ROOT,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full-scale ResNet1D center_tangent supplement over fixed-split datasets."
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(ROOT / "out" / "_active" / f"verify_resnet1d_center_tangent_fullscale_{time.strftime('%Y%m%d')}"),
    )
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--tangent-rank", type=int, default=4)
    parser.add_argument(
        "--python-bin",
        type=str,
        default=str(DEFAULT_PIA_PYTHON if DEFAULT_PIA_PYTHON.is_file() else Path.cwd() / "python"),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--search-roots",
        type=str,
        nargs="*",
        default=[str(ROOT / "out" / "_active")],
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--force-rerun", action="store_true", default=False)
    return parser


def _build_conditions(args: argparse.Namespace) -> List[MatrixCondition]:
    out_root = Path(args.out_root).resolve()
    datasets = _parse_dataset_list(args.datasets) if args.datasets else list(DEFAULT_DATASETS)
    conditions: List[MatrixCondition] = []
    for dataset in datasets:
        conditions.append(
            _build_e2_condition(
                dataset,
                out_root=out_root,
                seed=int(args.seed),
                e2_epochs=int(args.epochs),
                geometry="center_tangent",
                subproto_temperature=1.0,
                tangent_rank=int(args.tangent_rank),
                tangent_probe=False,
            )
        )
    return conditions


def main() -> None:
    args = build_argparser().parse_args()
    out_root = Path(args.out_root).resolve()
    _ensure_dir(out_root)
    search_roots = [Path(p).resolve() for p in args.search_roots]
    conditions = _build_conditions(args)

    manifest: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_root": str(out_root),
        "runner": str(ROOT / "scripts" / "hosts" / "run_resnet1d_local_closed_form_fixedsplit.py"),
        "python_bin": str(args.python_bin),
        "device": str(args.device),
        "dry_run": bool(args.dry_run),
        "force_rerun": bool(args.force_rerun),
        "required_match_fields": list(REQUIRED_MATCH_FIELDS),
        "search_roots": [str(p) for p in search_roots],
        "conditions": [],
    }

    for index, condition in enumerate(conditions, start=1):
        target_dir = Path(condition.target_run_dir)
        source_run_dir = None
        reason = ""
        if not args.force_rerun:
            source_run_dir, reason = _find_reusable_run(
                expected_meta=condition.expected_meta,
                search_roots=search_roots,
                require_dataflow=True,
            )
        status = "reused" if source_run_dir is not None else "run_required"
        manifest["conditions"].append(
            {
                "index": int(index),
                "dataset": condition.dataset,
                "label": condition.label,
                "arm": condition.arm,
                "status": status,
                "reason": reason,
                "target_run_dir": str(target_dir),
                "resolved_run_dir": source_run_dir,
                "expected_meta": dict(condition.expected_meta),
                "cli_args": list(condition.cli_args),
            }
        )

    manifest_path = out_root / "center_tangent_manifest.json"
    _write_json(manifest_path, manifest)
    if args.dry_run:
        print(f"[dry-run] wrote manifest -> {manifest_path}")
        return

    for entry, condition in zip(manifest["conditions"], conditions):
        if entry["status"] == "reused":
            print(f"[reuse] {entry['dataset']} / {entry['label']} -> {entry['resolved_run_dir']}", flush=True)
            continue
        print(f"[run] {entry['dataset']} / {entry['label']} -> {entry['target_run_dir']}", flush=True)
        try:
            _execute_condition(condition=condition, python_bin=args.python_bin, device=args.device)
        except Exception as exc:
            entry["status"] = "failed"
            entry["reason"] = f"runner_failed:{exc}"
            entry["resolved_run_dir"] = None
            _write_json(manifest_path, manifest)
            raise
        entry["status"] = "completed"
        entry["reason"] = "executed_runner"
        entry["resolved_run_dir"] = entry["target_run_dir"]
        _write_json(manifest_path, manifest)

    print(f"[done] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
