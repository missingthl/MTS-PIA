#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "hosts" / "run_resnet1d_local_closed_form_fixedsplit.py"
DEFAULT_PIA_PYTHON = Path("/home/THL/miniconda3/envs/pia/bin/python")
DEFAULT_DATASETS = [
    "har",
    "natops",
    "fingermovements",
    "selfregulationscp1",
    "basicmotions",
    "handmovementdirection",
    "uwavegesturelibrary",
    "epilepsy",
    "atrialfibrillation",
    "pendigits",
    "racketsports",
    "articularywordrecognition",
    "heartbeat",
    "selfregulationscp2",
    "libras",
    "japanesevowels",
    "cricket",
    "handwriting",
    "ering",
    "motorimagery",
    "ethanolconcentration",
]
REQUIRED_MATCH_FIELDS = [
    "dataset",
    "arm",
    "split_protocol",
    "runner_protocol",
    "seed",
    "host_backbone",
    "epochs",
    "train_batch_size",
    "test_batch_size",
    "closed_form_solve_mode",
    "prototype_geometry_mode",
    "local_support_mode",
    "prototype_aggregation",
    "routing_temperature",
    "class_prior_temperature",
    "subproto_temperature",
    "tangent_rank",
    "tangent_source",
    "tangent_probe",
]


@dataclass
class ProbeCondition:
    dataset: str
    label: str
    expected_meta: Dict[str, Any]
    cli_args: List[str]
    target_run_dir: str


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_dataset_list(raw: str | None) -> List[str]:
    if not raw:
        return list(DEFAULT_DATASETS)
    values = [item.strip().lower() for item in raw.split(",")]
    return [item for item in values if item]


def _values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return abs(float(expected) - float(actual)) <= 1e-12
        except (TypeError, ValueError):
            return False
    return expected == actual


def _is_candidate_complete(run_dir: Path) -> bool:
    return (
        (run_dir / "run_meta.json").is_file()
        and (run_dir / "summary.json").is_file()
        and (run_dir / "tangent_probe_summary.json").is_file()
        and (run_dir / "tangent_probe_full.json").is_file()
    )


def _candidate_matches(expected_meta: Dict[str, Any], run_dir: Path) -> tuple[bool, str]:
    if not _is_candidate_complete(run_dir):
        return False, "missing_required_artifacts"
    try:
        actual_meta = _read_json(run_dir / "run_meta.json")
    except Exception:
        return False, "broken_run_meta"
    for field in REQUIRED_MATCH_FIELDS:
        if field not in actual_meta:
            return False, f"missing_field:{field}"
        if field not in expected_meta:
            return False, f"missing_expected_field:{field}"
        if not _values_match(expected_meta[field], actual_meta[field]):
            return False, f"mismatch:{field}"
    return True, "matched"


def _iter_candidate_run_dirs(search_roots: Sequence[Path]) -> Iterable[Path]:
    seen: set[str] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for meta_path in root.rglob("run_meta.json"):
            run_dir = meta_path.parent.resolve()
            key = str(run_dir)
            if key in seen:
                continue
            seen.add(key)
            yield run_dir


def _find_reusable_run(*, expected_meta: Dict[str, Any], search_roots: Sequence[Path]) -> tuple[str | None, str]:
    for run_dir in _iter_candidate_run_dirs(search_roots):
        matched, reason = _candidate_matches(expected_meta, run_dir)
        if matched:
            return str(run_dir), "matched_existing_run"
    return None, "no_compatible_existing_run"


def _build_condition(
    dataset: str,
    *,
    out_root: Path,
    seed: int,
    epochs: int,
    subproto_temperature: float,
    tangent_rank: int,
) -> ProbeCondition:
    tau_tag = str(subproto_temperature).replace(".", "p")
    run_tag = f"stage2_resnet1d_e2_center_subproto_tangent_probe_tau{tau_tag}_rank{int(tangent_rank)}_{dataset}_seed{seed}"
    target_run_dir = out_root / "e2" / run_tag
    expected_meta = {
        "dataset": dataset,
        "arm": "e2",
        "split_protocol": "fixedsplit",
        "runner_protocol": "resnet1d_local_closed_form_fixedsplit",
        "seed": int(seed),
        "host_backbone": "ResNet1D",
        "epochs": int(epochs),
        "train_batch_size": 64,
        "test_batch_size": 128,
        "closed_form_solve_mode": "pinv",
        "prototype_geometry_mode": "center_subproto",
        "local_support_mode": "same_only",
        "prototype_aggregation": "pooled",
        "routing_temperature": 1.0,
        "class_prior_temperature": 1.0,
        "subproto_temperature": float(subproto_temperature),
        "tangent_rank": int(tangent_rank),
        "tangent_source": "subproto_offsets",
        "tangent_probe": True,
    }
    cli_args = [
        "--dataset",
        dataset,
        "--arm",
        "e2",
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--train-batch-size",
        "64",
        "--test-batch-size",
        "128",
        "--num-workers",
        "0",
        "--closed-form-solve-mode",
        "pinv",
        "--prototype-geometry-mode",
        "center_subproto",
        "--local-support-mode",
        "same_only",
        "--prototype-aggregation",
        "pooled",
        "--routing-temperature",
        "1.0",
        "--class-prior-temperature",
        "1.0",
        "--subproto-temperature",
        str(float(subproto_temperature)),
        "--tangent-rank",
        str(int(tangent_rank)),
        "--tangent-source",
        "subproto_offsets",
        "--dataflow-probe",
        "--tangent-probe",
        "--out-root",
        str(out_root),
        "--run-tag",
        run_tag,
    ]
    return ProbeCondition(
        dataset=dataset,
        label=f"center_subproto_tangent_probe_tau{subproto_temperature:.1f}_rank{int(tangent_rank)}",
        expected_meta=expected_meta,
        cli_args=cli_args,
        target_run_dir=str(target_run_dir),
    )


def _remove_path_if_exists(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _execute_condition(*, condition: ProbeCondition, python_bin: str, device: str) -> None:
    target_dir = Path(condition.target_run_dir)
    if target_dir.exists():
        _remove_path_if_exists(target_dir)
    command = [python_bin, str(RUNNER), *condition.cli_args, "--device", device]
    subprocess.run(command, cwd=str(ROOT), check=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a 21-dataset tangent probe matrix for ResNet1D + DLCR.")
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(ROOT / "out" / "_active" / f"verify_resnet1d_tangent_probe_{time.strftime('%Y%m%d')}"),
    )
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--subproto-temperature", type=float, default=1.0)
    parser.add_argument("--tangent-rank", type=int, default=3)
    parser.add_argument(
        "--python-bin",
        type=str,
        default=str(DEFAULT_PIA_PYTHON if DEFAULT_PIA_PYTHON.is_file() else Path(sys.executable).resolve()),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--search-roots", type=str, nargs="*", default=[str(ROOT / "out" / "_active")])
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--force-rerun", action="store_true", default=False)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    out_root = Path(args.out_root).resolve()
    _ensure_dir(out_root)
    search_roots = [Path(p).resolve() for p in args.search_roots]
    datasets = _parse_dataset_list(args.datasets)
    conditions = [
        _build_condition(
            dataset,
            out_root=out_root,
            seed=args.seed,
            epochs=args.epochs,
            subproto_temperature=float(args.subproto_temperature),
            tangent_rank=int(args.tangent_rank),
        )
        for dataset in datasets
    ]

    manifest: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_root": str(out_root),
        "runner": str(RUNNER),
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
        status = "pending"
        resolved_run_dir = None
        reason = ""
        if not args.force_rerun:
            reusable_run, reason = _find_reusable_run(
                expected_meta=condition.expected_meta,
                search_roots=search_roots,
            )
            if reusable_run is not None:
                status = "reused"
                resolved_run_dir = reusable_run
        if status == "pending":
            status = "run_required"
            reason = reason or "fresh_run_required"
            resolved_run_dir = str(target_dir)

        record = asdict(condition)
        record.update(
            {
                "index": int(index),
                "status": status,
                "resolved_run_dir": resolved_run_dir,
                "decision_reason": reason,
            }
        )
        manifest["conditions"].append(record)
        print(
            f"[{index:02d}/{len(conditions)}] {condition.dataset} :: {condition.label} -> {status} ({reason})",
            flush=True,
        )

        if status == "run_required" and not args.dry_run:
            _execute_condition(condition=condition, python_bin=str(args.python_bin), device=str(args.device))

    manifest_path = out_root / "tangent_probe_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"[done] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
