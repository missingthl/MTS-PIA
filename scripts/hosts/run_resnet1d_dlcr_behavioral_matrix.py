#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
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
    "local_readout_gate",
    "dataflow_probe",
    "routing_temperature",
    "class_prior_temperature",
    "subproto_temperature",
    "tangent_rank",
    "tangent_source",
    "tangent_probe",
]


@dataclass
class MatrixCondition:
    dataset: str
    label: str
    arm: str
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


def _tau_tag(value: float) -> str:
    return str(value).replace(".", "p")


def _values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return abs(float(expected) - float(actual)) <= 1e-12
        except (TypeError, ValueError):
            return False
    return expected == actual


def _is_candidate_complete(run_dir: Path, *, require_dataflow: bool) -> bool:
    if not (run_dir / "run_meta.json").is_file():
        return False
    if not (run_dir / "summary.json").is_file():
        return False
    if require_dataflow and not (run_dir / "dataflow_agreement_summary.json").is_file():
        return False
    return True


def _candidate_matches(expected_meta: Dict[str, Any], run_dir: Path, *, require_dataflow: bool) -> tuple[bool, str]:
    if not _is_candidate_complete(run_dir, require_dataflow=require_dataflow):
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


def _find_reusable_run(
    *,
    expected_meta: Dict[str, Any],
    search_roots: Sequence[Path],
    require_dataflow: bool,
) -> tuple[str | None, str]:
    for run_dir in _iter_candidate_run_dirs(search_roots):
        matched, reason = _candidate_matches(expected_meta, run_dir, require_dataflow=require_dataflow)
        if matched:
            return str(run_dir), "matched_existing_run"
    return None, "no_compatible_existing_run"


def _build_e0_condition(dataset: str, *, out_root: Path, seed: int, e0_epochs: int) -> MatrixCondition:
    run_tag = f"stage2_resnet1d_e0_behavioral_{dataset}_seed{seed}"
    target_run_dir = out_root / "e0" / run_tag
    expected_meta = {
        "dataset": dataset,
        "arm": "e0",
        "split_protocol": "fixedsplit",
        "runner_protocol": "resnet1d_local_closed_form_fixedsplit",
        "seed": int(seed),
        "host_backbone": "ResNet1D",
        "epochs": int(e0_epochs),
        "train_batch_size": 32,
        "test_batch_size": 32,
        "closed_form_solve_mode": "ridge_solve",
        "prototype_geometry_mode": "flat",
        "local_support_mode": "same_only",
        "prototype_aggregation": "pooled",
        "local_readout_gate": "none",
        "dataflow_probe": False,
        "routing_temperature": 1.0,
        "class_prior_temperature": 1.0,
        "subproto_temperature": 1.0,
        "tangent_rank": 0,
        "tangent_source": "subproto_offsets",
        "tangent_probe": False,
    }
    cli_args = [
        "--dataset",
        dataset,
        "--arm",
        "e0",
        "--epochs",
        str(e0_epochs),
        "--seed",
        str(seed),
        "--train-batch-size",
        "32",
        "--test-batch-size",
        "32",
        "--num-workers",
        "0",
        "--closed-form-solve-mode",
        "ridge_solve",
        "--prototype-geometry-mode",
        "flat",
        "--local-support-mode",
        "same_only",
        "--prototype-aggregation",
        "pooled",
        "--routing-temperature",
        "1.0",
        "--class-prior-temperature",
        "1.0",
        "--subproto-temperature",
        "1.0",
        "--tangent-rank",
        "0",
        "--tangent-source",
        "subproto_offsets",
        "--out-root",
        str(out_root),
        "--run-tag",
        run_tag,
    ]
    return MatrixCondition(
        dataset=dataset,
        label="E0",
        arm="e0",
        expected_meta=expected_meta,
        cli_args=cli_args,
        target_run_dir=str(target_run_dir),
    )


def _build_e2_condition(
    dataset: str,
    *,
    out_root: Path,
    seed: int,
    e2_epochs: int,
    geometry: str,
    subproto_temperature: float,
    tangent_rank: int = 0,
    tangent_probe: bool = False,
) -> MatrixCondition:
    if geometry == "center_only":
        label = "center_only"
        tau = 1.0
        tangent_rank = 0
    elif geometry == "center_tangent":
        tau = 1.0
        label = f"center_tangent_rank{int(tangent_rank)}"
    else:
        tau = float(subproto_temperature)
        label = f"center_subproto_tau{tau:.1f}"
    run_tag = f"stage2_resnet1d_e2_{label.replace('.', 'p')}_{dataset}_seed{seed}"
    target_run_dir = out_root / "e2" / run_tag
    expected_meta = {
        "dataset": dataset,
        "arm": "e2",
        "split_protocol": "fixedsplit",
        "runner_protocol": "resnet1d_local_closed_form_fixedsplit",
        "seed": int(seed),
        "host_backbone": "ResNet1D",
        "epochs": int(e2_epochs),
        "train_batch_size": 64,
        "test_batch_size": 128,
        "closed_form_solve_mode": "pinv",
        "prototype_geometry_mode": str(geometry),
        "local_support_mode": "same_only",
        "prototype_aggregation": "pooled",
        "local_readout_gate": "none",
        "dataflow_probe": True,
        "routing_temperature": 1.0,
        "class_prior_temperature": 1.0,
        "subproto_temperature": float(tau),
        "tangent_rank": int(tangent_rank),
        "tangent_source": "subproto_offsets",
        "tangent_probe": bool(tangent_probe),
    }
    cli_args = [
        "--dataset",
        dataset,
        "--arm",
        "e2",
        "--epochs",
        str(e2_epochs),
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
        str(geometry),
        "--local-support-mode",
        "same_only",
        "--prototype-aggregation",
        "pooled",
        "--routing-temperature",
        "1.0",
        "--class-prior-temperature",
        "1.0",
        "--subproto-temperature",
        str(tau),
        "--tangent-rank",
        str(int(tangent_rank)),
        "--tangent-source",
        "subproto_offsets",
        "--dataflow-probe",
        "--out-root",
        str(out_root),
        "--run-tag",
        run_tag,
    ]
    if tangent_probe:
        cli_args.append("--tangent-probe")
    return MatrixCondition(
        dataset=dataset,
        label=label,
        arm="e2",
        expected_meta=expected_meta,
        cli_args=cli_args,
        target_run_dir=str(target_run_dir),
    )


def _build_conditions(args: argparse.Namespace) -> List[MatrixCondition]:
    out_root = Path(args.out_root).resolve()
    datasets = _parse_dataset_list(args.datasets)
    conditions: List[MatrixCondition] = []
    for dataset in datasets:
        conditions.append(_build_e0_condition(dataset, out_root=out_root, seed=args.seed, e0_epochs=args.e0_epochs))
        conditions.append(
            _build_e2_condition(
                dataset,
                out_root=out_root,
                seed=args.seed,
                e2_epochs=args.e2_epochs,
                geometry="center_only",
                subproto_temperature=1.0,
            )
        )
        for tau in (1.0, 0.5, 0.2, 0.1):
            conditions.append(
                _build_e2_condition(
                    dataset,
                    out_root=out_root,
                    seed=args.seed,
                    e2_epochs=args.e2_epochs,
                    geometry="center_subproto",
                    subproto_temperature=tau,
                )
            )
        if bool(args.include_center_tangent):
            for rank in args.tangent_ranks:
                conditions.append(
                    _build_e2_condition(
                        dataset,
                        out_root=out_root,
                        seed=args.seed,
                        e2_epochs=args.e2_epochs,
                        geometry="center_tangent",
                        subproto_temperature=1.0,
                        tangent_rank=int(rank),
                        tangent_probe=False,
                    )
                )
    return conditions


def _remove_path_if_exists(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _execute_condition(
    *,
    condition: MatrixCondition,
    python_bin: str,
    device: str,
) -> None:
    target_dir = Path(condition.target_run_dir)
    if target_dir.exists():
        _remove_path_if_exists(target_dir)
    command = [
        python_bin,
        str(RUNNER),
        *condition.cli_args,
        "--device",
        device,
    ]
    subprocess.run(command, cwd=str(ROOT), check=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ResNet1D + DLCR behavioral matrix orchestrator.")
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(ROOT / "out" / "_active" / f"verify_resnet1d_dlcr_behavioral_matrix_{time.strftime('%Y%m%d')}"),
    )
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--e0-epochs", type=int, default=30)
    parser.add_argument("--e2-epochs", type=int, default=100)
    parser.add_argument(
        "--python-bin",
        type=str,
        default=str(DEFAULT_PIA_PYTHON if DEFAULT_PIA_PYTHON.is_file() else Path(sys.executable).resolve()),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--search-roots",
        type=str,
        nargs="*",
        default=[str(ROOT / "out" / "_active")],
        help="Directories to search for reusable runs.",
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--force-rerun", action="store_true", default=False)
    parser.add_argument("--include-center-tangent", action="store_true", default=False)
    parser.add_argument("--tangent-ranks", type=int, nargs="*", default=[1, 2, 3])
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    out_root = Path(args.out_root).resolve()
    _ensure_dir(out_root)
    search_roots = [Path(p).resolve() for p in args.search_roots]
    conditions = _build_conditions(args)

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
        require_dataflow = condition.arm == "e2"
        status = "pending"
        reason = ""
        source_run_dir = None

        if not args.force_rerun:
            source_run_dir, reason = _find_reusable_run(
                expected_meta=condition.expected_meta,
                search_roots=search_roots,
                require_dataflow=require_dataflow,
            )

        if source_run_dir is not None:
            status = "reused"
        else:
            status = "run_required"

        entry = {
            "index": int(index),
            "dataset": condition.dataset,
            "label": condition.label,
            "arm": condition.arm,
            "status": status,
            "reason": reason,
            "target_run_dir": str(target_dir),
            "resolved_run_dir": source_run_dir,
            "require_dataflow": bool(require_dataflow),
            "expected_meta": dict(condition.expected_meta),
            "cli_args": list(condition.cli_args),
        }
        manifest["conditions"].append(entry)

    manifest_path = out_root / "behavioral_matrix_manifest.json"
    _write_json(manifest_path, manifest)

    if args.dry_run:
        print(f"[dry-run] wrote manifest -> {manifest_path}")
        return

    for entry in manifest["conditions"]:
        if entry["status"] == "reused":
            print(
                f"[reuse] {entry['dataset']} / {entry['label']} -> {entry['resolved_run_dir']}",
                flush=True,
            )
            continue

        condition = next(
            item
            for item in conditions
            if item.dataset == entry["dataset"] and item.label == entry["label"] and item.arm == entry["arm"]
        )
        print(
            f"[run] {entry['dataset']} / {entry['label']} -> {entry['target_run_dir']}",
            flush=True,
        )
        try:
            _execute_condition(condition=condition, python_bin=args.python_bin, device=args.device)
        except subprocess.CalledProcessError as exc:
            entry["status"] = "failed"
            entry["reason"] = f"runner_failed:{exc.returncode}"
            entry["resolved_run_dir"] = None
            _write_json(manifest_path, manifest)
            raise
        entry["status"] = "completed"
        entry["reason"] = "executed_runner"
        entry["resolved_run_dir"] = str(Path(condition.target_run_dir).resolve())
        _write_json(manifest_path, manifest)

    print(f"[done] manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
