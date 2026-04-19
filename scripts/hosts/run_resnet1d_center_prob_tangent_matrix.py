#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "hosts" / "run_resnet1d_local_closed_form_fixedsplit.py"
SUMMARIZER = ROOT / "scripts" / "analysis" / "summarize_resnet1d_center_prob_tangent.py"
DEFAULT_PIA_PYTHON = Path("/home/THL/miniconda3/envs/pia/bin/python")
DEFAULT_BEHAVIORAL_RESULTS = ROOT / "out" / "_active" / "verify_resnet1d_dlcr_behavioral_matrix_20260418" / "behavioral_results_table.csv"
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
PHASE0_DATASETS = ["natops", "fingermovements", "selfregulationscp1", "epilepsy"]
THREAD_ENV = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "VECLIB_MAXIMUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
}
DEFAULT_RUNTIME_CONFIG = {
    "train_batch_size": 128,
    "test_batch_size": 256,
    "num_workers": 8,
}
FALLBACK_RUNTIME_CONFIG = {
    "train_batch_size": 64,
    "test_batch_size": 128,
    "num_workers": 4,
}


@dataclass(frozen=True)
class WorkerSpec:
    name: str
    visible_device: str
    cpu_affinity: str
    cpunodebind: str = "0"
    membind: str = "0"


@dataclass
class PhaseCondition:
    dataset: str
    label: str
    cli_args: List[str]
    target_run_dir: str
    meta: Dict[str, Any]


WORKERS = [
    WorkerSpec(name="worker_a", visible_device="0", cpu_affinity="0-25,104-129"),
    WorkerSpec(name="worker_b", visible_device="1", cpu_affinity="26-51,130-155"),
]
TASKSET_BIN = shutil.which("taskset")
NUMACTL_BIN = shutil.which("numactl")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_dataset_list(raw: str | None) -> List[str]:
    if not raw:
        return list(DEFAULT_DATASETS)
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _load_best_subproto_tau_map(path: Path) -> Dict[str, float]:
    rows = _read_csv(path)
    mapping: Dict[str, float] = {}
    for row in rows:
        dataset = str(row.get("dataset", "")).strip().lower()
        tau_raw = row.get("best_subproto_tau")
        if not dataset or tau_raw in (None, "", "None"):
            continue
        mapping[dataset] = float(tau_raw)
    return mapping


def _remove_path_if_exists(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _load_or_init_runtime_config(out_root: Path) -> Dict[str, Any]:
    config_path = out_root / "center_prob_tangent_runtime_config.json"
    if config_path.is_file():
        return _read_json(config_path)
    return dict(DEFAULT_RUNTIME_CONFIG)


def _persist_runtime_config(out_root: Path, config: Dict[str, Any], *, downgraded: bool) -> None:
    config_path = out_root / "center_prob_tangent_runtime_config.json"
    payload = dict(config)
    payload["downgraded"] = bool(downgraded)
    _write_json(config_path, payload)


def _build_common_cli(
    *,
    dataset: str,
    epochs: int,
    seed: int,
    out_root: Path,
    run_tag: str,
    runtime_config: Dict[str, Any],
) -> List[str]:
    return [
        "--dataset",
        dataset,
        "--arm",
        "e2",
        "--epochs",
        str(int(epochs)),
        "--seed",
        str(int(seed)),
        "--train-batch-size",
        str(int(runtime_config["train_batch_size"])),
        "--test-batch-size",
        str(int(runtime_config["test_batch_size"])),
        "--num-workers",
        str(int(runtime_config["num_workers"])),
        "--closed-form-solve-mode",
        "pinv",
        "--local-support-mode",
        "same_only",
        "--prototype-aggregation",
        "pooled",
        "--routing-temperature",
        "1.0",
        "--class-prior-temperature",
        "1.0",
        "--tangent-rank",
        "4",
        "--tangent-source",
        "subproto_offsets",
        "--dataflow-probe",
        "--out-root",
        str(out_root),
        "--run-tag",
        run_tag,
    ]


def _build_condition(
    *,
    phase_root: Path,
    dataset: str,
    label: str,
    epochs: int,
    seed: int,
    runtime_config: Dict[str, Any],
    geometry_mode: str,
    prob_tangent_version: str | None = None,
    rank_selection_mode: str | None = None,
    subproto_temperature: float = 1.0,
) -> PhaseCondition:
    run_tag = f"{label}_{dataset}_seed{seed}".replace(".", "p")
    cli_args = _build_common_cli(
        dataset=dataset,
        epochs=epochs,
        seed=seed,
        out_root=phase_root,
        run_tag=run_tag,
        runtime_config=runtime_config,
    )
    cli_args.extend(
        [
            "--prototype-geometry-mode",
            str(geometry_mode),
            "--subproto-temperature",
            str(float(subproto_temperature)),
        ]
    )
    if prob_tangent_version is not None:
        cli_args.extend(["--prob-tangent-version", str(prob_tangent_version)])
    if rank_selection_mode is not None:
        cli_args.extend(["--rank-selection-mode", str(rank_selection_mode)])
    return PhaseCondition(
        dataset=dataset,
        label=label,
        cli_args=cli_args,
        target_run_dir=str((phase_root / "e2" / run_tag).resolve()),
        meta={
            "dataset": dataset,
            "label": label,
            "epochs": int(epochs),
            "geometry_mode": str(geometry_mode),
            "prob_tangent_version": prob_tangent_version,
            "rank_selection_mode": rank_selection_mode,
            "subproto_temperature": float(subproto_temperature),
            "runtime_config": dict(runtime_config),
        },
    )


def _build_phase_conditions(
    *,
    phase: str,
    out_root: Path,
    datasets: Sequence[str],
    seed: int,
    runtime_config: Dict[str, Any],
    best_tau_map: Dict[str, float],
) -> List[PhaseCondition]:
    phase_root = out_root / phase
    if phase == "phase0_smoke":
        conditions: List[PhaseCondition] = []
        for dataset in PHASE0_DATASETS:
            for version in ("v1", "v2", "v3"):
                conditions.append(
                    _build_condition(
                        phase_root=phase_root,
                        dataset=dataset,
                        label=f"center_prob_tangent_{version}",
                        epochs=1,
                        seed=seed,
                        runtime_config=runtime_config,
                        geometry_mode="center_prob_tangent",
                        prob_tangent_version=version,
                        rank_selection_mode="mdl",
                    )
                )
        return conditions

    if phase == "phase05_short_formal":
        conditions = []
        for dataset in PHASE0_DATASETS:
            best_tau = float(best_tau_map[dataset])
            conditions.extend(
                [
                    _build_condition(
                        phase_root=phase_root,
                        dataset=dataset,
                        label="center_tangent_v1",
                        epochs=40,
                        seed=seed,
                        runtime_config=runtime_config,
                        geometry_mode="center_tangent",
                        subproto_temperature=1.0,
                    ),
                    _build_condition(
                        phase_root=phase_root,
                        dataset=dataset,
                        label="center_prob_tangent_v1",
                        epochs=40,
                        seed=seed,
                        runtime_config=runtime_config,
                        geometry_mode="center_prob_tangent",
                        prob_tangent_version="v1",
                        rank_selection_mode="mdl",
                    ),
                    _build_condition(
                        phase_root=phase_root,
                        dataset=dataset,
                        label="center_prob_tangent_v2",
                        epochs=40,
                        seed=seed,
                        runtime_config=runtime_config,
                        geometry_mode="center_prob_tangent",
                        prob_tangent_version="v2",
                        rank_selection_mode="mdl",
                    ),
                    _build_condition(
                        phase_root=phase_root,
                        dataset=dataset,
                        label="center_prob_tangent_v3",
                        epochs=40,
                        seed=seed,
                        runtime_config=runtime_config,
                        geometry_mode="center_prob_tangent",
                        prob_tangent_version="v3",
                        rank_selection_mode="mdl",
                    ),
                    _build_condition(
                        phase_root=phase_root,
                        dataset=dataset,
                        label="center_only",
                        epochs=40,
                        seed=seed,
                        runtime_config=runtime_config,
                        geometry_mode="center_only",
                        subproto_temperature=1.0,
                    ),
                    _build_condition(
                        phase_root=phase_root,
                        dataset=dataset,
                        label="best_center_subproto",
                        epochs=40,
                        seed=seed,
                        runtime_config=runtime_config,
                        geometry_mode="center_subproto",
                        subproto_temperature=best_tau,
                    ),
                ]
            )
        return conditions

    if phase == "phase1_fullscale":
        conditions = []
        for dataset in datasets:
            best_tau = float(best_tau_map[dataset])
            conditions.extend(
                [
                    _build_condition(
                        phase_root=phase_root,
                        dataset=dataset,
                        label="center_only",
                        epochs=100,
                        seed=seed,
                        runtime_config=runtime_config,
                        geometry_mode="center_only",
                        subproto_temperature=1.0,
                    ),
                    _build_condition(
                        phase_root=phase_root,
                        dataset=dataset,
                        label="best_center_subproto",
                        epochs=100,
                        seed=seed,
                        runtime_config=runtime_config,
                        geometry_mode="center_subproto",
                        subproto_temperature=best_tau,
                    ),
                    _build_condition(
                        phase_root=phase_root,
                        dataset=dataset,
                        label="center_prob_tangent_v3",
                        epochs=100,
                        seed=seed,
                        runtime_config=runtime_config,
                        geometry_mode="center_prob_tangent",
                        prob_tangent_version="v3",
                        rank_selection_mode="mdl",
                    ),
                ]
            )
        return conditions

    raise ValueError(f"unsupported phase: {phase}")


def _assign_conditions(conditions: Sequence[PhaseCondition]) -> List[List[PhaseCondition]]:
    buckets: List[List[PhaseCondition]] = [[] for _ in WORKERS]
    for idx, condition in enumerate(conditions):
        buckets[idx % len(WORKERS)].append(condition)
    return buckets


def _is_oom_failure(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    try:
        content = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    markers = [
        "cuda out of memory",
        "out of memory",
        "cublas_status_alloc_failed",
        "cuda error: out of memory",
    ]
    return any(marker in content for marker in markers)


def _run_worker(
    *,
    worker: WorkerSpec,
    conditions: Sequence[PhaseCondition],
    python_bin: str,
    phase_root: Path,
) -> List[Dict[str, Any]]:
    phase_logs = phase_root / "logs"
    _ensure_dir(phase_logs)
    records: List[Dict[str, Any]] = []
    for condition in conditions:
        target_dir = Path(condition.target_run_dir)
        if target_dir.exists():
            _remove_path_if_exists(target_dir)
        log_path = phase_logs / f"{Path(condition.target_run_dir).name}.log"
        env = os.environ.copy()
        env.update(THREAD_ENV)
        env["CUDA_VISIBLE_DEVICES"] = worker.visible_device
        command: List[str] = []
        if TASKSET_BIN is not None:
            command.extend([TASKSET_BIN, "-c", worker.cpu_affinity])
        if NUMACTL_BIN is not None:
            command.extend(
                [
                    NUMACTL_BIN,
                    f"--cpunodebind={worker.cpunodebind}",
                    f"--membind={worker.membind}",
                ]
            )
        command.extend(
            [
                python_bin,
                str(RUNNER),
                *condition.cli_args,
                "--device",
                "cuda:0",
            ]
        )
        started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.run(
                command,
                cwd=str(ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        status = "completed" if process.returncode == 0 else "failed"
        reason = "executed_runner" if process.returncode == 0 else f"runner_failed:{process.returncode}"
        if status == "failed" and _is_oom_failure(log_path):
            reason = "oom"
        records.append(
            {
                "dataset": condition.dataset,
                "label": condition.label,
                "status": status,
                "reason": reason,
                "resolved_run_dir": str(target_dir.resolve()) if status == "completed" else None,
                "log_path": str(log_path.resolve()),
                "worker": worker.name,
                "started_at": started_at,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    return records


def _run_phase_parallel(
    *,
    phase_root: Path,
    conditions: Sequence[PhaseCondition],
    python_bin: str,
) -> List[Dict[str, Any]]:
    assignments = _assign_conditions(conditions)
    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(WORKERS)) as executor:
        futures = []
        for worker, worker_conditions in zip(WORKERS, assignments):
            futures.append(
                executor.submit(
                    _run_worker,
                    worker=worker,
                    conditions=worker_conditions,
                    python_bin=python_bin,
                    phase_root=phase_root,
                )
            )
        for future in futures:
            records.extend(future.result())
    return records


def _update_manifest_with_records(manifest: Dict[str, Any], records: Sequence[Dict[str, Any]]) -> None:
    index = {(str(item["dataset"]), str(item["label"])): item for item in manifest["conditions"]}
    for record in records:
        entry = index[(str(record["dataset"]), str(record["label"]))]
        entry["status"] = record["status"]
        entry["reason"] = record["reason"]
        entry["resolved_run_dir"] = record["resolved_run_dir"]
        entry["log_path"] = record["log_path"]
        entry["worker"] = record["worker"]
        entry["started_at"] = record["started_at"]
        entry["finished_at"] = record["finished_at"]


def _run_phase_with_oom_fallback(
    *,
    phase: str,
    out_root: Path,
    conditions_builder,
    seed: int,
    datasets: Sequence[str],
    best_tau_map: Dict[str, float],
    python_bin: str,
) -> Dict[str, Any]:
    runtime_config = _load_or_init_runtime_config(out_root)
    downgraded = bool(runtime_config.get("downgraded", False))
    attempts: List[Dict[str, Any]] = []

    for attempt_idx in range(2):
        phase_root = out_root / phase
        if attempt_idx > 0 and phase_root.exists():
            shutil.rmtree(phase_root)
        conditions = conditions_builder(
            phase=phase,
            out_root=out_root,
            datasets=datasets,
            seed=seed,
            runtime_config=runtime_config,
            best_tau_map=best_tau_map,
        )
        manifest = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": phase,
            "out_root": str(out_root.resolve()),
            "phase_root": str((out_root / phase).resolve()),
            "runner": str(RUNNER.resolve()),
            "python_bin": python_bin,
            "taskset_available": bool(TASKSET_BIN),
            "numactl_available": bool(NUMACTL_BIN),
            "runtime_config": dict(runtime_config),
            "conditions": [
                {
                    "index": int(idx + 1),
                    "dataset": condition.dataset,
                    "label": condition.label,
                    "status": "pending",
                    "reason": "",
                    "resolved_run_dir": None,
                    "target_run_dir": condition.target_run_dir,
                    "meta": dict(condition.meta),
                    "cli_args": list(condition.cli_args),
                }
                for idx, condition in enumerate(conditions)
            ],
        }
        manifest_path = out_root / phase / f"center_prob_tangent_{phase}_manifest.json"
        _write_json(manifest_path, manifest)
        records = _run_phase_parallel(
            phase_root=out_root / phase,
            conditions=conditions,
            python_bin=python_bin,
        )
        _update_manifest_with_records(manifest, records)
        _write_json(manifest_path, manifest)
        attempts.append({"runtime_config": dict(runtime_config), "records": records})

        failed = [record for record in records if record["status"] != "completed"]
        oom_failed = [record for record in failed if record["reason"] == "oom"]
        if not failed:
            _persist_runtime_config(out_root, runtime_config, downgraded=downgraded)
            return {"manifest": manifest, "manifest_path": manifest_path, "attempts": attempts}
        if phase == "phase0_smoke" and oom_failed and runtime_config == DEFAULT_RUNTIME_CONFIG:
            runtime_config = dict(FALLBACK_RUNTIME_CONFIG)
            downgraded = True
            continue
        raise RuntimeError(f"{phase} failed; see manifest {manifest_path}")

    raise RuntimeError(f"{phase} failed after OOM fallback; see {out_root / phase}")


def _maybe_enforce_phase1_gate(out_root: Path) -> None:
    summary_path = out_root / "center_prob_tangent_phase05_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("phase1_fullscale requires center_prob_tangent_phase05_summary.json")
    summary = _read_json(summary_path)
    if not bool(summary.get("phase1_gate_passed", False)):
        raise RuntimeError("phase05_short_formal gate did not pass; refusing to launch phase1_fullscale")


def _run_summarizer(*, manifest_path: Path, e0_reference: Path, python_bin: str) -> None:
    if not SUMMARIZER.is_file():
        return
    subprocess.run(
        [
            python_bin,
            str(SUMMARIZER),
            "--manifest",
            str(manifest_path),
            "--e0-reference",
            str(e0_reference),
        ],
        cwd=str(ROOT),
        check=True,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Center prob tangent matrix launcher with dual-GPU DLCR scheduling.")
    parser.add_argument("--phase", type=str, required=True, choices=["phase0_smoke", "phase05_short_formal", "phase1_fullscale"])
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(ROOT / "out" / "_active" / f"verify_resnet1d_center_prob_tangent_{time.strftime('%Y%m%d')}"),
    )
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--python-bin",
        type=str,
        default=str(DEFAULT_PIA_PYTHON if DEFAULT_PIA_PYTHON.is_file() else Path(sys.executable).resolve()),
    )
    parser.add_argument("--behavioral-results", type=str, default=str(DEFAULT_BEHAVIORAL_RESULTS))
    parser.add_argument("--skip-phase1-gate", action="store_true", default=False)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    out_root = Path(args.out_root).resolve()
    _ensure_dir(out_root)
    if TASKSET_BIN is None:
        raise RuntimeError("required binary not found on PATH: taskset")
    datasets = _parse_dataset_list(args.datasets)
    best_tau_map = _load_best_subproto_tau_map(Path(args.behavioral_results).resolve())
    missing = [dataset for dataset in (PHASE0_DATASETS if args.phase != "phase1_fullscale" else datasets) if dataset not in best_tau_map]
    if missing:
        raise ValueError(f"missing best_subproto_tau for datasets: {missing}")
    if args.phase == "phase1_fullscale" and not args.skip_phase1_gate:
        _maybe_enforce_phase1_gate(out_root)

    result = _run_phase_with_oom_fallback(
        phase=str(args.phase),
        out_root=out_root,
        conditions_builder=_build_phase_conditions,
        seed=int(args.seed),
        datasets=datasets,
        best_tau_map=best_tau_map,
        python_bin=str(args.python_bin),
    )
    manifest_path = Path(result["manifest_path"]).resolve()
    if args.phase in {"phase05_short_formal", "phase1_fullscale"}:
        _run_summarizer(
            manifest_path=manifest_path,
            e0_reference=Path(args.behavioral_results).resolve(),
            python_bin=str(args.python_bin),
        )
    print(f"[done] manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
