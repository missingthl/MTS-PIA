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
DEFAULT_PHASE05_REFERENCE = ROOT / "out" / "_active" / "verify_resnet1d_center_prob_tangent_20260419" / "center_prob_tangent_phase05_table.csv"
DEFAULT_FULLSCALE_REFERENCE = ROOT / "out" / "_active" / "verify_resnet1d_center_prob_tangent_20260419" / "center_prob_tangent_fullscale_table.csv"
DEFAULT_PHASEB_GAUSSIAN_REFERENCE = ROOT / "out" / "_active" / "verify_resnet1d_center_prob_tangent_refined_20260420" / "center_prob_tangent_phaseB_short_formal_refined_gaussian_dimnorm_table.csv"
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
PHASEA_DATASETS = ["fingermovements", "heartbeat"]
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
    posterior_mode: str | None = None,
    posterior_student_dof: float | None = None,
    mdl_penalty_beta: float | None = None,
    gaussian_refine_variant: str | None = None,
    mdl_zero_rank_rescue_margin: float | None = None,
    local_solver_competition_mode: str | None = None,
    relative_solver_temperature: float | None = None,
    abs_gate_activity_floor: float | None = None,
    emit_mdl_rank_trace: bool = False,
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
    if posterior_mode is not None:
        cli_args.extend(["--posterior-mode", str(posterior_mode)])
    if posterior_student_dof is not None:
        cli_args.extend(["--posterior-student-dof", str(float(posterior_student_dof))])
    if mdl_penalty_beta is not None:
        cli_args.extend(["--mdl-penalty-beta", str(float(mdl_penalty_beta))])
    if gaussian_refine_variant is not None:
        cli_args.extend(["--gaussian-refine-variant", str(gaussian_refine_variant)])
    if mdl_zero_rank_rescue_margin is not None:
        cli_args.extend(["--mdl-zero-rank-rescue-margin", str(float(mdl_zero_rank_rescue_margin))])
    if local_solver_competition_mode is not None:
        cli_args.extend(["--local-solver-competition-mode", str(local_solver_competition_mode)])
    if relative_solver_temperature is not None:
        cli_args.extend(["--relative-solver-temperature", str(float(relative_solver_temperature))])
    if abs_gate_activity_floor is not None:
        cli_args.extend(["--abs-gate-activity-floor", str(float(abs_gate_activity_floor))])
    if emit_mdl_rank_trace:
        cli_args.append("--emit-mdl-rank-trace")
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
            "posterior_mode": posterior_mode,
            "posterior_student_dof": None if posterior_student_dof is None else float(posterior_student_dof),
            "mdl_penalty_beta": None if mdl_penalty_beta is None else float(mdl_penalty_beta),
            "gaussian_refine_variant": gaussian_refine_variant,
            "mdl_zero_rank_rescue_margin": None if mdl_zero_rank_rescue_margin is None else float(mdl_zero_rank_rescue_margin),
            "local_solver_competition_mode": local_solver_competition_mode,
            "relative_solver_temperature": None if relative_solver_temperature is None else float(relative_solver_temperature),
            "abs_gate_activity_floor": None if abs_gate_activity_floor is None else float(abs_gate_activity_floor),
            "emit_mdl_rank_trace": bool(emit_mdl_rank_trace),
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
    if phase == "phaseA_probe":
        conditions: List[PhaseCondition] = []
        for dataset in PHASEA_DATASETS:
            conditions.append(
                _build_condition(
                    phase_root=phase_root,
                    dataset=dataset,
                    label="center_prob_tangent_v3_refined_probe",
                    epochs=40,
                    seed=seed,
                    runtime_config=runtime_config,
                    geometry_mode="center_prob_tangent",
                    prob_tangent_version="v3",
                    rank_selection_mode="mdl",
                    posterior_mode="gaussian_dimnorm",
                    posterior_student_dof=3.0,
                    mdl_penalty_beta=1.0,
                    emit_mdl_rank_trace=True,
                )
            )
        return conditions

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

    if phase in {"phaseB_short_formal_refined_gaussian_dimnorm", "phaseB_short_formal_refined_student_t"}:
        posterior_mode = "gaussian_dimnorm" if phase.endswith("gaussian_dimnorm") else "student_t"
        conditions = []
        for dataset in PHASE0_DATASETS:
            for beta in (0.5, 1.0, 2.0, 4.0):
                beta_tag = str(beta).replace(".", "p")
                conditions.append(
                    _build_condition(
                        phase_root=phase_root,
                        dataset=dataset,
                        label=f"center_prob_tangent_v3_refined_b{beta_tag}",
                        epochs=40,
                        seed=seed,
                        runtime_config=runtime_config,
                        geometry_mode="center_prob_tangent",
                        prob_tangent_version="v3",
                        rank_selection_mode="mdl",
                        posterior_mode=posterior_mode,
                        posterior_student_dof=3.0,
                        mdl_penalty_beta=float(beta),
                    )
                )
        return conditions

    if phase in {"phaseB2_short_formal_refined_trace_floor", "phaseB2_short_formal_refined_trace_floor_mdl_margin"}:
        gaussian_refine_variant = "trace_floor" if phase.endswith("trace_floor") else "trace_floor_mdl_margin"
        conditions = []
        for dataset in PHASE0_DATASETS:
            for beta in (1.0, 2.0):
                beta_tag = str(beta).replace(".", "p")
                conditions.append(
                    _build_condition(
                        phase_root=phase_root,
                        dataset=dataset,
                        label=f"center_prob_tangent_v3_refined_{gaussian_refine_variant}_b{beta_tag}",
                        epochs=40,
                        seed=seed,
                        runtime_config=runtime_config,
                        geometry_mode="center_prob_tangent",
                        prob_tangent_version="v3",
                        rank_selection_mode="mdl",
                        posterior_mode="gaussian_dimnorm",
                        posterior_student_dof=3.0,
                        mdl_penalty_beta=float(beta),
                        gaussian_refine_variant=gaussian_refine_variant,
                        mdl_zero_rank_rescue_margin=0.03,
                    )
                )
        return conditions

    if phase == "phaseR1_relcomp":
        conditions = []
        for dataset in PHASE0_DATASETS:
            for beta in (1.0, 2.0):
                beta_tag = str(beta).replace(".", "p")
                conditions.extend(
                    [
                        _build_condition(
                            phase_root=phase_root,
                            dataset=dataset,
                            label=f"gaussian_baseline_b{beta_tag}",
                            epochs=40,
                            seed=seed,
                            runtime_config=runtime_config,
                            geometry_mode="center_prob_tangent",
                            prob_tangent_version="v3",
                            rank_selection_mode="mdl",
                            posterior_mode="gaussian_dimnorm",
                            posterior_student_dof=3.0,
                            mdl_penalty_beta=float(beta),
                            gaussian_refine_variant="trace_floor",
                            mdl_zero_rank_rescue_margin=0.03,
                            local_solver_competition_mode="none",
                            relative_solver_temperature=1.0,
                            abs_gate_activity_floor=1e-6,
                        ),
                        _build_condition(
                            phase_root=phase_root,
                            dataset=dataset,
                            label=f"gaussian_relcomp_b{beta_tag}",
                            epochs=40,
                            seed=seed,
                            runtime_config=runtime_config,
                            geometry_mode="center_prob_tangent",
                            prob_tangent_version="v3",
                            rank_selection_mode="mdl",
                            posterior_mode="gaussian_dimnorm",
                            posterior_student_dof=3.0,
                            mdl_penalty_beta=float(beta),
                            gaussian_refine_variant="trace_floor",
                            mdl_zero_rank_rescue_margin=0.03,
                            local_solver_competition_mode="relcomp",
                            relative_solver_temperature=1.0,
                            abs_gate_activity_floor=1e-6,
                        ),
                    ]
                )
        return conditions

    if phase == "phaseC_fullscale_refined":
        selected_phaseb_summary_path = out_root / "center_prob_tangent_phaseB_short_formal_refined_summary.json"
        if not selected_phaseb_summary_path.is_file():
            raise RuntimeError("phaseC_fullscale_refined requires center_prob_tangent_phaseB_short_formal_refined_summary.json")
        phaseb_summary = _read_json(selected_phaseb_summary_path)
        selected_mode = str(phaseb_summary.get("selected_posterior_mode") or "")
        selected_beta = phaseb_summary.get("selected_mdl_penalty_beta")
        if selected_mode not in {"gaussian_dimnorm", "student_t"} or selected_beta in (None, ""):
            raise RuntimeError("phaseC_fullscale_refined requires a valid selected posterior mode and mdl_penalty_beta from Phase B")
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
                        label="center_prob_tangent_v3_refined",
                        epochs=100,
                        seed=seed,
                        runtime_config=runtime_config,
                        geometry_mode="center_prob_tangent",
                        prob_tangent_version="v3",
                        rank_selection_mode="mdl",
                        posterior_mode=selected_mode,
                        posterior_student_dof=3.0,
                        mdl_penalty_beta=float(selected_beta),
                    ),
                ]
            )
        return conditions

    if phase == "phaseC2_fullscale_refined":
        selected_phaseb2_summary_path = out_root / "center_prob_tangent_phaseB2_short_formal_refined_summary.json"
        if not selected_phaseb2_summary_path.is_file():
            raise RuntimeError("phaseC2_fullscale_refined requires center_prob_tangent_phaseB2_short_formal_refined_summary.json")
        phaseb2_summary = _read_json(selected_phaseb2_summary_path)
        selected_variant = str(phaseb2_summary.get("selected_gaussian_refine_variant") or "")
        selected_beta = phaseb2_summary.get("selected_mdl_penalty_beta")
        if selected_variant not in {"trace_floor", "trace_floor_mdl_margin"} or selected_beta in (None, ""):
            raise RuntimeError("phaseC2_fullscale_refined requires a valid selected gaussian refine variant and mdl_penalty_beta from Phase B2")
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
                        label="center_prob_tangent_v3_refined_b2",
                        epochs=100,
                        seed=seed,
                        runtime_config=runtime_config,
                        geometry_mode="center_prob_tangent",
                        prob_tangent_version="v3",
                        rank_selection_mode="mdl",
                        posterior_mode="gaussian_dimnorm",
                        posterior_student_dof=3.0,
                        mdl_penalty_beta=float(selected_beta),
                        gaussian_refine_variant=selected_variant,
                        mdl_zero_rank_rescue_margin=0.03,
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


def _maybe_enforce_phasec_gate(out_root: Path) -> None:
    summary_path = out_root / "center_prob_tangent_phaseB_short_formal_refined_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("phaseC_fullscale_refined requires center_prob_tangent_phaseB_short_formal_refined_summary.json")
    summary = _read_json(summary_path)
    if not bool(summary.get("phase_c_gate_passed", False)):
        raise RuntimeError("Phase B refined gate did not pass; refusing to launch phaseC_fullscale_refined")


def _maybe_enforce_phasec2_gate(out_root: Path) -> None:
    summary_path = out_root / "center_prob_tangent_phaseB2_short_formal_refined_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("phaseC2_fullscale_refined requires center_prob_tangent_phaseB2_short_formal_refined_summary.json")
    summary = _read_json(summary_path)
    if not bool(summary.get("phase_c_gate_passed", False)):
        raise RuntimeError("Phase B2 refined gate did not pass; refusing to launch phaseC2_fullscale_refined")


def _run_summarizer(
    *,
    manifest_path: Path,
    e0_reference: Path,
    python_bin: str,
    reference_phase05_table: Path | None = None,
    reference_fullscale_table: Path | None = None,
    reference_phaseb_gaussian_table: Path | None = None,
) -> None:
    if not SUMMARIZER.is_file():
        return
    command = [
        python_bin,
        str(SUMMARIZER),
        "--manifest",
        str(manifest_path),
        "--e0-reference",
        str(e0_reference),
    ]
    if reference_phase05_table is not None:
        command.extend(["--reference-phase05-table", str(reference_phase05_table)])
    if reference_fullscale_table is not None:
        command.extend(["--reference-fullscale-table", str(reference_fullscale_table)])
    if reference_phaseb_gaussian_table is not None:
        command.extend(["--reference-phaseb-gaussian-table", str(reference_phaseb_gaussian_table)])
    subprocess.run(command, cwd=str(ROOT), check=True)


def _phase_summary_path(out_root: Path, *, phase: str) -> Path:
    return out_root / f"center_prob_tangent_{phase}_summary.json"


def _phase_table_path(out_root: Path, *, phase: str) -> Path:
    return out_root / f"center_prob_tangent_{phase}_table.csv"


def _run_refined_phase_b(
    *,
    out_root: Path,
    seed: int,
    best_tau_map: Dict[str, float],
    python_bin: str,
    e0_reference: Path,
    reference_phase05_table: Path | None,
    requested_posterior_mode: str,
) -> Path:
    modes = [requested_posterior_mode] if requested_posterior_mode != "auto" else ["gaussian_dimnorm", "student_t"]
    attempts: List[Dict[str, Any]] = []
    selected_summary: Dict[str, Any] | None = None
    selected_table: Path | None = None
    selected_summary_path: Path | None = None
    for posterior_mode in modes:
        phase = f"phaseB_short_formal_refined_{posterior_mode}"
        result = _run_phase_with_oom_fallback(
            phase=phase,
            out_root=out_root,
            conditions_builder=_build_phase_conditions,
            seed=seed,
            datasets=PHASE0_DATASETS,
            best_tau_map=best_tau_map,
            python_bin=python_bin,
        )
        manifest_path = Path(result["manifest_path"]).resolve()
        _run_summarizer(
            manifest_path=manifest_path,
            e0_reference=e0_reference,
            python_bin=python_bin,
            reference_phase05_table=reference_phase05_table,
            reference_fullscale_table=None,
        )
        summary_path = _phase_summary_path(out_root, phase=phase)
        table_path = _phase_table_path(out_root, phase=phase)
        summary = _read_json(summary_path)
        selected_summary = summary
        selected_table = table_path
        selected_summary_path = summary_path
        attempts.append(
            {
                "posterior_mode": posterior_mode,
                "summary_path": str(summary_path),
                "comparison_csv": str(table_path),
                "phase_c_gate_passed": bool(summary.get("phase_c_gate_passed", False)),
                "selected_candidate": summary.get("selected_candidate"),
            }
        )
        if bool(summary.get("phase_c_gate_passed", False)):
            break
        if requested_posterior_mode != "auto":
            break
    if selected_summary is None:
        raise RuntimeError("Phase B did not produce any summary")
    consolidated = {
        "phase": "phaseB_short_formal_refined",
        "attempts": attempts,
        "selected_posterior_mode": selected_summary.get("selected_posterior_mode"),
        "selected_mdl_penalty_beta": selected_summary.get("selected_mdl_penalty_beta"),
        "selected_candidate": selected_summary.get("selected_candidate"),
        "phase_c_gate_passed": bool(selected_summary.get("phase_c_gate_passed", False)),
        "selected_summary_path": None if selected_summary_path is None else str(selected_summary_path),
        "comparison_csv": None if selected_table is None else str(selected_table),
    }
    consolidated_path = out_root / "center_prob_tangent_phaseB_short_formal_refined_summary.json"
    _write_json(consolidated_path, consolidated)
    return consolidated_path


def _run_refined_phase_b2(
    *,
    out_root: Path,
    seed: int,
    best_tau_map: Dict[str, float],
    python_bin: str,
    e0_reference: Path,
    reference_phase05_table: Path | None,
    reference_phaseb_gaussian_table: Path | None,
) -> Path:
    subphases = [
        "phaseB2_short_formal_refined_trace_floor",
        "phaseB2_short_formal_refined_trace_floor_mdl_margin",
    ]
    attempts: List[Dict[str, Any]] = []
    all_conditions: List[Dict[str, Any]] = []
    for phase in subphases:
        result = _run_phase_with_oom_fallback(
            phase=phase,
            out_root=out_root,
            conditions_builder=_build_phase_conditions,
            seed=seed,
            datasets=PHASE0_DATASETS,
            best_tau_map=best_tau_map,
            python_bin=python_bin,
        )
        manifest_path = Path(result["manifest_path"]).resolve()
        manifest = _read_json(manifest_path)
        all_conditions.extend(list(manifest.get("conditions", [])))
        attempts.append(
            {
                "phase": phase,
                "manifest_path": str(manifest_path),
            }
        )

    combined_phase = "phaseB2_short_formal_refined"
    combined_phase_root = out_root / combined_phase
    _ensure_dir(combined_phase_root)
    combined_manifest_path = combined_phase_root / f"center_prob_tangent_{combined_phase}_manifest.json"
    combined_manifest = {
        "phase": combined_phase,
        "out_root": str(out_root),
        "phase_root": str(combined_phase_root),
        "attempts": attempts,
        "conditions": all_conditions,
    }
    _write_json(combined_manifest_path, combined_manifest)
    _run_summarizer(
        manifest_path=combined_manifest_path,
        e0_reference=e0_reference,
        python_bin=python_bin,
        reference_phase05_table=reference_phase05_table,
        reference_fullscale_table=None,
        reference_phaseb_gaussian_table=reference_phaseb_gaussian_table,
    )
    summary_path = _phase_summary_path(out_root, phase=combined_phase)
    summary = _read_json(summary_path)
    consolidated = {
        "phase": combined_phase,
        "attempts": attempts,
        "selected_gaussian_refine_variant": summary.get("selected_gaussian_refine_variant"),
        "selected_mdl_penalty_beta": summary.get("selected_mdl_penalty_beta"),
        "selected_candidate": summary.get("selected_candidate"),
        "phase_c_gate_passed": bool(summary.get("phase_c_gate_passed", False)),
        "selected_summary_path": str(summary_path),
        "comparison_csv": str(_phase_table_path(out_root, phase=combined_phase)),
    }
    consolidated_path = out_root / "center_prob_tangent_phaseB2_short_formal_refined_summary.json"
    _write_json(consolidated_path, consolidated)
    return consolidated_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Center prob tangent matrix launcher with dual-GPU DLCR scheduling.")
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=[
            "phase0_smoke",
            "phase05_short_formal",
            "phase1_fullscale",
            "phaseA_probe",
            "phaseB_short_formal_refined",
            "phaseB2_short_formal_refined",
            "phaseR1_relcomp",
            "phaseC_fullscale_refined",
            "phaseC2_fullscale_refined",
        ],
    )
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
    parser.add_argument("--reference-phase05-table", type=str, default=str(DEFAULT_PHASE05_REFERENCE))
    parser.add_argument("--reference-fullscale-table", type=str, default=str(DEFAULT_FULLSCALE_REFERENCE))
    parser.add_argument("--reference-phaseb-gaussian-table", type=str, default=str(DEFAULT_PHASEB_GAUSSIAN_REFERENCE))
    parser.add_argument("--posterior-mode", type=str, default="auto", choices=["auto", "gaussian_dimnorm", "student_t"])
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
    if args.phase == "phaseA_probe":
        required_datasets = PHASEA_DATASETS
    elif args.phase in {"phase05_short_formal", "phaseB_short_formal_refined", "phaseB2_short_formal_refined", "phaseR1_relcomp"}:
        required_datasets = PHASE0_DATASETS
    elif args.phase in {"phase1_fullscale", "phaseC_fullscale_refined", "phaseC2_fullscale_refined"}:
        required_datasets = datasets
    else:
        required_datasets = PHASE0_DATASETS
    requires_best_tau = args.phase in {
        "phase05_short_formal",
        "phase1_fullscale",
        "phaseB_short_formal_refined",
        "phaseB2_short_formal_refined",
        "phaseC_fullscale_refined",
        "phaseC2_fullscale_refined",
    }
    if requires_best_tau:
        missing = [dataset for dataset in required_datasets if dataset not in best_tau_map]
        if missing:
            raise ValueError(f"missing best_subproto_tau for datasets: {missing}")
    if args.phase == "phase1_fullscale" and not args.skip_phase1_gate:
        _maybe_enforce_phase1_gate(out_root)
    if args.phase == "phaseC_fullscale_refined" and not args.skip_phase1_gate:
        _maybe_enforce_phasec_gate(out_root)
    if args.phase == "phaseC2_fullscale_refined" and not args.skip_phase1_gate:
        _maybe_enforce_phasec2_gate(out_root)
    if args.phase == "phaseB_short_formal_refined":
        consolidated_path = _run_refined_phase_b(
            out_root=out_root,
            seed=int(args.seed),
            best_tau_map=best_tau_map,
            python_bin=str(args.python_bin),
            e0_reference=Path(args.behavioral_results).resolve(),
            reference_phase05_table=Path(args.reference_phase05_table).resolve() if args.reference_phase05_table else None,
            requested_posterior_mode=str(args.posterior_mode),
        )
        print(f"[done] phaseB summary -> {consolidated_path}", flush=True)
        return
    if args.phase == "phaseB2_short_formal_refined":
        consolidated_path = _run_refined_phase_b2(
            out_root=out_root,
            seed=int(args.seed),
            best_tau_map=best_tau_map,
            python_bin=str(args.python_bin),
            e0_reference=Path(args.behavioral_results).resolve(),
            reference_phase05_table=Path(args.reference_phase05_table).resolve() if args.reference_phase05_table else None,
            reference_phaseb_gaussian_table=Path(args.reference_phaseb_gaussian_table).resolve() if args.reference_phaseb_gaussian_table else None,
        )
        print(f"[done] phaseB2 summary -> {consolidated_path}", flush=True)
        return

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
    if args.phase in {"phase05_short_formal", "phase1_fullscale", "phaseA_probe", "phaseR1_relcomp", "phaseC_fullscale_refined", "phaseC2_fullscale_refined"}:
        _run_summarizer(
            manifest_path=manifest_path,
            e0_reference=Path(args.behavioral_results).resolve(),
            python_bin=str(args.python_bin),
            reference_phase05_table=Path(args.reference_phase05_table).resolve() if args.reference_phase05_table else None,
            reference_fullscale_table=Path(args.reference_fullscale_table).resolve() if args.reference_fullscale_table else None,
            reference_phaseb_gaussian_table=Path(args.reference_phaseb_gaussian_table).resolve() if args.reference_phaseb_gaussian_table else None,
        )
    print(f"[done] manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
