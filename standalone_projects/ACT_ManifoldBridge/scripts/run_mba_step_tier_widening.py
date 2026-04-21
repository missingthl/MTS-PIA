#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


DATASETS = ["natops", "heartbeat", "atrialfibrillation"]
MAIN_ARMS = [
    "mba",
    "mba_feedback_easy",
    "mba_wide",
    "mba_wide_feedback_easy",
    "mba_wide_feedback_hard",
]
REFILL_ARMS = [
    "mba_wide_feedback_easy_tau2",
    "mba_wide_feedback_hard_tau2",
]


@dataclass(frozen=True)
class TaskSpec:
    arm: str
    dataset: str
    seeds: str
    out_root: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_conda_env_prefix(conda_env: str) -> Optional[str]:
    proc = subprocess.run(["conda", "env", "list", "--json"], capture_output=True, text=True, check=True)
    payload = json.loads(proc.stdout)
    suffix = os.sep + conda_env
    for env_path in payload.get("envs", []):
        if env_path.endswith(suffix):
            return env_path
    return None


def _parse_devices(raw: str) -> List[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or ["cuda:0"]


def _configure_device_env(base_env: Dict[str, str], device_spec: str) -> tuple[Dict[str, str], str]:
    env = dict(base_env)
    if device_spec.startswith("cuda:"):
        physical_id = device_spec.split(":", 1)[1]
        env["CUDA_VISIBLE_DEVICES"] = physical_id
        return env, "cuda:0"
    return env, device_spec


def _build_task_command(task: TaskSpec, conda_env: str) -> List[str]:
    project_root = _project_root()
    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        str(project_root / "run_act_pilot.py"),
        "--dataset",
        task.dataset,
        "--algo",
        "lraes",
        "--model",
        "resnet1d",
        "--epochs",
        "30",
        "--k-dir",
        "10",
        "--pia-gamma",
        "0.1",
        "--multiplier",
        "1",
        "--batch-size",
        "64",
        "--lr",
        "1e-3",
        "--patience",
        "10",
        "--val-ratio",
        "0.2",
        "--seeds",
        task.seeds,
        "--theory-diagnostics",
        "--out-root",
        task.out_root,
    ]
    if task.arm == "mba":
        cmd.extend(["--pipeline", "mba", "--mba-candidate-mode", "core"])
    elif task.arm == "mba_feedback_easy":
        cmd.extend(["--pipeline", "mba_feedback", "--mba-candidate-mode", "core", "--feedback-margin-polarity", "easy"])
    elif task.arm == "mba_wide":
        cmd.extend(["--pipeline", "mba", "--mba-candidate-mode", "step_tiers"])
    elif task.arm == "mba_wide_feedback_easy":
        cmd.extend(["--pipeline", "mba_feedback", "--mba-candidate-mode", "step_tiers", "--feedback-margin-polarity", "easy"])
    elif task.arm == "mba_wide_feedback_hard":
        cmd.extend(["--pipeline", "mba_feedback", "--mba-candidate-mode", "step_tiers", "--feedback-margin-polarity", "hard"])
    elif task.arm == "mba_wide_feedback_easy_tau2":
        cmd.extend(
            [
                "--pipeline",
                "mba_feedback",
                "--mba-candidate-mode",
                "step_tiers",
                "--feedback-margin-polarity",
                "easy",
                "--feedback-margin-temperature",
                "2.0",
            ]
        )
    elif task.arm == "mba_wide_feedback_hard_tau2":
        cmd.extend(
            [
                "--pipeline",
                "mba_feedback",
                "--mba-candidate-mode",
                "step_tiers",
                "--feedback-margin-polarity",
                "hard",
                "--feedback-margin-temperature",
                "2.0",
            ]
        )
    else:
        raise ValueError(f"Unsupported arm: {task.arm}")
    return cmd


def _final_results_path(task: TaskSpec) -> Path:
    return Path(task.out_root) / "final_results.csv"


def _should_skip_existing(task: TaskSpec) -> bool:
    path = _final_results_path(task)
    return path.is_file() and path.stat().st_size > 0


def _run_single_task(
    *,
    task: TaskSpec,
    device_spec: str,
    base_root: Path,
    conda_env: str,
    base_env: Dict[str, str],
    dry_run: bool,
) -> bool:
    log_path = base_root / "_logs" / task.arm / f"{task.dataset}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    Path(task.out_root).mkdir(parents=True, exist_ok=True)
    if _should_skip_existing(task):
        print(f"[SKIP] {task.arm} | {task.dataset} | existing final_results.csv")
        return True

    env, torch_device = _configure_device_env(base_env, device_spec)
    cmd = _build_task_command(task, conda_env) + ["--device", torch_device]
    print(f"[RUN ] {task.arm} | {task.dataset} | seeds={task.seeds} | device={device_spec}")
    if dry_run:
        print(" ".join(cmd))
        return True

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(_repo_root()),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        print(f"[FAIL] {task.arm} | {task.dataset} | see {log_path}")
        return False
    print(f"[DONE] {task.arm} | {task.dataset}")
    return True


def _partition_tasks(tasks: List[TaskSpec], devices: List[str]) -> Dict[str, List[TaskSpec]]:
    buckets = {device: [] for device in devices}
    for idx, task in enumerate(tasks):
        device = devices[idx % len(devices)]
        buckets[device].append(task)
    return buckets


def _run_task_group(
    *,
    tasks: List[TaskSpec],
    base_root: Path,
    devices: List[str],
    conda_env: str,
    base_env: Dict[str, str],
    dry_run: bool,
) -> bool:
    buckets = _partition_tasks(tasks, devices)
    all_ok = True
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = []
        for device, device_tasks in buckets.items():
            if not device_tasks:
                continue
            futures.append(
                executor.submit(
                    lambda d=device, ts=device_tasks: [
                        _run_single_task(
                            task=task,
                            device_spec=d,
                            base_root=base_root,
                            conda_env=conda_env,
                            base_env=base_env,
                            dry_run=dry_run,
                        )
                        for task in ts
                    ]
                )
            )
        for future in futures:
            for ok in future.result():
                all_ok = all_ok and ok
    return all_ok


def _main_tasks(base_root: Path, seeds: str) -> List[TaskSpec]:
    tasks: List[TaskSpec] = []
    for dataset in DATASETS:
        for arm in MAIN_ARMS:
            tasks.append(TaskSpec(arm=arm, dataset=dataset, seeds=seeds, out_root=str(base_root / arm / dataset)))
    return tasks


def _read_collapse_candidates(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _refill_tasks(base_root: Path, collapse_rows: List[Dict[str, str]]) -> List[TaskSpec]:
    grouped: Dict[str, List[str]] = {}
    for row in collapse_rows:
        dataset = str(row["dataset"])
        seed = str(row["seed"])
        grouped.setdefault(dataset, []).append(seed)
    tasks: List[TaskSpec] = []
    for dataset, seeds in sorted(grouped.items()):
        seed_list = ",".join(sorted(set(seeds), key=int))
        for arm in REFILL_ARMS:
            tasks.append(TaskSpec(arm=arm, dataset=dataset, seeds=seed_list, out_root=str(base_root / arm / dataset)))
    return tasks


def _run_summary(base_root: Path, dry_run: bool) -> bool:
    summary_script = _project_root() / "scripts" / "summarize_mba_step_tier_widening.py"
    cmd = [sys.executable, str(summary_script), "--root", str(base_root)]
    print(f"[SUMM] {' '.join(cmd)}")
    if dry_run:
        return True
    proc = subprocess.run(cmd, cwd=str(_repo_root()))
    return proc.returncode == 0


def _write_manifest(base_root: Path, args: argparse.Namespace, devices: List[str]) -> None:
    payload = {
        "results_root": str(base_root),
        "devices": devices,
        "conda_env": args.conda_env,
        "prepend_conda_lib": bool(args.prepend_conda_lib),
        "datasets": DATASETS,
        "main_arms": MAIN_ARMS,
        "refill_arms": REFILL_ARMS,
        "seeds": args.seeds,
    }
    path = base_root / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MBA step-tier widening v1 matrix.")
    parser.add_argument("--mode", choices=["main", "refill", "all"], default="all")
    parser.add_argument(
        "--results-root",
        default="standalone_projects/ACT_ManifoldBridge/results/mba_step_tier_widening_v1",
    )
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--conda-env", default="pia")
    parser.add_argument("--prepend-conda-lib", action="store_true", default=True)
    parser.add_argument("--no-prepend-conda-lib", dest="prepend_conda_lib", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_root = (_repo_root() / args.results_root).resolve()
    devices = _parse_devices(args.devices)
    _write_manifest(base_root, args, devices)

    base_env = os.environ.copy()
    if args.prepend_conda_lib:
        prefix = _resolve_conda_env_prefix(args.conda_env)
        if prefix:
            lib_dir = os.path.join(prefix, "lib")
            existing = base_env.get("LD_LIBRARY_PATH", "")
            base_env["LD_LIBRARY_PATH"] = lib_dir if not existing else f"{lib_dir}:{existing}"
            print(f"[ENV ] prepended LD_LIBRARY_PATH with {lib_dir}")
        else:
            print(f"[WARN] could not resolve conda env prefix for {args.conda_env}")

    if args.mode in {"main", "all"}:
        if not _run_task_group(
            tasks=_main_tasks(base_root, args.seeds),
            base_root=base_root,
            devices=devices,
            conda_env=args.conda_env,
            base_env=base_env,
            dry_run=args.dry_run,
        ):
            return 1
        if not _run_summary(base_root, dry_run=args.dry_run):
            return 1

    if args.mode in {"refill", "all"}:
        collapse_rows = _read_collapse_candidates(base_root / "_summary" / "collapse_candidates.csv")
        refill_tasks = _refill_tasks(base_root, collapse_rows)
        if refill_tasks:
            if not _run_task_group(
                tasks=refill_tasks,
                base_root=base_root,
                devices=devices,
                conda_env=args.conda_env,
                base_env=base_env,
                dry_run=args.dry_run,
            ):
                return 1
            if not _run_summary(base_root, dry_run=args.dry_run):
                return 1
        else:
            print("[INFO] no selective-collapse rows found; skipping tau refill")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
