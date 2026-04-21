#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from run_acl_small_matrix import (
    _configure_device_env,
    _parse_devices,
    _project_root,
    _repo_root,
    _resolve_conda_env_prefix,
)


FAILURE_REVIEW_DATASETS = ["handwriting", "atrialfibrillation", "motorimagery"]
STABILITY_CONFIRM_DATASETS = ["natops", "japanesevowels", "heartbeat"]

MBA_ARM = "mba"
ACL_N4_ARM = "gcg_acl_n4"
ALIGN_1P0_ARM = "gcg_acl_align1p0"
LOSS_0P1_ARM = "gcg_acl_loss0p1"
TEMP_0P05_ARM = "gcg_acl_temp0p05"
TEMP_0P10_ARM = "gcg_acl_temp0p10"

FAILURE_REVIEW_ARMS = [ALIGN_1P0_ARM, LOSS_0P1_ARM, TEMP_0P05_ARM, TEMP_0P10_ARM]
STABILITY_CONFIRM_ARMS = [MBA_ARM, ACL_N4_ARM]

DEFAULT_RESULTS_ROOTS = {
    "failure_review": "standalone_projects/ACT_ManifoldBridge/results/acl_failure_review_v1",
    "stability_confirm": "standalone_projects/ACT_ManifoldBridge/results/acl_stability_confirm_v1",
}


@dataclass(frozen=True)
class TaskSpec:
    mode: str
    arm: str
    dataset: str
    seeds: str
    out_root: str


def _resolve_results_root(mode: str, raw_root: str | None) -> Path:
    root = raw_root or DEFAULT_RESULTS_ROOTS[mode]
    return (_repo_root() / root).resolve()


def _tasks_for_mode(
    base_root: Path,
    mode: str,
    failure_seeds: str,
    stability_seeds: str,
) -> List[TaskSpec]:
    if mode == "failure_review":
        datasets = FAILURE_REVIEW_DATASETS
        arms = FAILURE_REVIEW_ARMS
        seeds = failure_seeds
    elif mode == "stability_confirm":
        datasets = STABILITY_CONFIRM_DATASETS
        arms = STABILITY_CONFIRM_ARMS
        seeds = stability_seeds
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    tasks: List[TaskSpec] = []
    for dataset in datasets:
        for arm in arms:
            tasks.append(TaskSpec(mode=mode, arm=arm, dataset=dataset, seeds=seeds, out_root=str(base_root / arm / dataset)))
    return tasks


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
        "--out-root",
        task.out_root,
    ]

    if task.arm == MBA_ARM:
        cmd.extend(["--pipeline", "mba"])
        return cmd

    alignment_weight = "0.7"
    loss_weight = "0.2"
    temperature = "0.07"
    if task.arm == ALIGN_1P0_ARM:
        alignment_weight = "1.0"
    elif task.arm == LOSS_0P1_ARM:
        loss_weight = "0.1"
    elif task.arm == TEMP_0P05_ARM:
        temperature = "0.05"
    elif task.arm == TEMP_0P10_ARM:
        temperature = "0.10"
    elif task.arm != ACL_N4_ARM:
        raise ValueError(f"Unsupported arm: {task.arm}")

    cmd.extend(
        [
            "--pipeline",
            "gcg_acl",
            "--acl-warmup-epochs",
            "10",
            "--acl-candidates-per-anchor",
            "4",
            "--acl-positives-per-anchor",
            "1",
            "--acl-alignment-weight",
            alignment_weight,
            "--acl-loss-weight",
            loss_weight,
            "--acl-temperature",
            temperature,
        ]
    )
    return cmd


def _log_path(base_root: Path, task: TaskSpec) -> Path:
    return base_root / "_logs" / task.mode / task.arm / f"{task.dataset}.log"


def _final_results_path(task: TaskSpec) -> Path:
    return Path(task.out_root) / "final_results.csv"


def _should_skip_existing(task: TaskSpec) -> bool:
    final_csv = _final_results_path(task)
    return final_csv.is_file() and final_csv.stat().st_size > 0


def _write_run_manifest(base_root: Path, devices: List[str], args: argparse.Namespace) -> None:
    manifest = {
        "mode": args.mode,
        "devices": devices,
        "results_root": str(base_root),
        "frozen_root": str((_repo_root() / args.frozen_root).resolve()),
        "failure_seeds": args.failure_seeds,
        "stability_seeds": args.stability_seeds,
        "conda_env": args.conda_env,
        "prepend_conda_lib": bool(args.prepend_conda_lib),
        "datasets": {
            "failure_review": FAILURE_REVIEW_DATASETS,
            "stability_confirm": STABILITY_CONFIRM_DATASETS,
        },
        "arms": {
            "failure_review": FAILURE_REVIEW_ARMS,
            "stability_confirm": STABILITY_CONFIRM_ARMS,
        },
    }
    manifest_path = base_root / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _run_single_task(
    *,
    task: TaskSpec,
    device_spec: str,
    base_root: Path,
    conda_env: str,
    base_env: Dict[str, str],
    dry_run: bool,
) -> bool:
    log_path = _log_path(base_root, task)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    Path(task.out_root).mkdir(parents=True, exist_ok=True)

    if _should_skip_existing(task):
        print(f"[SKIP] {task.mode} | {task.arm} | {task.dataset} | existing final_results.csv")
        return True

    env, torch_device = _configure_device_env(base_env, device_spec)
    cmd = _build_task_command(task, conda_env) + ["--device", torch_device]
    print(f"[RUN ] {task.mode} | {task.arm} | {task.dataset} | device={device_spec}")
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
        print(f"[FAIL] {task.mode} | {task.arm} | {task.dataset} | see {log_path}")
        return False
    print(f"[DONE] {task.mode} | {task.arm} | {task.dataset}")
    return True


def _run_tasks_for_device(
    *,
    device_spec: str,
    tasks: List[TaskSpec],
    base_root: Path,
    conda_env: str,
    base_env: Dict[str, str],
    dry_run: bool,
) -> List[bool]:
    results: List[bool] = []
    for task in tasks:
        ok = _run_single_task(
            task=task,
            device_spec=device_spec,
            base_root=base_root,
            conda_env=conda_env,
            base_env=base_env,
            dry_run=dry_run,
        )
        results.append(ok)
        if not ok:
            break
    return results


def _partition_tasks(tasks: List[TaskSpec], devices: List[str]) -> Dict[str, List[TaskSpec]]:
    buckets = {device: [] for device in devices}
    for idx, task in enumerate(tasks):
        device = devices[idx % len(devices)]
        buckets[device].append(task)
    return buckets


def _run_mode(
    *,
    mode: str,
    base_root: Path,
    devices: List[str],
    conda_env: str,
    base_env: Dict[str, str],
    failure_seeds: str,
    stability_seeds: str,
    dry_run: bool,
) -> bool:
    tasks = _tasks_for_mode(base_root, mode, failure_seeds, stability_seeds)
    base_root.mkdir(parents=True, exist_ok=True)
    buckets = _partition_tasks(tasks, devices)

    all_ok = True
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = []
        for device, device_tasks in buckets.items():
            if not device_tasks:
                continue
            futures.append(
                executor.submit(
                    _run_tasks_for_device,
                    device_spec=device,
                    tasks=device_tasks,
                    base_root=base_root,
                    conda_env=conda_env,
                    base_env=base_env,
                    dry_run=dry_run,
                )
            )
        for future in futures:
            for ok in future.result():
                all_ok = all_ok and ok
    return all_ok


def _run_summary(base_root: Path, mode: str, frozen_root: Path, dry_run: bool) -> bool:
    summary_script = _project_root() / "scripts" / "summarize_acl_v1_followups.py"
    cmd = [
        sys.executable,
        str(summary_script),
        "--mode",
        mode,
        "--root",
        str(base_root),
        "--frozen-root",
        str(frozen_root),
    ]
    print(f"[SUMM] {mode} | {' '.join(cmd)}")
    if dry_run:
        return True
    proc = subprocess.run(cmd, cwd=str(_repo_root()))
    return proc.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ACL v1 failure-review and stability-confirm follow-up experiments.")
    parser.add_argument("--mode", choices=["failure_review", "stability_confirm"], required=True)
    parser.add_argument("--results-root")
    parser.add_argument(
        "--frozen-root",
        default="standalone_projects/ACT_ManifoldBridge/results/acl_small_matrix_v1",
    )
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--failure-seeds", default="1,2,3")
    parser.add_argument("--stability-seeds", default="4,5")
    parser.add_argument("--conda-env", default="pia")
    parser.add_argument("--prepend-conda-lib", action="store_true", default=True)
    parser.add_argument("--no-prepend-conda-lib", dest="prepend_conda_lib", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_root = _resolve_results_root(args.mode, args.results_root)
    frozen_root = (_repo_root() / args.frozen_root).resolve()
    devices = _parse_devices(args.devices)
    _write_run_manifest(base_root, devices, args)

    base_env = os.environ.copy()
    if args.prepend_conda_lib:
        prefix = _resolve_conda_env_prefix(args.conda_env)
        if prefix:
            lib_dir = os.path.join(prefix, "lib")
            existing = base_env.get("LD_LIBRARY_PATH", "")
            base_env["LD_LIBRARY_PATH"] = lib_dir if not existing else f"{lib_dir}:{existing}"
            print(f"[ENV ] prepended LD_LIBRARY_PATH with {lib_dir}")
        else:
            print(f"[WARN] could not resolve conda env prefix for {args.conda_env}; skipping LD_LIBRARY_PATH prepend")

    ok = _run_mode(
        mode=args.mode,
        base_root=base_root,
        devices=devices,
        conda_env=args.conda_env,
        base_env=base_env,
        failure_seeds=args.failure_seeds,
        stability_seeds=args.stability_seeds,
        dry_run=args.dry_run,
    )
    if not ok:
        return 1
    ok = _run_summary(base_root, args.mode, frozen_root, args.dry_run)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
