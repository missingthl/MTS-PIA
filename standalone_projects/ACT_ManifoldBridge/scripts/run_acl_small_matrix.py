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
from typing import Dict, List, Optional


STRUCTURE_DATASETS = ["natops", "japanesevowels", "handwriting", "libras"]
VOLATILITY_DATASETS = ["atrialfibrillation", "heartbeat", "motorimagery"]
SANITY_DATASETS = ["basicmotions"]
FULL_DATASETS = STRUCTURE_DATASETS + VOLATILITY_DATASETS + SANITY_DATASETS

MBA_ARM = "mba"
ACL_N4_ARM = "gcg_acl_n4"
ACL_N8_ARM = "gcg_acl_n8"
ARM_ORDER = [MBA_ARM, ACL_N4_ARM, ACL_N8_ARM]


@dataclass(frozen=True)
class TaskSpec:
    phase: str
    arm: str
    dataset: str
    seeds: str
    out_root: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_conda_env_prefix(conda_env: str) -> Optional[str]:
    cmd = ["conda", "env", "list", "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(proc.stdout)
    envs = payload.get("envs", [])
    suffix = os.sep + conda_env
    for env_path in envs:
        if env_path.endswith(suffix):
            return env_path
    return None


def _parse_devices(raw: str) -> List[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or ["cuda:0"]


def _phase_root(base_root: Path, phase: str) -> Path:
    if phase == "preflight":
        return base_root / "preflight"
    if phase == "full":
        return base_root
    raise ValueError(f"Unsupported phase: {phase}")


def _tasks_for_phase(base_root: Path, phase: str, full_seeds: str) -> List[TaskSpec]:
    root = _phase_root(base_root, phase)
    if phase == "preflight":
        datasets = SANITY_DATASETS
        seeds = "1"
    else:
        datasets = FULL_DATASETS
        seeds = full_seeds

    tasks: List[TaskSpec] = []
    for dataset in datasets:
        tasks.append(TaskSpec(phase=phase, arm=MBA_ARM, dataset=dataset, seeds=seeds, out_root=str(root / MBA_ARM / dataset)))
        tasks.append(TaskSpec(phase=phase, arm=ACL_N4_ARM, dataset=dataset, seeds=seeds, out_root=str(root / ACL_N4_ARM / dataset)))
        tasks.append(TaskSpec(phase=phase, arm=ACL_N8_ARM, dataset=dataset, seeds=seeds, out_root=str(root / ACL_N8_ARM / dataset)))
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
    else:
        cmd.extend(
            [
                "--pipeline",
                "gcg_acl",
                "--acl-warmup-epochs",
                "10",
                "--acl-positives-per-anchor",
                "1",
                "--acl-alignment-weight",
                "0.7",
                "--acl-loss-weight",
                "0.2",
                "--acl-temperature",
                "0.07",
                "--acl-candidates-per-anchor",
                "4" if task.arm == ACL_N4_ARM else "8",
            ]
        )

    return cmd


def _configure_device_env(base_env: Dict[str, str], device_spec: str) -> tuple[Dict[str, str], str]:
    env = dict(base_env)
    if device_spec.startswith("cuda:"):
        physical_id = device_spec.split(":", 1)[1]
        env["CUDA_VISIBLE_DEVICES"] = physical_id
        return env, "cuda:0"
    return env, device_spec


def _log_path(base_root: Path, task: TaskSpec) -> Path:
    return base_root / "_logs" / task.phase / task.arm / f"{task.dataset}.log"


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
        "seeds": args.seeds,
        "conda_env": args.conda_env,
        "prepend_conda_lib": bool(args.prepend_conda_lib),
        "datasets": {
            "structure": STRUCTURE_DATASETS,
            "volatility": VOLATILITY_DATASETS,
            "sanity": SANITY_DATASETS,
        },
        "arms": ARM_ORDER,
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
        print(f"[SKIP] {task.phase} | {task.arm} | {task.dataset} | existing final_results.csv")
        return True

    env, torch_device = _configure_device_env(base_env, device_spec)
    cmd = _build_task_command(task, conda_env) + ["--device", torch_device]
    print(f"[RUN ] {task.phase} | {task.arm} | {task.dataset} | device={device_spec}")
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
        print(f"[FAIL] {task.phase} | {task.arm} | {task.dataset} | see {log_path}")
        return False
    print(f"[DONE] {task.phase} | {task.arm} | {task.dataset}")
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
    results = []
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


def _run_phase(
    *,
    phase: str,
    base_root: Path,
    devices: List[str],
    conda_env: str,
    base_env: Dict[str, str],
    seeds: str,
    dry_run: bool,
) -> bool:
    tasks = _tasks_for_phase(base_root, phase, seeds)
    phase_root = _phase_root(base_root, phase)
    phase_root.mkdir(parents=True, exist_ok=True)
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


def _run_summary(base_root: Path, phase: str, fail_on_preflight: bool, dry_run: bool) -> bool:
    summary_script = _project_root() / "scripts" / "summarize_acl_small_matrix.py"
    cmd = [
        sys.executable,
        str(summary_script),
        "--root",
        str(_phase_root(base_root, phase)),
        "--phase",
        phase,
    ]
    if fail_on_preflight and phase == "preflight":
        cmd.append("--fail-on-preflight")
    print(f"[SUMM] {phase} | {' '.join(cmd)}")
    if dry_run:
        return True
    proc = subprocess.run(cmd, cwd=str(_repo_root()))
    return proc.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ACL v1 small-matrix experiments.")
    parser.add_argument("--mode", choices=["preflight", "full", "all"], default="all")
    parser.add_argument(
        "--results-root",
        default="standalone_projects/ACT_ManifoldBridge/results/acl_small_matrix_v1",
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

    phases = [args.mode] if args.mode in {"preflight", "full"} else ["preflight", "full"]
    for phase in phases:
        ok = _run_phase(
            phase=phase,
            base_root=base_root,
            devices=devices,
            conda_env=args.conda_env,
            base_env=base_env,
            seeds=args.seeds,
            dry_run=args.dry_run,
        )
        if not ok:
            return 1
        ok = _run_summary(base_root, phase, fail_on_preflight=True, dry_run=args.dry_run)
        if not ok:
            return 1
        if phase == "preflight" and args.mode == "all":
            print("[OK  ] preflight gates passed; continuing to full matrix")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
