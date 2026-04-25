#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.datasets import AEON_FIXED_SPLIT_SPECS


DEFAULT_OUT_ROOT = PROJECT_ROOT / "results" / "mba_vs_rc4_census_v1" / "resnet1d_sharedbudget_s123"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_value(args: List[str], default: str = "") -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return default


def _git_sha() -> str:
    return _git_value(["git", "rev-parse", "HEAD"], default="unknown")


def _git_dirty() -> bool:
    return bool(_git_value(["git", "status", "--porcelain"]))


@dataclass
class ArmSpec:
    arm: str
    pipeline: str
    algo: str
    extra_args: List[str] = field(default_factory=list)
    direction_bank_source: str = "lraes"
    onthefly_aug: bool = False
    aug_weight_mode: str = "none"
    utilization_mode: str = "core_concat"
    core_training_mode: str = "concat_all"


ARM_SPECS: List[ArmSpec] = [
    ArmSpec(
        arm="mba_core_lraes",
        pipeline="act",
        algo="lraes",
        extra_args=[],
        direction_bank_source="lraes",
        onthefly_aug=False,
        aug_weight_mode="none",
        utilization_mode="core_concat",
        core_training_mode="concat_all",
    ),
    ArmSpec(
        arm="mba_feedback_lraes",
        pipeline="mba_feedback",
        algo="lraes",
        extra_args=[
            "--onthefly-aug",
            "--aug-weight-mode",
            "focal",
            "--tau-max",
            "2.0",
            "--tau-min",
            "0.1",
            "--tau-warmup-ratio",
            "0.3",
        ],
        direction_bank_source="lraes",
        onthefly_aug=True,
        aug_weight_mode="focal",
        utilization_mode="feedback_weighted_ce",
        core_training_mode="n/a",
    ),
    ArmSpec(
        arm="rc4_osf",
        pipeline="mba_feedback",
        algo="adaptive",
        extra_args=[
            "--direction-bank-source",
            "orthogonal_fusion",
            "--onthefly-aug",
            "--aug-weight-mode",
            "focal",
            "--tau-max",
            "2.0",
            "--tau-min",
            "0.1",
            "--tau-warmup-ratio",
            "0.3",
            "--osf-alpha",
            "1.0",
            "--osf-beta",
            "0.5",
            "--osf-kappa",
            "1.0",
        ],
        direction_bank_source="orthogonal_fusion",
        onthefly_aug=True,
        aug_weight_mode="focal",
        utilization_mode="feedback_weighted_ce",
        core_training_mode="n/a",
    ),
    ArmSpec(
        arm="mba_core_rc4_fused_concat",
        pipeline="act",
        algo="rc4_fused",
        extra_args=[
            "--osf-alpha",
            "1.0",
            "--osf-beta",
            "0.5",
            "--osf-kappa",
            "1.0",
        ],
        direction_bank_source="orthogonal_fusion",
        onthefly_aug=False,
        aug_weight_mode="none",
        utilization_mode="core_concat",
        core_training_mode="concat_all",
    ),
    ArmSpec(
        arm="mba_core_spectral_osf_concat",
        pipeline="act",
        algo="spectral_osf",
        extra_args=[
            "--osf-alpha",
            "1.0",
            "--osf-beta",
            "0.5",
            "--osf-kappa",
            "1.0",
            "--spectral-osf-rho",
            "0.90",
        ],
        direction_bank_source="spectral_osf",
        onthefly_aug=False,
        aug_weight_mode="none",
        utilization_mode="core_concat",
        core_training_mode="concat_all",
    ),
]


@dataclass
class Job:
    job_id: int
    dataset: str
    arm: str
    pipeline: str
    algo: str
    out_root: str
    log_path: str
    manifest_path: str
    results_path: str
    command: List[str]
    command_str: str
    status: str = "pending"
    gpu_id: Optional[int] = None
    returncode: Optional[int] = None
    fail_reason: str = ""
    start_time: str = ""
    end_time: str = ""
    duration_sec: Optional[float] = None


def _parse_csv_arg(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _build_command(args, dataset: str, arm_spec: ArmSpec, out_root: Path) -> List[str]:
    cmd = [
        "conda",
        "run",
        "-n",
        "pia",
        "python",
        str(PROJECT_ROOT / "run_act_pilot.py"),
        "--dataset",
        dataset,
        "--model",
        args.model,
        "--pipeline",
        arm_spec.pipeline,
        "--algo",
        arm_spec.algo,
        "--seeds",
        args.seeds,
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--batch-size",
        str(args.batch_size),
        "--patience",
        str(args.patience),
        "--val-ratio",
        str(args.val_ratio),
        "--k-dir",
        str(args.k_dir),
        "--pia-gamma",
        str(args.pia_gamma),
        "--multiplier",
        str(args.multiplier),
        "--device",
        "cuda:0",
        "--out-root",
        str(out_root),
    ]
    cmd.extend(arm_spec.extra_args)
    return cmd


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)


def _job_to_row(job: Job) -> Dict[str, object]:
    return {
        "job_id": job.job_id,
        "dataset": job.dataset,
        "arm": job.arm,
        "pipeline": job.pipeline,
        "algo": job.algo,
        "status": job.status,
        "gpu_id": job.gpu_id,
        "returncode": job.returncode,
        "fail_reason": job.fail_reason,
        "start_time": job.start_time,
        "end_time": job.end_time,
        "duration_sec": job.duration_sec,
        "out_root": job.out_root,
        "results_path": job.results_path,
        "log_path": job.log_path,
        "manifest_path": job.manifest_path,
        "command": job.command_str,
    }


def _write_jobs_manifest(root: Path, jobs: List[Job]) -> None:
    rows = [_job_to_row(job) for job in jobs]
    json_path = root / "jobs_manifest.json"
    csv_path = root / "jobs_manifest.csv"
    _write_json(json_path, {"generated_at": _iso_now(), "jobs": rows})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["job_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_worker_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{_iso_now()}] {message}\n")


def _build_run_manifest(*, args, arm_spec: ArmSpec, dataset: str, gpu_id: Optional[int], job: Job) -> Dict[str, object]:
    return {
        "dataset": dataset,
        "arm": arm_spec.arm,
        "pipeline": arm_spec.pipeline,
        "algo": arm_spec.algo,
        "model": args.model,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "val_ratio": args.val_ratio,
        "loader_seed_rule": "same_as_experiment_seed",
        "multiplier": args.multiplier,
        "k_dir": args.k_dir,
        "pia_gamma": args.pia_gamma,
        "disable_safe_step": False,
        "eta_safe": 0.5,
        "direction_bank_source": arm_spec.direction_bank_source,
        "onthefly_aug": arm_spec.onthefly_aug,
        "aug_weight_mode": arm_spec.aug_weight_mode,
        "utilization_mode": arm_spec.utilization_mode,
        "core_training_mode": arm_spec.core_training_mode,
        "feedback_margin_temperature": 1.0,
        "aug_loss_weight": 1.0,
        "tau_max": 2.0 if arm_spec.onthefly_aug else None,
        "tau_min": 0.1 if arm_spec.onthefly_aug else None,
        "tau_warmup_ratio": 0.3 if arm_spec.onthefly_aug else None,
        "router_temperature": 0.05 if arm_spec.arm == "rc4_osf" else None,
        "router_min_prob": 0.10 if arm_spec.arm == "rc4_osf" else None,
        "router_smoothing": 0.5 if arm_spec.arm == "rc4_osf" else None,
        "router_reward": "feedback_weight" if arm_spec.arm == "rc4_osf" else None,
        "osf_alpha": 1.0 if arm_spec.arm in {"rc4_osf", "mba_core_rc4_fused_concat"} else None,
        "osf_beta": 0.5 if arm_spec.arm in {"rc4_osf", "mba_core_rc4_fused_concat"} else None,
        "osf_kappa": 1.0 if arm_spec.arm in {"rc4_osf", "mba_core_rc4_fused_concat"} else None,
        "git_commit_sha": args.git_commit_sha,
        "git_is_dirty": args.git_is_dirty,
        "physical_gpu_id": gpu_id,
        "command": job.command_str,
        "out_root": job.out_root,
        "results_path": job.results_path,
        "job_id": job.job_id,
        "status": job.status,
        "start_time": job.start_time,
        "end_time": job.end_time,
        "duration_sec": job.duration_sec,
        "returncode": job.returncode,
        "fail_reason": job.fail_reason,
    }


def _make_jobs(args, arm_specs: List[ArmSpec]) -> List[Job]:
    jobs: List[Job] = []
    out_root = Path(args.out_root).resolve()
    log_root = out_root / "logs"
    job_id = 0
    for dataset in args.datasets:
        for arm_spec in arm_specs:
            arm_out_root = out_root / arm_spec.arm / dataset
            log_path = log_root / f"{dataset}_{arm_spec.arm}.log"
            manifest_path = arm_out_root / "run_manifest.json"
            results_path = arm_out_root / f"{dataset}_results.csv"
            command = _build_command(args, dataset, arm_spec, arm_out_root)
            jobs.append(
                Job(
                    job_id=job_id,
                    dataset=dataset,
                    arm=arm_spec.arm,
                    pipeline=arm_spec.pipeline,
                    algo=arm_spec.algo,
                    out_root=str(arm_out_root),
                    log_path=str(log_path),
                    manifest_path=str(manifest_path),
                    results_path=str(results_path),
                    command=command,
                    command_str=" ".join(command),
                )
            )
            job_id += 1
    return jobs


def _run_worker(gpu_id: int, jobs_q: "queue.Queue[Job]", jobs: List[Job], args, state_lock: threading.Lock) -> None:
    worker_log = Path(args.out_root).resolve() / "logs" / f"gpu{gpu_id}_worker.log"
    while True:
        try:
            job = jobs_q.get_nowait()
        except queue.Empty:
            _append_worker_log(worker_log, "worker idle -> queue empty, exiting")
            return

        with state_lock:
            job.status = "running"
            job.gpu_id = gpu_id
            job.start_time = _iso_now()
            arm_spec = next(spec for spec in ARM_SPECS if spec.arm == job.arm)
            _write_json(Path(job.manifest_path), _build_run_manifest(args=args, arm_spec=arm_spec, dataset=job.dataset, gpu_id=gpu_id, job=job))
            _write_jobs_manifest(Path(args.out_root), jobs)

        _append_worker_log(worker_log, f"START job_id={job.job_id} dataset={job.dataset} arm={job.arm}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        Path(job.log_path).parent.mkdir(parents=True, exist_ok=True)
        start_perf = time.time()
        returncode = 1
        fail_reason = ""
        try:
            with open(job.log_path, "w", encoding="utf-8") as log_f:
                proc = subprocess.run(
                    job.command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            returncode = int(proc.returncode)
            if returncode != 0:
                fail_reason = f"subprocess_returncode_{returncode}"
        except Exception as exc:
            fail_reason = repr(exc)
        duration = time.time() - start_perf

        with state_lock:
            job.returncode = returncode
            job.end_time = _iso_now()
            job.duration_sec = duration
            if returncode == 0 and Path(job.results_path).is_file():
                job.status = "success"
            else:
                job.status = "failed"
                job.fail_reason = fail_reason or "missing_results_file"
            arm_spec = next(spec for spec in ARM_SPECS if spec.arm == job.arm)
            _write_json(Path(job.manifest_path), _build_run_manifest(args=args, arm_spec=arm_spec, dataset=job.dataset, gpu_id=gpu_id, job=job))
            _write_jobs_manifest(Path(args.out_root), jobs)

        _append_worker_log(
            worker_log,
            f"END job_id={job.job_id} dataset={job.dataset} arm={job.arm} status={job.status} duration_sec={duration:.2f}",
        )
        jobs_q.task_done()


def _print_dry_run(jobs: List[Job], gpus: List[int]) -> None:
    print(f"Dry-run: {len(jobs)} jobs across GPUs {gpus}")
    for idx, job in enumerate(jobs):
        gpu_id = gpus[idx % len(gpus)]
        print(f"[gpu{gpu_id}] job_id={job.job_id} dataset={job.dataset} arm={job.arm}")
        print(f"  out_root={job.out_root}")
        print(f"  log={job.log_path}")
        print(f"  cmd={job.command_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 20-dataset MBA vs RC-4 matrix with 4-GPU queue scheduling.")
    parser.add_argument("--out-root", type=str, default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--datasets", type=str, default=",".join(sorted(AEON_FIXED_SPLIT_SPECS.keys())))
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--model", type=str, default="resnet1d")
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--k-dir", type=int, default=10)
    parser.add_argument("--pia-gamma", type=float, default=0.1)
    parser.add_argument("--multiplier", type=int, default=10)
    parser.add_argument("--actual-arms", type=str, default=",".join(spec.arm for spec in ARM_SPECS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.datasets = sorted(_parse_csv_arg(args.datasets))
    args.gpus = [int(x) for x in _parse_csv_arg(args.gpus)]
    requested_arms = set(_parse_csv_arg(args.actual_arms))
    arm_specs = [spec for spec in ARM_SPECS if spec.arm in requested_arms]
    missing_arms = sorted(requested_arms - {spec.arm for spec in ARM_SPECS})
    if missing_arms:
        raise ValueError(f"Unknown --actual-arms entries: {missing_arms}")
    if not arm_specs:
        raise ValueError("No actual arms selected.")
    args.git_commit_sha = _git_sha()
    args.git_is_dirty = _git_dirty()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)

    jobs = _make_jobs(args, arm_specs)
    matrix_manifest = {
        "created_at": _iso_now(),
        "git_commit_sha": args.git_commit_sha,
        "git_is_dirty": args.git_is_dirty,
        "datasets": args.datasets,
        "gpus": args.gpus,
        "actual_arms": [asdict(spec) for spec in arm_specs],
        "shared_budget": {
            "model": args.model,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "val_ratio": args.val_ratio,
            "k_dir": args.k_dir,
            "pia_gamma": args.pia_gamma,
            "multiplier": args.multiplier,
            "disable_safe_step": False,
            "eta_safe": 0.5,
        },
        "job_count": len(jobs),
    }
    _write_json(out_root / "matrix_manifest.json", matrix_manifest)
    _write_jobs_manifest(out_root, jobs)

    if args.dry_run:
        _print_dry_run(jobs, args.gpus)
        return

    jobs_q: "queue.Queue[Job]" = queue.Queue()
    for job in jobs:
        jobs_q.put(job)

    state_lock = threading.Lock()
    threads = []
    for gpu_id in args.gpus:
        t = threading.Thread(target=_run_worker, args=(gpu_id, jobs_q, jobs, args, state_lock), daemon=False)
        threads.append(t)
        t.start()
        time.sleep(1.0)

    for t in threads:
        t.join()

    _write_jobs_manifest(out_root, jobs)
    failed = [job for job in jobs if job.status != "success"]
    print(f"Matrix run complete. success={len(jobs)-len(failed)} failed={len(failed)}")
    if failed:
        print("Failed jobs:")
        for job in failed:
            print(f"- job_id={job.job_id} dataset={job.dataset} arm={job.arm} reason={job.fail_reason}")


if __name__ == "__main__":
    main()
