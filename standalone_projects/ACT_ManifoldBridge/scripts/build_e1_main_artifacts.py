#!/usr/bin/env python3
"""Build auditable E1 artifacts from existing per-seed experiment CSV files.

This script intentionally does not run experiments.  It consolidates completed
per-seed runs into atomic audit tables, then derives the paper-facing E1 table
from those atoms.  Missing or subset methods are kept visible instead of being
silently dropped.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
OUT_DIR = RESULTS_DIR / "e1_main"
AUDIT_DOC = PROJECT_DIR / "docs" / "E1_DATA_AUDIT.md"

EXPERIMENT_TAG = "E1_main_v1"
FINAL20_DATASETS = (
    "articularywordrecognition",
    "atrialfibrillation",
    "basicmotions",
    "cricket",
    "epilepsy",
    "ering",
    "ethanolconcentration",
    "fingermovements",
    "handmovementdirection",
    "handwriting",
    "har",
    "heartbeat",
    "japanesevowels",
    "libras",
    "motorimagery",
    "natops",
    "pendigits",
    "racketsports",
    "selfregulationscp2",
    "uwavegesturelibrary",
)
SEEDS = ("1", "2", "3")
BACKBONE = "resnet1d"

E1_METHODS = (
    "no_aug",
    "raw_aug_jitter",
    "raw_aug_timewarp",
    "raw_mixup",
    "timegan_classwise",
    "diffusionts_classwise",
    "dba_sameclass",
    "wdba_sameclass",
    "rgw_sameclass",
    "dgw_sameclass",
    "csta_topk_uniform_top5",
)

E1_EXCLUDED = (
    "timevae_classwise_optional",
    "raw_aug_magnitude_warping",
    "raw_aug_window_slicing",
    "raw_aug_window_warping",
    "manifold_mixup",
    "raw_smote_flatten_balanced",
    "spawner_sameclass_style",
    "jobda_cleanroom",
    "random_cov_state",
    "pca_cov_state",
)


@dataclass(frozen=True)
class SourceSpec:
    path: str
    priority: int
    label: str


SOURCE_SPECS = (
    SourceSpec("full_scale_resnet1d_v1/per_seed_external.csv", 10, "full_scale_resnet1d_v1"),
    SourceSpec("csta_external_baselines_phase1/resnet1d_s123/per_seed_external.csv", 20, "phase1_pilot7"),
    SourceSpec("final20_minimal_baseline_v1/resnet1d_s123/per_seed_external.csv", 30, "final20_minimal_baseline"),
    SourceSpec("final20_addendum_mixup_v1/resnet1d_s123/per_seed_external.csv", 40, "final20_mixup_addendum"),
    SourceSpec("wdba_final20/resnet1d_s123/per_seed_external.csv", 50, "wdba_final20"),
    SourceSpec("csta_external_baselines_phase2_new/resnet1d_s123/per_seed_external.csv", 60, "phase2_guided_warp_pilot7"),
    SourceSpec("csta_external_baselines_local/resnet1d_s123/diffusionts/per_seed_external.csv", 70, "diffusionts_local_subset"),
    SourceSpec("csta_external_baselines_local/resnet1d_s123/diffusionts_recovery/per_seed_external.csv", 75, "diffusionts_recovery_subset"),
    SourceSpec("csta_pia_final20/resnet1d_s123/per_seed_external.csv", 80, "csta_u5_final20"),
)


METHOD_META: Dict[str, Dict[str, object]] = {
    "no_aug": {
        "display": "No-Aug",
        "family": "Reference",
        "source_level": "reference",
        "paper_title": "",
        "paper_year": "",
        "paper_venue": "",
        "paper_url": "",
        "code_url": "",
        "implementation_status": "native reference",
        "protocol": "backbone-only training",
        "cost_type": "none",
        "requires_generator": False,
        "requires_soft_label": False,
        "requires_hidden_state": False,
        "requires_alignment": False,
        "outputs_raw_samples": False,
        "is_proposed": False,
        "is_external_baseline": False,
        "notes": "Reference row used for deltas and W/T/L.",
    },
    "raw_aug_jitter": {
        "display": "Jitter",
        "family": "Temporal / vicinal heuristic",
        "source_level": "standard transform baseline",
        "paper_title": "An Empirical Survey of Data Augmentation for Time Series Classification with Neural Networks",
        "paper_year": "2021",
        "paper_venue": "PLOS ONE",
        "paper_url": "https://arxiv.org/abs/2007.15951",
        "code_url": "https://github.com/uchidalab/time_series_augmentation",
        "implementation_status": "native/tsaug-style adapter",
        "protocol": "offline hard-label raw time perturbation",
        "cost_type": "offline-transform",
        "requires_generator": False,
        "requires_soft_label": False,
        "requires_hidden_state": False,
        "requires_alignment": False,
        "outputs_raw_samples": True,
        "is_proposed": False,
        "is_external_baseline": True,
        "notes": "Standard transform baseline, not one standalone method paper per transform.",
    },
    "raw_aug_timewarp": {
        "display": "TimeWarp",
        "family": "Temporal / vicinal heuristic",
        "source_level": "standard transform baseline",
        "paper_title": "An Empirical Survey of Data Augmentation for Time Series Classification with Neural Networks",
        "paper_year": "2021",
        "paper_venue": "PLOS ONE",
        "paper_url": "https://arxiv.org/abs/2007.15951",
        "code_url": "https://github.com/uchidalab/time_series_augmentation",
        "implementation_status": "tsaug-style adapter",
        "protocol": "offline hard-label temporal warping",
        "cost_type": "offline-transform",
        "requires_generator": False,
        "requires_soft_label": False,
        "requires_hidden_state": False,
        "requires_alignment": False,
        "outputs_raw_samples": True,
        "is_proposed": False,
        "is_external_baseline": True,
        "notes": "Standard transform baseline.",
    },
    "raw_mixup": {
        "display": "Mixup",
        "family": "Temporal / vicinal heuristic",
        "source_level": "ICLR 2018",
        "paper_title": "mixup: Beyond Empirical Risk Minimization",
        "paper_year": "2018",
        "paper_venue": "ICLR",
        "paper_url": "https://arxiv.org/abs/1710.09412",
        "code_url": "https://github.com/facebookresearch/mixup-cifar10",
        "implementation_status": "native soft-label trainer following Mixup",
        "protocol": "training-time soft-label vicinal training",
        "cost_type": "training-time",
        "requires_generator": False,
        "requires_soft_label": True,
        "requires_hidden_state": False,
        "requires_alignment": False,
        "outputs_raw_samples": False,
        "is_proposed": False,
        "is_external_baseline": True,
        "notes": "No standalone offline augmentation artifact; augmentation cost may be N/A.",
    },
    "timegan_classwise": {
        "display": "TimeGAN",
        "family": "Deep generative",
        "source_level": "NeurIPS 2019",
        "paper_title": "Time-series Generative Adversarial Networks",
        "paper_year": "2019",
        "paper_venue": "NeurIPS",
        "paper_url": "https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks",
        "code_url": "https://github.com/jsyoon0823/TimeGAN",
        "implementation_status": "native style adapter pending full E1 runs",
        "protocol": "classwise GAN generator fitting + sampling",
        "cost_type": "generator-fit",
        "requires_generator": True,
        "requires_soft_label": False,
        "requires_hidden_state": False,
        "requires_alignment": False,
        "outputs_raw_samples": True,
        "is_proposed": False,
        "is_external_baseline": True,
        "notes": "E1 target GAN-class representative; no completed Final20 rows found at build time.",
    },
    "diffusionts_classwise": {
        "display": "Diffusion-TS",
        "family": "Deep generative",
        "source_level": "ICLR 2024",
        "paper_title": "Diffusion-TS: Interpretable Diffusion for General Time Series Generation",
        "paper_year": "2024",
        "paper_venue": "ICLR",
        "paper_url": "https://openreview.net/forum?id=4h1apFjO99",
        "code_url": "https://github.com/Y-debug-sys/Diffusion-TS",
        "implementation_status": "external-code adapter",
        "protocol": "classwise diffusion generator fitting + sampling",
        "cost_type": "generator-fit",
        "requires_generator": True,
        "requires_soft_label": False,
        "requires_hidden_state": False,
        "requires_alignment": False,
        "outputs_raw_samples": True,
        "is_proposed": False,
        "is_external_baseline": True,
        "notes": "Generative cost-utility stress test; current rows are subset coverage.",
    },
    "dba_sameclass": {
        "display": "DBA",
        "family": "Analytical / alignment preserving",
        "source_level": "Pattern Recognition 2011 / tslearn",
        "paper_title": "A global averaging method for dynamic time warping, with applications to clustering",
        "paper_year": "2011",
        "paper_venue": "Pattern Recognition",
        "paper_url": "https://doi.org/10.1016/j.patcog.2010.09.013",
        "code_url": "https://tslearn.readthedocs.io/en/stable/gen_modules/barycenters/tslearn.barycenters.dtw_barycenter_averaging.html",
        "implementation_status": "tslearn adapter",
        "protocol": "same-class DTW barycenter synthesis",
        "cost_type": "dtw-alignment",
        "requires_generator": False,
        "requires_soft_label": False,
        "requires_hidden_state": False,
        "requires_alignment": True,
        "outputs_raw_samples": True,
        "is_proposed": False,
        "is_external_baseline": True,
        "notes": "Library implementation of DBA-family synthesis.",
    },
    "wdba_sameclass": {
        "display": "wDBA",
        "family": "Analytical / alignment preserving",
        "source_level": "weighted DBA implementation",
        "paper_title": "A global averaging method for dynamic time warping, with applications to clustering",
        "paper_year": "2011",
        "paper_venue": "Pattern Recognition",
        "paper_url": "https://doi.org/10.1016/j.patcog.2010.09.013",
        "code_url": "https://tslearn.readthedocs.io/en/stable/gen_modules/barycenters/tslearn.barycenters.dtw_barycenter_averaging.html",
        "implementation_status": "weighted tslearn DBA adapter",
        "protocol": "anchor-weighted same-class DTW barycenter synthesis",
        "cost_type": "dtw-alignment",
        "requires_generator": False,
        "requires_soft_label": False,
        "requires_hidden_state": False,
        "requires_alignment": True,
        "outputs_raw_samples": True,
        "is_proposed": False,
        "is_external_baseline": True,
        "notes": "Weighted DBA-family implementation, not claimed as separate canonical paper.",
    },
    "rgw_sameclass": {
        "display": "RGW",
        "family": "Analytical / alignment preserving",
        "source_level": "ICPR 2020 / survey-listed",
        "paper_title": "Time Series Data Augmentation for Neural Networks by Time Warping with a Discriminative Teacher",
        "paper_year": "2020",
        "paper_venue": "ICPR",
        "paper_url": "https://ieeexplore.ieee.org/document/9413168",
        "code_url": "https://github.com/uchidalab/time_series_augmentation",
        "implementation_status": "clean-room guided warping adapter",
        "protocol": "same-class guided warping",
        "cost_type": "dtw-alignment",
        "requires_generator": False,
        "requires_soft_label": False,
        "requires_hidden_state": False,
        "requires_alignment": True,
        "outputs_raw_samples": True,
        "is_proposed": False,
        "is_external_baseline": True,
        "notes": "Current implementation is a clean-room adapter, not an official code reproduction.",
    },
    "dgw_sameclass": {
        "display": "DGW",
        "family": "Analytical / alignment preserving",
        "source_level": "ICPR 2020 / survey-listed",
        "paper_title": "Time Series Data Augmentation for Neural Networks by Time Warping with a Discriminative Teacher",
        "paper_year": "2020",
        "paper_venue": "ICPR",
        "paper_url": "https://ieeexplore.ieee.org/document/9413168",
        "code_url": "https://github.com/uchidalab/time_series_augmentation",
        "implementation_status": "clean-room discriminative guided warping adapter",
        "protocol": "discriminative guided warping",
        "cost_type": "dtw-alignment",
        "requires_generator": False,
        "requires_soft_label": False,
        "requires_hidden_state": False,
        "requires_alignment": True,
        "outputs_raw_samples": True,
        "is_proposed": False,
        "is_external_baseline": True,
        "notes": "Current implementation is a clean-room adapter, not an official code reproduction.",
    },
    "csta_topk_uniform_top5": {
        "display": "CoSTA-U5",
        "family": "Analytical / alignment preserving",
        "source_level": "proposed",
        "paper_title": "CoSTA / PIA",
        "paper_year": "",
        "paper_venue": "proposed",
        "paper_url": "",
        "code_url": "",
        "implementation_status": "native proposed method",
        "protocol": "train-only covariance-state proposal + safe-step + whitening-coloring realization",
        "cost_type": "covariance-bridge",
        "requires_generator": False,
        "requires_soft_label": False,
        "requires_hidden_state": False,
        "requires_alignment": False,
        "outputs_raw_samples": True,
        "is_proposed": True,
        "is_external_baseline": False,
        "notes": "Canonical U5 configuration; RandomCov/PCACov are internal controls and excluded from E1.",
    },
}


PER_SEED_FIELDS = [
    "run_id",
    "timestamp",
    "git_commit",
    "experiment_tag",
    "dataset",
    "dataset_group",
    "seed",
    "split_id",
    "split_protocol",
    "backbone",
    "backbone_config_id",
    "method",
    "method_display_name",
    "family",
    "source_level",
    "protocol",
    "cost_type",
    "target_aug_ratio",
    "actual_aug_ratio",
    "n_train_original",
    "n_aug_generated",
    "n_train_effective",
    "generation_success",
    "failure_reason",
    "macro_f1",
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "aug_cost_sec",
    "method_cost_sec",
    "generator_fit_sec",
    "sample_gen_sec",
    "dtw_alignment_sec",
    "cov_state_compute_sec",
    "bridge_realization_sec",
    "downstream_train_sec",
    "eval_sec",
    "total_pipeline_sec",
    "total_sec",
    "peak_cpu_mem_mb",
    "peak_gpu_mem_mb",
    "device",
    "num_threads",
    "config_path",
    "result_path",
    "log_path",
    "artifact_path",
]

AUG_ARTIFACT_FIELDS = [
    "dataset",
    "seed",
    "method",
    "target_aug_ratio",
    "actual_aug_ratio",
    "n_train_original",
    "n_aug_target",
    "n_aug_generated",
    "n_aug_accepted",
    "n_aug_rejected",
    "generation_success_rate",
    "class_balance_before",
    "class_balance_after",
    "artifact_hash",
    "artifact_path",
    "failure_reason",
]

COST_FIELDS = [
    "dataset",
    "seed",
    "method",
    "aug_preprocess_sec",
    "generator_fit_sec",
    "sample_gen_sec",
    "dtw_alignment_sec",
    "cov_state_compute_sec",
    "bridge_realization_sec",
    "aug_postprocess_sec",
    "downstream_train_sec",
    "eval_sec",
    "total_pipeline_sec",
    "method_cost_sec",
    "total_sec",
    "peak_cpu_mem_mb",
    "peak_gpu_mem_mb",
    "device",
    "num_threads",
]


def _float(value: object, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in fieldnames})


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_DIR.parents[1], text=True).strip()
    except Exception:
        return "unknown"


def _source_timestamp(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return ""


def _hash_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _nearest_config_path(result_csv: Path) -> str:
    cur = result_csv.parent
    for _ in range(4):
        candidate = cur / "run_config.json"
        if candidate.exists():
            return str(candidate.relative_to(PROJECT_DIR))
        cur = cur.parent
    return ""


def _status_success(row: Mapping[str, str]) -> bool:
    status = (row.get("status") or "").lower()
    return status in {"", "success", "ok"} and not row.get("fail_reason")


def _artifact_path(row: Mapping[str, str]) -> str:
    p = row.get("candidate_audit_path", "")
    if p:
        return p
    return ""


def _cost_partition(method: str, row: Mapping[str, str]) -> Dict[str, object]:
    total = _float(row.get("method_elapsed_sec"))
    out: Dict[str, object] = {
        "aug_cost_sec": "",
        "generator_fit_sec": "",
        "sample_gen_sec": "",
        "dtw_alignment_sec": "",
        "cov_state_compute_sec": "",
        "bridge_realization_sec": "",
        "downstream_train_sec": "",
        "eval_sec": "",
        "total_pipeline_sec": total if not math.isnan(total) else "",
    }
    if method == "no_aug":
        out["aug_cost_sec"] = 0.0
    elif method == "raw_mixup":
        # Mixup is training-time vicinal augmentation; no standalone offline cost.
        out["aug_cost_sec"] = ""
    elif method in {"timegan_classwise", "diffusionts_classwise"}:
        out["generator_fit_sec"] = _float(row.get("generator_fit_sec"), math.nan)
        out["sample_gen_sec"] = _float(row.get("sample_gen_sec"), math.nan)
        if not math.isnan(total):
            out["aug_cost_sec"] = total
    elif method in {"dba_sameclass", "wdba_sameclass", "rgw_sameclass", "dgw_sameclass"}:
        # Current runners expose method_elapsed_sec but not isolated DTW cost.
        if not math.isnan(total):
            out["dtw_alignment_sec"] = total
            out["aug_cost_sec"] = total
    elif method == "csta_topk_uniform_top5":
        cov = _float(row.get("cov_state_compute_sec"))
        bridge = _float(row.get("bridge_realization_sec"))
        if not math.isnan(cov):
            out["cov_state_compute_sec"] = cov
        if not math.isnan(bridge):
            out["bridge_realization_sec"] = bridge
        if not math.isnan(cov) or not math.isnan(bridge):
            out["aug_cost_sec"] = (0.0 if math.isnan(cov) else cov) + (0.0 if math.isnan(bridge) else bridge)
        elif not math.isnan(total):
            out["aug_cost_sec"] = total
    elif not math.isnan(total):
        out["aug_cost_sec"] = total
    return out


def _with_cost_aliases(cost: Mapping[str, object]) -> Dict[str, object]:
    """Expose the E1 paper-facing names while preserving legacy audit fields."""

    out = dict(cost)
    out["method_cost_sec"] = out.get("aug_cost_sec", "")
    out["total_sec"] = out.get("total_pipeline_sec", "")
    return out


def _collect_source_rows() -> List[Tuple[SourceSpec, Path, Dict[str, str]]]:
    rows: List[Tuple[SourceSpec, Path, Dict[str, str]]] = []
    for spec in SOURCE_SPECS:
        path = RESULTS_DIR / spec.path
        for row in _read_csv(path):
            method = row.get("method", "")
            if method not in E1_METHODS:
                continue
            if row.get("backbone", BACKBONE) != BACKBONE:
                continue
            rows.append((spec, path, row))
    return rows


def _dedupe_rows(source_rows: Sequence[Tuple[SourceSpec, Path, Dict[str, str]]]) -> List[Tuple[SourceSpec, Path, Dict[str, str]]]:
    best: Dict[Tuple[str, str, str, str], Tuple[int, SourceSpec, Path, Dict[str, str]]] = {}
    for spec, path, row in source_rows:
        key = (
            row.get("dataset", ""),
            row.get("seed", ""),
            row.get("backbone", BACKBONE),
            row.get("method", ""),
        )
        if not all(key):
            continue
        current = best.get(key)
        if current is None or spec.priority >= current[0]:
            best[key] = (spec.priority, spec, path, row)
    return [(spec, path, row) for _, spec, path, row in sorted(best.values(), key=lambda x: (x[2].as_posix(), x[3].get("dataset", ""), x[3].get("seed", ""), x[3].get("method", "")))]


def _infer_n_train(row: Mapping[str, str], dataset_registry: Mapping[str, Mapping[str, object]]) -> int:
    aug_count = _float(row.get("aug_count"))
    actual_ratio = _float(row.get("actual_aug_ratio"))
    if not math.isnan(aug_count) and not math.isnan(actual_ratio) and actual_ratio > 0:
        return int(round(aug_count / actual_ratio))
    ds = row.get("dataset", "")
    try:
        return int(dataset_registry.get(ds, {}).get("train_size", 0))
    except Exception:
        return 0


def _build_per_seed_rows(
    rows: Sequence[Tuple[SourceSpec, Path, Dict[str, str]]],
    dataset_registry: Mapping[str, Mapping[str, object]],
) -> List[Dict[str, object]]:
    git = _git_commit()
    out: List[Dict[str, object]] = []
    for spec, path, row in rows:
        method = row.get("method", "")
        meta = METHOD_META[method]
        dataset = row.get("dataset", "")
        seed = row.get("seed", "")
        aug_count = int(_float(row.get("aug_count"), 0.0))
        n_train = _infer_n_train(row, dataset_registry)
        actual_ratio = _float(row.get("actual_aug_ratio"), 0.0)
        target_ratio = _float(row.get("target_aug_ratio"), _float(row.get("aug_ratio"), actual_ratio))
        cost = _with_cost_aliases(_cost_partition(method, row))
        artifact = _artifact_path(row)
        result_rel = str(path.relative_to(PROJECT_DIR))
        source_config_path = _nearest_config_path(path)
        success = _status_success(row)
        out.append(
            {
                "run_id": f"{EXPERIMENT_TAG}:{dataset}:s{seed}:{BACKBONE}:{method}",
                "timestamp": _source_timestamp(path),
                "git_commit": git,
                "experiment_tag": EXPERIMENT_TAG,
                "dataset": dataset,
                "dataset_group": "Final20" if dataset in FINAL20_DATASETS else "Subset",
                "seed": seed,
                "split_id": f"{dataset}_seed{seed}",
                "split_protocol": "official UEA train/test with train-only internal validation split",
                "backbone": row.get("backbone", BACKBONE),
                "backbone_config_id": "resnet1d_default",
                "method": method,
                "method_display_name": meta["display"],
                "family": meta["family"],
                "source_level": meta["source_level"],
                "protocol": meta["protocol"],
                "cost_type": meta["cost_type"],
                "target_aug_ratio": target_ratio,
                "actual_aug_ratio": actual_ratio,
                "n_train_original": n_train,
                "n_aug_generated": aug_count,
                "n_train_effective": n_train + aug_count,
                "generation_success": success,
                "failure_reason": row.get("fail_reason", ""),
                "macro_f1": _float(row.get("aug_f1")),
                "accuracy": row.get("accuracy", ""),
                "balanced_accuracy": row.get("balanced_accuracy", ""),
                "precision_macro": row.get("precision_macro", ""),
                "recall_macro": row.get("recall_macro", ""),
                "peak_cpu_mem_mb": row.get("peak_cpu_mem_mb", ""),
                "peak_gpu_mem_mb": row.get("peak_gpu_mem_mb", ""),
                "device": row.get("device", "cuda" if row.get("backbone", BACKBONE) == BACKBONE else ""),
                "num_threads": row.get("num_threads", ""),
                "config_path": source_config_path,
                "result_path": result_rel,
                "log_path": "",
                "artifact_path": artifact,
                "_source_label": spec.label,
                **cost,
            }
        )
    return out


def _materialize_reconstructed_configs(per_seed_rows: List[Dict[str, object]]) -> None:
    """Write one compact config JSON per E1 atomic row.

    Several historical roots predate strict per-run config capture.  The
    reconstructed config preserves the normalized E1 identity plus the original
    source CSV/config path, so every atomic row has a concrete config artifact.
    """

    config_dir = OUT_DIR / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for row in per_seed_rows:
        run_id = str(row["run_id"])
        file_id = hashlib.sha1(run_id.encode("utf-8")).hexdigest()[:16]
        path = config_dir / f"{file_id}.json"
        payload = {
            "run_id": run_id,
            "experiment_tag": row.get("experiment_tag", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "split_protocol": row.get("split_protocol", ""),
            "backbone": row.get("backbone", ""),
            "backbone_config_id": row.get("backbone_config_id", ""),
            "method": row.get("method", ""),
            "method_display_name": row.get("method_display_name", ""),
            "family": row.get("family", ""),
            "protocol": row.get("protocol", ""),
            "target_aug_ratio": row.get("target_aug_ratio", ""),
            "actual_aug_ratio": row.get("actual_aug_ratio", ""),
            "source_result_path": row.get("result_path", ""),
            "source_config_path": row.get("config_path", ""),
            "source_label": row.get("_source_label", ""),
            "reconstructed_by": "scripts/build_e1_main_artifacts.py",
            "reconstruction_note": "Historical rows may not have had per-run config JSON; this file records the normalized E1 config and source pointers.",
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        row["config_path"] = str(path.relative_to(PROJECT_DIR))


def _parse_ts_metadata(path: Path) -> Tuple[int, int, bool, bool, Dict[str, int]]:
    """Return n_cases, seq_len, variable_length, missing, label counts."""

    n_cases = 0
    seq_lens: List[int] = []
    n_channels_seen: List[int] = []
    labels: Counter[str] = Counter()
    in_data = False
    missing = False
    with path.open(errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            lower = line.lower()
            if lower.startswith("@data"):
                in_data = True
                continue
            if not in_data or line.startswith("@"):
                continue
            n_cases += 1
            if "?" in line or "nan" in lower:
                missing = True
            parts = line.split(":")
            if len(parts) < 2:
                continue
            label = parts[-1].strip()
            labels[label] += 1
            dims = parts[:-1]
            n_channels_seen.append(len(dims))
            dim_lens = []
            for dim in dims:
                dim = dim.strip()
                if not dim:
                    dim_lens.append(0)
                else:
                    dim_lens.append(len([x for x in dim.split(",") if x != ""]))
            if dim_lens:
                seq_lens.append(max(dim_lens))
    variable = len(set(seq_lens)) > 1
    seq_len = max(seq_lens) if seq_lens else 0
    n_channels = max(n_channels_seen) if n_channels_seen else 0
    return n_cases, seq_len, variable, missing, dict(labels), n_channels


def _find_ts_pair(dataset: str) -> Tuple[Optional[Path], Optional[Path]]:
    root = PROJECT_DIR / "data" / "UEA30_aeon"
    candidates = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    ds_dir = candidates.get(dataset.lower())
    if ds_dir is None:
        # HAR is a project-local alias, not part of the canonical UEA30 tree.
        return None, None
    train = next(iter(sorted(ds_dir.glob("*_TRAIN.ts"))), None)
    test = next(iter(sorted(ds_dir.glob("*_TEST.ts"))), None)
    return train, test


def _build_dataset_registry(datasets: Iterable[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset in sorted(set(datasets)):
        train_ts, test_ts = _find_ts_pair(dataset)
        notes = []
        if train_ts is None or test_ts is None:
            rows.append(
                {
                    "dataset": dataset,
                    "source_archive": "unknown/local alias",
                    "n_classes": "",
                    "n_channels": "",
                    "seq_len": "",
                    "train_size": "",
                    "test_size": "",
                    "is_variable_length": "",
                    "has_missing_values": "",
                    "normalization": "per-run trainer normalization",
                    "label_distribution": "",
                    "final20_included": dataset in FINAL20_DATASETS,
                    "notes": "Dataset files not found in data/UEA30_aeon; likely project-local alias or legacy root.",
                }
            )
            continue
        n_train, seq_train, var_train, miss_train, labels_train, c_train = _parse_ts_metadata(train_ts)
        n_test, seq_test, var_test, miss_test, labels_test, c_test = _parse_ts_metadata(test_ts)
        labels_all = Counter(labels_train)
        labels_all.update(labels_test)
        if train_ts.parent.name.lower() != dataset.lower():
            notes.append(f"resolved_to={train_ts.parent.name}")
        rows.append(
            {
                "dataset": dataset,
                "source_archive": "UEA30_aeon",
                "n_classes": len(labels_all),
                "n_channels": max(c_train, c_test),
                "seq_len": max(seq_train, seq_test),
                "train_size": n_train,
                "test_size": n_test,
                "is_variable_length": var_train or var_test,
                "has_missing_values": miss_train or miss_test,
                "normalization": "per-run trainer normalization",
                "label_distribution": json.dumps(dict(sorted(labels_all.items())), sort_keys=True),
                "final20_included": dataset in FINAL20_DATASETS,
                "notes": "; ".join(notes),
            }
        )
    return rows


def _build_method_registry() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for method in E1_METHODS:
        meta = METHOD_META[method]
        rows.append(
            {
                "method": method,
                "method_display_name": meta["display"],
                "family": meta["family"],
                "source_level": meta["source_level"],
                "paper_title": meta["paper_title"],
                "paper_year": meta["paper_year"],
                "paper_venue": meta["paper_venue"],
                "paper_url": meta["paper_url"],
                "code_url": meta["code_url"],
                "implementation_status": meta["implementation_status"],
                "protocol": meta["protocol"],
                "cost_type": meta["cost_type"],
                "requires_generator": meta["requires_generator"],
                "requires_soft_label": meta["requires_soft_label"],
                "requires_hidden_state": meta["requires_hidden_state"],
                "requires_alignment": meta["requires_alignment"],
                "outputs_raw_samples": meta["outputs_raw_samples"],
                "is_proposed": meta["is_proposed"],
                "is_external_baseline": meta["is_external_baseline"],
                "notes": meta["notes"],
            }
        )
    return rows


def _build_aug_artifacts(per_seed_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in per_seed_rows:
        n_aug = int(_float(row.get("n_aug_generated"), 0.0))
        success = str(row.get("generation_success", "")).lower() == "true"
        artifact = str(row.get("artifact_path", ""))
        artifact_path = PROJECT_DIR / artifact if artifact and not Path(artifact).is_absolute() else Path(artifact) if artifact else Path()
        rows.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "method": row.get("method", ""),
                "target_aug_ratio": row.get("target_aug_ratio", ""),
                "actual_aug_ratio": row.get("actual_aug_ratio", ""),
                "n_train_original": row.get("n_train_original", ""),
                "n_aug_target": int(round(_float(row.get("target_aug_ratio"), 0.0) * _float(row.get("n_train_original"), 0.0))),
                "n_aug_generated": n_aug,
                "n_aug_accepted": n_aug if success else 0,
                "n_aug_rejected": 0 if success else n_aug,
                "generation_success_rate": 1.0 if success else 0.0,
                "class_balance_before": "",
                "class_balance_after": "",
                "artifact_hash": _hash_file(artifact_path) if artifact else "",
                "artifact_path": artifact,
                "failure_reason": row.get("failure_reason", ""),
            }
        )
    return rows


def _build_cost_audit(per_seed_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in per_seed_rows:
        rows.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "method": row.get("method", ""),
                "aug_preprocess_sec": "",
                "generator_fit_sec": row.get("generator_fit_sec", ""),
                "sample_gen_sec": row.get("sample_gen_sec", ""),
                "dtw_alignment_sec": row.get("dtw_alignment_sec", ""),
                "cov_state_compute_sec": row.get("cov_state_compute_sec", ""),
                "bridge_realization_sec": row.get("bridge_realization_sec", ""),
                "aug_postprocess_sec": "",
                "downstream_train_sec": row.get("downstream_train_sec", ""),
                "eval_sec": row.get("eval_sec", ""),
                "total_pipeline_sec": row.get("total_pipeline_sec", ""),
                "method_cost_sec": row.get("method_cost_sec", ""),
                "total_sec": row.get("total_sec", ""),
                "peak_cpu_mem_mb": row.get("peak_cpu_mem_mb", ""),
                "peak_gpu_mem_mb": row.get("peak_gpu_mem_mb", ""),
                "device": row.get("device", ""),
                "num_threads": row.get("num_threads", ""),
            }
        )
    return rows


def _dataset_method_means(per_seed_rows: Sequence[Mapping[str, object]]) -> Dict[Tuple[str, str], float]:
    values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in per_seed_rows:
        if str(row.get("generation_success", "")).lower() != "true":
            continue
        f1 = _float(row.get("macro_f1"))
        if math.isnan(f1):
            continue
        values[(str(row.get("method")), str(row.get("dataset")))].append(f1)
    return {k: mean(v) for k, v in values.items() if v}


def _mean_available(values: Iterable[object]) -> object:
    vals = [_float(v) for v in values]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return "N/A"
    return mean(vals)


def _build_main_table(per_seed_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    # 1. Dataset-balanced means
    ds_method_f1 = _dataset_method_means(per_seed_rows)
    no_aug_f1 = {d: v for (m, d), v in ds_method_f1.items() if m == "no_aug"}
    
    # 2. Collect times
    method_ds_aug_time: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    method_ds_total_time: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    
    for row in per_seed_rows:
        m, d = str(row.get("method")), str(row.get("dataset"))
        at = _float(row.get("method_cost_sec"), _float(row.get("aug_cost_sec")))
        tt = _float(row.get("total_sec"), _float(row.get("total_pipeline_sec"), _float(row.get("method_elapsed_sec"))))
        if not math.isnan(at):
            method_ds_aug_time[(m, d)].append(at)
        if not math.isnan(tt):
            method_ds_total_time[(m, d)].append(tt)
        
    # Seed-averaged times per dataset
    aug_time_map = {k: mean(v) for k, v in method_ds_aug_time.items()}
    total_time_map = {k: mean(v) for k, v in method_ds_total_time.items()}
    
    # 3. Method stats
    out: List[Dict[str, object]] = []
    costa_method = "csta_topk_uniform_top5"
    
    # Check coverage for all methods
    matrix = _coverage_matrix(per_seed_rows)
    eligible_methods = []
    for m in E1_METHODS:
        is_full = all(matrix[m][d] == len(SEEDS) for d in FINAL20_DATASETS)
        if is_full:
            eligible_methods.append(m)
            
    for method in E1_METHODS:
        if method not in eligible_methods:
            continue
            
        meta = METHOD_META[method]
        # F1 metrics
        covered = [d for d in FINAL20_DATASETS if (method, d) in ds_method_f1]
        paired = [d for d in covered if d in no_aug_f1]
        
        avg_f1 = mean([ds_method_f1[(method, d)] for d in covered]) if covered else math.nan
        delta = mean([ds_method_f1[(method, d)] - no_aug_f1[d] for d in paired]) if paired else math.nan
        
        wins = ties = losses = 0
        for d in paired:
            diff = ds_method_f1[(method, d)] - no_aug_f1[d]
            if diff > 1e-12: wins += 1
            elif diff < -1e-12: losses += 1
            else: ties += 1
            
        # Time metrics
        at_vals = [aug_time_map[(method, d)] for d in covered if (method, d) in aug_time_map]
        tt_vals = [total_time_map[(method, d)] for d in covered if (method, d) in total_time_map]
        
        avg_at = mean(at_vals) if at_vals else math.nan
        avg_tt = mean(tt_vals) if tt_vals else math.nan
        
        # Rel Cost vs CoSTA
        rel_costs = []
        for d in covered:
            m_at = aug_time_map.get((method, d))
            c_at = aug_time_map.get((costa_method, d))
            if m_at is not None and c_at is not None and c_at > 1e-6:
                rel_costs.append(m_at / c_at)
        rel_cost = mean(rel_costs) if rel_costs else math.nan
        
        out.append({
            "Method": meta["display"],
            "Family": meta["family"],
            "Avg_F1": avg_f1,
            "Delta_vs_NoAug": delta,
            "WTL_vs_NoAug": f"{wins}/{ties}/{losses}",
            "Aug_Time": avg_at,
            "Total_Time": avg_tt,
            "Rel_Cost": rel_cost,
            "_raw_method": method
        })
    return out


def _build_dataset_method_matrix(per_seed_rows: Sequence[Mapping[str, object]]) -> Tuple[List[Dict[str, object]], List[str]]:
    ds_method_f1 = _dataset_method_means(per_seed_rows)
    method_labels = {method: str(METHOD_META[method]["display"]) for method in E1_METHODS}
    fields = ["Dataset"] + [method_labels[method] for method in E1_METHODS]
    rows: List[Dict[str, object]] = []
    for dataset in FINAL20_DATASETS:
        row: Dict[str, object] = {"Dataset": dataset}
        for method in E1_METHODS:
            row[method_labels[method]] = ds_method_f1.get((method, dataset), "N/A")
        rows.append(row)
    return rows, fields


def _build_method_provenance_table(method_registry: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in method_registry:
        rows.append(
            {
                "Method": row.get("method_display_name", ""),
                "Engineering_Method": row.get("method", ""),
                "Source_Level": row.get("source_level", ""),
                "Protocol": row.get("protocol", ""),
                "Cost_Type": row.get("cost_type", ""),
                "Paper": row.get("paper_title", ""),
                "Venue": row.get("paper_venue", ""),
                "Year": row.get("paper_year", ""),
                "Implementation": row.get("implementation_status", ""),
                "Notes": row.get("notes", ""),
            }
        )
    return rows


def _write_markdown_table(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    lines = []
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        vals = []
        for f in fields:
            value = row.get(f, "")
            vals.append(_fmt(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n")


def _coverage_matrix(per_seed_rows: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = {method: {dataset: 0 for dataset in FINAL20_DATASETS} for method in E1_METHODS}
    seen: Dict[Tuple[str, str], set] = defaultdict(set)
    for row in per_seed_rows:
        method = str(row.get("method", ""))
        dataset = str(row.get("dataset", ""))
        seed = str(row.get("seed", ""))
        if method in E1_METHODS and dataset in FINAL20_DATASETS:
            seen[(method, dataset)].add(seed)
    for (method, dataset), seeds in seen.items():
        matrix[method][dataset] = len(seeds)
    return matrix


def _write_audit_report(
    per_seed_rows: Sequence[Mapping[str, object]],
    method_registry: Sequence[Mapping[str, object]],
    dataset_registry: Sequence[Mapping[str, object]],
    main_rows: Sequence[Mapping[str, object]],
) -> None:
    matrix = _coverage_matrix(per_seed_rows)
    lines: List[str] = []
    lines.append("# E1 Data Audit")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Experiment tag: `{EXPERIMENT_TAG}`")
    lines.append("")
    lines.append("## Method Coverage Matrix")
    lines.append("")
    lines.append("| Method | Final20 datasets with 3 seeds | Total rows | Missing runs |")
    lines.append("| --- | ---: | ---: | --- |")
    row_count = Counter(str(r.get("method")) for r in per_seed_rows)
    for method in E1_METHODS:
        complete = sum(1 for d in FINAL20_DATASETS if matrix[method][d] == len(SEEDS))
        missing = [
            f"{d}:s{','.join(s for s in SEEDS if s not in {str(r.get('seed')) for r in per_seed_rows if r.get('method') == method and r.get('dataset') == d})}"
            for d in FINAL20_DATASETS
            if matrix[method][d] != len(SEEDS)
        ]
        lines.append(f"| `{method}` | {complete}/20 | {row_count[method]} | {'; '.join(missing[:8])}{' ...' if len(missing) > 8 else ''} |")
    lines.append("")
    lines.append("## Dataset Coverage Matrix")
    lines.append("")
    lines.append("| Dataset | Methods complete at 3 seeds |")
    lines.append("| --- | ---: |")
    for dataset in FINAL20_DATASETS:
        complete_methods = sum(1 for method in E1_METHODS if matrix[method][dataset] == len(SEEDS))
        lines.append(f"| `{dataset}` | {complete_methods}/{len(E1_METHODS)} |")
    lines.append("")
    lines.append("## Missing Runs")
    lines.append("")
    missing_lines = []
    for method in E1_METHODS:
        for dataset in FINAL20_DATASETS:
            present = {str(r.get("seed")) for r in per_seed_rows if r.get("method") == method and r.get("dataset") == dataset}
            missing = [seed for seed in SEEDS if seed not in present]
            if missing:
                missing_lines.append(f"- `{method}` / `{dataset}` missing seeds: {', '.join(missing)}")
    lines.extend(missing_lines or ["- None"])
    lines.append("")
    lines.append("## Budget Consistency")
    lines.append("")
    inconsistent = []
    for row in per_seed_rows:
        method = str(row.get("method"))
        if method == "raw_mixup":
            continue
        target = _float(row.get("target_aug_ratio"))
        actual = _float(row.get("actual_aug_ratio"))
        if not math.isnan(target) and not math.isnan(actual) and abs(target - actual) > 1e-6:
            inconsistent.append(f"- `{method}` `{row.get('dataset')}` seed {row.get('seed')}: target={target}, actual={actual}")
    lines.extend(inconsistent[:50] or ["- No target/actual augmentation-ratio mismatch detected in available rows."])
    if len(inconsistent) > 50:
        lines.append(f"- ... {len(inconsistent) - 50} more")
    lines.append("")
    lines.append("## Cost Field Availability")
    lines.append("")
    lines.append("| Method | Rows | aug_cost | generator_fit | sample_gen | dtw_alignment | cov_state | bridge | total |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method in E1_METHODS:
        rows = [r for r in per_seed_rows if r.get("method") == method]
        def count(field: str) -> int:
            n = 0
            for r in rows:
                value = r.get(field, "")
                if value in {"", "N/A", None}:
                    continue
                if isinstance(value, float) and math.isnan(value):
                    continue
                if str(value).lower() == "nan":
                    continue
                n += 1
            return n
        lines.append(
            f"| `{method}` | {len(rows)} | {count('aug_cost_sec')} | {count('generator_fit_sec')} | {count('sample_gen_sec')} | {count('dtw_alignment_sec')} | {count('cov_state_compute_sec')} | {count('bridge_realization_sec')} | {count('total_pipeline_sec')} |"
        )
    lines.append("")
    lines.append("## W/T/L Consistency")
    lines.append("")
    for row in main_rows:
        method = row["Method"]
        wtl = str(row["WTL_vs_NoAug"])
        try:
            total = sum(int(x) for x in wtl.split("/"))
        except Exception:
            total = -1
        lines.append(f"- {method}: W/T/L={wtl}, paired dataset count={total}")
    lines.append("")
    lines.append("## Claim Support Checklist")
    lines.append("")
    lines.append("- RandomCov and PCACov are excluded from E1 and remain internal controls.")
    lines.append("- TimeVAE is excluded from E1.")
    lines.append("- Mixup is marked as training-time soft-label vicinal training; offline augmentation cost can be N/A.")
    lines.append("- TimeGAN currently has method metadata but no completed E1 rows in this workspace.")
    lines.append("- Diffusion-TS rows are subset coverage and must not be described as Final20 full unless completed.")
    lines.append("- RGW/DGW rows are subset coverage and clean-room adapters; do not call them official reproductions.")
    lines.append("- CoSTA-U5 uses canonical Final20 rows when available.")
    AUDIT_DOC.write_text("\n".join(lines) + "\n")


def main() -> None:
    source_rows = _dedupe_rows(_collect_source_rows())
    datasets = set(FINAL20_DATASETS)
    datasets.update(row.get("dataset", "") for _, _, row in source_rows)
    dataset_registry = _build_dataset_registry(datasets)
    dataset_registry_map = {str(row["dataset"]): row for row in dataset_registry}
    per_seed_rows = _build_per_seed_rows(source_rows, dataset_registry_map)
    _materialize_reconstructed_configs(per_seed_rows)
    method_registry = _build_method_registry()
    aug_artifacts = _build_aug_artifacts(per_seed_rows)
    cost_audit = _build_cost_audit(per_seed_rows)
    main_rows = _build_main_table(per_seed_rows)
    dataset_method_matrix, dataset_method_matrix_fields = _build_dataset_method_matrix(per_seed_rows)
    provenance_rows = _build_method_provenance_table(method_registry)

    _write_csv(OUT_DIR / "per_seed_e1_runs.csv", per_seed_rows, PER_SEED_FIELDS)
    _write_csv(
        OUT_DIR / "e1_method_registry.csv",
        method_registry,
        [
            "method",
            "method_display_name",
            "family",
            "source_level",
            "paper_title",
            "paper_year",
            "paper_venue",
            "paper_url",
            "code_url",
            "implementation_status",
            "protocol",
            "cost_type",
            "requires_generator",
            "requires_soft_label",
            "requires_hidden_state",
            "requires_alignment",
            "outputs_raw_samples",
            "is_proposed",
            "is_external_baseline",
            "notes",
        ],
    )
    _write_csv(
        OUT_DIR / "e1_dataset_registry.csv",
        dataset_registry,
        [
            "dataset",
            "source_archive",
            "n_classes",
            "n_channels",
            "seq_len",
            "train_size",
            "test_size",
            "is_variable_length",
            "has_missing_values",
            "normalization",
            "label_distribution",
            "final20_included",
            "notes",
        ],
    )
    _write_csv(OUT_DIR / "e1_aug_artifacts.csv", aug_artifacts, AUG_ARTIFACT_FIELDS)
    _write_csv(OUT_DIR / "e1_cost_audit.csv", cost_audit, COST_FIELDS)
    main_fields = ["Method", "Family", "Avg_F1", "Delta_vs_NoAug", "WTL_vs_NoAug", "Aug_Time", "Total_Time", "Rel_Cost"]
    _write_csv(OUT_DIR / "e1_main_table.csv", main_rows, main_fields)
    _write_markdown_table(OUT_DIR / "e1_main_table.md", main_rows, main_fields)
    _write_csv(OUT_DIR / "e1_dataset_method_matrix.csv", dataset_method_matrix, dataset_method_matrix_fields)
    _write_markdown_table(OUT_DIR / "e1_dataset_method_matrix.md", dataset_method_matrix, dataset_method_matrix_fields)
    provenance_fields = [
        "Method",
        "Engineering_Method",
        "Source_Level",
        "Protocol",
        "Cost_Type",
        "Paper",
        "Venue",
        "Year",
        "Implementation",
        "Notes",
    ]
    _write_csv(OUT_DIR / "e1_method_provenance_table.csv", provenance_rows, provenance_fields)
    _write_markdown_table(OUT_DIR / "e1_method_provenance_table.md", provenance_rows, provenance_fields)
    _write_audit_report(per_seed_rows, method_registry, dataset_registry, main_rows)

    print(f"Wrote {OUT_DIR / 'per_seed_e1_runs.csv'} ({len(per_seed_rows)} rows)")
    print(f"Wrote {OUT_DIR / 'e1_main_table.csv'}")
    print(f"Wrote {OUT_DIR / 'e1_dataset_method_matrix.csv'}")
    print(f"Wrote {AUDIT_DOC}")


if __name__ == "__main__":
    main()
