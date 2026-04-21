#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from datasets.trial_dataset_factory import (  # noqa: E402
    DEFAULT_ATRIALFIBRILLATION_ROOT,
    DEFAULT_ARTICULARYWORDRECOGNITION_ROOT,
    DEFAULT_BASICMOTIONS_ROOT,
    DEFAULT_CRICKET_ROOT,
    DEFAULT_EPILEPSY_ROOT,
    DEFAULT_ERING_ROOT,
    DEFAULT_ETHANOLCONCENTRATION_ROOT,
    DEFAULT_FINGERMOVEMENTS_ROOT,
    DEFAULT_HAR_ROOT,
    DEFAULT_HANDMOVEMENTDIRECTION_ROOT,
    DEFAULT_HANDWRITING_ROOT,
    DEFAULT_HEARTBEAT_ROOT,
    DEFAULT_JAPANESEVOWELS_ROOT,
    DEFAULT_LIBRAS_ROOT,
    DEFAULT_MITBIH_NPZ,
    DEFAULT_MOTORIMAGERY_ROOT,
    DEFAULT_NATOPS_ROOT,
    DEFAULT_PENDIGITS_ROOT,
    DEFAULT_RACKETSPORTS_ROOT,
    DEFAULT_SELFREGULATIONSCP1_ROOT,
    DEFAULT_SELFREGULATIONSCP2_ROOT,
    DEFAULT_UWAVEGESTURELIBRARY_ROOT,
    load_trials_for_dataset,
    normalize_dataset_name,
)
from models.resnet1d_adapter import ResNet1DAdapter  # noqa: E402
from models.resnet1d_local_closed_form import ResNet1DLocalClosedFormResidual  # noqa: E402
from models.resnet1d_residual_linear import ResNet1DResidualLinear  # noqa: E402


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj: Dict[str, object]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    _ensure_dir(os.path.dirname(path))
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_fixedsplit_arrays(args: argparse.Namespace):
    trials = load_trials_for_dataset(
        dataset=args.dataset,
        har_root=args.har_root,
        mitbih_npz=args.mitbih_npz,
        natops_root=args.natops_root,
        fingermovements_root=args.fingermovements_root,
        selfregulationscp1_root=args.selfregulationscp1_root,
        basicmotions_root=args.basicmotions_root,
        handmovementdirection_root=args.handmovementdirection_root,
        uwavegesturelibrary_root=args.uwavegesturelibrary_root,
        epilepsy_root=args.epilepsy_root,
        atrialfibrillation_root=args.atrialfibrillation_root,
        pendigits_root=args.pendigits_root,
        racketsports_root=args.racketsports_root,
        articularywordrecognition_root=args.articularywordrecognition_root,
        heartbeat_root=args.heartbeat_root,
        selfregulationscp2_root=args.selfregulationscp2_root,
        libras_root=args.libras_root,
        japanesevowels_root=args.japanesevowels_root,
        cricket_root=args.cricket_root,
        handwriting_root=args.handwriting_root,
        ering_root=args.ering_root,
        motorimagery_root=args.motorimagery_root,
        ethanolconcentration_root=args.ethanolconcentration_root,
    )
    train_trials = [t for t in trials if str(t.get("split", "")).lower() == "train"]
    test_trials = [t for t in trials if str(t.get("split", "")).lower() == "test"]
    if not train_trials or not test_trials:
        raise ValueError(
            f"dataset={args.dataset} does not expose fixed train/test splits in trial dicts"
        )
    train_x = np.stack([np.asarray(t["x_trial"], dtype=np.float32) for t in train_trials], axis=0)
    test_x = np.stack([np.asarray(t["x_trial"], dtype=np.float32) for t in test_trials], axis=0)
    train_y = np.asarray([int(t["label"]) for t in train_trials], dtype=np.int64)
    test_y = np.asarray([int(t["label"]) for t in test_trials], dtype=np.int64)
    num_classes = int(max(train_y.max(initial=0), test_y.max(initial=0)) + 1)
    return train_x, test_x, train_y, test_y, num_classes


def _build_model(args: argparse.Namespace, *, in_channels: int, num_classes: int) -> torch.nn.Module:
    if args.arm == "e0":
        return ResNet1DAdapter(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
        )
    if args.arm == "e1":
        return ResNet1DResidualLinear(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            init_beta=float(args.init_beta),
        )
    if args.arm == "e2":
        return ResNet1DLocalClosedFormResidual(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            prototypes_per_class=int(args.prototypes_per_class),
            routing_temperature=float(args.routing_temperature),
            class_prior_temperature=args.class_prior_temperature,
            subproto_temperature=args.subproto_temperature,
            ridge=float(args.closed_form_ridge),
            ridge_mode=str(args.closed_form_ridge_mode),
            ridge_trace_eps=float(args.closed_form_ridge_trace_eps),
            solve_mode=str(args.closed_form_solve_mode),
            pinv_rcond=float(args.closed_form_pinv_rcond),
            input_norm_mode=str(args.closed_form_input_norm_mode),
            input_norm_eps=float(args.closed_form_input_norm_eps),
            enable_probe=bool(args.closed_form_probe),
            init_beta=float(args.init_beta),
            detach_local_input=bool(args.detach_local_latent),
            support_mode=str(args.local_support_mode),
            prototype_aggregation=str(args.prototype_aggregation),
            prototype_geometry_mode=str(args.prototype_geometry_mode),
            tangent_rank=int(args.tangent_rank),
            tangent_source=str(args.tangent_source),
            prob_tangent_version=str(args.prob_tangent_version),
            rank_selection_mode=str(args.rank_selection_mode),
            posterior_mode=str(args.posterior_mode),
            posterior_student_dof=float(args.posterior_student_dof),
            mdl_penalty_beta=float(args.mdl_penalty_beta),
            gaussian_refine_variant=str(args.gaussian_refine_variant),
            mdl_zero_rank_rescue_margin=float(args.mdl_zero_rank_rescue_margin),
            local_solver_competition_mode=str(args.local_solver_competition_mode),
            relative_solver_temperature=float(args.relative_solver_temperature),
            abs_gate_activity_floor=float(args.abs_gate_activity_floor),
            readout_gate_mode=str(args.local_readout_gate),
        )
    raise ValueError(f"unknown arm: {args.arm}")


def _make_loaders(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
    *,
    train_batch_size: int,
    test_batch_size: int,
    num_workers: int,
    use_cuda: bool,
):
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_x).float(),
        torch.from_numpy(train_y).long(),
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(test_x).float(),
        torch.from_numpy(test_y).long(),
    )
    train_kwargs = {"batch_size": int(train_batch_size), "shuffle": True}
    test_kwargs = {"batch_size": int(test_batch_size), "shuffle": False}
    if use_cuda:
        cuda_kwargs = {"num_workers": int(num_workers), "pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    return (
        torch.utils.data.DataLoader(train_dataset, **train_kwargs),
        torch.utils.data.DataLoader(test_dataset, **test_kwargs),
    )


def _compute_fusion_alpha(args: argparse.Namespace, *, epoch: int) -> float:
    hold = max(0, int(args.fusion_warmup_hold_epochs))
    ramp = max(0, int(args.fusion_warmup_ramp_epochs))
    start = float(args.fusion_warmup_start_scale)
    if hold == 0 and ramp == 0:
        return 1.0
    if int(epoch) <= hold:
        return start
    if ramp == 0:
        return 1.0
    progress = min(1.0, max(0.0, (float(epoch) - float(hold)) / float(ramp)))
    return start + (1.0 - start) * progress


def _maybe_set_probe_context(model: torch.nn.Module, *, split: str, epoch: int) -> None:
    local_head = getattr(model, "local_head", None)
    if local_head is not None and hasattr(local_head, "set_probe_context"):
        local_head.set_probe_context(split=str(split), epoch=int(epoch))


def _export_probe_artifacts(model: torch.nn.Module, *, run_dir: str) -> None:
    local_head = getattr(model, "local_head", None)
    if local_head is None or not hasattr(local_head, "export_probe_rows"):
        return
    rows = local_head.export_probe_rows()
    if not rows:
        return
    probe_csv = os.path.join(run_dir, "closed_form_probe_rows.csv")
    _write_csv(probe_csv, rows)

    metrics = ["mean_trace", "condition_number", "effective_ridge", "weight_norm"]
    summary: Dict[str, object] = {
        "n_rows": int(len(rows)),
        "splits": sorted({str(r["split"]) for r in rows}),
        "metrics": {},
    }
    for metric in metrics:
        values = np.asarray([float(r[metric]) for r in rows], dtype=np.float64)
        summary["metrics"][metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(np.median(values)),
        }
        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=40)
        plt.title(metric)
        plt.xlabel(metric)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"closed_form_probe_{metric}.png"), dpi=160)
        plt.close()
    _write_json(os.path.join(run_dir, "closed_form_probe_summary.json"), summary)


def _export_dataflow_probe_artifacts(
    *,
    run_dir: str,
    first_batch_summary: Dict[str, object],
    stage_rows: List[Dict[str, object]],
    agreement_summary: Dict[str, object],
) -> None:
    _write_json(os.path.join(run_dir, "dataflow_summary.json"), first_batch_summary)
    _write_json(os.path.join(run_dir, "dataflow_agreement_summary.json"), agreement_summary)
    if stage_rows:
        _write_csv(os.path.join(run_dir, "dataflow_test_stage_rows.csv"), stage_rows)


def _export_mdl_rank_trace_artifact(
    *,
    run_dir: str,
    first_batch_summary: Dict[str, object] | None,
) -> None:
    if not first_batch_summary:
        return
    local_head_dataflow = first_batch_summary.get("local_head_dataflow")
    if not isinstance(local_head_dataflow, dict):
        return
    class_summaries = local_head_dataflow.get("class_summaries")
    if not isinstance(class_summaries, list):
        return
    trace_rows: List[Dict[str, object]] = []
    for class_summary in class_summaries:
        if not isinstance(class_summary, dict):
            continue
        rank_rows = class_summary.get("rank_rows")
        if not isinstance(rank_rows, list) or not rank_rows:
            continue
        trace_rows.append(
            {
                "class_idx": class_summary.get("class_idx"),
                "selected_rank": class_summary.get("selected_rank"),
                "ppca_sigma2": class_summary.get("ppca_sigma2"),
                "lw_shrinkage_alpha": class_summary.get("lw_shrinkage_alpha"),
                "posterior_mode": class_summary.get("posterior_mode"),
                "mdl_penalty_beta": class_summary.get("mdl_penalty_beta"),
                "gaussian_refine_variant": class_summary.get("gaussian_refine_variant"),
                "mdl_zero_rank_rescue_margin": class_summary.get("mdl_zero_rank_rescue_margin"),
                "trace_per_dim": class_summary.get("trace_per_dim"),
                "rank0_score": class_summary.get("rank0_score"),
                "rank1_score": class_summary.get("rank1_score"),
                "rank01_relative_gap": class_summary.get("rank01_relative_gap"),
                "zero_rank_rescued": class_summary.get("zero_rank_rescued"),
                "rank_rows": rank_rows,
            }
        )
    if not trace_rows:
        return
    payload = {
        "dataset": first_batch_summary.get("dataset"),
        "arm": first_batch_summary.get("arm"),
        "split": first_batch_summary.get("split"),
        "trace_rows": trace_rows,
    }
    _write_json(os.path.join(run_dir, "mdl_rank_trace.json"), payload)


def _export_tangent_probe_artifacts(model: torch.nn.Module, *, run_dir: str) -> None:
    local_head = getattr(model, "local_head", None)
    if local_head is None or not hasattr(local_head, "export_tangent_probe_payload"):
        return
    payload = local_head.export_tangent_probe_payload()
    if not payload:
        return
    summary = dict(payload.get("summary", {}))
    class_rows = list(payload.get("class_rows", []))
    _write_json(os.path.join(run_dir, "tangent_probe_summary.json"), summary)
    if class_rows:
        csv_rows: List[Dict[str, object]] = []
        for row in class_rows:
            csv_rows.append(
                {
                    "class_idx": row.get("class_idx"),
                    "prototype_geometry_mode": row.get("prototype_geometry_mode"),
                    "tangent_source": row.get("tangent_source"),
                    "requested_tangent_rank": row.get("requested_tangent_rank"),
                    "actual_tangent_rank": row.get("actual_tangent_rank"),
                    "rank95": row.get("rank95"),
                    "effective_rank": row.get("effective_rank"),
                    "top1_energy_ratio": row.get("top1_energy_ratio"),
                    "top1_top2_spectral_gap": row.get("top1_top2_spectral_gap"),
                }
            )
        _write_csv(os.path.join(run_dir, "tangent_probe_class_rows.csv"), csv_rows)
        _write_json(os.path.join(run_dir, "tangent_probe_full.json"), {"class_rows": class_rows, "summary": summary})


def _forward_outputs(model: torch.nn.Module, batch_x: torch.Tensor, *, args: argparse.Namespace):
    if args.arm == "e2":
        return model(batch_x, fusion_alpha=1.0, return_features=True)
    return model(batch_x, return_features=True)


def _make_run_dir(args: argparse.Namespace) -> str:
    tag = str(args.run_tag).strip() if str(args.run_tag).strip() else f"{args.dataset}_{args.arm}_seed{int(args.seed)}"
    out_root = args.out_root or os.path.join(
        "out",
        "_active",
        f"verify_resnet1d_local_closed_form_fixedsplit_{time.strftime('%Y%m%d')}",
    )
    return os.path.join(out_root, str(args.arm), tag)


def _normalized_entropy_from_probs(probs: np.ndarray) -> float:
    if probs.size <= 1:
        return 0.0
    probs = probs.astype(np.float64, copy=False)
    probs = probs / np.clip(probs.sum(), a_min=1e-12, a_max=None)
    valid = probs > 0
    entropy = float(-(probs[valid] * np.log(probs[valid])).sum())
    denom = float(np.log(float(probs.size)))
    if denom <= 0.0:
        return 0.0
    return entropy / denom


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ResNet-1D + DLCR fixed-split TSC runner.")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--arm", type=str, default="e0", choices=["e0", "e1", "e2"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--train-batch-size", type=int, default=64)
    p.add_argument("--test-batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out-root", type=str, default="")
    p.add_argument("--run-tag", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--init-beta", type=float, default=0.1)
    p.add_argument("--prototypes-per-class", type=int, default=4)
    p.add_argument("--routing-temperature", type=float, default=1.0)
    p.add_argument("--class-prior-temperature", type=float, default=None,
                   help="Temperature for class-center softmax routing. None = use routing-temperature.")
    p.add_argument("--subproto-temperature", type=float, default=None,
                   help="Temperature for sub-prototype softmax routing. None = use routing-temperature.")
    p.add_argument("--closed-form-ridge", type=float, default=1e-2)
    p.add_argument("--closed-form-ridge-mode", type=str, default="fixed", choices=["fixed", "trace_adaptive"])
    p.add_argument("--closed-form-ridge-trace-eps", type=float, default=1e-8)
    p.add_argument(
        "--closed-form-solve-mode",
        type=str,
        default="ridge_solve",
        choices=["ridge_solve", "pinv", "dual_ridge", "dual_pinv"],
    )
    p.add_argument("--closed-form-pinv-rcond", type=float, default=1e-4)
    p.add_argument("--closed-form-input-norm-mode", type=str, default="none", choices=["none", "l2_hypersphere"])
    p.add_argument("--closed-form-input-norm-eps", type=float, default=1e-8)
    p.add_argument("--closed-form-probe", action="store_true", default=False)
    p.add_argument("--dataflow-probe", action="store_true", default=False)
    p.add_argument("--local-support-mode", type=str, default="same_only", choices=["same_opp_balanced", "same_only", "same_opp_asym"])
    p.add_argument("--prototype-aggregation", type=str, default="pooled", choices=["pooled", "committee_mean"])
    p.add_argument("--prototype-geometry-mode", type=str, default="flat", choices=["flat", "center_subproto", "center_only", "center_tangent", "center_prob_tangent"])
    p.add_argument("--tangent-rank", type=int, default=2)
    p.add_argument("--tangent-source", type=str, default="subproto_offsets", choices=["subproto_offsets"])
    p.add_argument("--tangent-probe", action="store_true", default=False)
    p.add_argument("--prob-tangent-version", type=str, default="v1", choices=["v1", "v2", "v3"])
    p.add_argument("--rank-selection-mode", type=str, default="mdl", choices=["mdl", "bic"])
    p.add_argument("--posterior-mode", type=str, default="gaussian_dimnorm", choices=["gaussian_dimnorm", "student_t"])
    p.add_argument("--posterior-student-dof", type=float, default=3.0)
    p.add_argument("--mdl-penalty-beta", type=float, default=1.0)
    p.add_argument("--gaussian-refine-variant", type=str, default="base", choices=["base", "trace_floor", "trace_floor_mdl_margin"])
    p.add_argument("--mdl-zero-rank-rescue-margin", type=float, default=0.03)
    p.add_argument("--local-solver-competition-mode", type=str, default="none", choices=["none", "relcomp"])
    p.add_argument("--relative-solver-temperature", type=float, default=1.0)
    p.add_argument("--abs-gate-activity-floor", type=float, default=1e-6)
    p.add_argument("--emit-mdl-rank-trace", action="store_true", default=False)
    p.add_argument("--local-readout-gate", type=str, default="none", choices=["none", "consistency"])
    p.add_argument("--detach-local-latent", action="store_true", default=False)
    p.add_argument("--fusion-warmup-hold-epochs", type=int, default=0)
    p.add_argument("--fusion-warmup-ramp-epochs", type=int, default=0)
    p.add_argument("--fusion-warmup-start-scale", type=float, default=0.0)
    p.add_argument("--har-root", type=str, default=DEFAULT_HAR_ROOT)
    p.add_argument("--mitbih-npz", type=str, default=DEFAULT_MITBIH_NPZ)
    p.add_argument("--natops-root", type=str, default=DEFAULT_NATOPS_ROOT)
    p.add_argument("--fingermovements-root", type=str, default=DEFAULT_FINGERMOVEMENTS_ROOT)
    p.add_argument("--selfregulationscp1-root", type=str, default=DEFAULT_SELFREGULATIONSCP1_ROOT)
    p.add_argument("--basicmotions-root", type=str, default=DEFAULT_BASICMOTIONS_ROOT)
    p.add_argument("--handmovementdirection-root", type=str, default=DEFAULT_HANDMOVEMENTDIRECTION_ROOT)
    p.add_argument("--uwavegesturelibrary-root", type=str, default=DEFAULT_UWAVEGESTURELIBRARY_ROOT)
    p.add_argument("--epilepsy-root", type=str, default=DEFAULT_EPILEPSY_ROOT)
    p.add_argument("--atrialfibrillation-root", type=str, default=DEFAULT_ATRIALFIBRILLATION_ROOT)
    p.add_argument("--pendigits-root", type=str, default=DEFAULT_PENDIGITS_ROOT)
    p.add_argument("--racketsports-root", type=str, default=DEFAULT_RACKETSPORTS_ROOT)
    p.add_argument("--articularywordrecognition-root", type=str, default=DEFAULT_ARTICULARYWORDRECOGNITION_ROOT)
    p.add_argument("--heartbeat-root", type=str, default=DEFAULT_HEARTBEAT_ROOT)
    p.add_argument("--selfregulationscp2-root", type=str, default=DEFAULT_SELFREGULATIONSCP2_ROOT)
    p.add_argument("--libras-root", type=str, default=DEFAULT_LIBRAS_ROOT)
    p.add_argument("--japanesevowels-root", type=str, default=DEFAULT_JAPANESEVOWELS_ROOT)
    p.add_argument("--cricket-root", type=str, default=DEFAULT_CRICKET_ROOT)
    p.add_argument("--handwriting-root", type=str, default=DEFAULT_HANDWRITING_ROOT)
    p.add_argument("--ering-root", type=str, default=DEFAULT_ERING_ROOT)
    p.add_argument("--motorimagery-root", type=str, default=DEFAULT_MOTORIMAGERY_ROOT)
    p.add_argument("--ethanolconcentration-root", type=str, default=DEFAULT_ETHANOLCONCENTRATION_ROOT)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args.dataset = normalize_dataset_name(args.dataset)
    _set_seed(int(args.seed))

    run_dir = _make_run_dir(args)
    _ensure_dir(run_dir)

    train_x, test_x, train_y, test_y, num_classes = _load_fixedsplit_arrays(args)
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    model = _build_model(args, in_channels=int(train_x.shape[1]), num_classes=int(num_classes)).to(device)
    local_head = getattr(model, "local_head", None)
    if local_head is not None and hasattr(local_head, "reset_probe"):
        local_head.reset_probe()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_loader, test_loader = _make_loaders(
        train_x,
        test_x,
        train_y,
        test_y,
        train_batch_size=int(args.train_batch_size),
        test_batch_size=int(args.test_batch_size),
        num_workers=int(args.num_workers),
        use_cuda=bool(device.type == "cuda"),
    )

    run_meta = {
        "dataset": str(args.dataset),
        "arm": str(args.arm),
        "split_protocol": "fixedsplit",
        "runner_protocol": "resnet1d_local_closed_form_fixedsplit",
        "host_backbone": "ResNet1D",
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "train_batch_size": int(args.train_batch_size),
        "test_batch_size": int(args.test_batch_size),
        "num_classes": int(num_classes),
        "n_train": int(train_x.shape[0]),
        "n_test": int(test_x.shape[0]),
        "in_channels": int(train_x.shape[1]),
        "seq_len": int(train_x.shape[2]),
        "prototypes_per_class": int(args.prototypes_per_class),
        "routing_temperature": float(
            getattr(local_head, "routing_temperature", float(args.routing_temperature))
            if local_head is not None
            else float(args.routing_temperature)
        ),
        "class_prior_temperature": float(
            getattr(
                local_head,
                "class_prior_temperature",
                float(args.routing_temperature if args.class_prior_temperature is None else args.class_prior_temperature),
            )
            if local_head is not None
            else float(args.routing_temperature if args.class_prior_temperature is None else args.class_prior_temperature)
        ),
        "subproto_temperature": float(
            getattr(
                local_head,
                "subproto_temperature",
                float(args.routing_temperature if args.subproto_temperature is None else args.subproto_temperature),
            )
            if local_head is not None
            else float(args.routing_temperature if args.subproto_temperature is None else args.subproto_temperature)
        ),
        "closed_form_ridge": float(args.closed_form_ridge),
        "closed_form_ridge_mode": str(args.closed_form_ridge_mode),
        "closed_form_ridge_trace_eps": float(args.closed_form_ridge_trace_eps),
        "closed_form_solve_mode": str(args.closed_form_solve_mode),
        "closed_form_pinv_rcond": float(args.closed_form_pinv_rcond),
        "closed_form_input_norm_mode": str(args.closed_form_input_norm_mode),
        "closed_form_input_norm_eps": float(args.closed_form_input_norm_eps),
        "dataflow_probe": bool(args.dataflow_probe),
        "local_support_mode": str(args.local_support_mode),
        "prototype_aggregation": str(args.prototype_aggregation),
        "prototype_geometry_mode": str(args.prototype_geometry_mode),
        "tangent_rank": int(args.tangent_rank),
        "tangent_source": str(args.tangent_source),
        "tangent_probe": bool(args.tangent_probe),
        "prob_tangent_version": str(args.prob_tangent_version),
        "rank_selection_mode": str(args.rank_selection_mode),
        "posterior_mode": str(args.posterior_mode),
        "posterior_student_dof": float(args.posterior_student_dof),
        "mdl_penalty_beta": float(args.mdl_penalty_beta),
        "gaussian_refine_variant": str(args.gaussian_refine_variant),
        "mdl_zero_rank_rescue_margin": float(args.mdl_zero_rank_rescue_margin),
        "local_solver_competition_mode": str(args.local_solver_competition_mode),
        "relative_solver_temperature": float(args.relative_solver_temperature),
        "abs_gate_activity_floor": float(args.abs_gate_activity_floor),
        "local_readout_gate": str(args.local_readout_gate),
        "run_dir": run_dir,
    }
    _write_json(os.path.join(run_dir, "run_meta.json"), run_meta)

    start_time = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        n_seen = 0
        train_loss = 0.0
        train_correct = 0
        fusion_alpha = _compute_fusion_alpha(args, epoch=epoch)
        _maybe_set_probe_context(model, split="train", epoch=epoch)
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if args.arm == "e2":
                logits = model(batch_x, fusion_alpha=float(fusion_alpha))
            else:
                logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()

            batch_n = int(batch_y.shape[0])
            n_seen += batch_n
            train_loss += float(loss.item()) * batch_n
            train_correct += int((logits.argmax(dim=1) == batch_y).sum().item())

        if epoch == 1 or epoch % int(args.log_every) == 0 or epoch == int(args.epochs):
            avg_loss = train_loss / max(1, n_seen)
            avg_acc = float(train_correct) / max(1, n_seen)
            print(
                f"[{args.dataset}][{args.arm}] epoch {epoch}/{args.epochs} "
                f"train_loss={avg_loss:.6f} train_acc={avg_acc:.4f} fusion_alpha={fusion_alpha:.4f}",
                flush=True,
            )

    model.eval()
    all_pred: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    dataflow_first_batch_summary: Dict[str, object] | None = None
    stage_rows: List[Dict[str, object]] = []
    base_pred_all: List[np.ndarray] = []
    local_pred_all: List[np.ndarray] = []
    final_pred_all: List[np.ndarray] = []
    gate_values: List[np.ndarray] = []
    selected_top1_index_all: List[np.ndarray] = []
    selected_top1_weight_all: List[np.ndarray] = []
    selected_routing_entropy_all: List[np.ndarray] = []
    selected_cos_gap_all: List[np.ndarray] = []
    selected_routing_widths: List[int] = []
    selected_rank_all: List[np.ndarray] = []
    selected_lw_alpha_all: List[np.ndarray] = []
    selected_ppca_sigma2_all: List[np.ndarray] = []
    selected_posterior_confidence_all: List[np.ndarray] = []
    selected_posterior_log_confidence_all: List[np.ndarray] = []
    selected_posterior_residual_energy_all: List[np.ndarray] = []
    selected_posterior_residual_energy_per_dim_all: List[np.ndarray] = []
    selected_posterior_sigma2_eff_all: List[np.ndarray] = []
    selected_posterior_sigma2_used_all: List[np.ndarray] = []
    selected_trace_per_dim_all: List[np.ndarray] = []
    selected_rank0_score_all: List[np.ndarray] = []
    selected_rank1_score_all: List[np.ndarray] = []
    selected_rank01_relative_gap_all: List[np.ndarray] = []
    selected_zero_rank_rescued_all: List[np.ndarray] = []
    selected_abs_gate_all: List[np.ndarray] = []
    selected_relative_top1_weight_all: List[np.ndarray] = []
    selected_relative_top1_weight_entropy_all: List[np.ndarray] = []
    selected_relative_solver_margin_all: List[np.ndarray] = []
    selected_relative_competition_active_all: List[np.ndarray] = []
    selected_local_solver_weighted_delta_norm_all: List[np.ndarray] = []
    selected_relative_solver_top1_index_all: List[np.ndarray] = []
    with torch.no_grad():
        _maybe_set_probe_context(model, split="test", epoch=int(args.epochs))
        sample_offset = 0
        for batch_idx, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y_np = batch_y.numpy()
            if args.arm == "e2" and bool(args.dataflow_probe):
                outputs = model(batch_x, fusion_alpha=1.0, return_features=True)
                logits = outputs.final_logit
                base_pred = outputs.base_logit.argmax(dim=1).cpu().numpy()
                local_pred = outputs.local_closed_form_logit.argmax(dim=1).cpu().numpy()
                final_pred = outputs.final_logit.argmax(dim=1).cpu().numpy()
                base_pred_all.append(base_pred)
                local_pred_all.append(local_pred)
                final_pred_all.append(final_pred)
                if outputs.readout_gate is not None:
                    gate_values.append(outputs.readout_gate.detach().cpu().numpy())
                routing_payload = None
                local_head = getattr(model, "local_head", None)
                if local_head is not None and hasattr(local_head, "export_last_batch_routing_payload"):
                    routing_payload = local_head.export_last_batch_routing_payload()
                if routing_payload is not None:
                    selected_top1_index_all.append(np.asarray(routing_payload["top1_index"], dtype=np.int64))
                    selected_top1_weight_all.append(np.asarray(routing_payload["top1_weight"], dtype=np.float64))
                    selected_routing_entropy_all.append(np.asarray(routing_payload["routing_entropy"], dtype=np.float64))
                    selected_cos_gap_all.append(np.asarray(routing_payload["cos_top1_top2_gap"], dtype=np.float64))
                    selected_routing_widths.append(int(routing_payload["routing_width"]))
                    if "selected_rank" in routing_payload:
                        selected_rank_all.append(np.asarray(routing_payload["selected_rank"], dtype=np.int64))
                    if "lw_shrinkage_alpha" in routing_payload:
                        selected_lw_alpha_all.append(np.asarray(routing_payload["lw_shrinkage_alpha"], dtype=np.float64))
                    if "ppca_sigma2" in routing_payload:
                        selected_ppca_sigma2_all.append(np.asarray(routing_payload["ppca_sigma2"], dtype=np.float64))
                    if "posterior_confidence" in routing_payload:
                        selected_posterior_confidence_all.append(np.asarray(routing_payload["posterior_confidence"], dtype=np.float64))
                    if "posterior_log_confidence" in routing_payload:
                        selected_posterior_log_confidence_all.append(np.asarray(routing_payload["posterior_log_confidence"], dtype=np.float64))
                    if "posterior_residual_energy" in routing_payload:
                        selected_posterior_residual_energy_all.append(np.asarray(routing_payload["posterior_residual_energy"], dtype=np.float64))
                    if "posterior_residual_energy_per_dim" in routing_payload:
                        selected_posterior_residual_energy_per_dim_all.append(np.asarray(routing_payload["posterior_residual_energy_per_dim"], dtype=np.float64))
                    if "posterior_sigma2_eff" in routing_payload:
                        selected_posterior_sigma2_eff_all.append(np.asarray(routing_payload["posterior_sigma2_eff"], dtype=np.float64))
                    if "posterior_sigma2_used" in routing_payload:
                        selected_posterior_sigma2_used_all.append(np.asarray(routing_payload["posterior_sigma2_used"], dtype=np.float64))
                    if "trace_per_dim" in routing_payload:
                        selected_trace_per_dim_all.append(np.asarray(routing_payload["trace_per_dim"], dtype=np.float64))
                    if "rank0_score" in routing_payload:
                        selected_rank0_score_all.append(np.asarray(routing_payload["rank0_score"], dtype=np.float64))
                    if "rank1_score" in routing_payload:
                        selected_rank1_score_all.append(np.asarray(routing_payload["rank1_score"], dtype=np.float64))
                    if "rank01_relative_gap" in routing_payload:
                        selected_rank01_relative_gap_all.append(np.asarray(routing_payload["rank01_relative_gap"], dtype=np.float64))
                    if "zero_rank_rescued" in routing_payload:
                        selected_zero_rank_rescued_all.append(np.asarray(routing_payload["zero_rank_rescued"], dtype=np.float64))
                    if "abs_gate" in routing_payload:
                        selected_abs_gate_all.append(np.asarray(routing_payload["abs_gate"], dtype=np.float64))
                    if "relative_top1_weight" in routing_payload:
                        selected_relative_top1_weight_all.append(np.asarray(routing_payload["relative_top1_weight"], dtype=np.float64))
                    if "relative_top1_weight_entropy" in routing_payload:
                        selected_relative_top1_weight_entropy_all.append(np.asarray(routing_payload["relative_top1_weight_entropy"], dtype=np.float64))
                    if "relative_solver_margin" in routing_payload:
                        selected_relative_solver_margin_all.append(np.asarray(routing_payload["relative_solver_margin"], dtype=np.float64))
                    if "relative_competition_active" in routing_payload:
                        selected_relative_competition_active_all.append(np.asarray(routing_payload["relative_competition_active"], dtype=np.float64))
                    if "local_solver_weighted_delta_norm" in routing_payload:
                        selected_local_solver_weighted_delta_norm_all.append(np.asarray(routing_payload["local_solver_weighted_delta_norm"], dtype=np.float64))
                    if "relative_solver_top1_index" in routing_payload:
                        selected_relative_solver_top1_index_all.append(np.asarray(routing_payload["relative_solver_top1_index"], dtype=np.int64))
                if batch_idx == 0:
                    local_head = getattr(model, "local_head", None)
                    local_dataflow = None
                    if local_head is not None and hasattr(local_head, "export_last_dataflow_summary"):
                        local_dataflow = local_head.export_last_dataflow_summary()
                    dataflow_first_batch_summary = {
                        "dataset": str(args.dataset),
                        "arm": str(args.arm),
                        "split": "test",
                        "raw_input_shape": [int(v) for v in batch_x.shape],
                        "sequence_features_shape": [int(v) for v in outputs.sequence_features.shape],
                        "latent_shape": [int(v) for v in outputs.latent.shape],
                        "base_logit_shape": [int(v) for v in outputs.base_logit.shape],
                        "local_logit_shape": [int(v) for v in outputs.local_closed_form_logit.shape],
                        "final_logit_shape": [int(v) for v in outputs.final_logit.shape],
                        "readout_gate_shape": None if outputs.readout_gate is None else [int(v) for v in outputs.readout_gate.shape],
                        "beta": float(outputs.beta.detach().cpu().item()),
                        "local_head_dataflow": local_dataflow,
                    }
                for i in range(len(batch_y_np)):
                    local_top1_index = -1
                    local_top1_weight = 0.0
                    local_routing_entropy = 0.0
                    local_subproto_cos_gap = 0.0
                    local_routing_width = 0
                    if routing_payload is not None:
                        local_top1_index = int(routing_payload["top1_index"][i])
                        local_top1_weight = float(routing_payload["top1_weight"][i])
                        local_routing_entropy = float(routing_payload["routing_entropy"][i])
                        local_subproto_cos_gap = float(routing_payload["cos_top1_top2_gap"][i])
                        local_routing_width = int(routing_payload["routing_width"])
                    stage_rows.append(
                        {
                            "sample_index": int(sample_offset + i),
                            "label": int(batch_y_np[i]),
                            "base_pred": int(base_pred[i]),
                            "local_pred": int(local_pred[i]),
                            "final_pred": int(final_pred[i]),
                            "base_correct": int(base_pred[i] == batch_y_np[i]),
                            "local_correct": int(local_pred[i] == batch_y_np[i]),
                            "final_correct": int(final_pred[i] == batch_y_np[i]),
                            "base_local_agree": int(base_pred[i] == local_pred[i]),
                            "base_final_agree": int(base_pred[i] == final_pred[i]),
                            "local_final_agree": int(local_pred[i] == final_pred[i]),
                            "local_top1_subproto": int(local_top1_index),
                            "local_top1_weight": float(local_top1_weight),
                            "local_routing_entropy": float(local_routing_entropy),
                            "local_subproto_cos_gap": float(local_subproto_cos_gap),
                            "local_routing_width": int(local_routing_width),
                            "local_selected_rank": None if routing_payload is None or "selected_rank" not in routing_payload else int(routing_payload["selected_rank"][i]),
                            "local_lw_shrinkage_alpha": None if routing_payload is None or "lw_shrinkage_alpha" not in routing_payload else float(routing_payload["lw_shrinkage_alpha"][i]),
                            "local_ppca_sigma2": None if routing_payload is None or "ppca_sigma2" not in routing_payload else float(routing_payload["ppca_sigma2"][i]),
                            "local_posterior_confidence": None if routing_payload is None or "posterior_confidence" not in routing_payload else float(routing_payload["posterior_confidence"][i]),
                            "local_posterior_log_confidence": None if routing_payload is None or "posterior_log_confidence" not in routing_payload else float(routing_payload["posterior_log_confidence"][i]),
                            "local_posterior_residual_energy": None if routing_payload is None or "posterior_residual_energy" not in routing_payload else float(routing_payload["posterior_residual_energy"][i]),
                            "local_posterior_residual_energy_per_dim": None if routing_payload is None or "posterior_residual_energy_per_dim" not in routing_payload else float(routing_payload["posterior_residual_energy_per_dim"][i]),
                            "local_posterior_sigma2_eff": None if routing_payload is None or "posterior_sigma2_eff" not in routing_payload else float(routing_payload["posterior_sigma2_eff"][i]),
                            "local_posterior_sigma2_used": None if routing_payload is None or "posterior_sigma2_used" not in routing_payload else float(routing_payload["posterior_sigma2_used"][i]),
                            "local_trace_per_dim": None if routing_payload is None or "trace_per_dim" not in routing_payload else float(routing_payload["trace_per_dim"][i]),
                            "local_rank0_score": None if routing_payload is None or "rank0_score" not in routing_payload else float(routing_payload["rank0_score"][i]),
                            "local_rank1_score": None if routing_payload is None or "rank1_score" not in routing_payload else float(routing_payload["rank1_score"][i]),
                            "local_rank01_relative_gap": None if routing_payload is None or "rank01_relative_gap" not in routing_payload else float(routing_payload["rank01_relative_gap"][i]),
                            "local_zero_rank_rescued": None if routing_payload is None or "zero_rank_rescued" not in routing_payload else int(round(float(routing_payload["zero_rank_rescued"][i]))),
                            "local_abs_gate": None if routing_payload is None or "abs_gate" not in routing_payload else float(routing_payload["abs_gate"][i]),
                            "local_relative_top1_weight": None if routing_payload is None or "relative_top1_weight" not in routing_payload else float(routing_payload["relative_top1_weight"][i]),
                            "local_relative_top1_weight_entropy": None if routing_payload is None or "relative_top1_weight_entropy" not in routing_payload else float(routing_payload["relative_top1_weight_entropy"][i]),
                            "local_relative_solver_margin": None if routing_payload is None or "relative_solver_margin" not in routing_payload else float(routing_payload["relative_solver_margin"][i]),
                            "local_relative_competition_active": None if routing_payload is None or "relative_competition_active" not in routing_payload else float(routing_payload["relative_competition_active"][i]),
                            "local_local_solver_weighted_delta_norm": None if routing_payload is None or "local_solver_weighted_delta_norm" not in routing_payload else float(routing_payload["local_solver_weighted_delta_norm"][i]),
                            "local_relative_solver_top1_index": None if routing_payload is None or "relative_solver_top1_index" not in routing_payload else int(routing_payload["relative_solver_top1_index"][i]),
                        }
                    )
                sample_offset += len(batch_y_np)
                pred = final_pred
            else:
                logits = model(batch_x, fusion_alpha=1.0) if args.arm == "e2" else model(batch_x)
                pred = logits.argmax(dim=1).cpu().numpy()
            all_pred.append(pred)
            all_y.append(batch_y_np)

    y_true = np.concatenate(all_y, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    wallclock = float(time.time() - start_time)
    test_acc = float(accuracy_score(y_true, y_pred))
    test_macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    summary = {
        "dataset": str(args.dataset),
        "arm": str(args.arm),
        "seed": int(args.seed),
        "test_acc": test_acc,
        "test_macro_f1": test_macro_f1,
        "wallclock_seconds": wallclock,
    }
    _write_json(os.path.join(run_dir, "summary.json"), summary)
    _write_csv(
        os.path.join(run_dir, "per_dataset.csv"),
        [summary],
    )
    if bool(args.dataflow_probe) and args.arm == "e2" and stage_rows:
        y_true_stage = np.concatenate(all_y, axis=0)
        base_pred = np.concatenate(base_pred_all, axis=0)
        local_pred = np.concatenate(local_pred_all, axis=0)
        final_pred = np.concatenate(final_pred_all, axis=0)
        agreement_summary = {
            "base_acc": float(accuracy_score(y_true_stage, base_pred)),
            "local_acc": float(accuracy_score(y_true_stage, local_pred)),
            "final_acc": float(accuracy_score(y_true_stage, final_pred)),
            "base_local_agreement": float(np.mean(base_pred == local_pred)),
            "base_final_agreement": float(np.mean(base_pred == final_pred)),
            "local_final_agreement": float(np.mean(local_pred == final_pred)),
            "final_override_rate": float(np.mean(final_pred != base_pred)),
            "helpful_override_rate": float(np.mean((final_pred == y_true_stage) & (base_pred != y_true_stage))),
            "harmful_override_rate": float(np.mean((final_pred != y_true_stage) & (base_pred == y_true_stage))),
        }
        if gate_values:
            gate_concat = np.concatenate(gate_values, axis=0)
            agreement_summary["gate_mean"] = float(gate_concat.mean())
            agreement_summary["gate_min"] = float(gate_concat.min())
            agreement_summary["gate_max"] = float(gate_concat.max())
        if selected_top1_weight_all:
            top1_indices = np.concatenate(selected_top1_index_all, axis=0)
            top1_weights = np.concatenate(selected_top1_weight_all, axis=0)
            routing_entropy = np.concatenate(selected_routing_entropy_all, axis=0)
            cos_gap = np.concatenate(selected_cos_gap_all, axis=0)
            routing_width = int(max(selected_routing_widths))
            counts = np.bincount(top1_indices.clip(min=0), minlength=max(1, routing_width))
            probs = counts.astype(np.float64) / max(1.0, float(counts.sum()))
            occupancy_entropy = _normalized_entropy_from_probs(probs)
            raw_entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum()) if counts.sum() > 0 else 0.0
            agreement_summary["same_weight_max_mean"] = float(top1_weights.mean())
            agreement_summary["subproto_weight_entropy_mean"] = float(routing_entropy.mean())
            agreement_summary["subproto_cos_top1_top2_gap_mean"] = float(cos_gap.mean())
            agreement_summary["subproto_top1_occupancy_entropy"] = float(occupancy_entropy)
            agreement_summary["subproto_usage_effective_count"] = float(np.exp(raw_entropy))
            agreement_summary["subproto_top1_occupancy_counts"] = [int(v) for v in counts.tolist()]
        if selected_rank_all:
            selected_rank = np.concatenate(selected_rank_all, axis=0)
            lw_alpha = np.concatenate(selected_lw_alpha_all, axis=0) if selected_lw_alpha_all else np.zeros_like(selected_rank, dtype=np.float64)
            rank_counts = np.bincount(selected_rank.clip(min=0), minlength=max(1, int(selected_rank.max(initial=0)) + 1))
            agreement_summary["selected_rank_distribution"] = {str(idx): int(value) for idx, value in enumerate(rank_counts.tolist())}
            agreement_summary["mean_selected_rank"] = float(selected_rank.mean())
            agreement_summary["k0_fallback_rate"] = float(np.mean(selected_rank == 0))
            agreement_summary["lw_shrinkage_alpha_mean"] = float(lw_alpha.mean())
        if selected_ppca_sigma2_all:
            sigma2 = np.concatenate(selected_ppca_sigma2_all, axis=0)
            agreement_summary["ppca_sigma2_mean"] = float(sigma2.mean())
        if selected_posterior_confidence_all:
            posterior_confidence = np.concatenate(selected_posterior_confidence_all, axis=0)
            agreement_summary["posterior_confidence_mean"] = float(posterior_confidence.mean())
            agreement_summary["posterior_confidence_std"] = float(posterior_confidence.std())
            agreement_summary["posterior_confidence_q10"] = float(np.quantile(posterior_confidence, 0.10))
            agreement_summary["posterior_confidence_q50"] = float(np.quantile(posterior_confidence, 0.50))
            agreement_summary["posterior_confidence_q90"] = float(np.quantile(posterior_confidence, 0.90))
            agreement_summary["posterior_confidence_qgap"] = float(
                agreement_summary["posterior_confidence_q90"] - agreement_summary["posterior_confidence_q10"]
            )
            agreement_summary["posterior_confidence_qratio"] = float(
                agreement_summary["posterior_confidence_q90"] / max(agreement_summary["posterior_confidence_q10"], 1e-12)
            )
            agreement_summary["posterior_confidence_far_decay_mean"] = float((1.0 - posterior_confidence).mean())
        if selected_posterior_log_confidence_all:
            posterior_log_confidence = np.concatenate(selected_posterior_log_confidence_all, axis=0)
            agreement_summary["posterior_log_confidence_mean"] = float(posterior_log_confidence.mean())
            agreement_summary["posterior_log_confidence_std"] = float(posterior_log_confidence.std())
        if selected_posterior_residual_energy_all:
            posterior_residual_energy = np.concatenate(selected_posterior_residual_energy_all, axis=0)
            agreement_summary["posterior_residual_energy_mean"] = float(posterior_residual_energy.mean())
        if selected_posterior_residual_energy_per_dim_all:
            posterior_residual_energy_per_dim = np.concatenate(selected_posterior_residual_energy_per_dim_all, axis=0)
            agreement_summary["posterior_residual_energy_per_dim_mean"] = float(posterior_residual_energy_per_dim.mean())
        if selected_posterior_sigma2_eff_all:
            posterior_sigma2_eff = np.concatenate(selected_posterior_sigma2_eff_all, axis=0)
            agreement_summary["posterior_sigma2_eff_mean"] = float(posterior_sigma2_eff.mean())
        if selected_posterior_sigma2_used_all:
            posterior_sigma2_used = np.concatenate(selected_posterior_sigma2_used_all, axis=0)
            agreement_summary["posterior_sigma2_used_mean"] = float(posterior_sigma2_used.mean())
        if selected_trace_per_dim_all:
            trace_per_dim = np.concatenate(selected_trace_per_dim_all, axis=0)
            agreement_summary["trace_per_dim_mean"] = float(trace_per_dim.mean())
        if selected_rank0_score_all:
            rank0_score = np.concatenate(selected_rank0_score_all, axis=0)
            agreement_summary["rank0_score_mean"] = float(rank0_score.mean())
        if selected_rank1_score_all:
            rank1_score = np.concatenate(selected_rank1_score_all, axis=0)
            agreement_summary["rank1_score_mean"] = float(rank1_score.mean())
        if selected_rank01_relative_gap_all:
            rank01_relative_gap = np.concatenate(selected_rank01_relative_gap_all, axis=0)
            agreement_summary["rank01_relative_gap_mean"] = float(rank01_relative_gap.mean())
        if selected_zero_rank_rescued_all:
            zero_rank_rescued = np.concatenate(selected_zero_rank_rescued_all, axis=0)
            agreement_summary["zero_rank_rescued_rate"] = float(zero_rank_rescued.mean())
        if selected_abs_gate_all:
            abs_gate = np.concatenate(selected_abs_gate_all, axis=0)
            agreement_summary["abs_gate_mean"] = float(abs_gate.mean())
            agreement_summary["abs_gate_q10"] = float(np.quantile(abs_gate, 0.10))
            agreement_summary["abs_gate_q50"] = float(np.quantile(abs_gate, 0.50))
            agreement_summary["abs_gate_q90"] = float(np.quantile(abs_gate, 0.90))
        if selected_relative_top1_weight_all:
            relative_top1_weight = np.concatenate(selected_relative_top1_weight_all, axis=0)
            agreement_summary["relative_top1_weight_mean"] = float(relative_top1_weight.mean())
        if selected_relative_top1_weight_entropy_all:
            relative_top1_weight_entropy = np.concatenate(selected_relative_top1_weight_entropy_all, axis=0)
            agreement_summary["relative_top1_weight_entropy"] = float(relative_top1_weight_entropy.mean())
        if selected_relative_solver_margin_all:
            relative_solver_margin = np.concatenate(selected_relative_solver_margin_all, axis=0)
            agreement_summary["relative_solver_margin_mean"] = float(relative_solver_margin.mean())
        if selected_relative_competition_active_all:
            relative_competition_active = np.concatenate(selected_relative_competition_active_all, axis=0)
            agreement_summary["relative_competition_active_rate"] = float(relative_competition_active.mean())
        if selected_local_solver_weighted_delta_norm_all:
            local_solver_weighted_delta_norm = np.concatenate(selected_local_solver_weighted_delta_norm_all, axis=0)
            agreement_summary["local_solver_weighted_delta_norm_mean"] = float(local_solver_weighted_delta_norm.mean())
        if selected_relative_solver_top1_index_all:
            active_mask = None
            if selected_relative_competition_active_all:
                active_mask = np.concatenate(selected_relative_competition_active_all, axis=0) > 0.5
            top1_index = np.concatenate(selected_relative_solver_top1_index_all, axis=0)
            if active_mask is not None and active_mask.any():
                active_top1 = top1_index[active_mask]
                counts = np.bincount(active_top1.clip(min=0), minlength=max(1, int(active_top1.max(initial=0)) + 1))
                probs = counts.astype(np.float64) / max(1.0, float(counts.sum()))
                agreement_summary["solver_top1_occupancy_entropy"] = float(_normalized_entropy_from_probs(probs))
            else:
                agreement_summary["solver_top1_occupancy_entropy"] = 0.0
        local_head = getattr(model, "local_head", None)
        if local_head is not None and hasattr(local_head, "export_learned_prototype_geometry_summary"):
            learned_geometry = local_head.export_learned_prototype_geometry_summary()
            if learned_geometry is not None:
                agreement_summary.update({f"learned_{k}": v for k, v in learned_geometry.items()})
        if dataflow_first_batch_summary is None:
            dataflow_first_batch_summary = {"dataset": str(args.dataset), "arm": str(args.arm)}
        _export_dataflow_probe_artifacts(
            run_dir=run_dir,
            first_batch_summary=dataflow_first_batch_summary,
            stage_rows=stage_rows,
            agreement_summary=agreement_summary,
        )
        if bool(args.emit_mdl_rank_trace):
            _export_mdl_rank_trace_artifact(
                run_dir=run_dir,
                first_batch_summary=dataflow_first_batch_summary,
            )
    if bool(args.closed_form_probe) and args.arm == "e2":
        _export_probe_artifacts(model, run_dir=run_dir)
    if bool(args.tangent_probe) and args.arm == "e2":
        _export_tangent_probe_artifacts(model, run_dir=run_dir)
    print(
        f"[done][{args.dataset}][{args.arm}] acc={test_acc:.4f} macro_f1={test_macro_f1:.4f} wallclock={wallclock:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
