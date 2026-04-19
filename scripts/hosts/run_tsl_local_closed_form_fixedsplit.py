#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from models.patchtst_adapter import PatchTSTAdapter  # noqa: E402
from models.patchtst_local_closed_form import PatchTSTLocalClosedFormResidual  # noqa: E402
from models.patchtst_residual_linear import PatchTSTResidualLinear  # noqa: E402
from models.timesnet_adapter import TimesNetAdapter  # noqa: E402
from models.timesnet_local_closed_form import TimesNetLocalClosedFormResidual  # noqa: E402
from models.timesnet_residual_linear import TimesNetResidualLinear  # noqa: E402
from scripts.hosts.run_resnet1d_local_closed_form_fixedsplit import (  # noqa: E402
    _compute_fusion_alpha,
    _ensure_dir,
    _export_dataflow_probe_artifacts,
    _export_probe_artifacts,
    _export_tangent_probe_artifacts,
    _load_fixedsplit_arrays,
    _make_loaders,
    _maybe_set_probe_context,
    _normalized_entropy_from_probs,
    _set_seed,
    _write_csv,
    _write_json,
    build_argparser as _build_resnet_argparser,
)
from datasets.trial_dataset_factory import normalize_dataset_name  # noqa: E402


def _build_model(args: argparse.Namespace, *, in_channels: int, seq_len: int, num_classes: int) -> torch.nn.Module:
    backbone = str(args.host_backbone)
    common_kwargs = dict(
        in_channels=int(in_channels),
        seq_len=int(seq_len),
        num_classes=int(num_classes),
        init_beta=float(args.init_beta),
    )
    if backbone == "patchtst":
        if args.arm == "e0":
            return PatchTSTAdapter(
                in_channels=int(in_channels),
                seq_len=int(seq_len),
                num_classes=int(num_classes),
                d_model=int(args.backbone_d_model),
                d_ff=int(args.backbone_d_ff),
                e_layers=int(args.backbone_e_layers),
                n_heads=int(args.backbone_n_heads),
                factor=int(args.backbone_factor),
                dropout=float(args.backbone_dropout),
                activation=str(args.backbone_activation),
                patch_len=int(args.patch_len),
                patch_stride=int(args.patch_stride),
            )
        if args.arm == "e1":
            return PatchTSTResidualLinear(
                **common_kwargs,
                d_model=int(args.backbone_d_model),
                d_ff=int(args.backbone_d_ff),
                e_layers=int(args.backbone_e_layers),
                n_heads=int(args.backbone_n_heads),
                factor=int(args.backbone_factor),
                dropout=float(args.backbone_dropout),
                activation=str(args.backbone_activation),
                patch_len=int(args.patch_len),
                patch_stride=int(args.patch_stride),
            )
        if args.arm == "e2":
            return PatchTSTLocalClosedFormResidual(
                **common_kwargs,
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
                detach_local_input=bool(args.detach_local_latent),
                support_mode=str(args.local_support_mode),
                prototype_aggregation=str(args.prototype_aggregation),
                prototype_geometry_mode=str(args.prototype_geometry_mode),
                tangent_rank=int(args.tangent_rank),
                tangent_source=str(args.tangent_source),
                readout_gate_mode=str(args.local_readout_gate),
                d_model=int(args.backbone_d_model),
                d_ff=int(args.backbone_d_ff),
                e_layers=int(args.backbone_e_layers),
                n_heads=int(args.backbone_n_heads),
                factor=int(args.backbone_factor),
                dropout=float(args.backbone_dropout),
                activation=str(args.backbone_activation),
                patch_len=int(args.patch_len),
                patch_stride=int(args.patch_stride),
            )
    if backbone == "timesnet":
        if args.arm == "e0":
            return TimesNetAdapter(
                in_channels=int(in_channels),
                seq_len=int(seq_len),
                num_classes=int(num_classes),
                d_model=int(args.backbone_d_model),
                d_ff=int(args.backbone_d_ff),
                e_layers=int(args.backbone_e_layers),
                top_k=int(args.times_top_k),
                num_kernels=int(args.times_num_kernels),
                dropout=float(args.backbone_dropout),
                embed=str(args.times_embed),
                freq=str(args.times_freq),
            )
        if args.arm == "e1":
            return TimesNetResidualLinear(
                **common_kwargs,
                d_model=int(args.backbone_d_model),
                d_ff=int(args.backbone_d_ff),
                e_layers=int(args.backbone_e_layers),
                top_k=int(args.times_top_k),
                num_kernels=int(args.times_num_kernels),
                dropout=float(args.backbone_dropout),
                embed=str(args.times_embed),
                freq=str(args.times_freq),
            )
        if args.arm == "e2":
            return TimesNetLocalClosedFormResidual(
                **common_kwargs,
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
                detach_local_input=bool(args.detach_local_latent),
                support_mode=str(args.local_support_mode),
                prototype_aggregation=str(args.prototype_aggregation),
                prototype_geometry_mode=str(args.prototype_geometry_mode),
                tangent_rank=int(args.tangent_rank),
                tangent_source=str(args.tangent_source),
                readout_gate_mode=str(args.local_readout_gate),
                d_model=int(args.backbone_d_model),
                d_ff=int(args.backbone_d_ff),
                e_layers=int(args.backbone_e_layers),
                top_k=int(args.times_top_k),
                num_kernels=int(args.times_num_kernels),
                dropout=float(args.backbone_dropout),
                embed=str(args.times_embed),
                freq=str(args.times_freq),
            )
    raise ValueError(f"unknown backbone/arm: {backbone}/{args.arm}")


def _make_run_dir(args: argparse.Namespace) -> str:
    default_tag = f"{args.dataset}_{args.host_backbone}_{args.arm}_seed{int(args.seed)}"
    tag = str(args.run_tag).strip() if str(args.run_tag).strip() else default_tag
    out_root = args.out_root or os.path.join(
        "out",
        "_active",
        f"verify_tsl_local_closed_form_fixedsplit_{time.strftime('%Y%m%d')}",
    )
    return os.path.join(out_root, str(args.host_backbone), str(args.arm), tag)


def build_argparser(default_host_backbone: str | None = None) -> argparse.ArgumentParser:
    p = _build_resnet_argparser()
    p.description = "Time-Series-Library backbone + DLCR fixed-split TSC runner."
    p.add_argument(
        "--host-backbone",
        type=str,
        default=default_host_backbone,
        choices=["patchtst", "timesnet"],
        required=default_host_backbone is None,
    )
    p.add_argument("--backbone-d-model", type=int, default=128)
    p.add_argument("--backbone-d-ff", type=int, default=256)
    p.add_argument("--backbone-e-layers", type=int, default=3)
    p.add_argument("--backbone-dropout", type=float, default=0.1)
    p.add_argument("--backbone-n-heads", type=int, default=8)
    p.add_argument("--backbone-factor", type=int, default=1)
    p.add_argument("--backbone-activation", type=str, default="gelu")
    p.add_argument("--patch-len", type=int, default=16)
    p.add_argument("--patch-stride", type=int, default=8)
    p.add_argument("--times-top-k", type=int, default=3)
    p.add_argument("--times-num-kernels", type=int, default=4)
    p.add_argument("--times-embed", type=str, default="fixed")
    p.add_argument("--times-freq", type=str, default="h")
    return p


def main(default_host_backbone: str | None = None) -> None:
    args = build_argparser(default_host_backbone=default_host_backbone).parse_args()
    args.dataset = normalize_dataset_name(args.dataset)
    _set_seed(int(args.seed))

    run_dir = _make_run_dir(args)
    _ensure_dir(run_dir)

    train_x, test_x, train_y, test_y, num_classes = _load_fixedsplit_arrays(args)
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    model = _build_model(
        args,
        in_channels=int(train_x.shape[1]),
        seq_len=int(train_x.shape[2]),
        num_classes=int(num_classes),
    ).to(device)
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
        "runner_protocol": "tsl_local_closed_form_fixedsplit",
        "host_backbone": str(args.host_backbone),
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
        "local_readout_gate": str(args.local_readout_gate),
        "backbone_d_model": int(args.backbone_d_model),
        "backbone_d_ff": int(args.backbone_d_ff),
        "backbone_e_layers": int(args.backbone_e_layers),
        "backbone_dropout": float(args.backbone_dropout),
        "backbone_n_heads": int(args.backbone_n_heads),
        "backbone_factor": int(args.backbone_factor),
        "backbone_activation": str(args.backbone_activation),
        "patch_len": int(args.patch_len),
        "patch_stride": int(args.patch_stride),
        "times_top_k": int(args.times_top_k),
        "times_num_kernels": int(args.times_num_kernels),
        "times_embed": str(args.times_embed),
        "times_freq": str(args.times_freq),
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
            logits = model(batch_x, fusion_alpha=float(fusion_alpha)) if args.arm == "e2" else model(batch_x)
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
                f"[{args.dataset}][{args.host_backbone}][{args.arm}] epoch {epoch}/{args.epochs} "
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
    with torch.no_grad():
        _maybe_set_probe_context(model, split="test", epoch=int(args.epochs))
        sample_offset = 0
        for batch_idx, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y_np = batch_y.numpy()
            if args.arm == "e2" and bool(args.dataflow_probe):
                outputs = model(batch_x, fusion_alpha=1.0, return_features=True)
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
                if batch_idx == 0:
                    local_head = getattr(model, "local_head", None)
                    local_dataflow = None
                    if local_head is not None and hasattr(local_head, "export_last_dataflow_summary"):
                        local_dataflow = local_head.export_last_dataflow_summary()
                    dataflow_first_batch_summary = {
                        "dataset": str(args.dataset),
                        "host_backbone": str(args.host_backbone),
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
        "host_backbone": str(args.host_backbone),
        "arm": str(args.arm),
        "seed": int(args.seed),
        "test_acc": test_acc,
        "test_macro_f1": test_macro_f1,
        "wallclock_seconds": wallclock,
    }
    _write_json(os.path.join(run_dir, "summary.json"), summary)
    _write_csv(os.path.join(run_dir, "per_dataset.csv"), [summary])
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
    if bool(args.closed_form_probe) and args.arm == "e2":
        _export_probe_artifacts(model, run_dir=run_dir)
    if bool(args.tangent_probe) and args.arm == "e2":
        _export_tangent_probe_artifacts(model, run_dir=run_dir)
    print(
        f"[done][{args.dataset}][{args.host_backbone}][{args.arm}] "
        f"acc={test_acc:.4f} macro_f1={test_macro_f1:.4f} wallclock={wallclock:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
