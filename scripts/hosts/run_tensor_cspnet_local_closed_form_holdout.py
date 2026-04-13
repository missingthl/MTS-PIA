#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from models.local_closed_form_residual_head import TensorCSPNetLocalClosedFormResidual  # noqa: E402
from models.tensor_cspnet_adapter import (  # noqa: E402
    TensorCSPNetAdapter,
    ensure_tensor_cspnet_reference_on_path,
    get_tensor_cspnet_reference_root,
)
from models.tensor_cspnet_residual_linear import TensorCSPNetResidualLinear  # noqa: E402
from route_b_unified.risk_aware_axis_controller import (  # noqa: E402
    FragilityProbeConfig,
    compute_class_fragility_scores,
)
from scripts.support.fisher_pia_utils import FisherPIAConfig  # noqa: E402

ensure_tensor_cspnet_reference_on_path()

from utils.load_data import dataloader_in_main, load_BCIC  # noqa: E402
import utils.geoopt as ref_geoopt  # noqa: E402


@contextmanager
def _reference_repo_cwd():
    prev_cwd = os.getcwd()
    ref_root = get_tensor_cspnet_reference_root()
    os.chdir(ref_root)
    try:
        yield ref_root
    finally:
        os.chdir(prev_cwd)


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


def _subject_cache_path(cache_root: str, *, subject: int) -> str:
    return os.path.join(str(cache_root), f"bcic_holdout_tensor_subject{int(subject):02d}.npz")


def _load_subject_arrays(
    subject: int,
    *,
    cache_root: str,
    disable_subject_cache: bool,
):
    cache_path = _subject_cache_path(cache_root, subject=int(subject))
    if not bool(disable_subject_cache) and os.path.exists(cache_path):
        cached = np.load(cache_path)
        return (
            cached["train_x"],
            cached["test_x"],
            cached["train_y"],
            cached["test_y"],
        )

    with _reference_repo_cwd():
        dataset = load_BCIC(subject, alg_name="Tensor_CSPNet", scenario="Holdout")
        train_x, test_x, train_y, test_y = dataset.generate_training_valid_test_set_Holdout()

    if not bool(disable_subject_cache):
        _ensure_dir(str(cache_root))
        np.savez(
            cache_path,
            train_x=train_x,
            test_x=test_x,
            train_y=train_y,
            test_y=test_y,
        )

    return train_x, test_x, train_y, test_y


def _adjust_learning_rate(optimizer, *, initial_lr: float, decay: float, epoch: int) -> None:
    optimizer.lr = float(initial_lr) * (float(decay) ** (int(epoch) // 100))


def _resolve_spd_dtype(spd_precision: str) -> torch.dtype:
    value = str(spd_precision).strip().lower()
    if value == "fp64":
        return torch.float64
    if value == "fp32":
        return torch.float32
    raise ValueError(f"unsupported spd precision: {spd_precision}")


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


def _safe_rank_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    xx = np.asarray(x, dtype=np.float64).ravel()
    yy = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 2:
        return None
    if np.allclose(xx, xx[0]) or np.allclose(yy, yy[0]):
        return None
    xx_rank = np.argsort(np.argsort(xx)).astype(np.float64)
    yy_rank = np.argsort(np.argsort(yy)).astype(np.float64)
    if np.allclose(xx_rank, xx_rank[0]) or np.allclose(yy_rank, yy_rank[0]):
        return None
    return float(np.corrcoef(xx_rank, yy_rank)[0, 1])


def _build_model(
    args: argparse.Namespace,
    *,
    channel_num: int,
    device: torch.device,
    spd_dtype: torch.dtype,
) -> torch.nn.Module:
    if args.arm == "e0":
        model = TensorCSPNetAdapter(
            channel_num=int(channel_num),
            mlp=bool(args.mlp),
            dataset="BCIC",
            spd_dtype=spd_dtype,
        )
    elif args.arm == "e1":
        model = TensorCSPNetResidualLinear(
            channel_num=int(channel_num),
            mlp=bool(args.mlp),
            dataset="BCIC",
            init_beta=float(args.init_beta),
            spd_dtype=spd_dtype,
        )
    elif args.arm == "e2":
        model = TensorCSPNetLocalClosedFormResidual(
            channel_num=int(channel_num),
            mlp=bool(args.mlp),
            dataset="BCIC",
            prototypes_per_class=int(args.prototypes_per_class),
            routing_temperature=float(args.routing_temperature),
            ridge=float(args.closed_form_ridge),
            init_beta=float(args.init_beta),
            detach_local_input=bool(args.detach_local_latent),
            support_mode=str(args.local_support_mode),
            prototype_aggregation=str(args.prototype_aggregation),
            readout_gate_mode=str(args.local_readout_gate),
            spd_dtype=spd_dtype,
        )
    else:
        raise ValueError(f"unknown arm: {args.arm}")
    return model.to(device)


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
    feature_dtype: torch.dtype,
):
    train_dataset = dataloader_in_main(
        torch.from_numpy(train_x).to(dtype=feature_dtype),
        torch.LongTensor(train_y),
    )
    test_dataset = dataloader_in_main(
        torch.from_numpy(test_x).to(dtype=feature_dtype),
        torch.LongTensor(test_y),
    )

    train_kwargs = {"batch_size": int(train_batch_size), "shuffle": True}
    test_kwargs = {"batch_size": int(test_batch_size), "shuffle": False}
    if use_cuda:
        cuda_kwargs = {
            "num_workers": int(num_workers),
            "pin_memory": True,
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    return train_loader, test_loader


def _forward_outputs(model: torch.nn.Module, batch_x: torch.Tensor, *, args: argparse.Namespace):
    if args.arm == "e2":
        return model(batch_x, fusion_alpha=1.0, return_features=True)
    return model(batch_x, return_features=True)


def _collect_probe_bundle(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    latents: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    base_logits: List[np.ndarray] = []
    final_logits: List[np.ndarray] = []
    local_logits: List[np.ndarray] = []
    readout_gates: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            outputs = _forward_outputs(model, batch_x, args=args)

            latent = outputs.latent.detach().cpu().numpy()
            base_logit = outputs.base_logit.detach().cpu().numpy()
            final_logit = outputs.final_logit.detach().cpu().numpy()
            pred = np.argmax(final_logit, axis=-1)

            latents.append(latent)
            labels.append(batch_y.detach().cpu().numpy())
            preds.append(pred.astype(np.int64))
            base_logits.append(base_logit)
            final_logits.append(final_logit)

            if hasattr(outputs, "local_closed_form_logit") and outputs.local_closed_form_logit is not None:
                local_logits.append(outputs.local_closed_form_logit.detach().cpu().numpy())
            if hasattr(outputs, "readout_gate") and outputs.readout_gate is not None:
                readout_gates.append(outputs.readout_gate.detach().cpu().numpy())

    bundle: Dict[str, np.ndarray] = {
        "latent": np.concatenate(latents, axis=0),
        "y_true": np.concatenate(labels, axis=0).astype(np.int64),
        "y_pred": np.concatenate(preds, axis=0).astype(np.int64),
        "base_logit": np.concatenate(base_logits, axis=0),
        "final_logit": np.concatenate(final_logits, axis=0),
    }
    if local_logits:
        bundle["local_logit"] = np.concatenate(local_logits, axis=0)
    if readout_gates:
        bundle["readout_gate"] = np.concatenate(readout_gates, axis=0)
    return bundle


def _build_probe_outputs(
    *,
    model: torch.nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    train_bundle = _collect_probe_bundle(model, train_loader, device=device, args=args)
    test_bundle = _collect_probe_bundle(model, test_loader, device=device, args=args)

    fisher_cfg = FisherPIAConfig(
        knn_k=int(args.fragility_knn_k),
        interior_quantile=float(args.fragility_interior_quantile),
        boundary_quantile=float(args.fragility_boundary_quantile),
        hetero_k=int(args.fragility_hetero_k),
    )
    probe_cfg = FragilityProbeConfig(
        alpha=float(args.fragility_alpha),
        beta=float(args.fragility_beta),
        gamma=float(args.fragility_gamma),
        eps=float(args.fragility_eps),
    )
    _raw_scores, fragility_rows, _fragility_meta = compute_class_fragility_scores(
        train_bundle["latent"],
        train_bundle["y_true"],
        fisher_cfg=fisher_cfg,
        probe_cfg=probe_cfg,
    )

    num_classes = int(model.adapter.num_classes if hasattr(model, "adapter") else model.num_classes)
    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        test_bundle["y_true"],
        test_bundle["y_pred"],
        labels=labels,
        zero_division=0,
    )

    fragility_by_class = {
        int(row["class_id"]): float(row["fragility_score_norm"])
        for row in fragility_rows
    }
    class_metric_rows: List[Dict[str, object]] = []
    for class_idx in labels:
        mask = test_bundle["y_true"] == int(class_idx)
        row: Dict[str, object] = {
            "class_id": int(class_idx),
            "precision": float(precision[class_idx]),
            "recall": float(recall[class_idx]),
            "f1": float(f1[class_idx]),
            "support": int(support[class_idx]),
            "fragility_score_norm": float(fragility_by_class.get(int(class_idx), 0.0)),
        }
        if "readout_gate" in test_bundle:
            if np.any(mask):
                row["true_class_gate_mean"] = float(
                    np.mean(test_bundle["readout_gate"][mask, int(class_idx)])
                )
            else:
                row["true_class_gate_mean"] = 0.0
        class_metric_rows.append(row)

    fragility_vec = np.asarray([row["fragility_score_norm"] for row in class_metric_rows], dtype=np.float64)
    recall_vec = np.asarray([row["recall"] for row in class_metric_rows], dtype=np.float64)
    f1_vec = np.asarray([row["f1"] for row in class_metric_rows], dtype=np.float64)
    gate_vec = (
        np.asarray([row.get("true_class_gate_mean", np.nan) for row in class_metric_rows], dtype=np.float64)
        if any("true_class_gate_mean" in row for row in class_metric_rows)
        else None
    )

    summary = {
        "fragility_vs_recall_spearman": _safe_rank_corr(fragility_vec, recall_vec),
        "fragility_vs_f1_spearman": _safe_rank_corr(fragility_vec, f1_vec),
        "mean_fragility": float(np.mean(fragility_vec)) if fragility_vec.size else 0.0,
        "mean_recall": float(np.mean(recall_vec)) if recall_vec.size else 0.0,
        "mean_f1": float(np.mean(f1_vec)) if f1_vec.size else 0.0,
    }
    if gate_vec is not None:
        summary["fragility_vs_gate_spearman"] = _safe_rank_corr(fragility_vec, gate_vec)
        summary["mean_true_class_gate"] = float(np.nanmean(gate_vec)) if gate_vec.size else 0.0
    return fragility_rows, class_metric_rows, summary


def _train_one_subject(
    model: torch.nn.Module,
    optimizer,
    train_loader,
    test_loader,
    *,
    device: torch.device,
    args: argparse.Namespace,
    subject: int,
) -> Dict[str, float]:
    train_start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        _adjust_learning_rate(
            optimizer,
            initial_lr=float(args.initial_lr),
            decay=float(args.decay),
            epoch=int(epoch),
        )
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            if args.arm == "e2":
                fusion_alpha = _compute_fusion_alpha(args, epoch=epoch)
                logits = model(batch_x, fusion_alpha=float(fusion_alpha))
            else:
                fusion_alpha = 1.0
                logits = model(batch_x)
            output = F.log_softmax(logits, dim=-1)
            loss = F.nll_loss(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * int(batch_y.shape[0])
            pred = output.data.max(1, keepdim=True)[1]
            total_correct += int(pred.eq(batch_y.data.view_as(pred)).long().sum().item())
            total_count += int(batch_y.shape[0])

        if int(epoch) % int(args.log_every) == 0 or int(epoch) == int(args.epochs):
            epoch_loss = total_loss / max(1, total_count)
            epoch_acc = total_correct / max(1, total_count)
            beta_suffix = ""
            if args.arm == "e2" and hasattr(model, "beta"):
                beta_suffix = f" fusion_alpha={fusion_alpha:.4f} beta={float(model.beta.detach().cpu().item()):.4f}"
            print(
                f"[{args.arm}][subject={subject}] epoch={epoch}/{args.epochs} "
                f"train_loss={epoch_loss:.6f} train_acc={epoch_acc:.4f}{beta_suffix}",
                flush=True,
            )

    train_seconds = time.time() - train_start

    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_count = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if args.arm == "e2":
                logits = model(batch_x, fusion_alpha=1.0)
            else:
                logits = model(batch_x)
            output = F.log_softmax(logits, dim=-1)
            loss = F.nll_loss(output, batch_y)
            test_loss += float(loss.item()) * int(batch_y.shape[0])
            pred = output.data.max(1, keepdim=True)[1]
            test_correct += int(pred.eq(batch_y.data.view_as(pred)).long().sum().item())
            test_count += int(batch_y.shape[0])

    probe_outputs = None
    if bool(args.export_fragility_probe):
        probe_outputs = _build_probe_outputs(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            args=args,
        )

    return {
        "train_seconds": float(train_seconds),
        "test_macro_acc": float(test_correct / max(1, test_count)),
        "test_loss": float(test_loss / max(1, test_count)),
        "probe_outputs": probe_outputs,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tensor-CSPNet holdout with residual local closed-form arms.")
    p.add_argument("--arm", type=str, default="e0", choices=["e0", "e1", "e2"])
    p.add_argument("--dataset", type=str, default="BCIC", choices=["BCIC"])
    p.add_argument("--start-no", type=int, default=1)
    p.add_argument("--end-no", type=int, default=9)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--train-batch-size", type=int, default=29)
    p.add_argument("--test-batch-size", type=int, default=29)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--initial-lr", type=float, default=1e-3)
    p.add_argument("--decay", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--mlp", action="store_true", default=False)
    p.add_argument("--no-cuda", action="store_true", default=False)
    p.add_argument("--spd-precision", type=str, default="fp64", choices=["fp64", "fp32"])
    p.add_argument(
        "--cache-root",
        type=str,
        default=os.path.join(ROOT, "cache", "tensor_cspnet_bcic_holdout"),
    )
    p.add_argument("--disable-subject-cache", action="store_true", default=False)
    p.add_argument("--out-root", type=str, default=os.path.join(ROOT, "out", "_active", "verify_tensor_cspnet_local_closed_form_holdout_20260412"))
    p.add_argument("--run-tag", type=str, default="")
    p.add_argument("--init-beta", type=float, default=0.1)
    p.add_argument("--prototypes-per-class", type=int, default=4)
    p.add_argument("--routing-temperature", type=float, default=1.0)
    p.add_argument("--closed-form-ridge", type=float, default=1e-2)
    p.add_argument("--detach-local-latent", action="store_true", default=False)
    p.add_argument("--fusion-warmup-hold-epochs", type=int, default=0)
    p.add_argument("--fusion-warmup-ramp-epochs", type=int, default=0)
    p.add_argument("--fusion-warmup-start-scale", type=float, default=0.0)
    p.add_argument(
        "--local-support-mode",
        type=str,
        default="same_opp_balanced",
        choices=["same_opp_balanced", "same_only", "same_opp_asym"],
    )
    p.add_argument(
        "--prototype-aggregation",
        type=str,
        default="pooled",
        choices=["pooled", "committee_mean"],
    )
    p.add_argument(
        "--local-readout-gate",
        type=str,
        default="none",
        choices=["none", "consistency"],
    )
    p.add_argument("--export-fragility-probe", action="store_true", default=False)
    p.add_argument("--fragility-knn-k", type=int, default=20)
    p.add_argument("--fragility-interior-quantile", type=float, default=0.70)
    p.add_argument("--fragility-boundary-quantile", type=float, default=0.30)
    p.add_argument("--fragility-hetero-k", type=int, default=3)
    p.add_argument("--fragility-alpha", type=float, default=1.0)
    p.add_argument("--fragility-beta", type=float, default=1.0)
    p.add_argument("--fragility-gamma", type=float, default=1.0)
    p.add_argument("--fragility-eps", type=float, default=1e-8)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _set_seed(int(args.seed))

    use_cuda = not bool(args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    spd_dtype = _resolve_spd_dtype(str(args.spd_precision))

    run_tag = str(args.run_tag).strip() or time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(str(args.out_root), str(args.arm), f"seed{int(args.seed)}_{run_tag}")
    _ensure_dir(run_dir)

    meta = {
        "arm": str(args.arm),
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "start_no": int(args.start_no),
        "end_no": int(args.end_no),
        "epochs": int(args.epochs),
        "train_batch_size": int(args.train_batch_size),
        "test_batch_size": int(args.test_batch_size),
        "num_workers": int(args.num_workers),
        "initial_lr": float(args.initial_lr),
        "decay": float(args.decay),
        "mlp": bool(args.mlp),
        "use_cuda": bool(use_cuda),
        "device": str(device),
        "spd_precision": str(args.spd_precision),
        "cache_root": str(args.cache_root),
        "disable_subject_cache": bool(args.disable_subject_cache),
        "init_beta": float(args.init_beta),
        "prototypes_per_class": int(args.prototypes_per_class),
        "routing_temperature": float(args.routing_temperature),
        "closed_form_ridge": float(args.closed_form_ridge),
        "detach_local_latent": bool(args.detach_local_latent),
        "fusion_warmup_hold_epochs": int(args.fusion_warmup_hold_epochs),
        "fusion_warmup_ramp_epochs": int(args.fusion_warmup_ramp_epochs),
        "fusion_warmup_start_scale": float(args.fusion_warmup_start_scale),
        "local_support_mode": str(args.local_support_mode),
        "prototype_aggregation": str(args.prototype_aggregation),
        "local_readout_gate": str(args.local_readout_gate),
        "export_fragility_probe": bool(args.export_fragility_probe),
        "fragility_knn_k": int(args.fragility_knn_k),
        "fragility_interior_quantile": float(args.fragility_interior_quantile),
        "fragility_boundary_quantile": float(args.fragility_boundary_quantile),
        "fragility_hetero_k": int(args.fragility_hetero_k),
        "fragility_alpha": float(args.fragility_alpha),
        "fragility_beta": float(args.fragility_beta),
        "fragility_gamma": float(args.fragility_gamma),
        "fragility_eps": float(args.fragility_eps),
    }
    _write_json(os.path.join(run_dir, "run_meta.json"), meta)

    rows: List[Dict[str, object]] = []
    for subject in range(int(args.start_no), int(args.end_no) + 1):
        print(f"[{args.arm}] subject={subject} load_start", flush=True)
        train_x, test_x, train_y, test_y = _load_subject_arrays(
            int(subject),
            cache_root=str(args.cache_root),
            disable_subject_cache=bool(args.disable_subject_cache),
        )
        train_loader, test_loader = _make_loaders(
            train_x,
            test_x,
            train_y,
            test_y,
            train_batch_size=int(args.train_batch_size),
            test_batch_size=int(args.test_batch_size),
            num_workers=int(args.num_workers),
            use_cuda=bool(use_cuda),
            feature_dtype=spd_dtype,
        )

        model = _build_model(
            args,
            channel_num=int(train_x.shape[1] * train_x.shape[2]),
            device=device,
            spd_dtype=spd_dtype,
        )
        optimizer = ref_geoopt.optim.RiemannianAdam(model.parameters(), lr=float(args.initial_lr))

        subject_start = time.time()
        result = _train_one_subject(
            model,
            optimizer,
            train_loader,
            test_loader,
            device=device,
            args=args,
            subject=int(subject),
        )
        wallclock_seconds = time.time() - subject_start
        row = {
            "subject": int(subject),
            "test_macro_acc": float(result["test_macro_acc"]),
            "test_loss": float(result["test_loss"]),
            "train_seconds": float(result["train_seconds"]),
            "wallclock_seconds": float(wallclock_seconds),
            "arm": str(args.arm),
            "seed": int(args.seed),
        }
        rows.append(row)
        if bool(args.export_fragility_probe) and result.get("probe_outputs") is not None:
            fragility_rows, class_metric_rows, probe_summary = result["probe_outputs"]
            _write_csv(os.path.join(run_dir, f"subject_{int(subject):02d}_fragility_rows.csv"), fragility_rows)
            _write_csv(os.path.join(run_dir, f"subject_{int(subject):02d}_test_class_metrics.csv"), class_metric_rows)
            _write_json(
                os.path.join(run_dir, f"subject_{int(subject):02d}_probe_summary.json"),
                {
                    "subject": int(subject),
                    "arm": str(args.arm),
                    "seed": int(args.seed),
                    **probe_summary,
                },
            )
        print(
            f"[{args.arm}] subject={subject} done "
            f"acc={row['test_macro_acc']:.4f} loss={row['test_loss']:.6f} "
            f"wallclock={row['wallclock_seconds']:.1f}",
            flush=True,
        )

    mean_acc = float(np.mean([float(r["test_macro_acc"]) for r in rows])) if rows else 0.0
    summary = {
        "arm": str(args.arm),
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "subject_count": len(rows),
        "mean_test_macro_acc": mean_acc,
    }
    _write_csv(os.path.join(run_dir, "per_subject.csv"), rows)
    _write_json(os.path.join(run_dir, "summary.json"), summary)
    print(f"[{args.arm}] mean_test_macro_acc={mean_acc:.4f}", flush=True)
    print(f"[{args.arm}] run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
