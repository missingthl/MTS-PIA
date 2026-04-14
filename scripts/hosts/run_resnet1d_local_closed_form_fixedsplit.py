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
    DEFAULT_BASICMOTIONS_ROOT,
    DEFAULT_EPILEPSY_ROOT,
    DEFAULT_FINGERMOVEMENTS_ROOT,
    DEFAULT_HAR_ROOT,
    DEFAULT_HANDMOVEMENTDIRECTION_ROOT,
    DEFAULT_MITBIH_NPZ,
    DEFAULT_NATOPS_ROOT,
    DEFAULT_PENDIGITS_ROOT,
    DEFAULT_SELFREGULATIONSCP1_ROOT,
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
    p.add_argument("--local-support-mode", type=str, default="same_only", choices=["same_opp_balanced", "same_only", "same_opp_asym"])
    p.add_argument("--prototype-aggregation", type=str, default="pooled", choices=["pooled", "committee_mean"])
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
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "train_batch_size": int(args.train_batch_size),
        "test_batch_size": int(args.test_batch_size),
        "num_classes": int(num_classes),
        "n_train": int(train_x.shape[0]),
        "n_test": int(test_x.shape[0]),
        "in_channels": int(train_x.shape[1]),
        "seq_len": int(train_x.shape[2]),
        "closed_form_ridge": float(args.closed_form_ridge),
        "closed_form_ridge_mode": str(args.closed_form_ridge_mode),
        "closed_form_ridge_trace_eps": float(args.closed_form_ridge_trace_eps),
        "closed_form_solve_mode": str(args.closed_form_solve_mode),
        "closed_form_pinv_rcond": float(args.closed_form_pinv_rcond),
        "closed_form_input_norm_mode": str(args.closed_form_input_norm_mode),
        "closed_form_input_norm_eps": float(args.closed_form_input_norm_eps),
        "local_support_mode": str(args.local_support_mode),
        "prototype_aggregation": str(args.prototype_aggregation),
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
    with torch.no_grad():
        _maybe_set_probe_context(model, split="test", epoch=int(args.epochs))
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            logits = model(batch_x, fusion_alpha=1.0) if args.arm == "e2" else model(batch_x)
            pred = logits.argmax(dim=1).cpu().numpy()
            all_pred.append(pred)
            all_y.append(batch_y.numpy())

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
    if bool(args.closed_form_probe) and args.arm == "e2":
        _export_probe_artifacts(model, run_dir=run_dir)
    print(
        f"[done][{args.dataset}][{args.arm}] acc={test_acc:.4f} macro_f1={test_macro_f1:.4f} wallclock={wallclock:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
