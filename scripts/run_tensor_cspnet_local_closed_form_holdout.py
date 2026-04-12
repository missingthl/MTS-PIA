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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.local_closed_form_residual_head import TensorCSPNetLocalClosedFormResidual  # noqa: E402
from models.tensor_cspnet_adapter import (  # noqa: E402
    TensorCSPNetAdapter,
    ensure_tensor_cspnet_reference_on_path,
    get_tensor_cspnet_reference_root,
)
from models.tensor_cspnet_residual_linear import TensorCSPNetResidualLinear  # noqa: E402

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


def _adjust_learning_rate(optimizer, *, initial_lr: float, decay: float, epoch: int) -> None:
    optimizer.lr = float(initial_lr) * (float(decay) ** (int(epoch) // 100))


def _build_model(args: argparse.Namespace, *, channel_num: int, device: torch.device) -> torch.nn.Module:
    if args.arm == "e0":
        model = TensorCSPNetAdapter(
            channel_num=int(channel_num),
            mlp=bool(args.mlp),
            dataset="BCIC",
        )
    elif args.arm == "e1":
        model = TensorCSPNetResidualLinear(
            channel_num=int(channel_num),
            mlp=bool(args.mlp),
            dataset="BCIC",
            init_beta=float(args.init_beta),
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
):
    train_dataset = dataloader_in_main(
        torch.from_numpy(train_x).double(),
        torch.LongTensor(train_y),
    )
    test_dataset = dataloader_in_main(
        torch.from_numpy(test_x).double(),
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
            print(
                f"[{args.arm}][subject={subject}] epoch={epoch}/{args.epochs} "
                f"train_loss={epoch_loss:.6f} train_acc={epoch_acc:.4f}",
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
            logits = model(batch_x)
            output = F.log_softmax(logits, dim=-1)
            loss = F.nll_loss(output, batch_y)
            test_loss += float(loss.item()) * int(batch_y.shape[0])
            pred = output.data.max(1, keepdim=True)[1]
            test_correct += int(pred.eq(batch_y.data.view_as(pred)).long().sum().item())
            test_count += int(batch_y.shape[0])

    return {
        "train_seconds": float(train_seconds),
        "test_macro_acc": float(test_correct / max(1, test_count)),
        "test_loss": float(test_loss / max(1, test_count)),
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
    p.add_argument("--out-root", type=str, default=os.path.join(ROOT, "out", "_active", "verify_tensor_cspnet_local_closed_form_holdout_20260412"))
    p.add_argument("--run-tag", type=str, default="")
    p.add_argument("--init-beta", type=float, default=0.1)
    p.add_argument("--prototypes-per-class", type=int, default=4)
    p.add_argument("--routing-temperature", type=float, default=1.0)
    p.add_argument("--closed-form-ridge", type=float, default=1e-2)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _set_seed(int(args.seed))

    use_cuda = not bool(args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

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
        "init_beta": float(args.init_beta),
        "prototypes_per_class": int(args.prototypes_per_class),
        "routing_temperature": float(args.routing_temperature),
        "closed_form_ridge": float(args.closed_form_ridge),
    }
    _write_json(os.path.join(run_dir, "run_meta.json"), meta)

    rows: List[Dict[str, object]] = []
    for subject in range(int(args.start_no), int(args.end_no) + 1):
        print(f"[{args.arm}] subject={subject} load_start", flush=True)
        with _reference_repo_cwd():
            dataset = load_BCIC(subject, alg_name="Tensor_CSPNet", scenario="Holdout")
            train_x, test_x, train_y, test_y = dataset.generate_training_valid_test_set_Holdout()
        train_loader, test_loader = _make_loaders(
            train_x,
            test_x,
            train_y,
            test_y,
            train_batch_size=int(args.train_batch_size),
            test_batch_size=int(args.test_batch_size),
            num_workers=int(args.num_workers),
            use_cuda=bool(use_cuda),
        )

        model = _build_model(args, channel_num=int(train_x.shape[1] * train_x.shape[2]), device=device)
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
