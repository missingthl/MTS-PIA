#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from datasets.trial_dataset_factory import load_trials_for_dataset, normalize_dataset_name
from models.tensor_cspnet_adapter import (
    TensorCSPNetAdapter,
    ensure_tensor_cspnet_reference_on_path,
)

ensure_tensor_cspnet_reference_on_path()
import utils.geoopt as ref_geoopt


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


class ExtandedTensorCSPNetAdapter(torch.nn.Module):
    def __init__(self, channel_num: int, num_classes: int, mlp: bool, spd_dtype: torch.dtype):
        super().__init__()
        self.adapter = TensorCSPNetAdapter(
            channel_num=channel_num,
            mlp=mlp,
            dataset="fixedsplit_generic",
            spd_dtype=spd_dtype
        )
        self.num_classes = num_classes
        # Verify and inject classifier override if num classes differ 
        if self.adapter.num_classes != num_classes:
            latent_dim = self.adapter.base_model.Temporal_Block(-1).shape[-1] if hasattr(self.adapter.base_model, 'Temporal_Block') else 32
            # Force replacing the classifier
            in_f = self.adapter.base_model.Classifier.in_features if isinstance(self.adapter.base_model.Classifier, torch.nn.Linear) else \
                   self.adapter.base_model.Classifier[-1].in_features
            self.adapter.base_model.Classifier = torch.nn.Linear(in_f, num_classes).float()
    
    def forward(self, x):
        # x is [B, C, L]. Adapter expects [B, window_num, band_num, channels, time]
        x = x.unsqueeze(1).unsqueeze(2) 
        return self.adapter(x, return_features=False)


def _adjust_learning_rate(optimizer, *, initial_lr: float, decay: float, epoch: int) -> None:
    optimizer.lr = float(initial_lr) * (float(decay) ** (int(epoch) // 100))


def _train_one_seed(
    model: torch.nn.Module,
    optimizer,
    train_loader,
    test_loader,
    *,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
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
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * int(batch_y.shape[0])
            pred = logits.data.max(1, keepdim=True)[1]
            total_correct += int(pred.eq(batch_y.data.view_as(pred)).long().sum().item())
            total_count += int(batch_y.shape[0])

        if int(epoch) % int(args.log_every) == 0 or int(epoch) == int(args.epochs):
            epoch_loss = total_loss / max(1, total_count)
            epoch_acc = total_correct / max(1, total_count)
            print(
                f"[tensor-spdnet] epoch={epoch}/{args.epochs} "
                f"train_loss={epoch_loss:.6f} train_acc={epoch_acc:.4f}",
                flush=True,
            )

    train_seconds = time.time() - train_start

    model.eval()
    test_loss = 0.0
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            test_loss += float(loss.item()) * int(batch_y.shape[0])
            pred = logits.data.max(1, keepdim=True)[1]
            test_preds.extend(pred.cpu().numpy().ravel())
            test_labels.extend(batch_y.cpu().numpy().ravel())

    test_acc = accuracy_score(test_labels, test_preds)
    test_macro_f1 = f1_score(test_labels, test_preds, average="macro")

    return {
        "train_seconds": float(train_seconds),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "test_loss": float(test_loss / max(1, len(test_labels))),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--train-batch-size", type=int, default=32)
    p.add_argument("--test-batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--initial-lr", type=float, default=1e-3)
    p.add_argument("--decay", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--mlp", action="store_true", default=False)
    p.add_argument("--spd-precision", type=str, default="fp64", choices=["fp64", "fp32"])
    p.add_argument("--out-root", type=str, default="out/_active/verify_tensor_spdnet_fixedsplit")
    p.add_argument("--run-name", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _set_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spd_dtype = torch.float64 if args.spd_precision == "fp64" else torch.float32

    run_name = args.run_name.strip() or f"seed{int(args.seed)}"
    run_dir = os.path.join(str(args.out_root), "e0", run_name)
    _ensure_dir(run_dir)

    train_x, test_x, train_y, test_y, num_classes = _load_fixedsplit_arrays(args)
    train_loader, test_loader = _make_loaders(
        train_x, test_x, train_y, test_y,
        train_batch_size=int(args.train_batch_size),
        test_batch_size=int(args.test_batch_size),
        num_workers=int(args.num_workers),
        use_cuda=torch.cuda.is_available()
    )

    in_channels = int(train_x.shape[1])
    
    model = ExtandedTensorCSPNetAdapter(
        channel_num=in_channels,
        num_classes=num_classes,
        mlp=args.mlp,
        spd_dtype=spd_dtype
    ).to(device)

    # Tensor-SPDNet MUST use RiemannianAdam
    optimizer = ref_geoopt.optim.RiemannianAdam(model.parameters(), lr=float(args.initial_lr))

    run_start = time.time()
    result = _train_one_seed(
        model, optimizer, train_loader, test_loader,
        device=device, args=args, seed=int(args.seed)
    )
    wallclock = time.time() - run_start

    summary = {
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "test_acc": float(result["test_acc"]),
        "test_macro_f1": float(result["test_macro_f1"]),
        "test_loss": float(result["test_loss"]),
        "train_seconds": float(result["train_seconds"]),
        "wallclock_seconds": float(wallclock),
    }

    _write_json(os.path.join(run_dir, "summary.json"), summary)
    
    # Store pseudo summary_per_seed.csv to stay compatible with run aggregation
    pd_row = {
        "dataset": summary["dataset"],
        "seed": summary["seed"],
        "test_acc": summary["test_acc"],
        "test_macro_f1": summary["test_macro_f1"]
    }
    import pandas as pd
    pd.DataFrame([pd_row]).to_csv(os.path.join(run_dir, "summary_per_seed.csv"), index=False)

    print(f"[tensor-spdnet] summary {json.dumps(summary)}")

if __name__ == "__main__":
    main()
