from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from core.dual_stream_classifier import (
    DualStreamClassifier,
    DualStreamModelConfig,
    ManifoldOnlyClassifier,
    SpatialOnlyClassifier,
)
from core.dual_stream_dataset import (
    DualStreamState,
    DualStreamTrialDataset,
    collate_dual_stream_batch,
)


@dataclass(frozen=True)
class DualStreamEvalConfig:
    variant: str
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    device: str = "auto"


@dataclass
class DualStreamEvalResult:
    dataset: str
    seed: int
    variant: str
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    best_epoch: int
    history_rows: List[Dict[str, float]] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_device(text: str) -> torch.device:
    if str(text).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(text))


def _build_model(variant: str, model_cfg: DualStreamModelConfig) -> nn.Module:
    key = str(variant).strip().lower()
    if key == "spatial_only":
        return SpatialOnlyClassifier(model_cfg)
    if key == "manifold_only":
        return ManifoldOnlyClassifier(model_cfg)
    if key == "dual_stream":
        return DualStreamClassifier(model_cfg)
    raise ValueError(f"unknown dual-stream variant: {variant}")


def _compute_loss(model: nn.Module, outputs: Dict[str, torch.Tensor], labels: torch.Tensor, variant: str) -> torch.Tensor:
    ce = nn.CrossEntropyLoss()
    logits = outputs["logits"]
    loss = ce(logits, labels)
    if str(variant).strip().lower() == "dual_stream":
        aux_w = float(getattr(model, "aux_weight", 0.0))
        if aux_w > 0.0:
            loss = loss + aux_w * ce(outputs["spatial_logits"], labels) + aux_w * ce(outputs["manifold_logits"], labels)
    return loss


@torch.no_grad()
def _evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    *,
    variant: str,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    loss_rows: List[float] = []
    for batch in loader:
        labels = batch["labels"].to(device)
        raw_x = batch["raw_x"].to(device)
        z = batch["z"].to(device)
        outputs = model(raw_x, z)
        loss = _compute_loss(model, outputs, labels, variant)
        pred = torch.argmax(outputs["logits"], dim=1)
        loss_rows.append(float(loss.detach().cpu().item()))
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
    if not y_true:
        return {"acc": 0.0, "macro_f1": 0.0, "loss": 0.0}
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "loss": float(np.mean(loss_rows)) if loss_rows else 0.0,
    }


def evaluate_dual_stream(
    state: DualStreamState,
    *,
    seed: int,
    model_cfg: DualStreamModelConfig,
    eval_cfg: DualStreamEvalConfig,
) -> DualStreamEvalResult:
    _set_seed(int(seed))
    device = _resolve_device(eval_cfg.device)
    model = _build_model(eval_cfg.variant, model_cfg).to(device)

    train_ds = DualStreamTrialDataset(state.train)
    val_ds = DualStreamTrialDataset(state.val)
    test_ds = DualStreamTrialDataset(state.test)

    train_loader = DataLoader(train_ds, batch_size=int(eval_cfg.batch_size), shuffle=True, collate_fn=collate_dual_stream_batch)
    eval_train_loader = DataLoader(train_ds, batch_size=int(eval_cfg.batch_size), shuffle=False, collate_fn=collate_dual_stream_batch)
    val_loader = DataLoader(val_ds, batch_size=int(eval_cfg.batch_size), shuffle=False, collate_fn=collate_dual_stream_batch)
    test_loader = DataLoader(test_ds, batch_size=int(eval_cfg.batch_size), shuffle=False, collate_fn=collate_dual_stream_batch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(eval_cfg.lr), weight_decay=float(eval_cfg.weight_decay))
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_f1 = float("-inf")
    stale_epochs = 0
    history_rows: List[Dict[str, float]] = []

    for epoch in range(1, int(eval_cfg.epochs) + 1):
        model.train()
        loss_rows: List[float] = []
        for batch in train_loader:
            optimizer.zero_grad()
            labels = batch["labels"].to(device)
            raw_x = batch["raw_x"].to(device)
            z = batch["z"].to(device)
            outputs = model(raw_x, z)
            loss = _compute_loss(model, outputs, labels, eval_cfg.variant)
            loss.backward()
            optimizer.step()
            loss_rows.append(float(loss.detach().cpu().item()))

        train_metrics = _evaluate_split(model, eval_train_loader, variant=eval_cfg.variant, device=device)
        val_metrics = _evaluate_split(model, val_loader, variant=eval_cfg.variant, device=device)
        history_rows.append(
            {
                "epoch": float(epoch),
                "train_loss_epoch_mean": float(np.mean(loss_rows)) if loss_rows else 0.0,
                "train_acc": float(train_metrics["acc"]),
                "train_macro_f1": float(train_metrics["macro_f1"]),
                "val_acc": float(val_metrics["acc"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
            }
        )

        if float(val_metrics["macro_f1"]) > float(best_val_f1) + 1e-9:
            best_val_f1 = float(val_metrics["macro_f1"])
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(eval_cfg.patience):
                break

    model.load_state_dict(best_state)
    train_metrics = _evaluate_split(model, eval_train_loader, variant=eval_cfg.variant, device=device)
    val_metrics = _evaluate_split(model, val_loader, variant=eval_cfg.variant, device=device)
    test_metrics = _evaluate_split(model, test_loader, variant=eval_cfg.variant, device=device)

    return DualStreamEvalResult(
        dataset=str(state.dataset),
        seed=int(seed),
        variant=str(eval_cfg.variant),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        best_epoch=int(best_epoch),
        history_rows=history_rows,
        meta={
            "device": str(device),
            "epochs_requested": int(eval_cfg.epochs),
            "epochs_ran": int(len(history_rows)),
            "batch_size": int(eval_cfg.batch_size),
            "lr": float(eval_cfg.lr),
            "weight_decay": float(eval_cfg.weight_decay),
            "patience": int(eval_cfg.patience),
        },
    )
