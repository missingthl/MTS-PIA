from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from aeon.classification.convolution_based import MultiRocketHydraClassifier

# Local imports
from core.resnet1d import ResNet1DClassifier
from core.patchtst import PatchTST
from core.timesnet import TimesNet

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        if self.mode == 'min':
            current_score = -score
        else:
            current_score = score

        if self.best_score is None:
            self.best_score = current_score
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0

def build_model(n_kernels: int = 10000, random_state: int = 42, n_jobs: int = 1):
    return MultiRocketHydraClassifier(
        n_kernels=n_kernels,
        random_state=random_state,
        n_jobs=n_jobs
    )

def fit_eval_minirocket(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "stop_epoch": 0
    }

def _get_dev(device_str):
    use_cuda = torch.cuda.is_available() and str(device_str).startswith("cuda")
    return torch.device(device_str if use_cuda else "cpu")


class IndexedTensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], int(idx)


def _make_tensor_loader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    indexed: bool = False,
    seed: Optional[int] = None,
) -> DataLoader:
    if indexed:
        dataset = IndexedTensorDataset(X, y)
    else:
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())

    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
    }
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        kwargs["generator"] = generator
    return DataLoader(dataset, **kwargs)


def _evaluate(model, loader, dev, criterion):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev)
            logits = model(bx)
            loss = criterion(logits, by)
            total_loss += loss.item() * bx.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(by.cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return avg_loss, float(acc), float(f1)


def fit_eval_pytorch_model(
    model,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    epochs=100,
    lr=1e-3,
    batch_size=64,
    patience=10,
    device="cuda",
    use_cosine_annealing=False,
    return_model_obj=False,
    loader_seed: Optional[int] = None,
) -> Dict[str, float]:
    dev = _get_dev(device)
    model.to(dev)

    train_loader = _make_tensor_loader(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        indexed=False,
        seed=loader_seed,
    )

    val_loader = None
    if X_val is not None:
        val_loader = _make_tensor_loader(
            X_val,
            y_val,
            batch_size=batch_size,
            shuffle=False,
            indexed=False,
        )

    test_loader = _make_tensor_loader(
        X_test,
        y_test,
        batch_size=batch_size,
        shuffle=False,
        indexed=False,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    scheduler = None
    if use_cosine_annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
    early_stopping = EarlyStopping(patience=patience, mode='max') # Maximize Val F1
    
    best_val_f1 = 0
    best_val_loss = float('inf')
    stop_epoch = epochs
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(dev, non_blocking=True), by.to(dev, non_blocking=True)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        if scheduler:
            scheduler.step()
            
        if val_loader:
            v_loss, v_acc, v_f1 = _evaluate(model, val_loader, dev, criterion)
            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                best_val_loss = v_loss
            
            early_stopping(v_f1, model)
            if early_stopping.early_stop:
                stop_epoch = epoch + 1
                break
        else:
            # Fallback if no val set
            stop_epoch = epoch + 1

    if early_stopping.best_model_state:
        model.load_state_dict(early_stopping.best_model_state)
        
    _, t_acc, t_f1 = _evaluate(model, test_loader, dev, criterion)
    
    res = {
        "accuracy": t_acc,
        "macro_f1": t_f1,
        "best_val_f1": best_val_f1,
        "best_val_loss": best_val_loss,
        "stop_epoch": stop_epoch,
    }
    if return_model_obj:
        res["model_obj"] = model  # Return for theory probing
    return res


def _true_class_margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
    if logits.shape[1] <= 1:
        return true_logits
    masked = logits.clone()
    masked.scatter_(1, y.view(-1, 1), -torch.inf)
    other_logits = torch.max(masked, dim=1).values
    return true_logits - other_logits


def fit_eval_pytorch_model_weighted_aug_ce(
    model,
    X_train,
    y_train,
    X_aug,
    y_aug,
    X_val,
    y_val,
    X_test,
    y_test,
    epochs=100,
    lr=1e-3,
    batch_size=64,
    patience=10,
    device="cuda",
    use_cosine_annealing=False,
    feedback_margin_temperature: float = 1.0,
    aug_loss_weight: float = 1.0,
    loader_seed: Optional[int] = None,
) -> Dict[str, float]:
    dev = _get_dev(device)
    model.to(dev)

    orig_loader = _make_tensor_loader(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        indexed=False,
        seed=loader_seed,
    )
    aug_loader = None
    if X_aug is not None and y_aug is not None and len(y_aug) > 0:
        aug_loader = _make_tensor_loader(
            X_aug,
            y_aug,
            batch_size=batch_size,
            shuffle=True,
            indexed=False,
            seed=None if loader_seed is None else int(loader_seed) + 991,
        )

    val_loader = None
    if X_val is not None:
        val_loader = _make_tensor_loader(
            X_val,
            y_val,
            batch_size=batch_size,
            shuffle=False,
            indexed=False,
        )

    test_loader = _make_tensor_loader(
        X_test,
        y_test,
        batch_size=batch_size,
        shuffle=False,
        indexed=False,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    criterion_per_sample = nn.CrossEntropyLoss(reduction="none")
    scheduler = None
    if use_cosine_annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    early_stopping = EarlyStopping(patience=patience, mode="max")
    best_val_f1 = 0
    best_val_loss = float("inf")
    stop_epoch = epochs
    last_orig_ce_loss = 0.0
    last_weighted_aug_ce_loss = 0.0
    last_aug_margin_mean = 0.0
    weight_values: List[float] = []

    tau = max(1e-6, float(feedback_margin_temperature))
    lambda_aug = float(aug_loss_weight)

    for epoch in range(epochs):
        model.train()
        aug_iter = iter(aug_loader) if aug_loader is not None else None
        epoch_orig_losses: List[float] = []
        epoch_aug_losses: List[float] = []
        epoch_margins: List[float] = []
        for bx, by in orig_loader:
            bx, by = bx.to(dev, non_blocking=True), by.to(dev, non_blocking=True)
            optimizer.zero_grad()
            logits = model(bx)
            loss_orig = criterion(logits, by)
            loss = loss_orig
            epoch_orig_losses.append(float(loss_orig.item()))

            if aug_iter is not None and aug_loader is not None:
                try:
                    ax, ay = next(aug_iter)
                except StopIteration:
                    aug_iter = iter(aug_loader)
                    ax, ay = next(aug_iter)
                ax, ay = ax.to(dev, non_blocking=True), ay.to(dev, non_blocking=True)
                logits_aug = model(ax)
                margin = _true_class_margin(logits_aug, ay)
                weights = torch.sigmoid(margin / tau).detach()
                ce_aug = criterion_per_sample(logits_aug, ay)
                weighted_aug = torch.mean(weights * ce_aug)
                loss = loss + lambda_aug * weighted_aug
                epoch_aug_losses.append(float(weighted_aug.item()))
                epoch_margins.append(float(torch.mean(margin.detach()).item()))
                weight_values.extend(weights.detach().cpu().numpy().astype(float).tolist())

            loss.backward()
            optimizer.step()

        if epoch_orig_losses:
            last_orig_ce_loss = float(np.mean(epoch_orig_losses))
        if epoch_aug_losses:
            last_weighted_aug_ce_loss = float(np.mean(epoch_aug_losses))
        else:
            last_weighted_aug_ce_loss = 0.0
        if epoch_margins:
            last_aug_margin_mean = float(np.mean(epoch_margins))
        else:
            last_aug_margin_mean = 0.0

        if scheduler:
            scheduler.step()

        if val_loader:
            v_loss, v_acc, v_f1 = _evaluate(model, val_loader, dev, criterion)
            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                best_val_loss = v_loss
            early_stopping(v_f1, model)
            if early_stopping.early_stop:
                stop_epoch = epoch + 1
                break
        else:
            stop_epoch = epoch + 1

    if early_stopping.best_model_state:
        model.load_state_dict(early_stopping.best_model_state)

    _, t_acc, t_f1 = _evaluate(model, test_loader, dev, criterion)
    weights_np = np.asarray(weight_values, dtype=np.float64)
    return {
        "accuracy": t_acc,
        "macro_f1": t_f1,
        "best_val_f1": best_val_f1,
        "best_val_loss": best_val_loss,
        "stop_epoch": stop_epoch,
        "feedback_weight_mean": float(np.mean(weights_np)) if weights_np.size else 0.0,
        "feedback_weight_std": float(np.std(weights_np)) if weights_np.size else 0.0,
        "last_orig_ce_loss": float(last_orig_ce_loss),
        "last_weighted_aug_ce_loss": float(last_weighted_aug_ce_loss),
        "last_aug_margin_mean": float(last_aug_margin_mean),
    }

def fit_eval_resnet1d(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    epochs=30,
    lr=1e-3,
    batch_size=64,
    patience=10,
    device="cuda",
    return_model_obj=False,
    loader_seed: Optional[int] = None,
):
    in_channels = X_train.shape[1]
    num_classes = int(max(y_train.max(), (y_val.max() if y_val is not None else 0), y_test.max()) + 1)
    model = ResNet1DClassifier(in_channels, num_classes)
    return fit_eval_pytorch_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        epochs,
        lr,
        batch_size,
        patience,
        device,
        return_model_obj=return_model_obj,
        loader_seed=loader_seed,
    )


def fit_eval_resnet1d_weighted_aug_ce(
    X_train,
    y_train,
    X_aug,
    y_aug,
    X_val,
    y_val,
    X_test,
    y_test,
    epochs=30,
    lr=1e-3,
    batch_size=64,
    patience=10,
    device="cuda",
    feedback_margin_temperature: float = 1.0,
    aug_loss_weight: float = 1.0,
    loader_seed: Optional[int] = None,
):
    in_channels = X_train.shape[1]
    num_classes = int(max(y_train.max(), (y_val.max() if y_val is not None else 0), y_test.max()) + 1)
    model = ResNet1DClassifier(in_channels, num_classes)
    return fit_eval_pytorch_model_weighted_aug_ce(
        model,
        X_train,
        y_train,
        X_aug,
        y_aug,
        X_val,
        y_val,
        X_test,
        y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        device=device,
        feedback_margin_temperature=feedback_margin_temperature,
        aug_loss_weight=aug_loss_weight,
        loader_seed=loader_seed,
    )

def fit_eval_patchtst(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    epochs=100,
    lr=5e-4,
    batch_size=64,
    patience=15,
    device="cuda",
    return_model_obj=False,
    loader_seed: Optional[int] = None,
):
    in_channels = X_train.shape[1]
    seq_len = X_train.shape[2]
    num_classes = int(max(y_train.max(), (y_val.max() if y_val is not None else 0), y_test.max()) + 1)
    model = PatchTST(in_channels, seq_len, num_classes)
    return fit_eval_pytorch_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        epochs,
        lr,
        batch_size,
        patience,
        device,
        use_cosine_annealing=True,
        return_model_obj=return_model_obj,
        loader_seed=loader_seed,
    )


def fit_eval_patchtst_weighted_aug_ce(
    X_train,
    y_train,
    X_aug,
    y_aug,
    X_val,
    y_val,
    X_test,
    y_test,
    epochs=100,
    lr=5e-4,
    batch_size=64,
    patience=15,
    device="cuda",
    feedback_margin_temperature: float = 1.0,
    aug_loss_weight: float = 1.0,
    loader_seed: Optional[int] = None,
):
    in_channels = X_train.shape[1]
    seq_len = X_train.shape[2]
    num_classes = int(max(y_train.max(), (y_val.max() if y_val is not None else 0), y_test.max()) + 1)
    model = PatchTST(in_channels, seq_len, num_classes)
    return fit_eval_pytorch_model_weighted_aug_ce(
        model,
        X_train,
        y_train,
        X_aug,
        y_aug,
        X_val,
        y_val,
        X_test,
        y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        device=device,
        use_cosine_annealing=True,
        feedback_margin_temperature=feedback_margin_temperature,
        aug_loss_weight=aug_loss_weight,
        loader_seed=loader_seed,
    )

def fit_eval_timesnet(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    epochs=100,
    lr=5e-4,
    batch_size=32,
    patience=15,
    device="cuda",
    return_model_obj=False,
    loader_seed: Optional[int] = None,
):
    in_channels = X_train.shape[1]
    seq_len = X_train.shape[2]
    num_classes = int(max(y_train.max(), (y_val.max() if y_val is not None else 0), y_test.max()) + 1)
    model = TimesNet(in_channels, seq_len, num_classes)
    return fit_eval_pytorch_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        epochs,
        lr,
        batch_size,
        patience,
        device,
        return_model_obj=return_model_obj,
        loader_seed=loader_seed,
    )


def fit_eval_timesnet_weighted_aug_ce(
    X_train,
    y_train,
    X_aug,
    y_aug,
    X_val,
    y_val,
    X_test,
    y_test,
    epochs=100,
    lr=5e-4,
    batch_size=32,
    patience=15,
    device="cuda",
    feedback_margin_temperature: float = 1.0,
    aug_loss_weight: float = 1.0,
    loader_seed: Optional[int] = None,
):
    in_channels = X_train.shape[1]
    seq_len = X_train.shape[2]
    num_classes = int(max(y_train.max(), (y_val.max() if y_val is not None else 0), y_test.max()) + 1)
    model = TimesNet(in_channels, seq_len, num_classes)
    return fit_eval_pytorch_model_weighted_aug_ce(
        model,
        X_train,
        y_train,
        X_aug,
        y_aug,
        X_val,
        y_val,
        X_test,
        y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        device=device,
        feedback_margin_temperature=feedback_margin_temperature,
        aug_loss_weight=aug_loss_weight,
        loader_seed=loader_seed,
    )
