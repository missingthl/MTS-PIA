from __future__ import annotations

import os
import random
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


def _set_training_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i % (2**32 - 1))
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
    _set_training_seed(loader_seed)
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
    _set_training_seed(loader_seed)
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


def _project_router_probs(
    probs: np.ndarray,
    active_mask: np.ndarray,
    *,
    min_prob: float,
) -> np.ndarray:
    active_idx = np.where(active_mask)[0]
    out = np.zeros_like(probs, dtype=np.float64)
    if active_idx.size == 0:
        return np.full_like(probs, 1.0 / max(1, probs.size), dtype=np.float64)
    if active_idx.size == 1:
        out[active_idx[0]] = 1.0
        return out

    probs_active = np.asarray(probs[active_idx], dtype=np.float64)
    probs_active = np.maximum(probs_active, 0.0)
    if float(np.sum(probs_active)) <= 1e-12:
        probs_active = np.full((active_idx.size,), 1.0 / active_idx.size, dtype=np.float64)
    else:
        probs_active = probs_active / float(np.sum(probs_active))

    lower = float(min(max(min_prob, 0.0), (1.0 / active_idx.size) - 1e-6))
    residual_budget = 1.0 - lower * active_idx.size
    residual = np.maximum(probs_active - lower, 0.0)
    if float(np.sum(residual)) <= 1e-12:
        probs_active = np.full((active_idx.size,), 1.0 / active_idx.size, dtype=np.float64)
    else:
        probs_active = lower + residual / float(np.sum(residual)) * residual_budget

    out[active_idx] = probs_active
    out = out / max(float(np.sum(out)), 1e-12)
    return out


def _router_target_probs(
    rewards: np.ndarray,
    active_mask: np.ndarray,
    *,
    temperature: float,
    min_prob: float,
) -> np.ndarray:
    active_idx = np.where(active_mask)[0]
    out = np.zeros_like(rewards, dtype=np.float64)
    if active_idx.size == 0:
        return np.full_like(rewards, 1.0 / max(1, rewards.size), dtype=np.float64)
    if active_idx.size == 1:
        out[active_idx[0]] = 1.0
        return out

    temp = max(float(temperature), 1e-6)
    rewards_active = np.asarray(rewards[active_idx], dtype=np.float64)
    scaled = rewards_active / temp
    scaled = scaled - float(np.max(scaled))
    probs_active = np.exp(scaled)
    probs_active = probs_active / max(float(np.sum(probs_active)), 1e-12)
    out[active_idx] = probs_active
    return _project_router_probs(out, active_mask, min_prob=min_prob)


def fit_eval_pytorch_model_adaptive_aug_ce(
    model,
    X_train,
    y_train,
    X_aug_lraes,
    y_aug_lraes,
    X_aug_zpia,
    y_aug_zpia,
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
    router_temperature: float = 0.05,
    router_min_prob: float = 0.10,
    router_smoothing: float = 0.5,
    loader_seed: Optional[int] = None,
) -> Dict[str, float]:
    _set_training_seed(loader_seed)
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

    aug_loaders = {
        "lraes": None,
        "zpia": None,
    }
    if X_aug_lraes is not None and y_aug_lraes is not None and len(y_aug_lraes) > 0:
        aug_loaders["lraes"] = _make_tensor_loader(
            X_aug_lraes,
            y_aug_lraes,
            batch_size=batch_size,
            shuffle=True,
            indexed=False,
            seed=None if loader_seed is None else int(loader_seed) + 991,
        )
    if X_aug_zpia is not None and y_aug_zpia is not None and len(y_aug_zpia) > 0:
        aug_loaders["zpia"] = _make_tensor_loader(
            X_aug_zpia,
            y_aug_zpia,
            batch_size=batch_size,
            shuffle=True,
            indexed=False,
            seed=None if loader_seed is None else int(loader_seed) + 1991,
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
    last_weighted_aug_ce_loss_lraes = 0.0
    last_weighted_aug_ce_loss_zpia = 0.0
    last_aug_margin_mean = 0.0
    last_aug_margin_mean_lraes = 0.0
    last_aug_margin_mean_zpia = 0.0

    tau = max(1e-6, float(feedback_margin_temperature))
    lambda_aug = float(aug_loss_weight)
    active_mask = np.asarray(
        [aug_loaders["lraes"] is not None, aug_loaders["zpia"] is not None],
        dtype=bool,
    )
    router_probs = _project_router_probs(
        np.asarray([0.5, 0.5], dtype=np.float64),
        active_mask,
        min_prob=float(router_min_prob),
    )
    if int(np.sum(active_mask)) == 2:
        router_probs[:] = 0.5

    weight_values_all: List[float] = []
    weight_values_by_engine: Dict[str, List[float]] = {"lraes": [], "zpia": []}
    router_trace: List[Dict[str, float]] = []
    last_rewards = {"lraes": 0.0, "zpia": 0.0}

    for epoch in range(epochs):
        model.train()
        aug_iters = {name: (iter(loader) if loader is not None else None) for name, loader in aug_loaders.items()}
        epoch_orig_losses: List[float] = []
        epoch_engine_losses: Dict[str, List[float]] = {"lraes": [], "zpia": []}
        epoch_engine_rewards: Dict[str, List[float]] = {"lraes": [], "zpia": []}
        epoch_engine_margins: Dict[str, List[float]] = {"lraes": [], "zpia": []}
        epoch_probs = router_probs.copy()
        epoch_prob_tensor = torch.tensor(epoch_probs, dtype=torch.float32, device=dev).detach()

        for bx, by in orig_loader:
            bx, by = bx.to(dev, non_blocking=True), by.to(dev, non_blocking=True)
            optimizer.zero_grad()
            logits = model(bx)
            loss_orig = criterion(logits, by)
            loss = loss_orig
            epoch_orig_losses.append(float(loss_orig.item()))

            for engine_idx, engine_name in enumerate(["lraes", "zpia"]):
                aug_loader = aug_loaders[engine_name]
                aug_iter = aug_iters[engine_name]
                if aug_loader is None or aug_iter is None:
                    continue
                try:
                    ax, ay = next(aug_iter)
                except StopIteration:
                    aug_iter = iter(aug_loader)
                    aug_iters[engine_name] = aug_iter
                    ax, ay = next(aug_iter)
                ax, ay = ax.to(dev, non_blocking=True), ay.to(dev, non_blocking=True)
                logits_aug = model(ax)
                margin = _true_class_margin(logits_aug, ay)
                weights = torch.sigmoid(margin / tau).detach()
                ce_aug = criterion_per_sample(logits_aug, ay)
                weighted_aug = torch.mean(weights * ce_aug)
                loss = loss + lambda_aug * epoch_prob_tensor[engine_idx] * weighted_aug
                epoch_engine_losses[engine_name].append(float(weighted_aug.item()))
                epoch_engine_rewards[engine_name].append(float(torch.mean(weights).item()))
                epoch_engine_margins[engine_name].append(float(torch.mean(margin.detach()).item()))
                weights_np = weights.detach().cpu().numpy().astype(float)
                weight_values_all.extend(weights_np.tolist())
                weight_values_by_engine[engine_name].extend(weights_np.tolist())

            loss.backward()
            optimizer.step()

        last_orig_ce_loss = float(np.mean(epoch_orig_losses)) if epoch_orig_losses else 0.0
        last_weighted_aug_ce_loss_lraes = (
            float(np.mean(epoch_engine_losses["lraes"])) if epoch_engine_losses["lraes"] else 0.0
        )
        last_weighted_aug_ce_loss_zpia = (
            float(np.mean(epoch_engine_losses["zpia"])) if epoch_engine_losses["zpia"] else 0.0
        )
        combined_epoch_losses = epoch_engine_losses["lraes"] + epoch_engine_losses["zpia"]
        last_weighted_aug_ce_loss = float(np.mean(combined_epoch_losses)) if combined_epoch_losses else 0.0

        last_aug_margin_mean_lraes = (
            float(np.mean(epoch_engine_margins["lraes"])) if epoch_engine_margins["lraes"] else 0.0
        )
        last_aug_margin_mean_zpia = (
            float(np.mean(epoch_engine_margins["zpia"])) if epoch_engine_margins["zpia"] else 0.0
        )
        combined_epoch_margins = epoch_engine_margins["lraes"] + epoch_engine_margins["zpia"]
        last_aug_margin_mean = float(np.mean(combined_epoch_margins)) if combined_epoch_margins else 0.0

        rewards = np.asarray(
            [
                float(np.mean(epoch_engine_rewards["lraes"])) if epoch_engine_rewards["lraes"] else 0.0,
                float(np.mean(epoch_engine_rewards["zpia"])) if epoch_engine_rewards["zpia"] else 0.0,
            ],
            dtype=np.float64,
        )
        last_rewards = {"lraes": float(rewards[0]), "zpia": float(rewards[1])}
        p_target = _router_target_probs(
            rewards,
            active_mask,
            temperature=float(router_temperature),
            min_prob=float(router_min_prob),
        )
        router_probs = (1.0 - float(router_smoothing)) * router_probs + float(router_smoothing) * p_target
        router_probs = _project_router_probs(
            router_probs,
            active_mask,
            min_prob=float(router_min_prob),
        )

        router_trace.append(
            {
                "epoch": int(epoch + 1),
                "p_lraes": float(epoch_probs[0]),
                "p_zpia": float(epoch_probs[1]),
                "reward_lraes": float(rewards[0]),
                "reward_zpia": float(rewards[1]),
                "aug_loss_lraes": float(last_weighted_aug_ce_loss_lraes),
                "aug_loss_zpia": float(last_weighted_aug_ce_loss_zpia),
                "margin_lraes": float(last_aug_margin_mean_lraes),
                "margin_zpia": float(last_aug_margin_mean_zpia),
            }
        )

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
    weights_np = np.asarray(weight_values_all, dtype=np.float64)
    weights_np_lraes = np.asarray(weight_values_by_engine["lraes"], dtype=np.float64)
    weights_np_zpia = np.asarray(weight_values_by_engine["zpia"], dtype=np.float64)
    return {
        "accuracy": t_acc,
        "macro_f1": t_f1,
        "best_val_f1": best_val_f1,
        "best_val_loss": best_val_loss,
        "stop_epoch": stop_epoch,
        "feedback_weight_mean": float(np.mean(weights_np)) if weights_np.size else 0.0,
        "feedback_weight_std": float(np.std(weights_np)) if weights_np.size else 0.0,
        "feedback_weight_mean_lraes": float(np.mean(weights_np_lraes)) if weights_np_lraes.size else 0.0,
        "feedback_weight_mean_zpia": float(np.mean(weights_np_zpia)) if weights_np_zpia.size else 0.0,
        "last_orig_ce_loss": float(last_orig_ce_loss),
        "last_weighted_aug_ce_loss": float(last_weighted_aug_ce_loss),
        "last_weighted_aug_ce_loss_lraes": float(last_weighted_aug_ce_loss_lraes),
        "last_weighted_aug_ce_loss_zpia": float(last_weighted_aug_ce_loss_zpia),
        "last_aug_margin_mean": float(last_aug_margin_mean),
        "last_aug_margin_mean_lraes": float(last_aug_margin_mean_lraes),
        "last_aug_margin_mean_zpia": float(last_aug_margin_mean_zpia),
        "router_p_lraes_final": float(router_probs[0]),
        "router_p_zpia_final": float(router_probs[1]),
        "router_reward_lraes_last": float(last_rewards["lraes"]),
        "router_reward_zpia_last": float(last_rewards["zpia"]),
        "adaptive_best_engine_final": "lraes" if float(router_probs[0]) >= float(router_probs[1]) else "zpia",
        "router_trace": router_trace,
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
    _set_training_seed(loader_seed)
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
    _set_training_seed(loader_seed)
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


def fit_eval_resnet1d_adaptive_aug_ce(
    X_train,
    y_train,
    X_aug_lraes,
    y_aug_lraes,
    X_aug_zpia,
    y_aug_zpia,
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
    router_temperature: float = 0.05,
    router_min_prob: float = 0.10,
    router_smoothing: float = 0.5,
    loader_seed: Optional[int] = None,
):
    _set_training_seed(loader_seed)
    in_channels = X_train.shape[1]
    num_classes = int(max(y_train.max(), (y_val.max() if y_val is not None else 0), y_test.max()) + 1)
    model = ResNet1DClassifier(in_channels, num_classes)
    return fit_eval_pytorch_model_adaptive_aug_ce(
        model,
        X_train,
        y_train,
        X_aug_lraes,
        y_aug_lraes,
        X_aug_zpia,
        y_aug_zpia,
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
        router_temperature=router_temperature,
        router_min_prob=router_min_prob,
        router_smoothing=router_smoothing,
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
