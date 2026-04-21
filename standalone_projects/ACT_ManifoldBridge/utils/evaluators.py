from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("SupCon features must have shape [N, D]")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("SupCon labels must have shape [N]")
    if features.shape[0] <= 1:
        return features.new_zeros(())

    features = F.normalize(features, p=2, dim=-1)
    logits = torch.matmul(features, features.transpose(0, 1)) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.transpose(0, 1)).to(features.dtype)
    logits_mask = torch.ones_like(positive_mask) - torch.eye(features.shape[0], device=features.device, dtype=features.dtype)
    positive_mask = positive_mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not torch.any(valid):
        return features.new_zeros(())

    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (positive_count + 1e-12)
    return -mean_log_prob_pos[valid].mean()

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
    return_model_obj=False
) -> Dict[str, float]:
    dev = _get_dev(device)
    model.to(dev)
    
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if X_val is not None:
        val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
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


def fit_eval_resnet1d_acl(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    init_state_dict: Dict[str, torch.Tensor],
    selected_positive_map: Dict[int, List[np.ndarray]],
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 10,
    device: str = "cuda",
    acl_temperature: float = 0.07,
    acl_loss_weight: float = 0.2,
    return_model_obj: bool = False,
) -> Dict[str, float]:
    in_channels = X_train.shape[1]
    num_classes = int(max(y_train.max(), (y_val.max() if y_val is not None else 0), y_test.max()) + 1)
    model = ResNet1DClassifier(in_channels, num_classes)
    model.load_state_dict(copy.deepcopy(init_state_dict))

    dev = _get_dev(device)
    model.to(dev)

    train_ds = IndexedTensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None:
        val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience, mode='max')

    best_val_f1 = 0.0
    best_val_loss = float("inf")
    stop_epoch = epochs
    last_supcon_loss = 0.0
    last_ce_loss = 0.0

    for epoch in range(max(0, int(epochs))):
        model.train()
        running_cls = 0.0
        running_supcon = 0.0
        batch_count = 0

        for bx, by, bidx in train_loader:
            bx = bx.to(dev, non_blocking=True)
            by = by.to(dev, non_blocking=True)
            bidx_list = [int(i) for i in bidx.tolist()]

            optimizer.zero_grad()
            anchor_features = model.encode(bx)
            logits = model.classify(anchor_features)
            loss_cls = criterion(logits, by)
            loss_supcon = _compute_acl_supcon_loss(
                model,
                anchor_features,
                by,
                bidx_list,
                selected_positive_map,
                dev,
                temperature=acl_temperature,
            )
            loss = loss_cls + float(acl_loss_weight) * loss_supcon
            loss.backward()
            optimizer.step()

            running_cls += float(loss_cls.item())
            running_supcon += float(loss_supcon.item())
            batch_count += 1

        last_ce_loss = running_cls / max(1, batch_count)
        last_supcon_loss = running_supcon / max(1, batch_count)

        if val_loader:
            v_loss, _, v_f1 = _evaluate(model, val_loader, dev, criterion)
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
    res = {
        "accuracy": t_acc,
        "macro_f1": t_f1,
        "best_val_f1": best_val_f1,
        "best_val_loss": best_val_loss,
        "stop_epoch": stop_epoch,
        "last_ce_loss": float(last_ce_loss),
        "last_supcon_loss": float(last_supcon_loss),
        "selected_anchor_count": int(sum(1 for v in selected_positive_map.values() if v)),
        "selected_positive_count": int(sum(len(v) for v in selected_positive_map.values())),
    }
    if return_model_obj:
        res["model_obj"] = model
    return res


def fit_eval_resnet1d_continue_ce(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    init_state_dict: Dict[str, torch.Tensor],
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 10,
    device: str = "cuda",
    return_model_obj: bool = False,
) -> Dict[str, float]:
    in_channels = X_train.shape[1]
    num_classes = int(max(y_train.max(), (y_val.max() if y_val is not None else 0), y_test.max()) + 1)
    model = ResNet1DClassifier(in_channels, num_classes)
    model.load_state_dict(copy.deepcopy(init_state_dict))
    return fit_eval_pytorch_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        device=device,
        return_model_obj=return_model_obj,
    )


def _compute_acl_supcon_loss(
    model: nn.Module,
    anchor_features: torch.Tensor,
    anchor_labels: torch.Tensor,
    anchor_indices: List[int],
    selected_positive_map: Dict[int, List[np.ndarray]],
    dev: torch.device,
    *,
    temperature: float,
) -> torch.Tensor:
    selected_local_idx: List[int] = []
    positive_arrays: List[np.ndarray] = []
    positive_labels: List[int] = []

    for local_idx, anchor_idx in enumerate(anchor_indices):
        pos_list = selected_positive_map.get(int(anchor_idx), [])
        if not pos_list:
            continue
        selected_local_idx.append(int(local_idx))
        label_val = int(anchor_labels[local_idx].item())
        for pos_x in pos_list:
            positive_arrays.append(np.asarray(pos_x, dtype=np.float32))
            positive_labels.append(label_val)

    if not selected_local_idx or not positive_arrays:
        return anchor_features.new_zeros(())

    anchor_proj = model.project(anchor_features[selected_local_idx])
    anchor_sup_labels = anchor_labels[selected_local_idx]

    x_pos = torch.from_numpy(np.stack(positive_arrays)).float().to(dev, non_blocking=True)
    pos_features = model.encode(x_pos)
    pos_proj = model.project(pos_features)
    pos_labels = torch.tensor(positive_labels, device=dev, dtype=torch.long)

    supcon_features = torch.cat([anchor_proj, pos_proj], dim=0)
    supcon_labels = torch.cat([anchor_sup_labels, pos_labels], dim=0)
    return supervised_contrastive_loss(
        supcon_features,
        supcon_labels,
        temperature=temperature,
    )

def fit_eval_resnet1d(X_train, y_train, X_val, y_val, X_test, y_test, epochs=30, lr=1e-3, batch_size=64, patience=10, device="cuda", return_model_obj=False):
    in_channels = X_train.shape[1]
    num_classes = int(max(y_train.max(), (y_val.max() if y_val is not None else 0), y_test.max()) + 1)
    model = ResNet1DClassifier(in_channels, num_classes)
    return fit_eval_pytorch_model(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs, lr, batch_size, patience, device, return_model_obj=return_model_obj)

def fit_eval_patchtst(X_train, y_train, X_val, y_val, X_test, y_test, epochs=100, lr=5e-4, batch_size=64, patience=15, device="cuda", return_model_obj=False):
    in_channels = X_train.shape[1]
    seq_len = X_train.shape[2]
    num_classes = int(max(y_train.max(), (y_val.max() if y_val is not None else 0), y_test.max()) + 1)
    model = PatchTST(in_channels, seq_len, num_classes)
    return fit_eval_pytorch_model(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs, lr, batch_size, patience, device, use_cosine_annealing=True, return_model_obj=return_model_obj)

def fit_eval_timesnet(X_train, y_train, X_val, y_val, X_test, y_test, epochs=100, lr=5e-4, batch_size=32, patience=15, device="cuda", return_model_obj=False):
    in_channels = X_train.shape[1]
    seq_len = X_train.shape[2]
    num_classes = int(max(y_train.max(), (y_val.max() if y_val is not None else 0), y_test.max()) + 1)
    model = TimesNet(in_channels, seq_len, num_classes)
    return fit_eval_pytorch_model(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs, lr, batch_size, patience, device, return_model_obj=return_model_obj)
