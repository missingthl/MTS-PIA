from __future__ import annotations

import os
from contextlib import contextmanager
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
from act_lite_feedback import compute_margin_feedback
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


@contextmanager
def _temporary_eval_mode(model: nn.Module):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()

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
    aug_ce_mode: str = "selected",
    selected_alignment_map: Optional[Dict[int, List[float]]] = None,
    soft_gating: bool = False,
    gating_tau: float = 0.0,
    return_model_obj: bool = False,
    loader_seed: Optional[int] = None,
) -> Dict[str, float]:
    in_channels = X_train.shape[1]
    num_classes = int(max(y_train.max(), (y_val.max() if y_val is not None else 0), y_test.max()) + 1)
    model = ResNet1DClassifier(in_channels, num_classes)
    model.load_state_dict(copy.deepcopy(init_state_dict))

    dev = _get_dev(device)
    model.to(dev)

    train_loader = _make_tensor_loader(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        indexed=True,
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
    early_stopping = EarlyStopping(patience=patience, mode='max')

    best_val_f1 = 0.0
    best_val_loss = float("inf")
    stop_epoch = epochs
    last_supcon_loss = 0.0
    last_ce_loss = 0.0
    running_gating_weight = 0.0
    running_zero_weight_count = 0
    total_aug_samples = 0

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
            loss_cls_orig = criterion(logits, by)
            
            # Combined Feedback Control: Apply gating to both CE and SupCon to prevent decoupling
            loss_supcon, aug_info = _compute_acl_supcon_loss(
                model,
                anchor_features,
                by,
                bidx_list,
                selected_positive_map,
                dev,
                temperature=acl_temperature,
                selected_alignment_map=selected_alignment_map,
                soft_gating=soft_gating,
                gating_tau=gating_tau,
            )
            
            loss_cls_aug = torch.tensor(0.0, device=dev)
            if aug_info is not None and aug_ce_mode == "selected":
                # aug_info: (aug_features, aug_labels, Optional[aug_weights])
                aug_feat, aug_lab, aug_weights = aug_info
                aug_logits = model.classify(aug_feat)
                
                if soft_gating and aug_weights is not None:
                    # Apply alignment-guided soft gating: w = max(0, (a - tau)/(1 - tau))
                    w = torch.clamp((aug_weights - gating_tau) / (1.0 - gating_tau + 1e-8), min=0.0, max=1.0)
                    
                    # Log metrics
                    running_gating_weight += float(w.sum().item())
                    running_zero_weight_count += int((w <= 1e-5).sum().item())
                    total_aug_samples += int(w.shape[0])
                    
                    # Sample-wise weighted CE
                    raw_ce = F.cross_entropy(aug_logits, aug_lab, reduction='none')
                    loss_cls_aug = (raw_ce * w).mean()
                else:
                    loss_cls_aug = criterion(aug_logits, aug_lab)

            # Combined Loss: CE(orig) + [CE(aug) if selected] + alpha * SupCon(orig, aug)
            loss_cls = loss_cls_orig + loss_cls_aug
            loss = loss_cls + float(acl_loss_weight) * loss_supcon
            
            loss.backward()
            optimizer.step()

            running_cls += float(loss_cls_orig.item())# Still track orig separately if needed, or total
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
        "mean_aug_ce_weight": float(running_gating_weight / max(1, total_aug_samples)) if soft_gating else 1.0,
        "zero_weight_fraction": float(running_zero_weight_count / max(1, total_aug_samples)) if soft_gating else 0.0,
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
    loader_seed: Optional[int] = None,
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
        loader_seed=loader_seed,
    )


def fit_eval_resnet1d_weighted_aug_ce(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_aug: Optional[np.ndarray],
    y_aug: Optional[np.ndarray],
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    init_state_dict: Dict[str, torch.Tensor],
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 10,
    device: str = "cuda",
    feedback_margin_temperature: float = 1.0,
    feedback_margin_polarity: str = "easy",
    feedback_aug_weight: float = 1.0,
    loader_seed: Optional[int] = None,
    return_model_obj: bool = False,
) -> Dict[str, float]:
    in_channels = X_train.shape[1]
    num_classes = int(max(y_train.max(), (y_val.max() if y_val is not None else 0), y_test.max()) + 1)
    model = ResNet1DClassifier(in_channels, num_classes)
    model.load_state_dict(copy.deepcopy(init_state_dict))

    dev = _get_dev(device)
    model.to(dev)

    orig_loader = _make_tensor_loader(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        indexed=True,
        seed=loader_seed,
    )

    aug_loader = None
    aug_weight_sum = np.zeros((0,), dtype=np.float64)
    aug_margin_sum = np.zeros((0,), dtype=np.float64)
    aug_seen_count = np.zeros((0,), dtype=np.int64)
    if X_aug is not None and y_aug is not None and len(X_aug) > 0:
        aug_weight_sum = np.zeros((len(X_aug),), dtype=np.float64)
        aug_margin_sum = np.zeros((len(X_aug),), dtype=np.float64)
        aug_seen_count = np.zeros((len(X_aug),), dtype=np.int64)
        aug_seed = None if loader_seed is None else int(loader_seed) + 100_000
        aug_loader = _make_tensor_loader(
            X_aug,
            y_aug,
            batch_size=batch_size,
            shuffle=True,
            indexed=True,
            seed=aug_seed,
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
    early_stopping = EarlyStopping(patience=patience, mode="max")

    best_val_f1 = 0.0
    best_val_loss = float("inf")
    stop_epoch = epochs
    last_orig_ce_loss = 0.0
    last_weighted_aug_ce_loss = 0.0
    feedback_weight_mean = 0.0
    feedback_weight_std = 0.0
    feedback_reject_frac = 0.0
    last_aug_margin_mean = 0.0

    for epoch in range(max(0, int(epochs))):
        model.train()
        running_orig_loss = 0.0
        running_aug_loss = 0.0
        batch_count = 0
        epoch_weights: List[np.ndarray] = []
        epoch_margins: List[np.ndarray] = []
        aug_iter = iter(aug_loader) if aug_loader is not None else None

        for bx, by, _ in orig_loader:
            bx = bx.to(dev, non_blocking=True)
            by = by.to(dev, non_blocking=True)

            optimizer.zero_grad()
            orig_features = model.encode(bx)
            orig_logits = model.classify(orig_features)
            loss_orig = criterion(orig_logits, by)
            loss_aug = torch.tensor(0.0, device=dev)

            if aug_iter is not None:
                try:
                    ax, ay, aidx = next(aug_iter)
                except StopIteration:
                    aug_iter = iter(aug_loader)
                    ax, ay, aidx = next(aug_iter)

                ax = ax.to(dev, non_blocking=True)
                ay = ay.to(dev, non_blocking=True)
                aidx_np = aidx.detach().cpu().numpy().astype(np.int64, copy=False)

                # Keep BN statistics tied to the original supervision stream only.
                with _temporary_eval_mode(model):
                    aug_features = model.encode(ax)
                    aug_logits = model.classify(aug_features)

                aug_margins, aug_weights = compute_margin_feedback(
                    aug_logits,
                    ay,
                    temperature=feedback_margin_temperature,
                    polarity=feedback_margin_polarity,
                )
                raw_aug_ce = F.cross_entropy(aug_logits, ay, reduction="none")
                loss_aug = float(feedback_aug_weight) * (aug_weights * raw_aug_ce).mean()

                epoch_weights.append(aug_weights.detach().cpu().numpy())
                epoch_margins.append(aug_margins.detach().cpu().numpy())
                aug_weight_sum[aidx_np] += aug_weights.detach().cpu().numpy().astype(np.float64, copy=False)
                aug_margin_sum[aidx_np] += aug_margins.detach().cpu().numpy().astype(np.float64, copy=False)
                aug_seen_count[aidx_np] += 1

            loss = loss_orig + loss_aug
            loss.backward()
            optimizer.step()

            running_orig_loss += float(loss_orig.item())
            running_aug_loss += float(loss_aug.item())
            batch_count += 1

        last_orig_ce_loss = running_orig_loss / max(1, batch_count)
        last_weighted_aug_ce_loss = running_aug_loss / max(1, batch_count)

        if epoch_weights:
            flat_weights = np.concatenate(epoch_weights)
            flat_margins = np.concatenate(epoch_margins)
            feedback_weight_mean = float(np.mean(flat_weights))
            feedback_weight_std = float(np.std(flat_weights))
            feedback_reject_frac = float(np.mean(flat_weights <= 0.5))
            last_aug_margin_mean = float(np.mean(flat_margins))
        else:
            feedback_weight_mean = 0.0
            feedback_weight_std = 0.0
            feedback_reject_frac = 0.0
            last_aug_margin_mean = 0.0

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
    aug_feedback_weight_by_sample = np.divide(
        aug_weight_sum,
        np.maximum(aug_seen_count, 1),
        dtype=np.float64,
    ) if aug_weight_sum.size else np.zeros((0,), dtype=np.float64)
    aug_margin_by_sample = np.divide(
        aug_margin_sum,
        np.maximum(aug_seen_count, 1),
        dtype=np.float64,
    ) if aug_margin_sum.size else np.zeros((0,), dtype=np.float64)
    if aug_seen_count.size:
        unseen = aug_seen_count <= 0
        aug_feedback_weight_by_sample[unseen] = np.nan
        aug_margin_by_sample[unseen] = np.nan
    res = {
        "accuracy": t_acc,
        "macro_f1": t_f1,
        "best_val_f1": best_val_f1,
        "best_val_loss": best_val_loss,
        "stop_epoch": stop_epoch,
        "feedback_weight_mean": float(feedback_weight_mean),
        "feedback_weight_std": float(feedback_weight_std),
        "feedback_reject_frac": float(feedback_reject_frac),
        "last_orig_ce_loss": float(last_orig_ce_loss),
        "last_weighted_aug_ce_loss": float(last_weighted_aug_ce_loss),
        "last_aug_margin_mean": float(last_aug_margin_mean),
        "aug_feedback_weight_by_sample": aug_feedback_weight_by_sample,
        "aug_margin_by_sample": aug_margin_by_sample,
        "aug_seen_count_by_sample": aug_seen_count,
    }
    if return_model_obj:
        res["model_obj"] = model
    return res


def _compute_acl_supcon_loss(
    model: nn.Module,
    anchor_features: torch.Tensor,
    anchor_labels: torch.Tensor,
    anchor_indices: List[int],
    selected_positive_map: Dict[int, List[np.ndarray]],
    dev: torch.device,
    *,
    temperature: float,
    selected_alignment_map: Optional[Dict[int, List[float]]] = None,
    soft_gating: bool = False,
    gating_tau: float = 0.0,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]]:
    """
    Returns (supcon_loss, (aug_enc_features, aug_labels, Optional[aug_weights]))
    """
    selected_local_idx: List[int] = []
    positive_arrays: List[np.ndarray] = []
    positive_labels: List[int] = []
    positive_weights: List[float] = []

    for local_idx, anchor_idx in enumerate(anchor_indices):
        pos_list = selected_positive_map.get(int(anchor_idx), [])
        if not pos_list:
            continue
        
        weight_list = None
        if selected_alignment_map is not None:
            weight_list = selected_alignment_map.get(int(anchor_idx), [])

        selected_local_idx.append(int(local_idx))
        label_val = int(anchor_labels[local_idx].item())
        for i, pos_x in enumerate(pos_list):
            positive_arrays.append(np.asarray(pos_x, dtype=np.float32))
            positive_labels.append(label_val)
            if weight_list is not None and i < len(weight_list):
                positive_weights.append(float(weight_list[i]))
            else:
                positive_weights.append(1.0)

    if not selected_local_idx or not positive_arrays:
        return anchor_features.new_zeros(()), None

    anchor_proj = model.project(anchor_features[selected_local_idx])
    anchor_sup_labels = anchor_labels[selected_local_idx]

    x_pos = torch.from_numpy(np.stack(positive_arrays)).float().to(dev, non_blocking=True)
    pos_features = model.encode(x_pos)
    pos_proj = model.project(pos_features)
    pos_labels = torch.tensor(positive_labels, device=dev, dtype=torch.long)
    
    aug_weights_t = None
    if positive_weights:
        aug_weights_t = torch.tensor(positive_weights, device=dev, dtype=torch.float)

    supcon_features = torch.cat([anchor_proj, pos_proj], dim=0)
    supcon_labels = torch.cat([anchor_sup_labels, pos_labels], dim=0)
    
    supcon_loss = supervised_contrastive_loss(
        supcon_features,
        supcon_labels,
        temperature=temperature,
    )
    return supcon_loss, (pos_features, pos_labels, aug_weights_t)

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
