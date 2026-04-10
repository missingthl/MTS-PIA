from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from route_b_unified.trajectory_classifier import DynamicGRUClassifier, TrajectoryModelConfig
from route_b_unified.trajectory_pia_operator import TrajectoryPIAOperator, TrajectoryPIAOperatorConfig
from route_b_unified.trajectory_representation import TrajectoryRepresentationState


@dataclass(frozen=True)
class TrajectoryPIAEvalConfig:
    operator_mode: str
    gamma_main: float = 0.10
    smooth_lambda: float = 0.50
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    device: str = "auto"


@dataclass
class TrajectoryPIAEvalResult:
    dataset: str
    seed: int
    operator_mode: str
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    best_epoch: int
    diagnostics: Dict[str, float] = field(default_factory=dict)
    operator_meta: Dict[str, object] = field(default_factory=dict)
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


def _normalize_sequence(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return ((arr - np.asarray(mean, dtype=np.float32)[None, :]) / (np.asarray(std, dtype=np.float32)[None, :] + 1e-6)).astype(
        np.float32
    )


class _DynamicTrajectoryDataset(Dataset):
    def __init__(
        self,
        *,
        tids: Sequence[str],
        labels: Sequence[int],
        z_seq_list: Sequence[np.ndarray],
        dynamic_feature_mean: np.ndarray,
        dynamic_feature_std: np.ndarray,
        z_dim: int,
    ) -> None:
        self.tids = [str(v) for v in tids]
        self.labels = [int(v) for v in labels]
        self.z_seq_list = [_normalize_sequence(np.asarray(v, dtype=np.float32), dynamic_feature_mean, dynamic_feature_std) for v in z_seq_list]
        self.z_dim = int(z_dim)
        if len(self.tids) != len(self.labels) or len(self.tids) != len(self.z_seq_list):
            raise ValueError("trajectory dataset sizes must match")

    def __len__(self) -> int:
        return int(len(self.tids))

    def __getitem__(self, idx: int):
        return {
            "trial_id_str": str(self.tids[int(idx)]),
            "label": int(self.labels[int(idx)]),
            "z_static": np.zeros((self.z_dim,), dtype=np.float32),
            "z_seq": np.asarray(self.z_seq_list[int(idx)], dtype=np.float32),
        }


def _collate_dynamic_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not batch:
        raise ValueError("batch cannot be empty")
    tids = [str(item["trial_id_str"]) for item in batch]
    labels = torch.as_tensor([int(item["label"]) for item in batch], dtype=torch.long)
    z_static = torch.as_tensor(
        np.stack([np.asarray(item["z_static"], dtype=np.float32) for item in batch], axis=0),
        dtype=torch.float32,
    )
    seqs = [np.asarray(item["z_seq"], dtype=np.float32) for item in batch]
    feat_dim = int(seqs[0].shape[1])
    max_len = int(max(seq.shape[0] for seq in seqs))
    z_seq = np.zeros((len(seqs), max_len, feat_dim), dtype=np.float32)
    lengths = np.zeros((len(seqs),), dtype=np.int64)
    for i, seq in enumerate(seqs):
        k = int(seq.shape[0])
        z_seq[i, :k, :] = seq
        lengths[i] = k
    return {
        "trial_ids": tids,
        "labels": labels,
        "z_static": z_static,
        "z_seq": torch.as_tensor(z_seq, dtype=torch.float32),
        "seq_lengths": torch.as_tensor(lengths, dtype=torch.long),
    }


def _pairwise_mean_distance(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= 1:
        return 0.0
    diffs = arr[:, None, :] - arr[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    tri = dists[np.triu_indices(arr.shape[0], k=1)]
    return float(np.mean(tri)) if tri.size else 0.0


def compute_trajectory_diagnostics(
    z_seq_list: Sequence[np.ndarray],
    labels: Sequence[int],
    *,
    continuity_ratio: float,
) -> Dict[str, float]:
    seqs = [np.asarray(v, dtype=np.float32) for v in z_seq_list]
    y = np.asarray(list(labels), dtype=np.int64)
    traj_lens = [int(v.shape[0]) for v in seqs]
    step_mags: List[float] = []
    curvature_vals: List[float] = []
    seq_means: List[np.ndarray] = []
    delta_by_class: Dict[int, List[np.ndarray]] = {}

    for seq, cls in zip(seqs, y.tolist()):
        seq_means.append(np.mean(seq, axis=0).astype(np.float32))
        if int(seq.shape[0]) >= 2:
            delta = np.diff(seq, axis=0)
            step_mags.append(float(np.mean(np.linalg.norm(delta, axis=1))))
            delta_by_class.setdefault(int(cls), []).append(np.mean(delta, axis=0).astype(np.float32))
        if int(seq.shape[0]) >= 3:
            curv = seq[2:] - 2.0 * seq[1:-1] + seq[:-2]
            curvature_vals.append(float(np.mean(np.linalg.norm(curv, axis=1))))

    seq_means_arr = np.stack(seq_means, axis=0).astype(np.float32) if seq_means else np.zeros((0, 0), dtype=np.float32)
    class_dispersion_rows: List[float] = []
    for cls in sorted(set(y.tolist())):
        mask = y == int(cls)
        if int(np.sum(mask)) <= 1:
            continue
        class_dispersion_rows.append(_pairwise_mean_distance(seq_means_arr[mask]))

    mean_deltas: List[np.ndarray] = []
    for cls in sorted(delta_by_class):
        cls_arr = np.stack(delta_by_class[int(cls)], axis=0).astype(np.float32)
        mean_deltas.append(np.mean(cls_arr, axis=0).astype(np.float32))
    transition_sep = _pairwise_mean_distance(np.stack(mean_deltas, axis=0).astype(np.float32)) if len(mean_deltas) >= 2 else 0.0

    return {
        "trajectory_len_mean": float(np.mean(traj_lens)) if traj_lens else 0.0,
        "step_change_mean": float(np.mean(step_mags)) if step_mags else 0.0,
        "local_curvature_proxy": float(np.mean(curvature_vals)) if curvature_vals else 0.0,
        "classwise_dispersion": float(np.mean(class_dispersion_rows)) if class_dispersion_rows else 0.0,
        "transition_separation_proxy": float(transition_sep),
        "continuity_distortion_ratio": float(continuity_ratio),
    }


def _compute_loss(outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    return nn.CrossEntropyLoss()(outputs["logits"], labels)


@torch.no_grad()
def _evaluate_split(model: nn.Module, loader: DataLoader, *, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    loss_rows: List[float] = []
    for batch in loader:
        labels = batch["labels"].to(device)
        z_static = batch["z_static"].to(device)
        z_seq = batch["z_seq"].to(device)
        seq_lengths = batch["seq_lengths"].to(device)
        outputs = model(z_static, z_seq, seq_lengths)
        loss = _compute_loss(outputs, labels)
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


def _build_train_final(
    state: TrajectoryRepresentationState,
    *,
    operator_mode: str,
    gamma_main: float,
    smooth_lambda: float,
    operator_cfg: TrajectoryPIAOperatorConfig,
    prefit_operator: TrajectoryPIAOperator | None = None,
) -> tuple[List[str], List[int], List[np.ndarray], Dict[str, object], Dict[str, float]]:
    train_tids = [str(v) for v in state.train.tids.tolist()]
    train_labels = [int(v) for v in state.train.y.tolist()]
    train_seqs = [np.asarray(v, dtype=np.float32) for v in state.train.z_seq_list]

    if str(operator_mode) == "baseline":
        diag = compute_trajectory_diagnostics(train_seqs, train_labels, continuity_ratio=1.0)
        return train_tids, train_labels, train_seqs, {"mode": "baseline"}, diag

    operator = prefit_operator if prefit_operator is not None else TrajectoryPIAOperator(operator_cfg).fit(train_seqs)
    aug_seqs, _delta_list, op_meta = operator.transform_many(
        train_seqs,
        gamma_main=float(gamma_main),
        smooth_lambda=float(smooth_lambda),
    )
    aug_tids = [f"{tid}__aug_g{int(round(float(gamma_main) * 1000)):03d}_l{int(round(float(smooth_lambda) * 100)):02d}" for tid in train_tids]
    final_tids = list(train_tids) + list(aug_tids)
    final_labels = list(train_labels) + list(train_labels)
    final_seqs = list(train_seqs) + list(aug_seqs)
    diag = compute_trajectory_diagnostics(
        final_seqs,
        final_labels,
        continuity_ratio=float(op_meta["mean_continuity_distortion_ratio"]),
    )
    return final_tids, final_labels, final_seqs, op_meta, diag


def evaluate_trajectory_train_final(
    state: TrajectoryRepresentationState,
    *,
    seed: int,
    model_cfg: TrajectoryModelConfig,
    eval_cfg: TrajectoryPIAEvalConfig,
    train_tids: Sequence[str],
    train_labels: Sequence[int],
    train_z_seq_list: Sequence[np.ndarray],
    diagnostics: Dict[str, float],
    operator_meta: Dict[str, object],
) -> TrajectoryPIAEvalResult:
    _set_seed(int(seed))
    device = _resolve_device(eval_cfg.device)
    model = DynamicGRUClassifier(model_cfg).to(device)

    train_ds = _DynamicTrajectoryDataset(
        tids=train_tids,
        labels=train_labels,
        z_seq_list=train_z_seq_list,
        dynamic_feature_mean=state.dynamic_feature_mean,
        dynamic_feature_std=state.dynamic_feature_std,
        z_dim=int(state.z_dim),
    )
    val_ds = _DynamicTrajectoryDataset(
        tids=[str(v) for v in state.val.tids.tolist()],
        labels=[int(v) for v in state.val.y.tolist()],
        z_seq_list=state.val.z_seq_list,
        dynamic_feature_mean=state.dynamic_feature_mean,
        dynamic_feature_std=state.dynamic_feature_std,
        z_dim=int(state.z_dim),
    )
    test_ds = _DynamicTrajectoryDataset(
        tids=[str(v) for v in state.test.tids.tolist()],
        labels=[int(v) for v in state.test.y.tolist()],
        z_seq_list=state.test.z_seq_list,
        dynamic_feature_mean=state.dynamic_feature_mean,
        dynamic_feature_std=state.dynamic_feature_std,
        z_dim=int(state.z_dim),
    )

    train_loader = DataLoader(train_ds, batch_size=int(eval_cfg.batch_size), shuffle=True, collate_fn=_collate_dynamic_batch)
    eval_train_loader = DataLoader(train_ds, batch_size=int(eval_cfg.batch_size), shuffle=False, collate_fn=_collate_dynamic_batch)
    val_loader = DataLoader(val_ds, batch_size=int(eval_cfg.batch_size), shuffle=False, collate_fn=_collate_dynamic_batch)
    test_loader = DataLoader(test_ds, batch_size=int(eval_cfg.batch_size), shuffle=False, collate_fn=_collate_dynamic_batch)

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
            z_static = batch["z_static"].to(device)
            z_seq = batch["z_seq"].to(device)
            seq_lengths = batch["seq_lengths"].to(device)
            outputs = model(z_static, z_seq, seq_lengths)
            loss = _compute_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_rows.append(float(loss.detach().cpu().item()))

        train_metrics = _evaluate_split(model, eval_train_loader, device=device)
        val_metrics = _evaluate_split(model, val_loader, device=device)
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
    train_metrics = _evaluate_split(model, eval_train_loader, device=device)
    val_metrics = _evaluate_split(model, val_loader, device=device)
    test_metrics = _evaluate_split(model, test_loader, device=device)

    return TrajectoryPIAEvalResult(
        dataset=str(state.dataset),
        seed=int(seed),
        operator_mode=str(eval_cfg.operator_mode),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        best_epoch=int(best_epoch),
        diagnostics=diagnostics,
        operator_meta=operator_meta,
        history_rows=history_rows,
        meta={
            "device": str(device),
            "epochs_requested": int(eval_cfg.epochs),
            "epochs_ran": int(len(history_rows)),
            "batch_size": int(eval_cfg.batch_size),
            "lr": float(eval_cfg.lr),
            "weight_decay": float(eval_cfg.weight_decay),
            "patience": int(eval_cfg.patience),
            "train_final_size": int(len(train_ds)),
            "val_size": int(len(val_ds)),
            "test_size": int(len(test_ds)),
        },
    )


def evaluate_trajectory_pia_t2a(
    state: TrajectoryRepresentationState,
    *,
    seed: int,
    model_cfg: TrajectoryModelConfig,
    eval_cfg: TrajectoryPIAEvalConfig,
    operator_cfg: TrajectoryPIAOperatorConfig,
    prefit_operator: TrajectoryPIAOperator | None = None,
) -> TrajectoryPIAEvalResult:
    train_tids, train_labels, train_seqs, operator_meta, diagnostics = _build_train_final(
        state,
        operator_mode=str(eval_cfg.operator_mode),
        gamma_main=float(eval_cfg.gamma_main),
        smooth_lambda=float(eval_cfg.smooth_lambda),
        operator_cfg=operator_cfg,
        prefit_operator=prefit_operator,
    )

    return evaluate_trajectory_train_final(
        state,
        seed=int(seed),
        model_cfg=model_cfg,
        eval_cfg=eval_cfg,
        train_tids=train_tids,
        train_labels=train_labels,
        train_z_seq_list=train_seqs,
        diagnostics=diagnostics,
        operator_meta=operator_meta,
    )
