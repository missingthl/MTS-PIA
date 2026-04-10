from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from route_b_unified.types import RepresentationState


def _normalize_raw_trial(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True) + 1e-6
    return ((arr - mean) / std).astype(np.float32)


@dataclass
class DualStreamSplit:
    split_name: str
    trial_dicts: List[Dict[str, object]]
    X_z: np.ndarray
    y: np.ndarray
    tids: np.ndarray
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class DualStreamState:
    dataset: str
    seed: int
    split_meta: Dict[str, object]
    train: DualStreamSplit
    val: DualStreamSplit
    test: DualStreamSplit
    num_classes: int
    channels: int
    max_length: int
    z_dim: int
    meta: Dict[str, object] = field(default_factory=dict)


class DualStreamTrialDataset(Dataset):
    def __init__(self, split: DualStreamSplit):
        self.split = split
        if len(split.trial_dicts) != int(split.X_z.shape[0]) or len(split.trial_dicts) != int(split.y.shape[0]):
            raise ValueError(f"split={split.split_name} has mismatched trial / z / y sizes")
        tids = [str(t["trial_id_str"]) for t in split.trial_dicts]
        if tids != [str(v) for v in split.tids.tolist()]:
            raise ValueError(f"split={split.split_name} tid order mismatch between trial_dicts and z-space arrays")

    def __len__(self) -> int:
        return int(len(self.split.trial_dicts))

    def __getitem__(self, idx: int):
        trial = self.split.trial_dicts[int(idx)]
        return {
            "trial_id_str": str(trial["trial_id_str"]),
            "raw_x": _normalize_raw_trial(np.asarray(trial["x_trial"], dtype=np.float32)),
            "z": np.asarray(self.split.X_z[int(idx)], dtype=np.float32),
            "label": int(self.split.y[int(idx)]),
        }


def collate_dual_stream_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not batch:
        raise ValueError("batch cannot be empty")
    tids = [str(item["trial_id_str"]) for item in batch]
    labels = torch.as_tensor([int(item["label"]) for item in batch], dtype=torch.long)
    z = torch.as_tensor(np.stack([np.asarray(item["z"], dtype=np.float32) for item in batch], axis=0), dtype=torch.float32)

    raw_list = [np.asarray(item["raw_x"], dtype=np.float32) for item in batch]
    channels = int(raw_list[0].shape[0])
    max_len = int(max(arr.shape[1] for arr in raw_list))
    raw = np.zeros((len(raw_list), channels, max_len), dtype=np.float32)
    lengths = np.zeros((len(raw_list),), dtype=np.int64)
    for i, arr in enumerate(raw_list):
        if int(arr.shape[0]) != channels:
            raise ValueError("all samples in a batch must share the same channel count")
        t = int(arr.shape[1])
        raw[i, :, :t] = arr
        lengths[i] = t
    return {
        "trial_ids": tids,
        "raw_x": torch.as_tensor(raw, dtype=torch.float32),
        "raw_lengths": torch.as_tensor(lengths, dtype=torch.long),
        "z": z,
        "labels": labels,
    }


def _make_split(
    *,
    split_name: str,
    trial_dicts: Sequence[Dict[str, object]],
    X_z: np.ndarray,
    y: np.ndarray,
    tids: np.ndarray,
) -> DualStreamSplit:
    return DualStreamSplit(
        split_name=str(split_name),
        trial_dicts=list(trial_dicts),
        X_z=np.asarray(X_z, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        tids=np.asarray(tids, dtype=object),
        meta={
            "n_trials": int(len(trial_dicts)),
        },
    )


def build_dual_stream_state(rep_state: RepresentationState) -> DualStreamState:
    train_split = _make_split(
        split_name="train",
        trial_dicts=rep_state.train_trial_dicts,
        X_z=rep_state.X_train,
        y=rep_state.y_train,
        tids=rep_state.tid_train,
    )
    val_split = _make_split(
        split_name="val",
        trial_dicts=rep_state.val_trial_dicts,
        X_z=rep_state.X_val,
        y=rep_state.y_val,
        tids=rep_state.tid_val,
    )
    test_split = _make_split(
        split_name="test",
        trial_dicts=rep_state.test_trial_dicts,
        X_z=rep_state.X_test,
        y=rep_state.y_test,
        tids=rep_state.tid_test,
    )

    all_trials = list(train_split.trial_dicts) + list(val_split.trial_dicts) + list(test_split.trial_dicts)
    if not all_trials:
        raise ValueError("dual stream state requires at least one trial")
    channels = int(np.asarray(all_trials[0]["x_trial"], dtype=np.float32).shape[0])
    max_length = int(max(np.asarray(t["x_trial"], dtype=np.float32).shape[1] for t in all_trials))
    num_classes = int(len(np.unique(np.asarray(rep_state.y_train, dtype=np.int64))))

    return DualStreamState(
        dataset=str(rep_state.dataset),
        seed=int(rep_state.seed),
        split_meta=dict(rep_state.split_meta),
        train=train_split,
        val=val_split,
        test=test_split,
        num_classes=num_classes,
        channels=channels,
        max_length=max_length,
        z_dim=int(rep_state.X_train.shape[1]) if rep_state.X_train.ndim == 2 else 0,
        meta={
            "train_trials": int(len(train_split.trial_dicts)),
            "val_trials": int(len(val_split.trial_dicts)),
            "test_trials": int(len(test_split.trial_dicts)),
        },
    )
