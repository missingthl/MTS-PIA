from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from core.trajectory_representation import TrajectoryRepresentationState, TrajectorySplit


def _normalize_static(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return ((arr - np.asarray(mean, dtype=np.float32)) / (np.asarray(std, dtype=np.float32) + 1e-6)).astype(np.float32)


def _normalize_sequence(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return ((arr - np.asarray(mean, dtype=np.float32)[None, :]) / (np.asarray(std, dtype=np.float32)[None, :] + 1e-6)).astype(
        np.float32
    )


class TrajectoryTrialDataset(Dataset):
    def __init__(
        self,
        split: TrajectorySplit,
        *,
        static_feature_mean: np.ndarray,
        static_feature_std: np.ndarray,
        dynamic_feature_mean: np.ndarray,
        dynamic_feature_std: np.ndarray,
    ) -> None:
        self.split = split
        self.static_feature_mean = np.asarray(static_feature_mean, dtype=np.float32)
        self.static_feature_std = np.asarray(static_feature_std, dtype=np.float32)
        self.dynamic_feature_mean = np.asarray(dynamic_feature_mean, dtype=np.float32)
        self.dynamic_feature_std = np.asarray(dynamic_feature_std, dtype=np.float32)
        if len(split.trial_dicts) != int(split.X_static.shape[0]) or len(split.trial_dicts) != int(split.y.shape[0]):
            raise ValueError(f"split={split.split_name} has mismatched trial/static/y sizes")
        if len(split.trial_dicts) != int(len(split.z_seq_list)):
            raise ValueError(f"split={split.split_name} has mismatched trial/trajectory sizes")
        tids = [str(t["trial_id_str"]) for t in split.trial_dicts]
        if tids != [str(v) for v in split.tids.tolist()]:
            raise ValueError(f"split={split.split_name} tid order mismatch")

    def __len__(self) -> int:
        return int(len(self.split.trial_dicts))

    def __getitem__(self, idx: int):
        trial = self.split.trial_dicts[int(idx)]
        return {
            "trial_id_str": str(trial["trial_id_str"]),
            "label": int(self.split.y[int(idx)]),
            "z_static": _normalize_static(
                np.asarray(self.split.X_static[int(idx)], dtype=np.float32),
                self.static_feature_mean,
                self.static_feature_std,
            ),
            "z_seq": _normalize_sequence(
                np.asarray(self.split.z_seq_list[int(idx)], dtype=np.float32),
                self.dynamic_feature_mean,
                self.dynamic_feature_std,
            ),
        }


def collate_trajectory_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
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
        if int(seq.shape[1]) != feat_dim:
            raise ValueError("all sequences in a batch must share the same feature dimension")
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


def build_trajectory_datasets(
    state: TrajectoryRepresentationState,
) -> tuple[TrajectoryTrialDataset, TrajectoryTrialDataset, TrajectoryTrialDataset]:
    kwargs = {
        "static_feature_mean": state.static_feature_mean,
        "static_feature_std": state.static_feature_std,
        "dynamic_feature_mean": state.dynamic_feature_mean,
        "dynamic_feature_std": state.dynamic_feature_std,
    }
    return (
        TrajectoryTrialDataset(state.train, **kwargs),
        TrajectoryTrialDataset(state.val, **kwargs),
        TrajectoryTrialDataset(state.test, **kwargs),
    )
