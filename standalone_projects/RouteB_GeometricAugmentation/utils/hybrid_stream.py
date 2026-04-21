from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class HybridStreamDataset(Dataset):
    def __init__(
        self,
        raw_seq_list: Sequence[np.ndarray],
        de_feat_list: Sequence[np.ndarray],
        labels: Sequence[int],
    ) -> None:
        if len(raw_seq_list) != len(de_feat_list) or len(raw_seq_list) != len(labels):
            raise ValueError("raw_seq_list, de_feat_list, labels must have same length")
        self.raw_seq: List[np.ndarray] = list(raw_seq_list)
        self.de_feat: List[np.ndarray] = list(de_feat_list)
        self.labels: List[int] = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        manifold_x = torch.as_tensor(self.raw_seq[idx], dtype=torch.float32)
        dcnet_x = torch.as_tensor(self.de_feat[idx], dtype=torch.float32)
        y = torch.LongTensor([self.labels[idx]])
        return manifold_x, dcnet_x, y
