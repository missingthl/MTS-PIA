from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ManifoldAugDataset(Dataset):
    """
    On-the-fly CSTA manifold augmentation dataset.

    The dataset stores only lightweight ``(x_raw, sigma_orig, z_candidate)``
    tuples.  The whitening-coloring bridge is evaluated in ``__getitem__`` so
    large augmented raw arrays do not need to be materialized up front.
    """

    def __init__(
        self,
        anchor_x_raws: List[np.ndarray],
        anchor_sigma_origs: List[np.ndarray],
        z_cands: np.ndarray,
        y_cands: np.ndarray,
        mean_log: np.ndarray,
    ) -> None:
        if not (len(anchor_x_raws) == len(anchor_sigma_origs) == len(z_cands) == len(y_cands)):
            raise ValueError("ManifoldAugDataset inputs must have equal length.")
        self._x_raws = anchor_x_raws
        self._sigma_origs = anchor_sigma_origs
        self._z_cands = np.asarray(z_cands, dtype=np.float32)
        self._y_cands = np.asarray(y_cands, dtype=np.int64)
        self._mean_log = np.asarray(mean_log, dtype=np.float64)

    def __len__(self) -> int:
        return int(self._z_cands.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self._x_raws[idx]).float(),
            torch.from_numpy(self._sigma_origs[idx]).float(),
            torch.from_numpy(self._z_cands[idx]).float(),
            torch.tensor(int(self._y_cands[idx]), dtype=torch.long),
        )
