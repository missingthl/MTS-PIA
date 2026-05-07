from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ExternalAugResult:
    X_aug: np.ndarray
    y_aug: Optional[np.ndarray] = None
    y_aug_soft: Optional[np.ndarray] = None
    source_space: str = "raw_time"
    label_mode: str = "hard"
    uses_external_library: bool = False
    library_name: str = ""
    budget_matched: bool = True
    selection_rule: str = ""
    warning_count: int = 0
    fallback_count: int = 0
    meta: Dict[str, float] = field(default_factory=dict)


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def repeat_anchor_indices(n_train: int, multiplier: int) -> np.ndarray:
    return np.repeat(np.arange(int(n_train), dtype=np.int64), int(multiplier))


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((int(y.shape[0]), int(n_classes)), dtype=np.float32)
    out[np.arange(int(y.shape[0])), y.astype(np.int64)] = 1.0
    return out


def resample_ct(x_ct: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample one multivariate series from [C, T] to [C, target_len]."""
    x_ct = np.asarray(x_ct, dtype=np.float32)
    c, t = int(x_ct.shape[0]), int(x_ct.shape[1])
    target_len = int(target_len)
    if t == target_len:
        return x_ct.astype(np.float32, copy=True)
    if t <= 1:
        return np.repeat(x_ct, target_len, axis=1).astype(np.float32)
    src = np.linspace(0.0, 1.0, t)
    dst = np.linspace(0.0, 1.0, target_len)
    out = np.empty((c, target_len), dtype=np.float32)
    for ch in range(c):
        out[ch] = np.interp(dst, src, x_ct[ch]).astype(np.float32)
    return out


def class_to_indices(y_train: np.ndarray) -> Dict[int, np.ndarray]:
    y_train = np.asarray(y_train, dtype=np.int64)
    return {int(c): np.flatnonzero(y_train == c) for c in np.unique(y_train)}


def finite_stack(xs: List[np.ndarray]) -> np.ndarray:
    return np.nan_to_num(np.stack(xs, axis=0).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
