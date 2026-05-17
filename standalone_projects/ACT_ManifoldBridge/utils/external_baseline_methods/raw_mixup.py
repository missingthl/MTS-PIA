from __future__ import annotations

from typing import Optional

import numpy as np

from utils.external_baseline_methods.base import ExternalAugResult, one_hot, rng


def raw_mixup(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    alpha: float = 0.4,
    n_classes: Optional[int] = None,
) -> ExternalAugResult:
    gen = rng(seed)
    n_train = int(len(X_train))
    n_aug = int(multiplier) * n_train
    n_classes_i = int(n_classes if n_classes is not None else np.max(y_train) + 1)
    i = gen.integers(0, n_train, size=n_aug)
    j = gen.integers(0, n_train, size=n_aug)
    lam = gen.beta(float(alpha), float(alpha), size=(n_aug, 1, 1)).astype(np.float32)
    X_aug = lam * np.asarray(X_train[i], dtype=np.float32) + (1.0 - lam) * np.asarray(X_train[j], dtype=np.float32)

    lam_y = lam.reshape(n_aug, 1)
    y_i = one_hot(np.asarray(y_train[i], dtype=np.int64), n_classes_i)
    y_j = one_hot(np.asarray(y_train[j], dtype=np.int64), n_classes_i)
    y_soft = lam_y * y_i + (1.0 - lam_y) * y_j
    return ExternalAugResult(
        X_aug=X_aug.astype(np.float32),
        y_aug_soft=y_soft.astype(np.float32),
        source_space="raw_mixup",
        label_mode="soft",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="train_split_random_pair_beta",
        meta={"mixup_alpha": float(alpha)},
    )
