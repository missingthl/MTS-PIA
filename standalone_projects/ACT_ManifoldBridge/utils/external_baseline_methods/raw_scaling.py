from __future__ import annotations

import numpy as np

from utils.external_baseline_methods.base import ExternalAugResult, repeat_anchor_indices, rng


def raw_aug_scaling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    low: float = 0.8,
    high: float = 1.2,
) -> ExternalAugResult:
    gen = rng(seed)
    idx = repeat_anchor_indices(len(X_train), multiplier)
    X_src = np.asarray(X_train[idx], dtype=np.float32)
    factors = gen.uniform(float(low), float(high), size=(len(idx), 1, 1)).astype(np.float32)
    return ExternalAugResult(
        X_aug=X_src * factors,
        y_aug=np.asarray(y_train[idx], dtype=np.int64),
        source_space="raw_time",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="repeat_train_anchors_amplitude_uniform",
        meta={"scaling_low": float(low), "scaling_high": float(high)},
    )
