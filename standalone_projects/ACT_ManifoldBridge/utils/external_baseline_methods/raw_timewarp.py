from __future__ import annotations

import numpy as np

from utils.external_baseline_methods.base import ExternalAugResult, repeat_anchor_indices


def raw_aug_timewarp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    n_speed_change: int = 3,
    max_speed_ratio: float = 2.0,
) -> ExternalAugResult:
    try:
        from tsaug import TimeWarp
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("raw_aug_timewarp requires optional dependency `tsaug`.") from exc

    idx = repeat_anchor_indices(len(X_train), multiplier)
    X_src = np.asarray(X_train[idx], dtype=np.float32)
    np.random.seed(int(seed) % (2**32 - 1))
    X_tc = np.transpose(X_src, (0, 2, 1))
    X_aug = TimeWarp(
        n_speed_change=int(n_speed_change),
        max_speed_ratio=float(max_speed_ratio),
    ).augment(X_tc)
    X_aug = np.transpose(np.asarray(X_aug, dtype=np.float32), (0, 2, 1))
    return ExternalAugResult(
        X_aug=X_aug,
        y_aug=np.asarray(y_train[idx], dtype=np.int64),
        source_space="raw_time",
        label_mode="hard",
        uses_external_library=True,
        library_name="tsaug",
        budget_matched=True,
        selection_rule="repeat_train_anchors_timewarp",
    )
